"""Lossless per-frame compressed container for one scenario (s3d_v2+): `data.pack`.

Layout: b"S3DPACK1" | u64 little-endian header length | JSON header | chunks.
Header: {"arrays": {name: {"dtype", "shape", "filter"}}, "K", "T", "codec": "zlib-6",
         "chunks": [[offset, length] per (k, t, array in ARRAY_ORDER)]}  (offsets from the file start)
Chunks are ordered (camera k, frame t, array), so a window of consecutive frames of one camera is ONE
contiguous read. Every chunk is one frame of one array, zlib level 6, after a reversible integer filter:

  lr        (3,h,w)  uint8   "grad8":   r = x - (left + up - upleft) mod 256 (zero outside); inverse = 2D cumsum
  hr        (3,h,w)  uint8   "grad8_lr": grad8 of (hr - lr) mod 256 (lr/hr are identical off the medium)
  aux       (4,h,w)  fp16    "grad16_shuffle": grad of the uint16 bit patterns mod 2^16, then byte planes split
  hr_alpha  (h,w)    uint8   "grad8"
Only stdlib zlib + numpy (no new dependencies). Measured on s3d_v1 frames: lr 3.5x, hr 4.2x, aux 4.0x,
hr_alpha ~5x (whole item ~4x vs ~1.9x for plain zlib); decode is zlib inflate + two cumsums.
"""
from __future__ import annotations

import json
import os
import struct
import zlib
from pathlib import Path

import numpy as np

MAGIC = b"S3DPACK1"
ARRAY_ORDER = ("lr", "hr", "aux", "hr_alpha")
FILTERS = {"lr": "grad8", "hr": "grad8_lr", "aux": "grad16_shuffle", "hr_alpha": "grad8"}
LEVEL = 6
PACK_NAME = "data.pack"


def _grad(x: np.ndarray, dt) -> np.ndarray:
    """x - left - up + upleft over the last two axes (zero padded), wrapping in dtype `dt`."""
    x = x.astype(dt, copy=False)
    r = x.copy()
    r[..., :, 1:] -= x[..., :, :-1]
    r[..., 1:, :] -= x[..., :-1, :]
    r[..., 1:, 1:] += x[..., :-1, :-1]
    return r


def _ungrad(r: np.ndarray, dt) -> np.ndarray:
    return np.cumsum(np.cumsum(r, axis=-1, dtype=dt), axis=-2, dtype=dt)


def encode_frame(name: str, x: np.ndarray, lr: np.ndarray | None = None) -> bytes:
    f = FILTERS[name]
    if f == "grad8":
        r = _grad(x, np.uint8)
    elif f == "grad8_lr":
        r = _grad(x - lr, np.uint8)
    elif f == "grad16_shuffle":
        r = _grad(np.ascontiguousarray(x).view(np.uint16), np.uint16)
        r = np.ascontiguousarray(np.moveaxis(r.view(np.uint8).reshape(r.shape + (2,)), -1, 0))
    else:
        raise ValueError(f)
    return zlib.compress(np.ascontiguousarray(r).tobytes(), LEVEL)


def decode_frame(name: str, buf: bytes, shape, lr: np.ndarray | None = None) -> np.ndarray:
    f = FILTERS[name]
    raw = zlib.decompress(buf)
    if f in ("grad8", "grad8_lr"):
        x = _ungrad(np.frombuffer(raw, np.uint8).reshape(shape), np.uint8)
        if f == "grad8_lr":
            x += lr
        return x
    if f == "grad16_shuffle":
        b = np.frombuffer(raw, np.uint8).reshape((2,) + tuple(shape))
        r = np.ascontiguousarray(np.moveaxis(b, 0, -1)).view(np.uint16)[..., 0]
        return _ungrad(r, np.uint16).view(np.float16)
    raise ValueError(f)


_pool = None


def write_pack(path: Path, arrays: dict, workers: int = 8) -> int:
    """arrays: lr, hr (K,T,3,h,w) uint8, aux (K,T,4,h,w) fp16, hr_alpha (K,T,h,w) uint8. Atomic. Returns bytes.
    Frames are encoded on a thread pool (zlib releases the GIL)."""
    global _pool
    path = Path(path)
    K, T = arrays["lr"].shape[:2]
    jobs = [(n, arrays[n][k, t], arrays["lr"][k, t] if n == "hr" else None)
            for k in range(K) for t in range(T) for n in ARRAY_ORDER]
    if workers > 1:
        if _pool is None:
            from concurrent.futures import ThreadPoolExecutor
            _pool = ThreadPoolExecutor(workers)
        blobs = list(_pool.map(lambda j: encode_frame(*j), jobs))
    else:
        blobs = [encode_frame(*j) for j in jobs]
    head = {"arrays": {n: {"dtype": str(arrays[n].dtype), "shape": list(arrays[n].shape), "filter": FILTERS[n]}
                       for n in ARRAY_ORDER},
            "K": int(K), "T": int(T), "codec": f"zlib-{LEVEL}", "chunks": []}
    # header length depends on offsets: fix the width by formatting offsets after a first pass
    hb = b""
    for _ in range(20):
        off = len(MAGIC) + 8 + len(hb)
        chunks = []
        for b in blobs:
            chunks.append([off, len(b)])
            off += len(b)
        head["chunks"] = chunks
        nb = json.dumps(head, separators=(",", ":")).encode()
        if len(nb) == len(hb):
            hb = nb
            break
        hb = nb
    else:
        raise RuntimeError("pack header did not converge")
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(MAGIC + struct.pack("<Q", len(hb)) + hb)
        for b in blobs:
            f.write(b)
        size = f.tell()
    os.replace(tmp, path)
    return size


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        if f.read(8) != MAGIC:
            raise ValueError(f"{path}: not an S3DPACK1 file")
        (n,) = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(n))
    h["path"] = str(path)
    return h


def read_window(h: dict, k: int, t0: int, length: int, names=ARRAY_ORDER) -> dict:
    """Frames t0..t0+length-1 of camera k: {name: (L, ...) array}. One contiguous file read."""
    A = len(ARRAY_ORDER)
    T = h["T"]
    i0 = (k * T + t0) * A
    i1 = (k * T + t0 + length) * A
    ch = h["chunks"]
    start, end = ch[i0][0], ch[i1 - 1][0] + ch[i1 - 1][1]
    with open(h["path"], "rb") as f:
        f.seek(start)
        buf = memoryview(f.read(end - start))
    out = {}
    for n in names:
        meta = h["arrays"][n]
        out[n] = np.empty((length,) + tuple(meta["shape"][2:]), dtype=meta["dtype"])
    need_lr = "hr" in names
    for j in range(length):
        base = i0 + j * A
        lr = None
        for a, n in enumerate(ARRAY_ORDER):
            if n not in names and not (n == "lr" and need_lr):
                continue
            o, ln = ch[base + a]
            x = decode_frame(n, buf[o - start:o - start + ln], h["arrays"][n]["shape"][2:], lr)
            if n == "lr":
                lr = x
            if n in names:
                out[n][j] = x
    return out


def read_all(path: Path) -> dict:
    """Whole scenario as (K,T,...) arrays."""
    h = read_header(path)
    K, T = h["K"], h["T"]
    out = {n: np.empty(h["arrays"][n]["shape"], dtype=h["arrays"][n]["dtype"]) for n in ARRAY_ORDER}
    for k in range(K):
        w = read_window(h, k, 0, T)
        for n in ARRAY_ORDER:
            out[n][k] = w[n]
    return out
