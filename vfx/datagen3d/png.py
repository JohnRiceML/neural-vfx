"""Minimal PNG writer (no PIL in the venv): uint8 (H,W) gray or (H,W,3) RGB numpy -> file."""
from __future__ import annotations

import struct
import zlib

import numpy as np


def write_png(path, img: np.ndarray) -> None:
    img = np.ascontiguousarray(img, dtype=np.uint8)
    if img.ndim == 2:
        color, ch = 0, 1
    elif img.ndim == 3 and img.shape[2] == 3:
        color, ch = 2, 3
    else:
        raise ValueError(img.shape)
    h, w = img.shape[:2]
    raw = np.concatenate([np.zeros((h, 1), np.uint8), img.reshape(h, w * ch)], 1).tobytes()

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)
