"""Gate 3a checks: the native ray-marcher vs vfx.render3d.VolumeRenderer (native/README.md, "Gate 3a").

OWNER: native test engineer. Black box: the native side is reached only through nvfx_render.h / nvfx.h
(nvfx_render_ctypes.py, nvfx_ctypes.py). Reference: VolumeRenderer((1440, 2560), march_scale=0.5), default
RenderParams3D, draw(volume, camera, solid=state.solid, return_aux=True, velocity=state.velocity, prev_camera).

No top-level torch / warp / DLL import: the metric, camera and PNG helpers are pure numpy (unit-tested without a GPU).

Per (scene, camera) "case" (`run_case`):
  Python draw (cam, prev)                 -> rgb, alpha, depth, motion (moving camera + moving volume)
  native draw (cam, prev, AUX)       = A1 -> R1 colour, R2 alpha, R3 depth, R4 motion (moving camera)
  native draw (cam, prev, 0)              -> R5 (colour bit-identical to A1)
  native draw (cam, None, AUX) + Python (cam, None) -> R4 static camera (motion from the volume velocity only)
  native draw (cam, prev, AUX)       = A2 -> R6 (all four outputs bit-identical to A1; two other draws in between)
  Python draw (yaw(cam, 1 deg), prev)     -> NC R1 (must FAIL)
  motion with x/y swapped / negated       -> NC R4 (must FAIL)
"""
from __future__ import annotations

import hashlib
import math
import os
import struct
import zlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "native" / "build" / "test_results"
IMAGE_DIR = RESULT_DIR / "gate3a_images"
CACHE_DIR = ROOT / "native" / "build" / "gate3a_cache"

# ------------------------------------------------------------------ the contract (native/README.md, Gate 3a)
FRAME_H, FRAME_W = 1440, 2560
MARCH_SCALE = 0.5
STEPS = 150
SCENES = ("campfire", "plume")
CAMERAS = ("default", "orbit60", "closeup")
R1_MEAN_MAX, R1_P99_MAX = 1.5, 8.0           # 8-bit units, whole frame
R2_MEAN_MAX, R2_P99_MAX = 0.01, 0.05         # alpha
R3_MEAN_MAX = 0.01                           # depth, metres, where Python alpha > 0.05
R4_MEAN_MAX = 0.5                            # motion, px, where Python alpha > 0.05
AUX_MASK_ALPHA = 0.05
T2_BUDGET_MS = 1.5
T2_N, T2_WARM = 300, 30
CLOSEUP_MIN_COVERAGE = 1.0 / 3.0

# ------------------------------------------------------------------ harness choices (reported; not thresholds)
DOMAIN_CENTRE = (0.5, 0.5, 0.5)
ORBIT_DEG = 60.0                 # "an orbit of 60 degrees about the domain centre" (about the vertical axis)
PREV_ORBIT_DEG = 3.0             # moving camera: prev_camera = camera orbited by -3 deg about the domain centre
NC_YAW_DEG = 1.0                 # NC R1: Python camera rotated (yawed about its own position) by 1 deg
CLOSEUP_DISTS = (1.3, 1.15, 1.0, 0.9, 0.8, 0.7, 0.6)   # bench.camera_at(dist); the largest one reaching 1/3 coverage
BLOCK = 8                        # structured-error proxy block size
T2_VIEW_ORBIT_DEG = 0.25         # view-only redraws: the camera orbits this much per frame
DIFF_GAIN = 8


# ------------------------------------------------------------------ cameras (duck-typed fields3d.Camera)
@dataclass(frozen=True)
class Cam:
    position: tuple = (0.5, 0.55, -1.4)
    target: tuple = (0.5, 0.45, 0.5)
    up: tuple = (0.0, 1.0, 0.0)
    fov_y_deg: float = 40.0


def as_cam(c) -> Cam:
    return Cam(tuple(float(v) for v in c.position), tuple(float(v) for v in c.target), tuple(float(v) for v in c.up),
               float(c.fov_y_deg))


def to_vfx(c):
    from vfx.core.fields3d import Camera
    return Camera(position=tuple(c.position), target=tuple(c.target), up=tuple(c.up), fov_y_deg=float(c.fov_y_deg))


def _rot_y(v, deg: float, centre) -> tuple:
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    x, y, z = (v[0] - centre[0], v[1] - centre[1], v[2] - centre[2])
    return (centre[0] + ca * x + sa * z, centre[1] + y, centre[2] - sa * x + ca * z)


def orbit(c: Cam, deg: float, centre=DOMAIN_CENTRE) -> Cam:
    """Rotate position and target about the vertical axis through `centre` (right-handed about +y)."""
    return replace(c, position=_rot_y(c.position, deg, centre), target=_rot_y(c.target, deg, centre))


def yaw(c: Cam, deg: float) -> Cam:
    """Rotate the view direction by `deg` about the camera's own up axis (camera_basis: rt = up0 x fw, up = fw x rt),
    through the camera position: the view direction turns by exactly `deg`."""
    pos = np.asarray(c.position, np.float64)
    v = np.asarray(c.target, np.float64) - pos
    fw = v / np.linalg.norm(v)
    rt = np.cross(np.asarray(c.up, np.float64), fw)
    rt /= np.linalg.norm(rt)
    k = np.cross(fw, rt)
    a = math.radians(deg)
    vr = v * math.cos(a) + np.cross(k, v) * math.sin(a) + k * np.dot(k, v) * (1 - math.cos(a))
    return replace(c, target=tuple(float(x) for x in pos + vr))


def camera_at(dist: float) -> Cam:
    """vfx/render3d/bench.py camera_at (TARGET (0.5, 0.42, 0.5), pitch 0.20, fov 40), duplicated so the unit tests
    need no torch; test_unit_camera_at_matches_bench checks it against the source by AST."""
    t, pitch = (0.5, 0.42, 0.5), 0.20
    return Cam(position=(t[0], t[1] + dist * math.sin(pitch), t[2] - dist * math.cos(pitch)), target=t, fov_y_deg=40.0)


def cam_dict(c: Cam | None):
    return None if c is None else dict(position=list(c.position), target=list(c.target), up=list(c.up),
                                       fov_y_deg=c.fov_y_deg)


def angle_between_views(a: Cam, b: Cam) -> float:
    da = np.subtract(a.target, a.position)
    db = np.subtract(b.target, b.position)
    cs = float(np.dot(da, db) / (np.linalg.norm(da) * np.linalg.norm(db)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cs))))


# ------------------------------------------------------------------ metrics
def inv_cdf_percentile_int(absd: np.ndarray, q: float) -> int:
    """Exact q-th percentile (inverted CDF: smallest v with P(|d| <= v) >= q/100) of non-negative ints <= 255."""
    h = np.bincount(absd.ravel(), minlength=256)
    c = np.cumsum(h)
    return int(np.searchsorted(c, math.ceil(q / 100.0 * c[-1] - 1e-9)))


def structure_proxy(d: np.ndarray, block: int = BLOCK, seed: int = 0) -> dict:
    """Report-only proxy for "structured error" (NOT a gate; the lead reviews the x8 images).

    d: (H, W, C) signed error. B = per-channel 8x8 block mean of d. Dither-like (zero-mean, uncorrelated) error
    averages out in B; a bias over a region (a shifted edge, a wrong colour) survives.
      struct_energy       = sum(B^2) * block^2 / sum(d^2): the fraction of the error energy that is smooth at the
                            8x8 scale. Uncorrelated zero-mean noise gives ~1/block^2 = 0.016; a smooth error gives ~1.
      struct_energy_shuf  = the same with the error pixels randomly permuted (noise baseline, same histogram)
      struct_ratio        = struct_energy / struct_energy_shuf (~1 = no spatial structure beyond the histogram;
                            a global constant bias also gives ~1 but a high struct_energy)
      struct_l1           = mean|B| / mean|d| (sign-cancellation measure; equals 1 for any one-signed error)
      blocks_gt1          = fraction of blocks whose |mean error| > 1 level in any channel
    """
    H, W, Cc = d.shape
    Hb, Wb = H // block * block, W // block * block
    df = d[:Hb, :Wb].astype(np.float32)

    def stats(x):
        B = x.reshape(Hb // block, block, Wb // block, block, Cc).mean(axis=(1, 3))
        ma = float(np.abs(x).mean())
        e = float((x * x).sum(dtype=np.float64))
        return (float(np.abs(B).mean()) / ma if ma > 0 else 0.0,
                float((B * B).sum(dtype=np.float64)) * block * block / e if e > 0 else 0.0, B)

    l1, en, B = stats(df)
    rng = np.random.default_rng(seed)
    flat = df.reshape(-1, Cc)
    sh = flat[rng.permutation(flat.shape[0])].reshape(df.shape)
    _, ens, _ = stats(sh)
    return dict(struct_energy=en, struct_energy_shuf=ens,
                struct_ratio=(en / ens) if ens > 0 else (0.0 if en == 0 else math.inf),
                struct_l1=l1, blocks_gt1=float((np.abs(B) > 1.0).any(-1).mean()), block=block,
                noise_floor=1.0 / (block * block))


def r1_metrics(nat_rgb: np.ndarray, py_rgb: np.ndarray, proxy: bool = True) -> dict:
    assert nat_rgb.shape == py_rgb.shape and nat_rgb.dtype == np.uint8 and py_rgb.dtype == np.uint8
    d = nat_rgb.astype(np.int16) - py_rgb.astype(np.int16)
    a = np.abs(d).astype(np.uint8)
    mean = float(a.mean(dtype=np.float64))
    p99 = inv_cdf_percentile_int(a, 99.0)
    pix = a.max(-1)
    out = dict(mean_abs=mean, p99=p99, max_abs=int(a.max()), p99_pixel_maxch=inv_cdf_percentile_int(pix, 99.0),
               frac_values_ne=float((a > 0).mean()), frac_pixels_gt1=float((pix > 1).mean()),
               frac_pixels_gt8=float((pix > 8).mean()), mean_signed=[float(v) for v in d.reshape(-1, 3).mean(0)],
               bit_identical=bool(not a.any()))
    out["pass"] = bool(mean <= R1_MEAN_MAX and p99 <= R1_P99_MAX)
    if proxy:
        out["structure"] = structure_proxy(d)
    return out


def r2_metrics(nat_alpha: np.ndarray, py_alpha: np.ndarray) -> dict:
    a = np.abs(nat_alpha.astype(np.float32) - py_alpha.astype(np.float32))
    mean, p99 = float(a.mean(dtype=np.float64)), float(np.percentile(a, 99))
    return dict(mean_abs=mean, p99=p99, max_abs=float(a.max()), pass_=bool(mean <= R2_MEAN_MAX and p99 <= R2_P99_MAX))


def masked(py_alpha: np.ndarray) -> np.ndarray:
    return py_alpha.astype(np.float32) > AUX_MASK_ALPHA


def r3_metrics(nat_depth, py_depth, py_alpha) -> dict:
    m = masked(py_alpha)
    n = int(m.sum())
    if n == 0:
        return dict(n_px=0, pass_=False, mean_abs=math.nan, note="no pixel with Python alpha > 0.05")
    a = np.abs(nat_depth[m].astype(np.float32) - py_depth[m].astype(np.float32))
    mean = float(a.mean(dtype=np.float64))
    return dict(n_px=n, mean_abs=mean, p99=float(np.percentile(a, 99)), max_abs=float(a.max()),
                py_depth_median=float(np.median(py_depth[m])), pass_=bool(mean <= R3_MEAN_MAX))


def r4_metrics(nat_motion, py_motion, py_alpha) -> dict:
    """Gate on the mean end-point error |n - p|_2 over the mask (stricter than the mean per-component |.|, also
    reported). Motion arrays are (2, H, W): x right, y down."""
    m = masked(py_alpha)
    n = int(m.sum())
    if n == 0:
        return dict(n_px=0, pass_=False, epe_mean=math.nan, note="no pixel with Python alpha > 0.05")
    dx = nat_motion[0][m].astype(np.float32) - py_motion[0][m].astype(np.float32)
    dy = nat_motion[1][m].astype(np.float32) - py_motion[1][m].astype(np.float32)
    epe = np.sqrt(dx * dx + dy * dy)
    pm = np.sqrt(py_motion[0][m].astype(np.float32) ** 2 + py_motion[1][m].astype(np.float32) ** 2)
    mean = float(epe.mean(dtype=np.float64))
    return dict(n_px=n, epe_mean=mean, epe_p99=float(np.percentile(epe, 99)), epe_max=float(epe.max()),
                abs_x_mean=float(np.abs(dx).mean(dtype=np.float64)), abs_y_mean=float(np.abs(dy).mean(dtype=np.float64)),
                component_mean=float(0.5 * (np.abs(dx).mean(dtype=np.float64) + np.abs(dy).mean(dtype=np.float64))),
                py_motion_median_px=float(np.median(pm)),
                py_motion_mean_xy=[float(py_motion[0][m].mean(dtype=np.float64)),
                                   float(py_motion[1][m].mean(dtype=np.float64))],
                pass_=bool(mean <= R4_MEAN_MAX))


def motion_swapped(m: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(m[::-1])


def motion_negated(m: np.ndarray) -> np.ndarray:
    return -m


def bit_compare(a: np.ndarray, b: np.ndarray) -> dict:
    assert a.shape == b.shape and a.dtype == b.dtype
    ua = a.view(np.uint8).ravel() if a.dtype == np.uint8 else a.view(np.uint32).ravel()
    ub = b.view(np.uint8).ravel() if b.dtype == np.uint8 else b.view(np.uint32).ravel()
    ne = int(np.count_nonzero(ua != ub))
    return dict(identical=ne == 0, n_differ=ne, n=int(ua.size))


def sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def coverage(py_alpha: np.ndarray) -> dict:
    a = py_alpha.astype(np.float32)
    return dict(alpha_gt_005=float((a > 0.05).mean()), alpha_gt_002=float((a > 0.02).mean()))


# ------------------------------------------------------------------ images (minimal PNG encoder, zlib only)
def write_png(path: Path, img: np.ndarray, level: int = 4) -> None:
    img = np.ascontiguousarray(img)
    assert img.dtype == np.uint8
    if img.ndim == 2:
        img = img[..., None]
    h, w, c = img.shape
    ctype = {1: 0, 3: 2, 4: 6}[c]
    raw = np.empty((h, 1 + w * c), np.uint8)
    raw[:, 0] = 0                                   # filter type None
    raw[:, 1:] = img.reshape(h, w * c)

    def chunk(t: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw.tobytes(), level)) + chunk(b"IEND", b""))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(png)


def read_png(path: Path) -> np.ndarray:
    """Decoder for the files write_png produces (8-bit, filter 0, one IDAT); used by the unit test only."""
    b = Path(path).read_bytes()
    assert b[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, hdr = 8, b"", None
    while pos < len(b):
        n = struct.unpack(">I", b[pos:pos + 4])[0]
        t, data = b[pos + 4:pos + 8], b[pos + 8:pos + 8 + n]
        assert struct.unpack(">I", b[pos + 8 + n:pos + 12 + n])[0] == zlib.crc32(t + data) & 0xFFFFFFFF
        if t == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", data)
        elif t == b"IDAT":
            idat += data
        pos += 12 + n
    w, h, _, ctype = hdr[:4]
    c = {0: 1, 2: 3, 6: 4}[ctype]
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + w * c)
    assert (raw[:, 0] == 0).all()
    return raw[:, 1:].reshape(h, w, c)


def diff_image(nat_rgb: np.ndarray, py_rgb: np.ndarray, gain: int = DIFF_GAIN) -> np.ndarray:
    d = np.abs(nat_rgb.astype(np.int16) - py_rgb.astype(np.int16)) * gain
    return np.clip(d, 0, 255).astype(np.uint8)


def thumb(img: np.ndarray, f: int = 4, mode: str = "mean") -> np.ndarray:
    h, w, c = img.shape
    x = img[:h // f * f, :w // f * f].reshape(h // f, f, w // f, f, c)
    if mode == "max":
        return x.max(axis=(1, 3))
    return (x.astype(np.float32).mean(axis=(1, 3)) + 0.5).astype(np.uint8)


def contact_sheet(rows: list[list[np.ndarray]], gap: int = 4) -> np.ndarray:
    """rows of equally sized (h, w, 3) uint8 tiles -> one image with a grey grid."""
    h, w, _ = rows[0][0].shape
    ncol = max(len(r) for r in rows)
    out = np.full((len(rows) * (h + gap) + gap, ncol * (w + gap) + gap, 3), 64, np.uint8)
    for i, r in enumerate(rows):
        for j, t in enumerate(r):
            y, x = gap + i * (h + gap), gap + j * (w + gap)
            out[y:y + h, x:x + w] = t
    return out


# ------------------------------------------------------------------ GPU side (imports inside)
def gpu_snapshot() -> dict:
    from gate2_checks import gpu_snapshot as g
    return g()


def stats(xs) -> dict:
    a = np.asarray(xs, np.float64)
    if a.size == 0:
        return dict(n=0)
    return dict(median=float(np.median(a)), p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)),
                p95=float(np.percentile(a, 95)), min=float(a.min()), max=float(a.max()), n=int(a.size))


class PyScene:
    """Python Solver3D state after STEPS steps + numpy copies for the native upload."""

    def __init__(self, scene: str, steps: int = STEPS):
        from vfx.core.fields3d import CH3, Volume
        from vfx.sim3d import Solver3D, SolverConfig3D
        from nvfx_scenes import DT, get_scene
        self.name, self.steps = scene, steps
        self.sc = get_scene(scene)
        self.solver = Solver3D(64, SolverConfig3D(**self.sc.config_overrides))
        st = self.solver.reset(self.sc.obstacles)
        for i in range(steps):
            st = self.solver.step(st, self.sc.forces(i), DT) or st
        self.st = st
        self.i = steps
        self.CH3, self.Volume, self.DT = CH3, Volume, DT

    def volume(self):
        return self.Volume(self.st.data[self.CH3["density"]], self.st.data[self.CH3["temperature"]])

    def numpy_state(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.ascontiguousarray(self.st.data.detach().cpu().numpy(), np.float32),
                np.ascontiguousarray(self.st.solid.detach().cpu().numpy(), np.float32))

    def step(self):
        self.st = self.solver.step(self.st, self.sc.forces(self.i), self.DT) or self.st
        self.i += 1


class PyRender:
    def __init__(self):
        import torch
        from vfx.render3d import VolumeRenderer
        self.torch = torch
        self.r = VolumeRenderer((FRAME_H, FRAME_W), march_scale=MARCH_SCALE)

    def draw(self, ps: PyScene, cam: Cam, prev: Cam | None, aux: bool = True, reuse: bool = False) -> dict:
        t = self.torch
        res = self.r.draw(ps.volume(), to_vfx(cam), ps.st.solid, reuse, return_aux=aux,
                          velocity=ps.st.velocity if aux else None, prev_camera=to_vfx(prev) if prev is not None else None)
        if not aux:
            out = dict(color=res.cpu().numpy().copy())
        else:
            frame, a = res
            out = dict(color=frame.cpu().numpy().copy(), alpha=a["alpha"].float().cpu().numpy(),
                       depth=a["depth"].float().cpu().numpy(), motion=a["motion"].float().cpu().numpy())
        t.cuda.synchronize()
        return out

    def march_size(self) -> tuple[int, int]:
        return int(self.r.mw), int(self.r.mh)


def choose_closeup(pr: PyRender, ps: PyScene) -> tuple[Cam, list]:
    """The largest bench.camera_at distance whose Python alpha > 0.05 coverage is >= 1/3 (prev camera irrelevant)."""
    tried = []
    for d in CLOSEUP_DISTS:
        c = camera_at(d)
        cov = coverage(pr.draw(ps, c, None)["alpha"])
        tried.append(dict(dist=d, **cov))
        if cov["alpha_gt_005"] >= CLOSEUP_MIN_COVERAGE:
            return c, tried
    return camera_at(CLOSEUP_DISTS[-1]), tried


def cameras_for(pr: PyRender, ps: PyScene) -> tuple[dict, dict]:
    base = Cam()
    close, tried = choose_closeup(pr, ps)
    cams = {"default": base, "orbit60": orbit(base, ORBIT_DEG), "closeup": close}
    return cams, {"closeup_search": tried}


def prev_of(c: Cam) -> Cam:
    return orbit(c, -PREV_ORBIT_DEG)


def _strip(d: dict) -> dict:
    d = dict(d)
    if "pass_" in d:
        d["pass"] = d.pop("pass_")
    return d


def save_case_images(tag: str, nat_rgb, py_rgb, extra: dict | None = None) -> dict:
    files = {}
    for k, img in (("native", nat_rgb), ("python", py_rgb), (f"diff_x{DIFF_GAIN}", diff_image(nat_rgb, py_rgb))):
        p = IMAGE_DIR / f"{tag}_{k}.png"
        write_png(p, img)
        files[k] = str(p)
    for k, img in (extra or {}).items():
        p = IMAGE_DIR / f"{tag}_{k}.png"
        write_png(p, img)
        files[k] = str(p)
    return files


def run_case(nr, pr: PyRender, ps: PyScene, cam_name: str, cam: Cam) -> tuple[dict, dict, list]:
    """All per-(scene, camera) checks. Returns (results by check id, info, contact-sheet row). Arrays are dropped."""
    from nvfx_render_ctypes import NVFX_RENDER_AUX
    tag = f"{ps.name}_{cam_name}"
    prev = prev_of(cam)
    res: dict = {}
    py = pr.draw(ps, cam, prev)
    cov = coverage(py["alpha"])
    nr.draw(cam, prev, NVFX_RENDER_AUX)
    a1 = nr.all_outputs()
    nat_rgb = np.ascontiguousarray(a1["color"][..., :3])
    a_channel = dict(all_255=bool((a1["color"][..., 3] == 255).all()))

    r1 = r1_metrics(nat_rgb, py["color"])
    s = r1["structure"]
    res[f"R1_{tag}"] = dict(pass_=r1["pass"], r1=r1, coverage=cov, a_channel=a_channel, visual_review="PENDING (lead)",
                            summary=(f"R1 {tag}: mean abs {r1['mean_abs']:.3f} (<= {R1_MEAN_MAX}), p99 {r1['p99']} "
                                     f"(<= {R1_P99_MAX:g}), max {r1['max_abs']}; values differing {r1['frac_values_ne']:.3f}; "
                                     f"proxy struct_energy {s['struct_energy']:.3f} vs shuffled {s['struct_energy_shuf']:.3f} "
                                     f"(x{s['struct_ratio']:.2f}), blocks>1 {s['blocks_gt1']:.4f}; visual review PENDING"))
    r2 = r2_metrics(a1["alpha"], py["alpha"])
    res[f"R2_{tag}"] = dict(r2, summary=f"R2 {tag}: alpha mean abs {r2['mean_abs']:.5f} (<= {R2_MEAN_MAX}), p99 "
                                         f"{r2['p99']:.5f} (<= {R2_P99_MAX}), max {r2['max_abs']:.4f}")
    r3 = r3_metrics(a1["depth"], py["depth"], py["alpha"])
    res[f"R3_{tag}"] = dict(r3, summary=f"R3 {tag}: depth mean abs {r3.get('mean_abs', math.nan):.5f} m (<= {R3_MEAN_MAX}) "
                                         f"over {r3['n_px']} px with Python alpha > {AUX_MASK_ALPHA}")
    r4 = r4_metrics(a1["motion"], py["motion"], py["alpha"])
    res[f"R4_{tag}_movingcam"] = dict(r4, prev_orbit_deg=-PREV_ORBIT_DEG, summary=(
        f"R4 {tag} moving camera ({PREV_ORBIT_DEG:g} deg/step) + volume: EPE mean {r4.get('epe_mean', math.nan):.4f} px "
        f"(<= {R4_MEAN_MAX}), mean |dx|,|dy| {r4.get('abs_x_mean', math.nan):.4f}, {r4.get('abs_y_mean', math.nan):.4f}; "
        f"Python |motion| median {r4.get('py_motion_median_px', math.nan):.1f} px over {r4['n_px']} px"))
    for kind, fn in (("swapxy", motion_swapped), ("negate", motion_negated)):
        nc = r4_metrics(fn(a1["motion"]), py["motion"], py["alpha"])
        ok = not nc["pass_"]
        res[f"NC_R4_{kind}_{tag}"] = dict(expected="FAIL", check_pass=nc["pass_"], pass_=ok, metrics=nc, summary=(
            f"NC R4 {kind} {tag}: native motion {kind} vs Python: EPE mean {nc.get('epe_mean', math.nan):.3f} px -> R4 "
            f"{'PASS (control NOT OK)' if nc['pass_'] else 'FAIL (control ok)'}"))

    # R5: aux off vs on, colour bit-identical
    nr.draw(cam, prev, 0)
    c0 = nr.color()
    b5 = bit_compare(c0, a1["color"])
    res[f"R5_{tag}"] = dict(pass_=b5["identical"], **b5, summary=f"R5 {tag}: colour without AUX vs with AUX: "
                                                                 f"{'bit-identical' if b5['identical'] else str(b5['n_differ']) + ' bytes differ'}")
    del c0

    # R4 static camera (moving volume only)
    py_s = pr.draw(ps, cam, None)
    nr.draw(cam, None, NVFX_RENDER_AUX)
    nm = nr.motion()
    r4s = r4_metrics(nm, py_s["motion"], py_s["alpha"])
    col_static_same = bit_compare(nr.color(), a1["color"])["identical"]
    res[f"R4_{tag}_static"] = dict(r4s, colour_same_as_moving_cam=col_static_same, summary=(
        f"R4 {tag} static camera, moving volume: EPE mean {r4s.get('epe_mean', math.nan):.4f} px (<= {R4_MEAN_MAX}); "
        f"Python |motion| median {r4s.get('py_motion_median_px', math.nan):.2f} px, mean (x, y) "
        f"{', '.join(f'{v:.2f}' for v in r4s.get('py_motion_mean_xy', [math.nan, math.nan]))}"))
    del nm, py_s

    # R6: same inputs again (two different draws in between) -> all outputs bit-identical
    nr.draw(cam, prev, NVFX_RENDER_AUX)
    a2 = nr.all_outputs()
    b6 = {k: bit_compare(a2[k], a1[k]) for k in a1}
    ok6 = all(v["identical"] for v in b6.values())
    res[f"R6_{tag}"] = dict(pass_=ok6, per_output=b6, summary=f"R6 {tag}: repeat draw: " + ", ".join(
        f"{k} {'identical' if v['identical'] else str(v['n_differ']) + ' differ'}" for k, v in b6.items()))
    del a2

    # NC R1: Python with the camera yawed by 1 degree vs native
    cam_nc = yaw(cam, NC_YAW_DEG)
    py_nc = pr.draw(ps, cam_nc, prev_of(cam_nc), aux=False)
    nc1 = r1_metrics(nat_rgb, py_nc["color"], proxy=True)
    write_png(IMAGE_DIR / f"{tag}_nc_yaw1_diff_x{DIFF_GAIN}.png", diff_image(nat_rgb, py_nc["color"]))
    res[f"NC_R1_yaw1_{tag}"] = dict(expected="FAIL", check_pass=nc1["pass"], pass_=not nc1["pass"], r1=nc1,
                                    yaw_deg=angle_between_views(cam, cam_nc), summary=(
        f"NC R1 yaw {NC_YAW_DEG:g} deg {tag}: mean abs {nc1['mean_abs']:.3f}, p99 {nc1['p99']} -> R1 "
        f"{'PASS (control NOT OK)' if nc1['pass'] else 'FAIL (control ok)'}; proxy struct_energy "
        f"{nc1['structure']['struct_energy']:.3f} (x{nc1['structure']['struct_ratio']:.2f} shuffled)"))
    del py_nc

    files = save_case_images(tag, nat_rgb, py["color"])
    res[f"R1_{tag}"]["images"] = files
    row = [thumb(py["color"]), thumb(nat_rgb), thumb(diff_image(nat_rgb, py["color"]), mode="max")]
    info = dict(camera=cam_dict(cam), prev_camera=cam_dict(prev), coverage=cov, native_sha256=sha256(nat_rgb),
                python_sha256=sha256(py["color"]), py_rgb=py["color"])
    return res, info, row          # every entry carries `pass_` (the test turns it into `pass`)


def python_self_controls(pr: PyRender, ps: PyScene, cam: Cam) -> dict:
    """HC3: the comparator on Python vs Python. Same inputs twice -> R1 PASS (expected bit-identical); camera yawed by
    1 deg -> R1 must FAIL; motion swapped / negated -> R4 must FAIL. Needs CUDA only (no DLL)."""
    prev = prev_of(cam)
    a = pr.draw(ps, cam, prev)
    b = pr.draw(ps, cam, prev)
    out = {}
    r = r1_metrics(b["color"], a["color"], proxy=False)
    bi = {k: bit_compare(a[k], b[k])["identical"] for k in a}
    out[f"HC3_py_self_{ps.name}"] = dict(pass_=r["pass"], r1=r, bit_identical=bi, summary=(
        f"HC3 Python vs Python {ps.name}: R1 mean {r['mean_abs']:.3f}, p99 {r['p99']}; bit-identical {bi}"))
    cn = yaw(cam, NC_YAW_DEG)
    c = pr.draw(ps, cn, prev_of(cn), aux=False)
    r = r1_metrics(c["color"], a["color"], proxy=True)
    out[f"HC3_py_yaw1_{ps.name}"] = dict(expected="FAIL", check_pass=r["pass"], pass_=not r["pass"], r1=r, summary=(
        f"HC3 Python yaw 1 deg vs Python {ps.name}: mean {r['mean_abs']:.3f}, p99 {r['p99']} -> "
        f"{'PASS (control NOT OK)' if r['pass'] else 'FAIL (control ok)'}; proxy struct_energy {r['structure']['struct_energy']:.3f} (x{r['structure']['struct_ratio']:.2f})"))
    for kind, fn in (("swapxy", motion_swapped), ("negate", motion_negated)):
        m = r4_metrics(fn(a["motion"]), a["motion"], a["alpha"])
        out[f"HC3_py_r4_{kind}_{ps.name}"] = dict(expected="FAIL", check_pass=m["pass_"], pass_=not m["pass_"], metrics=m,
                                                  summary=f"HC3 Python motion {kind} vs Python {ps.name}: EPE "
                                                          f"{m['epe_mean']:.3f} px -> {'PASS (NOT OK)' if m['pass_'] else 'FAIL (ok)'}")
    return {k: _strip(v) for k, v in out.items()}


# ------------------------------------------------------------------ dither-mode cache (for the subprocess probe)
def write_dither_cache(path: Path, scenes: dict) -> None:
    """scenes: name -> dict(data, solid, cases: {cam_name: dict(camera, prev_camera, py_rgb)})"""
    arrays, meta = {}, {}
    for s, v in scenes.items():
        arrays[f"{s}__data"], arrays[f"{s}__solid"] = v["data"], v["solid"]
        meta[s] = {}
        for c, cv in v["cases"].items():
            arrays[f"{s}__{c}__py_rgb"] = cv["py_rgb"]
            meta[s][c] = dict(camera=cv["camera"], prev_camera=cv["prev_camera"])
    import json
    arrays["meta"] = np.array(json.dumps(meta))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


# ------------------------------------------------------------------ T2
def native_t2(ns, nr, ps_state: tuple, forces, first_step: int, cam: Cam, mode: str, n=T2_N, warm=T2_WARM) -> dict:
    """mode 'full': step the native solver, then draw (AUX) -> a new volume every frame.
    mode 'view': one full draw, then REUSE_VOLUME|AUX draws with the camera orbiting T2_VIEW_ORBIT_DEG per frame."""
    from nvfx_render_ctypes import NVFX_RENDER_AUX, NVFX_RENDER_REUSE_VOLUME
    from nvfx_scenes import DT
    ns.upload(*ps_state)
    i = first_step
    c = cam
    flags = NVFX_RENDER_AUX
    if mode == "view":
        nr.draw(c, prev_of(c), NVFX_RENDER_AUX)
        flags |= NVFX_RENDER_REUSE_VOLUME

    def frame():
        nonlocal i, c
        if mode == "full":
            ns.step(forces(i), DT)
            i += 1
            nr.draw(c, prev_of(c), flags)
        else:
            p = c
            c = orbit(c, T2_VIEW_ORBIT_DEG)
            nr.draw(c, p, flags)

    for _ in range(warm):
        frame()
    ns.wait_idle()
    base = nr.n_draws
    samples: dict = {}
    method = "non-blocking get_timings after each draw"

    def grab():
        t = nr.timings()
        if t is not None and t["draw_index"] > base:
            samples.setdefault(t["draw_index"], t)

    k = 0
    while len(samples) < n and k < 20 * n:
        frame()
        k += 1
        grab()
    if len(samples) < n:
        method = "wait_idle + get_timings after each draw (non-blocking reads gave too few distinct draws)"
        while len(samples) < n and k < 40 * n:
            frame()
            k += 1
            ns.wait_idle()
            grab()
    ns.wait_idle()
    grab()
    ts = list(samples.values())
    want_reuse = mode == "view"
    flag_ok = [bool(t["flags"] & NVFX_RENDER_REUSE_VOLUME) == want_reuse for t in ts]
    out = {s: stats([t[s] for t in ts]) for s in ("prep", "march", "post", "total")}
    out.update(method=method, draws=k, flags_seen=sorted({t["flags"] for t in ts}), flags_ok=bool(all(flag_ok)),
               n_flag_mismatch=int(len(flag_ok) - sum(flag_ok)))
    return out


def python_t2(pr: PyRender, ps: PyScene, cam: Cam, mode: str, n=T2_N, warm=T2_WARM) -> dict:
    """VolumeRenderer.timings (CUDA events, prep / view / total) with last_timings(sync=True) after each draw.
    'full': Python Solver3D step + draw (new volume); 'view': reuse_volume=True draws with an orbiting camera."""
    r = pr.r
    c = cam
    vel = ps.st.velocity
    if mode == "view":
        r.draw(ps.volume(), to_vfx(c), ps.st.solid, False, return_aux=True, velocity=vel, prev_camera=to_vfx(prev_of(c)))
    ts = []
    for k in range(warm + n):
        if mode == "full":
            ps.step()
            r.draw(ps.volume(), to_vfx(c), ps.st.solid, False, return_aux=True, velocity=ps.st.velocity,
                   prev_camera=to_vfx(prev_of(c)))
        else:
            p = c
            c = orbit(c, T2_VIEW_ORBIT_DEG)
            r.draw(ps.volume(), to_vfx(c), ps.st.solid, True, return_aux=True, velocity=vel, prev_camera=to_vfx(p))
        t = r.last_timings(sync=True)
        if k >= warm:
            ts.append(t)
    out = {s: stats([t[s] for t in ts]) for s in ("prep", "view", "total")}
    out["what"] = "VolumeRenderer.timings via last_timings(sync=True): CUDA events around the draw (prep, view, total)"
    return out


def env_dither_mode() -> str | None:
    return os.environ.get("NVFX_RENDER_DITHER_MODE")
