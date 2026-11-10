"""Shared helpers for the 3D gate tests (owner: 3D test/eval engineer).

The 3D modules are written concurrently by other engineers: every test that needs one skips with a
clear reason until the module imports and exports the CLAUDE.md "3D phase" interface, so
`pytest tests3d -q` always runs.
"""
from __future__ import annotations

import dataclasses
import math
import warnings

import pytest
import torch
import torch.nn.functional as F

from vfx.core.fields3d import AMBIENT_TEMP, DOMAIN_SIZE, Camera

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
DEV = "cuda"


def import_or_skip(modname: str, attrs: tuple[str, ...] = ()):
    """Import a module under test, skipping (not erroring) if it or a needed attr is missing."""
    try:
        mod = __import__(modname, fromlist=["_"])
    except ImportError as e:
        pytest.skip(f"{modname} not importable yet: {e}")
    missing = [a for a in attrs if not hasattr(mod, a)]
    if missing:
        pytest.skip(f"{modname} does not export {missing} yet (CLAUDE.md 3D-phase interface)")
    return mod


def make_config(sim, **overrides):
    """SolverConfig3D() with overrides (field names fixed by CLAUDE.md: same as 2D + heat_diffusion,
    confinement_floor)."""
    cfg = sim.SolverConfig3D()
    for k in overrides:
        if not hasattr(cfg, k):
            pytest.fail(f"SolverConfig3D has no field '{k}' (interface in CLAUDE.md requires it)")
    if dataclasses.is_dataclass(cfg):
        return dataclasses.replace(cfg, **overrides)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ----------------------------------------------------------------------------- fields
def world_grid(res: int):
    """Cell-centre world coordinates, each (D,H,W) indexed [z, y, x]; y = 0 is the bottom."""
    c = (torch.arange(res, device=DEV, dtype=torch.float32) + 0.5) * (DOMAIN_SIZE / res)
    zz, yy, xx = torch.meshgrid(c, c, c, indexing="ij")
    return xx, yy, zz


def blob(res: int, cx: float, cy: float, cz: float, sigma: float) -> torch.Tensor:
    xx, yy, zz = world_grid(res)
    return torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) / (2 * sigma ** 2))


def compact_blob(res: int, cx: float, cy: float, cz: float, radius: float) -> torch.Tensor:
    """(1 - r^2/R^2)^2 inside R, exactly 0 outside: a blob with a known screen footprint."""
    xx, yy, zz = world_grid(res)
    r2 = ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) / radius ** 2
    return (1 - r2).clamp(min=0) ** 2


def smooth_noise(res: int, channels: int, seed: int, coarse: int = 6, amp: float = 1.0) -> torch.Tensor:
    """Smooth random field (C,D,H,W): coarse Gaussian noise, trilinearly upsampled twice (C1-ish)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    z = torch.randn(1, channels, coarse, coarse, coarse, device=DEV, generator=g)
    mid = F.interpolate(z, size=(res // 2,) * 3, mode="trilinear", align_corners=False)
    mid = F.avg_pool3d(mid, 3, 1, 1, count_include_pad=False)
    return amp * F.interpolate(mid, size=(res,) * 3, mode="trilinear", align_corners=False)[0]


def fluid_mask(state) -> torch.Tensor:
    return state.solid < 0.5


def mass(state) -> float:
    """World-space smoke mass: sum(density) * dx^3 over fluid cells (float64 accumulate)."""
    d = state.ch("density")
    return float((d * fluid_mask(state)).double().sum().item()) * state.meta.dx ** 3


def check_finite(state, where=""):
    bad = (~torch.isfinite(state.data)).sum().item()
    assert bad == 0, f"{bad} non-finite values {where}"


# ----------------------------------------------------------------------------- camera geometry
def camera_rays(cam: Camera, h: int, w: int):
    """Unit ray directions (h,w,3) of a standard pinhole camera (row 0 = top), plus the origin.
    Horizontal handedness is NOT assumed by the tests that use this (scenes are mirror-symmetric in x
    or only the vertical / radial geometry matters)."""
    pos = torch.tensor(cam.position, device=DEV, dtype=torch.float32)
    tgt = torch.tensor(cam.target, device=DEV, dtype=torch.float32)
    up = torch.tensor(cam.up, device=DEV, dtype=torch.float32)
    fwd = tgt - pos
    fwd = fwd / fwd.norm()
    right = torch.linalg.cross(fwd, up)
    right = right / right.norm()
    true_up = torch.linalg.cross(right, fwd)
    th = math.tan(math.radians(cam.fov_y_deg) / 2)
    ys = (1 - 2 * (torch.arange(h, device=DEV, dtype=torch.float32) + 0.5) / h) * th
    xs = (2 * (torch.arange(w, device=DEV, dtype=torch.float32) + 0.5) / w - 1) * th * (w / h)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    d = fwd + xx[..., None] * right + yy[..., None] * true_up
    return pos, d / d.norm(dim=-1, keepdim=True)


def rays_hit_sphere(cam: Camera, h: int, w: int, center, radius: float) -> torch.Tensor:
    """(h,w) bool: the pixel-centre ray passes within `radius` of `center`."""
    o, d = camera_rays(cam, h, w)
    c = torch.tensor(center, device=DEV, dtype=torch.float32)
    oc = c - o
    t = (d * oc).sum(-1).clamp(min=0)
    closest = o + t[..., None] * d
    return (closest - c).norm(dim=-1) <= radius


def rays_hit_box(cam: Camera, h: int, w: int, lo=(0.0, 0.0, 0.0), hi=(DOMAIN_SIZE,) * 3) -> torch.Tensor:
    o, d = camera_rays(cam, h, w)
    lo_t = torch.tensor(lo, device=DEV)
    hi_t = torch.tensor(hi, device=DEV)
    inv = 1.0 / torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
    t0, t1 = (lo_t - o) * inv, (hi_t - o) * inv
    tmin = torch.minimum(t0, t1).amax(-1)
    tmax = torch.maximum(t0, t1).amin(-1)
    return tmax >= tmin.clamp(min=0)


def dilate2d(mask: torch.Tensor, px: int) -> torch.Tensor:
    return F.max_pool2d(mask.float()[None, None], 2 * px + 1, 1, px)[0, 0] > 0.5


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a.float() - b.float()) ** 2).mean().item()
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


# ----------------------------------------------------------------------------- host syncs
class SyncCounter:
    """Counts host syncs that torch itself detects (torch.cuda.set_sync_debug_mode('warn')).
    Warp-internal syncs are invisible to this; it is the 'where feasible' check."""

    def __enter__(self):
        torch.cuda.synchronize()
        self._cm = warnings.catch_warnings(record=True)
        self.records = self._cm.__enter__()
        warnings.simplefilter("always")
        self._prev = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("warn")
        return self

    def __exit__(self, *exc):
        torch.cuda.set_sync_debug_mode(self._prev)
        self._cm.__exit__(*exc)
        self.syncs = [str(r.message) for r in self.records
                      if "synchronizing" in str(r.message).lower() and "prototype" not in str(r.message).lower()]
        return False


def ambient_like(t: torch.Tensor) -> torch.Tensor:
    return torch.full_like(t, AMBIENT_TEMP)
