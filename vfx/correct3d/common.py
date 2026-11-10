"""Shared pieces of the correction study: scenes, restriction, diagnostics, a tiny PNG writer.

Everything goes through the public Solver3D interface (reset / step / project / divergence);
vfx/sim3d is not modified. Scenes use the sandbox dissipation (density 1.2 /s, cooling 5 /s).
"""
from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from vfx.core.fields3d import (AMBIENT_TEMP, CH3, Emitter3D, Forces3D, Impulse3D, Obstacle3D,
                               rasterize_obstacles_3d)

VEL = [CH3["vel_x"], CH3["vel_y"], CH3["vel_z"]]
# Datasets + checkpoints of this study live on D: (C: is full; lead's instruction), not DATA_ROOT.
import os as _os
from pathlib import Path as _Path
CORR_DATA = _Path(_os.environ.get("CORR3D_DATA", r"D:\neural-vfx-data"))
CORR_DATASETS = CORR_DATA / "datasets"
CORR_CKPT = CORR_DATA / "checkpoints" / "corr3d"
SANDBOX_SIM = {"density_dissipation": 1.2, "temperature_cooling": 5.0}


# ---------------------------------------------------------------------------------------- scenes
@dataclass
class Scene:
    name: str
    emitters: list
    obstacles: list = field(default_factory=list)
    wind: tuple = (0.0, 0.0, 0.0)
    impulses: list = field(default_factory=list)          # [(every, phase, Impulse3D)]
    sim: dict = field(default_factory=lambda: dict(SANDBOX_SIM))
    tags: list = field(default_factory=list)

    def forces(self, step: int) -> Forces3D:
        imps = [imp for every, phase, imp in self.impulses if step % every == phase]
        return Forces3D(emitters=list(self.emitters), impulses=imps, wind=tuple(self.wind))


def _fire(pos, r=0.06, T=1300.0, fuel=8.0, vy=0.35):
    return Emitter3D(pos=pos, radius=r, density_rate=0.15, temperature=T, fuel_rate=fuel, velocity=(0.0, vy, 0.0))


def _smoke(pos, r=0.07, T=620.0, rate=3.0, vy=0.55):
    return Emitter3D(pos=pos, radius=r, density_rate=rate, temperature=T, fuel_rate=0.0, velocity=(0.0, vy, 0.0))


def named_scenes() -> dict[str, Scene]:
    """The six characterization scenes (held out from training: hand-made, not from the sampler)."""
    c = (0.5, 0.09, 0.5)
    return {
        "fire": Scene("fire", [_fire(c)], tags=["fire"]),
        "smoke": Scene("smoke", [_smoke(c)], tags=["smoke"]),
        "smoke_sphere": Scene("smoke_sphere", [_smoke(c)], [Obstacle3D("sphere", (0.52, 0.45, 0.5), (0.1, 0.1, 0.1))],
                              tags=["smoke", "obstacle"]),
        "fire_wind": Scene("fire_wind", [_fire((0.35, 0.09, 0.5))], wind=(0.45, 0.0, 0.1), tags=["fire", "wind"]),
        "smoke_push": Scene("smoke_push", [_smoke(c)],
                            impulses=[(15, 7, Impulse3D(pos=(0.45, 0.35, 0.5), radius=0.12, velocity=(1.5, 0.0, 0.4)))],
                            tags=["smoke", "impulse"]),
        # OOD: in-flow box slab + strong wind + mixed emitters
        "ood_box_wind": Scene("ood_box_wind", [_fire((0.4, 0.09, 0.45)), _smoke((0.6, 0.09, 0.55), r=0.09, rate=4.0)],
                              [Obstacle3D("box", (0.55, 0.5, 0.5), (0.18, 0.04, 0.12))], wind=(1.2, 0.0, -0.3),
                              tags=["fire", "smoke", "obstacle", "box", "strong_wind", "ood"]),
    }


def sampled_scene(seed: int, split: str) -> Scene:
    """A scene from the route-B scenario sampler (vfx/datagen3d, read-only use), sandbox-like ranges."""
    from vfx.datagen3d.scenarios import sample_scenario
    sc = sample_scenario(seed, split)
    return Scene(f"{split}_{seed}", sc.emitters, sc.obstacles, sc.wind, sc.impulses, dict(sc.sim), list(sc.tags))


# ---------------------------------------------------------------------------------------- solvers
_SOLVERS: dict = {}


def get_solver(res: int, cfg_kw: dict | None = None, tag: str = ""):
    """Cached Solver3D per (res, tag); config replaced on every call (no graph recapture needed)."""
    from vfx.sim3d import Solver3D, SolverConfig3D
    key = (res, tag)
    if key not in _SOLVERS:
        _SOLVERS[key] = Solver3D(res, SolverConfig3D())
    s = _SOLVERS[key]
    s.config = SolverConfig3D(**(cfg_kw or {}))
    return s


def restrict(data: torch.Tensor, res: int = 64) -> torch.Tensor:
    """(C,N,N,N) -> (C,res,res,res) box average (the 'restricted fine' state)."""
    f = data.shape[-1] // res
    return data if f == 1 else F.avg_pool3d(data[None], f)[0]


def solid_at(obstacles, res, device="cuda"):
    return rasterize_obstacles_3d(list(obstacles), res, device=device)


# ---------------------------------------------------------------------------------------- diagnostics
def _coords(n, device):
    return (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n


@torch.no_grad()
def diagnostics(data: torch.Tensor) -> torch.Tensor:
    """GPU (8,): smoke mass, smoke centroid y, smoke 95% height, heat, heat centroid y,
    kinetic energy (0.5 sum |u|^2 dV), density-weighted mean v_y, smoke centroid x.
    Any resolution (volume integrals, world units)."""
    n = data.shape[-1]
    dv = 1.0 / n ** 3
    y = _coords(n, data.device).view(1, n, 1)
    x = _coords(n, data.device).view(1, 1, n)
    d = data[CH3["density"]].clamp_min(0)
    h = (data[CH3["temperature"]] - AMBIENT_TEMP).clamp_min(0)
    m = d.sum() * dv
    cy = (d * y).sum() * dv / (m + 1e-12)
    cx = (d * x).sum() * dv / (m + 1e-12)
    prof = d.sum(dim=(0, 2))                        # (n,) over y
    cdf = prof.cumsum(0) / (prof.sum() + 1e-12)
    h95 = ((cdf < 0.95).float().sum() + 0.5) / n
    H = h.sum() * dv
    hy = (h * y).sum() * dv / (H + 1e-12)
    u = data[VEL]
    ke = 0.5 * (u * u).sum() * dv
    vy = (d * data[CH3["vel_y"]]).sum() * dv / (m + 1e-12)
    return torch.stack([m, cy, h95, H, hy, ke, vy, cx])


DIAG_NAMES = ["mass", "cy", "h95", "heat", "hy", "ke", "vy", "cx"]


@torch.no_grad()
def ke_spectrum(vel: torch.Tensor) -> torch.Tensor:
    """(3,n,n,n) velocity -> (n//2,) kinetic-energy power per integer wavenumber (cycles/domain)."""
    n = vel.shape[-1]
    U = torch.fft.rfftn(vel, dim=(-3, -2, -1)) / n ** 3
    p = (U.real ** 2 + U.imag ** 2).sum(0)
    w = torch.full_like(p, 2.0)
    w[..., 0] = 1.0
    if n % 2 == 0:
        w[..., -1] = 1.0
    f = torch.fft.fftfreq(n, device=vel.device) * n
    fr = torch.fft.rfftfreq(n, device=vel.device) * n
    k = torch.sqrt(f[:, None, None] ** 2 + f[None, :, None] ** 2 + fr[None, None, :] ** 2)
    kb = torch.round(k).long().clamp_max(n // 2)
    out = torch.zeros(n // 2 + 1, device=vel.device)
    out.scatter_add_(0, kb.flatten(), (p * w).flatten())
    return out[: n // 2]


def band_power(spec: np.ndarray, bands=((1, 2), (3, 4), (5, 8), (9, 16), (17, 31))) -> list[float]:
    return [float(spec[a:b + 1].sum()) for a, b in bands]


BANDS = ["k1-2", "k3-4", "k5-8", "k9-16", "k17-31"]


# ---------------------------------------------------------------------------------------- images
def write_png(path, img: np.ndarray) -> None:
    img = np.ascontiguousarray(img, dtype=np.uint8)
    color, ch = (0, 1) if img.ndim == 2 else (2, 3)
    h, w = img.shape[:2]
    raw = np.concatenate([np.zeros((h, 1), np.uint8), img.reshape(h, w * ch)], 1).tobytes()

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


@torch.no_grad()
def mip(data: torch.Tensor, res: int = 64) -> torch.Tensor:
    """Front-view (along z) max-intensity projection of density (rows = y, top = high), (res,res) float GPU.
    Restricts to `res` first so every column shows the same resolution."""
    d = restrict(data[CH3["density"]][None], res)[0]
    return d.amax(dim=0).flip(0)


def to_u8(img: np.ndarray, vmax: float) -> np.ndarray:
    g = np.clip(img / max(vmax, 1e-6), 0, 1) ** 0.6
    return (g * 255).astype(np.uint8)


def label_strip(width: int, height: int = 1) -> np.ndarray:
    return np.full((height, width), 255, np.uint8)


def finite(x: float) -> bool:
    return math.isfinite(x)
