"""Synthetic stand-in dataset in the exact CLAUDE.md format (for building/testing the ML stack
before the real solver dataset exists). NOT physics: noisy density/heat blobs advected at 512^2
by a random time-varying divergence-free velocity, then 4x average-pooled to 128^2.

    .venv/Scripts/python -m vfx.train.synthetic --version synthetic --train 20 --test 4 --frames 32
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from vfx.core import paths
from vfx.core.fields import AMBIENT_TEMP, CHANNELS, DOMAIN_SIZE, HR_RES, LR_RES, SIM_DT, UPSCALE

DEV = "cuda"


def _noise(g: torch.Generator, n: int, octaves=(4, 8, 16, 32, 64, 128), decay=0.6) -> torch.Tensor:
    """Multi-scale value noise in roughly [-1, 1], (n, n)."""
    out = torch.zeros(1, 1, n, n, device=DEV)
    amp, tot = 1.0, 0.0
    for o in octaves:
        r = torch.randn(1, 1, o, o, device=DEV, generator=g)
        out += amp * F.interpolate(r, size=(n, n), mode="bicubic", align_corners=False)
        tot += amp
        amp *= decay
    return (out / tot * 1.5)[0, 0]


def _stream_velocity(t: float, modes, n: int) -> torch.Tensor:
    """Divergence-free (2,n,n) velocity (m/s) from a sum of Fourier stream-function modes."""
    c = (torch.arange(n, device=DEV, dtype=torch.float32) + 0.5) / n * DOMAIN_SIZE
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    u = torch.zeros(n, n, device=DEV)
    v = torch.zeros(n, n, device=DEV)
    for (kx, ky, a, ph, w) in modes:
        arg = 2 * math.pi * (kx * xx + ky * yy) + ph + w * t
        # psi = a/(2 pi |k|) sin(arg);  u = dpsi/dy, v = -dpsi/dx
        kn = math.hypot(kx, ky)
        cs = torch.cos(arg) * (a / kn)
        u += cs * ky
        v -= cs * kx
    return torch.stack([u, v])


def _advect(f: torch.Tensor, vel: torch.Tensor, dt: float) -> torch.Tensor:
    C, H, W = f.shape
    ys = (torch.arange(H, device=DEV) * 2 + 1) / H - 1
    xs = (torch.arange(W, device=DEV) * 2 + 1) / W - 1
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    gx = xx - vel[0] * dt * 2 / DOMAIN_SIZE
    gy = yy - vel[1] * dt * 2 / DOMAIN_SIZE
    grid = torch.stack([gx, gy], -1)[None]
    return F.grid_sample(f[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]


def gen_scenario(seed: int, frames: int, warmup: int = 12):
    g = torch.Generator(device=DEV).manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = HR_RES
    modes = []
    for _ in range(rng.integers(4, 9)):
        k = rng.integers(1, 5, size=2) * rng.choice([-1, 1], size=2)
        modes.append((int(k[0]), int(abs(k[1]) + 1), rng.uniform(0.05, 0.25), rng.uniform(0, 6.28), rng.uniform(-2, 2)))
    for _ in range(rng.integers(3, 7)):   # small-scale swirls -> HR detail
        k = rng.integers(6, 16, size=2) * rng.choice([-1, 1], size=2)
        modes.append((int(k[0]), int(k[1]), rng.uniform(0.01, 0.05), rng.uniform(0, 6.28), rng.uniform(-4, 4)))
    drift = rng.uniform(0.05, 0.3)
    fire = bool(rng.random() < 0.5)
    c = (torch.arange(n, device=DEV, dtype=torch.float32) + 0.5) / n
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    emitters = [(rng.uniform(0.2, 0.8), rng.uniform(0.08, 0.3), rng.uniform(0.03, 0.08)) for _ in range(rng.integers(1, 4))]
    dens = torch.zeros(n, n, device=DEV)
    heat = torch.zeros(n, n, device=DEV)
    fuel = torch.zeros(n, n, device=DEV)
    hr_out, lr_out = [], []
    for step in range(warmup + frames):
        t = step * SIM_DT
        vel = _stream_velocity(t, modes, n)
        vel[1] += drift
        # emit noisy density / heat (fresh small-scale detail every step)
        src = torch.zeros(n, n, device=DEV)
        for (ex, ey, r) in emitters:
            src += torch.exp(-((xx - ex) ** 2 + (yy - ey) ** 2) / (2 * r * r))
        mod = (1.0 + _noise(g, n)).clamp_min(0.0)
        dens = dens + SIM_DT * 3.0 * src * mod
        heat = heat + SIM_DT * (600.0 if fire else 150.0) * src * (1.0 + 0.5 * _noise(g, n)).clamp_min(0)
        if fire:
            fuel = fuel + SIM_DT * 2.0 * src
        st = _advect(torch.stack([dens, heat, fuel]), vel, SIM_DT)
        dens, heat, fuel = st[0] * 0.995, st[1] * 0.97, st[2] * 0.9
        if step >= warmup:
            temp = AMBIENT_TEMP + heat
            full = torch.stack([dens, vel[0], vel[1], temp, fuel])           # CHANNELS order
            hr_out.append(torch.stack([dens, temp]).half())
            lr_out.append(F.avg_pool2d(full[None], UPSCALE)[0].half())
    return torch.stack(hr_out), torch.stack(lr_out), fire


def build(version: str, n_train: int, n_test: int, frames: int) -> None:
    root = paths.dataset_dir(version)
    splits = {"train": list(range(1000, 1000 + n_train)), "test": list(range(9000, 9000 + n_test)), "ood": []}
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for split, seeds in splits.items():
        for seed in seeds:
            d = root / split / f"scn_{seed:06d}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "meta.json").unlink(missing_ok=True)
            hr, lr, fire = gen_scenario(seed, frames)
            np.save(d / "hr.npy", hr.cpu().numpy())
            np.save(d / "lr.npy", lr.cpu().numpy())
            np.save(d / "solid_hr.npy", np.zeros((HR_RES, HR_RES), np.uint8))
            np.save(d / "solid_lr.npy", np.zeros((LR_RES, LR_RES), np.uint8))
            meta = {"scenario": {"seed": seed, "synthetic": True}, "split": split, "solver_version": "synthetic",
                    "lr_res": LR_RES, "hr_res": HR_RES, "dt": SIM_DT, "frames": frames, "start_step": 0,
                    "channels": list(CHANNELS), "has_lr_sim": False,
                    "tags": ["synthetic", "fire" if fire else "smoke"], "flags": []}
            (d / "meta.json").write_text(json.dumps(meta, indent=1))
            print(f"{split} {seed} dmax={hr[:, 0].float().max().item():.2f} "
                  f"Tmax={hr[:, 1].float().max().item():.0f}", flush=True)
    card = {"version": version, "counts": {k: len(v) for k, v in splits.items()}, "solver_version": "synthetic",
            "frames": frames, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "procedural stand-in data generated by vfx/train/synthetic.py (not a solver)"}
    (root / "card.json").write_text(json.dumps(card, indent=1))
    (root / "splits.json").write_text(json.dumps(splits))
    print(f"wrote {root} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="synthetic")
    ap.add_argument("--train", type=int, default=20)
    ap.add_argument("--test", type=int, default=4)
    ap.add_argument("--frames", type=int, default=32)
    a = ap.parse_args()
    build(a.version, a.train, a.test, a.frames)
