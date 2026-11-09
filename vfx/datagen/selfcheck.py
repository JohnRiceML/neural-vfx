"""Datagen self-check (owner's quick check; gate tests live in tests/).

    .venv/Scripts/python -m vfx.datagen.selfcheck [--fake] [--keep] [--png-dir DIR] [--frames T]

Builds a 3-scenario dataset (1 train, 1 test, 1 ood) under version "selfcheck", checks
shapes/dtypes/finiteness/splits/loader, reports LR(pooled HR) vs independent LR sim agreement
at large scales, optionally writes PNGs, then deletes the dataset (unless --keep).
Uses the real vfx.sim.Solver when importable, else (or with --fake) a tiny torch fake solver.
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vfx.core import paths
from vfx.core.fields import (AMBIENT_TEMP, CH, DOMAIN_SIZE, HR_RES, LR_RES, NUM_CHANNELS, FieldMeta,
                             FieldState, Forces, rasterize_obstacles)

from . import build, runner
from .dataset import load_dataset
from .scenarios import sample_scenario

VERSION = "selfcheck"


# ----------------------------------------------------------------------------- fake solver
class FakeSolver:
    """Crude stand-in with the vfx.sim.Solver interface: emit, buoyancy, wind, burn,
    semi-Lagrangian advection, a few Jacobi projection iterations. For plumbing tests only."""

    def __init__(self, res: int, config=None, device="cuda"):
        self.res, self.dx, self.device = res, DOMAIN_SIZE / res, device
        c = (torch.arange(res, device=device, dtype=torch.float32) + 0.5) * self.dx
        self.yy, self.xx = torch.meshgrid(c, c, indexing="ij")

    def reset(self, obstacles=()):
        d = torch.zeros(NUM_CHANNELS, self.res, self.res, device=self.device)
        d[CH["temperature"]] = AMBIENT_TEMP
        return FieldState(d, rasterize_obstacles(list(obstacles), self.res, self.device),
                          FieldMeta.for_res(self.res))

    def advect(self, field, velocity, dt):
        f = field[None] if field.dim() == 3 else field[None, None]
        gx = (self.xx - dt * velocity[0]) / DOMAIN_SIZE * 2 - 1
        gy = (self.yy - dt * velocity[1]) / DOMAIN_SIZE * 2 - 1
        out = F.grid_sample(f, torch.stack([gx, gy], -1)[None], align_corners=False, padding_mode="border")
        return out[0] if field.dim() == 3 else out[0, 0]

    def project(self, state, iters=30):
        u, v = state.data[CH["vel_x"]], state.data[CH["vel_y"]]
        div = (torch.roll(u, -1, 1) - torch.roll(u, 1, 1) + torch.roll(v, -1, 0) - torch.roll(v, 1, 0)) / (2 * self.dx)
        p = torch.zeros_like(div)
        for _ in range(iters):
            p = 0.25 * (torch.roll(p, 1, 0) + torch.roll(p, -1, 0) + torch.roll(p, 1, 1) + torch.roll(p, -1, 1)
                        - self.dx * self.dx * div)
        u -= (torch.roll(p, -1, 1) - torch.roll(p, 1, 1)) / (2 * self.dx)
        v -= (torch.roll(p, -1, 0) - torch.roll(p, 1, 0)) / (2 * self.dx)
        return state

    def step(self, state, forces: Forces, dt):
        d = state.data
        for e in forces.emitters:
            m = ((self.xx - e.pos[0]) ** 2 + (self.yy - e.pos[1]) ** 2 <= e.radius ** 2).float()
            d[CH["density"]] += m * e.density_rate * dt
            d[CH["fuel"]] += m * e.fuel_rate * dt
            if e.temperature > 0:
                d[CH["temperature"]] = torch.maximum(d[CH["temperature"]], m * e.temperature)
            d[CH["vel_x"]] = d[CH["vel_x"]] * (1 - m) + m * e.velocity[0]
            d[CH["vel_y"]] = d[CH["vel_y"]] * (1 - m) + m * e.velocity[1]
        burn = torch.clamp(d[CH["fuel"]], max=2.0 * dt) * (d[CH["temperature"]] > 500)
        d[CH["fuel"]] -= burn
        d[CH["temperature"]] += burn * 300.0
        d[CH["density"]] += burn * 0.3
        d[CH["vel_x"]] += dt * forces.wind[0]
        d[CH["vel_y"]] += dt * (forces.wind[1] + 2.0 * (d[CH["temperature"]] - AMBIENT_TEMP) / AMBIENT_TEMP
                                - 0.1 * d[CH["density"]])
        d[CH["temperature"]] = AMBIENT_TEMP + (d[CH["temperature"]] - AMBIENT_TEMP) * 0.98
        fluid = 1 - state.solid
        d[CH["vel_x"]] *= fluid
        d[CH["vel_y"]] *= fluid
        self.project(state)
        d.copy_(self.advect(d, d[CH["vel_x"]:CH["vel_y"] + 1].clone(), dt))
        d[CH["vel_x"]] *= fluid
        d[CH["vel_y"]] *= fluid
        return state

    def divergence(self, state):
        return torch.zeros_like(state.solid)


def fake_factory(res):
    return FakeSolver(res), "fake-0"


# ----------------------------------------------------------------------------- png
def write_png(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    def chunk(t, data):
        c = struct.pack(">I", len(data)) + t + data
        return c + struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) \
        + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def _to_rgb(x: np.ndarray, lo: float, hi: float, hot: bool) -> np.ndarray:
    a = np.clip((x.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)[::-1]   # y=0 at bottom
    if hot:
        return (np.stack([np.clip(3 * a, 0, 1), np.clip(3 * a - 1, 0, 1), np.clip(3 * a - 2, 0, 1)], -1) * 255).astype(np.uint8)
    return (np.repeat(a[..., None], 3, -1) * 255).astype(np.uint8)


def _up(x: np.ndarray, n: int) -> np.ndarray:
    return np.kron(x, np.ones((n, n), dtype=x.dtype))


def save_pngs(root: Path, out: Path) -> list[Path]:
    """Per scenario, last frame: top row density, bottom row temperature;
    columns HR | pooled LR | independent LR sim (if any). Solids drawn blue."""
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for meta_p in sorted(root.glob("*/scn_*/meta.json")):
        d = meta_p.parent
        meta = json.loads(meta_p.read_text())
        hr = np.load(d / "hr.npy")[-1].astype(np.float32)
        lr = np.load(d / "lr.npy")[-1].astype(np.float32)
        cols = [(hr[0], hr[1]), (_up(lr[CH["density"]], 4), _up(lr[CH["temperature"]], 4))]
        if meta["has_lr_sim"]:
            ls = np.load(d / "lr_sim.npy")[-1].astype(np.float32)
            cols.append((_up(ls[CH["density"]], 4), _up(ls[CH["temperature"]], 4)))
        solid = np.load(d / "solid_hr.npy")[::-1].astype(bool)
        dmax = max(float(hr[0].max()), 1e-3)
        tmax = max(float(hr[1].max()), AMBIENT_TEMP + 1)
        rows = []
        for r, (lo, hi, hot) in enumerate([(0, dmax, False), (AMBIENT_TEMP, tmax, True)]):
            tiles = []
            for c in cols:
                img = _to_rgb(c[r], lo, hi, hot)
                img[solid] = (40, 60, 160)
                tiles.append(img)
                tiles.append(np.full((HR_RES, 4, 3), 128, np.uint8))
            rows.append(np.concatenate(tiles[:-1], 1))
            rows.append(np.full((4, rows[-1].shape[1], 3), 128, np.uint8))
        p = out / f"{meta['split']}_scn_{meta['scenario']['seed']:06d}.png"
        write_png(p, np.ascontiguousarray(np.concatenate(rows[:-1], 0)))
        written.append(p)
    return written


# ----------------------------------------------------------------------------- checks
def rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-8))


def large_scale_agreement(d: Path, pool: int = 4) -> dict:
    """Pooled-HR LR vs independent LR sim, after a further `pool`x average pool
    (128 -> 32: features >= ~1/32 of the box). Relative L2 of density and T - ambient."""
    lr = torch.from_numpy(np.load(d / "lr.npy").astype(np.float32))
    ls = torch.from_numpy(np.load(d / "lr_sim.npy").astype(np.float32))
    out = {}
    for name, off in (("density", 0.0), ("temperature", AMBIENT_TEMP)):
        a = lr[:, CH[name]] - off
        b = ls[:, CH[name]] - off
        out[f"{name}@128"] = rel_diff(a.numpy(), b.numpy())
        out[f"{name}@{LR_RES // pool}"] = rel_diff(F.avg_pool2d(a[:, None], pool).numpy(),
                                                   F.avg_pool2d(b[:, None], pool).numpy())
    return out


def check_scenario_sampling(n: int = 2000) -> None:
    for split in ("train", "test", "ood"):
        for s in (0, 7, 12345):
            assert sample_scenario(s, split) == sample_scenario(s, split), "non-deterministic"
    tags_seen = {"train": set(), "ood": set()}
    for s in range(n):
        tr = sample_scenario(s, "train")
        assert all(o.kind == "circle" for o in tr.obstacles), "box obstacle in train"
        assert 1 <= len(tr.emitters) <= 3 and len(tr.obstacles) <= 2
        assert all(0 < e.pos[0] < 1 and 0 < e.pos[1] < 0.35 for e in tr.emitters)
        tags_seen["train"].update(tr.tags)
        ood = sample_scenario(200_000 + s, "ood")
        assert "ood" in ood.tags and len(set(ood.tags) & {"box_obstacle", "strong_wind", "large_emitter"}) >= 1
        tags_seen["ood"].update(ood.tags)
    fire = sum("fire" in sample_scenario(s, "train").tags for s in range(n)) / n
    print(f"  sampling: deterministic, train never has box obstacles, fire fraction {fire:.2f}, "
          f"ood tags {sorted(tags_seen['ood'])}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--png-dir", type=str, default=None)
    a = ap.parse_args(argv)

    if not a.fake:
        try:
            from vfx.sim import Solver  # noqa: F401
        except Exception as e:                      # solver not landed yet
            print(f"  vfx.sim not importable ({type(e).__name__}: {e}); using fake solver")
            a.fake = True
    if a.fake:
        runner.SOLVER_FACTORY = fake_factory
        runner.reset_solver_cache()

    check_scenario_sampling()
    root = paths.dataset_dir(VERSION)
    shutil.rmtree(root, ignore_errors=True)
    try:
        card = build.build(VERSION, 1, 1, 1, a.frames)
        T = a.frames
        splits = json.loads((root / "splits.json").read_text())
        sets = {k: set(v) for k, v in splits.items()}
        assert not (sets["train"] & sets["test"]) and not (sets["train"] & sets["ood"]) and not (sets["test"] & sets["ood"])
        assert card["counts"] == {"train": 1, "test": 1, "ood": 1}

        expect = {"hr.npy": ((T, 2, HR_RES, HR_RES), np.float16), "lr.npy": ((T, 5, LR_RES, LR_RES), np.float16),
                  "solid_hr.npy": ((HR_RES, HR_RES), np.uint8), "solid_lr.npy": ((LR_RES, LR_RES), np.uint8),
                  "lr_sim.npy": ((T, 5, LR_RES, LR_RES), np.float16)}
        for split, seeds in splits.items():
            for seed in seeds:
                d = runner.scenario_dir(root, split, seed)
                meta = json.loads((d / "meta.json").read_text())
                for f, (shape, dt) in expect.items():
                    if f == "lr_sim.npy" and split == "train":
                        assert not (d / f).exists(); continue
                    x = np.load(d / f)
                    assert x.shape == shape and x.dtype == dt, (f, x.shape, x.dtype)
                    assert np.isfinite(x.astype(np.float32)).all(), f"non-finite in {d / f}"
                # pooled lr really is the pooled HR (density/temperature channels)
                hr = torch.from_numpy(np.load(d / "hr.npy").astype(np.float32))
                lr = np.load(d / "lr.npy").astype(np.float32)
                pooled = F.avg_pool2d(hr, 4).numpy()
                err = np.abs(pooled - lr[:, [CH["density"], CH["temperature"]]]).max()
                assert err < 1.0, f"lr is not pooled hr (max abs {err})"
                dmax = meta["stats"]["hr"]["max"][0]
                print(f"  {split}/scn_{seed:06d}: tags={meta['tags']} start_step={meta['start_step']} "
                      f"density max {dmax:.3f}, T max {meta['stats']['hr']['max'][1]:.0f} K, "
                      f"flags={meta['flags']}, sim {meta['sim_seconds']}")
                if meta["has_lr_sim"]:
                    agr = large_scale_agreement(d)
                    print("    LR(pooled HR) vs LR sim, relative L2: "
                          + ", ".join(f"{k}={v:.3f}" for k, v in agr.items()))

        for split in ("train", "test", "ood"):
            ds = load_dataset(VERSION, split)
            assert len(ds) == T - 1, (split, len(ds))
            it = ds[0]
            assert it["lr"].shape == (5, LR_RES, LR_RES) and it["hr"].shape == (2, HR_RES, HR_RES)
            assert it["hr_prev"].shape == (2, HR_RES, HR_RES) and it["solid_hr"].shape == (HR_RES, HR_RES)
            assert it["solid_lr"].shape == (LR_RES, LR_RES) and it["t"] >= 1
            assert all(it[k].dtype == torch.float32 for k in ("lr", "hr", "hr_prev", "solid_lr", "solid_hr"))
            assert ("lr_sim" in it) == (split != "train")
        dl = torch.utils.data.DataLoader(load_dataset(VERSION, "test"), batch_size=2, num_workers=2, shuffle=True)
        b = next(iter(dl))
        assert b["hr"].shape == (2, 2, HR_RES, HR_RES)
        del dl, ds, it, b                  # release mmaps (Windows cannot delete mapped files)
        print(f"  loader OK (workers=2); card bytes {card['bytes'] / 1e6:.1f} MB, flagged {card['flagged']}")

        # resumability: a second build call must skip everything
        msgs = []
        build.build(VERSION, 1, 1, 1, a.frames, log=msgs.append)
        assert any("0 to run" in m for m in msgs), msgs

        if a.png_dir:
            for p in save_pngs(root, Path(a.png_dir)):
                print(f"  png: {p}")
        print("SELFCHECK OK")
    finally:
        if not a.keep:
            import gc, time
            for _ in range(5):
                gc.collect()
                shutil.rmtree(root, ignore_errors=True)
                if not root.exists():
                    break
                time.sleep(1.0)
            if root.exists():
                print(f"  warning: could not fully delete {root}")


if __name__ == "__main__":
    main()
