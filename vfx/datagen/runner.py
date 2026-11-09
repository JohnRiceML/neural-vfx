"""run(scenario): simulate one scenario at HR (and LR for test/ood) and write its directory.

Frames are buffered on the GPU and read back once per scenario (offline tool).
Files are written first and meta.json last, so meta.json marks a complete directory.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from vfx.core import paths
from vfx.core.fields import (AMBIENT_TEMP, CHANNELS, CH, HR_CHANNELS, HR_RES, LR_RES, NUM_CHANNELS,
                             SIM_DT, UPSCALE, Forces, Scenario, rasterize_obstacles)

from .scenarios import DEFAULT_FRAMES, scenario_split

FILES = ("hr.npy", "lr.npy", "solid_hr.npy", "solid_lr.npy")
DEFAULT_VERSION = "scratch"

# Solver factory: (res) -> solver with the vfx.sim.Solver interface. Overridable (selfcheck
# installs a fake while vfx.sim is not ready). Returns (solver, solver_version).
SOLVER_FACTORY: Callable | None = None
_solver_cache: dict[int, tuple] = {}
_state_cache: dict[int, object] = {}   # persistent state buffers per res (solver graphs are keyed by ptr)
REUSE_BUFFERS = True


def _default_factory(res: int):
    from vfx.sim import SOLVER_VERSION, Solver, SolverConfig
    return Solver(res, SolverConfig()), SOLVER_VERSION


def get_solver(res: int):
    if res not in _solver_cache:
        _solver_cache[res] = (SOLVER_FACTORY or _default_factory)(res)
    return _solver_cache[res]


def reset_solver_cache() -> None:
    _solver_cache.clear()
    _state_cache.clear()


def fresh_state(solver, res: int, obstacles):
    """solver.reset(obstacles), copied into persistent buffers so the solver's captured
    CUDA graph (keyed by buffer pointers) is reused across scenarios instead of re-captured."""
    new = solver.reset(obstacles)
    if not REUSE_BUFFERS:
        return new
    st = _state_cache.get(res)
    if st is None:
        _state_cache[res] = new
        return new
    st.data.copy_(new.data)
    st.solid.copy_(new.solid)
    st.meta = new.meta
    return st


def scenario_dir(root: Path, split: str, seed: int) -> Path:
    return Path(root) / split / f"scn_{seed:06d}"


def is_complete(d: Path, with_lr_sim: bool) -> bool:
    need = FILES + (("lr_sim.npy",) if with_lr_sim else ())
    return (d / "meta.json").exists() and all((d / f).exists() for f in need)


def simulate(scenario: Scenario, res: int, frames: int, keep: str):
    """Run `scenario.num_steps` steps; return (a, b, solid, solver_version) with the last `frames` on GPU.

    keep="hr": (T,2,res,res) fp16 density,temperature + (T,5,res/4,res/4) fp16 pooled state
    keep="full": (T,5,res,res) fp16 full state, second return is None.
    Frame i is the state after step (num_steps - frames + i + 1).
    """
    solver, version = get_solver(res)
    state = fresh_state(solver, res, scenario.obstacles)
    forces = Forces(emitters=list(scenario.emitters), impulses=[], wind=tuple(scenario.wind))
    start = scenario.num_steps - frames
    if start < 0:
        raise ValueError(f"num_steps {scenario.num_steps} < frames {frames}")
    hr_idx = [CH[c] for c in HR_CHANNELS]
    if keep == "hr":
        a = torch.empty(frames, len(HR_CHANNELS), res, res, device="cuda", dtype=torch.float16)
        b = torch.empty(frames, NUM_CHANNELS, res // UPSCALE, res // UPSCALE, device="cuda", dtype=torch.float16)
    else:
        a = torch.empty(frames, NUM_CHANNELS, res, res, device="cuda", dtype=torch.float16)
        b = None
    for s in range(scenario.num_steps):
        state = solver.step(state, forces, SIM_DT)
        i = s - start
        if i >= 0:
            if keep == "hr":
                a[i] = state.data[hr_idx]
                b[i] = F.avg_pool2d(state.data[None], UPSCALE)[0]
            else:
                a[i] = state.data
    return a, b, state.solid, version


def _stats(x: torch.Tensor) -> dict:
    """Per-channel min/max/mean of a (T,C,H,W) tensor + finiteness (one small readback)."""
    xf = x.float()
    mn = xf.amin(dim=(0, 2, 3)); mx = xf.amax(dim=(0, 2, 3)); me = xf.mean(dim=(0, 2, 3))
    fin = torch.isfinite(xf).all()
    v = torch.cat([mn, mx, me, fin.float()[None]]).cpu().tolist()
    C = x.shape[1]
    return {"min": v[:C], "max": v[C:2 * C], "mean": v[2 * C:3 * C], "finite": bool(v[-1])}


def _save(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def tracking(lr: torch.Tensor, lr_sim: torch.Tensor) -> dict:
    """Per-frame relative L2 of pooled HR vs lr_sim after a further 4x pool (32²): density,
    temperature - ambient, velocity. Returns {"density": [..T], "temperature": [..], "velocity": [..]}."""
    a = F.avg_pool2d(lr.float(), UPSCALE)
    b = F.avg_pool2d(lr_sim.float(), UPSCALE)
    off = torch.zeros(NUM_CHANNELS, 1, 1, device=a.device)
    off[CH["temperature"]] = AMBIENT_TEMP
    a, b = a - off, b - off

    def rel(idx):
        num = (a[:, idx] - b[:, idx]).flatten(1).norm(dim=1)
        return num / (b[:, idx].flatten(1).norm(dim=1) + 1e-8)
    v = torch.stack([rel([CH["density"]]), rel([CH["temperature"]]), rel([CH["vel_x"], CH["vel_y"]])])
    v = v.cpu().tolist()
    return {"density": [round(x, 4) for x in v[0]], "temperature": [round(x, 4) for x in v[1]],
            "velocity": [round(x, 4) for x in v[2]]}


def run(scenario: Scenario, out_dir: str | Path | None = None, frames: int | None = None,
        with_lr_sim: bool | None = None, guide=None) -> dict:
    """Simulate and write one scenario directory; return its metadata dict.

    out_dir: the scenario directory (default: DATA_ROOT/datasets/scratch/<split>/scn_<seed>).
    frames: saved frames T (default DEFAULT_FRAMES); warm-up = scenario.num_steps - T.
    with_lr_sim: also run an independent LR_RES sim (default: True for test/ood; forced True
        when guided).
    guide: a vfx.datagen.guide.GuideConfig -> the HR run is guided toward the independent LR
        run (dataset v2); lr_sim.npy is written for every split. None = v1 behaviour.
    """
    split = scenario_split(scenario)
    frames = DEFAULT_FRAMES if frames is None else int(frames)
    if with_lr_sim is None:
        with_lr_sim = split in ("test", "ood")
    if guide is not None:
        with_lr_sim = True
    d = Path(out_dir) if out_dir is not None else scenario_dir(paths.dataset_dir(DEFAULT_VERSION), split, scenario.seed)
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").unlink(missing_ok=True)       # incomplete until rewritten

    t0 = time.perf_counter()
    lr_sim = None
    if guide is not None:
        from .guide import simulate_guided
        hr, lr, lr_sim, solid_hr, version = simulate_guided(scenario, frames, guide)
    else:
        hr, lr, solid_hr, version = simulate(scenario, HR_RES, frames, keep="hr")
    stats = {"hr": _stats(hr), "lr": _stats(lr)}
    track = tracking(lr, lr_sim) if lr_sim is not None else None
    _save(d / "hr.npy", hr.cpu().numpy())
    _save(d / "lr.npy", lr.cpu().numpy())
    _save(d / "solid_hr.npy", (solid_hr > 0.5).to(torch.uint8).cpu().numpy())
    _save(d / "solid_lr.npy", (rasterize_obstacles(scenario.obstacles, LR_RES) > 0.5).to(torch.uint8).cpu().numpy())
    del hr, lr
    t_hr = time.perf_counter() - t0
    t_lr = 0.0
    if lr_sim is not None:
        stats["lr_sim"] = _stats(lr_sim)
        _save(d / "lr_sim.npy", lr_sim.cpu().numpy())
        del lr_sim
    elif with_lr_sim:
        t1 = time.perf_counter()
        lr_sim, _, _, _ = simulate(scenario, LR_RES, frames, keep="full")
        stats["lr_sim"] = _stats(lr_sim)
        _save(d / "lr_sim.npy", lr_sim.cpu().numpy())
        t_lr = time.perf_counter() - t1
    else:
        (d / "lr_sim.npy").unlink(missing_ok=True)

    flags = []
    if not all(s["finite"] for s in stats.values()):
        flags.append("nonfinite")
    dmax = stats["hr"]["max"][0]
    if dmax < 1e-3:
        flags.append("empty")
    if dmax > 20.0 or stats["hr"]["max"][1] > 5000.0 or stats["hr"]["min"][1] < AMBIENT_TEMP - 50.0:
        flags.append("out_of_range")

    meta = {
        "scenario": dataclasses.asdict(scenario),
        "split": split,
        "solver_version": version,
        "lr_res": LR_RES, "hr_res": HR_RES, "dt": SIM_DT,
        "frames": frames,
        "start_step": scenario.num_steps - frames + 1,   # solver step count of frame 0
        "channels": list(CHANNELS), "hr_channels": list(HR_CHANNELS),
        "has_lr_sim": bool(with_lr_sim),
        "tags": list(scenario.tags),
        "stats": stats, "flags": flags,
        "sim_seconds": {"hr": round(t_hr, 3), "lr": round(t_lr, 3)},
    }
    if guide is not None:
        meta["guidance"] = guide.asdict()
        meta["tracking"] = track   # rel L2 at 32² of lr (pooled guided HR) vs lr_sim, per frame
    tmp = d / "meta.tmp.json"
    tmp.write_text(json.dumps(meta, indent=1))
    os.replace(tmp, d / "meta.json")
    return meta
