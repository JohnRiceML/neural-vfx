"""Training-pair collection for the velocity corrector (non-differentiable formulations).

Target = what the fine sim's LARGE scales do in one step minus what the coarse solver does, both from
(nearly) the same large-scale state, low-passed (cutoff cycles/domain) and projected by the 64^3
solver (its divergent part would be removed by the next step's projection anyway).

  mode "pre"    (one-step supervised, 'pre-computed'): run the 128^3 sim free; at sampled steps
                x = P64(R f_t),   target = R f_{t+1} - step64(x)
  mode "dagger" (data along the CORRECTED coarse trajectory): the 64^3 sim runs with the current
                corrector; the 128^3 sim is nudged toward it (vfx.datagen3d.guide3d, velocity before
                the fine step, scalars after, i.e. the mirror image of the route-B guide) so their large
                scales coincide; at sampled steps
                x = c_t (the coarse state the corrector sees at runtime),
                target = [R f_{t+1} - R f_t^guided] - [step64_uncorrected(c_t) - c_t]   (tendency difference)

Stored per scene as fp16 32^3 tensors (features() of x, 2x-pooled target):
    DATA_ROOT/datasets/corr3d_<name>/<split>_<seed>.pt   {"x": (n,6,32,32,32), "y": (n,3,32,32,32)}

    .venv/Scripts/python -m vfx.correct3d.collect --name r0 --mode pre --seeds 0-23
    .venv/Scripts/python -m vfx.correct3d.collect --name r1 --mode dagger --ckpt CK --seeds 0-23
"""
from __future__ import annotations

import argparse
import json
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from vfx.core.fields3d import SIM_DT
from .common import CORR_DATASETS as DATASETS
from .common import VEL, get_solver, restrict, sampled_scene
from .model import features
from .tune_classical import parse_seeds


def _lowpass_project(dv: torch.Tensor, proj_solver, proj_state, mask) -> torch.Tensor:
    from vfx.datagen3d.guide3d import lowpass
    v = lowpass(dv, mask)
    proj_state.data.zero_()
    proj_state.data[VEL] = v
    proj_solver.project(proj_state)
    return proj_state.data[VEL] * (proj_state.solid < 0.5)


@torch.no_grad()
def collect_scene(sc, mode: str, steps: int, warmup: int, every: int, coarse_kw: dict, cutoff: float,
                  corrector=None, guide_cfg=None):
    from vfx.datagen3d.guide3d import Guide3D, spectral_mask
    kw_c = dict(sc.sim); kw_c.update(coarse_kw)
    fs = get_solver(128, dict(sc.sim), "fine")
    cs = get_solver(64, kw_c, "c")          # main coarse trajectory (dagger) / one-step coarse (pre)
    us = get_solver(64, kw_c, "u")          # uncorrected one-step probe (dagger)
    ps = get_solver(64, {}, "p")            # target projection
    f = fs.reset(sc.obstacles)
    c = cs.reset(sc.obstacles)
    u = us.reset(sc.obstacles)
    p = ps.reset(sc.obstacles)
    mask = spectral_mask(64, cutoff, f.data.device)
    g = Guide3D(guide_cfg, f.solid, 64) if mode == "dagger" else None
    xs, ys = [], []
    for t in range(steps):
        frc = sc.forces(t)
        take = t >= warmup and (t - warmup) % every == 0
        if mode == "pre":
            if take:
                c.data.copy_(restrict(f.data))
                cs.project(c)
                x = c.data.clone()
                cs.step(c, frc, SIM_DT)
            fs.step(f, frc, SIM_DT)
            if take:
                dv = restrict(f.data)[VEL] - c.data[VEL]
                y = _lowpass_project(dv, ps, p, mask)
        else:
            g.velocity(f, c)
            if take:
                x = c.data.clone()
                u.data.copy_(c.data)
                us.step(u, frc, SIM_DT)
                r0 = restrict(f.data)[VEL].clone()
            fs.step(f, frc, SIM_DT)
            if take:
                dv = (restrict(f.data)[VEL] - r0) - (u.data[VEL] - x[VEL])
                y = _lowpass_project(dv, ps, p, mask)
            if corrector is not None:
                corrector.apply(c, SIM_DT)
            cs.step(c, frc, SIM_DT)
            g.scalars(f, c)
        if take:
            xs.append(features(x, c.solid)[0].half())
            ys.append(F.avg_pool3d(y[None], 2)[0].half())
    return torch.stack(xs).cpu(), torch.stack(ys).cpu(), bool(torch.isfinite(c.data).all())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["pre", "dagger"], default="pre")
    ap.add_argument("--seeds", default="0-23")
    ap.add_argument("--split", default="train")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--cutoff", type=float, default=10.0)
    ap.add_argument("--coarse_cfg", default="{}", help="JSON SolverConfig3D overrides for the 64^3 sim")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--guide_cutoff", type=float, default=8.0)
    a = ap.parse_args()
    out = DATASETS / f"corr3d_{a.name}"
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free / 2 ** 30
    seeds = parse_seeds(a.seeds)
    est = len(seeds) * (a.steps // a.every) * 9 * 32 ** 3 * 2 / 2 ** 30
    print(f"free {free:.1f} GiB, estimated {est:.2f} GiB", flush=True)
    assert free - est > 2.0, "not enough free disk (C: is shared and tight: keep >= 2 GiB free)"
    coarse_kw = json.loads(a.coarse_cfg)
    corrector = guide_cfg = None
    if a.mode == "dagger":
        from vfx.datagen3d.guide3d import GuideConfig3D
        guide_cfg = GuideConfig3D(k_vel=1.0, k_scalar=0.7, cutoff=a.guide_cutoff)
        if a.ckpt:
            from . import VelocityCorrector
            corrector = VelocityCorrector(a.ckpt)
    t0 = time.time()
    for seed in seeds:
        sc = sampled_scene(seed, a.split)
        rng = np.random.default_rng(seed)
        warm = int(rng.integers(20, 80))
        x, y, ok = collect_scene(sc, a.mode, a.steps, warm, a.every, coarse_kw, a.cutoff, corrector, guide_cfg)
        torch.save({"x": x, "y": y, "tags": sc.tags, "finite": ok}, out / f"{a.split}_{seed}.pt")
        rms = float(y.float().square().mean().sqrt())
        print(f"seed {seed}: {x.shape[0]} pairs, target rms {rms:.4f} m/s/step, finite {ok} "
              f"({time.time() - t0:.0f} s)", flush=True)
    (out / "card.json").write_text(json.dumps(vars(a), indent=1))


if __name__ == "__main__":
    main()
