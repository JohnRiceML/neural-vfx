"""Solver self-check (owner's quick look, NOT the gate tests; those live in tests/).

    .venv/Scripts/python -m vfx.sim.selfcheck [--out DIR] [--steps 150]

Runs a campfire and a smoke-plume-past-a-cylinder scene at 128^2 and 512^2, then prints
ms/step (CUDA events), max normalized divergence, determinism, LR-vs-HR large-scale agreement,
mass drift over 1000 steps in a sourceless closed box, and exponential decay with
density_dissipation > 0. Saves PNG strips (obstacles tinted blue):
    rows = snapshots, columns = [LR density | HR density | LR temperature | HR temperature]
"""
from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

from vfx.core.fields import AMBIENT_TEMP, CH, SIM_DT, Emitter, Forces, Obstacle
from vfx.sim import Solver, SolverConfig

DEFAULT_OUT = os.path.join(tempfile.gettempdir(), "neural-vfx-sim-selfcheck")


def campfire(t: float) -> Forces:
    # a log-like row of small fuel sources with slow, deterministic strength variation
    em = []
    for k, x in enumerate((0.42, 0.47, 0.5, 0.53, 0.58)):
        s = 1.0 + 0.35 * np.sin(2.3 * t + 1.7 * k)
        em.append(Emitter(pos=(x, 0.07), radius=0.035, density_rate=0.0, temperature=900.0,
                          fuel_rate=8.0 * s, velocity=(0.0, 0.4)))
    return Forces(emitters=em)


def plume(t: float) -> Forces:
    return Forces(emitters=[Emitter(pos=(0.45, 0.1), radius=0.05, density_rate=2.5, temperature=520.0,
                                    velocity=(0.0, 0.8))], wind=(0.4, 0.0))


SCENES = {
    "campfire": (campfire, []),
    "plume": (plume, [Obstacle("circle", (0.5, 0.52), (0.09, 0.09))]),
}


def timed_run(solver, state, forces_fn, steps, snaps):
    frames = []
    ms = []
    for i in range(steps):
        f = forces_fn(i * SIM_DT)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        solver.step(state, f, SIM_DT)
        e1.record()
        ms.append((e0, e1))
        if i + 1 in snaps:
            frames.append(state.data[[CH["density"], CH["temperature"]]].clone())
    torch.cuda.synchronize()
    t = np.array([a.elapsed_time(b) for a, b in ms])
    return frames, t


def colorize_density(d: np.ndarray) -> np.ndarray:
    v = np.clip(d / 1.5, 0, 1) ** 0.6
    return np.stack([v, v, v], -1)


def colorize_temp(T: np.ndarray) -> np.ndarray:
    x = np.clip((T - AMBIENT_TEMP) / (2500.0 - AMBIENT_TEMP), 0, 1)
    r = np.clip(x * 3.0, 0, 1)
    g = np.clip(x * 3.0 - 1.0, 0, 1)
    b = np.clip(x * 3.0 - 2.0, 0, 1)
    return np.stack([r, g, b], -1)


def to_img(lr, hr, size=384, solid_lr=None, solid_hr=None):
    """lr, hr: (2,H,W) tensors -> (size, 4*size, 3) uint8 strip, flipped so +y is up."""
    def up(t):
        return F.interpolate(t[None, None].float(), size=(size, size), mode="nearest")[0, 0].cpu().numpy()[::-1]

    def tint(img, solid):
        if solid is not None:
            m = up(solid) > 0.5
            img[m] = (0.15, 0.25, 0.6)
        return img
    cols = [tint(colorize_density(up(lr[0])), solid_lr), tint(colorize_density(up(hr[0])), solid_hr),
            tint(colorize_temp(up(lr[1])), solid_lr), tint(colorize_temp(up(hr[1])), solid_hr)]
    img = np.concatenate(cols, 1)
    img[:, [size, 2 * size, 3 * size]] = 0.3
    return (img * 255).astype(np.uint8)


def large_scale_err(lr_d, hr_d, sigma_lr=3.0):
    """Relative L2 between LR density and 4x-avg-pooled HR density after a Gaussian blur."""
    hr_p = F.avg_pool2d(hr_d[None, None], 4)[0, 0]
    k = int(3 * sigma_lr) * 2 + 1
    x = torch.arange(k, device=lr_d.device, dtype=torch.float32) - k // 2
    g = torch.exp(-x ** 2 / (2 * sigma_lr ** 2))
    g = g / g.sum()

    def blur(a):
        a = F.conv2d(F.pad(a[None, None], (k // 2, k // 2, 0, 0), mode="replicate"), g.view(1, 1, 1, k))
        a = F.conv2d(F.pad(a, (0, 0, k // 2, k // 2), mode="replicate"), g.view(1, 1, k, 1))
        return a[0, 0]
    a, b = blur(lr_d), blur(hr_p)
    return float((a - b).norm() / (b.norm() + 1e-8))


def save_png(img: np.ndarray, path: str):
    """PIL if installed, else pygame-ce (in the venv)."""
    try:
        from PIL import Image
        Image.fromarray(img).save(path)
    except ImportError:
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(img.transpose(1, 0, 2)))
        pygame.image.save(surf, path)


def budgets(f, dx):
    """Large-scale integrals of a (2,H,W) [density, temperature] frame."""
    d, T = f[0].double(), f[1].double()
    H = d.shape[0]
    y = (torch.arange(H, device=d.device, dtype=torch.float64) + 0.5) * dx
    m = d.sum() * dx * dx
    heat = (T - AMBIENT_TEMP).clamp(min=0).sum() * dx * dx
    yc = (d.sum(1) * y).sum() * dx * dx / (m + 1e-12)
    yh = ((T - AMBIENT_TEMP).clamp(min=0).sum(1) * y).sum() * dx * dx / (heat + 1e-12)
    return float(m), float(heat), float(yc), float(yh)


def mass_drift(res, steps=1000, dissipation=0.0):
    cfg = SolverConfig(density_dissipation=dissipation)
    s = Solver(res, cfg)
    st = s.reset([Obstacle("circle", (0.6, 0.6), (0.08, 0.08))])
    yy, xx = torch.meshgrid(*(torch.arange(res, device="cuda", dtype=torch.float32) + 0.5,) * 2, indexing="ij")
    yy, xx = yy / res, xx / res
    blob = torch.exp(-((xx - 0.35) ** 2 + (yy - 0.3) ** 2) / 0.01)
    st.data[CH["density"]] = blob * (st.solid < 0.5)
    st.data[CH["temperature"]] = AMBIENT_TEMP + 400.0 * blob * (st.solid < 0.5)
    st.data[CH["vel_x"]] = 0.5 * torch.sin(3.1 * yy)
    s.project(st)
    m0 = float(st.data[CH["density"]].double().sum())
    for _ in range(steps):
        s.step(st, Forces(), SIM_DT)
    m1 = float(st.data[CH["density"]].double().sum())
    expected = np.exp(-dissipation * steps * SIM_DT)
    return (m1 / m0) / expected - 1.0, bool(torch.isfinite(st.data).all())


def determinism(res, scene, steps=40):
    fn, obs = SCENES[scene]
    outs = []
    for _ in range(2):
        s = Solver(res)
        st = s.reset(obs)
        for i in range(steps):
            s.step(st, fn(i * SIM_DT), SIM_DT)
        outs.append(st.data.clone())
    return bool(torch.equal(outs[0], outs[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--steps", type=int, default=150)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    wp.config.quiet = True
    snaps = [args.steps // 3, 2 * args.steps // 3, args.steps]

    for name, (fn, obs) in SCENES.items():
        res_frames, solids = {}, {}
        for res in (128, 512):
            s = Solver(res)
            st = s.reset(obs)
            frames, t = timed_run(s, st, fn, args.steps, snaps)
            solids[res] = st.solid
            div = float(s.divergence(st).max())
            it = int(s.pressure.it.numpy()[0])
            Tmax = float(st.data[CH["temperature"]].max())
            print(f"{name:9s} {res:4d}^2  ms/step mean {t[5:].mean():.3f}  median {np.median(t[5:]):.3f}  "
                  f"max {t[5:].max():.3f} | max norm. div {div:.2e} | last PCG iters {it} | Tmax {Tmax:.0f} K "
                  f"| finite {bool(torch.isfinite(st.data).all())}")
            res_frames[res] = frames
        errs = [large_scale_err(a[0], b[0]) for a, b in zip(res_frames[128], res_frames[512])]
        print(f"{name:9s} LR vs pooled HR large-scale rel. L2 (blur 3 LR cells) at steps {snaps}: "
              + ", ".join(f"{e:.3f}" for e in errs))
        for k, (a, b) in enumerate(zip(res_frames[128], res_frames[512])):
            bl, bh = budgets(a, 1 / 128), budgets(b, 1 / 512)
            print(f"{name:9s} step {snaps[k]:3d} LR/HR: smoke mass {bl[0]:.4f}/{bh[0]:.4f}  heat {bl[1]:.1f}/{bh[1]:.1f} K m^2"
                  f"  smoke centroid y {bl[2]:.3f}/{bh[2]:.3f} m  heat centroid y {bl[3]:.3f}/{bh[3]:.3f} m")
        strip = np.concatenate([to_img(a, b, solid_lr=solids[128], solid_hr=solids[512])
                                for a, b in zip(res_frames[128], res_frames[512])], 0)
        save_png(strip, os.path.join(args.out, f"{name}.png"))

    for res in (128, 512):
        drift, ok = mass_drift(res)
        print(f"mass drift {res}^2, 1000 steps closed box, no sources: {100 * drift:+.5f} %  finite {ok}")
    for res in (128, 512):
        drift, ok = mass_drift(res, steps=300, dissipation=0.5)
        print(f"dissipation 0.5/s {res}^2, 300 steps: mass vs exp(-k t) {100 * drift:+.5f} %  finite {ok}")
    for res in (128, 512):
        print(f"determinism {res}^2 (campfire, 40 steps, two fresh solvers): bit-identical = "
              f"{determinism(res, 'campfire')}")
    print("PNGs in", args.out)


if __name__ == "__main__":
    main()
