"""3D solver self-check (owner's quick look, NOT the gate tests; those live in tests3d/).

    .venv/Scripts/python -m vfx.sim3d.selfcheck [--out DIR] [--steps 150] [--quick] [--bench]

Runs a campfire and a smoke plume past a sphere at 64^3 and 128^3, prints ms/step (CUDA events,
split by stage), divergence, PCG iterations, budgets, and the gates: divergence with/without a
sphere, mass drift over 1000 steps (closed box, no sources, dissipation 0), hot fluid rises,
nothing inside walls / obstacles, bit-identical reruns, 1000-step NaN stress run (strong fire +
impulses). --bench adds the 64/96/128/256 timing table (full and runtime quality) + VRAM.
--fused: the opt-in fused pressure solve (adaptive and fixed-cost levels) at 64^3: timings, cold
project(), stress divergence. --hires: the opt-in finer scalar grid (scalar_res 96/128/192):
fine-mass drift, restriction consistency, rasterisation, determinism, stress, cost + VRAM.
PNGs (one per scene): rows = snapshots; columns per resolution =
  [density mid-z slice | density max-projection along z | temperature slice | temperature MIP]
"""
from __future__ import annotations

import argparse
import os
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

from vfx.core.fields import AMBIENT_TEMP
from vfx.core.fields3d import CH3, SIM_DT, Emitter3D, Forces3D, Impulse3D, Obstacle3D
from vfx.sim3d import Solver3D, SolverConfig3D, fused_config, fused_fixed_config, runtime_config

DEFAULT_OUT = os.path.join(os.environ.get("LOCALAPPDATA", "."), "Temp", "neural-vfx-sim3d-selfcheck")


# ------------------------------------------------------------------ scenes
def campfire(t: float) -> Forces3D:
    """A small cluster of fuel sources (a 'log pile') with slow deterministic flicker."""
    em = []
    spots = [(0.5, 0.5), (0.44, 0.47), (0.56, 0.47), (0.47, 0.56), (0.55, 0.55)]
    for k, (x, z) in enumerate(spots):
        s = 1.0 + 0.35 * np.sin(2.3 * t + 1.7 * k)
        em.append(Emitter3D(pos=(x, 0.07, z), radius=0.045, density_rate=0.0, temperature=900.0,
                            fuel_rate=8.0 * s, velocity=(0.0, 0.4, 0.0)))
    return Forces3D(emitters=em)


def plume(t: float) -> Forces3D:
    return Forces3D(emitters=[Emitter3D(pos=(0.42, 0.1, 0.5), radius=0.06, density_rate=1.0, temperature=900.0,
                                        velocity=(0.0, 1.2, 0.0))], wind=(0.4, 0.0, 0.0))


SPHERE = Obstacle3D("sphere", (0.5, 0.55, 0.5), (0.12, 0.12, 0.12))
SCENES = {"campfire": (campfire, []), "plume": (plume, [SPHERE])}


# ------------------------------------------------------------------ images
def colorize_density(d):
    v = np.clip(d / 0.6, 0, 1) ** 0.6
    return np.stack([v, v, v], -1)


def colorize_temp(T):
    x = np.clip((T - AMBIENT_TEMP) / (1800.0 - AMBIENT_TEMP), 0, 1)
    return np.stack([np.clip(3 * x, 0, 1), np.clip(3 * x - 1, 0, 1), np.clip(3 * x - 2, 0, 1)], -1)


def panels(frame, solid, size=256):
    """frame (2,D,H,W) density/temperature -> (size, 4*size, 3) float: slice + MIP for each."""
    N = frame.shape[-1]

    def up(a):  # (H,W) torch -> (size,size) numpy, flipped so +y is up
        return F.interpolate(a[None, None].float(), size=(size, size), mode="nearest")[0, 0].cpu().numpy()[::-1]

    def tint(img, mask2d):
        m = up(mask2d) > 0.5
        img[m] = img[m] * 0.5 + np.array([0.1, 0.2, 0.45])
        return img
    mid = N // 2
    s_slice, s_proj = solid[mid], solid.amax(0)
    cols = [tint(colorize_density(up(frame[0, mid])), s_slice), tint(colorize_density(up(frame[0].amax(0))), s_proj),
            tint(colorize_temp(up(frame[1, mid])), s_slice), tint(colorize_temp(up(frame[1].amax(0))), s_proj)]
    img = np.concatenate(cols, 1)
    img[:, [size, 2 * size, 3 * size]] = 0.3
    return img


def save_png(img: np.ndarray, path: str):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.dtype != np.uint8 else img
    try:
        from PIL import Image
        Image.fromarray(img).save(path)
    except ImportError:
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame
        pygame.image.save(pygame.surfarray.make_surface(np.ascontiguousarray(img.transpose(1, 0, 2))), path)


# ------------------------------------------------------------------ measurements
def run_scene(res, fn, obs, steps, snaps, cfg=None):
    s = Solver3D(res, cfg)
    st = s.reset(obs)
    frames, ev = [], []
    for i in range(steps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        s.step(st, fn(i * SIM_DT), SIM_DT)
        b.record()
        ev.append((a, b))
        if i + 1 in snaps:
            frames.append(st.data[[CH3["density"], CH3["temperature"]]].clone())
    torch.cuda.synchronize()
    t = np.array([a.elapsed_time(b) for a, b in ev])
    return s, st, frames, t


def budgets(f, dx):
    d, T = f[0].double(), (f[1].double() - AMBIENT_TEMP).clamp(min=0)
    y = (torch.arange(d.shape[1], device=d.device, dtype=torch.float64) + 0.5) * dx
    m, h = d.sum() * dx ** 3, T.sum() * dx ** 3
    yc = (d.sum((0, 2)) * y).sum() * dx ** 3 / (m + 1e-12)
    yh = (T.sum((0, 2)) * y).sum() * dx ** 3 / (h + 1e-12)
    return float(m), float(h), float(yc), float(yh)


def profile(res, fn, obs, cfg, warm=60, n=60, scalar_res=None):
    s = Solver3D(res, cfg, scalar_res=scalar_res)
    st = s.reset(obs)
    for i in range(warm):
        s.step(st, fn(i * SIM_DT), SIM_DT)
    torch.cuda.synchronize()
    ev = []
    for i in range(warm, warm + n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        s.step(st, fn(i * SIM_DT), SIM_DT)
        b.record()
        ev.append((a, b))
    torch.cuda.synchronize()
    t = np.median([a.elapsed_time(b) for a, b in ev])
    pr = [s.step_profiled(st, fn(i * SIM_DT), SIM_DT) for i in range(warm + n, warm + n + 30)]
    p = {k: float(np.median([q[k] for q in pr])) for k in pr[0]}
    p["step"] = float(t)
    p["iters"] = s.pressure_iters()
    p["div"] = float(s.divergence(st).max())
    p["vram_mb"] = solver_vram_mb(s, st)
    return p


def solver_vram_mb(s, st):
    """Bytes held by the solver's own buffers + the state tensors (deterministic, excludes the CUDA
    context and other processes)."""
    tot = st.data.numel() * 4 + st.solid.numel() * 4
    if s.scalar_res is not None:
        tot += (s.hi.numel() + s.hi_solid.numel()) * 4
    seen = set()
    for obj in [o for o in (s, s.pressure, s.pressure_fused) if o is not None]:
        for v in vars(obj).values():
            arrs = [v] if isinstance(v, wp.array) else (
                [a for lv in v for a in lv.values() if isinstance(a, wp.array)] if isinstance(v, list) and v and isinstance(v[0], dict) else [])
            for a in arrs:
                if a.device.is_cuda and id(a) not in seen:
                    seen.add(id(a))
                    tot += a.capacity
    return tot / 2 ** 20


def gpu_status():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return f"GPU util, memory used (all processes) = {out}"
    except Exception as e:  # pragma: no cover
        return f"nvidia-smi failed: {e}"


def project_gate(res, with_sphere, cfg=None):
    torch.manual_seed(1)
    s = Solver3D(res, cfg)
    st = s.reset([SPHERE] if with_sphere else [])
    v = F.avg_pool3d(torch.randn(1, 3, res, res, res, device="cuda"), 3, 1, 1)[0]
    st.data[1:4] = v * (st.solid < 0.5)
    s.project(st)
    return float(s.divergence(st).max()), s.pressure_iters()


def mass_drift(res, steps=1000):
    s = Solver3D(res, SolverConfig3D(density_dissipation=0.0))
    st = s.reset([Obstacle3D("sphere", (0.6, 0.6, 0.55), (0.1, 0.1, 0.1))])
    c = (torch.arange(res, device="cuda", dtype=torch.float32) + 0.5) / res
    zz, yy, xx = torch.meshgrid(c, c, c, indexing="ij")
    blob = torch.exp(-((xx - 0.35) ** 2 + (yy - 0.3) ** 2 + (zz - 0.45) ** 2) / 0.01) * (st.solid < 0.5)
    st.data[CH3["density"]] = blob
    st.data[CH3["temperature"]] = AMBIENT_TEMP + 400.0 * blob
    st.data[CH3["vel_x"]] = 0.5 * torch.sin(3.1 * yy)
    st.data[CH3["vel_z"]] = 0.3 * torch.cos(2.7 * xx)
    s.project(st)
    m0 = float(st.data[CH3["density"]].double().sum())
    for _ in range(steps):
        s.step(st, Forces3D(), SIM_DT)
    m1 = float(st.data[CH3["density"]].double().sum())
    return m1 / m0 - 1.0, bool(torch.isfinite(st.data).all())


def hot_rises(res, steps=30):
    s = Solver3D(res)
    st = s.reset()
    c = (torch.arange(res, device="cuda", dtype=torch.float32) + 0.5) / res
    zz, yy, xx = torch.meshgrid(c, c, c, indexing="ij")
    blob = torch.exp(-((xx - 0.5) ** 2 + (yy - 0.3) ** 2 + (zz - 0.5) ** 2) / 0.006)
    st.data[CH3["temperature"]] = AMBIENT_TEMP + 300.0 * blob
    w = lambda: ((st.data[CH3["temperature"]] - AMBIENT_TEMP).clamp(min=0) * yy).sum() / (
        (st.data[CH3["temperature"]] - AMBIENT_TEMP).clamp(min=0).sum() + 1e-12)
    y0 = float(w())
    for _ in range(steps):
        s.step(st, Forces3D(), SIM_DT)
    return y0, float(w()), float(st.data[CH3["vel_y"]].max())


def walls_clean(st):
    solid = st.solid > 0.5
    d = st.data
    bad = (d[CH3["density"]][solid].abs().max() if solid.any() else torch.tensor(0.0)).item()
    bad = max(bad, d[1:4][:, solid].abs().max().item() if solid.any() else 0.0)
    return bad


def determinism(res, steps=40):
    outs = []
    for _ in range(2):
        s = Solver3D(res)
        st = s.reset([SPHERE])
        for i in range(steps):
            f = campfire(i * SIM_DT)
            f.impulses = [Impulse3D(pos=(0.5, 0.3, 0.5), radius=0.08, velocity=(1.0, 0.5, -0.5))] if i % 10 == 0 else []
            s.step(st, f, SIM_DT)
        outs.append(st.data.clone())
    return bool(torch.equal(outs[0], outs[1]))


def stress(res, steps=1000, cfg=None, scalar_res=None):
    s = Solver3D(res, cfg, scalar_res=scalar_res)
    st = s.reset([SPHERE, Obstacle3D("box", (0.25, 0.7, 0.3), (0.1, 0.02, 0.15))])
    rng = np.random.default_rng(0)
    vmax = 0.0
    for i in range(steps):
        f = campfire(i * SIM_DT)
        for e in f.emitters:
            e.fuel_rate *= 4.0
            e.temperature = 1500.0
        f.emitters.append(Emitter3D(pos=(0.3, 0.2, 0.7), radius=0.08, density_rate=6.0, temperature=1200.0,
                                    fuel_rate=20.0, velocity=(1.5, 2.0, 0.0)))
        if i % 7 == 0:
            p = rng.uniform(0.15, 0.85, 3)
            f.impulses = [Impulse3D(pos=tuple(p), radius=0.07, velocity=tuple(rng.uniform(-6, 6, 3)))]
        f.wind = (float(np.sin(0.01 * i)), 0.0, float(np.cos(0.013 * i)))
        s.step(st, f, SIM_DT)
        if i % 100 == 99:
            vmax = max(vmax, float(st.velocity.abs().max()))
    ok = bool(torch.isfinite(st.data).all()) and (s.scalar_res is None or bool(torch.isfinite(s.hi).all()))
    return ok, vmax, float(st.data[CH3["temperature"]].max()), float(s.divergence(st).max()), walls_clean(st)


# ------------------------------------------------------------------ finer scalar grid (opt-in)
def hires_mass(S, steps=1000):
    """Closed box, no sources, dissipation 0: fine-field density total drift."""
    s = Solver3D(64, SolverConfig3D(density_dissipation=0.0), scalar_res=S)
    st = s.reset([Obstacle3D("sphere", (0.6, 0.6, 0.55), (0.1, 0.1, 0.1))])
    c = (torch.arange(S, device="cuda", dtype=torch.float32) + 0.5) / S
    zz, yy, xx = torch.meshgrid(c, c, c, indexing="ij")
    blob = torch.exp(-((xx - 0.35) ** 2 + (yy - 0.3) ** 2 + (zz - 0.45) ** 2) / 0.01) * (s.hi_solid < 0.5)
    s.hi[0] = blob
    s.hi[1] = AMBIENT_TEMP + 400.0 * blob
    c = (torch.arange(64, device="cuda", dtype=torch.float32) + 0.5) / 64
    zz, yy, xx = torch.meshgrid(c, c, c, indexing="ij")
    st.data[CH3["vel_x"]] = 0.5 * torch.sin(3.1 * yy)
    st.data[CH3["vel_z"]] = 0.3 * torch.cos(2.7 * xx)
    s.project(st)
    m0 = float(s.hi[0].double().sum())
    for _ in range(steps):
        s.step(st, Forces3D(), SIM_DT)
    return float(s.hi[0].double().sum()) / m0 - 1.0, bool(torch.isfinite(s.hi).all())


def hires_restriction(S, steps=90):
    """Coarse density vs box-average of the fine field over coarse fluid cells: rel L1, mass rel."""
    s = Solver3D(64, scalar_res=S)
    st = s.reset([SPHERE])
    for i in range(steps):
        s.step(st, plume(i * SIM_DT), SIM_DT)
    fl = st.solid < 0.5
    rd = F.interpolate(s.hi[0][None, None], size=(64,) * 3, mode="area")[0, 0] * fl
    cd = st.data[0] * fl
    l1 = float((rd - cd).abs().sum() / cd.abs().sum())
    return l1, float(abs(rd.double().sum() - cd.double().sum()) / cd.double().sum())


def hires_determinism(S, cfg=None, steps=60):
    outs = []
    for _ in range(2):
        s = Solver3D(64, cfg, scalar_res=S)
        st = s.reset([SPHERE])
        for i in range(steps):
            s.step(st, campfire(i * SIM_DT), SIM_DT)
        outs.append((st.data.clone(), s.hi.clone()))
    return bool(torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1]))


def hires_raster(S):
    """One step from rest: fine emitter footprint vs the analytic (1-r^2/R^2)^2 at fine centers;
    fine solid == rasterize_obstacles_3d at S; fine solids hold exactly 0 / ambient."""
    from vfx.core.fields3d import rasterize_obstacles_3d
    s = Solver3D(64, SolverConfig3D(density_dissipation=0.0), scalar_res=S)
    st = s.reset([SPHERE])
    e = Emitter3D(pos=(0.31, 0.2, 0.47), radius=0.06, density_rate=3.0)
    s.step(st, Forces3D(emitters=[e]), SIM_DT)
    c = (torch.arange(S, device="cuda", dtype=torch.float64) + 0.5) / S
    zz, yy, xx = torch.meshgrid(c, c, c, indexing="ij")
    r2 = ((xx - e.pos[0]) ** 2 + (yy - e.pos[1]) ** 2 + (zz - e.pos[2]) ** 2) / e.radius ** 2
    want = torch.where(r2 < 1, (1 - r2) ** 2, torch.zeros_like(r2)) * e.density_rate * SIM_DT
    err = float((s.hi[0].double() - want).abs().max() / want.max())
    same = bool(torch.equal(s.hi_solid, rasterize_obstacles_3d([SPHERE], S)))
    m = s.hi_solid > 0.5
    clean = float(max(s.hi[0][m].abs().max(), (s.hi[1][m] - AMBIENT_TEMP).abs().max()))
    return err, same, clean


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--quick", action="store_true", help="scenes + images only")
    ap.add_argument("--bench", action="store_true", help="timing table 64/96/128/256")
    ap.add_argument("--fused", action="store_true", help="opt-in fused pressure solve checks + timings")
    ap.add_argument("--hires", action="store_true", help="opt-in finer scalar grid checks + timings")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    wp.config.quiet = True
    print("GPU before run:", gpu_status())
    snaps = [args.steps // 3, 2 * args.steps // 3, args.steps]

    for name, (fn, obs) in SCENES.items():
        rows = {}
        for res in (64, 128):
            s, st, frames, t = run_scene(res, fn, obs, args.steps, snaps)
            div = float(s.divergence(st).max())
            print(f"{name:8s} {res:3d}^3 ms/step median {np.median(t[10:]):.3f} mean {t[10:].mean():.3f} "
                  f"| max norm div {div:.2e} | PCG iters {int(s.pressure.it.numpy()[0])} "
                  f"| Tmax {float(st.data[CH3['temperature']].max()):.0f} K | solid clean {walls_clean(st):.1e} "
                  f"| finite {bool(torch.isfinite(st.data).all())}")
            for k, f in enumerate(frames):
                m, h, yc, yh = budgets(f, 1.0 / res)
                print(f"           step {snaps[k]:3d}: smoke {m:.4f} heat {h:.2f} K m^3  centroid y smoke {yc:.3f} heat {yh:.3f}")
            rows[res] = [panels(f, st.solid) for f in frames]
        img = np.concatenate([np.concatenate([a, np.full((a.shape[0], 8, 3), 1.0), b], 1)
                              for a, b in zip(rows[64], rows[128])], 0)
        save_png(img, os.path.join(args.out, f"{name}.png"))
    if not args.quick:
        for res in (64, 128):
            for sph in (False, True):
                d, it = project_gate(res, sph)
                print(f"gate div {res}^3 sphere={sph}: {d:.2e} (iters {it})  [< 1e-4]")
        for res in (64, 128):
            print(f"gate hot rises {res}^3: heat centroid y, max v_y after 1 s = {hot_rises(res)}")
        for res in (64, 128):
            drift, ok = mass_drift(res)
            print(f"gate mass {res}^3 1000 steps closed box: {100 * drift:+.5f} %  finite {ok}  [|.| < 0.1 %]")
        for res in (64, 128):
            print(f"gate determinism {res}^3: bit-identical = {determinism(res)}")
        for res in (64,):
            for label, cfg in (("full", None), ("runtime", runtime_config())):
                ok, vmax, Tm, dv, wc = stress(res, cfg=cfg)
                print(f"gate stress {res}^3 {label} 1000 steps: finite {ok}  max|u| {vmax:.2f} m/s  Tmax {Tm:.0f} K  "
                      f"div {dv:.2e}  solid clean {wc:.1e}")
    if args.fused:
        print("GPU before fused:", gpu_status())
        levels = (("fused adaptive", fused_config()), ("fused fixed (lower quality)", fused_fixed_config()))
        for label, cfg in levels:
            for sph in (False, True):
                d, it = project_gate(64, sph, cfg)
                print(f"fused cold project 64^3 sphere={sph} {label}: div {d:.2e} (V-cycles {it})  [< 1e-4]")
            ok, vmax, Tm, dv, wc = stress(64, cfg=cfg)
            print(f"fused stress 64^3 {label}: finite {ok} max|u| {vmax:.2f} div(end) {dv:.2e} solid clean {wc:.1e}")
        for label, cfg in (("default pcg", SolverConfig3D()),) + levels:
            for name, (fn, obs) in SCENES.items():
                p = profile(64, fn, obs, cfg)
                print(f"fused-bench 64^3 {label:28s} {name:8s} step {p['step']:.3f} ms | adv {p['advection']:.3f} "
                      f"forces {p['forces']:.3f} proj {p['projection']:.3f} | iters {p['iters']} div {p['div']:.1e}")
        print("GPU after fused:", gpu_status())
    if args.hires:
        print("GPU before hires:", gpu_status())
        for S in (128,):
            drift, ok = hires_mass(S)
            print(f"hires {S}^3 fine mass 1000 steps closed box: {100 * drift:+.5f} %  finite {ok}  [|.| < 0.1 %]")
            l1, mrel = hires_restriction(S)
            print(f"hires {S}^3 restriction vs 64^3 density: rel L1 {l1:.2e}, mass rel {mrel:.2e}")
            print(f"hires {S}^3 emitter rel err, solid == raster, solid clean: {hires_raster(S)}")
            print(f"hires {S}^3 determinism: {hires_determinism(S)} (fused: {hires_determinism(S, fused_config())})")
            ok, vmax, Tm, dv, wc = stress(64, cfg=None, scalar_res=S)
            print(f"hires {S}^3 stress 1000 steps: finite {ok} max|u| {vmax:.2f} Tmax {Tm:.0f} div {dv:.2e} "
                  f"solid clean {wc:.1e}")
        for S in (None, 96, 128, 192):
            for label, cfg in (("pcg", SolverConfig3D()), ("fused", fused_config())):
                p = profile(64, plume, [SPHERE], cfg, scalar_res=S)
                stages = " ".join(f"{k} {p[k]:.3f}" for k in ("advection", "scalars", "forces", "projection") if k in p)
                print(f"hires-bench 64^3 + scalars {S} {label:5s} plume step {p['step']:.3f} ms | {stages} "
                      f"| solver VRAM {p['vram_mb']:.0f} MiB")
        print("GPU after hires:", gpu_status())
    if args.bench:
        print("GPU before bench:", gpu_status())
        for res in (64, 96, 128, 256):
            for label, cfg in (("full", SolverConfig3D()), ("runtime", runtime_config())):
                for name, (fn, obs) in SCENES.items():
                    p = profile(res, fn, obs, cfg)
                    print(f"bench {res:3d}^3 {label:7s} {name:8s} step {p['step']:.3f} ms | adv {p['advection']:.3f} "
                          f"forces {p['forces']:.3f} proj {p['projection']:.3f} | iters {p['iters']} div {p['div']:.1e} "
                          f"| solver VRAM {p['vram_mb']:.0f} MiB")
        print("GPU after bench:", gpu_status())
    print("PNGs in", args.out)


if __name__ == "__main__":
    main()
