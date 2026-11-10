"""Gate 1 checks (native/README.md, "Gate 1 — native solver: what passes means"), backend-agnostic.

Every check takes a backend factory `make(res, config) -> solver with the NativeSolver interface` and
returns a dict: {"pass": bool, "summary": str, ...numbers}. The Python reference (vfx.sim3d.Solver3D) is
run in-process, in the same session, on the same inputs. Tests in test_gate1.py call these and record
the dicts; test_harness_controls.py calls them with deliberately broken fakes and expects failures.

Error measures (native/README.md), per channel c at one step, computed in float64 over all cells:
    rel_l2   = ||n - p||_2 / (||p - ambient_c||_2 + EPS)
    abs/peak = |n - p| / max|p - ambient_c|            (ambient 293 K for temperature, 0 otherwise)
EPS is not given by the contract: EPS = 1e-12 here. If max|p - ambient_c| == 0 (the reference channel is
exactly ambient, e.g. fuel in `plume`), abs/peak is 0 where n == p and +inf elsewhere (strict).
"""
from __future__ import annotations

import dataclasses
import subprocess
import time

import numpy as np
import torch

from vfx.core.fields import AMBIENT_TEMP
from vfx.core.fields3d import CHANNELS_3D, Forces3D, rasterize_obstacles_3d
from vfx.sim3d import Solver3D, SolverConfig3D, runtime_config
from vfx.sim3d import selfcheck as SC

from nvfx_scenes import DT, get_scene

RES = 64
EPS = 1e-12
AMBIENT = np.array([0.0, 0.0, 0.0, 0.0, AMBIENT_TEMP, 0.0])

# thresholds, quoted from native/README.md (do not edit to make a test pass)
P1_REL_L2, P1_P999, P1_MAX = 1e-4, 1e-3, 5e-2
P2_REL_L2 = 1e-4
P3_MASS_REL, P3_HEAT_REL, P3_CENTROID_M = 0.01, 0.01, 0.01
S1_DIV = 1e-4
S2_MASS_REL = 1e-3
T1_BUDGET_MS = 2.0
# S4 "max|u| and Tmax bounded as in Python": the contract gives no number. PROVISIONAL reading used here
# (flagged to the lead): finite, Tmax <= max_temperature (the Python hard cap), max|u| <= 1.5 x the
# Python max|u| of the same scenario measured in the same session.
S4_VMAX_FACTOR = 1.5

P1_CFG = SolverConfig3D(pressure_tol=0.0, pressure_max_iters=8, pressure_warm_start=False)
ZERO_SOURCES = dict(buoyancy_alpha=0.0, buoyancy_beta=0.0, vorticity_eps=0.0, density_dissipation=0.0,
                    temperature_cooling=0.0, burn_rate=0.0, heat_diffusion=0.0)
P1_STEPS = 120
P3_STEPS = 120
S1_STEPS = 120
S2_STEPS = 1000
S3_STEPS = 200
S4_STEPS = 1000
T1_WARM, T1_N = 60, 200


# ------------------------------------------------------------------ measures
def channel_errors(n: np.ndarray, p: np.ndarray) -> dict:
    """Per-channel rel_l2, p99.9 and max of abs/peak; arrays (6,D,H,W)."""
    rel, p999, mx, peak = [], [], [], []
    for c in range(n.shape[0]):
        pc = p[c].astype(np.float64)
        d = np.abs(n[c].astype(np.float64) - pc)
        dp = pc - AMBIENT[c]
        pk = float(np.abs(dp).max())
        rel.append(float(np.sqrt((d * d).sum()) / (np.sqrt((dp * dp).sum()) + EPS)))
        if pk > 0.0:
            ap = d / pk
        else:
            ap = np.where(d == 0.0, 0.0, np.inf)
        p999.append(float(np.percentile(ap, 99.9)))
        mx.append(float(ap.max()))
        peak.append(pk)
    return dict(rel_l2=rel, p999=p999, max=mx, peak=peak)


def rel_l2(n: np.ndarray, p: np.ndarray, amb: float = 0.0) -> float:
    d = n.astype(np.float64) - p.astype(np.float64)
    dp = p.astype(np.float64) - amb
    return float(np.sqrt((d * d).sum()) / (np.sqrt((dp * dp).sum()) + EPS))


def _np_solid(obstacles):
    return rasterize_obstacles_3d(list(obstacles), RES, device="cuda").cpu().numpy()


def gpu_snapshot() -> dict:
    out = {}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        name, util, used, tot = [s.strip() for s in r.stdout.strip().splitlines()[0].split(",")]
        out.update(gpu=name, util_pct=int(util), mem_used_mib=int(used), mem_total_mib=int(tot))
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        out["compute_apps"] = [s.strip() for s in r.stdout.strip().splitlines() if s.strip()]
    except Exception as e:  # pragma: no cover
        out["error"] = repr(e)
    out["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return out


# ------------------------------------------------------------------ P1
def check_p1(make, scene_name: str, steps=range(P1_STEPS)) -> dict:
    """Teacher-forced single-step parity. The Python trajectory runs with the DEFAULT config (it provides
    realistic states x_k); each x_k is then stepped once by (a) a separate Python one-step solver with
    P1_CFG (warm start off -> phi is zeroed in the rhs kernel, so the step is a pure function of
    x_k, forces, dt, config: see test_python_one_step_is_memoryless) and (b) the backend after
    upload(x_k) with P1_CFG."""
    sc = get_scene(scene_name)
    steps = list(steps)
    traj = Solver3D(RES, SolverConfig3D())
    st = traj.reset(sc.obstacles)
    one = Solver3D(RES, P1_CFG)
    ost = one.reset(sc.obstacles)
    b = make(RES, P1_CFG)
    b.reset(sc.obstacles)
    solid_ref = st.solid.cpu().numpy()
    solid_mismatch = int((b.download_solid() != solid_ref).sum())
    rows = []
    worst = None
    first_fail = None
    for k in range(max(steps) + 1):
        f = sc.forces(k)
        if k in steps:
            x = st.data.cpu().numpy()
            ost.data.copy_(st.data)
            one.step(ost, f, DT)
            ref = ost.data.cpu().numpy()
            b.upload(x)
            b.step(f, DT)
            n = b.download()
            e = channel_errors(n, ref)
            e["step"] = k
            rows.append(e)
            for c in range(6):
                for key, thr in (("rel_l2", P1_REL_L2), ("p999", P1_P999), ("max", P1_MAX)):
                    ratio = e[key][c] / thr
                    if worst is None or ratio > worst["ratio"]:
                        worst = dict(step=k, channel=CHANNELS_3D[c], measure=key, value=e[key][c], threshold=thr,
                                     ratio=ratio)
                    if ratio > 1.0 and first_fail is None:
                        first_fail = dict(step=k, channel=CHANNELS_3D[c], measure=key, value=e[key][c],
                                          threshold=thr)
        traj.step(st, f, DT)
    per_ch = {CHANNELS_3D[c]: dict(rel_l2_max=max(r["rel_l2"][c] for r in rows),
                                   p999_max=max(r["p999"][c] for r in rows),
                                   max_max=max(r["max"][c] for r in rows),
                                   zero_peak_steps=sum(1 for r in rows if r["peak"][c] == 0.0))
              for c in range(6)}
    ok = first_fail is None and solid_mismatch == 0
    summ = (f"P1 {scene_name} steps {steps[0]}..{steps[-1]}: worst {worst['measure']} {worst['value']:.3e} "
            f"({worst['channel']}, step {worst['step']}, {worst['ratio']:.3g}x threshold); solid mismatch "
            f"{solid_mismatch} cells")
    if first_fail:
        summ += f"; FIRST FAIL step {first_fail['step']} {first_fail['channel']} {first_fail['measure']} " \
                f"{first_fail['value']:.3e} > {first_fail['threshold']:.0e}"
    return {"pass": ok, "summary": summ, "worst": worst, "first_fail": first_fail, "per_channel": per_ch,
            "solid_mismatch_cells": solid_mismatch,
            "curve_rel_l2": [[r["step"]] + r["rel_l2"] for r in rows]}


# ------------------------------------------------------------------ P2
def _random_velocity(seed=1):
    torch.manual_seed(seed)
    return torch.nn.functional.avg_pool3d(torch.randn(1, 3, RES, RES, RES, device="cuda"), 3, 1, 1)[0]


def _project_cases():
    """(name, obstacles, data (6,N,N,N) numpy). selfcheck.project_gate's smoothed random field without /
    with the sphere, and the plume state at step 60 with that field added (non-trivial background)."""
    cases = []
    for name, obs in (("random", []), ("random+sphere", [SC.SPHERE])):
        s = Solver3D(RES)
        st = s.reset(obs)
        st.data[1:4] = _random_velocity() * (st.solid < 0.5)
        cases.append((name, obs, st.data.cpu().numpy()))
    sc = get_scene("plume")
    s = Solver3D(RES)
    st = s.reset(sc.obstacles)
    for k in range(60):
        s.step(st, sc.forces(k), DT)
    st.data[1:4] += 0.5 * _random_velocity(2) * (st.solid < 0.5)
    cases.append(("plume@60+random", sc.obstacles, st.data.cpu().numpy()))
    return cases


def check_p2_project(make) -> dict:
    """nvfx_solver_project vs Solver3D.project on the same field, both with P1_CFG (fixed 8 PCG iterations,
    warm start off: a fresh Python solver, the backend after upload)."""
    res = {}
    ok = True
    for name, obs, data in _project_cases():
        py = Solver3D(RES, P1_CFG)
        st = py.reset(obs)
        st.data.copy_(torch.from_numpy(data))
        py.project(st)
        ref = st.data.cpu().numpy()
        b = make(RES, P1_CFG)
        b.reset(obs)
        b.upload(data)
        b.project()
        n = b.download()
        rl = [rel_l2(n[c], ref[c]) for c in (1, 2, 3)]
        other = float(np.abs(n[[0, 4, 5]].astype(np.float64) - ref[[0, 4, 5]]).max())
        res[name] = dict(vel_rel_l2=rl, other_channels_max_abs=other,
                         div_after_python=float(py.divergence(st).max()))
        ok &= max(rl) <= P2_REL_L2
        b.close()
    worst = max((max(v["vel_rel_l2"]), k) for k, v in res.items())
    return {"pass": ok, "summary": f"P2 project: worst velocity rel_l2 {worst[0]:.3e} ({worst[1]}), "
                                   f"threshold {P2_REL_L2:.0e}", "cases": res}


def check_p2_zero_sources(make, scene_name: str, at_step: int = 60) -> dict:
    """One step with every source / force term zeroed (config) and no emitters/impulses/wind: pure advection
    + mass correction + projection, from the Python default-config state at `at_step`."""
    sc = get_scene(scene_name)
    traj = Solver3D(RES)
    st = traj.reset(sc.obstacles)
    for k in range(at_step):
        traj.step(st, sc.forces(k), DT)
    x = st.data.cpu().numpy()
    cfg = dataclasses.replace(P1_CFG, **ZERO_SOURCES)
    py = Solver3D(RES, cfg)
    pst = py.reset(sc.obstacles)
    pst.data.copy_(st.data)
    py.step(pst, Forces3D(), DT)
    ref = pst.data.cpu().numpy()
    b = make(RES, cfg)
    b.reset(sc.obstacles)
    b.upload(x)
    b.step(Forces3D(), DT)
    n = b.download()
    b.close()
    rl = [rel_l2(n[c], ref[c]) for c in (1, 2, 3)]
    allch = channel_errors(n, ref)
    ok = max(rl) <= P2_REL_L2
    return {"pass": ok, "summary": f"P2 zero-sources step {scene_name}@{at_step}: velocity rel_l2 "
                                   f"{', '.join(f'{v:.3e}' for v in rl)} (x,y,z), threshold {P2_REL_L2:.0e}",
            "vel_rel_l2": rl, "all_channels": allch}


# ------------------------------------------------------------------ P3
def budgets(data: np.ndarray) -> dict:
    d = data[0].astype(np.float64)
    T = data[4].astype(np.float64) - AMBIENT_TEMP
    dx = 1.0 / RES
    c = (np.arange(RES) + 0.5) * dx
    m = d.sum()
    cz = (d.sum((1, 2)) * c).sum() / m if m > 0 else np.nan
    cy = (d.sum((0, 2)) * c).sum() / m if m > 0 else np.nan
    cx = (d.sum((0, 1)) * c).sum() / m if m > 0 else np.nan
    return dict(mass=float(m * dx ** 3), heat=float(T.sum() * dx ** 3), centroid=[float(cx), float(cy), float(cz)])


def check_p3(make, scene_name: str) -> dict:
    sc = get_scene(scene_name)
    py = Solver3D(RES)
    st = py.reset(sc.obstacles)
    b = make(RES, SolverConfig3D())
    b.reset(sc.obstacles)
    curve = []
    for k in range(P3_STEPS):
        f = sc.forces(k)
        py.step(st, f, DT)
        b.step(f, DT)
        if (k + 1) % 10 == 0 or k + 1 == P3_STEPS:
            n, p = b.download(), st.data.cpu().numpy()
            curve.append([k + 1] + channel_errors(n, p)["rel_l2"])
    bp, bn = budgets(p), budgets(n)
    mass_rel = abs(bn["mass"] - bp["mass"]) / abs(bp["mass"])
    heat_rel = abs(bn["heat"] - bp["heat"]) / abs(bp["heat"])
    cdist = float(np.linalg.norm(np.array(bn["centroid"]) - np.array(bp["centroid"])))
    ok = mass_rel <= P3_MASS_REL and heat_rel <= P3_HEAT_REL and cdist <= P3_CENTROID_M
    b.close()
    return {"pass": bool(ok),
            "summary": f"P3 {scene_name} @120: mass {100 * mass_rel:.4f} % (<= 1 %), heat {100 * heat_rel:.4f} % "
                       f"(<= 1 %), density centroid {cdist:.2e} m (<= 0.01); final rel_l2 per channel "
                       f"{', '.join(f'{v:.2e}' for v in curve[-1][1:])}",
            "python": bp, "backend": bn, "mass_rel": mass_rel, "heat_rel": heat_rel, "centroid_dist_m": cdist,
            "curve_rel_l2_every10": curve}


# ------------------------------------------------------------------ S1
def check_s1(make, scene_name: str) -> dict:
    sc = get_scene(scene_name)
    b = make(RES, SolverConfig3D())
    b.reset(sc.obstacles)
    divs = []
    for k in range(S1_STEPS):
        b.step(sc.forces(k), DT)
        divs.append(float(b.divergence().max()))
    b.close()
    i = int(np.argmax(divs))
    ok = max(divs) < S1_DIV
    return {"pass": ok, "summary": f"S1 {scene_name}: max normalized div over steps 1..{S1_STEPS} = {divs[i]:.3e} "
                                   f"(step {i + 1}), final {divs[-1]:.3e}, threshold < {S1_DIV:.0e}",
            "max_div": divs[i], "worst_step": i + 1, "final_div": divs[-1], "per_step": divs}


# ------------------------------------------------------------------ S2
def check_s2(make, steps: int = S2_STEPS, python_figure: bool = True) -> dict:
    sc = get_scene("mass_box")
    cfg = dataclasses.replace(SolverConfig3D(), **sc.config_overrides)
    b = make(RES, cfg)
    b.reset(sc.obstacles)
    x0 = sc.initial(RES, _np_solid(sc.obstacles))
    b.upload(x0)
    b.project()
    m0 = float(b.download()[0].astype(np.float64).sum())
    for k in range(steps):
        b.step(sc.forces(k), DT)
    d = b.download()
    m1 = float(d[0].astype(np.float64).sum())
    b.close()
    drift = m1 / m0 - 1.0
    out = {"drift": drift, "m0": m0, "m1": m1, "finite": bool(np.isfinite(d).all())}
    if python_figure:
        pdrift, pfin = SC.mass_drift(RES, steps)
        out["python_drift"] = pdrift
    ok = abs(drift) < S2_MASS_REL and out["finite"]
    out.update({"pass": ok, "summary": f"S2 mass, closed box + sphere, {steps} steps: {100 * drift:+.5f} % "
                                       f"(|.| < 0.1 %)" + (f"; Python selfcheck.mass_drift same session "
                                                          f"{100 * out['python_drift']:+.5f} %" if python_figure else "")})
    return out


# ------------------------------------------------------------------ S3
def run_scene_states(make, scene_name: str, steps: int, cfg=None, b=None) -> np.ndarray:
    sc = get_scene(scene_name)
    own = b is None
    if own:
        b = make(RES, cfg if cfg is not None else SolverConfig3D())
    b.reset(sc.obstacles)
    for k in range(steps):
        b.step(sc.forces(k), DT)
    d = b.download()
    if own:
        b.close()
    return d


def bit_compare(a: np.ndarray, b: np.ndarray) -> dict:
    ua, ub = a.view(np.uint32), b.view(np.uint32)
    diff = ua != ub
    return dict(identical=bool(not diff.any()), differing_values=int(diff.sum()),
                max_abs=float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if diff.any() else 0.0)


def check_s3(make, steps: int = S3_STEPS) -> dict:
    a = run_scene_states(make, "determinism", steps)
    c = run_scene_states(make, "determinism", steps)
    r = bit_compare(a, c)
    return {"pass": r["identical"], **r,
            "summary": f"S3 two fresh solvers, {steps} steps campfire + sphere + impulses: bit-identical "
                       f"{r['identical']} ({r['differing_values']} differing values, max abs {r['max_abs']:.3e})"}


def check_s3_reset_reuse(make, steps: int = S3_STEPS) -> dict:
    """Contract check beside S3: reset() on a used solver (clears warm start, step counter) reproduces
    a fresh solver bit for bit."""
    b = make(RES, SolverConfig3D())
    a = run_scene_states(make, "determinism", steps, b=b)
    c = run_scene_states(make, "determinism", steps, b=b)
    b.close()
    r = bit_compare(a, c)
    return {"pass": r["identical"], **r,
            "summary": f"S3b same solver, reset between two {steps}-step runs: bit-identical {r['identical']} "
                       f"({r['differing_values']} differing values)"}


# ------------------------------------------------------------------ S4
def _stress_stats(data, solid, vmax):
    s = solid > 0.5
    walls = 0.0
    if s.any():
        walls = max(float(np.abs(data[0][s]).max()), float(np.abs(data[1:4][:, s]).max()))
    return dict(finite=bool(np.isfinite(data).all()), vmax=vmax, tmax=float(data[4].max()), solid_clean=walls)


def check_s4(make, cfg=None, label="full", python_figure: bool = True) -> dict:
    cfg = cfg if cfg is not None else SolverConfig3D()
    sc = get_scene("stress")
    b = make(RES, cfg)
    b.reset(sc.obstacles)
    vmax = 0.0
    for i in range(S4_STEPS):
        b.step(sc.forces(i), DT)
        if i % 100 == 99:
            vmax = max(vmax, float(np.abs(b.download()[1:4]).max()))   # selfcheck: velocity.abs().max()
    d = b.download()
    st = _stress_stats(d, b.download_solid(), vmax)
    st["div"] = float(b.divergence().max())
    b.close()
    out = {"backend": st, "config": label}
    ok = st["finite"] and st["tmax"] <= cfg.max_temperature
    if python_figure:
        pok, pv, pT, pdiv, pw = SC.stress(RES, S4_STEPS, cfg)
        out["python"] = dict(finite=pok, vmax=pv, tmax=pT, div=pdiv, solid_clean=pw)
        ok = ok and st["vmax"] <= S4_VMAX_FACTOR * pv
    out["pass"] = bool(ok)
    out["summary"] = (f"S4 stress {label} {S4_STEPS} steps: finite {st['finite']}, max|u| {st['vmax']:.3f} m/s, "
                      f"Tmax {st['tmax']:.0f} K, div {st['div']:.2e}, solid clean {st['solid_clean']:.1e}"
                      + (f" | Python same session: max|u| {pv:.3f}, Tmax {pT:.0f} K, div {pdiv:.2e}"
                         if python_figure else ""))
    return out


# ------------------------------------------------------------------ T1
def _stats(xs):
    a = np.asarray(xs, np.float64)
    return dict(median=float(np.median(a)), p95=float(np.percentile(a, 95)), n=int(a.size))


def python_timing(scene_name: str, cfg) -> dict:
    sc = get_scene(scene_name)
    s = Solver3D(RES, cfg)
    st = s.reset(sc.obstacles)
    k = 0
    for _ in range(T1_WARM):
        s.step(st, sc.forces(k), DT)
        k += 1
    torch.cuda.synchronize()
    ev = []
    for _ in range(T1_N):
        a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        s.step(st, sc.forces(k), DT)
        e.record()
        ev.append((a, e))
        k += 1
    torch.cuda.synchronize()
    step_ms = [a.elapsed_time(e) for a, e in ev]
    pr = []
    for _ in range(T1_N):
        pr.append(s.step_profiled(st, sc.forces(k), DT))
        k += 1
    out = {st_: _stats([p[st_] for p in pr]) for st_ in ("advection", "forces", "projection", "total")}
    out["step_graph"] = _stats(step_ms)
    out["pcg_iters_last"] = int(s.pressure.it.numpy()[0])
    return out


def backend_timing(make, scene_name: str, cfg) -> dict:
    sc = get_scene(scene_name)
    b = make(RES, cfg)
    b.set_timing(True)
    b.reset(sc.obstacles)
    k = 0
    for _ in range(T1_WARM):
        b.step(sc.forces(k), DT)
        k += 1
    b.wait_idle()
    samples = {}
    method = "non-blocking get_timings after each step"

    def grab():
        t = b.get_timings()
        if t is not None and t["step_index"] > T1_WARM:   # strictly after warm-up, 0- or 1-based
            samples.setdefault(t["step_index"], t)

    while len(samples) < T1_N and k < T1_WARM + 3000:
        b.step(sc.forces(k), DT)
        k += 1
        grab()
    if len(samples) < T1_N:
        method = "wait_idle + get_timings after each step (non-blocking reads gave too few distinct steps)"
        while len(samples) < T1_N and k < T1_WARM + 6000:
            b.step(sc.forces(k), DT)
            k += 1
            b.wait_idle()
            grab()
    b.wait_idle()
    grab()
    b.close()
    ts = list(samples.values())
    if not ts:
        return {"error": "no timings returned", "method": method, "steps_run": k}
    out = {st_: _stats([t[st_] for t in ts]) for st_ in ("advection", "forces", "projection", "total")}
    out["pcg_iters"] = _stats([t["pcg_iters"] for t in ts])
    out["method"] = method
    out["steps_run"] = k
    return out


def check_t1(make, scene_name: str, cfg_label: str) -> dict:
    cfg = SolverConfig3D() if cfg_label == "default" else runtime_config()
    g0 = gpu_snapshot()
    py = python_timing(scene_name, cfg)
    nat = backend_timing(make, scene_name, cfg)
    g1 = gpu_snapshot()
    ok = "error" not in nat and nat["total"]["n"] >= T1_N and nat["total"]["median"] <= T1_BUDGET_MS
    summ = (f"T1 {scene_name} {cfg_label}: backend total median {nat.get('total', {}).get('median', float('nan')):.3f} ms "
            f"(p95 {nat.get('total', {}).get('p95', float('nan')):.3f}, n {nat.get('total', {}).get('n', 0)}) "
            f"<= {T1_BUDGET_MS} ms; Python step_profiled total median {py['total']['median']:.3f} ms, graph step "
            f"{py['step_graph']['median']:.3f} ms; GPU util {g0.get('util_pct')}/{g1.get('util_pct')} %, "
            f"mem {g0.get('mem_used_mib')}/{g1.get('mem_used_mib')} MiB (before/after)")
    return {"pass": bool(ok), "summary": summ, "backend": nat, "python": py, "gpu_before": g0, "gpu_after": g1}
