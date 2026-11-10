"""3D feasibility benchmark (owner: 3D test/eval engineer). THE deliverable of the 3D phase.

    .venv/Scripts/python -m bench3d.run                    # measure, write bench3d/results/{latest.json,REPORT.md},
                                                           # compare to bench3d/baseline.json (exit 1 on >10% regression)
    .venv/Scripts/python -m bench3d.run --update-baseline  # measure and store as the new baseline
    .venv/Scripts/python -m bench3d.run --quick            # fewer iterations, one scene (smoke check of the harness)

Question answered: is the SPEC per-effect budget reachable on the RTX 5080?
    64^3 sim <= 2.0 ms, detail <= 2.5 ms, 1440p render <= 1.5 ms, total <= 6.0 ms, peak VRAM <= 4 GB.

Complete pipelines, each on two fixed scenes (campfire; smoke plume past a sphere), rendered at 2560x1440
with march_scale 0.5 and 1.0:
    raw64     64^3 sim -> render the 64^3 volume
    tri256    64^3 sim -> classical_upscale_3d (trilinear 256^3) -> render
    routeA    64^3 sim -> VolumeUpscaler (256^3, empty bricks skipped) -> render
    routeB    64^3 sim -> render the 64^3 volume -> ScreenDetail on the 1440p frame
    brute128  128^3 sim -> render
    brute256  256^3 sim -> render (the thing the neural path is meant to replace; its sim stage is the
              "direct 256^3 step time")
plus step-only lines for the solver's full-quality default config (sim64_default, sim256_default).

Method. Every (scene, pipeline) runs in its own subprocess (clean VRAM accounting; one crashing module
cannot poison the others). The scene is simulated 60 steps (2 s) to develop the plume, then frames are
run back to back: each frame = sim step + stages, with a torch.cuda.Event before the frame and after
every stage, so per-stage and whole-frame GPU times come from the same frames. mean / p50 / p95 over
`--iters` frames after `--warmup` untimed frames. `wall_ms_per_frame` is host wall-clock per frame
(includes Python launch overhead; if it exceeds the GPU frame time the pipeline is CPU-bound).
VRAM: torch peak allocated, and device-level used-memory growth since the worker's CUDA init (catches
Warp's own allocator; noisy if another process allocates meanwhile). The budget check uses the larger.
The 64^3 solver uses `vfx.sim3d.runtime_config()` when exported (game-budget setting) else the default.
Occupancy = fraction of 16^3-HR bricks (4^3 LR cells) and 32^3-HR bricks (8^3 LR) with density > 1e-3
or T > ambient + 1 K, measured on the 64^3 state.

Before timing, nvidia-smi is read (utilization over 3 samples, type-C compute processes). If another
compute process or > 10% utilization is seen the GPU is recorded as SHARED and a loud warning printed:
the lead re-runs on an idle GPU. On a shared GPU, regressions are reported but do not fail the exit code
unless --strict.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
BASELINE = HERE / "baseline.json"
REGRESSION_TOL = 0.10
BUDGET = {"sim": 2.0, "detail": 2.5, "render": 1.5, "frame": 6.0}
BUDGET_VRAM_MB = 4096
OUT_SIZE = (1440, 2560)                  # (h, w): 1440p
MARCH_SCALES = (0.5, 1.0)                # low first: Warp's pool keeps freed blocks, so 1.0 sees 0.5's leftovers
SCENES = ("campfire", "plume_sphere")
PIPELINES = ("raw64", "tri256", "routeA", "routeB", "brute128", "brute256", "sim64_default", "sim256_default")
NEURAL = ("routeA", "routeB")
WARM_STEPS = 60
RESULT_TAG = "BENCH3D_RESULT "


# ============================================================================ GPU sharing check
def gpu_check(samples: int = 3) -> dict:
    """nvidia-smi: utilization (several samples), memory, and type-C (compute) processes other than us."""
    info: dict = {"ok": False}
    try:
        utils, mems = [], []
        for i in range(samples):
            q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
            u, m, t = [float(x) for x in q.stdout.strip().splitlines()[0].split(",")]
            utils.append(u)
            mems.append(m)
            info["memory_total_mb"] = t
            if i + 1 < samples:
                time.sleep(0.5)
        full = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout
        procs = []
        for line in full.splitlines():
            m = re.match(r"\|\s+\d+\s+\S+\s+\S+\s+(\d+)\s+(C\+G|C|G)\s+(.+?)\s+(\S+)\s*\|", line)
            if m and m.group(2) == "C" and int(m.group(1)) != os.getpid():
                procs.append({"pid": int(m.group(1)), "name": m.group(3), "mem": m.group(4)})
        info.update(ok=True, utilization_pct=utils, memory_used_mb=mems, compute_processes=procs)
        reasons = []
        if procs:
            reasons.append(f"{len(procs)} other compute process(es): " + ", ".join(f"{p['pid']} {p['name']}" for p in procs))
        if statistics.fmean(utils) > 10:
            reasons.append(f"GPU utilization {statistics.fmean(utils):.0f}% before the benchmark")
        info["shared"] = bool(reasons)
        info["reasons"] = reasons
    except Exception as e:  # noqa: BLE001
        info.update(shared=None, reasons=[f"nvidia-smi failed: {e!r}"])
    return info


def loud_warning(lines: list[str]):
    bar = "!" * 78
    print()
    print(bar)
    print("!!  WARNING: GPU IS SHARED - timings are NOT representative")
    for l in lines:
        print("!!  " + l)
    print("!!  The lead re-runs bench3d on an idle GPU before any decision.")
    print(bar)
    print()


# ============================================================================ worker (one scene x pipeline)
def stats(ms: list[float]) -> dict:
    s = sorted(ms)
    return {"mean_ms": statistics.fmean(s), "p50_ms": s[len(s) // 2],
            "p95_ms": s[min(len(s) - 1, int(0.95 * len(s)))], "max_ms": s[-1], "n": len(s),
            "hitches_gt50ms": sum(x > 50.0 for x in s)}


def time_frames(stages, iters: int, warmup: int) -> dict:
    """stages: [(name, fn)], each fn() runs one stage; frame = all stages in order."""
    import torch

    for _ in range(warmup):
        for _, fn in stages:
            fn()
    torch.cuda.synchronize()
    evs = [[torch.cuda.Event(enable_timing=True) for _ in range(len(stages) + 1)] for _ in range(iters)]
    t0 = time.perf_counter()
    for e in evs:
        e[0].record()
        for i, (_, fn) in enumerate(stages):
            fn()
            e[i + 1].record()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1000 / iters
    out = {"stages": {n: stats([e[i].elapsed_time(e[i + 1]) for e in evs]) for i, (n, _) in enumerate(stages)},
           "frame": stats([e[0].elapsed_time(e[-1]) for e in evs]), "wall_ms_per_frame": wall}
    return out


def scene_def(name: str):
    from vfx.core.fields3d import Emitter3D, Forces3D, Obstacle3D

    if name == "campfire":
        return [], Forces3D(emitters=[Emitter3D(pos=(0.5, 0.1, 0.5), radius=0.08, density_rate=1.0, temperature=1300.0,
                                                fuel_rate=2.5, velocity=(0.0, 0.5, 0.0))])
    if name == "plume_sphere":
        return ([Obstacle3D("sphere", (0.5, 0.55, 0.5), (0.14, 0.14, 0.14))],
                Forces3D(emitters=[Emitter3D(pos=(0.5, 0.12, 0.5), radius=0.07, density_rate=1.5, temperature=450.0,
                                             velocity=(0.0, 1.0, 0.0))]))
    raise ValueError(name)


def synthetic_state(res: int):
    """Deterministic stand-in 64^3 plume when vfx.sim3d is missing (marked scene_source=synthetic)."""
    import torch
    import torch.nn.functional as F
    from vfx.core.fields3d import AMBIENT_TEMP, CH3, NUM_CHANNELS_3D, FieldMeta3D, FieldState3D

    g = torch.Generator(device="cuda").manual_seed(0)
    c = (torch.arange(res, device="cuda") + 0.5) / res
    zz, yy, xx = torch.meshgrid(c, c, c, indexing="ij")
    env = (1 - ((xx - 0.5) ** 2 + ((yy - 0.45) / 1.6) ** 2 + (zz - 0.5) ** 2) / 0.09).clamp(min=0) ** 2
    n = F.interpolate(torch.randn(1, 2, 8, 8, 8, device="cuda", generator=g), size=(res,) * 3, mode="trilinear")[0]
    data = torch.zeros(NUM_CHANNELS_3D, res, res, res, device="cuda")
    data[CH3["density"]] = env * (0.7 + 0.3 * n[0]).clamp(min=0)
    data[CH3["temperature"]] = AMBIENT_TEMP + 900 * env ** 2 * (0.5 + 0.5 * n[1]).clamp(0, 1)
    data[CH3["vel_y"]] = env
    return FieldState3D(data=data, solid=torch.zeros(res, res, res, device="cuda"), meta=FieldMeta3D.for_res(res))


def occupancy(state) -> dict:
    import torch.nn.functional as F
    from vfx.core.fields3d import AMBIENT_TEMP

    act = ((state.ch("density") > 1e-3) | (state.ch("temperature") > AMBIENT_TEMP + 1.0)).float()[None, None]
    return {"bricks16hr_4lr": F.max_pool3d(act, 4, 4).mean().item(),
            "bricks32hr_8lr": F.max_pool3d(act, 8, 8).mean().item(),
            "voxels": act.mean().item()}


def worker(scene: str, pipeline: str, iters: int, warmup: int, scales) -> dict:
    import torch

    torch.cuda.init()
    torch.empty(1, device="cuda")
    try:
        import warp as wp

        wp.config.quiet = True
        wp.init()
    except Exception:  # noqa: BLE001
        pass
    from vfx.core.fields3d import SIM_DT, Camera, Volume, rasterize_obstacles_3d

    def used_mb():
        f, t = torch.cuda.mem_get_info()
        return (t - f) / 2 ** 20

    used0 = used_mb()
    res_out: dict = {"scene": scene, "pipeline": pipeline, "configs": {}, "notes": []}
    obstacles, forces = scene_def(scene)
    sim_res = {"brute128": 128, "brute256": 256, "sim256_default": 256}.get(pipeline, 64)
    use_default_cfg = pipeline.endswith("_default")
    if sim_res == 256:
        iters = min(iters, 30)

    # ---------------------------------------------------------------- solver + warm-up scene
    solver = None
    try:
        import vfx.sim3d as sim

        cfg = None
        if not use_default_cfg and hasattr(sim, "runtime_config"):
            cfg = sim.runtime_config()
            res_out["sim_config"] = "runtime_config()"
        else:
            res_out["sim_config"] = "SolverConfig3D() default"
        res_out["solver_version"] = getattr(sim, "SOLVER3D_VERSION", None)
        solver = sim.Solver3D(sim_res, cfg)
        state = solver.reset(obstacles)
        for _ in range(WARM_STEPS):
            state = solver.step(state, forces, SIM_DT)
        torch.cuda.synchronize()
        res_out["scene_source"] = "solver"
        try:
            res_out["divergence_max_after_warmup"] = solver.divergence(state).max().item()
        except Exception as e:  # noqa: BLE001
            res_out["notes"].append(f"divergence() failed: {e!r}")
    except Exception as e:  # noqa: BLE001
        if pipeline in ("brute128", "brute256", "sim64_default", "sim256_default"):
            res_out["skipped"] = f"vfx.sim3d unavailable: {e!r}"
            return res_out
        res_out["notes"].append(f"vfx.sim3d unavailable ({e!r}); synthetic static 64^3 state, no sim stage")
        res_out["scene_source"] = "synthetic"
        state = synthetic_state(64)
    holder = {"st": state}
    if sim_res == 64:
        res_out["occupancy_start"] = occupancy(state)

    def sim_stage():
        holder["st"] = solver.step(holder["st"], forces, SIM_DT)

    stages_sim = [("sim", sim_stage)] if solver is not None else []

    if pipeline.endswith("_default"):
        t = time_frames(stages_sim, iters, warmup)
        t["vram"] = {"torch_peak_allocated_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
                     "device_used_delta_mb": used_mb() - used0}
        res_out["configs"]["step_only"] = t
        return res_out

    # ---------------------------------------------------------------- detail stage
    detail = None
    try:
        import vfx.runtime3d as rt
        if pipeline == "tri256":
            def detail():
                holder["vol"] = rt.classical_upscale_3d(holder["st"], factor=4, mode="trilinear")
        elif pipeline == "routeA":
            torch.manual_seed(0)
            vu = rt.VolumeUpscaler(None, device="cuda")
            res_out["volume_upscaler"] = {"untrained": getattr(vu, "untrained", None), "params": getattr(vu, "params", None),
                                          "sparse": getattr(vu, "sparse", None)}

            def detail():
                holder["vol"] = vu.infer(holder["st"])
        elif pipeline == "routeB":
            torch.manual_seed(0)
            sd = rt.ScreenDetail(None, device="cuda")
            res_out["screen_detail"] = {"untrained": getattr(sd, "untrained", None), "params": getattr(sd, "params", None)}

            def detail():
                holder["out"] = sd.infer(holder["img"], None)
    except Exception as e:  # noqa: BLE001
        if pipeline in ("tri256", "routeA", "routeB"):
            res_out["skipped"] = f"vfx.runtime3d unavailable for {pipeline}: {e!r}"
            return res_out

    hr_res = 256 if pipeline in ("tri256", "routeA") else sim_res
    solid = rasterize_obstacles_3d(obstacles, hr_res) if obstacles else None

    def raw_volume():
        holder["vol"] = Volume(density=holder["st"].ch("density"), temperature=holder["st"].ch("temperature"))

    try:
        import vfx.render3d as rd
    except Exception as e:  # noqa: BLE001
        res_out["skipped"] = f"vfx.render3d unavailable: {e!r}"
        return res_out
    cam = Camera()
    for ms in scales:
        key = f"ms{ms:g}"
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            renderer = rd.VolumeRenderer(OUT_SIZE, device="cuda", march_scale=ms)

            def render():
                holder["img"] = renderer.draw(holder["vol"], cam, solid)

            if pipeline in ("raw64", "routeB", "brute128", "brute256"):
                def render_raw():
                    raw_volume()
                    render()
                stages = stages_sim + [("render", render_raw)]
                if pipeline == "routeB":
                    stages.append(("detail", detail))
            else:
                stages = stages_sim + [("detail", detail), ("render", render)]
            t = time_frames(stages, iters, warmup)
            t["vram"] = {"torch_peak_allocated_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
                         "device_used_delta_mb": used_mb() - used0}
            t["renderer_timings"] = dict(getattr(renderer, "timings", {}) or {})
            if pipeline == "routeA":
                t["upscaler_timings"] = dict(getattr(vu, "timings", {}) or {})
            if pipeline == "routeB":
                t["screen_detail_timings"] = dict(getattr(sd, "timings", {}) or {})
            res_out["configs"][key] = t
            del renderer
        except Exception as e:  # noqa: BLE001
            import traceback
            res_out["configs"][key] = {"error": repr(e), "trace": traceback.format_exc()[-1500:]}
    if sim_res == 64:
        res_out["occupancy_end"] = occupancy(holder["st"])
    return res_out


def run_worker_subprocess(scene, pipeline, iters, warmup, scales, timeout) -> dict:
    cmd = [sys.executable, "-m", "bench3d.run", "--worker", scene, pipeline, "--iters", str(iters),
           "--warmup", str(warmup), "--scales", ",".join(f"{s:g}" for s in scales)]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"scene": scene, "pipeline": pipeline, "skipped": f"worker timed out after {timeout} s"}
    for line in p.stdout.splitlines():
        if line.startswith(RESULT_TAG):
            r = json.loads(line[len(RESULT_TAG):])
            r["worker_seconds"] = time.perf_counter() - t0
            return r
    tail = (p.stderr or p.stdout)[-2000:]
    return {"scene": scene, "pipeline": pipeline, "skipped": f"worker failed (exit {p.returncode}): {tail}"}


# ============================================================================ analysis + report
def vram_of(t: dict) -> float:
    v = t.get("vram", {})
    return max(v.get("torch_peak_allocated_mb", 0.0), v.get("device_used_delta_mb", 0.0))


def evaluate(res: dict) -> list[dict]:
    """One row per (scene, pipeline, march_scale) with pass/fail per budget line and the speed ratio."""
    rows = []
    by = {(r["scene"], r["pipeline"]): r for r in res["runs"]}
    for (scene, pipe), r in by.items():
        for key, t in r.get("configs", {}).items():
            if "error" in t or key == "step_only":
                continue
            st = t["stages"]
            row = {"scene": scene, "pipeline": pipe, "march_scale": key[2:],
                   "sim": st.get("sim", {}).get("mean_ms"), "detail": st.get("detail", {}).get("mean_ms", 0.0),
                   "render": st.get("render", {}).get("mean_ms"), "frame": t["frame"]["mean_ms"],
                   "frame_p50": t["frame"]["p50_ms"], "frame_p95": t["frame"]["p95_ms"],
                   "wall": t["wall_ms_per_frame"], "vram": vram_of(t), "hitches": t["frame"].get("hitches_gt50ms", 0),
                   "frame_max": t["frame"]["max_ms"]}
            row["pass"] = {k: (row[k] is not None and row[k] <= BUDGET[k]) for k in BUDGET}
            row["pass"]["vram"] = row["vram"] <= BUDGET_VRAM_MB
            row["pass_all"] = all(row["pass"].values())
            b = by.get((scene, "brute256"), {}).get("configs", {}).get(key)
            if b and "frame" in b:
                row["speedup_vs_brute256"] = b["frame"]["mean_ms"] / row["frame"]
            bd = by.get((scene, "sim256_default"), {}).get("configs", {}).get("step_only")
            if bd and b and "render" in b.get("stages", {}):
                row["speedup_vs_brute256_default"] = (bd["stages"]["sim"]["mean_ms"] + b["stages"]["render"]["mean_ms"]) / row["frame"]
            rows.append(row)
    return rows


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def write_report(res: dict, rows: list[dict], path: Path):
    g = res["gpu"]
    L = ["# bench3d — 3D feasibility report", "",
         f"Created {res['meta']['created']} on {res['meta'].get('gpu', '?')}, torch {res['meta'].get('torch', '?')}; "
         f"{res['meta']['iters']} timed frames (256^3: up to 30) after {res['meta']['warmup']} warm-up frames, scenes "
         f"pre-simulated {WARM_STEPS} steps. Output 2560x1440. GPU times from torch.cuda.Event.", ""]
    if g.get("shared"):
        L += ["> **WARNING: GPU WAS SHARED during this run — numbers are not representative.**", ">"]
        L += [f"> - {x}" for x in g.get("reasons", [])]
        L += ["> The lead re-runs `python -m bench3d.run` on an idle GPU before any decision.", ""]
    elif g.get("shared") is None:
        L += [f"> GPU sharing unknown: {g.get('reasons')}", ""]
    else:
        L += ["GPU idle at start (no other compute process, utilization <= 10%).", ""]
    if g.get("after"):
        L += [f"nvidia-smi after run: utilization {g['after'].get('utilization_pct')}%, "
              f"other compute processes {len(g['after'].get('compute_processes', []))}.", ""]
    L += ["## Budget (SPEC)", "", "| stage | budget |", "|---|---|",
          "| sim (64^3 step) | 2.0 ms |", "| detail (neural / upscale) | 2.5 ms |", "| render (1440p) | 1.5 ms |",
          "| frame total | 6.0 ms |", "| peak VRAM | 4096 MB |", "",
          "## Pipelines x stages (mean ms; ✔ within budget, ✘ over)", "",
          "| scene | pipeline | march | sim | detail | render | frame (p50 / p95) | frames > 50 ms (max) | host wall/frame | VRAM MB | all | x vs brute256 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def ck(row, k):
        return "✔" if row["pass"][k] else "✘"

    for r in sorted(rows, key=lambda r: (r["scene"], PIPELINES.index(r["pipeline"]), r["march_scale"])):
        sp = r.get("speedup_vs_brute256")
        L.append(f"| {r['scene']} | {r['pipeline']} | {r['march_scale']} | {fmt(r['sim'])} {ck(r, 'sim')} | "
                 f"{fmt(r['detail'])} {ck(r, 'detail')} | {fmt(r['render'])} {ck(r, 'render')} | "
                 f"{fmt(r['frame'])} ({fmt(r['frame_p50'])} / {fmt(r['frame_p95'])}) {ck(r, 'frame')} | "
                 f"{r['hitches']} ({r['frame_max']:.0f}) | {fmt(r['wall'])} | "
                 f"{r['vram']:.0f} {ck(r, 'vram')} | {'PASS' if r['pass_all'] else 'FAIL'} | "
                 f"{'-' if sp is None else f'{sp:.1f}x'} |")
    L += ["", "Pass/fail is judged on the mean (the spec budget is per frame, so hitches count). "
          "`frames > 50 ms` counts hitches (e.g. graph capture / autotuning when the active-brick count changes).",
          "", "`x vs brute256` = brute256 frame time (256^3 step + render, same solver config and march_scale) / "
          "this pipeline's frame time. `host wall/frame` > frame means the pipeline is launch/CPU-bound.", ""]
    L += ["## Step-only and scene info", "", "| scene | line | config | step mean / p50 / p95 ms | divergence after warm-up |",
          "|---|---|---|---|---|"]
    for r in res["runs"]:
        for key, t in r.get("configs", {}).items():
            if "error" in t or "sim" not in t.get("stages", {}):
                continue
            if key not in ("step_only", "ms1"):
                continue
            s = t["stages"]["sim"]
            L.append(f"| {r['scene']} | {r['pipeline']} ({key}) | {r.get('sim_config')} | "
                     f"{s['mean_ms']:.2f} / {s['p50_ms']:.2f} / {s['p95_ms']:.2f} | {fmt(r.get('divergence_max_after_warmup'), 6) if r.get('divergence_max_after_warmup') is not None else '-'} |")
    L += ["", "Occupancy of the 64^3 state (fraction of active bricks: density > 1e-3 or T > ambient + 1 K):", ""]
    for r in res["runs"]:
        if r["pipeline"] == "raw64" and "occupancy_start" in r:
            a, b = r["occupancy_start"], r.get("occupancy_end", {})
            L.append(f"- **{r['scene']}**: 16^3-HR bricks {a['bricks16hr_4lr'] * 100:.1f}% after warm-up -> "
                     f"{b.get('bricks16hr_4lr', float('nan')) * 100:.1f}% at end; 32^3-HR bricks "
                     f"{a['bricks32hr_8lr'] * 100:.1f}% -> {b.get('bricks32hr_8lr', float('nan')) * 100:.1f}%; "
                     f"active voxels {a['voxels'] * 100:.1f}%")
    for r in res["runs"]:
        if r["pipeline"] == "routeA":
            for key, t in r.get("configs", {}).items():
                if "upscaler_timings" in t:
                    L.append(f"- {r['scene']} routeA {key}: VolumeUpscaler.timings {json.dumps({k: round(v, 3) for k, v in t['upscaler_timings'].items()})}")
    sk = [(r["scene"], r["pipeline"], r["skipped"]) for r in res["runs"] if r.get("skipped")]
    er = [(r["scene"], r["pipeline"], k, t["error"]) for r in res["runs"] for k, t in r.get("configs", {}).items() if "error" in t]
    notes = [(r["scene"], r["pipeline"], n) for r in res["runs"] for n in r.get("notes", [])]
    if sk or er or notes:
        L += ["", "## Skipped / errors / notes", ""]
        L += [f"- {s} {p}: SKIPPED {why[:400]}" for s, p, why in sk]
        L += [f"- {s} {p} {k}: ERROR {e[:400]}" for s, p, k, e in er]
        L += [f"- {s} {p}: {n}" for s, p, n in notes]
    L += ["", "## Caveats", "",
          "- Untrained network weights in this phase: route A/B costs are for the default architectures, not a "
          "quality claim.",
          "- VRAM = max(torch peak allocated, device used-memory growth since the worker's CUDA init); the second "
          "includes Warp's pool and is noisy if another process allocates. Per-pipeline subprocesses keep pipelines apart.",
          "- Occupancy grows during the timed frames (continuous emitters), so later frames process more bricks.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def compare(res: dict, base: dict) -> list[str]:
    regs = []

    def flat(r):
        out = {}
        for run in r.get("runs", []):
            for key, t in run.get("configs", {}).items():
                if "error" in t:
                    continue
                for st, s in t.get("stages", {}).items():
                    out[f"{run['scene']}/{run['pipeline']}/{key}/{st}"] = s["p50_ms"]
                out[f"{run['scene']}/{run['pipeline']}/{key}/frame"] = t["frame"]["p50_ms"]
                out[f"{run['scene']}/{run['pipeline']}/{key}/vram_mb"] = vram_of(t)
        return out

    a, b = flat(res), flat(base)
    for k in sorted(a):
        if k not in b or b[k] <= 0:
            continue
        if k.endswith("vram_mb") and b[k] < 64:
            continue
        ratio = a[k] / b[k]
        if ratio > 1 + REGRESSION_TOL:
            regs.append(f"{k}: {a[k]:.3f} vs baseline {b[k]:.3f} (+{(ratio - 1) * 100:.1f}%)")
    return regs


def print_table(rows):
    print(f"\n{'scene':13s} {'pipeline':9s} {'ms':>4s} {'sim':>7s} {'detail':>7s} {'render':>7s} {'frame':>7s} "
          f"{'p95':>7s} {'wall':>7s} {'VRAM':>6s}  result  x/brute256")
    for r in sorted(rows, key=lambda r: (r["scene"], PIPELINES.index(r["pipeline"]), r["march_scale"])):
        fails = [k for k, v in r["pass"].items() if not v]
        sp = r.get("speedup_vs_brute256")
        print(f"{r['scene']:13s} {r['pipeline']:9s} {r['march_scale']:>4s} {fmt(r['sim']):>7s} {fmt(r['detail']):>7s} "
              f"{fmt(r['render']):>7s} {fmt(r['frame']):>7s} {fmt(r['frame_p95']):>7s} {fmt(r['wall']):>7s} "
              f"{r['vram']:6.0f}  {'PASS' if not fails else 'FAIL ' + ','.join(fails)}  {'-' if sp is None else f'{sp:.1f}x'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--pipelines", default=",".join(PIPELINES))
    ap.add_argument("--scales", default=",".join(f"{s:g}" for s in MARCH_SCALES))
    ap.add_argument("--quick", action="store_true", help="20 frames, campfire only")
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--strict", action="store_true", help="fail on regressions even when the GPU is shared")
    ap.add_argument("--timeout", type=int, default=1200, help="seconds per worker")
    ap.add_argument("--worker", nargs=2, metavar=("SCENE", "PIPELINE"), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    scales = [float(s) for s in args.scales.split(",")]

    if args.worker:
        r = worker(args.worker[0], args.worker[1], args.iters, args.warmup, scales)
        print(RESULT_TAG + json.dumps(r))
        return 0

    if args.quick:
        args.iters, args.warmup, args.scenes = 20, 5, "campfire"
    gpu = gpu_check()                                  # before this process touches CUDA
    print("nvidia-smi:", json.dumps({k: gpu.get(k) for k in ("utilization_pct", "memory_used_mb", "shared")}))
    if gpu.get("shared"):
        loud_warning(gpu["reasons"])

    import torch

    if not torch.cuda.is_available():
        print("CUDA not available")
        return 2
    res = {"gpu": gpu, "runs": [], "budget_ms": BUDGET, "budget_vram_mb": BUDGET_VRAM_MB,
           "meta": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "python": platform.python_version(),
                    "iters": args.iters, "warmup": args.warmup, "out_size_hw": OUT_SIZE, "march_scales": scales,
                    "created": time.strftime("%Y-%m-%d %H:%M:%S")}}
    for scene in args.scenes.split(","):
        for pipe in args.pipelines.split(","):
            print(f"[{scene} / {pipe}] ...", flush=True)
            r = run_worker_subprocess(scene, pipe, args.iters, args.warmup, scales, args.timeout)
            if r.get("skipped"):
                print(f"   SKIPPED: {r['skipped'][:300]}")
            for k, t in r.get("configs", {}).items():
                if "error" in t:
                    print(f"   {k}: ERROR {t['error'][:300]}")
                else:
                    print(f"   {k}: frame mean {t['frame']['mean_ms']:.3f} ms  " +
                          "  ".join(f"{n} {s['mean_ms']:.3f}" for n, s in t["stages"].items()))
            res["runs"].append(r)
    gpu_after = gpu_check(samples=1)
    res["gpu"]["after"] = {k: gpu_after.get(k) for k in ("utilization_pct", "compute_processes", "shared")}
    rows = evaluate(res)
    res["rows"] = rows
    print_table(rows)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "latest.json").write_text(json.dumps(res, indent=1))
    write_report(res, rows, RESULTS / "REPORT.md")
    print(f"wrote {RESULTS / 'latest.json'} and {RESULTS / 'REPORT.md'}")
    if gpu.get("shared"):
        loud_warning(gpu["reasons"])
    if args.update_baseline:
        BASELINE.write_text(json.dumps(res, indent=1))
        print(f"baseline updated: {BASELINE}" + ("  (WARNING: recorded on a SHARED GPU)" if gpu.get("shared") else ""))
        return 0
    if args.no_compare or not BASELINE.exists():
        if not BASELINE.exists():
            print("no bench3d/baseline.json yet; run with --update-baseline to create one")
        return 0
    base = json.loads(BASELINE.read_text())
    regs = compare(res, base)
    if not regs:
        print("PASS: no regression > 10% vs baseline")
        return 0
    print("REGRESSIONS > 10% vs baseline:\n  " + "\n  ".join(regs))
    if (gpu.get("shared") or base.get("gpu", {}).get("shared")) and not args.strict:
        print("not failing: this run or the baseline was on a SHARED GPU (use --strict to fail anyway)")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
