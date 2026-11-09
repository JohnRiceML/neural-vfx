"""Layer 5 performance harness (owner: test/eval engineer).

    .venv/Scripts/python -m bench.run                    # measure, write bench/results/latest.json,
                                                         # compare to bench/baseline.json (exit 1 on >10% regression)
    .venv/Scripts/python -m bench.run --update-baseline  # measure and store as the new baseline

Stages (fixed scenes, GPU time from torch.cuda.Event, per-iteration):
  trivial_torch / trivial_warp   M0 gate: harness reports timings for a trivial kernel
  solver_step_128                vfx.sim.Solver.step at 128^2, fire plume + obstacle, after warm-up
  upscaler_infer                 vfx.runtime.Upscaler.infer on live solver states (needs checkpoint)
  render_draw_1024               vfx.render.Renderer.draw at 1024^2
  frame_total                    step + infer (or bicubic) + draw, one CUDA-event span per frame
  peak VRAM                      torch.cuda.max_memory_allocated and device-level used memory
Stages whose module/checkpoint is missing are recorded as skipped (null) with a reason.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
BASELINE = HERE / "baseline.json"
REGRESSION_TOL = 0.10           # SPEC Layer 5: fail on > 10% regression
# SPEC performance budget (stated for the 3D target; reported for information in v0 2D)
BUDGET_MS = {"solver_step_128": 2.0, "upscaler_infer": 2.5, "render_draw_1024": 1.5, "frame_total": 6.0}
BUDGET_VRAM_MB = 4096


def time_cuda(fn, iters: int, warmup: int = 10, between=None) -> dict:
    """GPU time of fn() per call, each call bracketed by its own pair of CUDA events.
    `between` (untimed) runs before each timed call, e.g. to advance the simulation."""
    for _ in range(warmup):
        if between:
            between()
        fn()
    torch.cuda.synchronize()
    evs = []
    for _ in range(iters):
        if between:
            between()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        evs.append((s, e))
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in evs)
    return {"mean_ms": statistics.fmean(ms), "p50_ms": ms[len(ms) // 2],
            "p95_ms": ms[min(len(ms) - 1, int(0.95 * len(ms)))], "max_ms": ms[-1], "iters": iters}


def device_used_mb() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2 ** 20


class Bench:
    def __init__(self, iters: int):
        self.iters = iters
        self.res: dict = {"stages": {}, "skipped": {}}
        self.used0 = device_used_mb()
        self.used_peak = self.used0

    def sample_mem(self):
        self.used_peak = max(self.used_peak, device_used_mb())

    def record(self, name, t: dict | None, reason: str | None = None, **extra):
        if t is None:
            self.res["skipped"][name] = reason
            print(f"  {name:20s} SKIPPED: {reason}")
        else:
            t.update(extra)
            self.res["stages"][name] = t
            print(f"  {name:20s} mean {t['mean_ms']:8.3f} ms  p50 {t['p50_ms']:8.3f}  p95 {t['p95_ms']:8.3f}")
        self.sample_mem()


def trivial(b: Bench):
    x = torch.zeros(1 << 20, device="cuda")
    b.record("trivial_torch", time_cuda(lambda: x.add_(1.0), 200, 20))
    try:
        import warp as wp

        wp.init()

        @wp.kernel
        def _inc(a: wp.array(dtype=wp.float32)):
            i = wp.tid()
            a[i] = a[i] + 1.0

        wa = wp.from_torch(x)
        dev = wp.device_from_torch(x.device)
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
            b.record("trivial_warp", time_cuda(lambda: wp.launch(_inc, dim=x.numel(), inputs=[wa], device=dev), 200, 20))
    except Exception as e:  # noqa: BLE001 - report, don't crash the harness
        b.record("trivial_warp", None, f"warp unavailable: {e!r}")


def fixed_scene():
    from vfx.core.fields import Emitter, Forces, Obstacle

    obstacles = [Obstacle("circle", (0.55, 0.6), (0.08, 0.08))]
    forces = Forces(emitters=[Emitter(pos=(0.45, 0.12), radius=0.05, density_rate=1.0, temperature=1300.0,
                                      fuel_rate=2.5, velocity=(0.0, 0.5)),
                              Emitter(pos=(0.65, 0.15), radius=0.04, density_rate=1.5, temperature=330.0)])
    return obstacles, forces


def synthetic_state():
    """Deterministic stand-in LR state when vfx.sim is missing (marked in the results)."""
    import torch.nn.functional as F
    from vfx.core.fields import AMBIENT_TEMP, CH, LR_RES, NUM_CHANNELS, FieldMeta, FieldState

    g = torch.Generator(device="cuda").manual_seed(0)
    data = torch.zeros(NUM_CHANNELS, LR_RES, LR_RES, device="cuda")
    z = F.interpolate(torch.randn(1, 4, 16, 16, device="cuda", generator=g), size=(LR_RES, LR_RES), mode="bicubic")[0]
    data[CH["density"]] = z[0].relu()
    data[CH["vel_x"]], data[CH["vel_y"]] = 0.3 * z[1], 0.3 * z[2]
    data[CH["temperature"]] = AMBIENT_TEMP + 600 * z[3].relu()
    return FieldState(data=data, solid=torch.zeros(LR_RES, LR_RES, device="cuda"), meta=FieldMeta.for_res(LR_RES))


def run(iters: int) -> dict:
    from vfx.core.fields import LR_RES, SIM_DT, rasterize_obstacles, HR_RES
    from vfx.core.paths import default_checkpoint
    from vfx.eval.metrics import bicubic_upscale

    torch.cuda.reset_peak_memory_stats()
    b = Bench(iters)
    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    trivial(b)

    obstacles, forces = fixed_scene()
    # --- solver
    solver = state = None
    try:
        import vfx.sim as sim

        solver = sim.Solver(LR_RES)
        state = solver.reset(obstacles)
        for _ in range(60):                             # warm-up: developed plume
            state = solver.step(state, forces, SIM_DT)
        holder = {"s": state}

        def step():
            holder["s"] = solver.step(holder["s"], forces, SIM_DT)

        b.record("solver_step_128", time_cuda(step, iters, 10), solver_version=getattr(sim, "SOLVER_VERSION", None))
        state = holder["s"]
        scene_source = "solver"
    except ImportError as e:
        b.record("solver_step_128", None, f"vfx.sim not importable: {e}")
        state = synthetic_state()
        scene_source = "synthetic"
    except Exception as e:  # noqa: BLE001
        b.record("solver_step_128", None, f"solver error: {e!r}")
        solver = None
        state = synthetic_state()
        scene_source = "synthetic"
    b.res["scene_source"] = scene_source

    def advance():
        nonlocal state
        if solver is not None:
            state = solver.step(state, forces, SIM_DT)

    # --- upscaler
    upscaler = None
    ckpt = default_checkpoint()
    try:
        import vfx.runtime as rt

        if ckpt.exists():
            upscaler = rt.Upscaler(str(ckpt), device="cuda")
            upscaler.reset()
            holder = {}
            b.record("upscaler_infer", time_cuda(lambda: holder.__setitem__("o", upscaler.infer(state)), iters, 10,
                                                 between=advance), checkpoint=str(ckpt))
            b.res["upscaler_internal_timings"] = dict(getattr(upscaler, "timings", {}) or {})
        else:
            b.record("upscaler_infer", None, f"no checkpoint at {ckpt}")
    except ImportError as e:
        b.record("upscaler_infer", None, f"vfx.runtime not importable: {e}")
    except Exception as e:  # noqa: BLE001
        b.record("upscaler_infer", None, f"upscaler error: {e!r}")
        upscaler = None

    def upscale(s):
        return upscaler.infer(s) if upscaler is not None else bicubic_upscale(s)

    # --- renderer
    renderer = None
    try:
        import vfx.render as rd

        renderer = rd.Renderer((1024, 1024), device="cuda")
        hr = upscale(state)
        solid_hr = rasterize_obstacles(obstacles, HR_RES) if scene_source == "solver" else None
        b.record("render_draw_1024", time_cuda(lambda: renderer.draw(hr, solid_hr), iters, 10),
                 fields="neural" if upscaler is not None else "bicubic")
    except ImportError as e:
        b.record("render_draw_1024", None, f"vfx.render not importable: {e}")
    except Exception as e:  # noqa: BLE001
        b.record("render_draw_1024", None, f"renderer error: {e!r}")
        renderer = None

    # --- whole frame
    if solver is not None and renderer is not None:
        solid_hr = rasterize_obstacles(obstacles, HR_RES)

        def frame():
            advance()
            renderer.draw(upscale(state), solid_hr)

        b.record("frame_total", time_cuda(frame, iters, 10),
                 includes="step+" + ("neural" if upscaler is not None else "bicubic") + "+draw")
    else:
        b.record("frame_total", None, "needs solver and renderer")
    parts = [b.res["stages"].get(k, {}).get("mean_ms") for k in ("solver_step_128", "upscaler_infer", "render_draw_1024")]
    b.res["sum_of_stages_ms"] = sum(p for p in parts if p is not None) if any(p is not None for p in parts) else None

    torch.cuda.synchronize()
    b.sample_mem()
    b.res["vram"] = {"torch_peak_allocated_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
                     "torch_peak_reserved_mb": torch.cuda.max_memory_reserved() / 2 ** 20,
                     "device_used_peak_minus_start_mb": b.used_peak - b.used0,
                     "device_used_peak_mb": b.used_peak}
    b.res["meta"] = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "python": platform.python_version(),
                     "iters": iters, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    return b.res


def budget_report(res: dict):
    print("SPEC budget (3D targets, informational for v0 2D):")
    for k, lim in BUDGET_MS.items():
        t = res["stages"].get(k)
        if t:
            print(f"  {k:20s} {t['mean_ms']:7.3f} ms  budget {lim:.1f}  {'ok' if t['mean_ms'] <= lim else 'OVER'}")
    v = res["vram"]["torch_peak_allocated_mb"] + 0.0
    dv = res["vram"]["device_used_peak_minus_start_mb"]
    print(f"  VRAM torch peak {v:.0f} MB, device delta {dv:.0f} MB, budget {BUDGET_VRAM_MB} MB")


def compare(res: dict, base: dict) -> list[str]:
    """List of regressions > REGRESSION_TOL (median ms per stage - robust to clock/OS hiccups - and peak VRAM)."""
    regs = []
    if res.get("scene_source") != base.get("scene_source"):
        print(f"note: scene source {res.get('scene_source')} vs baseline {base.get('scene_source')}; "
              "stage timings not comparable - re-baseline once vfx.sim lands")
    for k, t in res["stages"].items():
        bt = base.get("stages", {}).get(k)
        if not bt:
            continue
        if res.get("scene_source") != base.get("scene_source"):
            continue
        if k.startswith("trivial_"):
            continue    # microsecond kernels: launch-overhead noise, reported but not gated
        ratio = t["p50_ms"] / max(bt["p50_ms"], 1e-9)
        flag = "REGRESSION" if ratio > 1 + REGRESSION_TOL else "ok"
        print(f"  {k:20s} p50 {t['p50_ms']:8.3f} ms vs baseline {bt['p50_ms']:8.3f}  ({(ratio - 1) * 100:+6.1f}%) {flag}")
        if ratio > 1 + REGRESSION_TOL:
            regs.append(f"{k}: p50 {t['p50_ms']:.3f} ms vs {bt['p50_ms']:.3f} ms (+{(ratio - 1) * 100:.1f}%)")
    for k in ("torch_peak_allocated_mb", "device_used_peak_minus_start_mb"):
        a, bb = res["vram"].get(k), base.get("vram", {}).get(k)
        if a is not None and bb and bb > 64 and a > bb * (1 + REGRESSION_TOL):
            regs.append(f"VRAM {k}: {a:.0f} MB vs {bb:.0f} MB")
    return regs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update-baseline", action="store_true", help="store this run as bench/baseline.json")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--no-compare", action="store_true")
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA not available: M0 gate fails (GPU not detected)")
        return 2
    res = run(args.iters)
    budget_report(res)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "latest.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {RESULTS / 'latest.json'}")
    if args.update_baseline:
        BASELINE.write_text(json.dumps(res, indent=1))
        print(f"baseline updated: {BASELINE}")
        return 0
    if args.no_compare:
        return 0
    if not BASELINE.exists():
        print("no bench/baseline.json yet; run with --update-baseline to create one")
        return 0
    regs = compare(res, json.loads(BASELINE.read_text()))
    if regs:
        print("FAIL: regressions > 10%:\n  " + "\n  ".join(regs))
        return 1
    print("PASS: no regression > 10% vs baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
