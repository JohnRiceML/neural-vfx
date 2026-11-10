"""Cost of DetailParticles by particle count and out_res (CUDA events, synchronized):

    python -m vfx.particles3d.bench --out DIR [--preset 1] [--counts 250000,1000000,4000000] [--res 128,192] [--legacy]

Per config: warm the 64^3 sim + particles up for --warmup ticks (ring full), then time --ticks ticks per mode:
  graph    the default path: the whole tick as one CUDA-graph replay (total only)
  eager    the same kernels launched directly, CUDA events between kernels (per-kernel ms; includes the Python
           launch gaps, so its total minus the graph total is the launch overhead the graph removes)
  unfused  separate deposit kernel, to split advect / deposit
  --legacy also sg_every=1: the curl noise evaluated on the whole velocity grid every tick (the previous path)
Render = VolumeRenderer.draw of the particle volume at 2560x1440, march_scale 0.5 (full draw: pack + light volumes
+ march + composite). Medians. Writes table.md + results.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vfx.core.fields3d import SIM_DT, Forces3D


def _med(xs):
    return float(np.median(xs)) if xs else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", type=int, default=1)
    ap.add_argument("--counts", default="250000,1000000,4000000")
    ap.add_argument("--res", default="128,192")
    ap.add_argument("--warmup", type=int, default=110)
    ap.add_argument("--ticks", type=int, default=20)
    ap.add_argument("--param", nargs="*")
    ap.add_argument("--legacy", action="store_true", help="also time sg_every=1 (noise evaluated every tick)")
    a = ap.parse_args(argv)
    from vfx.app3d.compare import orbit_cam
    from vfx.app3d.presets import PRESETS
    from vfx.render3d import VolumeRenderer
    from vfx.render3d.bench import gpu_note
    from vfx.sim3d import Solver3D, runtime_config

    from .compare import _parse_params
    from .particles import DetailParticles
    params = _parse_params(a.param)
    pre = PRESETS[a.preset]
    cfg = runtime_config(**pre.sim)
    forces = Forces3D(emitters=list(pre.emitters), wind=tuple(pre.wind))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rend = VolumeRenderer((1440, 2560), march_scale=0.5)
    cam = orbit_cam(-30.0)
    rows = []
    gpu0 = gpu_note()
    for res in [int(r) for r in a.res.split(",")]:
        for n in [int(c) for c in a.counts.split(",")]:
            solver = Solver3D(64, cfg)
            st = solver.reset(pre.obstacles)
            torch.cuda.reset_peak_memory_stats()
            m0 = torch.cuda.memory_allocated()
            dp = DetailParticles(n, res, config=cfg, **params)
            mem = (torch.cuda.memory_allocated() - m0) / 2 ** 20
            for _ in range(a.warmup):
                st = solver.step(st, forces, SIM_DT)
                dp.step(st, forces, SIM_DT)
            res_t = {}
            base = {k: dp.p[k] for k in ("graph", "fused", "sg_every")}
            modes = [("graph", dict(graph=True, fused=True)), ("eager", dict(graph=False, fused=True)),
                     ("unfused", dict(graph=False, fused=False))]
            if a.legacy:      # the per-tick noise evaluation this version replaced (sg_every = 1)
                modes += [("graph_sg1", dict(graph=True, fused=True, sg_every=1)),
                          ("eager_sg1", dict(graph=False, fused=True, sg_every=1))]
            for name, over in modes:
                dp.p.update(base)
                dp.p.update(over)
                acc = {}
                for i in range(a.ticks + 2):
                    st = solver.step(st, forces, SIM_DT)
                    dp.step(st, forces, SIM_DT)
                    t = dp.timings                      # syncs on the tick's last event
                    if i >= 2:                          # skip the capture / keyframe re-init ticks
                        for k, v in t.items():
                            acc.setdefault(k, []).append(v)
                res_t[name] = {k: _med(v) for k, v in acc.items()}
            dp.p.update(base)
            kern = {k: v for k, v in res_t["eager"].items() if k.startswith("k:")}
            print(f"  per kernel (eager, ms): " + "  ".join(f"{k[2:]} {v:.3f}" for k, v in kern.items()), flush=True)
            if a.legacy:
                kl = {k: v for k, v in res_t["eager_sg1"].items() if k.startswith("k:")}
                print(f"  per kernel (eager, sg_every=1): " + "  ".join(f"{k[2:]} {v:.3f}" for k, v in kl.items()),
                      flush=True)
            rms = []
            for _ in range(8):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                rend.draw(dp.volume(), cam)
                e1.record()
                e1.synchronize()
                rms.append(e0.elapsed_time(e1))
            d = dp.diagnostics(st)
            row = dict(out_res=res, count=n, alive=d["alive"], total_graph=res_t["graph"]["total"],
                       total_eager=res_t["eager"]["total"], compose=res_t["eager"]["compose"],
                       advect=res_t["unfused"]["advect"], deposit=res_t["unfused"]["deposit"],
                       total_unfused=res_t["unfused"]["total"], render=_med(rms[2:]), vram_mib=mem,
                       kernels=kern, mass_ratio=d["particle_mass"] / max(d["grid_mass"], 1e-12),
                       restrict_err=d["restrict_rel_err"])
            if a.legacy:
                row.update(total_graph_sg1=res_t["graph_sg1"]["total"], total_eager_sg1=res_t["eager_sg1"]["total"],
                           kernels_sg1={k: v for k, v in res_t["eager_sg1"].items() if k.startswith("k:")})
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if not isinstance(v, dict)},
                  flush=True)
            del dp
            torch.cuda.empty_cache()
    lines = ["| out_res | particles | alive | advect | deposit | compose | total (direct launches) | **total (CUDA graph)** "
             + ("| graph, sg_every=1 (old noise path) " if a.legacy else "") + "| render 1440p | VRAM MiB |",
             "|---|---|---|---|---|---|---|---|---|---|" + ("---|" if a.legacy else "")]
    for r in rows:
        lines.append(f"| {r['out_res']}³ | {r['count'] / 1e6:g}M | {r['alive'] / 1e6:.2f}M | {r['advect']:.2f} | "
                     f"{r['deposit']:.2f} | {r['compose']:.2f} | {r['total_eager']:.2f} | **{r['total_graph']:.2f}** | "
                     + (f"{r['total_graph_sg1']:.2f} | " if a.legacy else "")
                     + f"{r['render']:.2f} | {r['vram_mib']:.0f} |")
    lines += ["", "Per kernel, direct launches (ms, median; includes Python launch gaps):", ""]
    for r in rows:
        lines.append(f"* {r['out_res']}³ {r['count'] / 1e6:g}M: " + ", ".join(f"{k[2:]} {v:.3f}" for k, v in r["kernels"].items()))
        if a.legacy:
            lines.append(f"  * sg_every=1: " + ", ".join(f"{k[2:]} {v:.3f}" for k, v in r["kernels_sg1"].items()))
    gpu1 = gpu_note()
    (out / "table.md").write_text("\n".join(lines) + f"\n\nGPU before: {gpu0}\nGPU after: {gpu1}\n")
    (out / "results.json").write_text(json.dumps(dict(rows=rows, gpu=[gpu0, gpu1], preset=a.preset, params=params), indent=2))
    print("\n".join(lines))
    print("GPU:", gpu0, "|", gpu1)


if __name__ == "__main__":
    main()
