"""Cost of DetailParticles by particle count and out_res (CUDA events, synchronized):

    python -m vfx.particles3d.bench --out DIR [--preset 1] [--counts 250000,1000000,4000000] [--res 128,192]

Per config: warm the 64^3 sim + particles up for --warmup ticks (ring full), then time --ticks ticks twice:
fused (the default path: advect+seed+deposit in one kernel, then restrict+compose) and unfused (separate
deposit kernel, to split advect / deposit). Render = VolumeRenderer.draw of the particle volume at 2560x1440,
march_scale 0.5 (full draw: pack + light volumes + march + composite). Medians. Writes table.md + results.json.
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
            for fused in (True, False):
                dp.p["fused"] = fused
                acc = {}
                for _ in range(a.ticks):
                    st = solver.step(st, forces, SIM_DT)
                    dp.step(st, forces, SIM_DT)
                    for k, v in dp.timings.items():
                        acc.setdefault(k, []).append(v)
                res_t[fused] = {k: _med(v) for k, v in acc.items()}
            dp.p["fused"] = True
            rms = []
            for _ in range(8):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                rend.draw(dp.volume(), cam)
                e1.record()
                e1.synchronize()
                rms.append(e0.elapsed_time(e1))
            d = dp.diagnostics(st)
            row = dict(out_res=res, count=n, alive=d["alive"], total_fused=res_t[True]["total"],
                       advect_fused=res_t[True]["advect+deposit"], compose=res_t[True]["compose"],
                       advect=res_t[False]["advect"], deposit=res_t[False]["deposit"],
                       total_unfused=res_t[False]["total"], render=_med(rms[2:]), vram_mib=mem,
                       mass_ratio=d["particle_mass"] / max(d["grid_mass"], 1e-12), restrict_err=d["restrict_rel_err"])
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            del dp
            torch.cuda.empty_cache()
    lines = ["| out_res | particles | alive | advect | deposit | compose | **total (fused)** | total (unfused) | render 1440p | VRAM MiB |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['out_res']}³ | {r['count'] / 1e6:g}M | {r['alive'] / 1e6:.2f}M | {r['advect']:.2f} | "
                     f"{r['deposit']:.2f} | {r['compose']:.2f} | **{r['total_fused']:.2f}** | {r['total_unfused']:.2f} | "
                     f"{r['render']:.2f} | {r['vram_mib']:.0f} |")
    gpu1 = gpu_note()
    (out / "table.md").write_text("\n".join(lines) + f"\n\nGPU before: {gpu0}\nGPU after: {gpu1}\n")
    (out / "results.json").write_text(json.dumps(dict(rows=rows, gpu=[gpu0, gpu1], preset=a.preset, params=params), indent=2))
    print("\n".join(lines))
    print("GPU:", gpu0, "|", gpu1)


if __name__ == "__main__":
    main()
