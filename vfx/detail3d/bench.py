"""Cost of the advected-detail stage (GPU, CUDA events):

    python -m vfx.detail3d.bench [--preset 1] [--size 2560x1440] [--steps-warm 150] [--out results.json]

* step: AdvectedDetail.step per sim tick (amplitude + advection of both coordinate sets + texture copy).
* render: VolumeRenderer full (new volume: prep incl. detailed light volume + march) and view (camera-only
  redraw) with detail=None and with detail at each quality, march_scale 0.5 and 0.25; added = detail - none.
Trials are interleaved and the min over repeats of batches is reported (robust on a shared GPU).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch


def _time(fn, n=20, rep=7):
    best = []
    for _ in range(rep):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(n):
            fn()
        b.record()
        b.synchronize()
        best.append(a.elapsed_time(b) / n)
    return float(np.min(best)), float(np.median(best))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", type=int, default=1)
    ap.add_argument("--size", default="2560x1440")
    ap.add_argument("--steps-warm", type=int, default=150)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from vfx.app3d.compare import orbit_cam
    from vfx.app3d.presets import PRESETS
    from vfx.core.fields3d import CH3, SIM_DT, Forces3D, Volume, rasterize_obstacles_3d
    from vfx.detail3d import QUALITY, AdvectedDetail
    from vfx.render3d import VolumeRenderer
    from vfx.render3d.bench import gpu_note
    from vfx.sim3d import Solver3D, runtime_config
    W, H = (int(v) for v in a.size.lower().split("x"))
    pre = PRESETS[a.preset]
    s = Solver3D(64, runtime_config(**pre.sim))
    st = s.reset(pre.obstacles)
    dets = {q: AdvectedDetail(64, quality=q) for q in QUALITY}
    f = Forces3D(emitters=list(pre.emitters), wind=pre.wind)
    for _ in range(a.steps_warm):
        st = s.step(st, f, SIM_DT)
        for d in dets.values():
            d.step(st, SIM_DT)
    torch.cuda.synchronize()
    print("GPU before:", gpu_note())
    res = {"gpu_before": gpu_note(), "size": [W, H], "preset": a.preset}
    res["step_ms"] = {q: _time(lambda d=d: d.step(st, SIM_DT), 20, 9) for q, d in dets.items()}
    res["sim_step_ms"] = _time(lambda: s.step(st, f, SIM_DT), 10, 5)
    solid = rasterize_obstacles_3d(pre.obstacles, 128) if pre.obstacles else None
    vol = Volume(st.data[CH3["density"]], st.data[CH3["temperature"]])
    cam = orbit_cam(-30.0, 1.9)
    rows = []
    for ms in (0.5, 0.25):
        r = VolumeRenderer((H, W), march_scale=ms)
        variants = [("none", None)] + list(dets.items())
        for _, d in variants:                       # warm up / capture graphs
            r.draw(vol, cam, solid, detail=d)
            r.draw(vol, cam, solid, reuse_volume=True, detail=d)
        out = {k: {"full": [], "view": []} for k, _ in variants}
        for _ in range(3):                          # interleave
            for k, d in variants:
                out[k]["full"].append(_time(lambda d=d: r.draw(vol, cam, solid, detail=d), 10, 5)[0])
                r.draw(vol, cam, solid, detail=d)
                out[k]["view"].append(_time(lambda d=d: r.draw(vol, cam, solid, reuse_volume=True, detail=d), 10, 5)[0])
        for k, _ in variants:
            fm, vm = min(out[k]["full"]), min(out[k]["view"])
            rows.append({"march_scale": ms, "detail": k, "full": fm, "view": vm, "avg60": 0.5 * (fm + vm)})
    base = {(r_["march_scale"]): r_ for r_ in rows if r_["detail"] == "none"}
    for r_ in rows:
        b = base[r_["march_scale"]]
        r_["added_full"], r_["added_view"] = r_["full"] - b["full"], r_["view"] - b["view"]
    res["render"] = rows
    res["gpu_after"] = gpu_note()
    print("GPU after:", res["gpu_after"])
    print("detail step ms (min, median):", {q: tuple(round(x, 3) for x in v) for q, v in res["step_ms"].items()})
    print("sim step ms:", tuple(round(x, 3) for x in res["sim_step_ms"]))
    print(f"| march | detail | full ms | view ms | 60 Hz avg | added full | added view |")
    for r_ in rows:
        print(f"| {r_['march_scale']} | {r_['detail']} | {r_['full']:.2f} | {r_['view']:.2f} | {r_['avg60']:.2f} | "
              f"{r_['added_full']:+.2f} | {r_['added_view']:+.2f} |")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
