"""Guidance-strength sweep (evidence for GuideConfig3D defaults; see README "Guidance").

    .venv/Scripts/python -m vfx.datagen3d.tune_guide --out DIR [--steps 150] [--configs base,vel_only,...]

Per scene x config: tracking error at 16^3 (density, T-amb, velocity) every step (max and mean over the
whole run), 256^3 density power per radial band vs the UNGUIDED 256^3 run (geo-mean ratio over the last
`--spec-steps` steps), stability (finite, max |u|), plus PNGs: mid-z density slices and renders
(LR | guided HR | unguided HR) with one camera.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from vfx.core.fields3d import CH3, Camera, Volume, rasterize_obstacles_3d

from . import sim
from .guide3d import GuideConfig3D
from .scenarios import sample_scenario

CONFIGS = {
    "unguided": None,
    "base": GuideConfig3D(cutoff=12.0),                               # k_vel 1, k_s 0.7, cutoff 12
    "vel_only": GuideConfig3D(k_scalar=0.0, cutoff=12.0),
    "weak": GuideConfig3D(k_vel=0.5, k_scalar=0.35, cutoff=12.0),
    "c8": GuideConfig3D(cutoff=8.0),
    "c16": GuideConfig3D(cutoff=16.0),
    "strong_s": GuideConfig3D(k_scalar=1.0, cutoff=12.0),
    "fuel": GuideConfig3D(k_fuel=0.5),                                # cutoff 16 (default)
}
BANDS = ((4, 8), (8, 16), (16, 32), (32, 64), (64, 128))
SCENES = [("train", 3), ("train", 0), ("train", 1), ("train", 2), ("train", 5), ("ood", 200001)]


def radial_power(x: torch.Tensor) -> torch.Tensor:
    """(N,N,N) -> power summed per band in BANDS (cycles/domain)."""
    n = x.shape[-1]
    P = torch.fft.rfftn(x - x.mean()).abs() ** 2
    f = torch.fft.fftfreq(n, device=x.device) * n
    fr = torch.fft.rfftfreq(n, device=x.device) * n
    k = torch.sqrt(f[:, None, None] ** 2 + f[None, :, None] ** 2 + fr[None, None, :] ** 2)
    return torch.stack([P[(k >= a) & (k < b)].sum() for a, b in BANDS])


def default_camera() -> Camera:
    return Camera(position=(0.5 + 2.0 * math.sin(0.5), 0.75, 0.5 - 2.0 * math.cos(0.5)), target=(0.5, 0.42, 0.5))


def slice_img(d: torch.Tensor, vmax: float) -> np.ndarray:
    z = d.shape[0] // 2
    s = (d[z].float() / vmax).clamp(0, 1).sqrt().flip(0)   # y up -> row 0 top
    return (s.cpu().numpy() * 255).astype(np.uint8)


def save_png(path, arr):
    from .png import write_png
    write_png(path, arr)


def run_scene(split, seed, steps, spec_steps, configs, out: Path, renderer, sampler="v1", hr_res=256):
    sc = sample_scenario(seed, split, sampler=sampler)
    res = {}
    renders, slices = {}, {}
    solid_r = rasterize_obstacles_3d(sc.obstacles, 128) if sc.obstacles else None
    cam = default_camera()
    for name in configs:
        cfg = CONFIGS[name]
        tr, spec, spec_t = [], [], []
        t0 = time.perf_counter()
        vmax = None
        for s, lr, hr, t in sim.run_pair(sc, cfg, steps=steps, hr_res=hr_res):
            tr.append(t)
            if s >= steps - spec_steps:
                spec.append(radial_power(hr.data[CH3["density"]]))
                spec_t.append(radial_power(hr.data[CH3["temperature"]]))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tr = torch.stack(tr).cpu().numpy()
        spec = torch.stack(spec).mean(0).cpu().numpy()
        spec_t = torch.stack(spec_t).mean(0).cpu().numpy()
        umax = float(hr.velocity.abs().max())
        finite = bool(torch.isfinite(hr.data).all())
        res[name] = {"track_max": tr[10:].max(0).round(4).tolist(), "track_mean": tr[10:].mean(0).round(4).tolist(),
                     "track_last": tr[-1].round(4).tolist(), "spec": spec.tolist(), "spec_t": spec_t.tolist(), "umax": round(umax, 3),
                     "finite": finite, "sec": round(dt, 1), "ms_per_step": round(1000 * dt / steps, 1)}
        dmax = float(lr.data[CH3["density"]].max()) + 1e-6
        slices["lr"] = slice_img(torch.nn.functional.interpolate(lr.data[None, :1], scale_factor=hr_res // lr.data.shape[-1])[0, 0], dmax)
        slices[name] = slice_img(hr.data[CH3["density"]], dmax)
        renders["lr"] = renderer.draw(Volume(lr.data[0], lr.data[4]), cam, solid_r).cpu().numpy()
        renders[name] = renderer.draw(Volume(hr.data[0], hr.data[4]), cam, solid_r).cpu().numpy()
    base = np.asarray(res["unguided"]["spec"]) if "unguided" in res else None
    base_t = np.asarray(res["unguided"]["spec_t"]) if "unguided" in res else None
    for name in res:
        if base is not None:
            res[name]["spec_ratio"] = (np.asarray(res[name]["spec"]) / (base + 1e-12)).round(3).tolist()
            res[name]["spec_ratio_T"] = (np.asarray(res[name]["spec_t"]) / (base_t + 1e-12)).round(3).tolist()
        del res[name]["spec"], res[name]["spec_t"]
    tag = f"{split}_{seed}_{sc.kind}_hr{hr_res}"
    names = ["lr"] + list(configs)
    save_png(out / f"slices_{tag}.png", np.concatenate([slices[n] for n in names], 1))
    save_png(out / f"render_{tag}.png", np.concatenate([renders[n] for n in names], 1))
    return tag, sc.tags, res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--spec-steps", type=int, default=30)
    ap.add_argument("--configs", default="unguided,base,vel_only,weak,c8,c16,strong_s")
    ap.add_argument("--scenes", default=None, help="split:seed,...")
    ap.add_argument("--sampler", default="v1", help="scenario distribution: v1 | v2")
    ap.add_argument("--hr-res", type=int, default=256, help="guided HR resolution (multiple of 64)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from vfx.render3d import VolumeRenderer
    r = VolumeRenderer((360, 480), march_scale=1.0)
    configs = a.configs.split(",")
    scenes = SCENES if a.scenes is None else [(x.split(":")[0], int(x.split(":")[1])) for x in a.scenes.split(",")]
    allres = {}
    for split, seed in scenes:
        tag, tags, res = run_scene(split, seed, a.steps, a.spec_steps, configs, out, r, a.sampler, a.hr_res)
        allres[tag] = {"tags": tags, **res}
        print(tag, tags)
        for n, v in res.items():
            print(f"  {n:9s} track max d/T/u {v['track_max']} mean {v['track_mean']} "
                  f"spec d {v.get('spec_ratio')} T {v.get('spec_ratio_T')} umax {v['umax']} {v['ms_per_step']} ms/step")
        (out / "results.json").write_text(json.dumps(allres, indent=1))


if __name__ == "__main__":
    main()
