"""run(scenario, out_dir): simulate LR + guided HR, render K camera sequences, write one scenario dir.

Files are written via temp + rename and meta.json last, so meta.json marks a complete directory.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from vfx.core.fields3d import HR_RES_3D, LR_RES_3D, SIM_DT

from . import render as R
from . import sim
from .guide3d import DEFAULT_GUIDE, GuideConfig3D

FILES = ("lr.npy", "hr.npy", "aux.npy", "hr_alpha.npy")
DEFAULT_CAMS = 2
_renderer = None


def get_renderer() -> R.PairRenderer:
    global _renderer
    if _renderer is None:
        _renderer = R.PairRenderer()
    return _renderer


def scenario_dir(root, split: str, seed: int) -> Path:
    return Path(root) / split / f"scn_{seed:06d}"


def is_complete(d: Path) -> bool:
    return (d / "meta.json").exists() and all((d / f).exists() for f in FILES)


def _save(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def run(sc, out_dir, n_cams: int = DEFAULT_CAMS, guide: GuideConfig3D | None = DEFAULT_GUIDE,
        hr_res: int = HR_RES_3D) -> dict:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").unlink(missing_ok=True)
    t0 = time.perf_counter()
    pr = get_renderer()
    rng = np.random.default_rng([sc.seed, ("train", "test", "ood").index(sc.split), 0xCA4])
    T = sc.frames
    tracks = []
    cams = crops = None
    centroid = None
    t_render = 0.0
    for s, lr, hr, tr in sim.run_pair(sc, guide, hr_res=hr_res):
        tracks.append(tr)
        if s == sc.warmup - 1:
            centroid = R.effect_centroid(lr)
            cams = R.sample_cameras(rng, centroid, n_cams)
            crops = [R.choose_crop(rng, c, T, SIM_DT) for c in cams]
            pr.begin(sc.obstacles, cams, crops, T)
        if s >= sc.warmup:
            tr0 = time.perf_counter()
            pr.frame(s - sc.warmup, lr, hr, SIM_DT)
            t_render += time.perf_counter() - tr0
    tracks = torch.stack(tracks).cpu().numpy()              # (steps, 3) density, T-amb, velocity
    lr_np = pr.lr.cpu().numpy()
    hr_np = pr.hr.cpu().numpy()
    aux_np = pr.aux.cpu().numpy()
    ha_np = pr.hr_alpha.cpu().numpy()
    finite = bool(np.isfinite(aux_np).all())
    t_sim = time.perf_counter() - t0
    _save(d / "lr.npy", lr_np)
    _save(d / "hr.npy", hr_np)
    _save(d / "aux.npy", aux_np)
    _save(d / "hr_alpha.npy", ha_np)

    al = aux_np[:, :, 0].astype(np.float32)
    mot = np.linalg.norm(aux_np[:, :, 2:4].astype(np.float32), axis=2)
    cam_meta = []
    for k, c in enumerate(cams):
        cam_meta.append({
            "orbit": {kk: (list(v) if isinstance(v, tuple) else v) for kk, v in c.__dict__.items()},
            "moving": c.moving,
            "crop_xy_out": crops[k],                       # top-left of the CROP_OUT window, output px
            "frames": [{"position": list(c.at(i * SIM_DT).position), "target": list(c.at(i * SIM_DT).target),
                        "fov_y_deg": c.fov} for i in range(-1, T)],   # index 0 = camera one step before frame 0
            "domain_bbox_out": [R.domain_bbox(c.at(i * SIM_DT)) for i in range(T)],
            "lr_alpha_mean": round(float(al[k].mean()), 4),
            "hr_alpha_mean": round(float(ha_np[k].mean() / 255.0), 4),
            "motion_px_p50_p99": [round(float(np.percentile(mot[k][al[k] > 0.05], q)), 3) if (al[k] > 0.05).any() else 0.0
                                  for q in (50, 99)],
        })
    st = sc.warmup
    meta = {
        "scenario": sc.asdict(),
        "split": sc.split, "seed": sc.seed, "kind": sc.kind, "tags": list(sc.tags),
        "solver_version": sim.solver_version(),
        "lr_res": LR_RES_3D, "hr_res": hr_res, "dt": SIM_DT,
        "frames": T, "n_cams": n_cams,
        "start_step": st + 1,              # solver step count of frame 0 (frame i = state after step st+1+i)
        "effect_centroid": [round(v, 4) for v in centroid],
        "guidance": guide.asdict() if guide is not None else None,
        "tracking_16": {
            "what": "rel L2 at 16^3 (avg pool) of guided HR vs LR, columns density, T-ambient, velocity",
            "frames": tracks[st:st + T].round(4).tolist(),
            "max_after_step10": tracks[10:].max(0).round(4).tolist(),
            "mean_after_step10": tracks[10:].mean(0).round(4).tolist(),
        },
        "cams": cam_meta,
        "finite": finite and bool(np.isfinite(tracks).all()),
        "seconds": {"total": round(time.perf_counter() - t0, 2), "render": round(t_render, 2),
                    "sim_and_render": round(t_sim, 2)},
    }
    tmp = d / "meta.tmp.json"
    tmp.write_text(json.dumps(meta, indent=1))
    os.replace(tmp, d / "meta.json")
    return meta
