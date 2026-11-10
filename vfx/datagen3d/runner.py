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
from .pack import PACK_NAME, read_all, write_pack
from .scenarios import scene_tags

FILES = ("lr.npy", "hr.npy", "aux.npy", "hr_alpha.npy")      # storage "npy" (s3d_v1)
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
    d = Path(d)
    if not (d / "meta.json").exists():
        return False
    return (d / PACK_NAME).exists() or all((d / f).exists() for f in FILES)


def storage_of(d: Path) -> str:
    return "pack" if (Path(d) / PACK_NAME).exists() else "npy"


def load_arrays(d: Path) -> dict:
    """Whole scenario {lr, hr, aux, hr_alpha} as (K,T,...) numpy arrays, either storage."""
    d = Path(d)
    if storage_of(d) == "pack":
        return read_all(d / PACK_NAME)
    return {n: np.load(d / f"{n}.npy") for n in ("lr", "hr", "aux", "hr_alpha")}


# Per-camera validity: the v2 camera policy renders spare candidate cameras and keeps the first n valid ones.
# Thresholds mirror what the Layer-3 checks call "visible" / "in range" (with margin).
VALID = {"alpha_min_frame": 0.01, "alpha_mean": 0.02, "frame_mean": (4 / 255, 246 / 255), "lum_ratio": (0.6, 1.67)}
CAM_TAGS = {"fill_alpha_cover": 0.5, "flame_frac": 0.25}


def cam_stats(lr, hr, aux, ha) -> dict:
    """GPU tensors of one camera sequence: lr/hr (T,3,h,w) uint8, aux (T,4,h,w), ha (T,h,w) -> stats + reasons."""
    a = aux[:, 0].float()
    cover = (a > 0.02).float().mean((1, 2))
    lf, hf = lr.float() / 255, hr.float() / 255
    lum = lambda x: 0.2126 * x[:, 0] + 0.7152 * x[:, 1] + 0.0722 * x[:, 2]
    m = (a > 0.02) | (ha.float() > 0.02 * 255)
    ll, lh = lum(lf), lum(hf)
    ratio = float(lh[m].mean() / ll[m].mean().clamp(min=1e-4)) if bool(m.any()) else float("nan")
    flame = (lh > 0.45) & (hf[:, 0] - hf[:, 2] > 0.25) & m
    st = {"alpha_cover": round(float(cover.mean()), 4), "alpha_cover_min": round(float(cover.min()), 4),
          "lr_mean": round(float(lf.mean()), 4), "hr_mean": round(float(hf.mean()), 4),
          "lum_ratio_hr_lr": round(ratio, 4),
          "flame_frac": round(float(flame.sum() / m.sum().clamp(min=1)), 4)}
    why = []
    if st["alpha_cover_min"] < VALID["alpha_min_frame"] or st["alpha_cover"] < VALID["alpha_mean"]:
        why.append("effect barely visible")
    lo, hi = VALID["frame_mean"]
    if not (lo < st["lr_mean"] < hi and lo < st["hr_mean"] < hi):
        why.append("black/saturated")
    if not (VALID["lum_ratio"][0] <= ratio <= VALID["lum_ratio"][1]):
        why.append(f"hr/lr luminance {ratio:.2f}")
    st["invalid"] = why
    return st


def cam_tags(c, st: dict, T: int) -> list:
    t = []
    dist = float(np.mean([np.linalg.norm(np.subtract(c.at(i * SIM_DT).position, c.at(i * SIM_DT).target))
                          for i in range(T)]))
    if dist <= R.CAM_RANGES_V2["closeup_tag_dist"]:
        t.append("closeup")
    if st["alpha_cover"] >= CAM_TAGS["fill_alpha_cover"]:
        t.append("fill")
    if st["flame_frac"] >= CAM_TAGS["flame_frac"]:
        t.append("flame_dominated")
    info = getattr(c, "info", None) or {}
    if info.get("occluded_view"):
        t.append("occluded_view")
    if info.get("aim") == "emitter":
        t.append("emitter_aim")
    if c.moving:
        t.append("moving")
    return t


def scenario_cam_tags(tags: list, cam_meta: list) -> list:
    """Scenario-level tags derived from the camera tags (any camera)."""
    ct = set().union(*[set(c["tags"]) for c in cam_meta]) if cam_meta else set()
    out = [t for t in ("closeup", "fill", "flame_dominated") if t in ct]
    if "large_fire" in tags and "closeup" in ct:
        out.append("large_fire_closeup")
    return out


def _save(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def run(sc, out_dir, n_cams: int = DEFAULT_CAMS, guide: GuideConfig3D | None = DEFAULT_GUIDE,
        hr_res: int = HR_RES_3D, cam_policy: str = "v1", n_spare: int = 0, storage: str = "npy") -> dict:
    """cam_policy "v1": R.sample_cameras (s3d_v1); "v2": R.sample_cameras_v2 with `n_spare` extra candidate
    cameras rendered and the first `n_cams` valid ones kept (VALID). storage "npy" | "pack"."""
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
            if cam_policy == "v2":
                cams = R.sample_cameras_v2(rng, centroid, n_cams + n_spare, sc, T, SIM_DT)
            else:
                cams = R.sample_cameras(rng, centroid, n_cams)
            crops = [R.choose_crop(rng, c, T, SIM_DT) for c in cams]
            pr.begin(sc.obstacles, cams, crops, T)
        if s >= sc.warmup:
            tr0 = time.perf_counter()
            pr.frame(s - sc.warmup, lr, hr, SIM_DT)
            t_render += time.perf_counter() - tr0
    tracks = torch.stack(tracks).cpu().numpy()              # (steps, 3) density, T-amb, velocity
    stats = [cam_stats(pr.lr[k], pr.hr[k], pr.aux[k], pr.hr_alpha[k]) for k in range(len(cams))]
    valid = [k for k in range(len(cams)) if not stats[k]["invalid"]]
    keep = sorted((valid + [k for k in range(len(cams)) if k not in valid])[:n_cams])
    rejected = [{"orbit": {kk: (list(v) if isinstance(v, tuple) else v) for kk, v in cams[k].__dict__.items()},
                 **stats[k]} for k in range(len(cams)) if k not in keep]
    sel = torch.tensor(keep, device=pr.lr.device)
    lr_np = pr.lr[sel].cpu().numpy()
    hr_np = pr.hr[sel].cpu().numpy()
    aux_np = pr.aux[sel].cpu().numpy()
    ha_np = pr.hr_alpha[sel].cpu().numpy()
    cams = [cams[k] for k in keep]
    crops = [crops[k] for k in keep]
    stats = [stats[k] for k in keep]
    finite = bool(np.isfinite(aux_np).all())
    t_sim = time.perf_counter() - t0
    if storage == "pack":
        write_pack(d / PACK_NAME, {"lr": lr_np, "hr": hr_np, "aux": aux_np, "hr_alpha": ha_np})
    else:
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
            "stats": stats[k],
            "flagged": bool(stats[k]["invalid"]),
            "tags": cam_tags(c, stats[k], T),
        })
    st = sc.warmup
    tags = list(sc.tags) + [t for t in scene_tags(sc) if t not in sc.tags]
    tags += [t for t in scenario_cam_tags(tags, cam_meta) if t not in tags]
    meta = {
        "scenario": sc.asdict(),
        "split": sc.split, "seed": sc.seed, "kind": sc.kind, "tags": tags,
        "storage": storage, "cam_policy": cam_policy, "rejected_cams": rejected,
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
