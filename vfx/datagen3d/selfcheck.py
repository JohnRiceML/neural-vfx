"""Self-check of a built route-B dataset (or a tiny fresh build with --mini).

    .venv/Scripts/python -m vfx.datagen3d.selfcheck --version s3d_v1 [--png-dir DIR] [--max-scn N]
    .venv/Scripts/python -m vfx.datagen3d.selfcheck --mini            # builds s3d_selfcheck (2+1+1 scn), checks it

Checks: split disjointness + seed ranges; per scenario shapes / dtypes / finiteness / value ranges;
loader item shapes; lr/hr alignment (in pixels with no medium in either render - floor, obstacles,
background - the best integer shift of hr vs lr within +-3 px must be (0,0)); motion sanity (lr[t-1]
backward-warped by aux motion vs lr[t], MSE ratio no-warp / warp on medium pixels; static and moving
cameras separately); tracking summary. Writes lr|hr|alpha|motion PNG strips with --png-dir.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vfx.core import paths

from . import render as R
from .build import SEED_BASE, SEED_SPAN
from .dataset import load_sequences
from .runner import is_complete, scenario_dir

FAIL: list[str] = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def backward_warp(prev: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
    """prev (C,h,w), motion (2,h,w) px (x right, y down): out(p) = prev(p - motion)."""
    C, h, w = prev.shape
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    sx = xs - motion[0]
    sy = ys - motion[1]
    grid = torch.stack([(sx + 0.5) / w * 2 - 1, (sy + 0.5) / h * 2 - 1], -1)[None]
    return F.grid_sample(prev[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]


def alignment(lr: np.ndarray, hr: np.ndarray, a_lr: np.ndarray, a_hr: np.ndarray, r: int = 3):
    """lr, hr (3,h,w) uint8; alphas (h,w) in 0..1. Best (dy,dx) of hr vs lr on medium-free pixels."""
    L = lr.astype(np.float32).mean(0)
    H = hr.astype(np.float32).mean(0)
    m = (a_lr < 0.005) & (a_hr < 0.005)
    m[:r] = m[-r:] = False
    m[:, :r] = m[:, -r:] = False
    if m.sum() < 500:
        return None, None, None
    # use gradients so the smooth background gradient does not dominate
    gL = np.abs(np.diff(L, axis=1, append=L[:, -1:])) + np.abs(np.diff(L, axis=0, append=L[-1:]))
    gH = np.abs(np.diff(H, axis=1, append=H[:, -1:])) + np.abs(np.diff(H, axis=0, append=H[-1:]))
    best, err0 = None, None
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            e = float(np.abs(gL - np.roll(gH, (dy, dx), (0, 1)))[m].mean())
            if best is None or e < best[0]:
                best = (e, dy, dx)
            if dy == 0 and dx == 0:
                err0 = e
    return (best[1], best[2]), err0, float(np.abs(L - H)[m].mean())


SCALES = (-0.5, 0.0, 0.5, 1.0)


def motion_scan(lr: np.ndarray, aux: np.ndarray):
    """Per sequence (T,3,h,w), (T,4,h,w): mean over t of MSE(warp(lr[t-1], s * motion), lr[t]) on alpha > 0.1
    for s in SCALES (s = 0: no warp, 1: as stored). None if the sequence has too little medium."""
    out = {s: [] for s in SCALES}
    for t in range(1, lr.shape[0]):
        m = torch.from_numpy(aux[t, 0].astype(np.float32) > 0.1)
        if m.sum() < 200:
            continue
        cur = torch.from_numpy(lr[t].astype(np.float32))
        prev = torch.from_numpy(lr[t - 1].astype(np.float32))
        mot = torch.from_numpy(aux[t, 2:4].astype(np.float32))
        for s in SCALES:
            out[s].append(float(((backward_warp(prev, mot * s) - cur) ** 2).mean(0)[m].mean()))
    return {s: float(np.mean(v)) for s, v in out.items()} if out[0.0] else None


def camera_motion_check(split: str, seed: int) -> float:
    """Camera-only motion on our own pipeline: re-simulate the scenario's 64^3 run to frame 0, render the
    FROZEN volume from a moving camera at t-1 and t (crop + pool exactly as the build), warp by the stored-px
    motion. Returns MSE(no warp) / MSE(warp) on alpha > 0.1."""
    from . import sim
    from .scenarios import sample_scenario
    from vfx.core.fields3d import SIM_DT
    sc = sample_scenario(seed, split)
    st = None
    for s, lr, _, _ in sim.run_pair(sc, None, steps=sc.warmup + 1, with_hr=False):
        st = lr
    c = R.sample_cameras(np.random.default_rng(seed), R.effect_centroid(st), 1)[0]
    c.yaw_rate, c.pitch_rate, c.dolly_rate = 0.5, 0.1, 0.15
    pr = R.PairRenderer()
    crop = R.choose_crop(np.random.default_rng(0), c, 2, SIM_DT)
    pr.begin(sc.obstacles, [c], [crop], 2)
    pr.cams = [c]
    zero = st.data.clone()
    zero[1:4] = 0.0                                       # no fluid motion: camera only
    st.data.copy_(zero)
    pr.frame(0, st, None, SIM_DT)
    pr.frame(1, st, None, SIM_DT)
    lr = pr.lr[0].cpu().numpy()
    aux = pr.aux[0].cpu().numpy()
    r = motion_scan(lr, aux)
    return r[0.0] / (r[1.0] + 1e-6) if r else float("nan")


def strip(lr, hr, aux, ha):
    """One frame of one camera: lr | hr | alpha | |motion| as an RGB strip."""
    a = (np.clip(aux[0].astype(np.float32), 0, 1) * 255).astype(np.uint8)
    mo = np.linalg.norm(aux[2:4].astype(np.float32), axis=0)
    mo = (np.clip(mo / max(4.0, np.percentile(mo, 99)), 0, 1) * 255).astype(np.uint8)
    g = lambda x: np.repeat(x[..., None], 3, 2)
    return np.concatenate([lr.transpose(1, 2, 0), hr.transpose(1, 2, 0), g(a), g(mo)], 1)


def check_version(version: str, png_dir: str | None, max_scn: int | None):
    from .png import write_png
    root = paths.dataset_dir(version)
    print(f"[selfcheck3d] {root}")
    splits = json.loads((root / "splits.json").read_text())
    sets = {k: set(v) for k, v in splits.items()}
    check(not (sets["train"] & sets["test"]) and not (sets["train"] & sets["ood"]) and not (sets["test"] & sets["ood"]),
          "splits disjoint")
    check(all(SEED_BASE[k] <= s < SEED_BASE[k] + SEED_SPAN for k, v in splits.items() for s in v), "seed ranges")
    card_p = root / "card.json"
    if card_p.exists():
        card = json.loads(card_p.read_text())
        check(card["bytes"] <= 4e9, f"size {card['bytes'] / 1e9:.2f} GB <= 4 GB")
    shifts, err0s, absd, tracks = [], [], [], []
    scans = {False: [], True: []}
    n_done = n_missing = 0
    pngs = 0
    for split, seeds in splits.items():
        for seed in seeds[: max_scn or None]:
            d = scenario_dir(root, split, seed)
            if not is_complete(d):
                n_missing += 1
                continue
            n_done += 1
            meta = json.loads((d / "meta.json").read_text())
            K, T = meta["n_cams"], meta["frames"]
            lr = np.load(d / "lr.npy")
            hr = np.load(d / "hr.npy")
            aux = np.load(d / "aux.npy")
            ha = np.load(d / "hr_alpha.npy")
            s = R.CROP
            ok = (lr.shape == (K, T, 3, s, s) and hr.shape == lr.shape and aux.shape == (K, T, 4, s, s)
                  and ha.shape == (K, T, s, s) and lr.dtype == np.uint8 and hr.dtype == np.uint8
                  and aux.dtype == np.float16 and ha.dtype == np.uint8)
            fin = bool(np.isfinite(aux).all()) and meta["finite"]
            rng_ok = (aux[:, :, 0].min() >= 0 and aux[:, :, 0].max() <= 1.001 and aux[:, :, 1].min() >= 0)
            if not (ok and fin and rng_ok):
                check(False, f"{split}/scn_{seed:06d}: shapes/dtypes {ok} finite {fin} ranges {rng_ok}")
            tracks.append(meta["tracking_16"]["max_after_step10"])
            for k in range(K):
                for t in (0, T // 2, T - 1):
                    sh, e0, ad = alignment(lr[k, t], hr[k, t], aux[k, t, 0].astype(np.float32), ha[k, t] / 255.0)
                    if sh is not None:
                        shifts.append(sh)
                        err0s.append(e0)
                        absd.append(ad)
                r = motion_scan(lr[k], aux[k])
                if r is not None:
                    scans[bool(meta["cams"][k]["moving"])].append(r)
                if png_dir and pngs < 12 and (split != "train" or seed % 9 == 0):
                    Path(png_dir).mkdir(parents=True, exist_ok=True)
                    t = T // 2
                    name = f"{split}_scn{seed:06d}_cam{k}_{meta['kind']}{'_moving' if meta['cams'][k]['moving'] else ''}"
                    write_png(Path(png_dir) / f"{name}_strip.png", strip(lr[k, t], hr[k, t], aux[k, t], ha[k, t]))
                    write_png(Path(png_dir) / f"{name}_lr.png", lr[k, t].transpose(1, 2, 0))
                    write_png(Path(png_dir) / f"{name}_hr.png", hr[k, t].transpose(1, 2, 0))
                    pngs += 1
    check(n_done > 0, f"{n_done} complete scenarios ({n_missing} missing/incomplete)")
    if shifts:
        zero = sum(1 for s in shifts if s == (0, 0))
        check(zero == len(shifts), f"lr/hr alignment: best shift (0,0) in {zero}/{len(shifts)} frames "
                                   f"(gradient MAD at 0 shift median {np.median(err0s):.2f}, "
                                   f"abs diff on medium-free px median {np.median(absd):.2f} /255)")
    for moving, sc in scans.items():
        if not sc:
            continue
        e = {x: np.asarray([r[x] for r in sc]) for x in SCALES}
        sign_ok = float(np.mean(e[0.5] < e[-0.5]))
        name = "moving" if moving else "static"
        print(f"  info motion ({name} cams, n={len(sc)}): mean MSE of warped lr[t-1] vs lr[t] on alpha>0.1 by "
              f"motion scale: " + ", ".join(f"{x:+.1f}: {e[x].mean():.1f}" for x in SCALES)
              + f"; warp(1.0) better than none in {np.mean(e[1.0] < e[0.0]) * 100:.0f} %, "
              f"warp(0.5) in {np.mean(e[0.5] < e[0.0]) * 100:.0f} %")
        check(sign_ok >= 0.8, f"motion sign ({name} cams): warp(+0.5) beats warp(-0.5) in {sign_ok * 100:.0f} % of sequences")
    test_seeds = [s for s in splits.get("test", []) if is_complete(scenario_dir(root, "test", s))]
    if test_seeds:
        rc = camera_motion_check("test", test_seeds[0])
        check(rc > 3.0, f"camera-only motion (frozen volume, moving camera, own crop/pool): "
                        f"MSE no-warp/warp = {rc:.1f}")
    if tracks:
        tr = np.asarray(tracks)
        print(f"  info tracking rel L2 at 16^3 (per-scenario max over steps >= 10): density median "
              f"{np.median(tr[:, 0]):.3f} max {tr[:, 0].max():.3f}; T median {np.median(tr[:, 1]):.3f} max "
              f"{tr[:, 1].max():.3f}; velocity median {np.median(tr[:, 2]):.3f} max {tr[:, 2].max():.3f}")
    ds = load_sequences(version, "train", 8)
    if len(ds):
        it = ds[0]
        check(tuple(it["lr"].shape) == (8, 3, R.CROP, R.CROP) and it["lr"].dtype == torch.uint8
              and tuple(it["aux"].shape) == (8, 4, R.CROP, R.CROP) and it["aux"].dtype == torch.float16
              and it["hr"].shape == it["lr"].shape, f"loader item shapes ({len(ds)} train windows of 8)")
    return not FAIL


def mini():
    from .build import build
    from .scenarios import sample_scenario  # noqa: F401
    cfg = {"n_train": 2, "n_test": 1, "n_ood": 1, "frames": 8, "cams": 2}
    build("s3d_selfcheck", cfg)
    return check_version("s3d_selfcheck", None, None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--version")
    ap.add_argument("--png-dir")
    ap.add_argument("--max-scn", type=int)
    ap.add_argument("--mini", action="store_true")
    a = ap.parse_args(argv)
    ok = mini() if a.mini else check_version(a.version, a.png_dir, a.max_scn)
    print("[selfcheck3d]", "PASS" if ok else f"FAIL ({len(FAIL)})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
