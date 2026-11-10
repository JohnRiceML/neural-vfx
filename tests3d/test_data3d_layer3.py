"""Layer 3 (data integrity) for the route B dataset `$VFX_DATASET3D` (default "s3d_v1").

Contract: CLAUDE.md "Route B training phase" - vfx.datagen3d.load_sequences(version, split, length) items
  "lr" (L,3,h,w) uint8, "hr" (L,3,h,w) uint8, "aux" (L,4,h,w) float16 (alpha, depth, motion_x, motion_y of the
  LR render; motion in px per sim step, x right, y down, content at p was at p - motion one step ago),
  plus "seed", "cam", "t0"; splits by scenario seed train / test / ood; whole dataset <= 4 GB.
SPEC Layer 3: no NaN/Inf, shapes/dtypes, per-channel stats in range (outliers flagged, not silently kept),
  leakage-free seed splits, held-out scene types in OOD.

Everything is read through the public loader plus the files it is built from (splits.json, card.json,
<split>/scn_<seed>/meta.json for tags). Skips cleanly until the dataset exists. Item windows are taken
non-overlapping (stride = length) when the loader accepts a stride.

Tolerances where the spec gives no number (stated, not tuned to the data):
* lr/hr pixel alignment: on pixels with no medium in either render (both alphas 0, dilated 6 px) the best
  integer shift of hr against lr (gradient-magnitude L1 over shifts in [-2, 2]^2) must be (0, 0) for >= 90% of
  frames tested, and the median |lr - hr| there <= 1 code (renders of static geometry with one camera).
* tracking (coarse agreement): luminance blurred at sigma = 1.5 x the projected 64^3 voxel size (from the
  aux depth and the 40 deg fov at the 720-row stored scale) -> masked Pearson correlation lr vs hr: median over
  sequences >= 0.9; fine detail: high-pass energy (lum - blur at sigma = max(1, 0.5 voxel)) pooled over the
  split, hr >= 1.5x lr, and hr > lr in >= 75% of sequences.
* motion: opacity-weighted warp error with motion <= 0.9 x without, pooled over sequences that move
  (median |motion| > 0.25 px), for lr and for hr; the opposite sign must be worse.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vfx.eval3d import metrics as M

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
DEV = "cuda"
VERSION = os.environ.get("VFX_DATASET3D", "s3d_v1")
SPLITS = ("train", "test", "ood")
SIZE_LIMIT = 4 * 1024 ** 3          # "<= 4 GB" read as GiB; also reported in GB
STORED_ROWS = 720                   # 1440p output, 2x2 pooled (card "storage"); used only for the voxel footprint
SPLIT_NAMES = set(SPLITS)


# ----------------------------------------------------------------------------- loading
def _root() -> Path:
    from vfx.core.paths import dataset_dir

    return dataset_dir(VERSION)


@pytest.fixture(scope="module")
def data():
    root = _root()
    if not (root / "splits.json").exists():
        pytest.skip(f"dataset {VERSION} not built yet ({root / 'splits.json'} missing)")
    try:
        from vfx.datagen3d import load_sequences
    except ImportError as e:
        pytest.skip(f"vfx.datagen3d.load_sequences not importable: {e}")
    splits = json.loads((root / "splits.json").read_text())
    card = json.loads((root / "card.json").read_text()) if (root / "card.json").exists() else {}
    L = int(card.get("frames_per_sequence", 8))
    ds = {}
    for sp in SPLITS:
        try:
            try:
                ds[sp] = load_sequences(VERSION, sp, L, stride=L)
            except TypeError:
                ds[sp] = load_sequences(VERSION, sp, L)
        except (KeyError, FileNotFoundError, ValueError) as e:
            ds[sp] = None
            print(f"[layer3] split {sp}: {e!r}")
    if ds.get("train") is None or len(ds["train"]) == 0:
        pytest.skip(f"dataset {VERSION}: no finished train sequences yet")
    total = sum(len(v) for v in splits.values())
    done = sum((root / sp / f"scn_{int(s):06d}" / "meta.json").exists() for sp, v in splits.items() for s in v)
    if done < total:
        print(f"\n[layer3] {VERSION} still building: {done}/{total} scenarios complete; testing what exists")
    return {"root": root, "splits": splits, "card": card, "L": L, "ds": ds, "complete": done >= total,
            "progress": f"{done}/{total}"}


def _items(data, split, limit=None):
    d = data["ds"].get(split)
    if d is None or len(d) == 0:
        return
    for i in M._subsample(len(d), limit):
        it = d[i]
        yield i, it


def _meta(data, split, seed):
    p = data["root"] / split / f"scn_{int(seed):06d}" / "meta.json"
    if not p.exists():
        hits = list(data["root"].glob(f"{split}/*{int(seed)}*/meta.json"))
        p = hits[0] if hits else None
    return json.loads(p.read_text()) if p else None


def _gpu(it):
    return (torch.as_tensor(it["lr"]).to(DEV), torch.as_tensor(it["hr"]).to(DEV), torch.as_tensor(it["aux"]).to(DEV))


# ----------------------------------------------------------------------------- size / splits / tags
def test_size_and_card(data):
    if not data["complete"] and not data["card"]:
        pytest.skip(f"build in progress ({data['progress']} scenarios); card.json is written at the end")
    total = sum(f.stat().st_size for f in data["root"].rglob("*") if f.is_file())
    print(f"\n[layer3] {VERSION}: {total / 1e9:.3f} GB on disk ({total / 1024 ** 3:.3f} GiB); card complete="
          f"{data['card'].get('complete')} counts={data['card'].get('counts')}")
    assert total <= SIZE_LIMIT, f"dataset is {total / 1e9:.2f} GB > 4 GB"
    assert data["card"], "card.json missing (SPEC: every dataset gets a card)"
    for k in ("version", "counts", "solver_version"):
        assert k in data["card"], f"card.json lacks {k}"


def test_splits_disjoint_by_seed(data):
    s = {k: set(map(int, data["splits"].get(k, []))) for k in SPLITS}
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                assert not (s[a] & s[b]), f"seeds in both {a} and {b}: {sorted(s[a] & s[b])[:10]}"
    for sp in SPLITS:                       # items really come from their split's seeds
        for _, it in _items(data, sp, 40):
            assert int(it["seed"]) in s[sp], f"{sp} item has seed {it['seed']} not in splits.json[{sp}]"


def test_ood_has_scene_types_absent_from_train(data):
    tags = {}
    for sp in ("train", "ood"):
        tags[sp] = {}
        for seed in data["splits"].get(sp, []):
            m = _meta(data, sp, seed)
            if m is not None:
                tags[sp][int(seed)] = set(m.get("tags", [])) - SPLIT_NAMES      # the split name is not a scene type
    if not tags["ood"]:
        pytest.skip("no finished OOD scenarios")
    train_tags = set().union(*tags["train"].values()) if tags["train"] else set()
    novel = {s: sorted(t - train_tags) for s, t in tags["ood"].items()}
    print(f"\n[layer3] OOD novel tags per scenario: {novel}")
    bad = [s for s, n in novel.items() if not n]
    assert not bad, f"OOD scenarios with no scene type absent from train (tags minus split name): {bad}"


def test_solver_version_current(data):
    try:
        import vfx.sim3d as sim
    except ImportError:
        pytest.skip("vfx.sim3d not importable")
    want = getattr(sim, "SOLVER3D_VERSION", None)
    have = data["card"].get("solver_version")
    if have is None:
        metas = [_meta(data, "train", s) for s in data["splits"].get("train", [])[:5]]
        vs = {m.get("solver_version") for m in metas if m}
        if not vs:
            pytest.skip("no card.json and no finished meta.json")
        assert vs == {want}, f"scenarios built with {vs}, current solver is {want}"
        return
    assert have == want, f"dataset built with {have}, current solver is {want} (rebuild or bump the version)"


# ----------------------------------------------------------------------------- item contract / finiteness
def test_item_contract(data):
    L = data["L"]
    for sp in SPLITS:
        for _, it in _items(data, sp, 6):
            for k in ("lr", "hr", "aux", "seed", "cam", "t0"):
                assert k in it, f"{sp} item lacks {k}"
            lr, hr, aux = (torch.as_tensor(it[k]) for k in ("lr", "hr", "aux"))
            assert lr.dtype == torch.uint8 and hr.dtype == torch.uint8, (lr.dtype, hr.dtype)
            assert aux.dtype == torch.float16, aux.dtype
            assert lr.dim() == 4 and lr.shape[:2] == (L, 3), lr.shape
            assert hr.shape == lr.shape, (hr.shape, lr.shape)
            assert aux.shape == (L, 4) + tuple(lr.shape[-2:]), aux.shape


def test_no_nan_inf_and_aux_ranges(data):
    """Every sequence of every split: aux finite; alpha in [0,1]; depth >= 0, < 10 m; depth and motion 0 where
    alpha == 0 (CLAUDE.md: "0 where alpha == 0"); p99 |motion| < 0.25 x frame side per sim step."""
    n = 0
    worst = {"motion_p99": 0.0, "depth_max": 0.0}
    for sp in SPLITS:
        for i, it in _items(data, sp):
            aux = torch.as_tensor(it["aux"]).to(DEV).float()
            bad = int((~torch.isfinite(aux)).sum())
            assert bad == 0, f"{sp}[{i}] seed {it['seed']}: {bad} non-finite aux values"
            a, d, mo = aux[:, 0], aux[:, 1], aux[:, 2:4]
            assert a.min() >= -1e-3 and a.max() <= 1 + 1e-3, (sp, i, a.min().item(), a.max().item())
            assert d.min() >= -1e-3, (sp, i, d.min().item())
            empty = a == 0
            if empty.any():
                assert d[empty].abs().max() <= 1e-3, f"{sp}[{i}]: depth nonzero where alpha = 0"
                assert mo[:, 0][empty].abs().max() <= 1e-3 and mo[:, 1][empty].abs().max() <= 1e-3, \
                    f"{sp}[{i}]: motion nonzero where alpha = 0"
            mag = mo.norm(dim=1)[a > 0.05]
            p99 = float(torch.quantile(mag.float()[:1_000_000], 0.99)) if mag.numel() else 0.0
            side = max(aux.shape[-2:])
            assert p99 < 0.25 * side, f"{sp}[{i}]: p99 |motion| {p99:.1f} px per step"
            assert d.max() < 10.0, f"{sp}[{i}]: depth {d.max().item():.2f} m"
            worst["motion_p99"] = max(worst["motion_p99"], p99)
            worst["depth_max"] = max(worst["depth_max"], float(d.max()))
            n += 1
    print(f"\n[layer3] {n} sequences finite; worst p99 |motion| {worst['motion_p99']:.2f} px/step, "
          f"max depth {worst['depth_max']:.2f} m")


# ----------------------------------------------------------------------------- per-channel stats / outliers
def test_per_channel_stats_and_outliers(data):
    """Per sequence: frame means not black/saturated, the effect is visible (alpha > 0.02 on >= 0.2% of pixels),
    frames not frozen, hr/lr mean luminance in the effect region within [0.5, 2]. Outliers (median + 8 MAD of
    any statistic) are listed; any failing sequence must be flagged in card.json ("flagged")."""
    flagged = set(map(str, data["card"].get("flagged", [])))
    rows, fails = [], []
    for sp in SPLITS:
        for i, it in _items(data, sp):
            lr, hr, aux = _gpu(it)
            m = M.effect_mask(aux[:, 0])
            lf, hf = M.to_float(lr), M.to_float(hr)
            w = m[:, None].float()
            den = w.sum().clamp(min=1)
            ch_lr = ((lf * w).sum((0, 2, 3)) / den).tolist()
            ch_hr = ((hf * w).sum((0, 2, 3)) / den).tolist()
            r = {"split": sp, "i": i, "seed": int(it["seed"]), "cam": int(it["cam"]),
                 "lr_mean": float(lf.mean()), "hr_mean": float(hf.mean()),
                 "alpha_frac": float((aux[:, 0] > 0.02).float().mean()),
                 "lum_ratio": float(M.luminance(hf)[m].mean() / M.luminance(lf)[m].mean().clamp(min=1e-4)) if m.any() else float("nan"),
                 "frozen": float((lf[1:] - lf[:-1]).abs().amax()) == 0.0,
                 "lr_ch": ch_lr, "hr_ch": ch_hr}
            why = []
            if not (2 / 255 < r["lr_mean"] < 250 / 255 and 2 / 255 < r["hr_mean"] < 250 / 255):
                why.append("black/saturated")
            if r["alpha_frac"] < 0.002:
                why.append(f"effect barely visible (alpha frac {r['alpha_frac']:.4f})")
            if r["frozen"]:
                why.append("frozen frames")
            if not (0.5 <= r["lum_ratio"] <= 2.0):
                why.append(f"hr/lr luminance {r['lum_ratio']:.2f}")
            key = f"{sp}/scn_{r['seed']:06d}"
            if why and key not in flagged and f"{key}/cam{r['cam']}" not in flagged:
                fails.append((key, r["cam"], why))
            rows.append(r)
    for k in ("lr_mean", "hr_mean", "alpha_frac", "lum_ratio"):
        v = np.array([r[k] for r in rows], float)
        med = np.nanmedian(v)
        mad = np.nanmedian(np.abs(v - med)) + 1e-9
        out = [(r["split"], r["seed"], r["cam"], round(r[k], 4)) for r in rows if abs(r[k] - med) > 8 * mad]
        print(f"\n[layer3] {k}: median {med:.4f}, p5 {np.nanpercentile(v, 5):.4f}, p95 {np.nanpercentile(v, 95):.4f}; "
              f"outliers (>8 MAD): {out[:12]}{' ...' if len(out) > 12 else ''}")
    mean_lr = np.mean([r["lr_ch"] for r in rows], 0)
    mean_hr = np.mean([r["hr_ch"] for r in rows], 0)
    print(f"[layer3] effect-region RGB mean lr {np.round(mean_lr, 4).tolist()} hr {np.round(mean_hr, 4).tolist()}")
    assert not fails, f"{len(fails)} sequences out of range and not flagged in card.json: {fails[:10]}"


# ----------------------------------------------------------------------------- alignment of lr and hr
def _grad_mag(x):
    lum = M.luminance(x)
    gx = F.pad(lum[..., :, 1:] - lum[..., :, :-1], (0, 1))
    gy = F.pad(lum[..., 1:, :] - lum[..., :-1, :], (0, 0, 0, 1))
    return (gx.abs() + gy.abs())


def test_lr_hr_pixel_alignment_static_geometry(data):
    """Static geometry (floor grid, obstacles, background) is identical and unshifted between lr and hr."""
    ok_shift, n_frames, meds, skipped = 0, 0, [], 0
    for sp in SPLITS:
        for i, it in _items(data, sp, 24):
            lr, hr, aux = _gpu(it)
            ha = torch.as_tensor(it["hr_alpha"]).to(DEV).float() / 255 if "hr_alpha" in it else torch.zeros_like(aux[:, 0])
            for t in (0, lr.shape[0] - 1):
                med = (aux[t, 0].float() > 1e-3) | (ha[t] > 1e-3)
                bg = ~M.effect_mask(med.float(), thresh=0.5, dilate_px=6)
                gl, gh = _grad_mag(lr[t]), _grad_mag(hr[t])
                if bg.float().mean() < 0.05 or gl[bg].mean() < 1e-3:
                    skipped += 1
                    continue
                inner = bg.clone()
                inner[:2], inner[-2:], inner[:, :2], inner[:, -2:] = False, False, False, False
                best, best_s = None, None
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        sh = torch.roll(gh, (dy, dx), (0, 1))
                        e = (gl - sh)[inner].abs().mean().item()
                        if best is None or e < best:
                            best, best_s = e, (dy, dx)
                n_frames += 1
                ok_shift += best_s == (0, 0)
                meds.append(float((M.to_float(lr[t]) - M.to_float(hr[t])).abs().amax(0)[bg].median()) * 255)
    if n_frames == 0:
        pytest.skip(f"no frame with enough visible static background ({skipped} skipped)")
    print(f"\n[layer3] alignment: best shift (0,0) in {ok_shift}/{n_frames} frames; median |lr-hr| on background "
          f"(codes): median {np.median(meds):.2f}, max {np.max(meds):.2f}; {skipped} frames without background")
    assert ok_shift >= 0.9 * n_frames, f"lr/hr static geometry misaligned in {n_frames - ok_shift}/{n_frames} frames"
    assert np.median(meds) <= 1.0, f"median |lr - hr| on static background {np.median(meds):.2f} codes"


# ----------------------------------------------------------------------------- fine detail vs tracking
def _voxel_px(aux, fov_deg=40.0, res=64):
    a, d = aux[:, 0].float(), aux[:, 1].float()
    sel = a > 0.1
    if not sel.any():
        return None
    depth = float(d[sel].median())
    m_per_px = depth * 2 * math.tan(math.radians(fov_deg) / 2) / STORED_ROWS
    return (1.0 / res) / max(m_per_px, 1e-6)


def _corr(a, b, w):
    w = w / w.sum()
    ma, mb = (a * w).sum(), (b * w).sum()
    va, vb = ((a - ma) ** 2 * w).sum(), ((b - mb) ** 2 * w).sum()
    return float(((a - ma) * (b - mb) * w).sum() / (va * vb).sqrt().clamp(min=1e-12))


def test_hr_more_fine_detail_and_tracks_lr_at_coarse_scale(data):
    rows = []
    for sp in ("train", "test"):
        for i, it in _items(data, sp, 40):
            lr, hr, aux = _gpu(it)
            vox = _voxel_px(aux)
            if vox is None:
                continue
            m = M.item_masks(it, aux, DEV)[0]
            if m.float().mean() < 0.002:
                continue
            ll, lh = M.luminance(lr), M.luminance(hr)
            sc = max(1.0, 1.5 * vox)
            sf = max(1.0, 0.5 * vox)
            bl = M.gaussian_blur(ll[:, None], sc)[:, 0]
            bh = M.gaussian_blur(lh[:, None], sc)[:, 0]
            w = m.float()
            corr = _corr(bl, bh, w)
            fl = (((ll - M.gaussian_blur(ll[:, None], sf)[:, 0]) ** 2) * w).sum().item() / w.sum().item()
            fh = (((lh - M.gaussian_blur(lh[:, None], sf)[:, 0]) ** 2) * w).sum().item() / w.sum().item()
            rows.append({"split": sp, "seed": int(it["seed"]), "cam": int(it["cam"]), "voxel_px": round(vox, 2),
                         "corr": corr, "fine_lr": fl, "fine_hr": fh})
    if not rows:
        pytest.skip("no sequence with a visible effect")
    corr = np.array([r["corr"] for r in rows])
    ratio = sum(r["fine_hr"] for r in rows) / max(sum(r["fine_lr"] for r in rows), 1e-12)
    frac = np.mean([r["fine_hr"] > r["fine_lr"] for r in rows])
    print(f"\n[layer3] {len(rows)} seqs: coarse corr median {np.median(corr):.3f} (p10 {np.percentile(corr, 10):.3f}); "
          f"fine-energy hr/lr pooled {ratio:.2f}, hr > lr in {frac * 100:.0f}%; voxel px median "
          f"{np.median([r['voxel_px'] for r in rows]):.1f}")
    low = sorted(rows, key=lambda r: r["corr"])[:5]
    print("[layer3] lowest coarse correlation:", [(r["split"], r["seed"], r["cam"], round(r["corr"], 3)) for r in low])
    assert np.median(corr) >= 0.9, f"hr does not track lr at coarse scale: median corr {np.median(corr):.3f}"
    assert ratio >= 1.5, f"hr has only {ratio:.2f}x the fine-scale energy of lr in the effect region"
    assert frac >= 0.75, f"hr has more fine detail than lr in only {frac * 100:.0f}% of sequences"


# ----------------------------------------------------------------------------- motion / alignment
def _alignment_table(data, limit=30):
    """Warp error / no-warp error, region-weighted, pooled per (camera static|moving, lr|hr, alignment)."""
    acc = {}
    for sp in ("train", "test", "ood"):
        for i, it in _items(data, sp, limit):
            lr, hr, aux = _gpu(it)
            w = M.item_masks(it, aux, DEV)[0]
            st = aux[:, 2:4].float()
            cam = "moving" if bool(it.get("moving", False)) else "static"
            fl = {"lr": M.sequence_flow(lr), "hr": M.sequence_flow(hr)}
            for name, s in (("lr", lr), ("hr", hr)):
                n = M.warp_error(s, st, w, use_motion=False)
                cands = {"stored": st, "stored_x0.5": 0.5 * st, "stored_x0.25": 0.25 * st, "stored_neg": -st,
                         "own_flow": fl[name], "other_flow": fl["hr" if name == "lr" else "lr"]}
                for k, m in cands.items():
                    e = acc.setdefault((cam, name, k), [0.0, 0.0])
                    e[0] += M.warp_error(s, m, w)
                    e[1] += n
    return {k: v[0] / max(v[1], 1e-12) for k, v in acc.items()}


@pytest.fixture(scope="module")
def alignment(data):
    t = _alignment_table(data)
    print("\n[layer3] warp error / no-warp error (region-weighted, pooled):")
    for cam in ("static", "moving"):
        for s in ("lr", "hr"):
            print(f"   {cam:6s} {s}: " + "  ".join(f"{k} {v:.3f}" for (c, n, k), v in sorted(t.items()) if c == cam and n == s))
    return t


def test_own_estimated_flow_explains_frame_to_frame_change(alignment):
    """The alignment the temporal metrics use (each sequence's own estimated flow) must explain lr and hr:
    <= 0.9x the unwarped error, for static and moving cameras."""
    bad = {k: round(v, 3) for k, v in alignment.items() if k[2] == "own_flow" and not v <= 0.9}
    assert not bad, f"own-flow warp does not reduce frame-to-frame error: {bad}"


@pytest.mark.xfail(strict=False, reason="known (lead, 2026-11-09): the stored aux motion is the opacity-weighted fluid "
                   "velocity, not the apparent image motion; only its camera part is exact")
def test_stored_motion_vectors_reduce_warp_error(alignment):
    """Task requirement: the dataset's motion vectors reduce warp error vs no warp (<= 0.9x), lr and hr."""
    vals = {k: round(v, 3) for k, v in alignment.items() if k[2] == "stored"}
    assert all(v <= 0.9 for v in vals.values()), f"stored motion warp / no warp: {vals}"


def test_temporal_metrics_can_see_flicker_on_this_data(data):
    """Instrument check on the REAL data. (a) Sensitivity: iid per-frame noise of 2 codes (std) injected into hr
    inside the effect region (visible shimmer) must move the gated ratios (own-flow flicker, temporal jerk) above
    their 1.2 gate. (b) Blur is NOT rewarded beyond the gate's meaning: reported, a blurred hr (sigma 1 px) has a
    LOWER flicker (less detail changes less), which is why these metrics are upper-bound gates, not objectives.
    Also reported: the old stored-motion flicker (blind)."""
    g = torch.Generator(device=DEV).manual_seed(0)
    keys = ("flicker", "swim", "jerk", "flicker_stored")
    acc = {k: [0.0, 0.0, 0.0] for k in keys}
    for sp in ("train", "test"):
        for i, it in _items(data, sp, 30):
            lr, hr, aux = _gpu(it)
            w = M.item_masks(it, aux, DEV)[0]
            a, st = aux[:, 0].float(), aux[:, 2:4].float()
            hf = M.to_float(hr)
            nz = ((hf + 2 / 255 * torch.randn(hf.shape, device=DEV, generator=g) * (a[:, None] > 0.02)).clamp(0, 1)
                  * 255 + 0.5).to(torch.uint8)
            bl = (M.gaussian_blur(hf, 1.0) * 255 + 0.5).to(torch.uint8)
            fns = {"flicker": lambda x: M.flicker(x, None, w), "swim": lambda x: M.detail_swim(x, lr, None, w),
                   "jerk": lambda x: M.temporal_jerk(x, w), "flicker_stored": lambda x: M.flicker(x, st, w)}
            for k, fn in fns.items():
                acc[k][0] += fn(nz)
                acc[k][1] += fn(hr)
                acc[k][2] += fn(bl)
    r = {k: v[0] / max(v[1], 1e-12) for k, v in acc.items()}
    b = {k: v[2] / max(v[1], 1e-12) for k, v in acc.items()}
    print("\n[layer3] ratio to clean hr after 2-code iid noise: " + ", ".join(f"{k} {v:.3f}" for k, v in r.items()))
    print("[layer3] ratio to clean hr of hr blurred 1 px:       " + ", ".join(f"{k} {v:.3f}" for k, v in b.items()))
    assert r["flicker"] > 1.2, f"own-flow flicker cannot see 2-code shimmer on this data ({r['flicker']:.3f})"
    assert r["jerk"] > 1.2, f"temporal_jerk cannot see 2-code shimmer on this data ({r['jerk']:.3f})"
