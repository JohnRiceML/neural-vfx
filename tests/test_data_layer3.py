"""Layer 3: data integrity, on dataset version $VFX_DATASET (default "v1").

Checks the on-disk format from CLAUDE.md "Dataset format (v0)" directly with numpy (independent of
the loader), then the vfx.datagen.load_dataset contract. Skips if the dataset is not built yet.
"""
from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vfx.core.fields import AMBIENT_TEMP, CH, HR_RES, LR_RES, NUM_CHANNELS, Obstacle, rasterize_obstacles
from vfx.core.paths import dataset_dir
from tests.helpers import dataset_version, import_or_skip

VERSION = dataset_version()
ROOT = dataset_dir(VERSION)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SPLITS = ("train", "test", "ood")

pytestmark = pytest.mark.skipif(not (ROOT / "splits.json").exists(),
                                reason=f"dataset '{VERSION}' not built at {ROOT}")

# Physical ranges (independent of the generator's own flagging thresholds)
RANGES = {
    "density": (-1e-2, 20.0),
    "vel": (-30.0, 30.0),                    # m/s, 1 m box
    "temperature": (AMBIENT_TEMP - 50.0, 5000.0),
    "fuel": (-1e-2, 50.0),
}
MAX_BYTES = 6.5e9                            # "Whole dataset must stay under ~6 GB"


def _load_json(p):
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def card():
    p = ROOT / "card.json"
    assert p.exists(), "card.json missing"
    return _load_json(p)


@pytest.fixture(scope="module")
def splits():
    return _load_json(ROOT / "splits.json")


def scn_dir(split, seed):
    return ROOT / split / f"scn_{int(seed):06d}"


def _stats(arr: torch.Tensor):
    """per-channel (nonfinite, min, max, mean) of a (T, C, H, W) float tensor."""
    a = arr.float()
    nonfinite = (~torch.isfinite(a)).sum(dim=(0, 2, 3))
    a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return dict(nonfinite=nonfinite.tolist(), min=a.amin(dim=(0, 2, 3)).tolist(),
                max=a.amax(dim=(0, 2, 3)).tolist(), mean=a.mean(dim=(0, 2, 3)).tolist())


@pytest.fixture(scope="module")
def scan(splits):
    """One pass over every scenario: shapes, dtypes, stats, fingerprints. Problems are collected."""
    out = {"scenarios": {}, "problems": [], "fingerprints": {s: [] for s in SPLITS}}
    for split in SPLITS:
        for seed in splits.get(split, []):
            d = scn_dir(split, seed)
            key = f"{split}/{d.name}"
            rec = {"split": split, "seed": int(seed), "dir": d}
            if not (d / "meta.json").exists():
                out["problems"].append(f"{key}: missing meta.json")
                continue
            rec["meta"] = _load_json(d / "meta.json")
            arrs = {}
            for name in ("lr", "hr", "solid_lr", "solid_hr", "lr_sim"):
                p = d / f"{name}.npy"
                arrs[name] = np.load(p, mmap_mode="r") if p.exists() else None
            rec["shapes"] = {k: (None if v is None else (tuple(v.shape), str(v.dtype))) for k, v in arrs.items()}
            if arrs["lr"] is not None and arrs["hr"] is not None:
                lr = torch.from_numpy(np.array(arrs["lr"])).to(DEV)
                hr = torch.from_numpy(np.array(arrs["hr"])).to(DEV)
                rec["lr_stats"] = _stats(lr)
                rec["hr_stats"] = _stats(hr)
                # channel-order / pooling consistency: lr density,temperature == avgpool(hr)
                pooled = F.avg_pool2d(hr.float(), UPS := hr.shape[-1] // lr.shape[-1])
                for i, c in enumerate(("density", "temperature")):
                    ref = lr[:, CH[c]].float()
                    err = (pooled[:, i] - ref).abs().max().item()
                    scale = ref.abs().max().item() + 1e-6
                    rec[f"pool_err_{c}"] = err / scale
                # fingerprint: LR density at 32^2 per frame (for near-duplicate search)
                fp = F.adaptive_avg_pool2d(lr[:, CH["density"]].float()[:, None], 32)[:, 0].flatten(1)
                out["fingerprints"][split].append((key, fp))
                del lr, hr
            if arrs["lr_sim"] is not None:
                rec["lr_sim_stats"] = _stats(torch.from_numpy(np.array(arrs["lr_sim"])).to(DEV))
            if arrs["solid_lr"] is not None:
                rec["solid_lr"] = np.asarray(arrs["solid_lr"])
            if arrs["solid_hr"] is not None:
                rec["solid_hr"] = np.asarray(arrs["solid_hr"])
            out["scenarios"][key] = rec
    return out


def _flagged(card) -> set[str]:
    fl = card.get("flagged", {}) or {}
    return set(fl.keys()) if isinstance(fl, dict) else set(fl)


# ----------------------------------------------------------------------------- card / splits
def test_card_and_counts(card, splits):
    for k in ("version", "counts", "solver_version", "created"):
        assert k in card, f"card.json missing '{k}'"
    assert card["version"] == VERSION
    for s in SPLITS:
        n = len(splits.get(s, []))
        assert card["counts"].get(s) == n, f"card count for {s} = {card['counts'].get(s)}, splits.json has {n}"
        dirs = [p for p in (ROOT / s).glob("scn_*")] if (ROOT / s).exists() else []
        assert len(dirs) >= n, f"{s}: {len(dirs)} scenario dirs on disk < {n} in splits.json"


def test_all_scenarios_present(scan):
    assert not scan["problems"], "\n".join(scan["problems"])


def test_dataset_size_budget():
    total = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
    print(f"dataset {VERSION}: {total / 1e9:.2f} GB")
    assert total < MAX_BYTES, f"{total / 1e9:.2f} GB > ~6 GB budget"


@pytest.mark.xfail(strict=False, reason="M2 gate (SPEC): >= 500 train scenarios + OOD set; informational for v0")
def test_m2_gate_counts(splits):
    n_train, n_ood = len(splits.get("train", [])), len(splits.get("ood", []))
    print(f"M2 gate: train {n_train} (need >= 500), ood {n_ood} (need > 0)")
    assert n_train >= 500 and n_ood > 0


# ----------------------------------------------------------------------------- schema
def test_shapes_dtypes(scan):
    bad = []
    for key, r in scan["scenarios"].items():
        sh = r["shapes"]
        if sh["lr"] is None or sh["hr"] is None:
            bad.append(f"{key}: lr/hr missing")
            continue
        (ls, ld), (hs, hd) = sh["lr"], sh["hr"]
        T = ls[0]
        if ls != (T, NUM_CHANNELS, LR_RES, LR_RES) or ld != "float16":
            bad.append(f"{key}: lr {ls} {ld}")
        if hs != (T, 2, HR_RES, HR_RES) or hd != "float16":
            bad.append(f"{key}: hr {hs} {hd} (T must match lr={T})")
        if T < 2:
            bad.append(f"{key}: only {T} frames")
        if sh["solid_lr"] != ((LR_RES, LR_RES), "uint8"):
            bad.append(f"{key}: solid_lr {sh['solid_lr']}")
        if sh["solid_hr"] != ((HR_RES, HR_RES), "uint8"):
            bad.append(f"{key}: solid_hr {sh['solid_hr']}")
        if r["split"] in ("test", "ood"):
            if sh["lr_sim"] != ((T, NUM_CHANNELS, LR_RES, LR_RES), "float16"):
                bad.append(f"{key}: lr_sim {sh['lr_sim']} (required for test/ood)")
    assert not bad, "\n".join(bad[:50])


def test_meta_fields(scan, card):
    bad = []
    for key, r in scan["scenarios"].items():
        m = r["meta"]
        for k in ("scenario", "solver_version", "lr_res", "hr_res", "dt", "start_step", "tags"):
            if k not in m:
                bad.append(f"{key}: meta missing {k}")
        if m.get("lr_res") != LR_RES or m.get("hr_res") != HR_RES:
            bad.append(f"{key}: lr_res/hr_res {m.get('lr_res')}/{m.get('hr_res')}")
        if m.get("scenario", {}).get("seed") != r["seed"]:
            bad.append(f"{key}: meta seed {m.get('scenario', {}).get('seed')} != dir seed {r['seed']}")
        if m.get("solver_version") != card.get("solver_version"):
            bad.append(f"{key}: solver_version {m.get('solver_version')} != card {card.get('solver_version')}")
        if "channels" in m and tuple(m["channels"]) != ("density", "vel_x", "vel_y", "temperature", "fuel"):
            bad.append(f"{key}: channels {m['channels']}")
    assert not bad, "\n".join(bad[:50])


def test_solver_version_matches_current_solver(card):
    sim = import_or_skip("vfx.sim", ("SOLVER_VERSION",))
    assert card["solver_version"] == sim.SOLVER_VERSION, (
        f"dataset built with solver {card['solver_version']}, current solver is {sim.SOLVER_VERSION}: "
        "the network would be trained on a simulator the runtime does not run")


def test_no_nan_inf(scan):
    bad = []
    for key, r in scan["scenarios"].items():
        for arr in ("lr_stats", "hr_stats", "lr_sim_stats"):
            if arr in r and any(r[arr]["nonfinite"]):
                bad.append(f"{key}: {arr[:-6]} non-finite per channel {r[arr]['nonfinite']}")
    assert not bad, "\n".join(bad)


def test_channel_order_lr_is_pooled_hr(scan):
    """lr density/temperature (channels 0 and 3) must equal 4x avg-pool of hr (channels 0, 1)."""
    bad = []
    for key, r in scan["scenarios"].items():
        for c in ("density", "temperature"):
            e = r.get(f"pool_err_{c}")
            if e is not None and e > 2e-2:                 # float16 storage tolerance (relative to max)
                bad.append(f"{key}: lr {c} vs avgpool(hr) rel err {e:.3e}")
    assert not bad, "\n".join(bad[:50])


def test_solids_match_obstacles(scan):
    bad = []
    for key, r in scan["scenarios"].items():
        obs = [Obstacle(kind=o["kind"], pos=tuple(o["pos"]), size=tuple(o["size"]))
               for o in r["meta"].get("scenario", {}).get("obstacles", [])]
        if "solid_lr" in r:
            exp = rasterize_obstacles(obs, LR_RES, "cpu").numpy().astype(np.uint8)
            if not np.array_equal(r["solid_lr"], exp):
                bad.append(f"{key}: solid_lr != rasterize_obstacles(meta obstacles), "
                           f"{int((r['solid_lr'] != exp).sum())} cells differ")
        if "solid_hr" in r:
            exp = rasterize_obstacles(obs, HR_RES, "cpu").numpy().astype(np.uint8)
            diff = int((r["solid_hr"] != exp).sum())
            if diff > 0.01 * max(int(exp.sum()), 1):
                bad.append(f"{key}: solid_hr differs from obstacles in {diff} cells")
        for s in ("solid_lr", "solid_hr"):
            if s in r and not set(np.unique(r[s]).tolist()) <= {0, 1}:
                bad.append(f"{key}: {s} not binary")
    assert not bad, "\n".join(bad[:50])


# ----------------------------------------------------------------------------- statistics
def _range_violations(r):
    v = []
    for arr in ("lr_stats", "lr_sim_stats"):
        st = r.get(arr)
        if not st:
            continue
        for name, idxs in (("density", [CH["density"]]), ("vel", [CH["vel_x"], CH["vel_y"]]),
                           ("temperature", [CH["temperature"]]), ("fuel", [CH["fuel"]])):
            lo, hi = RANGES[name]
            for i in idxs:
                if st["min"][i] < lo or st["max"][i] > hi:
                    v.append(f"{arr[:-6]}.{name} [{st['min'][i]:.3g}, {st['max'][i]:.3g}] not in [{lo}, {hi}]")
    st = r.get("hr_stats")
    if st:
        for i, name in enumerate(("density", "temperature")):
            lo, hi = RANGES[name]
            if st["min"][i] < lo or st["max"][i] > hi:
                v.append(f"hr.{name} [{st['min'][i]:.3g}, {st['max'][i]:.3g}] not in [{lo}, {hi}]")
        if st["max"][0] < 1e-3:
            v.append("hr density empty (max < 1e-3)")
    return v


def test_physical_ranges_or_flagged(scan, card):
    """Out-of-range or empty scenarios must be listed in card['flagged'] (flagged, not silently kept)."""
    flagged = _flagged(card)
    unflagged, flagged_seen = [], []
    for key, r in scan["scenarios"].items():
        v = _range_violations(r)
        if v:
            (flagged_seen if key in flagged else unflagged).append(f"{key}: " + "; ".join(v))
    if flagged_seen:
        print("flagged (kept, listed in card):\n  " + "\n  ".join(flagged_seen))
    assert not unflagged, "out-of-range scenarios NOT flagged in card.json:\n" + "\n".join(unflagged[:50])


def test_statistical_outliers_report(scan, card):
    """Robust outlier report on per-scenario max density / max |vel| / max T (median + 8*MAD).
    Prints the table; fails only if a statistical outlier is also unflagged AND > 3x the median max."""
    keys = list(scan["scenarios"])
    if len(keys) < 5:
        pytest.skip("too few scenarios for outlier statistics")
    feats = {
        "max_density": [scan["scenarios"][k]["hr_stats"]["max"][0] for k in keys],
        "max_T": [scan["scenarios"][k]["hr_stats"]["max"][1] for k in keys],
        "max_speed": [max(abs(scan["scenarios"][k]["lr_stats"]["min"][i]), abs(scan["scenarios"][k]["lr_stats"]["max"][i]))
                      for k in keys for i in [CH["vel_x"]]],
    }
    flagged = _flagged(card)
    rows, gross = [], []
    for name, vals in feats.items():
        a = np.asarray(vals)
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med))) + 1e-9
        print(f"{name}: median {med:.3g}, MAD {mad:.3g}, range [{a.min():.3g}, {a.max():.3g}]")
        for k, x in zip(keys, a):
            if x > med + 8 * mad:
                rows.append(f"{k}: {name} {x:.3g} (median {med:.3g})" + (" [flagged]" if k in flagged else ""))
                if x > 3 * med and k not in flagged and med > 0:
                    gross.append(rows[-1])
    if rows:
        msg = "statistical outliers:\n  " + "\n  ".join(rows)
        print(msg)
        warnings.warn(msg)
    assert not gross, "gross unflagged outliers:\n" + "\n".join(gross)


def test_density_zero_inside_solids(scan, splits):
    bad = []
    for key, r in scan["scenarios"].items():
        if "solid_hr" not in r or not r["solid_hr"].any():
            continue
        hr = np.load(r["dir"] / "hr.npy", mmap_mode="r")
        solid = torch.from_numpy(r["solid_hr"]).bool().to(DEV)
        deep = (-F.max_pool2d(-solid.float()[None, None], 5, 1, 2)[0, 0]) > 0.5   # 2 cells inside
        if not deep.any():
            continue
        d = torch.from_numpy(np.array(hr[:, 0])).float().to(DEV)
        m = d[:, deep].abs().max().item()
        if m > 1e-2:
            bad.append(f"{key}: density {m:.3g} inside obstacle")
    assert not bad, "\n".join(bad[:50])


def test_frames_are_consecutive_not_frozen(scan):
    bad = []
    for key, r in scan["scenarios"].items():
        hr = np.load(r["dir"] / "hr.npy", mmap_mode="r")
        d = torch.from_numpy(np.array(hr[:, 0])).float().to(DEV)
        if d.max().item() < 1e-3:
            continue
        same = [(t) for t in range(1, d.shape[0]) if torch.equal(d[t], d[t - 1])]
        if same:
            bad.append(f"{key}: frames {same} identical to previous frame")
    assert not bad, "\n".join(bad[:50])


# ----------------------------------------------------------------------------- splits / leakage
def test_split_seeds_disjoint(splits):
    sets = {s: set(map(int, splits.get(s, []))) for s in SPLITS}
    for i, a in enumerate(SPLITS):
        assert len(sets[a]) == len(splits.get(a, [])), f"duplicate seeds within {a}"
        for b in SPLITS[i + 1:]:
            inter = sets[a] & sets[b]
            assert not inter, f"seeds in both {a} and {b}: {sorted(inter)[:10]}"


def test_no_near_duplicate_frames_across_splits(scan):
    """No test/ood frame may be a near copy of a train frame (relative L2 on 32^2 LR density < 2%)."""
    train = scan["fingerprints"]["train"]
    if not train:
        pytest.skip("no train scenarios")
    tr = torch.cat([fp for _, fp in train])
    tr_keys = [k for k, fp in train for _ in range(fp.shape[0])]
    tr_norm = tr.norm(dim=1)
    hits = []
    for split in ("test", "ood"):
        for key, fp in scan["fingerprints"][split]:
            n = fp.norm(dim=1)
            live = n > 1e-3 * math.sqrt(fp.shape[1])           # ignore (near-)empty frames
            if not live.any():
                continue
            dist = torch.cdist(fp[live], tr)                       # (t, N_train_frames)
            rel = dist / (n[live][:, None] + tr_norm[None, :] + 1e-12)
            best, idx = rel.min(dim=1)
            for b, i in zip(best.tolist(), idx.tolist()):
                if b < 0.02:
                    hits.append(f"{key} ~ {tr_keys[i]} (rel dist {b:.4f})")
    assert not hits, f"{len(hits)} near-duplicate frames across splits:\n" + "\n".join(hits[:30])


def test_ood_has_unseen_scene_types(scan, splits):
    if not splits.get("ood"):
        pytest.fail("no OOD split (SPEC Layer 3: held-out scene types never seen in training)")
    ignore = set(SPLITS) | {"early", "late"}
    train_tags = set()
    for key, r in scan["scenarios"].items():
        if r["split"] == "train":
            train_tags |= set(r["meta"].get("tags", []))
    no_new = []
    new_tags = set()
    for key, r in scan["scenarios"].items():
        if r["split"] != "ood":
            continue
        new = set(r["meta"].get("tags", [])) - train_tags - ignore
        new_tags |= new
        if not new:
            no_new.append(f"{key}: tags {r['meta'].get('tags')}")
    print(f"OOD-only tags: {sorted(new_tags)}")
    assert new_tags, "OOD split has no tags absent from train"
    assert not no_new, "OOD scenarios with no unseen scene type:\n" + "\n".join(no_new)


# ----------------------------------------------------------------------------- loader contract
def test_load_dataset_contract(splits):
    dg = import_or_skip("vfx.datagen", ("load_dataset",))
    try:
        ds = dg.load_dataset(VERSION, "train")
    except TypeError:
        ds = dg.load_dataset(VERSION)
    assert len(ds) > 0
    a = ds[0]
    for k, shape in (("lr", (NUM_CHANNELS, LR_RES, LR_RES)), ("hr_prev", (2, HR_RES, HR_RES)),
                     ("hr", (2, HR_RES, HR_RES)), ("solid_lr", (LR_RES, LR_RES)), ("solid_hr", (HR_RES, HR_RES))):
        assert k in a, f"item missing '{k}'"
        assert tuple(a[k].shape) == shape, f"{k} shape {tuple(a[k].shape)} != {shape}"
        assert a[k].dtype == torch.float32, f"{k} dtype {a[k].dtype}"
    assert "seed" in a and "t" in a
    seed, t = int(a["seed"]), int(a["t"])
    split = next(s for s in SPLITS if seed in set(map(int, splits.get(s, []))))
    d = scn_dir(split, seed)
    lr = np.load(d / "lr.npy", mmap_mode="r")
    hr = np.load(d / "hr.npy", mmap_mode="r")
    assert torch.equal(a["lr"], torch.from_numpy(np.asarray(lr[t], dtype=np.float32))), "item lr != lr.npy[t]"
    assert torch.equal(a["hr"], torch.from_numpy(np.asarray(hr[t], dtype=np.float32))), "item hr != hr.npy[t]"
    assert torch.equal(a["hr_prev"], torch.from_numpy(np.asarray(hr[t - 1], dtype=np.float32))), \
        "hr_prev must be hr.npy[t-1] (one SIM_DT earlier)"
