"""Build a versioned paired dataset.

    .venv/Scripts/python -m vfx.datagen.build --version v1 [--n-train N --n-test N --n-ood N --frames T]
    .venv/Scripts/python -m vfx.datagen.build --version v2            # guided, 16 frames (VERSION_PRESETS)
    .venv/Scripts/python -m vfx.datagen.build --version vX --guided   # guided HR for any version name
    .venv/Scripts/python -m vfx.datagen.build --version v1 --plan     # size + time estimate only
    .venv/Scripts/python -m vfx.datagen.build --version v1 --reflag   # recompute card flags only

Resumable: scenario dirs that are already complete (meta.json + all npy) are skipped.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np

from vfx.core import paths
from vfx.core.fields import HR_CHANNELS, HR_RES, LR_RES, NUM_CHANNELS

from . import runner
from .guide import DEFAULT_GUIDE, GuideConfig
from .scenarios import DEFAULT_FRAMES, OOD_RANGES, TRAIN_RANGES, WARMUP_RANGE, sample_scenario

# Disjoint seed ranges per split (leakage-proof by construction).
SEED_BASE = {"train": 0, "test": 100_000, "ood": 200_000}
SEED_SPAN = 100_000
DEFAULTS = {"n_train": 500, "n_test": 32, "n_ood": 32, "frames": DEFAULT_FRAMES, "guided": False}
# Per-version defaults (CLI flags still override). v2: HR guided toward the independent LR sim,
# lr_sim in every split, 16-frame sequences; fewer scenarios to stay under the size budget.
VERSION_PRESETS = {
    "v2": {"n_train": 200, "n_test": 32, "n_ood": 32, "frames": 16, "guided": True},
}
SIZE_LIMIT_GB = 6.0

# Statistical outlier rule (same features and rule as tests/test_data_layer3.py): a scenario
# whose max density / max T / max |vel_x| exceeds median + OUTLIER_MADS * MAD over ALL scenarios
# of the version is listed in card["flagged"] as "outlier_<feature>" (kept, not dropped).
OUTLIER_MADS = 8.0
OUTLIER_MARGIN = 0.98      # flag slightly inside the rule so float rounding never un-flags


def split_seeds(n_train: int, n_test: int, n_ood: int) -> dict[str, list[int]]:
    out = {}
    for split, n in (("train", n_train), ("test", n_test), ("ood", n_ood)):
        assert 0 <= n <= SEED_SPAN
        out[split] = [SEED_BASE[split] + i for i in range(n)]
    return out


def frame_bytes(with_lr_sim: bool) -> int:
    hr = len(HR_CHANNELS) * HR_RES * HR_RES * 2
    lr = NUM_CHANNELS * LR_RES * LR_RES * 2
    return hr + lr + (lr if with_lr_sim else 0)


def estimate_bytes(n_train, n_test, n_ood, frames, guided: bool = False) -> int:
    solids = HR_RES * HR_RES + LR_RES * LR_RES
    return (n_train * (frames * frame_bytes(guided) + solids)
            + (n_test + n_ood) * (frames * frame_bytes(True) + solids))


def measure_step_ms(res: int, seeds=(0, 1, 2), warm: int = 40, n: int = 30) -> float:
    """Wall time per solver step (synchronized), averaged over a few developed train scenes."""
    import torch
    from vfx.core.fields import SIM_DT, Forces
    total = 0.0
    for seed in seeds:
        sc = sample_scenario(SEED_BASE["train"] + seed, "train")
        solver, _ = runner.get_solver(res)
        state = runner.fresh_state(solver, res, sc.obstacles)
        forces = Forces(emitters=sc.emitters, wind=sc.wind)
        for _ in range(warm):
            state = solver.step(state, forces, SIM_DT)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            state = solver.step(state, forces, SIM_DT)
        torch.cuda.synchronize()
        total += time.perf_counter() - t0
    return total * 1000.0 / (n * len(seeds))


def plan(seeds: dict, frames: int, measure: bool = True, guided: bool = False) -> dict:
    steps_hr = steps_lr = 0
    for split, ss in seeds.items():
        for s in ss:
            n = sample_scenario(s, split, frames).num_steps
            steps_hr += n
            if split != "train" or guided:
                steps_lr += n
    p = {"bytes": estimate_bytes(len(seeds["train"]), len(seeds["test"]), len(seeds["ood"]), frames, guided),
         "steps_hr": steps_hr, "steps_lr": steps_lr}
    if measure:
        p["ms_per_step_hr"] = measure_step_ms(HR_RES)
        p["ms_per_step_lr"] = measure_step_ms(LR_RES)
        # + ~25 ms/scenario for readback and np.save (guidance adds ~1 ms/step, not modelled)
        n_scn = sum(len(v) for v in seeds.values())
        p["est_seconds"] = (steps_hr * p["ms_per_step_hr"] + steps_lr * p["ms_per_step_lr"]) / 1000 + 0.025 * n_scn
    return p


# ----------------------------------------------------------------------------- flags
def _features(meta: dict) -> dict[str, float]:
    st = meta["stats"]
    vx = 1                                                     # CH["vel_x"]
    return {"max_density": st["hr"]["max"][0], "max_T": st["hr"]["max"][1],
            "max_speed": max(abs(st["lr"]["min"][vx]), abs(st["lr"]["max"][vx]))}


def compute_flags(metas: dict[str, dict]) -> tuple[dict, dict]:
    """metas: {"split/scn_x": meta}. Returns (flagged {key: [flags]}, outlier summary)."""
    flagged = {k: list(m.get("flags", [])) for k, m in metas.items() if m.get("flags")}
    keys = list(metas)
    summary = {"rule": f"x > median + {OUTLIER_MADS:g}*MAD over all scenarios (x {OUTLIER_MARGIN} margin), "
                       "features: hr max density, hr max T, lr max |vel_x|"}
    if len(keys) < 5:
        return flagged, summary
    feats = [_features(metas[k]) for k in keys]
    for name in feats[0]:
        a = np.asarray([f[name] for f in feats], dtype=np.float64)
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med))) + 1e-9
        thr = med + OUTLIER_MADS * mad * OUTLIER_MARGIN
        hits = [k for k, x in zip(keys, a) if x > thr]
        for k in hits:
            flagged.setdefault(k, []).append(f"outlier_{name}")
        summary[name] = {"median": round(med, 4), "mad": round(mad, 4), "threshold": round(thr, 4),
                         "n_flagged": len(hits)}
    return flagged, summary


def _read_metas(root: Path, seeds: dict) -> dict[str, dict]:
    out = {}
    for split in ("train", "test", "ood"):
        for seed in seeds.get(split, []):
            p = runner.scenario_dir(root, split, seed) / "meta.json"
            if p.exists():
                out[f"{split}/scn_{int(seed):06d}"] = json.loads(p.read_text())
    return out


def reflag(version: str, log=print) -> dict:
    """Recompute card['flagged'] (hard flags from meta + statistical outliers) without rebuilding."""
    root = paths.dataset_dir(version)
    card = json.loads((root / "card.json").read_text())
    seeds = json.loads((root / "splits.json").read_text())
    flagged, summary = compute_flags(_read_metas(root, seeds))
    card["flagged"] = flagged
    card["outliers"] = summary
    card["reflagged"] = _dt.datetime.now().isoformat(timespec="seconds")
    (root / "card.json").write_text(json.dumps(card, indent=1))
    log(f"[datagen] {version}: {len(flagged)} flagged scenarios -> card.json")
    return card


# ----------------------------------------------------------------------------- build
def _card(version, seeds, frames, solver_version, metas, build_seconds, root: Path, guide) -> dict:
    total = sum(f.stat().st_size for f in root.rglob("*.npy"))
    flagged, outliers = compute_flags(metas)
    card = {
        "version": version,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "solver_version": solver_version,
        "counts": {k: len(v) for k, v in seeds.items()},
        "seed_ranges": {k: [SEED_BASE[k], SEED_BASE[k] + SEED_SPAN - 1] for k in seeds},
        "frames_per_scenario": frames,
        "pairs_per_scenario": frames - 1,
        "lr_res": LR_RES, "hr_res": HR_RES,
        "param_ranges": {"train_test": TRAIN_RANGES, "ood": OOD_RANGES, "warmup_steps": WARMUP_RANGE},
        "lr_sim_splits": ["train", "test", "ood"] if guide is not None else ["test", "ood"],
        "guidance": guide.asdict() if guide is not None else None,
        "flagged": flagged,
        "outliers": outliers,
        "bytes": total,
        "build_seconds": round(build_seconds, 1),
        "format": "see vfx/datagen/README.md",
    }
    if guide is not None:
        tr = {c: [] for c in ("density", "temperature", "velocity")}
        fire = {c: [] for c in tr}
        for m in metas.values():
            for c in tr:
                tr[c].append(max(m["tracking"][c]))
                if "fire" in m["tags"]:
                    fire[c].append(max(m["tracking"][c]))
        card["tracking_rel_l2_32"] = {
            "what": "per-scenario max over frames of rel L2(avgpool4(lr), avgpool4(lr_sim)) at 32²",
            **{c: {"median": round(float(np.median(v)), 4), "p95": round(float(np.percentile(v, 95)), 4),
                   "max": round(float(np.max(v)), 4),
                   "fire_max": round(float(np.max(fire[c])), 4) if fire[c] else None} for c, v in tr.items()}}
    return card


def build(version: str, n_train: int, n_test: int, n_ood: int, frames: int,
          force_size: bool = False, guided: bool = False, guide: GuideConfig | None = None,
          log=print) -> dict:
    root = paths.dataset_dir(version)
    seeds = split_seeds(n_train, n_test, n_ood)
    guide = (guide or DEFAULT_GUIDE) if guided else None
    est = estimate_bytes(n_train, n_test, n_ood, frames, guided)
    log(f"[datagen] {version}: train={n_train} test={n_test} ood={n_ood} frames={frames} "
        f"guided={guided} -> {est / 1e9:.2f} GB at {root}")
    if est > SIZE_LIMIT_GB * 1e9 and not force_size:
        raise SystemExit(f"estimated {est / 1e9:.2f} GB > {SIZE_LIMIT_GB} GB limit (use --force-size)")
    root.mkdir(parents=True, exist_ok=True)
    (root / "splits.json").write_text(json.dumps(seeds))

    jobs = [(split, s) for split in ("train", "test", "ood") for s in seeds[split]]
    todo = [(sp, s) for sp, s in jobs
            if not runner.is_complete(runner.scenario_dir(root, sp, s), guided or sp != "train")]
    log(f"[datagen] {len(jobs) - len(todo)} complete, {len(todo)} to run")
    cost = {j: sample_scenario(j[1], j[0], frames).num_steps * (1.25 if (guided or j[0] != "train") else 1.0)
            for j in todo}
    total_cost = sum(cost.values()) or 1.0

    t0 = time.perf_counter()
    done_cost = 0.0
    for i, (split, seed) in enumerate(todo):
        sc = sample_scenario(seed, split, frames)
        meta = runner.run(sc, runner.scenario_dir(root, split, seed), frames, guide=guide)
        done_cost += cost[(split, seed)]
        el = time.perf_counter() - t0
        eta = el / done_cost * (total_cost - done_cost)
        if i < 3 or i % 10 == 0 or i == len(todo) - 1:
            tr = (f" track d={max(meta['tracking']['density']):.3f} T={max(meta['tracking']['temperature']):.3f}"
                  if guide is not None else "")
            log(f"[datagen] {i + 1}/{len(todo)} {split} scn_{seed:06d} steps={sc.num_steps} "
                f"tags={','.join(sc.tags)}{' FLAGS=' + ','.join(meta['flags']) if meta['flags'] else ''}{tr} "
                f"| {el:6.0f}s elapsed, ETA {eta:6.0f}s")

    metas = _read_metas(root, seeds)
    solver_version = next(iter(metas.values()))["solver_version"]
    card = _card(version, seeds, frames, solver_version, metas, time.perf_counter() - t0, root, guide)
    (root / "card.json").write_text(json.dumps(card, indent=1))
    log(f"[datagen] done: {card['bytes'] / 1e9:.2f} GB, {len(card['flagged'])} flagged, "
        f"{card['build_seconds']:.0f}s this session")
    return card


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="vfx.datagen.build")
    ap.add_argument("--version", required=True)
    ap.add_argument("--n-train", type=int)
    ap.add_argument("--n-test", type=int)
    ap.add_argument("--n-ood", type=int)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--guided", action="store_true", default=None,
                    help="guide the 512² run toward the independent 128² run (v2 scheme)")
    ap.add_argument("--plan", action="store_true", help="print size/time estimate and exit")
    ap.add_argument("--reflag", action="store_true", help="recompute card.json flags only, no simulation")
    ap.add_argument("--force-size", action="store_true", help=f"allow > {SIZE_LIMIT_GB} GB")
    a = ap.parse_args(argv)
    if a.reflag:
        reflag(a.version)
        return
    cfg = {**DEFAULTS, **VERSION_PRESETS.get(a.version, {})}
    for k in ("n_train", "n_test", "n_ood", "frames", "guided"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    if cfg["frames"] < 2:
        ap.error("--frames must be >= 2 (items need t >= 1)")
    if a.plan:
        p = plan(split_seeds(cfg["n_train"], cfg["n_test"], cfg["n_ood"]), cfg["frames"], guided=cfg["guided"])
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()}, indent=1))
        print(f"~{p['bytes'] / 1e9:.2f} GB, ~{p['est_seconds'] / 60:.1f} min")
        return
    build(a.version, cfg["n_train"], cfg["n_test"], cfg["n_ood"], cfg["frames"], a.force_size, cfg["guided"])


if __name__ == "__main__":
    main(sys.argv[1:])
