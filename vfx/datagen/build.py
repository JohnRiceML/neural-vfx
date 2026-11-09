"""Build a versioned paired dataset.

    .venv/Scripts/python -m vfx.datagen.build --version v1 [--n-train N --n-test N --n-ood N --frames T]
    .venv/Scripts/python -m vfx.datagen.build --version v1 --plan     # size + time estimate only

Resumable: scenario dirs that are already complete (meta.json + all npy) are skipped.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

from vfx.core import paths
from vfx.core.fields import HR_CHANNELS, HR_RES, LR_RES, NUM_CHANNELS

from . import runner
from .scenarios import DEFAULT_FRAMES, OOD_RANGES, TRAIN_RANGES, WARMUP_RANGE, sample_scenario

# Disjoint seed ranges per split (leakage-proof by construction).
SEED_BASE = {"train": 0, "test": 100_000, "ood": 200_000}
SEED_SPAN = 100_000
DEFAULTS = {"n_train": 500, "n_test": 32, "n_ood": 32, "frames": DEFAULT_FRAMES}
SIZE_LIMIT_GB = 6.0


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


def estimate_bytes(n_train, n_test, n_ood, frames) -> int:
    solids = HR_RES * HR_RES + LR_RES * LR_RES
    return (n_train * (frames * frame_bytes(False) + solids)
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


def plan(seeds: dict, frames: int, measure: bool = True) -> dict:
    steps_hr = steps_lr = 0
    for split, ss in seeds.items():
        for s in ss:
            n = sample_scenario(s, split, frames).num_steps
            steps_hr += n
            if split != "train":
                steps_lr += n
    p = {"bytes": estimate_bytes(len(seeds["train"]), len(seeds["test"]), len(seeds["ood"]), frames),
         "steps_hr": steps_hr, "steps_lr": steps_lr}
    if measure:
        p["ms_per_step_hr"] = measure_step_ms(HR_RES)
        p["ms_per_step_lr"] = measure_step_ms(LR_RES)
        # + ~25 ms/scenario for readback and np.save
        n_scn = sum(len(v) for v in seeds.values())
        p["est_seconds"] = (steps_hr * p["ms_per_step_hr"] + steps_lr * p["ms_per_step_lr"]) / 1000 + 0.025 * n_scn
    return p


def _card(version, seeds, frames, solver_version, flagged, build_seconds, root: Path) -> dict:
    total = sum(f.stat().st_size for f in root.rglob("*.npy"))
    return {
        "version": version,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "solver_version": solver_version,
        "counts": {k: len(v) for k, v in seeds.items()},
        "seed_ranges": {k: [SEED_BASE[k], SEED_BASE[k] + SEED_SPAN - 1] for k in seeds},
        "frames_per_scenario": frames,
        "pairs_per_scenario": frames - 1,
        "lr_res": LR_RES, "hr_res": HR_RES,
        "param_ranges": {"train_test": TRAIN_RANGES, "ood": OOD_RANGES, "warmup_steps": WARMUP_RANGE},
        "flagged": flagged,
        "bytes": total,
        "build_seconds": round(build_seconds, 1),
        "format": "see vfx/datagen/README.md",
    }


def build(version: str, n_train: int, n_test: int, n_ood: int, frames: int,
          force_size: bool = False, log=print) -> dict:
    root = paths.dataset_dir(version)
    seeds = split_seeds(n_train, n_test, n_ood)
    est = estimate_bytes(n_train, n_test, n_ood, frames)
    log(f"[datagen] {version}: train={n_train} test={n_test} ood={n_ood} frames={frames} "
        f"-> {est / 1e9:.2f} GB at {root}")
    if est > SIZE_LIMIT_GB * 1e9 and not force_size:
        raise SystemExit(f"estimated {est / 1e9:.2f} GB > {SIZE_LIMIT_GB} GB limit (use --force-size)")
    root.mkdir(parents=True, exist_ok=True)
    (root / "splits.json").write_text(json.dumps(seeds))

    jobs = [(split, s) for split in ("train", "test", "ood") for s in seeds[split]]
    todo = [(sp, s) for sp, s in jobs if not runner.is_complete(runner.scenario_dir(root, sp, s), sp != "train")]
    log(f"[datagen] {len(jobs) - len(todo)} complete, {len(todo)} to run")
    cost = {j: sample_scenario(j[1], j[0], frames).num_steps * (1.25 if j[0] != "train" else 1.0) for j in todo}
    total_cost = sum(cost.values()) or 1.0

    t0 = time.perf_counter()
    done_cost = 0.0
    solver_version = None
    for i, (split, seed) in enumerate(todo):
        sc = sample_scenario(seed, split, frames)
        meta = runner.run(sc, runner.scenario_dir(root, split, seed), frames)
        solver_version = meta["solver_version"]
        done_cost += cost[(split, seed)]
        el = time.perf_counter() - t0
        eta = el / done_cost * (total_cost - done_cost)
        if i < 3 or i % 10 == 0 or i == len(todo) - 1:
            log(f"[datagen] {i + 1}/{len(todo)} {split} scn_{seed:06d} steps={sc.num_steps} "
                f"tags={','.join(sc.tags)}{' FLAGS=' + ','.join(meta['flags']) if meta['flags'] else ''} "
                f"| {el:6.0f}s elapsed, ETA {eta:6.0f}s")

    flagged = {}
    for split, seed in jobs:
        m = json.loads((runner.scenario_dir(root, split, seed) / "meta.json").read_text())
        solver_version = solver_version or m["solver_version"]
        if m["flags"]:
            flagged[f"{split}/scn_{seed:06d}"] = m["flags"]
    card = _card(version, seeds, frames, solver_version, flagged, time.perf_counter() - t0, root)
    (root / "card.json").write_text(json.dumps(card, indent=1))
    log(f"[datagen] done: {card['bytes'] / 1e9:.2f} GB, {len(flagged)} flagged, "
        f"{card['build_seconds']:.0f}s this session")
    return card


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="vfx.datagen.build")
    ap.add_argument("--version", required=True)
    ap.add_argument("--n-train", type=int, default=DEFAULTS["n_train"])
    ap.add_argument("--n-test", type=int, default=DEFAULTS["n_test"])
    ap.add_argument("--n-ood", type=int, default=DEFAULTS["n_ood"])
    ap.add_argument("--frames", type=int, default=DEFAULTS["frames"])
    ap.add_argument("--plan", action="store_true", help="print size/time estimate and exit")
    ap.add_argument("--force-size", action="store_true", help=f"allow > {SIZE_LIMIT_GB} GB")
    a = ap.parse_args(argv)
    if a.frames < 2:
        ap.error("--frames must be >= 2 (items need t >= 1)")
    if a.plan:
        p = plan(split_seeds(a.n_train, a.n_test, a.n_ood), a.frames)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()}, indent=1))
        print(f"~{p['bytes'] / 1e9:.2f} GB, ~{p['est_seconds'] / 60:.1f} min")
        return
    build(a.version, a.n_train, a.n_test, a.n_ood, a.frames, a.force_size)


if __name__ == "__main__":
    main(sys.argv[1:])
