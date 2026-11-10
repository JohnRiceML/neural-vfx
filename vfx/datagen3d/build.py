"""Build a versioned route-B (screen-space) dataset.

    .venv/Scripts/python -m vfx.datagen3d.build --version s3d_v1            # build (resumable)
    .venv/Scripts/python -m vfx.datagen3d.build --version s3d_v1 --plan     # size + time estimate only
    .venv/Scripts/python -m vfx.datagen3d.build --version s3d_v1 --card     # rewrite card.json only

Resumable: complete scenario dirs (meta.json + all .npy) are skipped; rerun the same command.
Refuses to start if the estimate exceeds 4 GB or the data disk has < 10 GB free.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from vfx.core import paths
from vfx.core.fields3d import HR_RES_3D, LR_RES_3D, SIM_DT

from . import render as R
from . import runner, sim
from .guide3d import DEFAULT_GUIDE
from .scenarios import DEFAULT_FRAMES, KIND_PROBS, KINDS, OOD_RANGES, RANGES, WARMUP_RANGE, sample_scenario

SEED_BASE = {"train": 0, "test": 100_000, "ood": 200_000}
SEED_SPAN = 100_000
VERSION_PRESETS = {
    "s3d_v1": {"n_train": 96, "n_test": 12, "n_ood": 12, "frames": DEFAULT_FRAMES, "cams": 2},
}
DEFAULTS = {"n_train": 8, "n_test": 2, "n_ood": 2, "frames": DEFAULT_FRAMES, "cams": runner.DEFAULT_CAMS}
SIZE_LIMIT_GB = 4.0
MIN_FREE_GB = 10.0
BYTES_PER_PX = 3 + 3 + 8 + 1          # lr, hr uint8 RGB; aux 4 x fp16; hr_alpha uint8


def split_seeds(n_train, n_test, n_ood) -> dict:
    return {sp: [SEED_BASE[sp] + i for i in range(n)] for sp, n in
            (("train", n_train), ("test", n_test), ("ood", n_ood))}


def estimate_bytes(n_scn: int, frames: int, cams: int) -> int:
    return n_scn * (cams * frames * R.CROP * R.CROP * BYTES_PER_PX + 4 * 128)


def free_gb(root: Path) -> float:
    p = root
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def measure_step_ms(n: int = 30) -> float:
    """Wall ms per paired step (64^3 runtime + guided 256^3 + guidance), after a short warm-up."""
    import torch
    sc = sample_scenario(0, "train")
    t = None
    for s, *_ in sim.run_pair(sc, DEFAULT_GUIDE, steps=20 + n):
        if s == 19:
            torch.cuda.synchronize()
            t = time.perf_counter()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1000.0 / n


def plan(cfg: dict, measure: bool = True) -> dict:
    seeds = split_seeds(cfg["n_train"], cfg["n_test"], cfg["n_ood"])
    steps = sum(sample_scenario(s, sp, cfg["frames"]).num_steps for sp, ss in seeds.items() for s in ss)
    n = sum(len(v) for v in seeds.values())
    p = {"scenarios": n, "sequences": n * cfg["cams"], "steps": steps,
         "bytes": estimate_bytes(n, cfg["frames"], cfg["cams"])}
    if measure:
        p["ms_per_step"] = measure_step_ms()
        # + rendering: 2 x cams full 1440p draws per saved frame (~3 ms each incl. crop/pool) + ~1 s/scn I/O
        p["est_seconds"] = steps * p["ms_per_step"] / 1000 + n * (cfg["frames"] * cfg["cams"] * 2 * 0.003 + 1.0)
    return p


def _card(version, cfg, seeds, root: Path, build_seconds: float) -> dict:
    metas = {}
    for sp, ss in seeds.items():
        for s in ss:
            p = runner.scenario_dir(root, sp, s) / "meta.json"
            if p.exists():
                metas[f"{sp}/scn_{s:06d}"] = json.loads(p.read_text())
    total = sum(f.stat().st_size for f in root.rglob("*.npy"))
    tr = np.asarray([m["tracking_16"]["max_after_step10"] for m in metas.values()]) if metas else np.zeros((0, 3))
    trf = np.asarray([np.max(m["tracking_16"]["frames"], 0) for m in metas.values()]) if metas else np.zeros((0, 3))
    moving = [f"{k}/cam{j}" for k, m in metas.items() for j, c in enumerate(m["cams"]) if c["moving"]]
    kinds = {}
    for m in metas.values():
        kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
    r = runner.get_renderer() if runner._renderer is not None else None
    summ = lambda a: {"median": round(float(np.median(a)), 4), "p95": round(float(np.percentile(a, 95)), 4),
                      "max": round(float(np.max(a)), 4)} if len(a) else None
    return {
        "version": version,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "solver_version": sim.solver_version(),
        "counts": {k: len(v) for k, v in seeds.items()},
        "complete": len(metas),
        "sequences_per_scenario": cfg["cams"], "frames_per_sequence": cfg["frames"],
        "seed_ranges": {k: [SEED_BASE[k], SEED_BASE[k] + SEED_SPAN - 1] for k in seeds},
        "lr_res": LR_RES_3D, "hr_res": HR_RES_3D, "dt": SIM_DT,
        "lr_solver": "Solver3D(64, runtime_config(**scenario.sim))  (= sandbox)",
        "hr_solver": "Solver3D(256, SolverConfig3D(**scenario.sim)) guided toward the LR run",
        "guidance": DEFAULT_GUIDE.asdict(),
        "tracking_rel_l2_16": {
            "what": "rel L2 at 16^3 of guided HR vs LR; per scenario max over all steps >= 10 (incl. warm-up) "
                    "and max over the saved frames; columns density, T-ambient, velocity",
            "all_steps": {c: summ(tr[:, j]) for j, c in enumerate(("density", "temperature", "velocity"))},
            "saved_frames": {c: summ(trf[:, j]) for j, c in enumerate(("density", "temperature", "velocity"))},
        },
        "param_ranges": {"kinds": dict(zip(KINDS, KIND_PROBS)), "train_test": RANGES, "ood": OOD_RANGES,
                         "warmup_steps": WARMUP_RANGE, "cameras": R.CAM_RANGES},
        "kinds": kinds,
        "renderer": R.renderer_settings(r.r) if r is not None else None,
        "storage": {
            "lr": "(K,T,3,256,256) uint8 sRGB, render of raw 64^3, 2x2-avg-pooled crop of the 1440p frame",
            "hr": "(K,T,3,256,256) uint8, render of guided 256^3, identical camera/crop/settings",
            "aux": "(K,T,4,256,256) fp16: alpha (0..1), depth (m, alpha-weighted), motion x,y in STORED px "
                   "per SIM_DT (= renderer output px / 2), alpha-weighted pooled; 0 where alpha = 0",
            "hr_alpha": "(K,T,256,256) uint8: HR render alpha * 255",
        },
        "moving_camera_sequences": moving,
        "bytes": total,
        "build_seconds": round(build_seconds, 1),
        "format": "see vfx/datagen3d/README.md",
    }


def build(version: str, cfg: dict, force: bool = False, log=print) -> dict:
    root = paths.dataset_dir(version)
    seeds = split_seeds(cfg["n_train"], cfg["n_test"], cfg["n_ood"])
    n = sum(len(v) for v in seeds.values())
    est = estimate_bytes(n, cfg["frames"], cfg["cams"])
    fg = free_gb(root)
    log(f"[datagen3d] {version}: {cfg} -> {n} scenarios, {n * cfg['cams']} sequences, "
        f"~{est / 1e9:.2f} GB at {root}; {fg:.1f} GB free")
    if est > SIZE_LIMIT_GB * 1e9:
        raise SystemExit(f"estimated {est / 1e9:.2f} GB > {SIZE_LIMIT_GB} GB limit")
    if fg < MIN_FREE_GB and not force:
        raise SystemExit(f"only {fg:.1f} GB free on the data disk (< {MIN_FREE_GB} GB): refusing to build")
    root.mkdir(parents=True, exist_ok=True)
    (root / "splits.json").write_text(json.dumps(seeds))
    jobs = [(sp, s) for sp in ("train", "test", "ood") for s in seeds[sp]]
    todo = [(sp, s) for sp, s in jobs if not runner.is_complete(runner.scenario_dir(root, sp, s))]
    log(f"[datagen3d] {len(jobs) - len(todo)} complete, {len(todo)} to run")
    cost = {j: sample_scenario(j[1], j[0], cfg["frames"]).num_steps + 25 for j in todo}
    total = sum(cost.values()) or 1
    t0 = time.perf_counter()
    done = 0
    for i, (sp, s) in enumerate(todo):
        sc = sample_scenario(s, sp, cfg["frames"])
        m = runner.run(sc, runner.scenario_dir(root, sp, s), n_cams=cfg["cams"])
        done += cost[(sp, s)]
        el = time.perf_counter() - t0
        tr = m["tracking_16"]["max_after_step10"]
        log(f"[datagen3d] {i + 1}/{len(todo)} {sp} scn_{s:06d} {sc.kind} steps={sc.num_steps} "
            f"track16 d={tr[0]:.3f} T={tr[1]:.3f} u={tr[2]:.3f} {'' if m['finite'] else 'NONFINITE '}"
            f"moving={[c['moving'] for c in m['cams']]} {m['seconds']['total']:.1f}s | "
            f"{el:6.0f}s elapsed, ETA {el / done * (total - done):6.0f}s", flush=True)
        if free_gb(root) < 2.0:
            raise SystemExit("data disk below 2 GB free: stopping (resumable)")
    card = _card(version, cfg, seeds, root, time.perf_counter() - t0)
    (root / "card.json").write_text(json.dumps(card, indent=1))
    log(f"[datagen3d] done: {card['complete']} scenarios, {card['bytes'] / 1e9:.2f} GB, "
        f"{card['build_seconds']:.0f}s this session")
    if card["bytes"] > SIZE_LIMIT_GB * 1e9:
        log(f"[datagen3d] WARNING: {card['bytes'] / 1e9:.2f} GB exceeds the {SIZE_LIMIT_GB} GB budget")
    return card


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="vfx.datagen3d.build")
    ap.add_argument("--version", required=True)
    for k in ("n-train", "n-test", "n-ood", "frames", "cams"):
        ap.add_argument(f"--{k}", type=int)
    ap.add_argument("--plan", action="store_true", help="print size/time estimate and exit")
    ap.add_argument("--card", action="store_true", help="rewrite card.json from the meta.json files")
    ap.add_argument("--force", action="store_true", help=f"build even with < {MIN_FREE_GB} GB free")
    a = ap.parse_args(argv)
    cfg = {**DEFAULTS, **VERSION_PRESETS.get(a.version, {})}
    for k in ("n_train", "n_test", "n_ood", "frames", "cams"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    if cfg["frames"] < 8:
        ap.error("--frames must be >= 8")
    if a.plan:
        p = plan(cfg)
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in p.items()}, indent=1))
        print(f"~{p['bytes'] / 1e9:.2f} GB, ~{p['est_seconds'] / 60:.1f} min, "
              f"{free_gb(paths.dataset_dir(a.version)):.1f} GB free")
        return
    if a.card:
        root = paths.dataset_dir(a.version)
        seeds = json.loads((root / "splits.json").read_text())
        card = _card(a.version, cfg, seeds, root, 0.0)
        old = root / "card.json"
        if old.exists():
            card["build_seconds"] = json.loads(old.read_text()).get("build_seconds", 0.0)
        old.write_text(json.dumps(card, indent=1))
        return
    build(a.version, cfg, a.force)


if __name__ == "__main__":
    main(sys.argv[1:])
