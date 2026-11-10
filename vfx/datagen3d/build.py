"""Build a versioned route-B (screen-space) dataset.

    .venv/Scripts/python -m vfx.datagen3d.build --version s3d_v1            # build (resumable)
    .venv/Scripts/python -m vfx.datagen3d.build --version s3d_v1 --plan     # size + time estimate only
    .venv/Scripts/python -m vfx.datagen3d.build --version s3d_v1 --card     # rewrite card.json only

Resumable: complete scenario dirs (meta.json + data files) are skipped; rerun the same command.
Refuses to start if the estimate exceeds the version's size limit (v1 4 GB, v2 6 GB) or the data disk has
< 9 GB free. s3d_v2: v2 scenario sampler + camera policy, lossless pack storage, and the s3d_v1 test / ood
scenarios imported bit-identically (converted to pack) as the first 12 seeds of v2's test / ood.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
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
from .scenarios import (DEFAULT_FRAMES, KIND_PROBS, KINDS, OOD_RANGES, RANGES, V2_KIND_PROBS, V2_KINDS,
                        V2_OOD_RANGES, V2_RANGES, WARMUP_RANGE, sample_scenario, scene_tags)

SEED_BASE = {"train": 0, "test": 100_000, "ood": 200_000}
SEED_SPAN = 100_000
VERSION_PRESETS = {
    "s3d_v1": {"n_train": 96, "n_test": 12, "n_ood": 12, "frames": DEFAULT_FRAMES, "cams": 2},
    "s3d_v2": {"n_train": 384, "n_test": 24, "n_ood": 24, "frames": DEFAULT_FRAMES, "cams": 3, "sampler": "v2",
               "cam_policy": "v2", "n_spare": 2, "storage": "pack", "size_limit_gb": 6.0,
               "import_from": "s3d_v1", "import": {"test": 12, "ood": 12}},
}
DEFAULTS = {"n_train": 8, "n_test": 2, "n_ood": 2, "frames": DEFAULT_FRAMES, "cams": runner.DEFAULT_CAMS,
            "sampler": "v1", "cam_policy": "v1", "n_spare": 0, "storage": "npy", "size_limit_gb": 4.0,
            "import_from": None, "import": {}}
SIZE_LIMIT_GB = 4.0
MIN_FREE_GB = 9.0
PAUSE_FREE_GB = 4.0
RESUME_FREE_GB = 7.0
PACK_RATIO_EST = 3.45                 # lossless ratio of the pack storage (3.48 on a 10-scenario v2 pilot; 3.2-4x on v1);
                                      # the build also stops when the files on disk reach the size limit
BYTES_PER_PX = 3 + 3 + 8 + 1          # lr, hr uint8 RGB; aux 4 x fp16; hr_alpha uint8


def split_seeds(n_train, n_test, n_ood) -> dict:
    return {sp: [SEED_BASE[sp] + i for i in range(n)] for sp, n in
            (("train", n_train), ("test", n_test), ("ood", n_ood))}


def estimate_bytes(n_scn: int, frames: int, cams: int, storage: str = "npy") -> int:
    raw = n_scn * (cams * frames * R.CROP * R.CROP * BYTES_PER_PX + 4 * 128)
    return int(raw / PACK_RATIO_EST) if storage == "pack" else raw


def storage_bytes(root: Path) -> int:
    return sum(f.stat().st_size for f in Path(root).rglob("*") if f.is_file())


def imported(cfg, split: str, seed: int) -> bool:
    return seed - SEED_BASE[split] < cfg.get("import", {}).get(split, 0)


def import_scenario(cfg, split: str, seed: int, out_dir: Path) -> dict:
    """Copy one scenario of cfg["import_from"] into out_dir: arrays bit-identical (re-stored as pack when the
    target storage is pack), meta.json extended with the v2 derived scenario / camera tags and stats."""
    import torch
    src = runner.scenario_dir(paths.dataset_dir(cfg["import_from"]), split, seed)
    if not runner.is_complete(src):
        raise SystemExit(f"cannot import {src}: not complete")
    meta = json.loads((src / "meta.json").read_text())
    A = runner.load_arrays(src)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").unlink(missing_ok=True)
    if cfg["storage"] == "pack":
        runner.write_pack(out_dir / runner.PACK_NAME, A)
    else:
        for n, a in A.items():
            runner._save(out_dir / f"{n}.npy", a)
    T = meta["frames"]
    for k, c in enumerate(meta["cams"]):
        o = {kk: (tuple(v) if isinstance(v, list) else v) for kk, v in c["orbit"].items() if kk != "info"}
        cam = R.OrbitCam(**o)
        g = lambda n: torch.from_numpy(np.ascontiguousarray(A[n][k])).cuda()
        st = runner.cam_stats(g("lr"), g("hr"), g("aux"), g("hr_alpha"))
        c["stats"] = st
        c["flagged"] = False            # v1 data is kept as built (v1 passed its checks)
        c["tags"] = runner.cam_tags(cam, st, T)
    tags = list(meta["tags"]) + [t for t in scene_tags(meta["scenario"]) if t not in meta["tags"]]
    tags += [t for t in runner.scenario_cam_tags(tags, meta["cams"]) if t not in tags]
    meta["tags"] = tags
    meta["imported_from"] = cfg["import_from"]
    meta["storage"] = cfg["storage"]
    meta["cam_policy"] = "v1"
    tmp = out_dir / "meta.tmp.json"
    tmp.write_text(json.dumps(meta, indent=1))
    os.replace(tmp, out_dir / "meta.json")
    return meta


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
    steps = sum(sample_scenario(s, sp, cfg["frames"], cfg["sampler"]).num_steps for sp, ss in seeds.items()
                for s in ss if not imported(cfg, sp, s))
    n = sum(len(v) for v in seeds.values())
    p = {"scenarios": n, "sequences": n * cfg["cams"], "steps": steps,
         "bytes": estimate_bytes(n, cfg["frames"], cfg["cams"], cfg["storage"])}
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
    total = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    raw = sum(len(m["cams"]) * m["frames"] * R.CROP * R.CROP * BYTES_PER_PX for m in metas.values())
    tr = np.asarray([m["tracking_16"]["max_after_step10"] for m in metas.values()]) if metas else np.zeros((0, 3))
    trf = np.asarray([np.max(m["tracking_16"]["frames"], 0) for m in metas.values()]) if metas else np.zeros((0, 3))
    moving = [f"{k}/cam{j}" for k, m in metas.items() for j, c in enumerate(m["cams"]) if c["moving"]]
    kinds = {}
    for m in metas.values():
        kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
    tag_counts, cam_tag_counts, n_seq = {}, {}, {}
    for key, m in metas.items():
        sp = key.split("/")[0]
        tc = tag_counts.setdefault(sp, {"_scenarios": 0})
        tc["_scenarios"] += 1
        for t in m["tags"]:
            tc[t] = tc.get(t, 0) + 1
        cc = cam_tag_counts.setdefault(sp, {"_sequences": 0})
        for c in m["cams"]:
            cc["_sequences"] += 1
            for t in c.get("tags", []):
                cc[t] = cc.get(t, 0) + 1
        n_seq[sp] = n_seq.get(sp, 0) + len(m["cams"])
    sort = lambda d: {k: dict(sorted(v.items(), key=lambda x: -x[1])) for k, v in d.items()}
    flagged = [f"{k}/cam{j}" for k, m in metas.items() for j, c in enumerate(m["cams"]) if c.get("flagged")]
    imports = sorted(k for k, m in metas.items() if m.get("imported_from"))
    r = runner.get_renderer() if runner._renderer is not None else None
    summ = lambda a: {"median": round(float(np.median(a)), 4), "p95": round(float(np.percentile(a, 95)), 4),
                      "max": round(float(np.max(a)), 4)} if len(a) else None
    return {
        "version": version,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "solver_version": sim.solver_version(),
        "counts": {k: len(v) for k, v in seeds.items()},
        "sequences": n_seq,
        "complete": len(metas),
        "sequences_per_scenario": cfg["cams"], "frames_per_sequence": cfg["frames"],
        "sampler": cfg["sampler"], "cam_policy": cfg["cam_policy"], "spare_cameras": cfg["n_spare"],
        "imported": {"from": cfg.get("import_from"), "per_split": cfg.get("import", {}), "scenarios": imports,
                     "note": "bit-identical copies of the source version's scenarios (2 cameras each); meta.json "
                             "gains the derived tags/stats"} if cfg.get("import_from") else None,
        "tag_counts": sort(tag_counts),
        "camera_tag_counts": sort(cam_tag_counts),
        "tag_definitions": {
            "large_fire": "a fire emitter with radius >= 0.08 m (v1 train max 0.075)",
            "multi_fire": ">= 2 fire emitters", "fire_only / smoke_only": "emitter types present",
            "hot_fire": "fire emitter >= 1450 K", "sooty_fire": "fire emitter density_rate > 0.3",
            "occluder": "floor-standing box >= 0.1 m tall (walls / pillars / crates beside fires)",
            "slab": "thin mid-air box above the emitters (preset-4 kind)",
            "closeup (camera)": "mean orbit distance <= 1.35 m (512-px window spans <= 0.35 m)",
            "fill (camera)": "mean share of stored px with lr alpha > 0.02 >= 0.5",
            "flame_dominated (camera)": ">= 25 % of effect px are flame (hr luminance > 0.45 and R - B > 0.25)",
            "occluded_view / emitter_aim / moving (camera)": "camera placed past a floor occluder / aimed at an "
                                                             "emitter / orbit moves",
            "scenario closeup / fill / flame_dominated": "any of its cameras has the tag",
            "large_fire_closeup": "large_fire and a closeup camera",
        },
        "flagged": flagged,
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
        "param_ranges": ({"kinds": dict(zip(V2_KINDS, V2_KIND_PROBS)), "train_test": V2_RANGES,
                          "train_test_shared_v1": RANGES, "ood": V2_OOD_RANGES, "warmup_steps": WARMUP_RANGE,
                          "cameras": R.CAM_RANGES_V2, "camera_validity": runner.VALID,
                          "v1_for_imported": {"kinds": dict(zip(KINDS, KIND_PROBS)), "ood": OOD_RANGES,
                                              "cameras": R.CAM_RANGES}}
                         if cfg["sampler"] == "v2" else
                         {"kinds": dict(zip(KINDS, KIND_PROBS)), "train_test": RANGES, "ood": OOD_RANGES,
                          "warmup_steps": WARMUP_RANGE, "cameras": R.CAM_RANGES}),
        "kinds": kinds,
        "renderer": R.renderer_settings(r.r) if r is not None else None,
        "storage_format": ("data.pack per scenario: lossless per-frame zlib-6 + integer predictors, see "
                           "vfx/datagen3d/pack.py" if cfg["storage"] == "pack" else "npy per array"),
        "storage": {
            "lr": "(K,T,3,256,256) uint8 sRGB, render of raw 64^3, 2x2-avg-pooled crop of the 1440p frame",
            "hr": "(K,T,3,256,256) uint8, render of guided 256^3, identical camera/crop/settings",
            "aux": "(K,T,4,256,256) fp16: alpha (0..1), depth (m, alpha-weighted), motion x,y in STORED px "
                   "per SIM_DT (= renderer output px / 2), alpha-weighted pooled; 0 where alpha = 0",
            "hr_alpha": "(K,T,256,256) uint8: HR render alpha * 255",
        },
        "moving_camera_sequences": moving,
        "bytes": total,
        "raw_bytes": raw,
        "compression_ratio": round(raw / max(total, 1), 3),
        "size_limit_gb": cfg["size_limit_gb"],
        "build_seconds": round(build_seconds, 1),
        "format": "see vfx/datagen3d/README.md",
    }


def build(version: str, cfg: dict, force: bool = False, log=print) -> dict:
    root = paths.dataset_dir(version)
    seeds = split_seeds(cfg["n_train"], cfg["n_test"], cfg["n_ood"])
    n = sum(len(v) for v in seeds.values())
    est = estimate_bytes(n, cfg["frames"], cfg["cams"], cfg["storage"])
    limit = cfg["size_limit_gb"]
    fg = free_gb(root)
    log(f"[datagen3d] {version}: {cfg} -> {n} scenarios, {n * cfg['cams']} sequences, "
        f"~{est / 1e9:.2f} GB at {root}; {fg:.1f} GB free")
    if est > limit * 1e9:
        raise SystemExit(f"estimated {est / 1e9:.2f} GB > {limit} GB limit")
    if fg < MIN_FREE_GB and not force:
        raise SystemExit(f"only {fg:.1f} GB free on the data disk (< {MIN_FREE_GB} GB): refusing to build")
    root.mkdir(parents=True, exist_ok=True)
    (root / "splits.json").write_text(json.dumps(seeds))
    # held-out splits first: if the size guard stops the build, only the train count shrinks
    jobs = [(sp, s) for sp in ("test", "ood", "train") for s in seeds[sp]]
    todo = [(sp, s) for sp, s in jobs if not runner.is_complete(runner.scenario_dir(root, sp, s))]
    log(f"[datagen3d] {len(jobs) - len(todo)} complete, {len(todo)} to run")
    cost = {j: 3 if imported(cfg, *j) else sample_scenario(j[1], j[0], cfg["frames"], cfg["sampler"]).num_steps + 25
            for j in todo}
    total = sum(cost.values()) or 1
    t0 = time.perf_counter()
    done = 0
    for i, (sp, s) in enumerate(todo):
        d = runner.scenario_dir(root, sp, s)
        if imported(cfg, sp, s):
            ts = time.perf_counter()
            m = import_scenario(cfg, sp, s, d)
            done += cost[(sp, s)]
            el = time.perf_counter() - t0
            log(f"[datagen3d] {i + 1}/{len(todo)} {sp} scn_{s:06d} imported from {cfg['import_from']} "
                f"tags={m['tags']} {time.perf_counter() - ts:.1f}s | {el:6.0f}s elapsed", flush=True)
            continue
        sc = sample_scenario(s, sp, cfg["frames"], cfg["sampler"])
        m = runner.run(sc, d, n_cams=cfg["cams"], cam_policy=cfg["cam_policy"], n_spare=cfg["n_spare"],
                       storage=cfg["storage"])
        done += cost[(sp, s)]
        el = time.perf_counter() - t0
        tr = m["tracking_16"]["max_after_step10"]
        sz = sum(f.stat().st_size for f in d.iterdir()) / 1e6
        ctag = ["".join(x[0] for x in c["tags"] if x in ("closeup", "fill", "flame_dominated")) or "-" for c in m["cams"]]
        log(f"[datagen3d] {i + 1}/{len(todo)} {sp} scn_{s:06d} {sc.kind} steps={sc.num_steps} "
            f"track16 d={tr[0]:.3f} T={tr[1]:.3f} u={tr[2]:.3f} {'' if m['finite'] else 'NONFINITE '}"
            f"cams(c/f/f)={ctag} rej={len(m['rejected_cams'])} flag={sum(c['flagged'] for c in m['cams'])} "
            f"{sz:.1f}MB {m['seconds']['total']:.1f}s | {el:6.0f}s elapsed, ETA {el / done * (total - done):6.0f}s",
            flush=True)
        if free_gb(root) < PAUSE_FREE_GB:      # other jobs on this disk swing free space by ~15 GB (pagefile /
            waited = 0                         # temp files): pause instead of writing into a nearly full disk
            while free_gb(root) < RESUME_FREE_GB and waited < 3600:
                if waited == 0:
                    log(f"[datagen3d] data disk {free_gb(root):.1f} GB free < {PAUSE_FREE_GB} GB: pausing until "
                        f">= {RESUME_FREE_GB} GB", flush=True)
                time.sleep(30)
                waited += 30
            if free_gb(root) < RESUME_FREE_GB:
                raise SystemExit("data disk stayed low for an hour: stopping (resumable)")
            if waited:
                log(f"[datagen3d] resumed after {waited} s ({free_gb(root):.1f} GB free)", flush=True)
        if storage_bytes(root) > limit * 1e9:
            raise SystemExit(f"dataset reached the {limit} GB size limit after {i + 1} scenarios: stopping")
    card = _card(version, cfg, seeds, root, time.perf_counter() - t0)
    (root / "card.json").write_text(json.dumps(card, indent=1))
    log(f"[datagen3d] done: {card['complete']} scenarios, {card['bytes'] / 1e9:.2f} GB, "
        f"{card['build_seconds']:.0f}s this session")
    if card["bytes"] > limit * 1e9:
        log(f"[datagen3d] WARNING: {card['bytes'] / 1e9:.2f} GB exceeds the {limit} GB budget")
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
