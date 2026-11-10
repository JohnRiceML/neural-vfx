"""Tuned-classical baseline: sweep 64^3 SolverConfig3D overrides against restricted 128^3 references
on TRAIN scenes (datagen3d sampler, train split). The learned corrector must beat the winner.

    .venv/Scripts/python -m vfx.correct3d.tune_classical --seeds 0-7 --steps 300 --out DIR

score (lower is better, averaged over scenes) =
    smoke-centroid |err| / 0.05 m + 95%-height |err| / 0.05 m + |log KE ratio| + mean |log| KE-spectrum
    ratio over k = 1..8 + rel. L2 error of the time-mean density at 16^3
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from vfx.sim3d import SolverConfig3D

from .common import sampled_scene
from .rollout import rollout, summarize

BASE_BETA = SolverConfig3D().buoyancy_beta
BASE_ALPHA = SolverConfig3D().buoyancy_alpha


def score(s: dict) -> float:
    return (s["cy_err"] / 0.05 + s["h95_err"] / 0.05 + abs(np.log(s["ke_ratio"])) + s["spec_logerr_k1_8"]
            + s["meanD_err16"])


def parse_seeds(txt: str) -> list[int]:
    if "-" in txt:
        a, b = txt.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in txt.split(",")]


def grid():
    for eps, bscale, floor in itertools.product((6.0, 4.0, 3.0, 2.0, 1.0, 0.0), (1.0, 1.2, 1.4), (0.5,)):
        yield {"vorticity_eps": eps, "buoyancy_beta": BASE_BETA * bscale, "buoyancy_alpha": BASE_ALPHA * bscale,
               "confinement_floor": floor}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--split", default="train")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--noise_floor", action="store_true")
    a = ap.parse_args()
    scenes = [sampled_scene(s, a.split) for s in parse_seeds(a.seeds)]
    refs = [rollout(sc, 128, a.steps, mip_steps=()) for sc in scenes]
    if a.noise_floor:
        # same 128^3 scene with a 1 mm/s nudge at step 5: the chaotic spread no correction can remove
        from vfx.core.fields3d import Forces3D, Impulse3D

        def ev_for(sc):
            def ev(t, st, s):
                f = sc.forces(t)
                if t == 5:
                    return Forces3D(f.emitters, f.impulses + [Impulse3D((0.5, 0.3, 0.5), 0.1, (1e-3, 0, 0))], f.wind)
                return f
            return ev
        ss = [summarize(rollout(sc, 128, a.steps, mip_steps=(), events=ev_for(sc)), r) for sc, r in zip(scenes, refs)]
        row = {k: float(np.mean([s[k] for s in ss])) for k in ("cy_err", "h95_err", "ke_ratio", "spec_logerr_k1_8",
                                                                "meanD_err16", "meanU_err16")}
        row["score"] = float(np.mean([score(s) for s in ss]))
        print("NOISE FLOOR (128 vs perturbed 128):", row, flush=True)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "noise_floor.json").write_text(json.dumps(row, indent=1))
        return
    rows = []
    for cfg in grid():
        ss = [summarize(rollout(sc, 64, a.steps, cfg_kw=cfg, mip_steps=()), r) for sc, r in zip(scenes, refs)]
        row = {"cfg": cfg, "score": float(np.mean([score(s) for s in ss]))}
        for k in ("cy_err", "h95_err", "ke_ratio", "spec_logerr_k1_8", "meanD_err16", "meanU_err16"):
            row[k] = float(np.mean([s[k] for s in ss]))
        row["band_ratio"] = np.exp(np.mean([np.log(s["band_ratio"]) for s in ss], 0)).round(3).tolist()
        rows.append(row)
        print(f"eps {cfg['vorticity_eps']:.1f} buoy x{cfg['buoyancy_beta'] / BASE_BETA:.2f}: score {row['score']:.3f} "
              f"cy {row['cy_err']:.3f} h95 {row['h95_err']:.3f} KE {row['ke_ratio']:.2f} spec {row['spec_logerr_k1_8']:.2f} "
              f"meanD {row['meanD_err16']:.3f} meanU {row['meanU_err16']:.3f} bands {row['band_ratio']}", flush=True)
    rows.sort(key=lambda r: r["score"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tune_classical.json").write_text(json.dumps(rows, indent=1))
    print("BEST", rows[0]["cfg"], rows[0]["score"])


if __name__ == "__main__":
    main()
