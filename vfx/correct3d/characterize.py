"""Step 1: what is there to gain? 64^3 vs restricted 128^3 (and 256^3) on the six named scenes.

    .venv/Scripts/python -m vfx.correct3d.characterize --steps 300 --fine256 smoke,fire --out DIR

Writes DIR/characterize.json (per scene: summaries vs the 128^3 reference, end centroids, band ratios,
height profiles of the time-mean density) and DIR/char_<scene>.png (MIPs at 4 times, 64 | 128 | 256).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .common import named_scenes, to_u8, write_png
from .rollout import rollout, summarize


def height_profile(mean_d: torch.Tensor) -> list[float]:
    p = mean_d.sum(dim=(0, 2))                         # (64,) over y
    p = torch.nn.functional.avg_pool1d(p[None, None], 4)[0, 0]
    return [round(float(v), 3) for v in (p / (p.sum() + 1e-12)).cpu()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--fine256", default="smoke,fire")
    ap.add_argument("--scenes", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scenes = named_scenes()
    names = a.scenes.split(",") if a.scenes else list(scenes)
    f256 = set(x for x in a.fine256.split(",") if x)
    report = {}
    for name in names:
        sc = scenes[name]
        t0 = time.time()
        r = {res: rollout(sc, res, a.steps) for res in (64, 128)}
        if name in f256:
            r[256] = rollout(sc, 256, a.steps)
        ref = r[128]
        rep = {"time_s": round(time.time() - t0, 1)}
        for res, ro in r.items():
            rep[str(res)] = summarize(ro, ref)
            rep[str(res)]["profile"] = height_profile(ro["mean_d"])
            rep[str(res)]["finite"] = ro["final_finite"]
            rep[str(res)]["heat_ratio"] = float(ro["diag"][30:, 3].mean() / ref["diag"][30:, 3].mean())
            rep[str(res)]["spec"] = [float(x) for x in ro["spec"][:16]]
        # where is the error: time-mean density difference split by height thirds (64 vs ref)
        dd = (r[64]["mean_d"] - ref["mean_d"])
        thirds = [float(dd[:, i * 21:(i + 1) * 21].abs().sum() / (ref["mean_d"].abs().sum() + 1e-12)) for i in range(3)]
        rep["abs_err_by_height_third_low_mid_high"] = thirds
        report[name] = rep
        # image: rows = times, columns = resolutions
        cols = [64, 128] + ([256] if 256 in r else [])
        ts = sorted(r[64]["mips"])[1:]
        vmax = max(float(np.percentile(r[c]["mips"][t], 99.5)) for c in cols for t in ts) + 1e-6
        rows = []
        for t in ts:
            tiles = [np.kron(to_u8(r[c]["mips"][t], vmax), np.ones((4, 4), np.uint8)) for c in cols]
            sep = np.full((256, 4), 128, np.uint8)
            rows.append(np.concatenate(sum([[x, sep] for x in tiles], [])[:-1], 1))
            rows.append(np.full((4, rows[-1].shape[1]), 128, np.uint8))
        write_png(out / f"char_{name}.png", np.concatenate(rows[:-1], 0))
        s64, s256 = rep["64"], rep.get("256")
        print(f"{name:13s} cy_end 64={s64['cy_end']:.3f} 128={s64['cy_end_ref']:.3f}"
              + (f" 256={s256['cy_end']:.3f}" if s256 else "")
              + f" | 64: mass {s64['mass_ratio']:.2f} heat {s64['heat_ratio']:.2f} KE {s64['ke_ratio']:.2f} "
              f"vy {s64['vy_ratio']:.2f} bands {[round(x, 2) for x in s64['band_ratio']]} "
              f"meanD_err {s64['meanD_err16']:.2f} meanU_err {s64['meanU_err16']:.2f}"
              + (f" | 256 vs 128: KE {s256['ke_ratio']:.2f} bands {[round(x, 2) for x in s256['band_ratio']]} "
                 f"meanD_err {s256['meanD_err16']:.2f}" if s256 else "") + f"  ({rep['time_s']} s)", flush=True)
        torch.cuda.empty_cache()
    (out / "characterize.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
