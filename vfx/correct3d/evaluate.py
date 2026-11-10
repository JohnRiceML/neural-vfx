"""Held-out evaluation, stability under user perturbations, and cost of the velocity corrector.

    .venv/Scripts/python -m vfx.correct3d.evaluate compare   --ckpt CK --classical '{json}' --out DIR
    .venv/Scripts/python -m vfx.correct3d.evaluate stability --ckpt CK --steps 3000 --out DIR
    .venv/Scripts/python -m vfx.correct3d.evaluate timing    --ckpt CK

compare: per held-out scene (6 named + sampled test / ood seeds), 64^3 raw | tuned-classical | learned
vs the restricted 128^3 reference; writes compare.json and cmp_<scene>.png (front MIPs at several
times, columns raw | classical | learned | ref128).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from vfx.core.fields3d import SIM_DT, Forces3D, Impulse3D, Obstacle3D

from . import VelocityCorrector
from .common import (DIAG_NAMES, get_solver, named_scenes, sampled_scene, solid_at, to_u8, write_png)
from .rollout import rollout, summarize
from .tune_classical import parse_seeds, score

KEYS = ("cy_err", "h95_err", "ke_ratio", "spec_logerr_k1_8", "meanD_err16", "meanU_err16")


def scenes_for(a):
    out = []
    named = named_scenes()
    for n in [x for x in a.named.split(",") if x]:
        out.append(named[n])
    for s in parse_seeds(a.test) if a.test not in ("", "none") else []:
        out.append(sampled_scene(s, "test"))
    for s in parse_seeds(a.ood) if a.ood not in ("", "none") else []:
        out.append(sampled_scene(s, "ood"))
    return out


def cmd_compare(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    classical = json.loads(a.classical)
    learned_cfg = json.loads(a.learned_cfg)
    corr = VelocityCorrector(a.ckpt, gain=a.gain)
    mips = (60, 120, 180, 240, a.steps - 1)
    report, agg = {}, {k: [] for k in ("raw", "classical", "learned")}
    for sc in scenes_for(a):
        ref = rollout(sc, 128, a.steps, mip_steps=mips)
        runs = {"raw": rollout(sc, 64, a.steps, mip_steps=mips),
                "classical": rollout(sc, 64, a.steps, cfg_kw=classical, mip_steps=mips),
                "learned": rollout(sc, 64, a.steps, cfg_kw=learned_cfg, corrector=corr, mip_steps=mips)}
        rep = {}
        for k, r in runs.items():
            s = summarize(r, ref)
            s["score"] = score(s)
            s["finite"] = r["final_finite"]
            if "cstats" in r:
                cs = r["cstats"]
                s["corr_rms"] = float(cs[:, 0].mean())
                s["fallback_frac"] = float(cs[:, 2].mean())
            rep[k] = s
            agg[k].append(s)
        report[sc.name] = {"tags": sc.tags, **rep}
        print(f"{sc.name:14s} " + " | ".join(
            f"{k}: sc {v['score']:.2f} cy {v['cy_end']:.3f} (ref {v['cy_end_ref']:.3f}) KE {v['ke_ratio']:.2f} "
            f"spec {v['spec_logerr_k1_8']:.2f} mD {v['meanD_err16']:.2f}" for k, v in rep.items()), flush=True)
        # image: rows = times, columns raw | classical | learned | ref
        cols = [runs["raw"], runs["classical"], runs["learned"], ref]
        ts = [60, 120, 240, a.steps - 1]
        vmax = max(float(np.percentile(c["mips"][t], 99.5)) for c in cols for t in ts) + 1e-6
        rows = []
        for t in ts:
            tiles = [np.kron(to_u8(c["mips"][t], vmax), np.ones((3, 3), np.uint8)) for c in cols]
            sep = np.full((192, 3), 110, np.uint8)
            rows.append(np.concatenate(sum([[x, sep] for x in tiles], [])[:-1], 1))
            rows.append(np.full((3, rows[-1].shape[1]), 110, np.uint8))
        write_png(out / f"cmp_{sc.name}.png", np.concatenate(rows[:-1], 0))
        torch.cuda.empty_cache()
    summ = {}
    for k, L in agg.items():
        summ[k] = {m: float(np.mean([s[m] for s in L])) for m in KEYS + ("score",)}
        summ[k]["ke_logerr"] = float(np.mean([abs(math.log(s["ke_ratio"])) for s in L]))
        summ[k]["band_ratio_geo"] = np.exp(np.mean([np.log(s["band_ratio"]) for s in L], 0)).round(3).tolist()
        print(k, {m: round(v, 3) if isinstance(v, float) else v for m, v in summ[k].items()})
    report["_mean"] = summ
    (out / f"compare_{a.tag}.json").write_text(json.dumps(report, indent=1))


def perturbation_events(sc, steps, seed=0):
    """User-style perturbations: random strong pushes, emitters off / on, an obstacle box toggled
    in the flow, a wind gust. Returns events(step, state, solver) for rollout()."""
    rng = np.random.default_rng(seed)
    pushes = {}
    for t in range(40, steps, 37):
        a = rng.uniform(0, 2 * math.pi)
        s = rng.uniform(1.0, 4.0)
        pushes[t] = Impulse3D(pos=tuple(rng.uniform(0.25, 0.75, 3)), radius=float(rng.uniform(0.08, 0.2)),
                              velocity=(s * math.cos(a), float(rng.uniform(-1.5, 2.5)), s * math.sin(a)))
    off = (int(steps * 0.3), int(steps * 0.43))
    box_on = (int(steps * 0.53), int(steps * 0.73))
    gust = (int(steps * 0.8), int(steps * 0.87))
    box = solid_at(list(sc.obstacles) + [Obstacle3D("box", (0.5, 0.45, 0.5), (0.2, 0.03, 0.15))], 64)
    plain = solid_at(sc.obstacles, 64)

    def ev(t, st, s):
        if t == box_on[0] and st.solid.shape[-1] == 64:
            st.solid.copy_(box)
        if t == box_on[1] and st.solid.shape[-1] == 64:
            st.solid.copy_(plain)
        f = sc.forces(t)
        em = [] if off[0] <= t < off[1] else list(f.emitters)
        imps = list(f.impulses) + ([pushes[t]] if t in pushes else [])
        wind = (1.5, 0.3, -0.8) if gust[0] <= t < gust[1] else f.wind
        return Forces3D(emitters=em, impulses=imps, wind=wind)
    return ev


def cmd_stability(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    corr = VelocityCorrector(a.ckpt, gain=a.gain)
    learned_cfg = json.loads(a.learned_cfg)
    res = {}
    for sc in scenes_for(a):
        r = {}
        for k, c, kw in (("raw", None, {}), ("learned", corr, learned_cfg)):
            ro = rollout(sc, 64, a.steps, cfg_kw=kw, corrector=c, mip_steps=(), events=perturbation_events(sc, a.steps),
                         div_every=50, spec_every=50, tag="stab")
            d = ro["diag"]
            q = len(d) // 6
            r[k] = {"finite": ro["final_finite"], "max_speed_end": ro["max_speed"],
                    "ke_by_sixth": [float(d[i * q:(i + 1) * q, 5].mean()) for i in range(6)],
                    "mass_by_sixth": [float(d[i * q:(i + 1) * q, 0].mean()) for i in range(6)],
                    "div_max": float(ro["div"].max()), "div_p50": float(np.median(ro["div"]))}
            if "cstats" in ro:
                cs = ro["cstats"]
                r[k]["fallback_frac"] = float(cs[:, 2].mean())
                r[k]["corr_rms_max"] = float(cs[:, 0][cs[:, 2] < 0.5].max()) if (cs[:, 2] < 0.5).any() else 0.0
                r[k]["corr_over_u_mean"] = float((cs[:, 0] / (cs[:, 1] + 1e-9)).mean())
        r["ke_ratio_learned_over_raw_by_sixth"] = [x / max(y, 1e-12) for x, y in
                                                   zip(r["learned"]["ke_by_sixth"], r["raw"]["ke_by_sixth"])]
        res[sc.name] = r
        print(sc.name, json.dumps(r), flush=True)
    (out / f"stability_{a.tag}.json").write_text(json.dumps(res, indent=1))


def cmd_timing(a):
    corr = VelocityCorrector(a.ckpt)
    sc = named_scenes()["smoke_sphere"]
    s = get_solver(64, dict(sc.sim))
    st = s.reset(sc.obstacles)
    for t in range(60):
        s.step(st, sc.forces(t), SIM_DT)
    res = {}
    for name, use_graph in (("graph", True), ("eager", False)):
        corr.use_graph = use_graph
        ts, tstep = [], []
        for t in range(260):
            e = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            torch.cuda._sleep(2_000_000)
            e[0].record()
            corr.apply(st, SIM_DT)
            e[1].record()
            s.step(st, sc.forces(t), SIM_DT)
            e[2].record()
            e[2].synchronize()
            if t >= 60:
                ts.append(e[0].elapsed_time(e[1]))
                tstep.append(e[1].elapsed_time(e[2]))
        res[name] = {"corrector_ms_median": float(np.median(ts)), "corrector_ms_p90": float(np.percentile(ts, 90)),
                     "step_ms_median": float(np.median(tstep))}
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["compare", "stability", "timing"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--classical", default="{}")
    ap.add_argument("--learned_cfg", default="{}", help="64^3 config the learned run uses (base it was trained on)")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--named", default="fire,smoke,smoke_sphere,fire_wind,smoke_push,ood_box_wind")
    ap.add_argument("--test", default="100000-100003")
    ap.add_argument("--ood", default="200000-200002")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    {"compare": cmd_compare, "stability": cmd_stability, "timing": cmd_timing}[a.cmd](a)


if __name__ == "__main__":
    main()
