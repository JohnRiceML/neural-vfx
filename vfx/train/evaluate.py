"""Held-out evaluation: autoregressive rollout of the runtime Upscaler (model fed its own output,
exactly as in the app) vs bicubic, on full 512^2 frames.

All metrics come from the gate implementation `vfx.eval.metrics.evaluate_upscaler` (one definition of
flicker: the prediction is warped with the velocity of the LR sequence that drove it, the reference is
HR warped with its own pooled velocity). The baseline is `vfx.eval.metrics.bicubic_upscale`.

    .venv/Scripts/python -m vfx.train.evaluate --version v2 --split test --lr-key lr_sim [--ckpt PATH]
    .venv/Scripts/python -m vfx.train.evaluate --scorecard [--ckpt PATH]   # v2 test/ood lr_sim + live flicker
"""
from __future__ import annotations

import argparse
import json

import torch

from vfx.runtime import Upscaler

_BIC_CACHE: dict = {}


def _bicubic(version, split, lr_key, max_scenarios, max_frames):
    from vfx.eval import metrics as M
    key = (version, split, lr_key, max_scenarios, max_frames)
    if key not in _BIC_CACHE:
        _BIC_CACHE[key] = M.evaluate_upscaler(M.bicubic_upscale, version, split, lr_key, max_scenarios, max_frames)
    return _BIC_CACHE[key]


@torch.no_grad()
def evaluate(ckpt, version: str, split: str = "test", max_scenarios: int | None = None,
             lr_key: str = "lr", sparse: bool = True, max_frames: int | None = None) -> dict:
    from vfx.eval import metrics as M
    ups = Upscaler(str(ckpt), sparse=sparse, cuda_graph=True)
    n = M.evaluate_upscaler(ups, version, split, lr_key, max_scenarios, max_frames)
    b = _bicubic(version, split, lr_key, max_scenarios, max_frames)
    r = {"net_l1_d": n.l1_density, "bic_l1_d": b.l1_density, "net_l1_T": n.l1_temperature,
         "bic_l1_T": b.l1_temperature, "net_spec": n.spectrum_error, "bic_spec": b.spectrum_error,
         "net_flicker": n.flicker, "bic_flicker": b.flicker, "ref_flicker": n.flicker_ref,
         "flicker_ratio_vs_ref": n.flicker_ratio, "bic_flicker_ratio_vs_ref": b.flicker_ratio}
    r["spec_improvement"] = 1.0 - r["net_spec"] / max(r["bic_spec"], 1e-12)
    r["l1_improvement"] = 1.0 - r["net_l1_d"] / max(r["bic_l1_d"], 1e-12)
    r["n_scenarios"], r["metrics_source"], r["lr_key"] = n.n_scenarios, "vfx.eval.metrics.evaluate_upscaler", lr_key
    return r


def scorecard(ckpt, version: str = "v2", live: bool = True, max_scenarios: int | None = None) -> dict:
    """The audited goal metrics for one checkpoint: v2 test/ood with lr_sim input, test with pooled lr
    (the Layer 4 L1 gate input) + live flicker."""
    out = {}
    for split, key in (("test", "lr_sim"), ("ood", "lr_sim"), ("test", "lr")):
        out[f"{split}/{key}"] = evaluate(ckpt, version, split, max_scenarios, key)
    if live:
        from vfx.eval.live_flicker import live_flicker
        out["live"] = live_flicker(300, 30, str(ckpt))
    return out


def _fmt(sc: dict) -> str:
    lines = []
    for k, r in sc.items():
        if k == "live":
            for s, v in r.items():
                lines.append(f"live/{s:15s} flicker ratio neural {v.get('ratio_neural', float('nan')):.3f} "
                             f"bicubic {v['ratio_bicubic']:.3f}  neural/bicubic {v.get('neural_over_bicubic', float('nan')):.3f}")
        else:
            lines.append(f"{k:12s} flicker ratio {r['flicker_ratio_vs_ref']:.3f} (bic {r['bic_flicker_ratio_vs_ref']:.3f}) | "
                         f"spec {r['net_spec']:.4f} vs {r['bic_spec']:.4f} ({100 * r['spec_improvement']:.0f}% below) | "
                         f"L1 d {r['net_l1_d']:.5f} vs {r['bic_l1_d']:.5f} ({100 * r['l1_improvement']:+.1f}%) | "
                         f"L1 T {r['net_l1_T']:.2f} vs {r['bic_l1_T']:.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--version", default="v2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--lr-key", default="lr", help="lr (pooled HR) or lr_sim (the 128^2 sim)")
    ap.add_argument("--max-scenarios", type=int, default=None)
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--scorecard", action="store_true", help="v2 test/ood lr_sim + live flicker")
    ap.add_argument("--no-live", action="store_true")
    ap.add_argument("--out", default=None, help="write the result JSON here")
    a = ap.parse_args()
    from vfx.core import paths
    ck = a.ckpt or paths.default_checkpoint()
    if a.scorecard:
        res = scorecard(ck, a.version, not a.no_live, a.max_scenarios)
        print(_fmt(res))
    else:
        res = evaluate(ck, a.version, a.split, a.max_scenarios, a.lr_key, sparse=not a.dense)
        print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"ckpt": str(ck), **res}, f, indent=1, default=str)
