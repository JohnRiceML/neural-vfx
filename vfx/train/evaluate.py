"""Held-out evaluation: autoregressive rollout of the runtime Upscaler (model fed its own output,
exactly as in the app) vs bicubic, on full 512^2 frames.

    .venv/Scripts/python -m vfx.train.evaluate --ckpt <path> --version synthetic --split test
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from vfx.core.fields import CH, DOMAIN_SIZE, HR_RES, LR_RES, SIM_DT, FieldMeta, FieldState
from vfx.runtime import Upscaler, classical_upscale, warp_prev
from vfx.train.metrics import get_metrics


def _flicker(cur, prev, vel):
    if prev is None:
        return None
    return (cur - warp_prev(prev, vel, DOMAIN_SIZE / HR_RES, SIM_DT)).abs().mean().item()


@torch.no_grad()
def evaluate(ckpt, version: str, split: str = "test", max_scenarios: int | None = None,
             lr_key: str = "lr", sparse: bool = True, max_frames: int | None = None) -> dict:
    from vfx.datagen import load_dataset
    spec, l1, src = get_metrics()
    ds = load_dataset(version, split, include_lr_sim=(lr_key == "lr_sim"))
    ups = Upscaler(str(ckpt), sparse=sparse, cuda_graph=True)
    acc = {k: [] for k in ("net_l1_d", "bic_l1_d", "net_l1_T", "bic_l1_T", "net_spec", "bic_spec",
                           "net_flicker", "bic_flicker", "ref_flicker", "frac")}
    scns = ds.scenarios[:max_scenarios] if max_scenarios else ds.scenarios
    for s in scns:
        hr = np.load(s["dir"] / "hr.npy", mmap_mode="r")
        lr = np.load(s["dir"] / f"{lr_key}.npy", mmap_mode="r")
        solid = torch.from_numpy(np.load(s["dir"] / "solid_lr.npy")).float().cuda()
        ups.reset()
        prev = {"net": None, "bic": None, "ref": None}
        prev_vel = None
        T = hr.shape[0] if max_frames is None else min(hr.shape[0], max_frames)
        for t in range(T):
            data = torch.from_numpy(np.asarray(lr[t], np.float32)).cuda()
            st = FieldState(data, solid, FieldMeta.for_res(LR_RES))
            o = ups.infer(st)
            net = torch.stack([o.density, o.temperature]).float()
            b = classical_upscale(st)
            bic = torch.stack([b.density, b.temperature]).float()
            ref = torch.from_numpy(np.asarray(hr[t], np.float32)).cuda()
            vel = data[CH["vel_x"]:CH["vel_y"] + 1]
            acc["net_l1_d"].append(l1(net[0], ref[0])); acc["bic_l1_d"].append(l1(bic[0], ref[0]))
            acc["net_l1_T"].append(l1(net[1], ref[1])); acc["bic_l1_T"].append(l1(bic[1], ref[1]))
            acc["net_spec"].append(spec(net[0], ref[0])); acc["bic_spec"].append(spec(bic[0], ref[0]))
            acc["frac"].append(ups.timings["processed_fraction"])
            for name, x in (("net", net[0]), ("bic", bic[0]), ("ref", ref[0])):
                f = _flicker(x[None], prev[name], prev_vel)
                if f is not None:
                    acc[f"{name}_flicker"].append(f)
                prev[name] = x[None]
            prev_vel = vel
    r = {k: float(np.mean(v)) for k, v in acc.items() if v}
    r["spec_improvement"] = 1.0 - r["net_spec"] / max(r["bic_spec"], 1e-12)
    r["l1_improvement"] = 1.0 - r["net_l1_d"] / max(r["bic_l1_d"], 1e-12)
    r["flicker_ratio_vs_ref"] = r["net_flicker"] / max(r["ref_flicker"], 1e-12)
    r["n_scenarios"], r["metrics_source"], r["lr_key"] = len(scns), src, lr_key
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--version", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--lr-key", default="lr", help="lr (pooled HR) or lr_sim (independent LR sim)")
    ap.add_argument("--max-scenarios", type=int, default=None)
    ap.add_argument("--dense", action="store_true")
    a = ap.parse_args()
    from vfx.core import paths
    print(json.dumps(evaluate(a.ckpt or paths.default_checkpoint(), a.version, a.split, a.max_scenarios,
                              a.lr_key, sparse=not a.dense), indent=1))
