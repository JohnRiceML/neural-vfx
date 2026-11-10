"""Train route B (ScreenNet / ScreenDetail) on vfx.datagen3d sequences.

    .venv/Scripts/python -m vfx.train3d.train --config vfx/train3d/configs/<name>.json [--minutes M]

Sample = K consecutive frames of one camera sequence (random crop biased to the effect, h-flip with motion_x
negated), all on the GPU. The net is run exactly as ScreenDetail runs it at process_scale 1.0 (the stored crops
ARE the runtime's processed resolution): out_k = clamp(lr_k + net(lr_k, aux_k, prev_k)), cold start prev_0 = lr_0,
prev_k = out_{k-1} (warped by prev_warp x motion when that is non-zero) with gradient through the chain.
Without use_prev the K frames are independent forward passes (single-frame net) and K only feeds the temporal term.
Loss per frame (weights = 0.1 + effect mask): L1 + grad + low-band log spectrum of luminance (eval bins) +
temporal change matching (out_k - out_{k-1}) vs (hr_k - hr_{k-1}).
End of run: vfx.eval3d.metrics.evaluate_screen_detail on test and ood through the real ScreenDetail runtime
(checkpoint path, process_scale 1.0), summary.json, EXPERIMENTS.md entry. Never promotes (see promote.py).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vfx.core import paths
from vfx.model3d import ScreenNet, ScreenNetConfig, load_screen_net, param_count, save_screen_net
from vfx.train3d import losses as L
from vfx.train3d.data import GPUWindows

EXPERIMENTS = Path(__file__).resolve().parent / "EXPERIMENTS.md"
CKPT_DIR = paths.CHECKPOINTS / "screen3d"

DEFAULTS = {
    "name": "screen3d", "dataset": "s3d_v1", "hypothesis": "", "init": None,
    "model": {"width": 64, "n_blocks": 8, "trunk_stride": 4, "hr_feats": 8, "hr_hidden": 16, "hr_kernel": 3,
              "use_aux": True, "use_prev": False, "prev_warp": 0.0},
    "train": {"minutes": 20.0, "batch": 12, "crop": 192, "frames": 3, "lr": 3e-4, "min_lr": 1e-5,
              "warmup_steps": 200, "grad_clip": 1.0, "seed": 0, "hflip": True, "log_every": 50, "val_every": 500,
              "ema": 0.999, "detach_prev": False, "max_train_sequences": None, "bg_weight": 0.1,
              "weight_decay": 1e-4, "holdout_sequences": 0},
    "loss": {"l1": 1.0, "grad": 0.5, "spec": 0.02, "spec_max_frac": 0.5, "temporal": 0.5, "ssim": 0.0, "speckle": 0.0},
    "eval": {"splits": ["test", "ood"], "max_sequences": None},
}


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def warp(x, motion, s):
    """x(p - s*motion(p)), pixel-centre grid, border clamp: the same formula ScreenDetail uses."""
    B, _, h, w = x.shape
    ys = (torch.arange(h, device=x.device, dtype=torch.float32) * 2 + 1) / h - 1
    xs = (torch.arange(w, device=x.device, dtype=torch.float32) * 2 + 1) / w - 1
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    m = motion.float() * s
    grid = torch.stack([xx[None] - m[:, 0] * (2.0 / w), yy[None] - m[:, 1] * (2.0 / h)], -1)
    return F.grid_sample(x, grid.to(x.dtype), mode="bilinear", padding_mode="border", align_corners=False)


def rollout(model, lr, hr, aux, mask, cfg):
    """Returns (total loss, terms dict, outputs (B,K,3,H,W))."""
    wl, tc = cfg["loss"], cfg["train"]
    mc = model.cfg
    B, K = lr.shape[:2]
    w = (tc["bg_weight"] + mask).unsqueeze(2)                 # (B,K,1,H,W)
    if not mc.use_prev:
        x, a = lr.flatten(0, 1), aux.flatten(0, 1)
        with torch.autocast("cuda", dtype=torch.float16):
            d = model(x, a if mc.use_aux else None, None).float()
        out = (x + d).view_as(lr)
    else:
        outs, prev = [], lr[:, 0]
        for k in range(K):
            pv = warp(prev, aux[:, k, 2:4], mc.prev_warp) if mc.prev_warp else prev
            with torch.autocast("cuda", dtype=torch.float16):
                d = model(lr[:, k], aux[:, k] if mc.use_aux else None, pv).float()
            o = lr[:, k] + d
            outs.append(o)
            prev = o.clamp(0, 1)
            if tc["detach_prev"]:
                prev = prev.detach()
        out = torch.stack(outs, 1)
    P, T, W = out.flatten(0, 1), hr.flatten(0, 1), w.flatten(0, 1)
    terms = {"l1": L.wl1(P, T, W), "grad": L.grad_l1(P, T, W),
             "spec": L.spectral_loss(P, T, wl["spec_max_frac"])}
    if wl.get("speckle", 0.0):
        terms["speckle"] = L.speckle_l1(P, T, W)
    if wl.get("ssim", 0.0):
        terms["ssim"] = L.ms_dssim(P, T, W)
    if K > 1:
        terms["temporal"] = L.temporal_l1(out[:, 1:].flatten(0, 1), out[:, :-1].flatten(0, 1),
                                          hr[:, 1:].flatten(0, 1), hr[:, :-1].flatten(0, 1), w[:, 1:].flatten(0, 1))
    else:
        terms["temporal"] = torch.zeros((), device=lr.device)
    total = sum(wl[k] * v for k, v in terms.items())
    return total, terms, out


@torch.no_grad()
def quick_val(model, batch, cfg):
    """Training-curve numbers on a fixed batch: masked L1 and spectral term, model vs the lr input."""
    lr, hr, aux, mask = batch
    model.eval()
    _, terms, out = rollout(model, lr, hr, aux, mask, cfg)
    model.train()
    P, T, X = out.flatten(0, 1).clamp(0, 1), hr.flatten(0, 1), lr.flatten(0, 1)
    m = mask.flatten(0, 1).unsqueeze(1)
    return {"l1": float(L.wl1(P, T, m)), "l1_lr": float(L.wl1(X, T, m)),
            "ssim": float(L.ms_dssim(P, T, m)), "ssim_lr": float(L.ms_dssim(X, T, m)),
            "spec": float(L.spectral_loss(P, T, 1.0)), "spec_lr": float(L.spectral_loss(X, T, 1.0))}


def final_eval(ckpt: Path, cfg) -> dict:
    from vfx.eval3d import metrics as M
    sh = M.tune_unsharp(cfg["dataset"], "train")
    sd = M.load_screen_detail_for_dataset(ckpt)
    out = {"sharpen": {"sigma": sh["sigma"], "amount": sh["amount"]}}
    keys = ("l1", "spectrum_error", "spectrum_log_ratio", "ms_dssim", "flicker_ratio", "flicker_ratio_moving",
            "jerk_ratio", "swim_ratio", "swim_ratio_moving")
    for sp in cfg["eval"]["splits"]:
        r = M.evaluate_screen_detail(sd, cfg["dataset"], sp, length=16, sharpen=sh,
                                     max_sequences=cfg["eval"]["max_sequences"])
        out[sp] = {name: {k: round(m[k], 5) for k in keys if k in m} for name, m in r["methods"].items()}
        out[sp]["n_sequences"] = r["n_sequences"]
    return out


def beats_baselines(ev: dict, split="test") -> bool:
    m = ev[split]
    return all(m["model"][k] < min(m["lr"][k], m["sharpen"][k]) for k in ("l1", "spectrum_error", "ms_dssim"))


def train(cfg):
    tc = cfg["train"]
    torch.manual_seed(tc["seed"])
    torch.backends.cudnn.benchmark = True
    gen = np.random.default_rng(tc["seed"])
    t_load = time.time()
    ho = tc["holdout_sequences"]
    data = GPUWindows(cfg["dataset"], "train", max_sequences=tc["max_train_sequences"],
                      seq_range=(0, -ho) if ho else None)
    hold = GPUWindows(cfg["dataset"], "train", seq_range=(-ho, 1 << 30)) if ho else None
    test = GPUWindows(cfg["dataset"], "test")
    print(f"data: train {data.N} seqs ({data.nbytes() / 2**30:.2f} GiB on GPU), test {test.N}; "
          f"{time.time() - t_load:.0f} s", flush=True)
    K = tc["frames"]
    val_batches = {"test": test.fixed_batch(16, K, 256), "train": data.fixed_batch(16, K, 256)}
    if hold is not None:      # model selection ONLY on held-out train scenarios (never on test)
        val_batches["hold"] = hold.fixed_batch(32, K, 256)
    if cfg.get("init"):
        model = load_screen_net(cfg["init"]).train()
    else:
        model = ScreenNet(ScreenNetConfig(**cfg["model"])).cuda()
    model = model.to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval() if tc["ema"] else None
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler("cuda")
    run_dir = paths.RUNS / "screen3d" / f"{cfg['name']}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    logf = open(run_dir / "log.jsonl", "a")
    print(f"run {run_dir}  params {param_count(model)}  model {asdict(model.cfg)}", flush=True)

    def log(step, d):
        logf.write(json.dumps({"step": step, **d}) + "\n")
        logf.flush()

    budget, t0, step, n_bad, curve = tc["minutes"] * 60, time.time(), 0, 0, []
    best = (float("inf"), None, 0)
    model.train()
    while True:
        frac = (time.time() - t0) / budget
        if frac >= 1.0:
            break
        lr_now = (tc["min_lr"] + 0.5 * (tc["lr"] - tc["min_lr"]) * (1 + math.cos(math.pi * frac))) \
            * min(1.0, (step + 1) / tc["warmup_steps"])
        for g in opt.param_groups:
            g["lr"] = lr_now
        b = data.sample(tc["batch"], K, tc["crop"], gen, hflip=tc["hflip"])
        total, terms, _ = rollout(model, *b, cfg)
        if not torch.isfinite(total):
            n_bad += 1
            if n_bad >= 50:
                raise RuntimeError("50 consecutive non-finite losses")
            step += 1
            continue
        n_bad = 0
        opt.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        scaler.step(opt)
        scaler.update()
        if ema is not None:
            with torch.no_grad():
                dec = min(tc["ema"], (1 + step) / (10 + step))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - dec)
        step += 1
        if step % tc["log_every"] == 0 or step == 1:
            d = {"loss": float(total), **{k: float(v) for k, v in terms.items()}, "lr": lr_now,
                 "elapsed_min": round((time.time() - t0) / 60, 3)}
            log(step, d)
            print(f"step {step:6d} " + " ".join(f"{k}={v:.4g}" for k, v in d.items()), flush=True)
        if step % tc["val_every"] == 0:
            ev_m = ema if ema is not None else model
            v = {f"{sp}_{k}": x for sp, bt in val_batches.items() for k, x in quick_val(ev_m, bt, cfg).items()}
            v["elapsed_min"] = round((time.time() - t0) / 60, 3)
            if "hold_l1" in v:     # selection score: l1 + ms-ssim surrogate + spectrum, each relative to lr
                sc = v["hold_l1"] / v["hold_l1_lr"] + v["hold_ssim"] / v["hold_ssim_lr"] + v["hold_spec"] / v["hold_spec_lr"]
                v["hold_score"] = sc
                if sc < best[0]:
                    best = (sc, copy.deepcopy(ev_m.state_dict()), step)
            curve.append({"step": step, **v})
            log(step, {"val": v})
            print("  val " + " ".join(f"{k}={x:.5g}" for k, x in v.items()), flush=True)
    final = ema if ema is not None else model
    if best[1] is not None:
        final.load_state_dict(best[1])
        print(f"selected step {best[2]} (hold-out score {best[0]:.4f})", flush=True)
    ck = run_dir / "model.pt"
    save_screen_net(final, ck, {"dataset": cfg["dataset"], "step": step, "run": run_dir.name, "init": cfg.get("init")})
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ck, CKPT_DIR / f"{run_dir.name}.pt")
    summary = {"selected_step": best[2] if best[1] is not None else step, "steps": step, "minutes": round((time.time() - t0) / 60, 2), "params": param_count(final),
               "val_curve_last": curve[-1] if curve else None}
    del data, test, val_batches, hold
    torch.cuda.empty_cache()
    try:
        ev = final_eval(ck, cfg)
        summary["eval"] = ev
        summary["beats_both_baselines_test"] = beats_baselines(ev)
    except Exception as e:  # noqa: BLE001
        summary["eval"] = f"failed: {e!r}"
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (run_dir / "curve.json").write_text(json.dumps(curve, indent=1))
    with open(EXPERIMENTS, "a", encoding="utf-8") as f:
        f.write(f"\n## {run_dir.name}  ({time.strftime('%Y-%m-%d %H:%M')})\n"
                f"- config: `{cfg.get('_config_path', '?')}`  dataset: `{cfg['dataset']}`  init: `{cfg.get('init')}`\n"
                f"- hypothesis: {cfg.get('hypothesis') or '(none given)'}\n"
                f"- result: {json.dumps(_compact(summary), default=str)}\n"
                "- decision: (to be filled in by the ML engineer)\n")
    print(json.dumps(summary, indent=1, default=str))
    print(f"checkpoint: {ck} (copy: {CKPT_DIR / (run_dir.name + '.pt')}); promote with vfx.train3d.promote")
    return summary


def _compact(s):
    out = {k: s.get(k) for k in ("steps", "selected_step", "minutes", "params")}
    ev = s.get("eval")
    if isinstance(ev, dict):
        for sp in ("test", "ood"):
            if sp in ev:
                out[sp] = {n: {k: v for k, v in m.items() if k in ("l1", "spectrum_error", "ms_dssim", "flicker_ratio",
                                                                     "jerk_ratio", "swim_ratio_moving")}
                           for n, m in ev[sp].items() if isinstance(m, dict)}
        out["beats_both_baselines_test"] = s.get("beats_both_baselines_test")
    else:
        out["eval"] = ev
    out["val_last"] = s.get("val_curve_last")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--minutes", type=float, default=None)
    a = ap.parse_args(argv)
    cfg = _merge(DEFAULTS, json.loads(Path(a.config).read_text()))
    cfg["_config_path"] = a.config
    if a.minutes is not None:
        cfg["train"]["minutes"] = a.minutes
    train(cfg)


if __name__ == "__main__":
    main()
