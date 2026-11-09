"""Config-driven training of DetailNet (non-adversarial M3 baseline).

    .venv/Scripts/python -m vfx.train.train --config vfx/train/configs/demo.json
    .venv/Scripts/python -m vfx.train.train --config vfx/train/configs/synthetic.json --overfit 400

Each step samples random HR crops (with matching LR crops) of frames t-2, t-1, t and builds the
"previous frame" input per sample from one of three sources, to fight exposure bias:
  * teacher  : warp(hr[t-1] + relative noise)                     (prob 1 - p_cold - p_rollout)
  * rollout  : warp(model(lr[t-1], warp(hr[t-2])))  model's own output, no grad  (p_rollout)
  * cold     : bicubic(lr[t])  — what the runtime uses after reset()           (p_cold)
Loss (normalized units) = L1 + grad + high-frequency + radial log-spectrum + temporal-change
(temporal only on rollout samples, where it is not identical to L1).
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch

from vfx.core import paths
from vfx.core.fields import CH, DOMAIN_SIZE, HR_RES
from vfx.model import DetailNet, DetailNetConfig, param_count, save_model
from vfx.runtime.warp import upscale_lr_tensor, warp_prev
from vfx.train import losses as L
from vfx.train.data import CropSampler

DX_HR = DOMAIN_SIZE / HR_RES
_DT = slice(CH["density"], CH["temperature"] + 1, CH["temperature"] - CH["density"])
_VEL = slice(CH["vel_x"], CH["vel_y"] + 1)
EXPERIMENTS = Path(__file__).resolve().parent / "EXPERIMENTS.md"

DEFAULTS = {
    "name": "demo",
    "dataset": "v1",
    "hypothesis": "",
    "model": {},
    "train": {"minutes": 15.0, "max_steps": None, "batch": 16, "crop_hr": 256, "lr": 1e-3, "min_lr": 2e-5,
              "warmup_steps": 200, "weight_decay": 0.0, "grad_clip": 1.0, "amp": True, "seed": 0,
              "preload": True, "log_every": 50, "val_every": 500, "ckpt_every_minutes": 5.0, "dt": 1.0 / 30.0},
    "loss": {"l1": 1.0, "grad": 0.5, "hf": 1.0, "spec": 0.05, "temporal": 0.5},
    "exposure": {"p_cold": 0.1, "p_rollout": 0.4, "noise_sigma": 0.1},
    "eval": {"split": "test", "max_scenarios": 8, "max_frames": None},
}


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def _git_commit() -> str:
    try:
        g = paths.REPO_ROOT / ".git"
        head = (g / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            return (g / head.split()[1]).read_text().strip()[:12]
        return head[:12]
    except Exception:
        return "unknown"


class Logger:
    def __init__(self, run_dir: Path):
        self.f = open(run_dir / "log.jsonl", "a")
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(str(run_dir))
        except Exception:
            self.tb = None

    def log(self, step: int, d: dict):
        self.f.write(json.dumps({"step": step, **d}) + "\n")
        self.f.flush()
        if self.tb:
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, step)


def build_prev(model, lr, hr, ex, dt, generator=None):
    """Per-sample previous-frame input (physical units) and rollout mask. lr (B,2,5,h,w), hr (B,3,2,H,W)."""
    B = lr.shape[0]
    dev = lr.device
    lr_prev, lr_cur = lr[:, 0], lr[:, 1]
    vel = lr_cur[:, _VEL]
    # teacher + relative noise
    sig = torch.rand(B, 1, 1, 1, device=dev) * ex["noise_sigma"]
    pn = model.norm_hr(hr[:, 1])
    pn = (pn + sig * torch.randn_like(pn) * (pn.abs() + 0.01)).clamp_min(0)
    teacher = warp_prev(model.denorm_hr(pn), vel, DX_HR, dt)
    # rollout from model's own previous output
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        p1 = model(lr_prev, warp_prev(hr[:, 0], lr_prev[:, _VEL], DX_HR, dt)).float()
    roll = warp_prev(p1, vel, DX_HR, dt)
    cold = upscale_lr_tensor(lr_cur[:, _DT], "bicubic")
    r = torch.rand(B, 1, 1, 1, device=dev)
    is_cold = r < ex["p_cold"]
    is_roll = (r >= ex["p_cold"]) & (r < ex["p_cold"] + ex["p_rollout"])
    prev = torch.where(is_cold, cold, torch.where(is_roll, roll, teacher))
    return prev, roll, is_roll


def compute_loss(model, lr, hr, cfg):
    ex, wl, dt = cfg["exposure"], cfg["loss"], cfg["train"]["dt"]
    prev, roll, is_roll = build_prev(model, lr, hr, ex, dt)
    with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["train"]["amp"]):
        pred = model.forward_normalized(lr[:, 1], prev)
    pred = pred.float()
    tgt = model.norm_hr(hr[:, 2])
    terms = {"l1": (pred - tgt).abs().mean(), "grad": L.grad_l1(pred, tgt), "hf": L.hf_l1(pred, tgt),
             "spec": L.spectral_loss(pred, tgt)}
    if wl["temporal"] > 0 and bool(is_roll.any()):
        m = is_roll.view(-1)
        ref_prev = model.norm_hr(warp_prev(hr[:, 1], lr[:, 1, _VEL], DX_HR, dt))
        terms["temporal"] = L.temporal_l1(pred[m], model.norm_hr(roll[m]), tgt[m], ref_prev[m])
    total = sum(wl[k] * v for k, v in terms.items())
    return total, terms


@torch.no_grad()
def quick_val(model, sampler, cfg, n_batches=4):
    """Teacher-forced crop metrics on the held-out split: model vs bicubic (normalized units)."""
    model.eval()
    acc = {"val_l1": 0.0, "val_bic_l1": 0.0, "val_spec": 0.0, "val_bic_spec": 0.0}
    for _ in range(n_batches):
        lr, hr = sampler.fixed_batch()
        prev = warp_prev(hr[:, 1], lr[:, 1, _VEL], DX_HR, cfg["train"]["dt"])
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model(lr[:, 1], prev).float()
        bic = upscale_lr_tensor(lr[:, 1, _DT], "bicubic")
        pn, bn, tn = model.norm_hr(pred), model.norm_hr(bic), model.norm_hr(hr[:, 2])
        acc["val_l1"] += (pn - tn).abs().mean().item() / n_batches
        acc["val_bic_l1"] += (bn - tn).abs().mean().item() / n_batches
        acc["val_spec"] += L.spectral_loss(pn[:, :1], tn[:, :1]).item() / n_batches
        acc["val_bic_spec"] += L.spectral_loss(bn[:, :1], tn[:, :1]).item() / n_batches
    model.train()
    return acc


def append_experiment(cfg, run_dir, summary):
    lines = [f"\n## {run_dir.name}  ({time.strftime('%Y-%m-%d %H:%M')})\n",
             f"- config: `{cfg.get('_config_path', '?')}`  dataset: `{cfg['dataset']}`  commit: `{_git_commit()}`\n",
             f"- hypothesis: {cfg.get('hypothesis') or '(none given)'}\n",
             f"- result: {json.dumps(summary)}\n",
             "- decision: (to be filled in by the ML engineer)\n"]
    with open(EXPERIMENTS, "a", encoding="utf-8") as f:
        f.writelines(lines)


def train(cfg: dict, overfit_steps: int = 0) -> dict:
    tc = cfg["train"]
    torch.manual_seed(tc["seed"])
    torch.backends.cudnn.benchmark = True
    sampler = CropSampler(cfg["dataset"], "train", tc["crop_hr"], tc["batch"], tc["preload"], seed=tc["seed"])
    mcfg = dict(cfg["model"])
    if mcfg.get("vel_scale", "auto") == "auto":
        mcfg["vel_scale"] = round(max(2.0 * sampler.velocity_std(), 0.05), 4)
    model = DetailNet(DetailNetConfig(**mcfg)).cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=tc["amp"])
    run_dir = paths.RUNS / f"{cfg['name']}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    log = Logger(run_dir)
    n_params = param_count(model)
    print(f"run {run_dir}  params {n_params}  model {asdict(model.cfg)}", flush=True)

    val_sampler = None
    if not overfit_steps:
        try:
            val_sampler = CropSampler(cfg["dataset"], cfg["eval"]["split"], tc["crop_hr"], tc["batch"], False, seed=123)
        except Exception as e:
            print("no validation split:", e)

    fixed = sampler.fixed_batch() if overfit_steps else None
    budget = tc["minutes"] * 60.0
    max_steps = overfit_steps or tc["max_steps"] or 10 ** 9
    t0 = time.time()
    last_ck = t0
    step = 0
    hist = []
    n_bad = 0

    def save(tag="latest"):
        p = run_dir / "model.pt"
        save_model(model, p, {"dataset": cfg["dataset"], "step": step, "commit": _git_commit(), "run": run_dir.name})
        return p

    model.train()
    while step < max_steps:
        frac = min(1.0, (time.time() - t0) / budget) if not overfit_steps else step / max_steps
        if not overfit_steps and frac >= 1.0:
            break
        lr_now = tc["min_lr"] + 0.5 * (tc["lr"] - tc["min_lr"]) * (1 + math.cos(math.pi * frac))
        lr_now *= min(1.0, (step + 1) / tc["warmup_steps"])
        for g in opt.param_groups:
            g["lr"] = lr_now
        lr_b, hr_b = fixed if overfit_steps else sampler.next()
        if overfit_steps:   # pure fitting check: teacher-forced, no noise
            cfg_o = _merge(cfg, {"exposure": {"p_cold": 0.0, "p_rollout": 0.0, "noise_sigma": 0.0}})
            total, terms = compute_loss(model, lr_b, hr_b, cfg_o)
        else:
            total, terms = compute_loss(model, lr_b, hr_b, cfg)
        if not torch.isfinite(total):          # skip (and count) non-finite steps; abort if persistent
            n_bad += 1
            if n_bad >= 50:
                raise RuntimeError(f"50 consecutive non-finite losses at step {step}; aborting run {run_dir}")
            step += 1
            continue
        n_bad = 0
        opt.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        scaler.step(opt)
        scaler.update()
        step += 1
        if step % tc["log_every"] == 0 or step == 1:
            d = {"loss": total.item(), **{k: v.item() for k, v in terms.items()}, "lr": lr_now,
                 "elapsed_min": (time.time() - t0) / 60}
            hist.append(d)
            log.log(step, d)
            print(f"step {step:6d} " + " ".join(f"{k}={v:.4g}" for k, v in d.items()), flush=True)
        if val_sampler and step % tc["val_every"] == 0:
            v = quick_val(model, val_sampler, cfg)
            log.log(step, v)
            print("  val " + " ".join(f"{k}={x:.4g}" for k, x in v.items()), flush=True)
        if time.time() - last_ck > tc["ckpt_every_minutes"] * 60:
            save()
            last_ck = time.time()

    ck = save()
    summary = {"steps": step, "minutes": round((time.time() - t0) / 60, 2), "params": n_params,
               "final_loss": hist[-1]["loss"] if hist else None}
    if overfit_steps:
        summary["overfit_first_l1"] = hist[0]["l1"]
        summary["overfit_last_l1"] = hist[-1]["l1"]
        print(json.dumps(summary))
        return summary
    if val_sampler:
        summary.update({k: round(v, 5) for k, v in quick_val(model, val_sampler, cfg, 8).items()})
    try:
        from vfx.train.evaluate import evaluate
        ev = evaluate(ck, cfg["dataset"], cfg["eval"]["split"], cfg["eval"]["max_scenarios"],
                      max_frames=cfg["eval"]["max_frames"])
        summary["rollout_eval"] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in ev.items()}
    except Exception as e:
        summary["rollout_eval"] = f"failed: {e!r}"
    ev = summary["rollout_eval"]
    ok = isinstance(ev, dict) and math.isfinite(ev.get("net_l1_d", float("nan")))         and math.isfinite(ev.get("net_spec", float("nan")))
    summary["promoted_to_default"] = ok and cfg.get("promote", True)
    if summary["promoted_to_default"]:
        paths.default_checkpoint().parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ck, paths.default_checkpoint())
    else:
        print("WARNING: checkpoint NOT copied to the default path (non-finite eval or promote=false)")
    log.log(step, {"summary": summary})
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    append_experiment(cfg, run_dir, summary)
    print(json.dumps(summary, indent=1))
    print(f"checkpoint: {ck}" + (f"  (copied to {paths.default_checkpoint()})" if summary["promoted_to_default"] else ""))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overfit", type=int, default=0, help="N steps on one fixed batch (sanity check)")
    ap.add_argument("--minutes", type=float, default=None, help="override train.minutes")
    ap.add_argument("--dataset", default=None, help="override dataset version")
    a = ap.parse_args(argv)
    cfg = _merge(DEFAULTS, json.loads(Path(a.config).read_text()))
    cfg["_config_path"] = a.config
    if a.minutes is not None:
        cfg["train"]["minutes"] = a.minutes
    if a.dataset:
        cfg["dataset"] = a.dataset
    train(cfg, a.overfit)


if __name__ == "__main__":
    main()
