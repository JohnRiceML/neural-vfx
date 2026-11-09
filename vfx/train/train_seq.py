"""Sequence (self-feedback) training of DetailNet on dataset v2 with the 128^2 sim (`lr_sim`) as input.

    .venv/Scripts/python -m vfx.train.train_seq --config vfx/train/configs/v2_seq.json

Each sample is a window of K+1 consecutive frames (random 256^2 HR crop, matching 64^2 LR crop,
h-flip with vel_x negated). The model is unrolled exactly like the runtime Upscaler:
  frame 0: cold start, prev = bicubic(lr_sim[0])              (p_cold; as after Upscaler.reset())
           or teacher start, prev = warp(hr[0] + noise)       (then frame 0 is not predicted)
  frame k: pred_k = model(lr_sim[k], warp_prev(pred_{k-1}, vel lr_sim[k-1]))
with gradient through the whole chain (optionally truncated every `tbptt` steps).
Loss per predicted frame (normalized units, a margin of the crop is ignored because the warp clamps
at the crop border, unlike the full frame): L1 + grad + HF + low-band log spectrum (as run 3) +
  temporal : change matching  (pred_k - W pred_{k-1}) vs (hr_k - W hr_{k-1}), both warped by the
             lr_sim velocity (v2 pairs follow the same flow);
  flicker  : hinge  relu(F(pred) - alpha * F(bicubic)) per sample, F = mean |x_k - W x_{k-1}| of
             density — the gate's flicker, judged against the bicubic upscale of the same input.
At the end: scorecard (v2 test/ood with lr_sim via vfx.eval.metrics + live flicker) appended to
EXPERIMENTS.md. Never auto-promotes; promotion to paths.default_checkpoint() is a manual decision.
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from vfx.core import paths
from vfx.core.fields import CH, DOMAIN_SIZE, HR_RES, UPSCALE
from vfx.model import DetailNet, DetailNetConfig, load_model, param_count, save_model
from vfx.runtime.warp import upscale_lr_tensor, warp_prev
from vfx.train import losses as L
from vfx.train.train import Logger, _git_commit, _merge

DX_HR = DOMAIN_SIZE / HR_RES
_DT = slice(CH["density"], CH["temperature"] + 1, CH["temperature"] - CH["density"])
_VEL = slice(CH["vel_x"], CH["vel_y"] + 1)
EXPERIMENTS = Path(__file__).resolve().parent / "EXPERIMENTS.md"

DEFAULTS = {
    "name": "v2_seq", "dataset": "v2", "hypothesis": "", "init": None, "model": {},
    "train": {"minutes": 15.0, "batch": 8, "crop_hr": 256, "steps_k": 6, "tbptt": 0, "lr": 3e-4, "min_lr": 1e-5,
              "warmup_steps": 100, "grad_clip": 1.0, "seed": 0, "preload": True, "log_every": 50,
              "val_every": 500, "ckpt_every_minutes": 5.0, "dt": 1.0 / 30.0, "margin": 16, "lr_key": "lr_sim", "p_pooled": 0.0},
    "loss": {"l1": 1.0, "grad": 0.5, "hf": 1.0, "spec": 0.002, "spec_max_frac": 0.5, "temporal": 0.5,
             "flicker": 4.0, "flicker_alpha": 0.8},
    "exposure": {"p_cold": 0.5, "noise_sigma": 0.1},
}


class SeqCropSampler:
    """Random windows of K+1 frames: lr (B,K+1,5,h,w) from `lr_key`, hr (B,K+1,2,H,W), GPU float32."""

    def __init__(self, version, split, length, crop_hr=256, batch=8, preload=True, hflip=True, seed=0,
                 lr_key="lr_sim", active_thresh=0.01, active_tries=4, p_pooled=0.0):
        """p_pooled > 0: per sample, the input is the pooled HR (`lr.npy`) with this probability instead
        of `lr_key` (keeps the model calibrated for both input distributions)."""
        from vfx.datagen import load_dataset
        ds = load_dataset(version, split, include_lr_sim=(lr_key == "lr_sim"))
        mm = None if preload else "r"
        self.p_pooled = p_pooled
        self.scn, self.pooled = [], []
        for s in ds.scenarios:
            if "nonfinite" in s.get("flags", []) or s["frames"] < length:
                continue
            self.scn.append((np.load(s["dir"] / "hr.npy", mmap_mode=mm), np.load(s["dir"] / f"{lr_key}.npy", mmap_mode=mm)))
            if p_pooled > 0:
                self.pooled.append(np.load(s["dir"] / "lr.npy", mmap_mode=mm))
        if not self.scn:
            raise RuntimeError(f"{version}/{split}: no scenarios with >= {length} frames")
        self.frames = np.array([a[0].shape[0] for a in self.scn])
        w = (self.frames - length + 1).astype(np.float64)
        self.p = w / w.sum()
        self.L, self.B, self.hflip = length, batch, hflip
        self.H = self.scn[0][0].shape[-1]
        self.ch = min(crop_hr, self.H)
        self.cl = self.ch // UPSCALE
        self.rng = np.random.default_rng(seed)
        self.thresh, self.tries = active_thresh, active_tries
        self._q = None

    def _sample_np(self):
        B, Lw, cl, ch = self.B, self.L, self.cl, self.ch
        lrr = self.H // UPSCALE
        lrs = np.empty((B, Lw, 5, cl, cl), np.float16)
        hrs = np.empty((B, Lw, 2, ch, ch), np.float16)
        flips = np.zeros(B, bool)
        for i, k in enumerate(self.rng.choice(len(self.scn), size=B, p=self.p)):
            hr, lr = self.scn[k]
            if self.p_pooled > 0 and self.rng.random() < self.p_pooled:
                lr = self.pooled[k]
            t0 = int(self.rng.integers(0, self.frames[k] - Lw + 1))
            for _ in range(self.tries):
                y = int(self.rng.integers(0, lrr - cl + 1))
                x = int(self.rng.integers(0, lrr - cl + 1))
                if lr[t0 + Lw - 1, 0, y:y + cl, x:x + cl].max() > self.thresh:
                    break
            lrs[i] = lr[t0:t0 + Lw, :, y:y + cl, x:x + cl]
            hrs[i] = hr[t0:t0 + Lw, :, y * UPSCALE:y * UPSCALE + ch, x * UPSCALE:x * UPSCALE + ch]
            flips[i] = self.hflip and self.rng.random() < 0.5
        return lrs, hrs, flips

    def _to_gpu(self, lrs, hrs, flips):
        lr = torch.from_numpy(lrs).pin_memory().cuda(non_blocking=True).float()
        hr = torch.from_numpy(hrs).pin_memory().cuda(non_blocking=True).float()
        if flips.any():                       # mirror x, negate vel_x (vertical flip is not valid: buoyancy)
            f = torch.from_numpy(flips).cuda().view(-1, 1, 1, 1, 1)
            sign = torch.ones(1, 1, 5, 1, 1, device="cuda")
            sign[:, :, CH["vel_x"]] = -1.0
            lr = torch.where(f, lr.flip(-1) * sign, lr)
            hr = torch.where(f, hr.flip(-1), hr)
        return lr, hr

    def next(self):
        if self._q is None:
            self._q = queue.Queue(maxsize=6)
            for _ in range(3):
                threading.Thread(target=lambda: [self._q.put(self._sample_np()) for _ in iter(int, 1)], daemon=True).start()
        return self._to_gpu(*self._q.get())

    def batch_det(self):
        return self._to_gpu(*self._sample_np())


def _flick(x, x_prev_w):
    """Per-sample mean |x_k - W x_{k-1}| of density (normalized). (B,)"""
    return (x[:, 0] - x_prev_w[:, 0]).abs().mean(dim=(-2, -1))


def rollout_loss(model, lr, hr, cfg, train=True):
    """Unroll the model over the window. Returns (total, terms, stats)."""
    tc, wl, ex = cfg["train"], cfg["loss"], cfg["exposure"]
    dt, m, K1 = tc["dt"], tc["margin"], lr.shape[1]
    B = lr.shape[0]
    c = (slice(None), slice(None), slice(m, -m), slice(m, -m)) if m > 0 else (slice(None),) * 4
    cold = torch.rand(B, device=lr.device) < ex["p_cold"]
    # previous-frame input for frame 1 (teacher) / frame 0 (cold)
    sig = torch.rand(B, 1, 1, 1, device=lr.device) * ex["noise_sigma"]
    h0 = model.norm_hr(hr[:, 0])
    h0 = (h0 + sig * torch.randn_like(h0) * (h0.abs() + 0.01)).clamp_min(0)
    teacher_prev = warp_prev(model.denorm_hr(h0), lr[:, 0, _VEL], DX_HR, dt)
    bic = [upscale_lr_tensor(lr[:, k, _DT], "bicubic") for k in range(K1)]
    with torch.autocast("cuda", dtype=torch.float16):
        p0 = model.forward_normalized(lr[:, 0], bic[0]).float()
    # frame-0 output: cold samples use the prediction, teacher samples the (noisy) HR frame 0
    prev_out_n = torch.where(cold.view(-1, 1, 1, 1), p0, h0)
    terms = {k: 0.0 for k in ("l1", "grad", "hf", "spec", "temporal", "flicker")}
    n_rec = 0
    st = {"f_pred": 0.0, "f_bic": 0.0, "f_hr": 0.0, "l1_bic": 0.0}

    def rec(pn, tn, w=None):
        nonlocal n_rec
        terms["l1"] = terms["l1"] + (pn - tn).abs().mean()
        terms["grad"] = terms["grad"] + L.grad_l1(pn, tn)
        terms["hf"] = terms["hf"] + L.hf_l1(pn, tn)
        terms["spec"] = terms["spec"] + L.spectral_loss(pn, tn, max_frac=wl["spec_max_frac"])
        n_rec += 1

    # frame 0 reconstruction only for cold samples (teacher samples copy HR)
    if bool(cold.any()):
        cm = cold
        rec(p0[cm][c], model.norm_hr(hr[cm, 0])[c])
    tbptt = tc.get("tbptt", 0) or 0
    for k in range(1, K1):
        vel = lr[:, k - 1, _VEL]
        prev_phys = model.denorm_hr(model.clamp_normalized(prev_out_n))
        if tbptt and k % tbptt == 0:
            prev_phys = prev_phys.detach()
        prev_w = warp_prev(prev_phys, vel, DX_HR, dt)
        with torch.autocast("cuda", dtype=torch.float16):
            pn = model.forward_normalized(lr[:, k], prev_w).float()
        tn = model.norm_hr(hr[:, k])
        rec(pn[c], tn[c])
        pw_n = model.norm_hr(prev_w)
        hw_n = model.norm_hr(warp_prev(hr[:, k - 1], vel, DX_HR, dt))
        bn, bpw = model.norm_hr(bic[k]), model.norm_hr(warp_prev(bic[k - 1], vel, DX_HR, dt))
        terms["temporal"] = terms["temporal"] + L.temporal_l1(pn[c], pw_n[c], tn[c], hw_n[c])
        fp, fb = _flick(pn[c], pw_n[c]), _flick(bn[c], bpw[c])
        terms["flicker"] = terms["flicker"] + torch.relu(fp - wl["flicker_alpha"] * fb).mean() / fb.mean().clamp_min(1e-6) * 1e-3
        with torch.no_grad():
            st["f_pred"] += fp.mean().item() if not train else 0.0
            st["f_bic"] += fb.mean().item() if not train else 0.0
            st["f_hr"] += _flick(tn[c], hw_n[c]).mean().item() if not train else 0.0
            st["l1_bic"] += (bn[c] - tn[c]).abs().mean().item() if not train else 0.0
        prev_out_n = pn
    nk = K1 - 1
    for kk in ("l1", "grad", "hf", "spec"):
        terms[kk] = terms[kk] / max(n_rec, 1)
    terms["temporal"] = terms["temporal"] / nk
    terms["flicker"] = terms["flicker"] / nk
    total = sum(wl[k] * v for k, v in terms.items())
    if not train:
        st = {k: v / nk for k, v in st.items()}
    return total, terms, st


@torch.no_grad()
def seq_val(model, sampler, cfg, n=4):
    model.eval()
    acc = {"val_l1": 0.0, "val_bic_l1": 0.0, "val_flick_pred_over_bic": 0.0, "val_flick_hr_over_bic": 0.0}
    for _ in range(n):
        lr, hr = sampler.batch_det()
        cfg_v = _merge(cfg, {"exposure": {"p_cold": 1.0, "noise_sigma": 0.0}})
        _, terms, st = rollout_loss(model, lr, hr, cfg_v, train=False)
        acc["val_l1"] += float(terms["l1"]) / n
        acc["val_bic_l1"] += st["l1_bic"] / n
        acc["val_flick_pred_over_bic"] += st["f_pred"] / max(st["f_bic"], 1e-9) / n
        acc["val_flick_hr_over_bic"] += st["f_hr"] / max(st["f_bic"], 1e-9) / n
    model.train()
    return acc


def train(cfg):
    tc = cfg["train"]
    torch.manual_seed(tc["seed"])
    torch.backends.cudnn.benchmark = True
    K1 = tc["steps_k"] + 1
    print("loading data ...", flush=True)
    sampler = SeqCropSampler(cfg["dataset"], "train", K1, tc["crop_hr"], tc["batch"], tc["preload"], seed=tc["seed"],
                             lr_key=tc["lr_key"], p_pooled=tc.get("p_pooled", 0.0))
    val = SeqCropSampler(cfg["dataset"], "test", K1, tc["crop_hr"], tc["batch"], False, hflip=False, seed=123,
                         lr_key=tc["lr_key"])
    if cfg.get("init"):
        model = load_model(cfg["init"]).train()
    else:
        mcfg = dict(cfg["model"])
        if mcfg.get("vel_scale", "auto") == "auto":
            v = sampler._sample_np()[0][:, :, _VEL].astype(np.float32)
            mcfg["vel_scale"] = round(max(2.0 * float(v.std()), 0.05), 4)
        model = DetailNet(DetailNetConfig(**mcfg)).cuda()
    model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"])
    scaler = torch.amp.GradScaler("cuda")
    run_dir = paths.RUNS / f"{cfg['name']}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    log = Logger(run_dir)
    print(f"run {run_dir}  params {param_count(model)}  model {asdict(model.cfg)}", flush=True)
    budget, t0, last_ck, step, n_bad = tc["minutes"] * 60, time.time(), time.time(), 0, 0
    ck = run_dir / "model.pt"

    def save():
        save_model(model, ck, {"dataset": cfg["dataset"], "step": step, "commit": _git_commit(), "run": run_dir.name,
                               "init": cfg.get("init"), "lr_key": tc["lr_key"]})

    hist = []
    model.train()
    while True:
        frac = (time.time() - t0) / budget
        if frac >= 1.0:
            break
        lr_now = (tc["min_lr"] + 0.5 * (tc["lr"] - tc["min_lr"]) * (1 + math.cos(math.pi * frac))) \
            * min(1.0, (step + 1) / tc["warmup_steps"])
        for g in opt.param_groups:
            g["lr"] = lr_now
        lr_b, hr_b = sampler.next()
        total, terms, _ = rollout_loss(model, lr_b, hr_b, cfg)
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
        step += 1
        if step % tc["log_every"] == 0 or step == 1:
            d = {"loss": total.item(), **{k: float(v) for k, v in terms.items()}, "lr": lr_now,
                 "elapsed_min": (time.time() - t0) / 60}
            hist.append(d)
            log.log(step, d)
            print(f"step {step:6d} " + " ".join(f"{k}={v:.4g}" for k, v in d.items()), flush=True)
        if step % tc["val_every"] == 0:
            v = seq_val(model, val, cfg)
            log.log(step, v)
            print("  val " + " ".join(f"{k}={x:.4g}" for k, x in v.items()), flush=True)
        if time.time() - last_ck > tc["ckpt_every_minutes"] * 60:
            save()
            last_ck = time.time()
    save()
    summary = {"steps": step, "minutes": round((time.time() - t0) / 60, 2), "params": param_count(model),
               "final_loss": hist[-1]["loss"] if hist else None}
    summary.update({k: round(v, 5) for k, v in seq_val(model, val, cfg, 8).items()})
    del sampler
    try:
        from vfx.train.evaluate import scorecard
        sc = scorecard(ck, cfg["dataset"])
        summary["scorecard"] = _compact(sc)
    except Exception as e:
        summary["scorecard"] = f"failed: {e!r}"
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    with open(EXPERIMENTS, "a", encoding="utf-8") as f:
        f.write(f"\n## {run_dir.name}  ({time.strftime('%Y-%m-%d %H:%M')})\n"
                f"- config: `{cfg.get('_config_path', '?')}`  dataset: `{cfg['dataset']}` input `{tc['lr_key']}`  "
                f"init: `{cfg.get('init')}`  commit: `{_git_commit()}`\n"
                f"- hypothesis: {cfg.get('hypothesis') or '(none given)'}\n"
                f"- result: {json.dumps(summary, default=str)}\n"
                "- decision: (to be filled in by the ML engineer)\n")
    print(json.dumps(summary, indent=1, default=str))
    print(f"checkpoint: {ck}  (not promoted; compare scorecards and copy manually)")
    return summary


def _compact(sc):
    out = {}
    for k, r in sc.items():
        if k == "live":
            for s, v in r.items():
                out[f"live/{s}"] = {"ratio_neural": round(v.get("ratio_neural", float("nan")), 3),
                                    "ratio_bicubic": round(v["ratio_bicubic"], 3),
                                    "neural_over_bicubic": round(v.get("neural_over_bicubic", float("nan")), 3)}
        else:
            out[k] = {"flicker_ratio": round(r["flicker_ratio_vs_ref"], 3),
                      "bic_flicker_ratio": round(r["bic_flicker_ratio_vs_ref"], 3),
                      "spec": round(r["net_spec"], 4), "bic_spec": round(r["bic_spec"], 4),
                      "spec_improvement": round(r["spec_improvement"], 3),
                      "l1_d": round(r["net_l1_d"], 6), "bic_l1_d": round(r["bic_l1_d"], 6),
                      "l1_improvement": round(r["l1_improvement"], 4)}
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
