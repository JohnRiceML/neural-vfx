"""Train CorrNet on collected (features, target) pairs.

    .venv/Scripts/python -m vfx.correct3d.train --data r0[,r1] --out CKPT --minutes 6 [--init CKPT]

Loss: MSE of net(x) vs target / OUT_SCALE at 32^3, random horizontal symmetries. Validation: the
last `--val` scenes (by file order) of the first dataset; reports R^2 = 1 - MSE / E[y^2] (0 = the identity).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .common import CORR_DATASETS as DATASETS
from .model import OUT_SCALE, CorrNet, augment


def load(names: list[str], val: int):
    tr_x, tr_y, va_x, va_y = [], [], [], []
    for i, n in enumerate(names):
        files = sorted((DATASETS / f"corr3d_{n}").glob("*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
        for j, fp in enumerate(files):
            d = torch.load(fp, mmap=True)
            is_val = j >= len(files) - val
            (va_x if is_val else tr_x).append(d["x"].cuda())     # mmap -> GPU per file (RAM is tight)
            (va_y if is_val else tr_y).append(d["y"].cuda())
            del d
    cat = lambda L: torch.cat(L) if L else None
    return cat(tr_x), cat(tr_y), cat(va_x), cat(va_y)


@torch.no_grad()
def r2(net, x, y, bs=16) -> float:
    se = sq = 0.0
    for i in range(0, x.shape[0], bs):
        with torch.autocast("cuda", dtype=torch.float16):
            p = net(x[i:i + bs].float())
        t = y[i:i + bs].float() / OUT_SCALE
        se += float((p.float() - t).square().sum())
        sq += float(t.square().sum())
    return 1.0 - se / max(sq, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=6.0)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val", type=int, default=3)
    ap.add_argument("--init", default=None)
    a = ap.parse_args()
    torch.manual_seed(0)
    x, y, vx, vy = load(a.data.split(","), a.val)
    print(f"train {x.shape[0]} pairs, val {vx.shape[0]}; target rms {float(y.float().square().mean().sqrt()):.4f} m/s/step",
          flush=True)
    net = CorrNet(a.width).cuda()
    if a.init:
        net.load_state_dict(torch.load(a.init, weights_only=False)["state_dict"])
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler()
    t0 = time.time()
    total = a.minutes * 60
    it = 0
    log = []
    while time.time() - t0 < total:
        frac = (time.time() - t0) / total
        for gp in opt.param_groups:
            gp["lr"] = a.lr * 0.5 * (1 + torch.cos(torch.tensor(frac * 3.14159)).item())
        idx = torch.randint(0, x.shape[0], (a.bs,), device="cuda")
        fb, yb = augment(x[idx].float(), y[idx].float())
        with torch.autocast("cuda", dtype=torch.float16):
            p = net(fb)
        loss = (p.float() - yb / OUT_SCALE).square().mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        it += 1
        if it % 500 == 0:
            net.eval()
            v = r2(net, vx, vy)
            net.train()
            log.append({"it": it, "s": round(time.time() - t0), "loss": float(loss), "val_r2": v})
            print(f"it {it} {time.time() - t0:.0f}s loss {float(loss):.4f} val R2 {v:.3f}", flush=True)
    net.eval()
    res = {"train_r2": r2(net, x[:256], y[:256]), "val_r2": r2(net, vx, vy), "iters": it}
    print(res, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "width": a.width, "meta": {**vars(a), **res, "log": log}}, a.out)
    Path(a.out).with_suffix(".json").write_text(json.dumps({**vars(a), **res, "log": log}, indent=1))


if __name__ == "__main__":
    main()
