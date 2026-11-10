"""Detail-oracle controls on real data: validate the Layer 4 detail gates (metrics.DETAIL_GATE_METRICS).

Dataset $VFX_DATASET3D (default s3d_v1), test split, 12 whole sequences (non-overlapping, evenly spaced).
Controls, all scored against hr on the item's effect region exactly as evaluate_screen_detail scores a model:
  lr              the unmodified lr render
  det{s}          lr + D shifted by s px diagonally (s = 0, 1, 2, 4), D = hr - G_sigma2(hr): the reference's OWN fine
                  detail, i.e. perfectly plausible filaments, exactly registered (s = 0) or displaced
  blur            lr blurred (sigma 1 px)
  sharpen         lr through the unsharp mask tuned on train (the Layer 4 baseline)
Requirements (lead, 2026-11-09): on every gate metric each det{s} beats lr, blur and sharpen, and det2 / det4 score
nearly as well as det0 (<= 1.25x det0, or better). The pixel-aligned metrics are printed to document why they are no
longer gates (det2 ~ lr on l1 / ms_dssim; det0 worse than lr on full-band spectrum_error).
Measured 2026-11-09 (12 seqs): l1 lr 0.0393, det0/1/2/4 0.0361/0.0372/0.0389/0.0410; ms_dssim lr 0.198,
det 0.141/0.162/0.195/0.228; spectrum_error lr 0.266, det0 0.366, det1 0.186.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from vfx.eval3d import metrics as M

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
DEV = "cuda"
VERSION = os.environ.get("VFX_DATASET3D", "s3d_v1")
N_SEQ = 12
NEAR = 1.25
SHIFTS = (0, 1, 2, 4)


@pytest.fixture(scope="module")
def oracle():
    try:
        ds = M._load(VERSION, "test", 16, stride=16)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"dataset {VERSION} test split not available: {e!r}")
    if len(ds) == 0:
        pytest.skip("empty test split")
    sh = M.tune_unsharp(VERSION, "train")
    q = lambda x: (x.clamp(0, 1) * 255 + 0.5).to(torch.uint8)
    acc: dict = {}
    for i in M._subsample(len(ds), N_SEQ):
        it = ds[i]
        lr = M.to_float(torch.as_tensor(it["lr"]).to(DEV))
        hr = M.to_float(torch.as_tensor(it["hr"]).to(DEV))
        aux = torch.as_tensor(it["aux"]).to(DEV)
        mask, _ = M.item_masks(it, aux, DEV)
        det = hr - M.gaussian_blur(hr, 2.0)
        seqs = {"lr": q(lr), "blur": q(M.gaussian_blur(lr, 1.0)), "sharpen": M.unsharp(q(lr), sh["sigma"], sh["amount"])}
        for s in SHIFTS:
            seqs[f"det{s}"] = q(lr + (torch.roll(det, (s, s), dims=(-2, -1)) if s else det))
        ref = q(hr)
        fr = M.sequence_flow(ref)
        for n, x in seqs.items():
            r = {"l1": M.l1(x, ref, mask), "ms_dssim": M.ms_dssim(x, ref, mask),
                 "spectrum_error": M.spectrum_error(x, ref, mask)}
            r.update(M.fair_detail_scores(x, ref, mask, flow_ref=fr))
            for k, v in r.items():
                acc.setdefault(n, {}).setdefault(k, []).append(v)
    res = {n: {k: float(np.nanmean(v)) for k, v in d.items()} for n, d in acc.items()}
    keys = ("l1", "ms_dssim", "spectrum_error") + M.DETAIL_GATE_METRICS + ("tof",)
    print("\n" + "control".ljust(9) + "".join(k[:14].rjust(15) for k in keys))
    for n, r in res.items():
        print(n.ljust(9) + "".join(f"{r[k]:15.4f}" for k in keys))
    return res


@pytest.mark.parametrize("metric", M.DETAIL_GATE_METRICS)
def test_oracle_detail_passes_gate_at_every_shift(oracle, metric):
    for s in SHIFTS:
        d = oracle[f"det{s}"][metric]
        for base in ("lr", "blur", "sharpen"):
            assert d < oracle[base][metric], f"{metric}: reference detail shifted {s}px {d:.4f} >= {base} {oracle[base][metric]:.4f}"


@pytest.mark.parametrize("metric", M.DETAIL_GATE_METRICS)
def test_shifted_oracle_nearly_as_good_as_registered(oracle, metric):
    d0 = oracle["det0"][metric]
    for s in (2, 4):
        assert oracle[f"det{s}"][metric] <= NEAR * d0, \
            f"{metric}: shift {s}px {oracle[f'det{s}'][metric]:.4f} > {NEAR} x registered {d0:.4f} (metric needs registration)"


def test_pixel_metrics_cannot_see_displaced_detail(oracle):
    """Documents the reason for the gate change (not a requirement on the code under test)."""
    o = oracle
    print(f"ms_dssim det2/lr {o['det2']['ms_dssim'] / o['lr']['ms_dssim']:.3f}, det4/lr {o['det4']['ms_dssim'] / o['lr']['ms_dssim']:.3f}; "
          f"l1 det2/lr {o['det2']['l1'] / o['lr']['l1']:.3f}; spectrum_error det0/lr {o['det0']['spectrum_error'] / o['lr']['spectrum_error']:.3f}")
    assert o["det4"]["ms_dssim"] > o["lr"]["ms_dssim"] or o["det4"]["l1"] > o["lr"]["l1"]
