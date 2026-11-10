"""Unit tests + controls for the distribution-level and reference-relative temporal metrics in
vfx/eval3d/metrics.py (sim-carried detail phase): band_spectrum, gradient / local-contrast histogram distances,
patch SWD, tLP (+ log ratio, pop) and tOF.

Synthetic "plume" sequence (no dataset, no other 3D module): a soft blob translating (1.5, -1.0) px/frame with a
band-pass noise detail layer modulated by the blob. In the REFERENCE the detail is glued to the blob (sampled at
p - v t) and evolves slowly (rotation between two noise fields, 0.06 rad/frame). Controls:
  indep   an independent realization of the same detail statistics, also glued and evolving ("procedural detail
          done right": never pixel-aligned with the reference)
  blur    the reference blurred (Gaussian sigma 1.5 px)
  noise   the reference + iid per-frame noise (std 0.03)
  stuck   the same detail field, but fixed in SCREEN space while the blob moves ("stuck to the screen")
  pop     glued detail whose noise field is re-rolled every 6 frames (periodic regeneration pop)
Requirements (from the task contract): the reference scores best (0 / ratio 1); blur, noise, stuck and pop score
worse than the reference AND worse than `indep` on the metric meant to catch them; `indep` beats blur on every
distribution metric even though l1 / MS-DSSIM prefer the blur (the reason these metrics exist).
Margins stated here (no spec number exists): a control must score >= 1.5x `indep` (or exceed it by the stated
absolute margin where `indep` is ~0).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vfx.eval3d import metrics as M

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
DEV = "cuda"
H, W, L = 160, 192, 24
V = (1.5, -1.0)
MARGIN = 1.5


def _noise_field(seed: int, size: int = 320) -> torch.Tensor:
    """Band-pass noise (difference of Gaussians, sigma 1.0 and 3.5 px), unit std, (size, size)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(1, 1, size, size, generator=g).to(DEV)
    f = M.gaussian_blur(z.expand(1, 3, size, size), 1.0)[:, :1] - M.gaussian_blur(z.expand(1, 3, size, size), 3.5)[:, :1]
    f = f[0, 0]
    return (f - f.mean()) / f.std()


def _sample(field: torch.Tensor, shift) -> torch.Tensor:
    """field sampled at (p - shift) for the (H, W) frame grid, bilinear (field is larger; frame at its centre)."""
    S = field.shape[0]
    ys = torch.arange(H, device=DEV, dtype=torch.float32) + (S - H) / 2 - shift[1]
    xs = torch.arange(W, device=DEV, dtype=torch.float32) + (S - W) / 2 - shift[0]
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([2 * gx / (S - 1) - 1, 2 * gy / (S - 1) - 1], -1)[None]
    return F.grid_sample(field[None, None], grid, mode="bilinear", align_corners=True)[0, 0]


def plume(mode: str = "ref"):
    """(L,3,H,W) float frames and (L,H,W) alpha."""
    seeds = {"indep": (11, 12)}.get(mode, (1, 2))
    fa, fb = _noise_field(seeds[0]), _noise_field(seeds[1])
    ys = torch.arange(H, device=DEV, dtype=torch.float32)[:, None]
    xs = torch.arange(W, device=DEV, dtype=torch.float32)[None, :]
    frames, alphas = [], []
    g = torch.Generator(device="cpu").manual_seed(99)
    for t in range(L):
        sh = (V[0] * t, V[1] * t)
        cx, cy = W * 0.4 + sh[0], H * 0.6 + sh[1]
        env = torch.exp(-(((xs - cx) / 38.0) ** 2 + ((ys - cy) / 30.0) ** 2))
        th = 0.06 * t
        if mode == "pop":
            k = t // 6
            fa, fb = _noise_field(100 + 2 * k), _noise_field(101 + 2 * k)
        dsh = (0.0, 0.0) if mode == "stuck" else sh
        d = math.cos(th) * _sample(fa, dsh) + math.sin(th) * _sample(fb, dsh)
        y = 0.12 + 0.55 * env + 0.09 * env * d
        img = torch.stack([y * 1.05, y, y * 0.92]).clamp(0, 1)
        if mode == "noise":
            img = (img + 0.03 * torch.randn(img.shape, generator=g).to(DEV)).clamp(0, 1)
        frames.append(img)
        alphas.append(env)
    x = torch.stack(frames)
    if mode == "blur":
        x = M.gaussian_blur(x, 1.5)
    return x, torch.stack(alphas)


@pytest.fixture(scope="module")
def seqs():
    ref, alpha = plume("ref")
    mask = M.effect_mask(alpha)
    out = {m: plume(m)[0] for m in ("indep", "blur", "noise", "stuck", "pop")}
    return ref, mask, out


@pytest.fixture(scope="module")
def scores(seqs):
    ref, mask, out = seqs
    fr = M.sequence_flow(ref)
    res = {"ref": M.fair_detail_scores(ref, ref, mask, flow_ref=fr)}
    for k, x in out.items():
        res[k] = M.fair_detail_scores(x, ref, mask, flow_ref=fr)
        res[k]["l1"] = M.l1(x, ref, mask)
        res[k]["ms_dssim"] = M.ms_dssim(x, ref, mask)
        res[k]["flicker_ratio_legacy"] = M.flicker(x, None, mask) / M.flicker(ref, None, mask)
    keys = ["l1", "ms_dssim", "band_distance", "band_2-4px", "grad_hist", "contrast_hist", "patch_swd", "tlp",
            "tlp_log_ratio", "pop", "tof", "flicker_ratio_legacy"]
    print("\n" + "control".ljust(8) + "".join(k[:13].rjust(14) for k in keys))
    for name, r in res.items():
        print(name.ljust(8) + "".join(f"{r.get(k, float('nan')):14.4f}" for k in keys))
    return res


# ----------------------------------------------------------------------------- unit checks
def test_wasserstein1_known_values():
    a = torch.linspace(0, 1, 1001, device=DEV)
    assert M.wasserstein1(a, a) == 0.0
    assert abs(M.wasserstein1(a, a + 0.25) - 0.25) < 1e-5
    assert abs(M.wasserstein1(a, 2 * a) - 0.5) < 2e-3           # E|x - 2x| = E x = 0.5
    g = torch.Generator(device="cpu").manual_seed(0)
    b = a[torch.randperm(1001, generator=g).to(DEV)][:400]       # different size, same distribution
    assert M.wasserstein1(a, b) < 0.04                           # sampling error of 400 draws ~ 0.015


def test_identity_is_zero(seqs):
    ref, mask, _ = seqs
    r = M.fair_detail_scores(ref, ref, mask)
    for k in ("band_distance", "grad_hist", "contrast_hist", "patch_swd", "tlp", "tof"):
        assert r[k] == pytest.approx(0.0, abs=1e-9), (k, r[k])
    assert r["pop"] == pytest.approx(1.0) and r["tlp_log_ratio"] == pytest.approx(0.0, abs=1e-9)


def test_band_spectrum_white_noise_is_flat_and_shift_invariant():
    g = torch.Generator(device="cpu").manual_seed(0)
    a = (0.5 + 0.1 * torch.randn(4, 3, 128, 128, generator=g)).to(DEV)
    b = (0.5 + 0.1 * torch.randn(4, 3, 128, 128, generator=g)).to(DEV)     # independent realization
    r = M.band_spectrum(a, b)
    assert r["distance"] < 0.05, r                       # same statistics, different pixels -> ~0
    rolled = torch.roll(a, (17, -9), dims=(-2, -1))
    assert M.band_spectrum(rolled, a)["distance"] < 0.05
    assert M.l1(rolled, a) > 0.05                        # ...while a pixel metric sees a large error
    blur = M.gaussian_blur(a, 1.5)
    rb = M.band_spectrum(blur, a)["bands"]
    assert rb["2-4px"] < -1.0 and rb["2-4px"] < rb["8-16px"] < 0, rb   # blur removes the finest band most


def test_patch_swd_deterministic_and_size_independent(seqs):
    ref, mask, out = seqs
    a = M.patch_swd(out["blur"], ref, mask)
    assert a == M.patch_swd(out["blur"], ref, mask)
    assert M.patch_swd(ref, ref, mask) == 0.0


def test_lap_distance_scales():
    x = torch.zeros(1, 3, 64, 64, device=DEV)
    y = x.clone()
    y[..., ::2, :] = 0.2                                   # finest-scale stripes
    assert M.lap_distance(x, x).item() == 0.0
    assert M.lap_distance(x, y).item() > 0
    assert M.lap_distance(x, torch.full_like(x, 0.3)).item() < 1e-6     # pure brightness change is not detail


# ----------------------------------------------------------------------------- controls (spatial)
def test_unaligned_detail_beats_blur_on_distribution_metrics(scores):
    """The reason for these metrics: an independent realization loses to blur on l1 and MS-DSSIM, but must win on
    every distribution metric."""
    s = scores
    assert s["indep"]["l1"] > s["blur"]["l1"], "premise: pixel metrics prefer blur over unaligned detail"
    assert s["indep"]["ms_dssim"] > s["blur"]["ms_dssim"], "premise: SSIM prefers blur over unaligned detail"
    for k in ("band_distance", "grad_hist", "contrast_hist", "patch_swd"):
        assert s["blur"][k] > MARGIN * s["indep"][k], (k, s["blur"][k], s["indep"][k])
    assert s["blur"]["band_2-4px"] < -0.3, "blur must show as missing power in the finest band"


@pytest.mark.parametrize("ctrl", ["noise"])
def test_noise_detected_spatially(scores, ctrl):
    s = scores
    for k in ("band_distance", "grad_hist", "patch_swd"):
        assert s[ctrl][k] > MARGIN * s["indep"][k], (k, s[ctrl][k], s["indep"][k])
    assert s[ctrl]["band_2-4px"] > 0.3, "noise must show as excess power in the finest band"


# ----------------------------------------------------------------------------- controls (temporal)
def test_blur_scores_worse_temporally_too_little_change(scores):
    s = scores
    assert s["blur"]["tlp"] > MARGIN * s["indep"]["tlp"] and s["blur"]["tlp"] > 0.1
    assert s["blur"]["tlp_log_ratio"] < -0.1, "blur must read as too little change"
    # documents the legacy trap (not a requirement on the new metrics): warp-error flicker PREFERS the blur
    print(f"legacy flicker ratio of blur {s['blur']['flicker_ratio_legacy']:.3f} (< 1 = rewarded)")


def test_noise_scores_worse_temporally_too_much_change(scores):
    s = scores
    assert s["noise"]["tlp"] > MARGIN * s["indep"]["tlp"] and s["noise"]["tlp"] > 0.1
    assert s["noise"]["tlp_log_ratio"] > 0.1, "per-frame noise must read as too much change"


def test_stuck_to_screen_detail_scores_worse(scores):
    s = scores
    assert s["stuck"]["tof"] > MARGIN * s["indep"]["tof"] and s["stuck"]["tof"] > 0.2, \
        (s["stuck"]["tof"], s["indep"]["tof"])
    assert s["stuck"]["tlp"] > MARGIN * s["indep"]["tlp"], (s["stuck"]["tlp"], s["indep"]["tlp"])


def test_periodic_pop_scores_worse(scores):
    s = scores
    assert s["pop"]["pop"] > MARGIN * max(s["indep"]["pop"], 1.0), (s["pop"]["pop"], s["indep"]["pop"])
    assert s["pop"]["tlp"] > MARGIN * s["indep"]["tlp"], (s["pop"]["tlp"], s["indep"]["tlp"])


def test_reference_scores_best_on_every_new_metric(scores):
    keys = ("band_distance", "grad_hist", "contrast_hist", "patch_swd", "tlp", "tof")
    for c in ("indep", "blur", "noise", "stuck", "pop"):
        for k in keys:
            assert scores[c][k] > scores["ref"][k], (c, k)
