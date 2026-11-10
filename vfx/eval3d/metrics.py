"""Image-space metrics for route B (screen-space detail) — owner: 3D test/eval engineer.

Training and the Layer 4 gates call these. Importable without the dataset or any other 3D module.
Self-contained on purpose (own warp, own blur, own sharpen) so the metrics never depend on the code they judge.

Conventions
-----------
* Frames: (..., 3, h, w), uint8 (0..255) or float (0..1). Everything is converted to float32 in [0, 1].
  Row 0 = top of the screen (renderer convention).
* aux (CLAUDE.md "Route B training phase"): (..., 4, h, w) = alpha, depth, motion_x, motion_y of the LR render.
  motion is in PIXELS PER SIM STEP, x right, y down, and means: the content at pixel p of frame t was at
  p - motion_t(p) in frame t-1. So frame t-1 is brought to frame t by the BACKWARD warp
        W(x_{t-1}, m_t)(p) = x_{t-1}(p - m_t(p))      (bilinear, border-clamped; `warp_backward`)
  using the motion stored WITH frame t.
* Effect region (`effect_mask`): alpha > ALPHA_THRESH (0.02), dilated by `dilate_px` (default
  max(2, round(max(h, w) / 128)) px, i.e. ~11 px at 1440p, ~4 px at 480 wide). Every metric is a mean over
  this region only, so the empty background (identical in every method) does not dilute it. The mask is
  ALWAYS computed from the LR render's alpha and shared by every sequence being compared (prediction, lr,
  sharpen, hr): all methods are scored on the same pixels.

Metrics (all "lower is better", 0 = identical)
----------------------------------------------
l1             masked mean over pixels and the 3 channels of |pred - ref| (0..1 units).
spectrum_error THE detail metric. Luminance Y = 0.2126 R + 0.7152 G + 0.0722 B (of the 0..1 values). Crop to
               the bounding box of the effect mask (union over the frames given), subtract each frame's mean,
               multiply by a 2D Hann window, P = |FFT2|^2 / sum(window^2) (white noise of variance s^2 -> s^2
               per mode). Radial frequency f = sqrt((ky/H)^2 + (kx/W)^2) in cycles/pixel (works for any crop
               shape), averaged inside N_BINS = 32 linear bins with edges linspace(1/64, 0.5, 33) and over
               frames (spectral average, then log). Bins without modes (small crops) are ignored.
                   error = mean_bins | log10(P_pred + FLOOR) - log10(P_ref + FLOOR) |     (decades)
               FLOOR = (1/255)^2 / 12, the power of uint8 quantization noise: differences below what an 8-bit
               frame can represent do not count. `spectrum_log_ratio` = the signed mean (< 0 blurrier than
               the reference, > 0 more high-frequency energy / noise).
ms_dssim       structural: 1 - sum_s w_s SSIM_s, SSIM of luminance (Gaussian window sigma 1.5, C1 = 0.01^2,
               C2 = 0.03^2) at 3 scales (full, /2, /4 by 2x2 average pooling), w = (0.5, 0.3, 0.2), each SSIM_s a
               masked mean (mask average-pooled with the image). No downloaded weights.
flicker        warped-frame flicker of a sequence (L, 3, h, w) with ITS OWN motion (L, 2, h, w):
                   mean_{t=1..L-1} masked_mean_p | x_t(p) - W(x_{t-1}, m_t)(p) |
               with weight mask_t AND (source pixel p - m_t(p) inside the frame), m = the sequence's own estimated
               apparent motion (see below). Judged as a RATIO to the hr reference: flicker(pred) / flicker(hr).
temporal_jerk  second temporal difference, unwarped (the sensitive flicker check; see its docstring), ratio to hr.
detail_swim    view consistency of the ADDED detail: the flicker of the signed detail layer
               D = output - input (pred - lr), relative to the same for the reference layer hr - lr:
                   swim_ratio = flicker(pred - lr, flow(pred)) / flicker(hr - lr, flow(hr))
               Detail that is glued to the content moves with the motion field and scores like the reference;
               detail that stays in screen space or re-rolls each frame "swims" and scores high.

Which motion is legitimate (revised 2026-11-09 after measuring on s3d_v1; the 2D lesson stands: warping a
sequence with ANOTHER sequence's motion measures how the two differ, not flicker):
* The stored aux `motion` is the opacity-weighted FLUID velocity, not the apparent image motion. Near the source a
  plume's image is quasi-steady while gas flows through it; higher up it moves at ~0.25-0.5x the stored motion;
  only the camera part is exact, and background / floor carry 0 even when the camera moves. Measured on s3d_v1
  (88 sequences, warp error relative to no warp, region-weighted): stored motion lr 2.13 / hr 1.57 (static
  camera), 1.35 / 1.18 (moving); x0.5: 1.38 / 1.15; x0.25: 1.12 / 1.02. Warping with it penalizes correct,
  temporally coherent output and makes the metric blind (2-code noise moved the ratio only to 1.05).
* So the temporal metrics warp every sequence with ITS OWN apparent motion, estimated from that sequence:
  `sequence_flow` (pyramidal Lucas-Kanade on luminance, 4 levels, Gaussian windows sigma 4 px, smoothed 2 px: a
  coarse, smooth flow that follows plume and camera motion but cannot absorb per-frame noise). Measured: own flow
  leaves lr 0.40 / hr 0.53 (static), lr 0.32 / hr 0.43 (moving) of the unwarped error; a flow taken from the OTHER
  sequence (lr flow on hr, hr flow on lr) only 0.69-0.87. A shared flow from lr would therefore favour lr-like
  (blurred) output; an own flow treats prediction and reference alike: flicker(pred, flow(pred)) /
  flicker(hr, flow(hr)). Detail swim warps the detail layer with the flow of the image it belongs to
  (out - lr with flow(out); hr - lr with flow(hr)): detail glued to the content scores like the reference,
  detail fixed in screen space or re-rolled each frame scores high.
* temporal_jerk needs no alignment at all and is the most sensitive shimmer check.
* Flicker, jerk and swim are GATES (ratio to hr <= 1.2), not objectives: less detail always changes less frame to
  frame (the lr baseline scores ~0.5-0.8), so never minimize them in training; minimizing them rewards blur.
* Temporal metrics are weighted by the same dilated effect region as the spatial ones (the own flow also
  compensates camera motion of the background, which the stored motion did not).
* The live check (no dataset) uses the same definition on the 2x2-pooled frames (the stored / processed scale).
* Spatial metrics (l1, spectrum, ms_dssim) use the dilated union of the lr alpha and, when the dataset provides
  it, the hr alpha ("hr_alpha"): smoke that exists only in hr is part of what the net must produce.

Baselines every score is compared to: the unmodified lr frame (`identity`) and a classical unsharp mask
(`unsharp`, sigma and amount tuned on train by `tune_unsharp`). A learned net must beat both.

Evaluation-time code: .item() / host syncs are fine here (not the frame loop).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "LUMA", "ALPHA_THRESH", "FLOOR", "N_BINS", "BIN_EDGES",
    "to_float", "luminance", "effect_mask", "default_dilate_px",
    "l1", "radial_power_spectrum", "spectrum_error", "spectrum_log_ratio", "band_power",
    "ssim_map", "ms_dssim",
    "warp_backward", "estimate_flow", "sequence_flow", "warp_error", "flicker", "temporal_jerk", "detail_swim", "item_masks",
    "gaussian_blur", "unsharp", "identity", "tune_unsharp",
    "aux_dict", "run_sequence", "score_sequence", "evaluate_screen_detail",
]

LUMA = (0.2126, 0.7152, 0.0722)
ALPHA_THRESH = 0.02
FLOOR = (1.0 / 255.0) ** 2 / 12.0
N_BINS = 32
BIN_EDGES = np.linspace(1.0 / 64.0, 0.5, N_BINS + 1)
MS_WEIGHTS = (0.5, 0.3, 0.2)


# ============================================================================ basics
def to_float(x) -> torch.Tensor:
    """uint8 -> float32 / 255; float -> float32 (assumed 0..1). numpy accepted."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(x))
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor/ndarray, got {type(x)}")
    if x.dtype == torch.uint8:
        return x.float() * (1.0 / 255.0)
    return x.float()


def luminance(x) -> torch.Tensor:
    """(..., 3, h, w) -> (..., h, w), Rec.709 weights on the stored (sRGB-encoded) values."""
    x = to_float(x)
    if x.shape[-3] != 3:
        raise ValueError(f"expected (...,3,h,w), got {tuple(x.shape)}")
    w = torch.tensor(LUMA, device=x.device, dtype=x.dtype).view(3, 1, 1)
    return (x * w).sum(-3)


def default_dilate_px(h: int, w: int) -> int:
    return max(2, int(round(max(h, w) / 128)))


def effect_mask(alpha, thresh: float = ALPHA_THRESH, dilate_px: int | None = None) -> torch.Tensor:
    """(..., h, w) alpha -> bool mask (..., h, w): alpha > thresh, dilated by a (2r+1)^2 square."""
    a = to_float(alpha)
    h, w = a.shape[-2:]
    r = default_dilate_px(h, w) if dilate_px is None else int(dilate_px)
    m = (a > thresh).float().reshape(-1, 1, h, w)
    if r > 0:
        m = F.max_pool2d(m, 2 * r + 1, 1, r)
    return (m > 0.5).reshape(a.shape)


def _weights(mask, like: torch.Tensor) -> torch.Tensor:
    """mask (..., h, w) bool/float or None -> float weights broadcastable to like (..., C, h, w)."""
    if mask is None:
        return torch.ones(like.shape[:-3] + (1,) + like.shape[-2:], device=like.device)
    m = torch.as_tensor(mask).to(like.device)
    m = (m > 0).float() if m.dtype in (torch.bool, torch.uint8) else m.float()
    return m.unsqueeze(-3)


def _masked_mean(err: torch.Tensor, mask) -> float:
    """err (..., C, h, w) -> mean over masked pixels and channels."""
    w = _weights(mask, err)
    den = (w.expand_as(err)).sum()
    if den.item() == 0:
        return float("nan")
    return float(((err * w).sum() / den).item())


def l1(pred, ref, mask=None) -> float:
    """Masked mean |pred - ref| over pixels and channels, frames (..., C, h, w) in 0..1 units."""
    p, r = to_float(pred), to_float(ref)
    r = r.to(p.device)
    if p.shape != r.shape:
        raise ValueError(f"shape mismatch {tuple(p.shape)} vs {tuple(r.shape)}")
    return _masked_mean((p - r).abs(), mask)


# ============================================================================ spectrum
def _bbox(mask: torch.Tensor | None, h: int, w: int, min_size: int = 16):
    if mask is None:
        return 0, h, 0, w
    m = mask.reshape(-1, h, w).any(0)
    if not m.any():
        return 0, h, 0, w
    ys = torch.nonzero(m.any(1)).flatten()
    xs = torch.nonzero(m.any(0)).flatten()
    y0, y1, x0, x1 = int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1

    def grow(a, b, L):
        if b - a >= min_size:
            return a, b
        c = (a + b) // 2
        a = max(0, min(c - min_size // 2, L - min_size))
        return a, min(L, a + min_size)
    y0, y1 = grow(y0, y1, h)
    x0, x1 = grow(x0, x1, w)
    return y0, y1, x0, x1


def radial_power_spectrum(frames, mask=None, edges=BIN_EDGES):
    """Binned radial power spectrum of luminance (see module docstring).

    frames (..., 3, h, w) or luminance (..., h, w) if it has no channel dim of size 3 at -3 — pass 3-channel
    frames to be safe. mask (..., h, w) selects the crop (bbox of the union). Returns (power (n_bins,) float64
    with NaN for empty bins, counts (n_bins,) number of FFT modes per bin)."""
    x = to_float(frames)
    lum = luminance(x) if x.dim() >= 3 and x.shape[-3] == 3 else x
    h, w = lum.shape[-2:]
    lum = lum.reshape(-1, h, w)
    y0, y1, x0, x1 = _bbox(None if mask is None else torch.as_tensor(mask).to(lum.device), h, w)
    c = lum[:, y0:y1, x0:x1].double()
    H, W = c.shape[-2:]
    c = c - c.mean(dim=(-2, -1), keepdim=True)
    win = torch.outer(torch.hann_window(H, periodic=False, dtype=torch.float64, device=c.device),
                      torch.hann_window(W, periodic=False, dtype=torch.float64, device=c.device))
    if H < 3 or W < 3:
        win = torch.ones_like(win)
    P = torch.fft.fft2(c * win).abs().pow(2).mean(0) / win.pow(2).sum()
    fy = torch.fft.fftfreq(H, device=c.device, dtype=torch.float64)
    fx = torch.fft.fftfreq(W, device=c.device, dtype=torch.float64)
    f = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    e = torch.as_tensor(edges, device=c.device, dtype=torch.float64)
    idx = torch.bucketize(f.flatten(), e, right=True) - 1          # bin i: e[i] <= f < e[i+1]
    nb = len(edges) - 1
    keep = (idx >= 0) & (idx < nb)
    s = torch.zeros(nb, dtype=torch.float64, device=c.device).scatter_add_(0, idx[keep], P.flatten()[keep])
    n = torch.zeros(nb, dtype=torch.float64, device=c.device).scatter_add_(0, idx[keep], torch.ones_like(P.flatten()[keep]))
    power = torch.where(n > 0, s / n.clamp(min=1), torch.full_like(s, float("nan")))
    return power.cpu().numpy(), n.cpu().numpy()


def _log_diff(pred, ref, mask):
    pp, _ = radial_power_spectrum(pred, mask)
    pr, _ = radial_power_spectrum(ref, mask)
    d = np.log10(pp + FLOOR) - np.log10(pr + FLOOR)
    return d[np.isfinite(d)]


def spectrum_error(pred, ref, mask=None) -> float:
    """Mean over frequency bins of |log10 P_pred - log10 P_ref| of luminance in the effect bbox (decades)."""
    d = _log_diff(pred, ref, mask)
    return float(np.abs(d).mean()) if d.size else float("nan")


def spectrum_log_ratio(pred, ref, mask=None) -> float:
    """Signed version of spectrum_error: < 0 = less high-frequency power than ref (blur), > 0 = more."""
    d = _log_diff(pred, ref, mask)
    return float(d.mean()) if d.size else float("nan")


def band_power(frames, mask=None, f_lo: float = 0.0, f_hi: float = 0.5) -> float:
    """Mean binned power over the bins whose centre lies in [f_lo, f_hi) cycles/pixel."""
    p, _ = radial_power_spectrum(frames, mask)
    c = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
    sel = (c >= f_lo) & (c < f_hi) & np.isfinite(p)
    return float(p[sel].mean()) if sel.any() else float("nan")


# ============================================================================ structural
def _gauss_kernel(sigma: float, device) -> torch.Tensor:
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


def _sep_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """x (N, C, h, w), separable Gaussian, reflect padding."""
    k = _gauss_kernel(sigma, x.device)
    r = k.numel() // 2
    C = x.shape[1]
    kx = k.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    mode = "reflect" if min(x.shape[-2:]) > r else "replicate"
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode=mode), kx, groups=C)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode=mode), ky, groups=C)


def ssim_map(a: torch.Tensor, b: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """a, b (N, 1, h, w) luminance 0..1 -> SSIM map (N, 1, h, w)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = _sep_blur(a, sigma), _sep_blur(b, sigma)
    saa = _sep_blur(a * a, sigma) - mu_a ** 2
    sbb = _sep_blur(b * b, sigma) - mu_b ** 2
    sab = _sep_blur(a * b, sigma) - mu_a * mu_b
    return ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))


def ms_dssim(pred, ref, mask=None, weights=MS_WEIGHTS) -> float:
    """1 - sum_s w_s * masked_mean(SSIM at scale s) on luminance (scales 1, 1/2, 1/4)."""
    a = luminance(pred)
    b = luminance(ref).to(a.device)
    h, w = a.shape[-2:]
    a, b = a.reshape(-1, 1, h, w), b.reshape(-1, 1, h, w)
    m = torch.ones_like(a) if mask is None else torch.as_tensor(mask).to(a.device).float().reshape(-1, 1, h, w)
    total = 0.0
    for i, ws in enumerate(weights):
        if i:
            if min(a.shape[-2:]) < 8:
                break
            a, b, m = F.avg_pool2d(a, 2), F.avg_pool2d(b, 2), F.avg_pool2d(m, 2)
        s = ssim_map(a, b)
        den = m.sum()
        val = (s * m).sum() / den if den > 0 else s.mean()
        total += ws * float(val.item())
    return 1.0 - total


# ============================================================================ temporal
def warp_backward(x: torch.Tensor, motion: torch.Tensor):
    """out(p) = x(p - motion(p)), bilinear, border clamp.

    x (N, C, h, w) float; motion (N, 2, h, w) px (x right, y down). Returns (out, valid) with valid (N, h, w)
    True where the source point lies inside the frame (pixels whose content came from off-screen are not
    judged)."""
    N, C, h, w = x.shape
    m = motion.float().to(x.device)
    ys = torch.arange(h, device=x.device, dtype=torch.float32)
    xs = torch.arange(w, device=x.device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    sx = gx[None] - m[:, 0]
    sy = gy[None] - m[:, 1]
    grid = torch.stack([2.0 * sx / max(w - 1, 1) - 1.0, 2.0 * sy / max(h - 1, 1) - 1.0], dim=-1)
    out = F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="border", align_corners=True)
    valid = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    return out, valid


def estimate_flow(prev, cur, levels: int = 4, iters: int = 4, sigma: float = 4.0, smooth: float = 2.0,
                  reg: float = 1e-5) -> torch.Tensor:
    """Image-space (apparent) motion between two frames by pyramidal Lucas-Kanade on luminance.

    prev, cur: (N, 3, h, w) or (N, h, w) luminance, uint8 or float. Returns motion (N, 2, h, w) in px with the
    contract convention cur(p) ~= prev(p - motion(p)) (so `warp_backward(prev, motion)` ~= cur). Gaussian
    windows of `sigma` px at every pyramid level (coarse levels see large motion), Tikhonov `reg`, update clamped to
    2 px per iteration, result Gaussian-smoothed (`smooth` px): a deliberately SMOOTH, coarse flow, so it follows
    the apparent motion of the plume / camera and not per-frame noise or fine detail."""
    a = prev if prev.dim() == 3 else luminance(prev)
    b = cur if cur.dim() == 3 else luminance(cur)
    a, b = to_float(a), to_float(b).to(a.device)
    pa, pb = [a[:, None]], [b[:, None]]
    for _ in range(levels - 1):
        if min(pa[-1].shape[-2:]) < 16:
            break
        pa.append(F.avg_pool2d(pa[-1], 2))
        pb.append(F.avg_pool2d(pb[-1], 2))
    m = torch.zeros(a.shape[0], 2, *pa[-1].shape[-2:], device=a.device)
    for lv in range(len(pa) - 1, -1, -1):
        A, B = pa[lv], pb[lv]
        if m.shape[-2:] != A.shape[-2:]:
            m = F.interpolate(m, size=A.shape[-2:], mode="bilinear", align_corners=False) * 2.0
        for _ in range(iters):
            wp, _ = warp_backward(A, m)
            gx = F.pad(wp[..., :, 2:] - wp[..., :, :-2], (1, 1, 0, 0), mode="replicate") * 0.5
            gy = F.pad(wp[..., 2:, :] - wp[..., :-2, :], (0, 0, 1, 1), mode="replicate") * 0.5
            e = wp - B
            S = lambda t: _sep_blur(t, sigma)
            sxx, sxy, syy = S(gx * gx) + reg, S(gx * gy), S(gy * gy) + reg
            bx, by = S(gx * e), S(gy * e)
            det = (sxx * syy - sxy * sxy).clamp(min=1e-12)
            dx = (syy * bx - sxy * by) / det
            dy = (sxx * by - sxy * bx) / det
            # warped(p) = prev(p - m); prev(p - m - d) ~= warped - g.d -> g.d = warped - cur
            m = m + torch.cat([dx, dy], 1).clamp(-2.0, 2.0)
    if smooth > 0:
        m = _sep_blur(m, smooth)
    return m


def sequence_flow(seq) -> torch.Tensor:
    """(L, 3, h, w) -> (L, 2, h, w) estimated apparent motion, flow[t] maps frame t-1 to t (flow[0] = 0).
    `evaluate_screen_detail` estimates it from the lr sequence ONLY and uses it for every method (module docstring)."""
    x = to_float(seq)
    fl = torch.zeros(x.shape[0], 2, *x.shape[-2:], device=x.device)
    if x.shape[0] > 1:
        fl[1:] = estimate_flow(x[:-1], x[1:])
    return fl


def warp_error(seq, motion, mask=None, use_motion: bool = True, motion_offset: int = 0) -> float:
    """Mean over t >= 1 of masked mean |x_t - W(x_{t-1}, m)|, m = motion[t - motion_offset] (offset 0 is the
    contract; 1 = the motion stored with frame t-1, a diagnostic for an off-by-one in the data).
    use_motion=False compares with the unwarped previous frame (m = 0). seq may be signed float."""
    x = seq.float() if isinstance(seq, torch.Tensor) and seq.is_floating_point() else to_float(seq)
    L = x.shape[0]
    if L < 2:
        raise ValueError("need at least 2 frames")
    mo = torch.as_tensor(motion).to(x.device).float()
    if mo.shape[0] != L or mo.shape[1] != 2:
        raise ValueError(f"motion must be (L,2,h,w) with L={L}, got {tuple(mo.shape)}")
    mk = None if mask is None else torch.as_tensor(mask).to(x.device)
    t0 = 1 + motion_offset
    vals = []
    for t in range(t0, L):
        m = mo[t - motion_offset:t - motion_offset + 1] if use_motion else torch.zeros_like(mo[t:t + 1])
        wp, valid = warp_backward(x[t - 1:t], m)
        wgt = valid.float()
        if mk is not None:
            wgt = wgt * mk[t:t + 1].float()
        e = (x[t:t + 1] - wp).abs()
        den = wgt.sum() * x.shape[1]
        if den.item() > 0:
            vals.append(float(((e * wgt.unsqueeze(1)).sum() / den).item()))
    return float(np.mean(vals)) if vals else float("nan")


def flicker(seq, motion=None, mask=None) -> float:
    """Warped-frame flicker of a sequence (L, 3, h, w) with its OWN motion (L, 2, h, w) (module docstring).
    motion=None (what evaluate_screen_detail uses): the sequence's own apparent motion, `sequence_flow(seq)`."""
    if motion is None:
        motion = sequence_flow(seq)
    return warp_error(seq, motion, mask, use_motion=True)


def temporal_jerk(seq, mask=None) -> float:
    """Second temporal difference, unwarped: mean_{t=1..L-2} masked_mean |x_{t+1} - 2 x_t + x_{t-1}|.

    Smooth change (fades, constant-velocity camera or content motion, steady Eulerian plumes) cancels to first
    order; per-frame flicker does not: iid noise of std s gives 2 s sqrt(3/pi) (= 1.95 s). Added 2026-11-09 because
    the motion-warped `flicker` is nearly blind on s3d renders (the material-velocity motion field does not
    describe the apparent motion of a steady plume; see tests3d/README.md). Judged as a ratio to hr. Also
    computed on the effect region."""
    x = seq.float() if isinstance(seq, torch.Tensor) and seq.is_floating_point() else to_float(seq)
    L = x.shape[0]
    if L < 3:
        raise ValueError("need at least 3 frames")
    mk = None if mask is None else torch.as_tensor(mask).to(x.device)
    vals = []
    for t in range(1, L - 1):
        e = (x[t + 1] - 2 * x[t] + x[t - 1]).abs()
        if mk is None:
            vals.append(float(e.mean()))
        else:
            w = mk[t].float()
            den = w.sum() * x.shape[1]
            if den.item() > 0:
                vals.append(float((e * w).sum() / den))
    return float(np.mean(vals)) if vals else float("nan")


def detail_swim(out_seq, in_seq, motion=None, mask=None) -> float:
    """Flicker of the signed detail layer out - in (module docstring). 0 if out == in.
    motion=None: the apparent motion of the OUTPUT sequence (`sequence_flow(out_seq)`), i.e. the detail is
    expected to move with the image it is part of; for the reference call it with hr as out_seq."""
    if motion is None:
        motion = sequence_flow(out_seq)
    d = to_float(out_seq) - to_float(in_seq).to(to_float(out_seq).device)
    return warp_error(d, motion, mask, use_motion=True)


# ============================================================================ baselines
def gaussian_blur(x, sigma: float) -> torch.Tensor:
    """(..., C, h, w) float -> blurred, same shape."""
    x = to_float(x)
    sh = x.shape
    y = _sep_blur(x.reshape(-1, sh[-3], sh[-2], sh[-1]), sigma)
    return y.reshape(sh)


def identity(frame: torch.Tensor, aux=None) -> torch.Tensor:
    """Baseline 1: the unmodified lr frame (h, w, 3) uint8."""
    return frame


def unsharp(x, sigma: float, amount: float) -> torch.Tensor:
    """Classical unsharp mask: x + amount * (x - G_sigma * x), clamped. uint8 in -> uint8 out (rounded),
    float in -> float out. Accepts (..., 3, h, w), or a single (h, w, 3) uint8 runtime frame."""
    hwc = isinstance(x, torch.Tensor) and x.dim() == 3 and x.shape[-1] == 3 and x.shape[0] != 3
    t = x.permute(2, 0, 1) if hwc else x
    is_u8 = t.dtype == torch.uint8
    f = to_float(t)
    y = (f + amount * (f - gaussian_blur(f, sigma))).clamp(0.0, 1.0)
    if is_u8:
        y = (y * 255.0 + 0.5).to(torch.uint8)
    return y.permute(1, 2, 0).contiguous() if hwc else y


def make_unsharp(sigma: float, amount: float) -> Callable:
    def fn(frame, aux=None):
        return unsharp(frame, sigma, amount)
    fn.__name__ = f"unsharp(sigma={sigma:g}, amount={amount:g})"
    return fn


UNSHARP_SIGMAS = (0.75, 1.0, 1.5, 2.0, 3.0)
UNSHARP_AMOUNTS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def _cache_dir() -> Path:
    from vfx.core.paths import RUNS

    d = RUNS / "eval3d"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tune_unsharp(version: str, split: str = "train", length: int = 8, max_sequences: int = 24,
                 device: str = "cuda", use_cache: bool = True, sigmas=UNSHARP_SIGMAS, amounts=UNSHARP_AMOUNTS) -> dict:
    """Grid-search the unsharp mask on `split` (train) of the dataset.

    Objective (lower better) = mean over {l1, spectrum_error, ms_dssim} of metric(sharpened, hr) / metric(lr, hr),
    each metric pooled over the sequences (masked to the effect region). amount = 0 (identity) scores 1.0, so the
    tuned sharpen is never worse than lr on this combined score on train. Cached in RUNS/eval3d/unsharp_<version>.json.
    Returns {"sigma", "amount", "objective", "grid": [...]}."""
    cache = _cache_dir() / f"unsharp_{version}_{split}.json"
    if use_cache and cache.exists():
        return json.loads(cache.read_text())
    ds = _load(version, split, length)
    idx = _subsample(len(ds), max_sequences)
    items = []
    for i in idx:
        it = ds[i]
        lr = torch.as_tensor(it["lr"]).to(device)
        hr = torch.as_tensor(it["hr"]).to(device)
        aux = torch.as_tensor(it["aux"]).to(device)
        items.append((lr, hr, effect_mask(aux[:, 0])))
    base = {k: [] for k in ("l1", "spec", "ssim")}
    for lr, hr, m in items:
        base["l1"].append(l1(lr, hr, m))
        base["spec"].append(spectrum_error(lr, hr, m))
        base["ssim"].append(ms_dssim(lr, hr, m))
    b = {k: max(float(np.nanmean(v)), 1e-12) for k, v in base.items()}
    grid = [{"sigma": 0.0, "amount": 0.0, "objective": 1.0, "l1": 1.0, "spec": 1.0, "ssim": 1.0}]   # = lr itself
    for s in sigmas:
        for a in amounts:
            acc = {k: [] for k in b}
            for lr, hr, m in items:
                sh = unsharp(lr, s, a)
                acc["l1"].append(l1(sh, hr, m))
                acc["spec"].append(spectrum_error(sh, hr, m))
                acc["ssim"].append(ms_dssim(sh, hr, m))
            rel = {k: float(np.nanmean(v)) / b[k] for k, v in acc.items()}
            grid.append({"sigma": s, "amount": a, "objective": float(np.mean(list(rel.values()))), **rel})
    best = min(grid, key=lambda g: g["objective"])
    out = {"sigma": best["sigma"], "amount": best["amount"], "objective": best["objective"], "version": version,
           "split": split, "n_sequences": len(items), "length": length, "grid": grid,
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    cache.write_text(json.dumps(out, indent=1))
    return out


# ============================================================================ running a model like the runtime
def aux_dict(aux_t: torch.Tensor) -> dict:
    """(4, h, w) stored aux -> the CLAUDE.md runtime dict {"alpha", "depth", "motion" (2, h, w)} (float16 cuda)."""
    a = aux_t.half()
    return {"alpha": a[0], "depth": a[1], "motion": a[2:4]}


def load_screen_detail_for_dataset(path, device: str = "cuda"):
    """ScreenDetail from a checkpoint, configured for DATASET frames: the stored crops are already at the
    runtime's processed resolution (2x2-pooled 1440p, = ScreenDetail's default process_scale 0.5), so the
    runtime must not downscale them again: process_scale=1.0."""
    from vfx.runtime3d import ScreenDetail

    try:
        return ScreenDetail(str(path), device=device, process_scale=1.0)
    except TypeError:
        return ScreenDetail(str(path), device=device)


def _infer_fn(model_or_callable):
    """-> (infer(frame_hwc_u8, aux_dict) -> frame_hwc_u8, reset() or None, name)."""
    m = model_or_callable
    if isinstance(m, (str, Path)):
        m = load_screen_detail_for_dataset(m)
    if hasattr(m, "infer"):
        return m.infer, getattr(m, "reset", None), type(m).__name__
    if callable(m):
        return m, getattr(m, "reset", None), getattr(m, "__name__", type(m).__name__)
    raise TypeError(f"need ScreenDetail-like .infer, a callable (frame, aux) -> frame, or a checkpoint path; got {type(m)}")


@torch.no_grad()
def run_sequence(model_or_callable, lr: torch.Tensor, aux: torch.Tensor, extra_aux: dict | None = None) -> torch.Tensor:
    """Run autoregressively, exactly as the runtime does: reset(), then for t = 0..L-1 one infer() call on the
    (h, w, 3) uint8 cuda frame with that frame's aux dict. lr (L, 3, h, w) uint8, aux (L, 4, h, w).
    Returns (L, 3, h, w) uint8 (outputs copied: the runtime reuses its buffers)."""
    infer, reset, _ = _infer_fn(model_or_callable)
    if reset is not None:
        reset()
    outs = []
    for t in range(lr.shape[0]):
        frame = lr[t].permute(1, 2, 0).contiguous()
        ad = aux_dict(aux[t])
        if extra_aux:
            ad.update(extra_aux)
        o = infer(frame, ad)
        if o.dtype != torch.uint8:
            raise TypeError(f"infer must return uint8 (h,w,3), got {o.dtype}")
        outs.append(o.permute(2, 0, 1).clone())
    return torch.stack(outs)


def score_sequence(out: torch.Tensor, lr: torch.Tensor, hr: torch.Tensor, aux: torch.Tensor, mask=None,
                   tweight=None) -> dict:
    """All metrics of one output sequence (L,3,h,w) vs hr. mask: spatial region (default: dilated lr alpha);
    tweight: per-pixel weight of the temporal metrics (default: the same region). Temporal metrics warp `out`
    with its OWN estimated apparent motion (sequence_flow(out)), never the stored fluid motion (module docstring)."""
    if mask is None:
        mask = effect_mask(aux[:, 0])
    if tweight is None:
        tweight = mask
    motion = sequence_flow(out) if out.shape[0] > 1 else None
    r = {"l1": l1(out, hr, mask), "spectrum_error": spectrum_error(out, hr, mask),
         "spectrum_log_ratio": spectrum_log_ratio(out, hr, mask), "ms_dssim": ms_dssim(out, hr, mask)}
    if out.shape[0] > 1:
        r["flicker"] = flicker(out, motion, tweight)
        r["swim"] = detail_swim(out, lr, motion, tweight)
    if out.shape[0] > 2:
        r["jerk"] = temporal_jerk(out, tweight)
    return r


def _load(version: str, split: str, length: int, stride: int | None = None):
    """load_sequences(version, split, length) of vfx.datagen3d; non-overlapping windows (stride = length) when
    the loader accepts a stride (it is not part of the CLAUDE.md signature)."""
    from vfx.datagen3d import load_sequences

    if stride is not None:
        try:
            return load_sequences(version, split, length, stride=stride)
        except TypeError:
            pass
    return load_sequences(version, split, length)


def _subsample(n: int, k: int | None) -> list[int]:
    if k is None or k >= n:
        return list(range(n))
    return sorted({int(round(i * (n - 1) / max(k - 1, 1))) for i in range(k)})


def _scalar(v):
    if isinstance(v, torch.Tensor):
        return v.item() if v.numel() == 1 else v.tolist()
    if isinstance(v, np.ndarray):
        return v.item() if v.size == 1 else v.tolist()
    return v


def item_masks(it: dict, aux: torch.Tensor, device="cuda"):
    """(spatial mask, temporal weight) of a dataset item (module docstring)."""
    a = aux[:, 0].float().clamp(0, 1)
    if it.get("hr_alpha") is not None:
        ha = to_float(torch.as_tensor(it["hr_alpha"]).to(device))
        a_sp = torch.maximum(a, ha)
    else:
        a_sp = a
    return effect_mask(a_sp), a


def _pooled(per, name, key, ref_key, sel=lambda p: True):
    num = [p[name][key] for p in per if sel(p) and key in p[name] and np.isfinite(p[name][key])]
    den = [p[ref_key] for p in per if sel(p) and key in p[name] and np.isfinite(p[name][key])]
    if not num or sum(den) <= 0:
        return float("nan")
    return float(sum(num) / sum(den))


def evaluate_screen_detail(model_or_callable, version: str, split: str = "test", length: int = 16,
                           max_sequences: int | None = None, sharpen: dict | None = None, baselines: bool = True,
                           device: str = "cuda", extra_aux: dict | None = None) -> dict:
    """Score route B on a dataset split against the hr renders, with the lr and unsharp baselines.

    model_or_callable: a ScreenDetail-like object (.infer(frame, aux), optional .reset()), a plain callable
        (frame (h,w,3) uint8 cuda, aux dict) -> (h,w,3) uint8, a checkpoint path (loaded once with
        process_scale=1.0, see load_screen_detail_for_dataset), or None (baselines only).
        It is run autoregressively per sequence exactly as the runtime runs (`run_sequence`): reset, then one
        infer per frame with that frame's CLAUDE.md aux dict {"alpha", "depth", "motion"} (+ extra_aux).
    Windows are non-overlapping (stride = length when the loader supports it); the default length 16 is a whole
        s3d_v1 sequence, so each (scenario, camera) is one cold-started rollout.
    sharpen: {"sigma", "amount"}; default = tune_unsharp(version, "train") (cached).
    Returns {
      "version", "split", "length", "n_sequences", "n_frames", "sharpen": {...},
      "methods": {name: {"l1", "spectrum_error", "spectrum_log_ratio", "ms_dssim",   # mean over sequences
                         "flicker", "flicker_ratio",          # pooled: sum flicker / sum flicker(hr), all seqs
                         "flicker_ratio_static", "flicker_ratio_moving",
                         "swim", "swim_ratio", "swim_ratio_moving",     # pooled: sum swim / sum swim(hr - lr)
                         "jerk", "jerk_ratio"}},                     # pooled: sum jerk / sum jerk(hr)
      "reference": {"flicker_hr", "swim_hr"},
      "per_sequence": [{"seed", "cam", "t0", "moving", name: {...}, "flicker_hr", "swim_hr"}]}
    Method names: "model" (if given), "lr", "sharpen". Lower is better everywhere.
    """
    if isinstance(model_or_callable, (str, Path)):
        model_or_callable = load_screen_detail_for_dataset(model_or_callable, device)
    ds = _load(version, split, length, stride=length)
    methods: dict[str, Callable] = {}
    if model_or_callable is not None:
        methods["model"] = model_or_callable
    sp = None
    if baselines:
        methods["lr"] = identity
        sp = sharpen or tune_unsharp(version, "train", device=device)
        methods["sharpen"] = make_unsharp(sp["sigma"], sp["amount"])
    per = []
    n_frames = 0
    for i in _subsample(len(ds), max_sequences):
        it = ds[i]
        lr = torch.as_tensor(it["lr"]).to(device)
        hr = torch.as_tensor(it["hr"]).to(device)
        aux = torch.as_tensor(it["aux"]).to(device)
        mask, _ = item_masks(it, aux, device)
        tw = mask
        motion = sequence_flow(hr) if hr.shape[0] > 1 else None      # hr's OWN apparent motion
        rec = {"index": i, "seed": _scalar(it.get("seed")), "cam": _scalar(it.get("cam")), "t0": _scalar(it.get("t0")),
               "moving": bool(_scalar(it.get("moving", False))), "mask_fraction": float(mask.float().mean().item())}
        if lr.shape[0] > 1:
            rec["flicker_hr"] = flicker(hr, motion, tw)
            rec["swim_hr"] = detail_swim(hr, lr, motion, tw)
        if lr.shape[0] > 2:
            rec["jerk_hr"] = temporal_jerk(hr, tw)
        for name, fn in methods.items():
            out = run_sequence(fn, lr, aux, extra_aux)
            rec[name] = score_sequence(out, lr, hr, aux, mask, tw)
        per.append(rec)
        n_frames += lr.shape[0]
    res = {"version": version, "split": split, "length": length, "n_sequences": len(per), "n_frames": n_frames,
           "sharpen": None if sp is None else {k: sp[k] for k in ("sigma", "amount")}, "methods": {},
           "reference": {}, "per_sequence": per}
    if not per:
        return res
    res["reference"] = {"jerk_hr": float(np.nanmean([p.get("jerk_hr", np.nan) for p in per])),
                        "flicker_hr": float(np.nanmean([p.get("flicker_hr", np.nan) for p in per])),
                        "swim_hr": float(np.nanmean([p.get("swim_hr", np.nan) for p in per]))}
    mov = lambda p: p["moving"]
    sta = lambda p: not p["moving"]
    for name in methods:
        agg = {k: float(np.nanmean([p[name][k] for p in per]))
               for k in ("l1", "spectrum_error", "spectrum_log_ratio", "ms_dssim")}
        if "flicker" in per[0][name]:
            agg.update(flicker=float(np.nanmean([p[name]["flicker"] for p in per])),
                       flicker_ratio=_pooled(per, name, "flicker", "flicker_hr"),
                       flicker_ratio_static=_pooled(per, name, "flicker", "flicker_hr", sta),
                       flicker_ratio_moving=_pooled(per, name, "flicker", "flicker_hr", mov),
                       swim=float(np.nanmean([p[name]["swim"] for p in per])),
                       swim_ratio=_pooled(per, name, "swim", "swim_hr"),
                       swim_ratio_moving=_pooled(per, name, "swim", "swim_hr", mov))
        if "jerk" in per[0][name]:
            agg.update(jerk=float(np.nanmean([p[name]["jerk"] for p in per])),
                       jerk_ratio=_pooled(per, name, "jerk", "jerk_hr"))
        res["methods"][name] = agg
    return res
