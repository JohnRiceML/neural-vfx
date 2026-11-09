"""Losses in normalized units (B,2,H,W): L1, gradient, high-frequency band, radial log spectrum,
temporal change consistency."""
from __future__ import annotations

import torch
import torch.nn.functional as F

_BINS: dict = {}


def grad_l1(p, t):
    dpx, dtx = p[..., :, 1:] - p[..., :, :-1], t[..., :, 1:] - t[..., :, :-1]
    dpy, dty = p[..., 1:, :] - p[..., :-1, :], t[..., 1:, :] - t[..., :-1, :]
    return (dpx - dtx).abs().mean() + (dpy - dty).abs().mean()


def highpass(x, k: int = 4):
    low = F.interpolate(F.avg_pool2d(x, k), scale_factor=k, mode="bilinear", align_corners=False)
    return x - low


def hf_l1(p, t, k: int = 4):
    return (highpass(p, k) - highpass(t, k)).abs().mean()


def _radial(H, W, device):
    key = (H, W, str(device))
    if key not in _BINS:
        fy = torch.fft.fftfreq(H, device=device) * H
        fx = torch.fft.rfftfreq(W, device=device) * W
        r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2).round().long()
        nb = min(H, W) // 2
        r = r.clamp(max=nb)          # bin nb collects the corners (dropped below)
        cnt = torch.bincount(r.flatten(), minlength=nb + 1).float()
        win = torch.hann_window(H, device=device)[:, None] * torch.hann_window(W, device=device)[None, :]
        _BINS[key] = (r.flatten(), cnt, nb, win)
    return _BINS[key]


def radial_power(x):
    """x (B,C,H,W) fp32 -> (B,C,nb) radially averaged power spectrum (Hann-windowed)."""
    B, C, H, W = x.shape
    r, cnt, nb, win = _radial(H, W, x.device)
    xw = (x - x.mean(dim=(-2, -1), keepdim=True)) * win
    P = torch.fft.rfft2(xw, norm="ortho").abs() ** 2
    P = P.reshape(B * C, -1)
    out = torch.zeros(B * C, nb + 1, device=x.device).index_add_(1, r, P) / cnt
    return out[:, 1:nb].reshape(B, C, nb - 1)


def spectral_loss(p, t, eps: float = 1e-8):
    """Mean |log P_pred - log P_ref| over radial frequency bins (computed in fp32)."""
    with torch.autocast("cuda", enabled=False):
        Pp, Pt = radial_power(p.float()), radial_power(t.float())
        scale = Pt.mean(dim=-1, keepdim=True) * 1e-4 + eps     # floor relative to the signal energy
        return (torch.log(Pp + scale) - torch.log(Pt + scale)).abs().mean()


def temporal_l1(pred, prev_pred_warped, target, prev_target_warped):
    """Match the frame-to-frame change: (pred_t - W(pred_{t-1})) vs (ref_t - W(ref_{t-1}))."""
    return ((pred - prev_pred_warped) - (target - prev_target_warped)).abs().mean()
