"""Image-space training losses for route B (frames (B,3,H,W) in 0..1). The evaluation metrics live in
vfx/eval3d/metrics.py (test engineer); these are differentiable training surrogates only."""
from __future__ import annotations

import torch
import torch.nn.functional as F

LUMA = (0.2126, 0.7152, 0.0722)
FLOOR = (1.0 / 255.0) ** 2 / 12.0        # uint8 quantization noise power (same floor as the eval metric)
_CACHE: dict = {}


def luma(x):
    w = torch.tensor(LUMA, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * w).sum(1, keepdim=True)


def wl1(p, t, w):
    """Weighted L1, w (B,1,H,W)."""
    return ((p - t).abs() * w).sum() / (w.sum() * p.shape[1]).clamp_min(1.0)


def grad_l1(p, t, w):
    dpx, dtx = p[..., :, 1:] - p[..., :, :-1], t[..., :, 1:] - t[..., :, :-1]
    dpy, dty = p[..., 1:, :] - p[..., :-1, :], t[..., 1:, :] - t[..., :-1, :]
    return wl1(dpx, dtx, w[..., :, 1:]) + wl1(dpy, dty, w[..., 1:, :])


def _bins(H, W, device):
    key = (H, W, str(device))
    if key not in _CACHE:
        fy = torch.fft.fftfreq(H, device=device)
        fx = torch.fft.rfftfreq(W, device=device)
        f = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)          # cycles / pixel
        nb = 32
        edges = torch.linspace(1.0 / 64.0, 0.5, nb + 1, device=device)
        idx = (torch.bucketize(f.flatten(), edges, right=True) - 1)
        keep = (idx >= 0) & (idx < nb)
        idx = torch.where(keep, idx, torch.full_like(idx, nb))
        cnt = torch.bincount(idx, minlength=nb + 1).float()
        win = torch.outer(torch.hann_window(H, periodic=False, device=device),
                          torch.hann_window(W, periodic=False, device=device))
        _CACHE[key] = (idx, cnt, nb, win, win.pow(2).sum())
    return _CACHE[key]


def radial_power(y):
    """y (B,1,H,W) luminance -> (B, 32) mean power per radial bin (eval metric's bins: linear in cycles/px)."""
    B, _, H, W = y.shape
    idx, cnt, nb, win, wsum = _bins(H, W, y.device)
    yc = (y - y.mean(dim=(-2, -1), keepdim=True))[:, 0] * win
    P = torch.fft.rfft2(yc).abs().pow(2) / wsum
    s = torch.zeros(B, nb + 1, device=y.device).index_add_(1, idx, P.reshape(B, -1))
    return (s / cnt.clamp_min(1.0))[:, :nb]


def spectral_loss(p, t, max_frac: float = 1.0, min_std: float = 0.01):
    """Mean |ln(P_pred + F) - ln(P_ref + F)| over the lowest `max_frac` of the eval metric's 32 bins, luminance,
    per sample; nearly flat targets are skipped. fp32."""
    with torch.autocast("cuda", enabled=False):
        lp, lt = luma(p.float()), luma(t.float())
        Pp, Pt = radial_power(lp), radial_power(lt)
        nb = max(1, int(round(Pp.shape[-1] * max_frac)))
        e = (torch.log(Pp[:, :nb] + FLOOR) - torch.log(Pt[:, :nb] + FLOOR)).abs().mean(-1)
        w = (lt.std(dim=(-2, -1))[:, 0] > min_std).float()
        return (e * w).sum() / w.sum().clamp_min(1.0)


def _gblur(x, sigma=1.5):
    r = 4
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))


def ms_dssim(p, t, w, weights=(0.5, 0.3, 0.2)):
    """Differentiable surrogate of the structural metric: 1 - sum_s w_s masked-mean SSIM of luminance at
    scales 1, 1/2, 1/4 (Gaussian sigma 1.5, C1 = 0.01^2, C2 = 0.03^2). fp32; w (B,1,H,W)."""
    with torch.autocast("cuda", enabled=False):
        a, b, m = luma(p.float()), luma(t.float()), w.float()
        C1, C2, tot = 0.01 ** 2, 0.03 ** 2, 0.0
        for i, ws in enumerate(weights):
            if i:
                a, b, m = F.avg_pool2d(a, 2), F.avg_pool2d(b, 2), F.avg_pool2d(m, 2)
            ma, mb = _gblur(a), _gblur(b)
            saa, sbb, sab = _gblur(a * a) - ma ** 2, _gblur(b * b) - mb ** 2, _gblur(a * b) - ma * mb
            s = ((2 * ma * mb + C1) * (2 * sab + C2)) / ((ma ** 2 + mb ** 2 + C1) * (saa + sbb + C2))
            tot = tot + ws * (s * m).sum() / m.sum().clamp_min(1.0)
        return 1.0 - tot


def med3(x):
    """3x3 median (replicate padding), per channel."""
    B, C, H, W = x.shape
    u = F.unfold(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3).view(B, C, 9, H, W)
    return u.median(dim=2).values


def speckle_l1(p, t, w):
    """Isolated-pixel structure of luminance: (x - med3(x)) of the prediction vs the reference. An isolated dot the reference
    does not have costs its full amplitude; real filaments (median-preserved) cost little."""
    lp, lt = luma(p), luma(t)
    return wl1(lp - med3(lp), lt - med3(lt), w)


def temporal_l1(p, p_prev, t, t_prev, w):
    """Match the (unwarped) frame-to-frame change of the reference: (p_t - p_{t-1}) vs (h_t - h_{t-1})."""
    return wl1(p - p_prev, t - t_prev, w)
