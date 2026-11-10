"""Conditional discriminators for route B (tempoGAN-style, SPEC M4) — training only, never shipped.

Spatial D (one frame): sees the candidate y (output or hr) together with its condition, the lr frame and lr alpha,
plus the detail layer y - lr explicitly:  [y, lr, y - lr, alpha] = 10 channels.
So it judges "is this detail plausible FOR THIS INPUT", not "is this texture plausible".
Temporal D (K consecutive frames, unwarped): [y_k, lr_k for k, y_k - y_{k-1}, lr_k - lr_{k-1}, alpha_mid].
No warping: the stored fluid motion is not the image motion (EXPERIMENTS.md probe 0, tests3d/README); real
pairs are unwarped too, so the D learns how detail may change between frames given how the input changed.

Both: PatchGAN, spectral norm on every conv, LeakyReLU 0.2, receptive field ~ 34 px (spatial). Hinge loss,
per-patch weights = effect mask pooled to the patch grid (background patches are identical in real and fake).
`features=True` returns intermediate activations for the feature-matching term (paired: fake and real share the
same condition, so |phi(fake) - phi(real)| is a learned, conditional perceptual distance; no pretrained weights).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SN = nn.utils.spectral_norm


def _blk(cin, cout, stride):
    return nn.Sequential(SN(nn.Conv2d(cin, cout, 4 if stride == 2 else 3, stride, 1)), nn.LeakyReLU(0.2, inplace=True))


class PatchD(nn.Module):
    def __init__(self, cin: int, width: int = 64, n_down: int = 2, n_flat: int = 2):
        super().__init__()
        layers, c = [], cin
        for i in range(n_down):
            co = width * (2 ** i)
            layers.append(_blk(c, co, 2))
            c = co
        for _ in range(n_flat):
            layers.append(_blk(c, c, 1))
        self.layers = nn.ModuleList(layers)
        self.out = SN(nn.Conv2d(c, 1, 3, 1, 1))
        self.stride = 2 ** n_down

    def forward(self, x, features=False):
        feats = []
        for l in self.layers:
            x = l(x)
            feats.append(x)
        o = self.out(x)
        return (o, feats) if features else o


def spatial_input(y, lr, alpha):
    """y, lr (N,3,H,W) 0..1, alpha (N,1,H,W)."""
    return torch.cat([y * 2 - 1, lr * 2 - 1, (y - lr) * 4, alpha * 2 - 1], 1)


def temporal_input(y, lr, alpha):
    """y, lr (B,K,3,H,W), alpha (B,K,1,H,W) -> (B, 6K + 6(K-1) + 1, H, W)."""
    K = y.shape[1]
    parts = [y.flatten(1, 2) * 2 - 1, lr.flatten(1, 2) * 2 - 1,
             ((y[:, 1:] - y[:, :-1]) * 8).flatten(1, 2), ((lr[:, 1:] - lr[:, :-1]) * 8).flatten(1, 2),
             alpha[:, K // 2] * 2 - 1]
    return torch.cat(parts, 1)


def temporal_channels(K: int) -> int:
    return 6 * K + 6 * (K - 1) + 1


def patch_weight(mask, stride):
    """mask (N,1,H,W) -> (N,1,H/s,W/s) fraction of effect pixels per patch cell."""
    return F.avg_pool2d(mask.float(), stride)


def _wmean(x, w):
    return (x * w).sum() / w.sum().clamp_min(1.0)


def d_hinge(real, fake, w):
    return _wmean(F.relu(1 - real), w) + _wmean(F.relu(1 + fake), w)


def g_hinge(fake, w):
    return -_wmean(fake, w)


def feature_match(f_fake, f_real, w_full):
    """Mean over layers of masked |phi_fake - phi_real| (real detached)."""
    tot = 0.0
    for a, b in zip(f_fake, f_real):
        w = F.interpolate(w_full, size=a.shape[-2:], mode="area")
        tot = tot + _wmean((a - b.detach()).abs().mean(1, keepdim=True), w)
    return tot / len(f_fake)


def r1_penalty(d_out_real, x_real, w):
    g, = torch.autograd.grad(_wmean(d_out_real, w) * w.sum(), x_real, create_graph=True)
    return g.pow(2).flatten(1).sum(1).mean()


def translate_aug(xs, max_shift: int, gen: torch.Generator | None = None):
    """Same random integer translation (reflect pad + crop) applied to every tensor of the list (DiffAugment-style,
    keeps real/fake/condition aligned). xs: list of (N,C,H,W)."""
    if max_shift <= 0:
        return xs
    N = xs[0].shape[0]
    dx = torch.randint(-max_shift, max_shift + 1, (1,), generator=gen).item()
    dy = torch.randint(-max_shift, max_shift + 1, (1,), generator=gen).item()
    return [torch.roll(x, (dy, dx), (-2, -1)) for x in xs]
