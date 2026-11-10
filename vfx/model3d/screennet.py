"""ScreenNet: 2D residual detail on the rendered frame (route B).

  rgb (n,3,H,W) in [0,1], aux (n,A,H,W), prev_out_warped (n,3,H,W)
      ── concat ── PixelUnshuffle(r) ── conv3x3 → N residual blocks (C) @ H/r x W/r
      ── conv3x3 → F*r^2 → PixelShuffle(r) → F ch @ H x W
  HR head: concat[F feats, rgb, prev] → conv k → ReLU → conv k → Δrgb ; out = rgb + Δrgb

H, W must be multiples of r. The last conv is zero-initialised: untrained output == input frame.
aux channels (A = 4 when use_aux): transmittance, depth (normalized), screen velocity x, y (px/frame / 16).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

AUX_CHANNELS = 4


@dataclass
class ScreenNetConfig:
    width: int = 32
    n_blocks: int = 4
    trunk_stride: int = 4       # r: trunk runs at 1/r of the processed resolution
    hr_feats: int = 4
    hr_hidden: int = 8
    hr_kernel: int = 1
    use_aux: bool = True        # transmittance, depth, velocity
    use_prev: bool = True       # previous output, warped by screen velocity
    res_scale: float = 0.1


class ResBlock2D(nn.Module):
    def __init__(self, c, res_scale):
        super().__init__()
        self.res_scale = res_scale
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.res_scale * self.c2(F.relu(self.c1(x), inplace=True))


class ScreenNet(nn.Module):
    def __init__(self, cfg: ScreenNetConfig | None = None, **kw):
        super().__init__()
        cfg = cfg or ScreenNetConfig(**kw)
        self.cfg = cfg
        r, c = cfg.trunk_stride, cfg.width
        cin = 3 + (AUX_CHANNELS if cfg.use_aux else 0) + (3 if cfg.use_prev else 0)
        self.n_in = cin
        self.unshuffle = nn.PixelUnshuffle(r)
        self.shuffle = nn.PixelShuffle(r)
        self.head = nn.Conv2d(cin * r * r, c, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock2D(c, cfg.res_scale) for _ in range(cfg.n_blocks)])
        self.tail = nn.Conv2d(c, cfg.hr_feats * r * r, 3, padding=1)
        k = cfg.hr_kernel
        hr_in = cfg.hr_feats + 3 + (3 if cfg.use_prev else 0)
        self.hr1 = nn.Conv2d(hr_in, cfg.hr_hidden, k, padding=k // 2)
        self.hr2 = nn.Conv2d(cfg.hr_hidden, 3, k, padding=k // 2)
        nn.init.zeros_(self.hr2.weight)
        nn.init.zeros_(self.hr2.bias)

    def forward(self, rgb, aux=None, prev=None):
        parts = [rgb]
        if self.cfg.use_aux:
            parts.append(aux)
        if self.cfg.use_prev:
            parts.append(prev)
        x = torch.cat(parts, dim=1)
        x = F.relu(self.head(self.unshuffle(x)), inplace=True)
        x = self.blocks(x)
        x = F.relu(self.shuffle(self.tail(x)))
        hr = [x, rgb] + ([prev] if self.cfg.use_prev else [])
        return self.hr2(F.relu(self.hr1(torch.cat(hr, dim=1)), inplace=True))   # Δrgb


def save_screen_net(model: ScreenNet, path, extra: dict | None = None) -> None:
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_screen_net(path, device="cuda") -> ScreenNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ScreenNet(ScreenNetConfig(**ck["config"]))
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()
