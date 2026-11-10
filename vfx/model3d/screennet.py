"""ScreenNet: 2D residual detail on the rendered frame (route B).

  rgb (n,3,H,W) in [0,1], aux (n,4,H,W) raw renderer units, prev (n,3,H,W) in [0,1] (optional)
      ── normalize aux, concat ── PixelUnshuffle(r) ── conv3x3 → N residual blocks (C) @ H/r x W/r
      ── conv3x3 → F*r^2 → PixelShuffle(r) → F ch @ H x W
  HR head: concat[F feats, rgb, prev] → conv k → ReLU → conv k → Δrgb ; out = rgb + Δrgb

H, W must be multiples of r (and of grey_lowfreq when set). The last conv is zero-initialised: untrained output == input frame.
aux channels (AUX_CHANNELS = 4), in the units of the processed image (vfx/render3d README, pooled like
vfx/datagen3d): alpha 0..1, depth in metres (0 where alpha == 0), motion x, y in processed px per SIM_DT.
They are normalized INSIDE the model (depth * depth_scale, motion * motion_scale, clipped to ±input_clip),
so callers pass physical units.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

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
    use_aux: bool = True        # alpha, depth, motion
    use_prev: bool = True       # previous output (runtime warps it by prev_warp x motion; 0 = unwarped)
    res_scale: float = 0.1
    depth_scale: float = 0.5    # 1/m
    motion_scale: float = 0.125  # 1/(px per step)
    input_clip: float = 8.0
    prev_warp: float = 0.0      # fraction of `motion` used to align the previous output (runtime + training)
    alpha_gate: float = 0.0     # k > 0: |Δrgb| <= k * alpha (detail only where the medium is; no halos on background)
    grey_lowfreq: int = 0       # k > 0: the part of Δrgb coarser than k px is made grey (no large-scale colour shift)


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
        self.register_buffer("aux_scale", torch.tensor([1.0, cfg.depth_scale, cfg.motion_scale, cfg.motion_scale])
                             .view(1, AUX_CHANNELS, 1, 1), persistent=False)

    def norm_aux(self, aux):
        k = self.cfg.input_clip
        return (aux * self.aux_scale.to(aux.dtype)).clamp(-k, k)

    def forward(self, rgb, aux=None, prev=None):
        parts = [rgb]
        if self.cfg.use_aux:
            parts.append(self.norm_aux(aux))
        if self.cfg.use_prev:
            parts.append(prev)
        x = torch.cat(parts, dim=1)
        x = F.relu(self.head(self.unshuffle(x)), inplace=True)
        x = self.blocks(x)
        x = F.relu(self.shuffle(self.tail(x)))
        hr = [x, rgb] + ([prev] if self.cfg.use_prev else [])
        d = self.hr2(F.relu(self.hr1(torch.cat(hr, dim=1)), inplace=True))      # Δrgb
        k = self.cfg.grey_lowfreq
        if k:       # keep fine detail in colour, but large-scale corrections may only change brightness
            lo = F.interpolate(F.avg_pool2d(d, k), scale_factor=k, mode="bilinear", align_corners=False)
            d = d - lo + lo.mean(1, keepdim=True)
        g = self.cfg.alpha_gate
        if g and aux is not None:
            lim = aux[:, 0:1].clamp(0.0, 1.0) * g
            d = torch.minimum(torch.maximum(d, -lim), lim)
        return d


def save_screen_net(model: ScreenNet, path, extra: dict | None = None) -> None:
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_screen_net(path, device="cuda") -> ScreenNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    names = {f.name for f in fields(ScreenNetConfig)}
    m = ScreenNet(ScreenNetConfig(**{k: v for k, v in ck["config"].items() if k in names}))
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()
