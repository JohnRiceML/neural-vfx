"""ScreenNet: 2D residual detail on the rendered frame (route B).

  rgb (n,3,H,W) in [0,1], aux (n,4,H,W) raw renderer units, prev (n,3,H,W) in [0,1] (optional)
      ── normalize aux, concat ── PixelUnshuffle(r) ── conv3x3 → N residual blocks (C) @ H/r x W/r
      ── conv3x3 → F*r^2 → PixelShuffle(r) → F ch @ H x W
  optional mid path (mid_width > 0): trunk ── conv → PixelShuffle(2) ⊕ PixelUnshuffle(2)(rgb, aux) ── conv, res blocks
      @ H/2 x W/2 ── conv (zero-init) → PixelShuffle(2) ── added to the F HR features
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
    mid_width: int = 0          # > 0: extra stride-2 path (trunk features x2 + the input at 1/2 res), added to the HR features
    mid_blocks: int = 1
    chroma_proj: bool = False   # Δrgb projected per pixel onto span{rgb, grey}: detail may scale the input colour or add grey, never new hues
    gate_dilate: int = 0        # > 0: alpha_gate uses alpha max-pooled over (2k+1)^2 px (room for thin wisps at plume edges)


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
        if cfg.mid_width:      # zero-initialised output: a checkpoint without it loads and behaves unchanged
            m, cin2 = cfg.mid_width, (3 + (AUX_CHANNELS if cfg.use_aux else 0)) * 4
            self.mid_up = nn.Conv2d(c, m * 4, 3, padding=1)
            self.mid_in = nn.Conv2d(m + cin2, m, 3, padding=1)
            self.mid_blocks = nn.Sequential(*[ResBlock2D(m, cfg.res_scale * 5) for _ in range(cfg.mid_blocks)])
            self.mid_out = nn.Conv2d(m, cfg.hr_feats * 4, 3, padding=1)
            nn.init.zeros_(self.mid_out.weight)
            nn.init.zeros_(self.mid_out.bias)
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
        x0 = x
        x = F.relu(self.head(self.unshuffle(x)), inplace=True)
        x = self.blocks(x)
        f = self.shuffle(self.tail(x))
        if self.cfg.mid_width:
            u = F.pixel_shuffle(F.relu(self.mid_up(x)), 2)
            s2 = F.pixel_unshuffle(x0[:, :3 + (AUX_CHANNELS if self.cfg.use_aux else 0)], 2)
            mm = self.mid_blocks(F.relu(self.mid_in(torch.cat([u, s2], 1)), inplace=True))
            f = f + F.pixel_shuffle(self.mid_out(mm), 2)
        x = F.relu(f)
        hr = [x, rgb] + ([prev] if self.cfg.use_prev else [])
        d = self.hr2(F.relu(self.hr1(torch.cat(hr, dim=1)), inplace=True))      # Δrgb
        if self.cfg.chroma_proj:
            d = chroma_project(d, rgb)
        k = self.cfg.grey_lowfreq
        if k:       # keep fine detail in colour, but large-scale corrections may only change brightness
            lo = F.interpolate(F.avg_pool2d(d, k), scale_factor=k, mode="bilinear", align_corners=False)
            d = d - lo + lo.mean(1, keepdim=True)
        g = self.cfg.alpha_gate
        if g and aux is not None:
            a = aux[:, 0:1].clamp(0.0, 1.0)
            if self.cfg.gate_dilate:
                a = F.max_pool2d(a, 2 * self.cfg.gate_dilate + 1, 1, self.cfg.gate_dilate)
            lim = a * g
            d = torch.minimum(torch.maximum(d, -lim), lim)
        return d


def chroma_project(d, rgb, eps: float = 1e-3):
    """Least-squares projection of d (n,3,H,W) onto span{rgb, (1,1,1)} per pixel (fp32; ridge eps for grey pixels)."""
    dt = d.dtype
    d, c = d.float(), rgb.float()
    cc, c1, d1, dc = (c * c).sum(1, keepdim=True), c.sum(1, keepdim=True), d.sum(1, keepdim=True), (d * c).sum(1, keepdim=True)
    a11, a12, a22 = cc + eps, c1, 3.0
    det = a11 * a22 - a12 * a12 + eps
    a = (a22 * dc - a12 * d1) / det
    b = (a11 * d1 - a12 * dc) / det
    return (a * c + b).to(dt)


def save_screen_net(model: ScreenNet, path, extra: dict | None = None) -> None:
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_screen_net(path, device="cuda") -> ScreenNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    names = {f.name for f in fields(ScreenNetConfig)}
    m = ScreenNet(ScreenNetConfig(**{k: v for k, v in ck["config"].items() if k in names}))
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()


def load_screen_net_extended(path, device="cuda", **override) -> ScreenNet:
    """Checkpoint weights in a (possibly larger) architecture: config = checkpoint config updated by `override`;
    parameters the checkpoint lacks (e.g. the zero-initialised mid path) keep their init. Same output as the checkpoint
    as long as the override only adds zero-initialised paths."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    names = {f.name for f in fields(ScreenNetConfig)}
    cfg = {k: v for k, v in ck["config"].items() if k in names}
    cfg.update(override)
    m = ScreenNet(ScreenNetConfig(**cfg))
    missing, unexpected = m.load_state_dict(ck["state_dict"], strict=False)
    if unexpected or any(not k.startswith("mid_") for k in missing):
        raise ValueError(f"incompatible override: missing {missing}, unexpected {unexpected}")
    return m.to(device).train()
