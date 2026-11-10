"""VolumeNet: 3D residual detail upscaler (route A), density + temperature, x4 (or x2) per axis.

Same recipe as the 2D DetailNet, moved to 3D and to bricks:

  lr brick (n,6,S,S,S), S = b + 2*halo ──norm── conv3x3x3 ──┐
  prev_warped core (n,2,sb,sb,sb) ──norm── unshuffle3d(s) ── conv1x1x1 ── zero-pad halo ──┤ + → ReLU
                                                                                         ▼
                      N residual blocks (width C) at LR, on S^3 → crop halo → b^3
                      conv1x1x1 → F*s^3 ch → pixel_shuffle3d(s) → F ch @ (sb)^3
                                                                                         ▼
  HR head: concat[F feats, prev (2), base (2)] → conv k → ReLU → conv k → Δ(2)   (k = 1 by default)
  out = base + Δ  (base = trilinear(lr density, T), supplied by the runtime)  → denorm, clamp

The last HR conv is zero-initialised, so an untrained net returns exactly the trilinear base.

Block variants (all conv3d, so cuDNN picks the kernels):
  full : 3x3x3 → ReLU → 3x3x3                                    54 C^2 params / block
  sep  : (2+1)D, [1x3x3, 3x1x1] → ReLU → [1x3x3, 3x1x1]          24 C^2
  mp   : multi-pass 2D (SPEC "Multi-pass GAN" idea): 2D convs over slices along one axis, the axis
         cycling z, y, x across convs: 1x3x3 → ReLU → 3x1x3 (next block 3x3x1 → 1x3x3 ...)  18 C^2
  dw   : depthwise 3x3x3 + pointwise 1x1x1, twice                 2 (C^2 + 27 C)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from vfx.core.fields3d import CH3, NUM_CHANNELS_3D
from vfx.core.fields import AMBIENT_TEMP

_AXIS_K = [(1, 3, 3), (3, 1, 3), (3, 3, 1)]


@dataclass
class VolumeNetConfig:
    width: int = 32            # trunk width C at LR
    n_blocks: int = 2          # residual blocks at LR
    block: str = "full"        # full | sep | mp | dw
    hr_feats: int = 4          # channels after the 3D pixel shuffle
    hr_hidden: int = 8         # HR head hidden width
    hr_kernel: int = 1         # 1 = pointwise HR head (cheap); 3 = 3x3x3 (zero-padded at brick borders)
    scale: int = 4
    # normalization (physical -> network units); stored in the checkpoint
    density_scale: float = 1.0
    vel_scale: float = 1.0
    temp_offset: float = AMBIENT_TEMP
    temp_scale: float = 1000.0
    fuel_scale: float = 1.0
    res_scale: float = 0.1
    input_clip: float = 20.0
    output_clip: float = 50.0


def pixel_shuffle3d(x: torch.Tensor, s: int) -> torch.Tensor:
    n, c, d, h, w = x.shape
    f = c // (s ** 3)
    x = x.reshape(n, f, s, s, s, d, h, w).permute(0, 1, 5, 2, 6, 3, 7, 4)
    return x.reshape(n, f, d * s, h * s, w * s)


def pixel_unshuffle3d(x: torch.Tensor, s: int) -> torch.Tensor:
    n, c, D, H, W = x.shape
    d, h, w = D // s, H // s, W // s
    x = x.reshape(n, c, d, s, h, s, w, s).permute(0, 1, 3, 5, 7, 2, 4, 6)
    return x.reshape(n, c * s ** 3, d, h, w)


def _conv(cin, cout, k, groups=1):
    k = (k, k, k) if isinstance(k, int) else k
    return nn.Conv3d(cin, cout, k, padding=tuple(kk // 2 for kk in k), groups=groups)


class Block3D(nn.Module):
    def __init__(self, c: int, kind: str, idx: int, res_scale: float):
        super().__init__()
        self.res_scale = res_scale
        if kind == "full":
            self.a, self.b = _conv(c, c, 3), _conv(c, c, 3)
        elif kind == "sep":
            self.a = nn.Sequential(_conv(c, c, (1, 3, 3)), _conv(c, c, (3, 1, 1)))
            self.b = nn.Sequential(_conv(c, c, (1, 3, 3)), _conv(c, c, (3, 1, 1)))
        elif kind == "mp":
            self.a = _conv(c, c, _AXIS_K[(2 * idx) % 3])
            self.b = _conv(c, c, _AXIS_K[(2 * idx + 1) % 3])
        elif kind == "dw":
            self.a = nn.Sequential(_conv(c, c, 3, groups=c), _conv(c, c, 1))
            self.b = nn.Sequential(_conv(c, c, 3, groups=c), _conv(c, c, 1))
        else:
            raise ValueError(kind)

    def forward(self, x):
        return x + self.res_scale * self.b(F.relu(self.a(x), inplace=True))


class VolumeNet(nn.Module):
    def __init__(self, cfg: VolumeNetConfig | None = None, **kw):
        super().__init__()
        cfg = cfg or VolumeNetConfig(**kw)
        self.cfg = cfg
        s, c = cfg.scale, cfg.width
        self.head = _conv(NUM_CHANNELS_3D, c, 3)
        self.prev_in = _conv(2 * s ** 3, c, 1)
        self.blocks = nn.Sequential(*[Block3D(c, cfg.block, i, cfg.res_scale) for i in range(cfg.n_blocks)])
        self.tail = _conv(c, cfg.hr_feats * s ** 3, 1)
        self.hr1 = _conv(cfg.hr_feats + 4, cfg.hr_hidden, cfg.hr_kernel)
        self.hr2 = _conv(cfg.hr_hidden, 2, cfg.hr_kernel)
        nn.init.zeros_(self.hr2.weight)       # untrained net == trilinear base
        nn.init.zeros_(self.hr2.bias)
        sc = torch.ones(NUM_CHANNELS_3D)
        off = torch.zeros(NUM_CHANNELS_3D)
        sc[CH3["density"]] = cfg.density_scale
        for k in ("vel_x", "vel_y", "vel_z"):
            sc[CH3[k]] = cfg.vel_scale
        sc[CH3["temperature"]] = cfg.temp_scale
        sc[CH3["fuel"]] = cfg.fuel_scale
        off[CH3["temperature"]] = cfg.temp_offset
        self.register_buffer("lr_off", off.view(1, -1, 1, 1, 1), persistent=False)
        self.register_buffer("lr_inv", (1.0 / sc).view(1, -1, 1, 1, 1), persistent=False)
        self.register_buffer("hr_off", torch.tensor([0.0, cfg.temp_offset]).view(1, 2, 1, 1, 1), persistent=False)
        self.register_buffer("hr_scale", torch.tensor([cfg.density_scale, cfg.temp_scale]).view(1, 2, 1, 1, 1),
                             persistent=False)

    def norm_hr(self, x):
        return (x - self.hr_off.to(x.dtype)) / self.hr_scale.to(x.dtype)

    def denorm_hr(self, x):
        return x * self.hr_scale.to(x.dtype) + self.hr_off.to(x.dtype)

    def trunk(self, lr, prevn_unshuffled, halo: int = 0):
        """LR part. lr (n,6,S,S,S) physical; prevn_unshuffled (n,2*s^3,b,b,b) = pixel_unshuffle3d of the
        normalized, clipped warped previous output. Returns the tail output (n, hr_feats*s^3, b, b, b)."""
        k, dt = self.cfg.input_clip, lr.dtype
        lrn = ((lr - self.lr_off.to(dt)) * self.lr_inv.to(dt)).clamp(-k, k)
        p = self.prev_in(prevn_unshuffled)
        if halo:
            p = F.pad(p, (halo,) * 6)
        x = F.relu(self.head(lrn) + p, inplace=True)
        x = self.blocks(x)
        if halo:
            x = x[:, :, halo:-halo, halo:-halo, halo:-halo]
        return self.tail(x)

    def forward_normalized(self, lr, prev, base, halo: int = 0):
        """lr (n,6,S,S,S) physical with `halo` LR cells of context on each side; prev, base (n,2,sb,sb,sb)
        physical for the core b = S - 2*halo. Returns the normalized unclamped prediction (n,2,sb,sb,sb)."""
        k, s = self.cfg.input_clip, self.cfg.scale
        prevn = self.norm_hr(prev).clamp(-k, k)
        basen = self.norm_hr(base)
        x = pixel_shuffle3d(self.trunk(lr, pixel_unshuffle3d(prevn, s), halo), s)
        x = torch.cat([F.relu(x), prevn, basen.clamp(-k, k)], dim=1)
        return basen + self.hr2(F.relu(self.hr1(x), inplace=True))

    def forward(self, lr, prev, base, halo: int = 0):
        y = self.forward_normalized(lr, prev, base, halo).clamp(0.0, self.cfg.output_clip)
        return self.denorm_hr(y)


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def save_volume_net(model: VolumeNet, path, extra: dict | None = None) -> None:
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_volume_net(path, device="cuda") -> VolumeNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = VolumeNet(VolumeNetConfig(**ck["config"]))
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()
