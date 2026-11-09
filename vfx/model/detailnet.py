"""DetailNet: 4x residual detail upscaler for density + temperature (M3-style, non-adversarial).

Physical units in and out; normalization lives inside the model.

  lr (B,5,h,w) ──norm──┐
  prev_hr_warped ──norm── PixelUnshuffle(4) ──┤ concat (5+32 ch) @ h×w
                                              ▼
                     conv3x3 → N residual blocks (C ch) → conv3x3 → PixelShuffle(4) → F ch @ 4h×4w
                                              ▼
  HR head @ 4h×4w:  concat[F feats, prev_warped (2), smooth base (2)] → conv3x3 → ReLU → conv3x3 → Δ (2)
  out = base + Δ  (base = bicubic(lr density, temperature), normalized)  → denorm → density ≥ 0, T ≥ ambient

Almost all FLOPs run at LR (h×w); the HR head is two thin convs, which keeps 512² inference cheap.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from vfx.core.fields import AMBIENT_TEMP, CH, UPSCALE

# density (0) and temperature (3) as a slice, so no index tensor is created (CUDA-graph safe)
_DT_SLICE = slice(CH["density"], CH["temperature"] + 1, CH["temperature"] - CH["density"])


@dataclass
class DetailNetConfig:
    lr_channels: int = 64        # trunk width at LR
    n_blocks: int = 6            # residual blocks at LR
    hr_feats: int = 8            # channels after PixelShuffle
    hr_hidden: int = 16          # HR head hidden width
    scale: int = UPSCALE
    # normalization (physical -> network units); stored in the checkpoint
    density_scale: float = 1.0
    vel_scale: float = 1.0       # m/s
    temp_offset: float = AMBIENT_TEMP
    temp_scale: float = 1000.0   # K
    fuel_scale: float = 1.0
    # stability (fp16): residual branch scale (EDSR-style), clip of normalized trunk inputs/outputs
    res_scale: float = 0.1
    input_clip: float = 20.0
    output_clip: float = 50.0


class ResBlock(nn.Module):
    def __init__(self, c: int, res_scale: float = 0.1):
        super().__init__()
        self.res_scale = res_scale
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.res_scale * self.c2(F.relu(self.c1(x), inplace=True))


class DetailNet(nn.Module):
    def __init__(self, cfg: DetailNetConfig | None = None, **kw):
        super().__init__()
        cfg = cfg or DetailNetConfig(**kw)
        self.cfg = cfg
        s = cfg.scale
        c = cfg.lr_channels
        self.head = nn.Conv2d(5 + 2 * s * s, c, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(c, cfg.res_scale) for _ in range(cfg.n_blocks)])
        self.tail = nn.Conv2d(c, cfg.hr_feats * s * s, 3, padding=1)
        self.shuffle = nn.PixelShuffle(s)
        self.unshuffle = nn.PixelUnshuffle(s)
        self.hr1 = nn.Conv2d(cfg.hr_feats + 4, cfg.hr_hidden, 3, padding=1)
        self.hr2 = nn.Conv2d(cfg.hr_hidden, 2, 3, padding=1)
        nn.init.zeros_(self.hr2.weight)   # start as the smooth upsample
        nn.init.zeros_(self.hr2.bias)
        lr_scale = torch.tensor([cfg.density_scale, cfg.vel_scale, cfg.vel_scale, cfg.temp_scale, cfg.fuel_scale])
        lr_off = torch.zeros(5)
        lr_off[CH["temperature"]] = cfg.temp_offset
        self.register_buffer("lr_off", lr_off.view(1, 5, 1, 1), persistent=False)
        self.register_buffer("lr_inv", (1.0 / lr_scale).view(1, 5, 1, 1), persistent=False)
        self.register_buffer("hr_off", torch.tensor([0.0, cfg.temp_offset]).view(1, 2, 1, 1), persistent=False)
        self.register_buffer("hr_scale", torch.tensor([cfg.density_scale, cfg.temp_scale]).view(1, 2, 1, 1),
                             persistent=False)

    # --- normalization helpers (also used by losses) ---
    def norm_hr(self, x):
        return (x - self.hr_off.to(x.dtype)) / self.hr_scale.to(x.dtype)

    def denorm_hr(self, x):
        return x * self.hr_scale.to(x.dtype) + self.hr_off.to(x.dtype)

    def forward_normalized(self, lr: torch.Tensor, prev_hr_warped: torch.Tensor) -> torch.Tensor:
        """Returns the unclamped prediction in normalized units (B,2,H,W). Used by training."""
        dt = lr.dtype
        lrn = (lr - self.lr_off.to(dt)) * self.lr_inv.to(dt)
        prevn = self.norm_hr(prev_hr_warped.to(dt))
        base = F.interpolate(lrn[:, _DT_SLICE], scale_factor=self.cfg.scale,
                             mode="bicubic", align_corners=False)
        k = self.cfg.input_clip
        x = torch.cat([lrn.clamp(-k, k), self.unshuffle(prevn.clamp(-k, k))], dim=1)
        x = F.relu(self.head(x), inplace=True)
        x = self.blocks(x)
        x = self.shuffle(self.tail(x))
        x = torch.cat([F.relu(x), prevn.clamp(-k, k), base.clamp(-k, k)], dim=1)
        delta = self.hr2(F.relu(self.hr1(x), inplace=True))
        return base + delta

    def clamp_normalized(self, y: torch.Tensor) -> torch.Tensor:
        # density >= 0, temperature >= ambient (temp_offset == ambient); upper clip guards long rollouts
        return y.clamp(0.0, self.cfg.output_clip)

    def forward(self, lr: torch.Tensor, prev_hr_warped: torch.Tensor) -> torch.Tensor:
        """lr (B,5,h,w) physical; prev_hr_warped (B,2,4h,4w) density, T(K) -> (B,2,4h,4w) density, T(K)."""
        return self.denorm_hr(self.clamp_normalized(self.forward_normalized(lr, prev_hr_warped)))


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def save_model(model: DetailNet, path, extra: dict | None = None) -> None:
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_model(path, device="cuda") -> DetailNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = DetailNet(DetailNetConfig(**ck["config"]))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()
