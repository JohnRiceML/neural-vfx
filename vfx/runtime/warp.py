"""Semi-Lagrangian warp of the previous HR frame and classical upscaling baselines.

`warp_prev` is shared by training (vfx/train) and inference (Upscaler) so both see the exact
same warping. Conventions follow vfx.core.fields: (C, H, W), row 0 = bottom (y up), velocity in m/s.
grid_sample maps normalized -1 to row/col 0, so +vel_y moves content toward higher row indices.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from vfx.core.fields import AMBIENT_TEMP, CH, UPSCALE, FieldState, HRFields

_GRID_CACHE: dict = {}


def _base_grid(H: int, W: int, device, dtype) -> torch.Tensor:
    """(1, H, W, 2) normalized pixel-center coordinates (x, y), align_corners=False."""
    key = (H, W, str(device), dtype)
    g = _GRID_CACHE.get(key)
    if g is None:
        ys = (torch.arange(H, device=device, dtype=torch.float32) * 2 + 1) / H - 1
        xs = (torch.arange(W, device=device, dtype=torch.float32) * 2 + 1) / W - 1
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        g = torch.stack([xx, yy], dim=-1).unsqueeze(0).to(dtype)
        _GRID_CACHE[key] = g
    return g


def upsample_velocity(lr_velocity: torch.Tensor, size: tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    """(B,2,h,w) -> (B,2,H,W). Velocity is a physical quantity (m/s), so values are not rescaled."""
    return F.interpolate(lr_velocity, size=size, mode=mode, align_corners=False)


def warp_prev(prev_hr: torch.Tensor, lr_velocity: torch.Tensor, dx_hr: float, dt: float,
              vel_mode: str = "bilinear") -> torch.Tensor:
    """Advect prev HR fields one step along the (upsampled) LR velocity, semi-Lagrangian.

    prev_hr: (2,H,W) or (B,2,H,W); lr_velocity: (2,h,w) or (B,2,h,w) m/s (vel_x, vel_y).
    out(x) = prev(x - u(x) * dt), bilinear sampling, clamped at the box border.
    Returns the same rank/dtype as prev_hr.
    """
    squeeze = prev_hr.dim() == 3
    if squeeze:
        prev_hr = prev_hr.unsqueeze(0)
    if lr_velocity.dim() == 3:
        lr_velocity = lr_velocity.unsqueeze(0)
    B, _, H, W = prev_hr.shape
    vel = upsample_velocity(lr_velocity.to(prev_hr.dtype), (H, W), vel_mode)
    # displacement in normalized units: cells = u*dt/dx; normalized = 2*cells/W
    sx = 2.0 * dt / (dx_hr * W)
    sy = 2.0 * dt / (dx_hr * H)
    disp = torch.stack([vel[:, 0] * sx, vel[:, 1] * sy], dim=-1)          # (B,H,W,2)
    grid = _base_grid(H, W, prev_hr.device, prev_hr.dtype) - disp
    out = F.grid_sample(prev_hr, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out.squeeze(0) if squeeze else out


def upscale_lr_tensor(lr: torch.Tensor, mode: str = "bicubic", scale: int = UPSCALE) -> torch.Tensor:
    """(B,2,h,w) density,temperature -> (B,2,sh,sw), clamped to physical range."""
    if mode == "nearest":
        up = F.interpolate(lr, scale_factor=scale, mode="nearest")
    else:
        up = F.interpolate(lr, scale_factor=scale, mode=mode, align_corners=False)
    d = up[:, 0:1].clamp_min(0.0)
    t = up[:, 1:2].clamp_min(AMBIENT_TEMP)
    return torch.cat([d, t], dim=1)


def classical_upscale(lr_state: FieldState, mode: str = "bicubic") -> HRFields:
    """Baseline 4x upscale of density + temperature. mode in {"bicubic","bilinear","nearest"}."""
    idx = [CH["density"], CH["temperature"]]
    x = lr_state.data[idx].unsqueeze(0)
    up = upscale_lr_tensor(x, mode)[0]
    return HRFields(density=up[0], temperature=up[1])
