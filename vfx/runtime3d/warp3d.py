"""3D semi-Lagrangian warp of the previous HR volume, brick sampling, classical 3D upscale.

Conventions (vfx.core.fields3d): (C, D, H, W), index [c, z, y, x]; velocity channels (vx, vy, vz) in m/s.
grid_sample's grid last dim is (x, y, z) -> (W, H, D); align_corners=False + padding "border" is the
same sampling as F.interpolate(trilinear, align_corners=False), so brick sampling == dense upsample.
`warp_prev_3d` is the dense reference (for training); `sample_bricks` computes the same warp (and the
trilinear base) only at the HR voxels of active bricks.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from vfx.core.fields import AMBIENT_TEMP
from vfx.core.fields3d import CH3, UPSCALE_3D, FieldState3D, Volume

_DT = slice(CH3["density"], CH3["temperature"] + 1, CH3["temperature"] - CH3["density"])   # graph-safe
_GRID_CACHE: dict = {}


def base_grid_3d(D: int, H: int, W: int, device, dtype) -> torch.Tensor:
    """(1, D, H, W, 3) normalized voxel-centre coordinates (x, y, z), align_corners=False."""
    key = (D, H, W, str(device), dtype)
    g = _GRID_CACHE.get(key)
    if g is None:
        def ax(n):
            return (torch.arange(n, device=device, dtype=torch.float32) * 2 + 1) / n - 1
        zz, yy, xx = torch.meshgrid(ax(D), ax(H), ax(W), indexing="ij")
        g = torch.stack([xx, yy, zz], dim=-1).unsqueeze(0).to(dtype)
        _GRID_CACHE[key] = g
    return g


def trilinear_up(x: torch.Tensor, s: int) -> torch.Tensor:
    """(B,C,d,h,w) -> (B,C,sd,sh,sw), == F.interpolate(trilinear, align_corners=False), done as a 1D
    pass along D then a 2D bilinear pass over HW (the fp16 trilinear kernel is ~3x slower on sm_120)."""
    n, c, d, h, w = x.shape
    a = F.interpolate(x.reshape(n, c, d, h * w), size=(d * s, h * w), mode="bilinear", align_corners=False)
    a = F.interpolate(a.view(n, c * d * s, h, w), scale_factor=s, mode="bilinear", align_corners=False)
    return a.view(n, c, d * s, h * s, w * s)


def warp_prev_3d(prev_hr: torch.Tensor, lr_velocity: torch.Tensor, dx_hr: float, dt: float) -> torch.Tensor:
    """out(x) = prev(x - u(x) dt); u = trilinear-upsampled LR velocity. prev (2,D,H,W) or (B,2,D,H,W);
    lr_velocity (3,d,h,w) or (B,3,d,h,w) m/s. Same rank/dtype as prev_hr."""
    sq = prev_hr.dim() == 4
    if sq:
        prev_hr = prev_hr.unsqueeze(0)
    if lr_velocity.dim() == 4:
        lr_velocity = lr_velocity.unsqueeze(0)
    B, _, D, H, W = prev_hr.shape
    vel = trilinear_up(lr_velocity.to(prev_hr.dtype), D // lr_velocity.shape[-3])
    s = 2.0 * dt / dx_hr
    disp = torch.stack([vel[:, 0] * (s / W), vel[:, 1] * (s / H), vel[:, 2] * (s / D)], dim=-1)
    grid = base_grid_3d(D, H, W, prev_hr.device, prev_hr.dtype) - disp
    out = F.grid_sample(prev_hr, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out.squeeze(0) if sq else out


class BrickGrid:
    """Normalized HR sample coordinates for bricks of `sb` HR voxels in an N^3 HR volume (nb^3 bricks)."""

    def __init__(self, N: int, sb: int, device, dtype=torch.half):
        self.N, self.sb, self.nb = N, sb, N // sb
        l = torch.arange(sb, device=device, dtype=torch.float32) * (2.0 / N)
        lz, ly, lx = torch.meshgrid(l, l, l, indexing="ij")
        self.local = torch.stack([lx, ly, lz], dim=-1).to(dtype)               # (sb,sb,sb,3)
        o = (torch.arange(self.nb, device=device, dtype=torch.float32) * sb * 2 + 1) / N - 1
        oz, oy, ox = torch.meshgrid(o, o, o, indexing="ij")
        self.offset = torch.stack([ox, oy, oz], dim=-1).reshape(-1, 3).to(dtype)  # (nb^3, 3), id = (z*nb+y)*nb+x

    def grid(self, ids: torch.Tensor) -> torch.Tensor:
        """(n,) brick ids -> (1, n*sb, sb, sb, 3) grid stacking the bricks along D."""
        g = self.local.unsqueeze(0) + self.offset[ids].view(-1, 1, 1, 1, 3)
        return g.view(1, -1, self.sb, self.sb, 3)


def _unstack(x: torch.Tensor, n: int, sb: int) -> torch.Tensor:
    """(1, C, n*sb, sb, sb) -> (n, C, sb, sb, sb)."""
    C = x.shape[1]
    return x.view(C, n, sb, sb, sb).transpose(0, 1)


def sample_bricks(samp_lr: torch.Tensor, prev_hr: torch.Tensor, grid: torch.Tensor, n: int, sb: int,
                  dx_hr: float, dt: float):
    """samp_lr (1,5,d,h,w) = [density, T, vx_prev, vy_prev, vz_prev]; prev_hr (1,2,N,N,N).
    Returns base (n,2,sb^3) trilinear of (density,T) clamped, prev_warped (n,2,sb^3), at brick voxels."""
    s = F.grid_sample(samp_lr, grid, mode="bilinear", padding_mode="border", align_corners=False)
    N = prev_hr.shape[-1]
    k = 2.0 * dt / (dx_hr * N)
    disp = (s[:, 2:5] * k).permute(0, 2, 3, 4, 1)                              # (1, n*sb, sb, sb, 3)
    pw = F.grid_sample(prev_hr, grid - disp, mode="bilinear", padding_mode="border", align_corners=False)
    base = torch.cat([s[:, 0:1].clamp_min(0.0), s[:, 1:2].clamp_min(AMBIENT_TEMP)], dim=1)
    return _unstack(base, n, sb), _unstack(pw, n, sb)


def clamp_dt(x: torch.Tensor) -> torch.Tensor:
    """(B,2,...) density >= 0, T >= ambient. Clamping at LR is enough: trilinear weights are convex."""
    return torch.cat([x[:, 0:1].clamp_min(0.0), x[:, 1:2].clamp_min(AMBIENT_TEMP)], dim=1)


def upscale_lr_tensor_3d(x: torch.Tensor, factor: int = UPSCALE_3D, mode: str = "trilinear") -> torch.Tensor:
    """(B,2,d,h,w) density, T -> (B,2,fd,fh,fw), in the physical range."""
    x = clamp_dt(x)
    if mode == "nearest":       # broadcast copy; faster than F.interpolate(nearest) in fp16
        n, c, d, h, w = x.shape
        s = factor
        return x.view(n, c, d, 1, h, 1, w, 1).expand(n, c, d, s, h, s, w, s).reshape(n, c, d * s, h * s, w * s)
    return trilinear_up(x, factor)


def classical_upscale_3d(state: FieldState3D, factor: int = UPSCALE_3D, mode: str = "trilinear",
                         dtype: torch.dtype | None = torch.half) -> Volume:
    """Baseline: density + temperature upsampled `factor`x per axis. mode in {"trilinear","nearest"}.
    dtype=None keeps the state's dtype (float32); default float16 halves the memory traffic."""
    x = state.data[_DT].unsqueeze(0)
    if dtype is not None:
        x = x.to(dtype)
    up = upscale_lr_tensor_3d(x, factor, mode)[0]
    return Volume(density=up[0], temperature=up[1])
