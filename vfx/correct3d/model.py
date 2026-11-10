"""CorrNet: a tiny 3D net that maps the local 64^3 state to a large-scale velocity correction.

Input  (B,6,64,64,64) raw state [density, vx, vy, vz, T, fuel] + solid (B,1,64,64,64)
       -> features at 32^3 (2x box average): u/U, density/D, (T-amb)/TS, solid fraction    6 channels
Net    32^3: conv3 6->C, conv3 stride 2 -> 16^3, conv3 dil 2, conv3 dil 2 (16^3), up x2 + skip,
       conv3 2C->C, conv1 C->3                                     (receptive field ~ 40 cells at 64^3)
Output (B,3,32,32,32) correction in m/s per SIM_DT step; the caller upsamples trilinear x2 to 64^3.

`features()` is shared by training (which stores 32^3 features) and runtime, so both see identical inputs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vfx.core.fields3d import AMBIENT_TEMP

U_SCALE = 1.0       # m/s
D_SCALE = 1.0       # density units
T_SCALE = 400.0     # K above ambient
OUT_SCALE = 0.02    # m/s per step: the network predicts correction / OUT_SCALE


def features(data: torch.Tensor, solid: torch.Tensor) -> torch.Tensor:
    """data (B,6,N,N,N) or (6,N,N,N), solid (B,N,N,N) or (N,N,N) -> (B,6,N/2,N/2,N/2) float32."""
    if data.dim() == 4:
        data, solid = data[None], solid[None]
    x = torch.cat([data[:, 1:4] / U_SCALE, data[:, 0:1] / D_SCALE, (data[:, 4:5] - AMBIENT_TEMP) / T_SCALE,
                   solid[:, None]], 1)
    return F.avg_pool3d(x, 2)


class CorrNet(nn.Module):
    def __init__(self, width: int = 16):
        super().__init__()
        C = width
        self.width = C
        self.e1 = nn.Conv3d(6, C, 3, padding=1)
        self.e2 = nn.Conv3d(C, C, 3, stride=2, padding=1)
        self.m1 = nn.Conv3d(C, C, 3, padding=2, dilation=2)
        self.m2 = nn.Conv3d(C, C, 3, padding=2, dilation=2)
        self.d1 = nn.Conv3d(2 * C, C, 3, padding=1)
        self.out = nn.Conv3d(C, 3, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """f: (B,6,32,32,32) features -> (B,3,32,32,32) correction in units of OUT_SCALE m/s."""
        a = F.gelu(self.e1(f))
        b = F.gelu(self.e2(a))
        b = F.gelu(self.m1(b)) + b
        b = F.gelu(self.m2(b)) + b
        u = F.interpolate(b, scale_factor=2, mode="trilinear", align_corners=False)
        c = F.gelu(self.d1(torch.cat([a, u], 1)))
        return self.out(c) * (1.0 - f[:, 5:6])          # zero inside (mostly) solid cells


def augment(f: torch.Tensor, y: torch.Tensor, gen: torch.Generator | None = None):
    """Random horizontal symmetry (gravity is along y): flips of x / z and the x<->z swap, applied to
    features (B,6,...) [ux,uy,uz,d,T,solid] and targets (B,3,...) [ux,uy,uz] (vectors transform too).
    Dims: (B,C,z,y,x)."""
    r = torch.randint(0, 8, (1,), generator=gen).item()
    if r & 1:   # flip x
        f, y = f.flip(-1), y.flip(-1)
        f = f.clone(); y = y.clone()
        f[:, 0] *= -1; y[:, 0] *= -1
    if r & 2:   # flip z
        f, y = f.flip(-3), y.flip(-3)
        f = f.clone(); y = y.clone()
        f[:, 2] *= -1; y[:, 2] *= -1
    if r & 4:   # swap x <-> z
        f = f.transpose(-1, -3)[:, [2, 1, 0, 3, 4, 5]]
        y = y.transpose(-1, -3)[:, [2, 1, 0]]
    return f.contiguous(), y.contiguous()
