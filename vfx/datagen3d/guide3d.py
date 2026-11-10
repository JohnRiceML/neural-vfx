"""Guided HR 3D simulation: a 256^3 run follows an independent 64^3 run at large scales.

Port of vfx/datagen/guide.py (2D dataset v2) to 3D. The 64^3 sim is what the game runs (runtime
config, sandbox dissipation), so route B's input is a render of it; unguided, a 256^3 run of the
same scene drifts away (chaotic plumes; the 64^3 plume also rises more slowly), so a render of it
is not a consistent target. Every step:

    L(x)  = spectral low-pass exp(-(k/cutoff)^8), k in cycles/domain, on the mirror-extended 64^3
            grid (no wall jumps), then trilinear upsampling x4
    vel  += k_vel * L(vel_lr - avgpool4(vel_hr))                     BEFORE the HR step (additive;
            the HR projection then removes the divergence the nudge introduced)
    s    *= exp(k_scalar * L(clamp(log((lp(s_lr)+eps) / (lp(avgpool4(s_hr))+eps)), -1, 1)))
            AFTER the HR step, for s in {density, temperature - ambient} (multiplicative: rescales
            the existing HR structure, never paints smooth haze; additive lost 25-60 % of the
            fine-scale power in 2D)
    fuel += k_fuel * L(fuel_lr - avgpool4(fuel_hr))                  (default 0: off)

Solid cells are never touched; density stays >= 0, temperature >= min(T_old, ambient).
vfx/sim3d is not modified: the nudge edits `state.data` in place between steps (same torch stream).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vfx.core.fields3d import AMBIENT_TEMP, CH3


@dataclass(frozen=True)
class GuideConfig3D:
    k_vel: float = 1.0          # per-step additive relaxation of the velocity large scales
    k_scalar: float = 0.7       # per-step multiplicative (log-ratio) relaxation of density, T - ambient
    k_fuel: float = 0.0         # per-step additive relaxation of fuel
    cutoff: float = 16.0        # low-pass cutoff, cycles per domain (64^3 Nyquist = 32); swept 8/12/16
    eps_density: float = 0.01   # log-ratio regularizers
    eps_temp: float = 5.0       # K
    log_clamp: float = 1.0      # |log ratio| cap per step

    def asdict(self) -> dict:
        d = dataclasses.asdict(self)
        d["scheme"] = ("vfx/datagen3d/guide3d.py: before the HR step vel += k_vel*L(vel_lr - pool4(vel_hr)); "
                       "after it density, T-ambient *= exp(k_scalar*L(clamp(log-ratio))); fuel += k_fuel*L(dfuel); "
                       "L = exp(-(k/cutoff)^8) mirror-padded spectral low-pass on 64^3 + trilinear x4")
        return d


DEFAULT_GUIDE = GuideConfig3D()


def spectral_mask(n: int, cutoff: float, device) -> torch.Tensor:
    """rfftn mask for the (2n)^3 mirror-extended grid; k in cycles per (original) domain."""
    f = torch.fft.fftfreq(2 * n, device=device) * n
    fr = torch.fft.rfftfreq(2 * n, device=device) * n
    k = torch.sqrt(f[:, None, None] ** 2 + f[None, :, None] ** 2 + fr[None, None, :] ** 2)
    return torch.exp(-(k / cutoff) ** 8)


def lowpass(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(C,n,n,n) -> (C,n,n,n), even (mirror) extension so walls do not ring."""
    n = x.shape[-1]
    y = torch.cat([x, x.flip(-1)], -1)
    y = torch.cat([y, y.flip(-2)], -2)
    y = torch.cat([y, y.flip(-3)], -3)
    y = torch.fft.irfftn(torch.fft.rfftn(y, dim=(-3, -2, -1)) * mask, s=y.shape[-3:], dim=(-3, -2, -1))
    return y[..., :n, :n, :n]


def up(x: torch.Tensor, factor: int) -> torch.Tensor:
    return F.interpolate(x[None], scale_factor=factor, mode="trilinear", align_corners=False)[0]


class Guide3D:
    """Applies the large-scale relaxation of an HR FieldState3D toward an LR one, in place."""

    VEL = [CH3["vel_x"], CH3["vel_y"], CH3["vel_z"]]
    SCAL = [CH3["density"], CH3["temperature"]]

    def __init__(self, cfg: GuideConfig3D, hr_solid: torch.Tensor, lr_res: int):
        self.cfg = cfg
        dev = hr_solid.device
        self.factor = hr_solid.shape[-1] // lr_res
        self.mask = spectral_mask(lr_res, cfg.cutoff, dev)
        self.fluid = (hr_solid < 0.5).float()
        self.offset = torch.tensor([0.0, AMBIENT_TEMP], device=dev).view(2, 1, 1, 1)
        self.eps = torch.tensor([cfg.eps_density, cfg.eps_temp], device=dev).view(2, 1, 1, 1)

    def _pool(self, x):
        return F.avg_pool3d(x[None], self.factor)[0]

    @torch.no_grad()
    def velocity(self, hr, lr) -> None:
        c, d = self.cfg, hr.data
        if not c.k_vel:
            return
        dv = lowpass(lr.data[self.VEL] - self._pool(d[self.VEL]), self.mask)
        d[self.VEL] += c.k_vel * up(dv, self.factor) * self.fluid

    @torch.no_grad()
    def scalars(self, hr, lr) -> None:
        c, d = self.cfg, hr.data
        if c.k_scalar:
            s_lr = lr.data[self.SCAL] - self.offset
            s_hr = self._pool(d[self.SCAL]) - self.offset
            a = lowpass(s_lr, self.mask).clamp_min(0.0)
            b = lowpass(s_hr, self.mask).clamp_min(0.0)
            logr = torch.log((a + self.eps) / (b + self.eps)).clamp(-c.log_clamp, c.log_clamp)
            g = torch.exp(c.k_scalar * up(logr, self.factor) * self.fluid)
            t = d[CH3["temperature"]]
            t_floor = torch.clamp_max(t, AMBIENT_TEMP)
            d[CH3["density"]] = (d[CH3["density"]] * g[0]).clamp_min(0.0)
            d[CH3["temperature"]] = torch.maximum((t - AMBIENT_TEMP) * g[1] + AMBIENT_TEMP, t_floor)
        if c.k_fuel:
            f = CH3["fuel"]
            df = lowpass((lr.data[f] - self._pool(d[f]))[None], self.mask)
            d[f] = (d[f] + c.k_fuel * up(df, self.factor)[0] * self.fluid).clamp_min(0.0)


def tracking_error(hr_data: torch.Tensor, lr_data: torch.Tensor, res: int = 16) -> torch.Tensor:
    """Relative L2 at res^3 (avg-pooled) of HR vs LR: (density, T - ambient, velocity), on GPU (3,)."""
    a = F.adaptive_avg_pool3d(hr_data[None], res)[0]
    b = F.adaptive_avg_pool3d(lr_data[None], res)[0]
    T = CH3["temperature"]
    a[T] -= AMBIENT_TEMP
    b[T] -= AMBIENT_TEMP

    def rel(idx):
        return (a[idx] - b[idx]).norm() / (b[idx].norm() + 1e-8)
    return torch.stack([rel([CH3["density"]]), rel([T]), rel([CH3["vel_x"], CH3["vel_y"], CH3["vel_z"]])])
