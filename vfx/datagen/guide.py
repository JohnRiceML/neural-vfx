"""Guided HR simulation (dataset v2): the 512² run follows an independent 128² run at large scales.

The network's runtime input is an independent 128² sim (`lr_sim`). Unguided, the 512² reference
drifts away from it (chaotic plumes), so (lr_sim -> hr) is not a consistent training pair. Here
the 128² sim runs freely and the HR state's large scales are relaxed toward it every step
(velocity BEFORE the 512² step, when both states hold the same step count; scalars right AFTER
it when scalar_post, the v2 default, so the saved frame sees the correction):

    L(x)  = spectral low-pass exp(-(k/cutoff)^8), k in cycles/domain, on the mirror-extended
            128² grid (no wall jumps), then bilinear upsampling x4
    vel  += k_vel * L(vel_lr - avgpool4(vel_hr))                               (additive)
    s    *= exp(k_scalar * L(log((lp(s_lr)+eps) / (lp(avgpool4(s_hr))+eps))))  (multiplicative)
            for s in {density, temperature - ambient}; plus an optional small additive term
            k_scalar_add * L(s_lr - avgpool4(s_hr)) that can create smoke where HR has none
    fuel += k_fuel * L(fuel_lr - avgpool4(fuel_hr))                           (default 0: off)

The solver step follows the velocity nudge, so its projection removes the divergence the nudge
introduced and every saved velocity is solver output. Why this split (measured, see README): additive scalar
nudging paints smooth haze over the HR and kills 30-50 % of its high-wavenumber density power;
velocity nudging alone keeps the fine detail but density drifts (LR transport is more diffusive);
a multiplicative scalar correction rescales the existing HR structure instead of overwriting it.
Solid cells are never touched; density stays >= 0, temperature >= min(T_old, ambient).
vfx/sim is not modified: the nudge edits `state.data` in place between steps (same torch stream).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vfx.core.fields import (AMBIENT_TEMP, CH, HR_CHANNELS, HR_RES, LR_RES, NUM_CHANNELS, SIM_DT,
                             UPSCALE, Forces, Scenario)

from . import runner


@dataclass(frozen=True)
class GuideConfig:
    # Defaults = the v2 setting, chosen by the sweep documented in README.md.
    k_vel: float = 1.0          # per-step additive relaxation of vel_x, vel_y large scales
    k_scalar: float = 0.7       # per-step multiplicative (log-ratio) relaxation of density, T - ambient
    k_scalar_add: float = 0.0   # per-step additive relaxation of density, T (creates smoke; hazes)
    k_fuel: float = 0.0         # per-step additive relaxation of fuel
    cutoff: float = 32.0        # low-pass cutoff, cycles per domain (LR Nyquist = 64)
    eps_density: float = 0.01   # log-ratio regularizers
    eps_temp: float = 5.0       # K
    scalar_post: bool = True    # apply the scalar/fuel part AFTER the solver step (velocity stays pre-step)

    def asdict(self) -> dict:
        d = dataclasses.asdict(self)
        d["scheme"] = ("pre-step relaxation of HR toward lr_sim, see vfx/datagen/guide.py: "
                       "vel += k_vel*L(dvel); density, T-ambient *= exp(k_scalar*L(log-ratio)) "
                       "[+ k_scalar_add*L(diff)]; fuel += k_fuel*L(dfuel); "
                       "L = exp(-(k/cutoff)^8) mirror-padded spectral low-pass + bilinear x4")
        return d


DEFAULT_GUIDE = GuideConfig()


def _spectral_mask(n: int, cutoff: float, device) -> torch.Tensor:
    """rfft2 mask for the 2n x 2n mirror-extended grid; k in cycles per (original) domain."""
    fy = torch.fft.fftfreq(2 * n, device=device) * n          # 2n samples span 2 domains
    fx = torch.fft.rfftfreq(2 * n, device=device) * n
    k = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    return torch.exp(-(k / cutoff) ** 8)


def _lowpass(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(C,n,n) -> (C,n,n), even (mirror) extension so walls do not ring."""
    n = x.shape[-1]
    y = torch.cat([x, x.flip(-1)], -1)
    y = torch.cat([y, y.flip(-2)], -2)
    y = torch.fft.irfft2(torch.fft.rfft2(y) * mask, s=y.shape[-2:])
    return y[..., :n, :n]


def _up(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x[None], scale_factor=UPSCALE, mode="bilinear", align_corners=False)[0]


class Guide:
    """Applies the large-scale relaxation of an HR FieldState toward an LR FieldState, in place."""

    VEL = [CH["vel_x"], CH["vel_y"]]
    SCAL = [CH["density"], CH["temperature"]]

    def __init__(self, cfg: GuideConfig, hr_solid: torch.Tensor):
        self.cfg = cfg
        dev = hr_solid.device
        self.mask = _spectral_mask(LR_RES, cfg.cutoff, dev)
        self.fluid = (hr_solid < 0.5).float()
        self.offset = torch.tensor([0.0, AMBIENT_TEMP], device=dev).view(2, 1, 1)
        self.eps = torch.tensor([cfg.eps_density, cfg.eps_temp], device=dev).view(2, 1, 1)

    def __call__(self, hr, lr, vel: bool = True, scal: bool = True) -> None:
        c, d = self.cfg, hr.data
        pooled = F.avg_pool2d(d[None], UPSCALE)[0]
        if c.k_vel and vel:
            dv = _lowpass(lr.data[self.VEL] - pooled[self.VEL], self.mask)
            d[self.VEL] += c.k_vel * _up(dv) * self.fluid
        if (c.k_scalar or c.k_scalar_add) and scal:
            s_lr = lr.data[self.SCAL] - self.offset
            s_hr = pooled[self.SCAL] - self.offset
            s = d[self.SCAL] - self.offset                     # (2,H,W) copy
            if c.k_scalar:
                a = _lowpass(s_lr, self.mask).clamp_min(0.0)
                b = _lowpass(s_hr, self.mask).clamp_min(0.0)
                logr = torch.log((a + self.eps) / (b + self.eps)).clamp(-1.0, 1.0)
                s = s * torch.exp(c.k_scalar * _up(logr) * self.fluid)
            if c.k_scalar_add:
                s = s + c.k_scalar_add * _up(_lowpass(s_lr - s_hr, self.mask)) * self.fluid
            t_floor = torch.minimum(d[CH["temperature"]], torch.full_like(s[1], AMBIENT_TEMP))
            d[CH["density"]] = s[0].clamp_min(0.0)
            d[CH["temperature"]] = torch.maximum(s[1] + AMBIENT_TEMP, t_floor)
        if c.k_fuel and scal:
            f = CH["fuel"]
            df = _lowpass((lr.data[f] - pooled[f])[None], self.mask)[0]
            d[f] = (d[f] + c.k_fuel * _up(df[None])[0] * self.fluid).clamp_min(0.0)


def simulate_guided(scenario: Scenario, frames: int, cfg: GuideConfig | None = DEFAULT_GUIDE,
                    observe=None):
    """Run the free 128² sim and the (guided) 512² sim side by side.

    Returns (hr (T,2,512,512) fp16, lr (T,5,128,128) fp16 = avgpool4(HR state),
             lr_sim (T,5,128,128) fp16, solid_hr, solver_version), all on the GPU.
    Frame i of all three is the state after step (num_steps - frames + i + 1).
    cfg=None runs the HR sim unguided (diagnostics). observe(step, hr_state, lr_state) is called
    after every step (diagnostics only).
    """
    hs, version = runner.get_solver(HR_RES)
    ls, _ = runner.get_solver(LR_RES)
    hr = runner.fresh_state(hs, HR_RES, scenario.obstacles)
    lr = runner.fresh_state(ls, LR_RES, scenario.obstacles)
    forces = Forces(emitters=list(scenario.emitters), impulses=[], wind=tuple(scenario.wind))
    start = scenario.num_steps - frames
    if start < 0:
        raise ValueError(f"num_steps {scenario.num_steps} < frames {frames}")
    guide = Guide(cfg, hr.solid) if cfg is not None else None
    hr_idx = [CH[c] for c in HR_CHANNELS]
    o_hr = torch.empty(frames, len(HR_CHANNELS), HR_RES, HR_RES, device="cuda", dtype=torch.float16)
    o_lr = torch.empty(frames, NUM_CHANNELS, LR_RES, LR_RES, device="cuda", dtype=torch.float16)
    o_sim = torch.empty(frames, NUM_CHANNELS, LR_RES, LR_RES, device="cuda", dtype=torch.float16)
    for s in range(scenario.num_steps):
        post = guide is not None and cfg.scalar_post
        if guide is not None:
            guide(hr, lr, scal=not post)       # both hold the state after s steps
        hr = hs.step(hr, forces, SIM_DT)
        lr = ls.step(lr, forces, SIM_DT)
        if post:
            guide(hr, lr, vel=False)           # scalars: saved frame sees the correction directly
        if observe is not None:
            observe(s, hr, lr)
        i = s - start
        if i >= 0:
            o_hr[i] = hr.data[hr_idx]
            o_lr[i] = F.avg_pool2d(hr.data[None], UPSCALE)[0]
            o_sim[i] = lr.data
    return o_hr, o_lr, o_sim, hr.solid, version
