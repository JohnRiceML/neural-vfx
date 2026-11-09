"""Quality metrics for the neural upscaler (owner: test/eval engineer).

Everything here is self-contained on purpose: the warp used by `flicker` and the bicubic
baseline are implemented locally so the metrics never depend on the code they judge
(vfx.runtime.warp_prev / classical_upscale).

Conventions follow vfx.core.fields: index [.., y, x], y = 0 is the bottom row, velocity
(vel_x, vel_y) in m/s, dx in meters, dt in seconds.

Metrics are evaluation-time code, so `.item()` / host syncs are fine here (not the frame loop).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from vfx.core.fields import (
    CH,
    SIM_DT,
    UPSCALE,
    FieldMeta,
    FieldState,
    HRFields,
)

__all__ = [
    "radial_power_spectrum",
    "spectrum_error",
    "l1",
    "semi_lagrangian_warp",
    "flicker",
    "flicker_step",
    "grad_energy",
    "bicubic_upscale",
    "scenario_dirs",
    "load_scenario",
    "lr_frame_state",
    "is_complete_scenario",
    "evaluate_upscaler",
    "EvalResult",
]


def _as_tensor(x) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(x))
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor or ndarray, got {type(x)}")
    return x.float()


# ----------------------------------------------------------------------------- spectrum
def radial_power_spectrum(x) -> torch.Tensor:
    """Radially summed power spectrum E(k) of a square field.

    x: (H, W) or (N, H, W) with H == W. For N > 1 the spectra are averaged over N.
    Returns a 1-D tensor of length H // 2 + 1, index k = integer radial wavenumber
    (cycles per domain). E(k) = sum over FFT modes with round(|(kx, ky)|) == k of
    |FFT(x)|^2 / (H * W)^2. E(0) is the mean^2 (DC) and is NOT used by spectrum_error.
    Modes with radius > H // 2 (the square's corners) are dropped.
    """
    x = _as_tensor(x)
    if x.dim() == 2:
        x = x[None]
    if x.dim() != 3 or x.shape[-1] != x.shape[-2]:
        raise ValueError(f"expected (H,W) or (N,H,W) square field, got {tuple(x.shape)}")
    n = x.shape[-1]
    power = torch.fft.fft2(x.double()).abs().pow(2).mean(0) / float(n * n) ** 2   # (H, W) fp64
    f = torch.fft.fftfreq(n, d=1.0 / n, device=x.device)                      # integer freqs
    ky, kx = torch.meshgrid(f, f, indexing="ij")
    kbin = torch.sqrt(kx * kx + ky * ky).round().long()
    kmax = n // 2
    keep = kbin <= kmax
    spec = torch.zeros(kmax + 1, device=x.device, dtype=power.dtype)
    spec.scatter_add_(0, kbin[keep], power[keep])
    return spec


def spectrum_error(pred, ref, rel_floor: float = 1e-10) -> float:
    """Log radial power-spectrum error of DENSITY, in decades (log10 units).

    pred, ref: density, (H, W) or (N, H, W) (N frames are averaged in spectral space first),
    same shape, square.

        E_p(k), E_r(k) = radial_power_spectrum(pred), radial_power_spectrum(ref)
        floor          = rel_floor * sum_{k>=1} E_r(k) + 1e-30
        error          = mean_{k=1..H//2} | log10(E_p(k) + floor) - log10(E_r(k) + floor) |

    DC (k = 0) is excluded: total mass is judged by l1, this metric judges detail. Every
    wavenumber shell has equal weight, so high frequencies (most of the shells) dominate:
    an over-smoothed upscale scores badly, injected noise also scores badly.
    0.0 means identical spectra; 1.0 means energy off by 10x on average.
    The floor keeps numerically empty shells (e.g. an all-zero frame) from producing
    arbitrarily large values.
    """
    p = radial_power_spectrum(pred)
    r = radial_power_spectrum(ref)
    if p.shape != r.shape:
        raise ValueError(f"shape mismatch {tuple(p.shape)} vs {tuple(r.shape)}")
    floor = rel_floor * r[1:].sum() + 1e-30
    d = (torch.log10(p[1:] + floor) - torch.log10(r[1:] + floor)).abs()
    return float(d.mean().item())


def l1(pred, ref) -> float:
    """Mean absolute error over all elements (same shape)."""
    p, r = _as_tensor(pred), _as_tensor(ref).to(_as_tensor(pred).device)
    if p.shape != r.shape:
        raise ValueError(f"shape mismatch {tuple(p.shape)} vs {tuple(r.shape)}")
    return float((p - r).abs().mean().item())


# ----------------------------------------------------------------------------- flicker
def semi_lagrangian_warp(x: torch.Tensor, velocity: torch.Tensor, dx: float, dt: float) -> torch.Tensor:
    """Advect x by `velocity` for one step dt (first-order backtrace + bilinear sample).

    x: (H, W) or (C, H, W). velocity: (2, h, w) m/s (vel_x, vel_y); bilinearly resampled to
    (H, W) if needed (m/s values are resolution independent). dx: cell size of x in meters.
    out[y, x] = x(y - dt*vy/dx, x - dt*vx/dx), clamped to the domain border.
    """
    x = _as_tensor(x)
    velocity = _as_tensor(velocity).to(x.device)
    squeeze = x.dim() == 2
    if squeeze:
        x = x[None]
    C, H, W = x.shape
    if velocity.shape[-2:] != (H, W):
        velocity = F.interpolate(velocity[None], size=(H, W), mode="bilinear", align_corners=False)[0]
    iy = torch.arange(H, device=x.device, dtype=torch.float32)
    ix = torch.arange(W, device=x.device, dtype=torch.float32)
    gy, gx = torch.meshgrid(iy, ix, indexing="ij")
    src_x = gx - dt * velocity[0] / dx
    src_y = gy - dt * velocity[1] / dx
    grid = torch.stack([2.0 * src_x / (W - 1) - 1.0, 2.0 * src_y / (H - 1) - 1.0], dim=-1)[None]
    out = F.grid_sample(x[None], grid, mode="bilinear", padding_mode="border", align_corners=True)[0]
    return out[0] if squeeze else out


def flicker_step(prev: torch.Tensor, cur: torch.Tensor, prev_velocity: torch.Tensor, dx: float, dt: float) -> float:
    """One term of `flicker`: mean | cur - warp(prev, prev_velocity) | (for streaming use)."""
    return float((_as_tensor(cur) - semi_lagrangian_warp(prev, prev_velocity, dx, dt)).abs().mean().item())


def grad_energy(seq) -> float:
    """Mean absolute forward difference (x and y averaged) of a (.., H, W) field, per cell.
    Detail measure used only for the diagnostic `flicker_ratio_detail_norm`."""
    s = _as_tensor(seq)
    gx = (s[..., :, 1:] - s[..., :, :-1]).abs().mean()
    gy = (s[..., 1:, :] - s[..., :-1, :]).abs().mean()
    return float(((gx + gy) / 2).item())


def flicker(seq, velocities, dx: float, dt: float) -> float:
    """Warped-frame flicker: mean_t mean_pixels | x_t - warp(x_{t-1}, v_{t-1}) |, t = 1..T-1.

    seq: (T, H, W) a scalar sequence (usually HR density). velocities: (T, 2, h, w) m/s, the
    flow at each frame (LR velocity is fine; it is resampled). Frame t-1 -> t is advected by
    the velocity of frame t-1 (vfx.sim advects first, with the start-of-step velocity, which
    is the velocity stored with frame t-1). dx: cell size of `seq` in meters, dt: seconds
    between frames.

    `velocities` MUST be the flow that actually transports `seq`, i.e. the velocity of the
    sequence that drove it (for an upscaler output: the velocity of the LR input it was fed).
    Warping a sequence with another run's velocity measures the divergence between the two
    runs, not temporal instability (see evaluate_upscaler and tests/README.md).

    Sources and diffusion make this non-zero even for the reference, so judge a prediction
    by the RATIO flicker(pred, v_pred_driver) / flicker(reference, v_reference) (v1 gate: <= 1.2).
    Response to real flicker: per-frame iid noise of std 0.2% of the frame max roughly doubles
    the value on dataset v1 HR density (tests/README.md).
    """
    seq = _as_tensor(seq)
    velocities = _as_tensor(velocities).to(seq.device)
    if seq.dim() != 3 or velocities.dim() != 4 or velocities.shape[1] != 2:
        raise ValueError(f"bad shapes seq {tuple(seq.shape)} vel {tuple(velocities.shape)}")
    if seq.shape[0] != velocities.shape[0]:
        raise ValueError("seq and velocities need the same T")
    T = seq.shape[0]
    if T < 2:
        raise ValueError("need at least 2 frames")
    total = 0.0
    for t in range(1, T):
        total += flicker_step(seq[t - 1], seq[t], velocities[t - 1], dx, dt)
    return total / (T - 1)


# ----------------------------------------------------------------------------- baseline
def bicubic_upscale(lr_state: FieldState, scale: int = UPSCALE) -> HRFields:
    """Independent classical baseline: bicubic (2D 'tricubic') upsampling of density, temperature.

    Density is clamped >= 0 (bicubic overshoots). Used as the reference baseline for the
    Layer 4 gates so the gate does not depend on vfx.runtime.classical_upscale.
    """
    d = lr_state.data[[CH["density"], CH["temperature"]]].float()[None]
    up = F.interpolate(d, scale_factor=scale, mode="bicubic", align_corners=False)[0]
    return HRFields(density=up[0].clamp_min(0.0), temperature=up[1])


# ----------------------------------------------------------------------------- dataset access
def scenario_dirs(version: str, split: str) -> list[Path]:
    """Scenario directories of a split, in splits.json order."""
    from vfx.core.paths import dataset_dir

    root = dataset_dir(version)
    splits = json.loads((root / "splits.json").read_text())
    return [root / split / f"scn_{int(s):06d}" for s in splits[split]]


def load_scenario(d: Path, mmap: bool = True) -> dict:
    """Raw on-disk arrays (numpy, memory-mapped) + meta for one scenario dir."""
    mode = "r" if mmap else None
    out = {"dir": Path(d), "meta": json.loads((Path(d) / "meta.json").read_text())}
    for name in ("lr", "hr", "solid_lr", "solid_hr", "lr_sim"):
        p = Path(d) / f"{name}.npy"
        out[name] = np.load(p, mmap_mode=mode) if p.exists() else None
    return out


def lr_frame_state(frame: np.ndarray, solid: np.ndarray, dt: float = SIM_DT, step: int = 0,
                   device: str = "cuda") -> FieldState:
    """Build a FieldState from one stored LR frame (5, h, w) float16."""
    data = torch.from_numpy(np.asarray(frame, dtype=np.float32)).to(device)
    sol = torch.from_numpy(np.asarray(solid, dtype=np.float32)).to(device)
    meta = FieldMeta.for_res(data.shape[-1], dt)
    meta.step = step
    return FieldState(data=data.contiguous(), solid=sol, meta=meta)


@dataclass
class EvalResult:
    """Scores of one upscaler on one (split, source). Flicker keys (see evaluate_upscaler):

    flicker / flicker_ref / flicker_ratio   AUDITED (gated): pred warped with the velocity of the
                                            sequence that drove it (`source`); reference = HR warped
                                            with its own (pooled-HR) velocity.
    flicker_legacy_hrvel / flicker_ratio_legacy_hrvel
                                            PRE-AUDIT definition (vfx/train/EXPERIMENTS.md entries up
                                            to 2026-11-09 12:08): pred always warped with the pooled-HR
                                            velocity. Identical to the audited value for source="lr";
                                            for "lr_sim" on v1 (independent run) it mostly measures how
                                            far the LR sim's flow diverged from the HR run. Not gated.
    flicker_ref_srcvel                      diagnostic: HR warped with the `source` velocity (meaningful
                                            only when source follows HR, e.g. v2 lr_sim).
    flicker_ratio_detail_norm               diagnostic: (flicker / grad_energy) of pred over the same
                                            for HR. Saturates under real noise (noise also adds gradient
                                            energy), so it is NOT a gate.
    """
    split: str
    source: str
    n_scenarios: int = 0
    n_frames: int = 0
    n_skipped_incomplete: int = 0
    spectrum_error: float = float("nan")      # decades, frame-weighted mean over scenarios
    l1_density: float = float("nan")
    l1_temperature: float = float("nan")      # kelvin
    flicker: float = float("nan")
    flicker_ref: float = float("nan")
    flicker_ratio: float = float("nan")       # mean of per-scenario flicker / flicker_ref
    flicker_legacy_hrvel: float = float("nan")
    flicker_ratio_legacy_hrvel: float = float("nan")
    flicker_ref_srcvel: float = float("nan")
    flicker_ratio_detail_norm: float = float("nan")
    per_scenario: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def is_complete_scenario(d: Path, need: tuple[str, ...] = ("lr", "hr", "solid_lr")) -> bool:
    """meta.json is written last by datagen, so it marks a finished scenario directory."""
    d = Path(d)
    return (d / "meta.json").exists() and all((d / f"{n}.npy").exists() for n in need)


def evaluate_upscaler(upscaler, version: str, split: str = "test", source: str = "lr",
                      max_scenarios: int | None = None, max_frames: int | None = None,
                      device: str = "cuda") -> EvalResult:
    """Run an upscaler over a dataset split and score it against the stored HR reference.

    upscaler: object with .infer(FieldState) -> HRFields (and optional .reset(), called once per
              scenario so temporal state does not leak), or a plain callable FieldState -> HRFields.
    source:   "lr"     = 4x average-pooled HR (the training input distribution), or
              "lr_sim" = the 128^2 sim (the real runtime situation). In v1 it is an independent
              run of the same scenario; in v2 the HR run is guided to follow it (matched pair).
    All stored frames are used (8 in v1, 16 in v2) unless max_frames is given. Scenario dirs
    that are not finished (no meta.json yet) are skipped and counted in n_skipped_incomplete.

    Flicker (audited 2026-11-09): each sequence is warped with the velocity of the flow that
    actually transports it:
        flicker       = flicker(pred density, velocity of `source`)   # the flow the upscaler followed
        flicker_ref   = flicker(HR density,   velocity of "lr")       # pooled HR = HR's own flow
        flicker_ratio = mean over scenarios of flicker / flicker_ref  (v1 gate <= 1.2)
    Both velocities are 128^2, so pred and reference are warped with equally coarse velocity.
    For source="lr" this is identical to the pre-audit definition. See EvalResult for the
    legacy and diagnostic keys.
    """
    infer: Callable = getattr(upscaler, "infer", upscaler)
    reset = getattr(upscaler, "reset", None)
    res = EvalResult(split=split, source=source)
    dirs = scenario_dirs(version, split)
    finished = [d for d in dirs if is_complete_scenario(d)]
    res.n_skipped_incomplete = len(dirs) - len(finished)
    dirs = finished
    if max_scenarios is not None:
        dirs = dirs[:max_scenarios]
    vsl = slice(CH["vel_x"], CH["vel_y"] + 1)
    acc = {k: 0.0 for k in ("spec", "l1d", "l1t", "fl", "flr", "fll", "flrs")}
    ratios, ratios_legacy, ratios_norm = [], [], []
    for d in dirs:
        scn = load_scenario(d)
        src = scn[source]
        if src is None:
            raise FileNotFoundError(f"{d} has no {source}.npy")
        meta = scn["meta"]
        dt = float(meta.get("dt", SIM_DT))
        T = src.shape[0] if max_frames is None else min(src.shape[0], max_frames)
        if reset is not None:
            reset()
        pd, pt = [], []
        with torch.no_grad():
            for t in range(T):
                st = lr_frame_state(src[t], scn["solid_lr"], dt, int(meta.get("start_step", 0)) + t, device)
                out = infer(st)
                pd.append(out.density.to(torch.float32, copy=True))   # infer may reuse buffers
                pt.append(out.temperature.to(torch.float32, copy=True))
        pd = torch.stack(pd)
        pt = torch.stack(pt)
        hr = torch.from_numpy(np.asarray(scn["hr"][:T], dtype=np.float32)).to(device)
        vel_hr = torch.from_numpy(np.asarray(scn["lr"][:T, vsl], dtype=np.float32)).to(device)
        vel_src = torch.from_numpy(np.asarray(src[:T, vsl], dtype=np.float32)).to(device)
        dx_hr = 1.0 / hr.shape[-1]
        s = spectrum_error(pd, hr[:, 0])
        ld = l1(pd, hr[:, 0])
        lt = l1(pt, hr[:, 1])
        nan = float("nan")
        fl = flicker(pd, vel_src, dx_hr, dt) if T > 1 else nan          # audited
        flr = flicker(hr[:, 0], vel_hr, dx_hr, dt) if T > 1 else nan     # reference, own flow
        fll = flicker(pd, vel_hr, dx_hr, dt) if T > 1 else nan           # legacy
        flrs = flicker(hr[:, 0], vel_src, dx_hr, dt) if T > 1 else nan   # diagnostic
        ratio = fl / max(flr, 1e-12)
        ratio_legacy = fll / max(flr, 1e-12)
        g_p, g_r = max(grad_energy(pd), 1e-12), max(grad_energy(hr[:, 0]), 1e-12)
        ratio_norm = (fl / g_p) / max(flr / g_r, 1e-12)
        res.per_scenario.append(dict(seed=meta.get("scenario", {}).get("seed"), dir=d.name, frames=T,
                                     spectrum_error=s, l1_density=ld, l1_temperature=lt,
                                     flicker=fl, flicker_ref=flr, flicker_ratio=ratio,
                                     flicker_legacy_hrvel=fll, flicker_ratio_legacy_hrvel=ratio_legacy,
                                     flicker_ref_srcvel=flrs, flicker_ratio_detail_norm=ratio_norm,
                                     tags=meta.get("tags", [])))
        res.n_scenarios += 1
        res.n_frames += T
        for k, v in (("spec", s), ("l1d", ld), ("l1t", lt), ("fl", fl), ("flr", flr), ("fll", fll), ("flrs", flrs)):
            acc[k] += v * T
        ratios.append(ratio)
        ratios_legacy.append(ratio_legacy)
        ratios_norm.append(ratio_norm)
        del pd, pt, hr, vel_hr, vel_src
    if res.n_frames:
        n = res.n_frames
        res.spectrum_error, res.l1_density, res.l1_temperature = acc["spec"] / n, acc["l1d"] / n, acc["l1t"] / n
        res.flicker, res.flicker_ref = acc["fl"] / n, acc["flr"] / n
        res.flicker_legacy_hrvel, res.flicker_ref_srcvel = acc["fll"] / n, acc["flrs"] / n
        res.flicker_ratio = float(np.mean(ratios))
        res.flicker_ratio_legacy_hrvel = float(np.mean(ratios_legacy))
        res.flicker_ratio_detail_norm = float(np.mean(ratios_norm))
    return res
