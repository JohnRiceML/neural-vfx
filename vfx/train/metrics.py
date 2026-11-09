"""Quick metrics for training-time validation. The gate metrics are vfx.eval.metrics (test engineer);
these are used if that module is not available yet, and are reported under their own names."""
from __future__ import annotations

import torch

from vfx.train.losses import radial_power


def spectrum_error(pred: torch.Tensor, ref: torch.Tensor, eps_rel: float = 1e-6) -> float:
    """Mean |log10 P_pred - log10 P_ref| over radial bins of the density field. (H,W) or (B,H,W)."""
    p = pred.float().reshape(-1, 1, *pred.shape[-2:])
    r = ref.float().reshape(-1, 1, *ref.shape[-2:])
    Pp, Pr = radial_power(p), radial_power(r)
    floor = Pr.mean(dim=-1, keepdim=True) * eps_rel + 1e-12
    return (torch.log10(Pp + floor) - torch.log10(Pr + floor)).abs().mean().item()


def l1(pred: torch.Tensor, ref: torch.Tensor) -> float:
    return (pred.float() - ref.float()).abs().mean().item()


def get_metrics():
    """(spectrum_error, l1, source) — prefers the gate implementations in vfx.eval.metrics."""
    try:
        from vfx.eval import metrics as m
        return m.spectrum_error, m.l1, "vfx.eval.metrics"
    except Exception:
        return spectrum_error, l1, "vfx.train.metrics"
