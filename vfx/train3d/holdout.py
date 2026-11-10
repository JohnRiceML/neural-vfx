"""Model selection on held-out TRAIN scenarios (never test/ood), whole 16-frame sequences, eval-metric definitions.

`HoldoutSet(windows)` keeps the hold-out GPUWindows; `evaluate(model)` runs the net over every frame as the runtime
does at process_scale 1.0 (out = uint8(clamp(lr + net))) and returns pooled numbers, each also relative to lr:
  l1, spectrum_error, ms_dssim             vfx.eval3d.metrics on the dilated lr/hr-alpha union (same as eval)
  jerk_ratio                               metrics.temporal_jerk(out) / temporal_jerk(hr)  (unwarped, cheap)
  midband                                  log10 band power 0.04-0.2 cyc/px (5-25 px structure), out / hr (0 = match;
                                           lr is negative). Own diagnostic: the eval spectrum is dominated by the top
                                           (dither) bins, see EXPERIMENTS.md "detail oracle".
  chroma_speckle                           isolated colour outliers per frame (the purple dots)
  speckle                                  isolated-pixel count: |Y - median3x3(Y)| > SPECKLE_T inside the effect
                                           region, per 256^2 frame; also for hr and lr (the reference's own count)
`select_ok(r, rule)` / `select_key(r, rule)` implement the pre-registered selection rule (EXPERIMENTS.md).
"""
from __future__ import annotations

import numpy as np
import torch

from vfx.train3d import losses as L

SPECKLE_T = 0.08


MIDBAND = (0.04, 0.2)       # cycles/px: 5-25 px structures (filaments, billows), below the dither-dominated top bins


def chroma_speckle_count(x, mask, t: float = 0.06):
    """Isolated colour outliers (the purple dots): |c - median3x3(c)| > t for chroma c = rgb - luma, any channel."""
    c = x.float() - L.luma(x.float())
    s = ((c - L.med3(c)).abs().amax(1)) > t
    return float((s & mask).sum()) / x.shape[0]


def speckle_count(x, mask):
    """x (N,3,H,W) 0..1, mask (N,H,W) bool -> isolated luminance outliers per frame (float)."""
    y = L.luma(x.float())
    s = (y - L.med3(y)).abs()[:, 0] > SPECKLE_T
    return float((s & mask).sum()) / x.shape[0]


class HoldoutSet:
    def __init__(self, windows, max_sequences: int | None = None):
        from vfx.eval3d import metrics as M
        self.M = M
        self.w = windows
        self.idx = list(range(windows.N if max_sequences is None else min(max_sequences, windows.N)))
        self._ref = None

    @torch.no_grad()
    def _frames(self, model, n):
        lr = self.w.lr[n].float() / 255.0
        if model is None:
            return self.w.lr[n]
        aux = self.w.aux[n].float()
        with torch.autocast("cuda", dtype=torch.float16):
            d = model(lr, aux if model.cfg.use_aux else None, None).float()
        return ((lr + d).clamp(0, 1) * 255.0 + 0.5).to(torch.uint8)

    @torch.no_grad()
    def _score(self, out_fn):
        M = self.M
        agg = {"l1": [], "spectrum_error": [], "ms_dssim": [], "jerk": [], "jerk_hr": [], "speckle": [], "midband": [],
               "chroma_speckle": []}
        for n in self.idx:
            lr, hr, aux = self.w.lr[n], self.w.hr[n], self.w.aux[n]
            a = torch.maximum(aux[:, 0].float().clamp(0, 1), self.w.hr_alpha[n].float() / 255.0)
            mask = M.effect_mask(a)
            out = out_fn(n)
            agg["l1"].append(M.l1(out, hr, mask))
            agg["spectrum_error"].append(M.spectrum_error(out, hr, mask))
            agg["ms_dssim"].append(M.ms_dssim(out, hr, mask))
            agg["jerk"].append(M.temporal_jerk(out, mask))
            agg["jerk_hr"].append(M.temporal_jerk(hr, mask))
            o = out.float() / 255.0
            agg["speckle"].append(speckle_count(o, mask))
            agg["chroma_speckle"].append(chroma_speckle_count(o, mask))
            agg["midband"].append(float(np.log10(max(M.band_power(out, mask, *MIDBAND), 1e-12)
                                                 / max(M.band_power(hr, mask, *MIDBAND), 1e-12))))
        r = {k: float(np.mean(v)) for k, v in agg.items() if k not in ("jerk", "jerk_hr")}
        r["jerk_ratio"] = float(np.sum(agg["jerk"]) / max(np.sum(agg["jerk_hr"]), 1e-12))
        return r

    def reference(self):
        """lr and hr numbers on the same sequences (computed once)."""
        if self._ref is None:
            lr = self._score(lambda n: self.w.lr[n])
            hr_sp = self._score(lambda n: self.w.hr[n])
            self._ref = {"lr": lr, "hr_speckle": hr_sp["speckle"], "hr_chroma_speckle": hr_sp["chroma_speckle"]}
        return self._ref

    def evaluate(self, model):
        was = model.training
        model.eval()
        r = self._score(lambda n: self._frames(model, n))
        model.train(was)
        ref = self.reference()
        for k in ("l1", "spectrum_error", "ms_dssim"):
            r[k + "_rel"] = r[k] / ref["lr"][k]
        r["speckle_rel_hr"] = r["speckle"] / max(ref["hr_speckle"], 1.0)
        r["chroma_speckle_rel_hr"] = r["chroma_speckle"] / max(ref["hr_chroma_speckle"], 1.0)
        return r


def select_ok(r, rule: dict) -> bool:
    return (r["l1_rel"] <= rule.get("max_l1_rel", 1.0) and r["ms_dssim_rel"] <= rule.get("max_ms_dssim_rel", 1.0)
            and r["jerk_ratio"] <= rule.get("max_jerk_ratio", 1.1)
            and r["speckle_rel_hr"] <= rule.get("max_speckle_rel_hr", 1.5)
            and r.get("chroma_speckle_rel_hr", 0.0) <= rule.get("max_speckle_rel_hr", 1.5))


def select_key(r, rule: dict) -> float:
    if rule.get("key") == "midband":       # |log10 mid-band power / hr's|: 5-25 px structure closest to the reference
        return abs(r["midband"])
    return r["spectrum_error"]
