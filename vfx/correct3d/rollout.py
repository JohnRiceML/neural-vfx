"""One instrumented rollout of a Scene at any resolution, optionally with a velocity corrector.

Returns time series of `common.diagnostics`, the mean KE spectrum at 64^3 (restricted) over the
second half, time-mean restricted density / velocity (64^3) over the second half, and front-view
density MIPs at chosen steps. One host sync at the end (GPU lists), as the frame-loop rule asks.
"""
from __future__ import annotations

import numpy as np
import torch

from vfx.core.fields3d import SIM_DT

from .common import VEL, diagnostics, get_solver, ke_spectrum, mip, restrict


@torch.no_grad()
def rollout(scene, res: int, steps: int = 300, cfg_kw: dict | None = None, corrector=None,
            mip_steps=(60, 120, 180, 240, 299), spec_every: int = 5, mean_from: int | None = None,
            tag: str = "", events=None, div_every: int = 0) -> dict:
    """corrector: object with .apply(state, dt) called BEFORE every solver step (or None).
    events: optional callable(step, state, solver) -> Forces3D | None to override forces (perturbations)."""
    kw = dict(scene.sim)
    kw.update(cfg_kw or {})
    s = get_solver(res, kw, tag)
    st = s.reset(scene.obstacles)
    if corrector is not None and hasattr(corrector, "reset"):
        corrector.reset()
    mean_from = steps // 2 if mean_from is None else mean_from
    diags, specs, mips, divs, cstats = [], [], {}, [], []
    mean_d = torch.zeros(64, 64, 64, device=st.data.device)
    mean_u = torch.zeros(3, 64, 64, 64, device=st.data.device)
    nmean = 0
    for t in range(steps):
        f = events(t, st, s) if events is not None else None
        if f is None:
            f = scene.forces(t)
        if corrector is not None:
            corrector.apply(st, SIM_DT)
            if hasattr(corrector, "last_stats"):
                cstats.append(corrector.last_stats)
        s.step(st, f, SIM_DT)
        diags.append(diagnostics(st.data))
        if t >= mean_from:
            r = restrict(st.data, 64)
            mean_d += r[0]
            mean_u += r[VEL]
            nmean += 1
            if (t - mean_from) % spec_every == 0:
                specs.append(ke_spectrum(r[VEL]))
        if t in mip_steps:
            mips[t] = mip(st.data)
        if div_every and (t % div_every == 0 or t == steps - 1):
            divs.append(s.divergence(st).max())
    out = {
        "diag": torch.stack(diags).cpu().numpy(),
        "spec": torch.stack(specs).mean(0).cpu().numpy() if specs else None,
        "mean_d": (mean_d / max(nmean, 1)),
        "mean_u": (mean_u / max(nmean, 1)),
        "mips": {k: v.cpu().numpy() for k, v in mips.items()},
        "final_finite": bool(torch.isfinite(st.data).all().item()),
        "max_speed": float(st.data[VEL].norm(dim=0).max().item()),
    }
    if divs:
        out["div"] = torch.stack(divs).cpu().numpy()
    if cstats:
        out["cstats"] = torch.stack(cstats).cpu().numpy()
    return out


def summarize(a: dict, ref: dict, t0: int = 30) -> dict:
    """Scalar comparison of rollout `a` against reference rollout `ref` (both from rollout())."""
    da, dr = a["diag"][t0:], ref["diag"][t0:]
    cy_err = float(np.abs(da[:, 1] - dr[:, 1]).mean())
    h95_err = float(np.abs(da[:, 2] - dr[:, 2]).mean())
    hy_err = float(np.abs(da[:, 4] - dr[:, 4]).mean())
    ke_ratio = float(da[:, 5].mean() / max(dr[:, 5].mean(), 1e-12))
    mass_ratio = float(da[:, 0].mean() / max(dr[:, 0].mean(), 1e-12))
    vy_ratio = float(da[:, 6].mean() / max(dr[:, 6].mean(), 1e-12))
    md, mr = a["mean_d"], ref["mean_d"]
    pool = lambda x: torch.nn.functional.avg_pool3d(x[None, None], 4)[0, 0]
    d_err16 = float(((pool(md) - pool(mr)).norm() / (pool(mr).norm() + 1e-8)).item())
    ua, ur = a["mean_u"], ref["mean_u"]
    poolu = lambda x: torch.nn.functional.avg_pool3d(x[None], 4)[0]
    u_err16 = float(((poolu(ua) - poolu(ur)).norm() / (poolu(ur).norm() + 1e-8)).item())
    out = dict(cy_err=cy_err, h95_err=h95_err, hy_err=hy_err, ke_ratio=ke_ratio, mass_ratio=mass_ratio,
               vy_ratio=vy_ratio, meanD_err16=d_err16, meanU_err16=u_err16,
               cy_end=float(da[-30:, 1].mean()), cy_end_ref=float(dr[-30:, 1].mean()))
    if a["spec"] is not None and ref["spec"] is not None:
        from .common import band_power
        bp, br = band_power(a["spec"]), band_power(ref["spec"])
        out["band_ratio"] = [x / max(y, 1e-20) for x, y in zip(bp, br)]
        out["spec_logerr_k1_8"] = float(np.abs(np.log(a["spec"][1:9] / np.maximum(ref["spec"][1:9], 1e-20))).mean())
    return out
