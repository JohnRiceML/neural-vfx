"""Paired 3D simulation: the free 64^3 runtime sim + a 256^3 sim guided to follow it.

LR  = Solver3D(64, runtime_config(**scenario.sim))   exactly the sandbox's solver setting
HR  = Solver3D(256, SolverConfig3D(**scenario.sim))  full-quality pressure (reference)
Both get identical Forces3D every step. Solvers and state buffers are cached per resolution so the
solver's CUDA graphs (keyed by buffer pointers) are captured once per process.
"""
from __future__ import annotations

from typing import Iterator

import torch

from vfx.core.fields3d import HR_RES_3D, LR_RES_3D, SIM_DT

from .guide3d import Guide3D, GuideConfig3D, tracking_error

_solvers: dict = {}
_states: dict = {}


def solver(res: int, runtime: bool):
    from vfx.sim3d import Solver3D, SolverConfig3D
    key = (res, runtime)
    if key not in _solvers:
        _solvers[key] = Solver3D(res, SolverConfig3D())
    return _solvers[key]


def configure(s, overrides: dict, runtime: bool) -> None:
    from vfx.sim3d import SolverConfig3D, runtime_config
    s.config = runtime_config(**overrides) if runtime else SolverConfig3D(**overrides)


def fresh_state(s, key, obstacles):
    new = s.reset(obstacles)
    st = _states.get(key)
    if st is None:
        _states[key] = new
        return new
    st.data.copy_(new.data)
    st.solid.copy_(new.solid)
    st.meta = new.meta
    return st


def solver_version() -> str:
    from vfx.sim3d import SOLVER3D_VERSION
    return SOLVER3D_VERSION


def run_pair(sc, guide: GuideConfig3D | None, hr_res: int = HR_RES_3D, lr_res: int = LR_RES_3D,
             steps: int | None = None, with_hr: bool = True) -> Iterator[tuple]:
    """Yield (step, lr_state, hr_state, track) after every solver step (step = 0-based count of steps
    done - 1). track = GPU (3,) rel L2 at 16^3 (density, T-amb, velocity) of HR vs LR (None if no HR).
    guide=None runs the HR unguided. States are cached buffers: copy what you keep."""
    ls = solver(lr_res, True)
    configure(ls, sc.sim, True)
    lr = fresh_state(ls, ("lr", lr_res), sc.obstacles)
    hs = hr = g = None
    if with_hr:
        hs = solver(hr_res, False)
        configure(hs, sc.sim, False)
        hr = fresh_state(hs, ("hr", hr_res), sc.obstacles)
        g = Guide3D(guide, hr.solid, lr_res) if guide is not None else None
    n = sc.num_steps if steps is None else steps
    for s in range(n):
        f = sc.forces(s)
        if hr is not None:
            if g is not None:
                g.velocity(hr, lr)              # both hold the state after s steps
            hr = hs.step(hr, f, SIM_DT)
        lr = ls.step(lr, f, SIM_DT)
        tr = None
        if hr is not None:
            if g is not None:
                g.scalars(hr, lr)
            tr = tracking_error(hr.data, lr.data)
        yield s, lr, hr, tr


def free_cache() -> None:
    _solvers.clear()
    _states.clear()
    torch.cuda.empty_cache()
