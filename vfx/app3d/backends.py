"""Resolve the real 3D solver / upscalers, falling back to vfx.app3d._stubs when imports fail."""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field


@dataclass
class Backends3D:
    Solver: type
    SolverConfig: type                         # full-quality config (128^3 reference)
    runtime_config: object                     # () -> config for the 64^3 game-budget sim
    classical_upscale: object
    volume_upscaler: object | None = None      # route A instance (untrained ok) or None
    screen_detail: object | None = None        # route B instance (untrained ok) or None
    reasons: dict = field(default_factory=dict)  # mode -> why unavailable
    stubs: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def _short(e: BaseException) -> str:
    return f"{type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"


def _smoke_test(Solver, config, device) -> None:
    """One reset + step at 64^3 so compile/runtime errors surface here, not mid-frame."""
    import torch
    from vfx.core.fields3d import LR_RES_3D, SIM_DT, Emitter3D, Forces3D
    s = Solver(LR_RES_3D, config(), device=device)
    st = s.reset([])
    s.step(st, Forces3D(emitters=[Emitter3D((0.5, 0.2, 0.5), 0.05, 1.0, 600.0)]), SIM_DT)
    torch.cuda.synchronize()


def load_backends(device: str = "cuda", force_stub_sim: bool = False, checkpoint_a=None,
                  checkpoint_b=None, warmup: bool = False) -> Backends3D:
    from vfx.app3d import _stubs
    notes, stubs, reasons = [], [], {}
    try:
        if force_stub_sim:
            raise ImportError("forced by --stub-sim")
        from vfx.sim3d import Solver3D as Solver, SolverConfig3D as SolverConfig
        try:
            from vfx.sim3d import runtime_config
        except ImportError:
            runtime_config = SolverConfig
        _smoke_test(Solver, runtime_config, device)      # a broken in-progress solver -> stub
    except Exception as e:
        if not isinstance(e, ImportError):
            traceback.print_exc()
        Solver, SolverConfig = _stubs.Solver3D, _stubs.SolverConfig3D
        runtime_config = SolverConfig
        stubs.append("sim3d.Solver3D")
        notes.append(f"sim3d: {_short(e)}")
    try:
        from vfx.runtime3d import classical_upscale_3d as cu
    except Exception as e:
        cu = _stubs.classical_upscale_3d
        stubs.append("runtime3d.classical_upscale_3d")
        notes.append(f"classical_upscale_3d: {_short(e)}")
    va = sb = None
    try:
        from vfx.runtime3d import VolumeUpscaler
        va = VolumeUpscaler(checkpoint_a, device=device)
        if warmup and hasattr(va, "warmup"):
            from vfx.core.fields3d import SIM_DT
            t0 = time.perf_counter()
            va.warmup(SIM_DT)          # pre-capture CUDA graphs for every brick-count bucket (no mid-run hitches)
            notes.append(f"VolumeUpscaler.warmup {time.perf_counter() - t0:.1f} s")
    except Exception as e:
        reasons["neural_vol"] = _short(e)
        if not isinstance(e, (ImportError, FileNotFoundError)):
            traceback.print_exc()
    try:
        from vfx.runtime3d import ScreenDetail
        sb = ScreenDetail(checkpoint_b, device=device)
    except Exception as e:
        reasons["neural_screen"] = _short(e)
        if not isinstance(e, (ImportError, FileNotFoundError)):
            traceback.print_exc()
    return Backends3D(Solver, SolverConfig, runtime_config, cu, va, sb, reasons, stubs, notes)
