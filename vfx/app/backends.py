"""Resolve the real solver / upscalers, falling back to vfx.app._stubs when imports fail."""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field


@dataclass
class Backends:
    Solver: type
    SolverConfig: type
    classical_upscale: object
    upscaler: object | None            # neural Upscaler instance, or None if unavailable
    neural_reason: str = ""            # why neural is unavailable
    stubs: list = field(default_factory=list)   # names of stand-ins in use
    notes: list = field(default_factory=list)


def _short(e: BaseException) -> str:
    return f"{type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"


def load_backends(checkpoint: str | None = None, device: str = "cuda", force_stub_sim: bool = False) -> Backends:
    from vfx.app import _stubs
    notes, stubs = [], []
    try:
        if force_stub_sim:
            raise ImportError("forced by --stub-sim")
        from vfx.sim import Solver, SolverConfig            # noqa: F401
    except Exception as e:                                   # module not landed / broken
        Solver, SolverConfig = _stubs.Solver, _stubs.SolverConfig
        stubs.append("sim.Solver")
        notes.append(f"sim: {_short(e)}")
    try:
        from vfx.runtime.warp import classical_upscale
    except Exception as e:
        try:
            from vfx.runtime import classical_upscale
        except Exception:
            classical_upscale = _stubs.classical_upscale
            stubs.append("runtime.classical_upscale")
            notes.append(f"classical_upscale: {_short(e)}")
    upscaler, reason = None, ""
    try:
        from vfx.runtime import Upscaler
        upscaler = Upscaler(checkpoint, device=device)
        if hasattr(upscaler, "warmup"):
            upscaler.warmup()          # pre-capture CUDA graphs for every crop shape (no mid-run hitches)
    except Exception as e:
        reason = _short(e)
        if not isinstance(e, (FileNotFoundError, ImportError)):
            traceback.print_exc()
    return Backends(Solver, SolverConfig, classical_upscale, upscaler, reason, stubs, notes)
