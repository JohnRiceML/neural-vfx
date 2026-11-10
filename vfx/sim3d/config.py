"""3D solver configuration. Field names match the 2D SolverConfig (tests zero them to isolate terms).

World units (meters, seconds, kelvin) so a 64^3 and a 256^3 solver driven by the same Forces3D
agree at large scales.
"""
from __future__ import annotations

from dataclasses import dataclass

# Version of the DEFAULT path (pressure_mode "pcg", scalar_res None). Unchanged since 0.1.0: the
# default step is bit-identical, so datasets / checkpoints built with it stay valid. Opt-in options
# are recorded in state.meta.solver_version as "+suffixes" (see solver_version_tag) instead.
SOLVER3D_VERSION: str = "sim3d-warp-0.1.0"
# Code revision: 0.2.0 adds the opt-in fused pressure solve and the finer scalar grid.
SOLVER3D_REVISION: str = "0.2.0"


@dataclass
class SolverConfig3D:
    # --- buoyancy: f_y = -buoyancy_alpha * density + buoyancy_beta * (T - T_ambient)
    buoyancy_alpha: float = 0.6        # m/s^2 per density unit (smoke weight)
    buoyancy_beta: float = 0.004       # m/s^2 per kelvin above ambient (heat lift)
    # --- vorticity confinement: f = vorticity_eps * dx * (N x omega_s)   [1/s]
    #     omega_s = [1,2,1]^3-smoothed vorticity, N = grad|w| / sqrt(|grad|w||^2 + (floor |w|)^2)
    vorticity_eps: float = 6.0
    # --- decay (rates per second, exponential)
    density_dissipation: float = 0.05
    temperature_cooling: float = 4.0
    # --- combustion (first order above ignition_temp, 100 K ramp; heat saturates at max_temperature)
    burn_rate: float = 8.0
    ignition_temp: float = 550.0
    burn_heat: float = 4000.0
    smoke_from_fuel: float = 0.35
    # --- pressure (MG-preconditioned CG, GPU-side stopping test, no host sync)
    #     stops when max |div u| dx / max|u| < pressure_tol or after pressure_max_iters.
    #     pressure_tol = 0 -> always run pressure_max_iters (fixed cost, runtime setting).
    pressure_tol: float = 2e-5
    pressure_max_iters: int = 40
    # start CG from the previous step's solution (phi is kept in the solver between calls)
    pressure_warm_start: bool = True
    # --- extras
    max_temperature: float = 3000.0    # K, hard cap + combustion saturation
    wind_marker_temp: float = 300.0    # K above ambient that counts as a full wind marker
    heat_diffusion: float = 5.0e-5     # m^2/s, temperature AND fuel (Lewis 1); explicit, capped
    confinement_floor: float = 0.5     # 0 = plain unit N
    # --- vorticity confinement, resolution-consistent (opt-in; 0 = off, the default path):
    #     eps_used = vorticity_eps * res / confinement_ref_res, i.e. the same eps * dx (confinement per
    #     unit vorticity) as a confinement_ref_res^3 run with vorticity_eps. 64^3 with ref 128 -> eps 3.
    confinement_ref_res: int = 0
    # --- pressure solver choice (opt-in; "pcg" is the default, bit-identical path above)
    #     "fused": multigrid as a stationary iteration, no CG (pressure_fused.py). It runs
    #     pressure_cycles V-cycles, then - if pressure_tol > 0 - more cycles while the normalized
    #     residual >= pressure_tol, up to pressure_max_iters cycles (GPU-side test, graph conditional).
    #     pressure_tol = 0 gives a fixed cost: a LOWER-QUALITY LEVEL that misses the 1e-4 gate on a cold
    #     start and in the impulse stress run (README: fused levels).
    pressure_mode: str = "pcg"
    pressure_cycles: int = 2           # fused: V-cycles always run
    pressure_smooth: int = 4           # fused: exact-K sweeps before / between / after the V-cycles
    pressure_smoother: str = "chebyshev"   # fused: "chebyshev" (weights on [0.5, 2]) | "jacobi" (omega 0.8)


def fused_config(**kw) -> SolverConfig3D:
    """Fused pressure solve, adaptive: 2 V-cycles + 4 Chebyshev sweeps, then more cycles until the
    normalized residual < 5e-5 (at most 24 cycles). The tail handles cold starts and sudden changes:
    measured <= 8.8e-5 cold, <= 7.5e-5 in the 1000-step stress run (README)."""
    base = dict(pressure_mode="fused", pressure_tol=5e-5, pressure_max_iters=24)
    base.update(kw)
    return SolverConfig3D(**base)


def fused_fixed_config(**kw) -> SolverConfig3D:
    """LOWER-QUALITY LEVEL: fused solve at a fixed cost (2 V-cycles, no stopping test). Measured: max
    normalized div 1.5e-4 (median 5e-5..9e-5) in the warm-started campfire / plume runs, ~1e-2 after a
    cold project(), up to 7.6e-3 on impulse steps of the stress run."""
    base = dict(pressure_mode="fused", pressure_tol=0.0)
    base.update(kw)
    return SolverConfig3D(**base)


def solver_version_tag(cfg: SolverConfig3D, scalar_res: int | None = None) -> str:
    """SOLVER3D_VERSION, plus a suffix per opt-in option in use (empty for the default path)."""
    tag = SOLVER3D_VERSION
    if cfg.confinement_ref_res:
        tag += f"+confref{cfg.confinement_ref_res}"
    if cfg.pressure_mode != "pcg":
        tag += f"+{cfg.pressure_mode}-c{cfg.pressure_cycles}s{cfg.pressure_smooth}{cfg.pressure_smoother[:4]}"
        tag += "-adaptive" if cfg.pressure_tol > 0 else "-fixed"
    if scalar_res is not None:
        tag += f"+scalars{scalar_res}"
    return tag


def runtime_config(**kw) -> SolverConfig3D:
    """Game-budget setting: a fixed 4 PCG iterations (pressure_tol = 0 keeps only the float32
    floor / stagnation stop), so the cost is constant. Meets the 1e-4 divergence gate in open
    scenes but NOT next to obstacles (see README). Everything else as the full-quality defaults."""
    base = dict(pressure_tol=0.0, pressure_max_iters=4)
    base.update(kw)
    return SolverConfig3D(**base)
