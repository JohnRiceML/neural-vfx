"""3D solver configuration. Field names match the 2D SolverConfig (tests zero them to isolate terms).

World units (meters, seconds, kelvin) so a 64^3 and a 256^3 solver driven by the same Forces3D
agree at large scales.
"""
from __future__ import annotations

from dataclasses import dataclass

SOLVER3D_VERSION: str = "sim3d-warp-0.1.0"


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


def runtime_config(**kw) -> SolverConfig3D:
    """Game-budget setting: a fixed 4 PCG iterations (pressure_tol = 0 keeps only the float32
    floor / stagnation stop), so the cost is constant. Meets the 1e-4 divergence gate in open
    scenes but NOT next to obstacles (see README). Everything else as the full-quality defaults."""
    base = dict(pressure_tol=0.0, pressure_max_iters=4)
    base.update(kw)
    return SolverConfig3D(**base)
