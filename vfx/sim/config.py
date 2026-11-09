"""Solver configuration. Field names are fixed by CLAUDE.md (tests zero them to isolate terms).

All parameters are in world units (meters, seconds, kelvin) so a 128^2 and a 512^2 solver
driven by the same Forces agree at large scales.
"""
from __future__ import annotations

from dataclasses import dataclass

SOLVER_VERSION: str = "sim2d-warp-0.2.0"


@dataclass
class SolverConfig:
    # --- buoyancy: f_y = -buoyancy_alpha * density + buoyancy_beta * (T - T_ambient)
    buoyancy_alpha: float = 0.6        # m/s^2 per density unit (smoke weight)
    buoyancy_beta: float = 0.004       # m/s^2 per kelvin above ambient (heat lift)
    # --- vorticity confinement: f = vorticity_eps * dx * (N x omega)   [vorticity_eps in 1/s]
    #     The dx factor makes the per-step velocity change of a grid-scale vortex resolution
    #     independent (it re-injects what the dx-scaled numerical dissipation removes).
    #     omega is [1,2,1]^2-smoothed first and N = grad|w| / sqrt(|grad|w||^2 + (floor*|w|)^2),
    #     so N fades out smoothly on vorticity ridges instead of flipping sign cell by cell.
    vorticity_eps: float = 6.0
    # --- decay (rates per second, exponential): d *= exp(-k dt), T-T_amb *= exp(-k dt)
    density_dissipation: float = 0.05
    temperature_cooling: float = 4.0
    # --- combustion: fuel burns where T > ignition_temp (100 K linear ramp) at burn_rate (1/s);
    #     each fuel unit burned adds burn_heat kelvin (scaled by the remaining headroom to
    #     max_temperature) and smoke_from_fuel density.
    burn_rate: float = 8.0
    ignition_temp: float = 550.0
    burn_heat: float = 4000.0
    smoke_from_fuel: float = 0.35
    # --- pressure: PCG (multigrid preconditioner) stops when max normalized divergence
    #     (|div u| dx / max|u|) < pressure_tol, or after pressure_max_iters iterations.
    #     Checked on the GPU (CUDA-graph conditional loop): no host sync. 0 = fixed iterations.
    pressure_tol: float = 2e-5
    pressure_max_iters: int = 40
    # --- extras (not part of the fixed name set; safe defaults)
    max_temperature: float = 3000.0   # K, hard cap + combustion saturation
    wind_marker_temp: float = 300.0    # kelvin above ambient that counts as a "full" wind marker
    # m^2/s, diffusivity of temperature AND fuel (Lewis number 1). Gives the ignition front a
    # finite world-space thickness; without it the bistable burn makes a 1-cell temperature
    # step that the buoyancy turns into grid-scale noise. 0 disables. The effective value is
    # capped at 0.24 * n_sub * dx^2 / dt (n_sub = 2 per 256 cells), i.e. ~9e-5 at 512^2, 30 Hz.
    heat_diffusion: float = 5.0e-5
    confinement_floor: float = 0.5     # see vorticity confinement above (0 = plain unit N)
