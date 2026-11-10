"""3D sandbox presets (world units: 1 m cube, y up). Keys 1-4.

Each preset carries `sim`: SolverConfig3D overrides applied to BOTH the 64^3 runtime solver and the
128^3 reference (so the comparison stays like for like). The solver's own defaults (density
dissipation 0.05 /s) are physical-study values: in the closed 1 m box the smoke never leaves, so
after ~30 s it fills the box as a grey haze. A game effect dissipates, so the presets raise
`density_dissipation` (and `temperature_cooling` where the hot gas otherwise pools under the lid)
until each reaches a steady, localized plume. Emitter rates are raised to keep the plume near the
source as dense as before. Measured steady state: see README ("Presets").
"""
from __future__ import annotations

from dataclasses import dataclass, field

from vfx.core.fields3d import Emitter3D, Obstacle3D


def fire_emitter(pos, radius=0.05, fuel=7.0) -> Emitter3D:
    return Emitter3D(pos=tuple(pos), radius=radius, density_rate=0.0, temperature=1300.0, fuel_rate=fuel,
                     velocity=(0.0, 0.35, 0.0))


def smoke_emitter(pos, radius=0.07, rate=2.5) -> Emitter3D:
    return Emitter3D(pos=tuple(pos), radius=radius, density_rate=rate, temperature=620.0,
                     velocity=(0.0, 0.6, 0.0))


def puff_emitter(pos, radius=0.06) -> Emitter3D:
    return Emitter3D(pos=tuple(pos), radius=radius, density_rate=8.0, temperature=520.0)


# game-effect dissipation shared by every preset (per-preset entries override)
GAME_SIM = dict(density_dissipation=1.2, temperature_cooling=5.0)


@dataclass
class Preset:
    name: str
    emitters: list
    obstacles: list = field(default_factory=list)
    wind: tuple = (0.0, 0.0, 0.0)
    sim: dict = field(default_factory=lambda: dict(GAME_SIM))   # SolverConfig3D overrides


PRESETS = {
    1: Preset("campfire", [fire_emitter((0.5, 0.09, 0.5), 0.065, fuel=10.0)],
              [Obstacle3D("box", (0.5, 0.025, 0.5), (0.11, 0.025, 0.11))]),
    2: Preset("smoke plume past a sphere", [smoke_emitter((0.46, 0.08, 0.5), 0.08, rate=4.0)],
              [Obstacle3D("sphere", (0.52, 0.48, 0.5), (0.12, 0.12, 0.12))]),
    3: Preset("twin torches + wind", [fire_emitter((0.28, 0.1, 0.45), fuel=10.0),
                                      fire_emitter((0.62, 0.1, 0.58), fuel=10.0)],
              [], wind=(0.45, 0.0, 0.1)),
    4: Preset("box obstacle", [smoke_emitter((0.4, 0.08, 0.5), 0.07, 3.2), fire_emitter((0.6, 0.08, 0.45), 0.045, fuel=10.0)],
              [Obstacle3D("box", (0.48, 0.55, 0.5), (0.2, 0.03, 0.16))]),
}

DEFAULT_OBSTACLE = Obstacle3D("sphere", (0.5, 0.5, 0.5), (0.11, 0.11, 0.11))
