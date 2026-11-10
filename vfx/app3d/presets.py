"""3D sandbox presets (world units: 1 m cube, y up). Keys 1-4."""
from __future__ import annotations

from dataclasses import dataclass, field

from vfx.core.fields3d import Emitter3D, Obstacle3D


def fire_emitter(pos, radius=0.05) -> Emitter3D:
    return Emitter3D(pos=tuple(pos), radius=radius, density_rate=0.0, temperature=1300.0, fuel_rate=7.0,
                     velocity=(0.0, 0.35, 0.0))


def smoke_emitter(pos, radius=0.07, rate=2.5) -> Emitter3D:
    return Emitter3D(pos=tuple(pos), radius=radius, density_rate=rate, temperature=620.0,
                     velocity=(0.0, 0.6, 0.0))


def puff_emitter(pos, radius=0.06) -> Emitter3D:
    return Emitter3D(pos=tuple(pos), radius=radius, density_rate=8.0, temperature=520.0)


@dataclass
class Preset:
    name: str
    emitters: list
    obstacles: list = field(default_factory=list)
    wind: tuple = (0.0, 0.0, 0.0)


PRESETS = {
    1: Preset("campfire", [fire_emitter((0.5, 0.09, 0.5), 0.065)],
              [Obstacle3D("box", (0.5, 0.025, 0.5), (0.11, 0.025, 0.11))]),
    2: Preset("smoke plume past a sphere", [smoke_emitter((0.46, 0.08, 0.5), 0.08)],
              [Obstacle3D("sphere", (0.52, 0.48, 0.5), (0.12, 0.12, 0.12))]),
    3: Preset("twin torches + wind", [fire_emitter((0.28, 0.1, 0.45)), fire_emitter((0.62, 0.1, 0.58))],
              [], wind=(0.45, 0.0, 0.1)),
    4: Preset("box obstacle", [smoke_emitter((0.4, 0.08, 0.5), 0.07, 2.0), fire_emitter((0.6, 0.08, 0.45), 0.045)],
              [Obstacle3D("box", (0.48, 0.55, 0.5), (0.2, 0.03, 0.16))]),
}

DEFAULT_OBSTACLE = Obstacle3D("sphere", (0.5, 0.5, 0.5), (0.11, 0.11, 0.11))
