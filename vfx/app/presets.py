"""Demo presets (world units: 1 m box, y up). Keys 1-4 in the app."""
from __future__ import annotations

from dataclasses import dataclass, field

from vfx.core.fields import Emitter, Obstacle


def fire_emitter(pos, radius=0.045) -> Emitter:
    return Emitter(pos=pos, radius=radius, density_rate=0.0, temperature=1200.0, fuel_rate=6.0,
                   velocity=(0.0, 0.35))


def smoke_emitter(pos, radius=0.05, rate=0.9) -> Emitter:
    return Emitter(pos=pos, radius=radius, density_rate=rate, temperature=420.0, velocity=(0.0, 0.4))


@dataclass
class Preset:
    name: str
    emitters: list
    obstacles: list = field(default_factory=list)
    wind: tuple = (0.0, 0.0)


PRESETS = {
    1: Preset("campfire", [fire_emitter((0.5, 0.08), 0.06)],
              [Obstacle("box", (0.5, 0.02), (0.12, 0.02))]),
    2: Preset("smoke stack + circle", [smoke_emitter((0.5, 0.1), 0.06)],
              [Obstacle("circle", (0.5, 0.48), (0.1, 0.1))]),
    3: Preset("twin torches + wind", [fire_emitter((0.3, 0.1)), fire_emitter((0.7, 0.1))],
              [], wind=(0.5, 0.0)),
    4: Preset("box obstacles (OOD)", [smoke_emitter((0.32, 0.1), 0.05, 0.7), fire_emitter((0.65, 0.08), 0.04)],
              [Obstacle("box", (0.42, 0.5), (0.16, 0.035)), Obstacle("box", (0.72, 0.72), (0.06, 0.06))]),
}

DEFAULT_OBSTACLE = Obstacle("circle", (0.5, 0.5), (0.09, 0.09))
