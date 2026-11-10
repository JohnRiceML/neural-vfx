"""Gate 1 scenes, as named in native/README.md, built from vfx/sim3d/selfcheck.py.

OWNER: native test engineer. Shared by native/tools/dump_reference.py and native/tests/.

Each scene is a `Scene` with: obstacles, a forces(i) function (i = step index, forces for step i), an
optional initial-state builder (None = reset state), an optional `project_first` flag (the initial
state is projected once before stepping, as selfcheck.mass_drift does) and the config override.

  campfire      selfcheck.campfire, no obstacles
  plume         selfcheck.plume + selfcheck.SPHERE
  determinism   selfcheck.determinism: campfire + SPHERE, an impulse every 10 steps
  stress        selfcheck.stress: 4x fuel at 1500 K, extra 1200 K jet, an impulse every 7 steps
                (rng = default_rng(0)), rotating wind, SPHERE + box
  mass_box      selfcheck.mass_drift: closed box, sphere, density blob + shear velocity, no sources,
                density_dissipation 0, projected once before stepping
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from vfx.core.fields import AMBIENT_TEMP, SIM_DT
from vfx.core.fields3d import CH3, NUM_CHANNELS_3D, Emitter3D, Forces3D, Impulse3D, Obstacle3D
from vfx.sim3d import selfcheck as SC

DT = SIM_DT


@dataclass
class Scene:
    name: str
    obstacles: List[Obstacle3D]
    forces: Callable[[int], Forces3D]
    initial: Optional[Callable[[int, np.ndarray], np.ndarray]] = None   # (res, solid) -> (6,N,N,N)
    project_first: bool = False
    config_overrides: dict = field(default_factory=dict)


def _campfire(i: int) -> Forces3D:
    return SC.campfire(i * DT)


def _plume(i: int) -> Forces3D:
    return SC.plume(i * DT)


def _determinism(i: int) -> Forces3D:
    f = SC.campfire(i * DT)
    f.impulses = [Impulse3D(pos=(0.5, 0.3, 0.5), radius=0.08, velocity=(1.0, 0.5, -0.5))] if i % 10 == 0 else []
    return f


class _StressForces:
    """Replays selfcheck.stress's force sequence. The rng is consumed in step order, so the sequence is
    generated once up front and indexed (any i, any order)."""

    def __init__(self, steps: int = 1000):
        rng = np.random.default_rng(0)
        self.seq = []
        for i in range(steps):
            f = SC.campfire(i * DT)
            for e in f.emitters:
                e.fuel_rate *= 4.0
                e.temperature = 1500.0
            f.emitters.append(Emitter3D(pos=(0.3, 0.2, 0.7), radius=0.08, density_rate=6.0, temperature=1200.0,
                                        fuel_rate=20.0, velocity=(1.5, 2.0, 0.0)))
            if i % 7 == 0:
                p = rng.uniform(0.15, 0.85, 3)
                f.impulses = [Impulse3D(pos=tuple(p), radius=0.07, velocity=tuple(rng.uniform(-6, 6, 3)))]
            f.wind = (float(np.sin(0.01 * i)), 0.0, float(np.cos(0.013 * i)))
            self.seq.append(f)

    def __call__(self, i: int) -> Forces3D:
        if i >= len(self.seq):
            raise IndexError(f"stress scenario defined for {len(self.seq)} steps")
        return dataclasses.replace(self.seq[i])


def _mass_box_initial(res: int, solid: np.ndarray) -> np.ndarray:
    """selfcheck.mass_drift's initial state, computed with torch on cuda exactly as there."""
    import torch
    c = (torch.arange(res, device="cuda", dtype=torch.float32) + 0.5) / res
    zz, yy, xx = torch.meshgrid(c, c, c, indexing="ij")
    st_solid = torch.from_numpy(solid).cuda()
    blob = torch.exp(-((xx - 0.35) ** 2 + (yy - 0.3) ** 2 + (zz - 0.45) ** 2) / 0.01) * (st_solid < 0.5)
    data = torch.zeros(NUM_CHANNELS_3D, res, res, res, device="cuda", dtype=torch.float32)
    data[CH3["temperature"]] = AMBIENT_TEMP
    data[CH3["density"]] = blob
    data[CH3["temperature"]] = AMBIENT_TEMP + 400.0 * blob
    data[CH3["vel_x"]] = 0.5 * torch.sin(3.1 * yy)
    data[CH3["vel_z"]] = 0.3 * torch.cos(2.7 * xx)
    return data.cpu().numpy()


def _none(i: int) -> Forces3D:
    return Forces3D()


def get_scene(name: str) -> Scene:
    if name == "campfire":
        return Scene("campfire", [], _campfire)
    if name == "plume":
        return Scene("plume", [SC.SPHERE], _plume)
    if name == "determinism":
        return Scene("determinism", [SC.SPHERE], _determinism)
    if name == "stress":
        return Scene("stress", [SC.SPHERE, Obstacle3D("box", (0.25, 0.7, 0.3), (0.1, 0.02, 0.15))], _StressForces())
    if name == "mass_box":
        return Scene("mass_box", [Obstacle3D("sphere", (0.6, 0.6, 0.55), (0.1, 0.1, 0.1))], _none,
                     initial=_mass_box_initial, project_first=True, config_overrides=dict(density_dissipation=0.0))
    raise KeyError(name)


SCENE_NAMES = ("campfire", "plume", "determinism", "stress", "mass_box")


def forces_to_json(f: Forces3D) -> dict:
    return dict(emitters=[dataclasses.asdict(e) for e in f.emitters],
                impulses=[dataclasses.asdict(m) for m in f.impulses], wind=list(f.wind))


def forces_from_json(d: dict) -> Forces3D:
    return Forces3D(emitters=[Emitter3D(**{**e, "pos": tuple(e["pos"]), "velocity": tuple(e["velocity"])})
                              for e in d["emitters"]],
                    impulses=[Impulse3D(**{**m, "pos": tuple(m["pos"]), "velocity": tuple(m["velocity"])})
                              for m in d["impulses"]],
                    wind=tuple(d["wind"]))
