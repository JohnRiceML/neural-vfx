"""3D shared schema. OWNER: lead. Mirrors fields.py (2D); changing it needs lead approval.

Conventions:
  * Layout (C, D, H, W). Index [c, z, y, x]. x right, y UP (buoyancy +y, gravity -y), z depth.
  * Domain is a cube of side DOMAIN_SIZE meters; dx = DOMAIN_SIZE / res. World units everywhere
    (meters, m/s, kelvin, per-second rates), so a 64^3 and a 256^3 run of one scene match at large scales.
  * Velocity collocated (cell-centred) at the module boundary, m/s.
  * float32 in the solver, float16 at the network boundary. Tensors stay on cuda.
  * The six box faces are walls. `solid` marks static obstacles only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch

from .fields import AMBIENT_TEMP, DOMAIN_SIZE, SIM_DT

LR_RES_3D: int = 64                 # runtime classical solver (spec: 64^3)
UPSCALE_3D: int = 4
HR_RES_3D: int = LR_RES_3D * UPSCALE_3D   # 256

CHANNELS_3D: Tuple[str, ...] = ("density", "vel_x", "vel_y", "vel_z", "temperature", "fuel")
CH3 = {name: i for i, name in enumerate(CHANNELS_3D)}
NUM_CHANNELS_3D = len(CHANNELS_3D)

Vec3 = Tuple[float, float, float]


@dataclass
class FieldMeta3D:
    res: Tuple[int, int, int]            # (D, H, W)
    dx: float
    dt: float
    channels: Tuple[str, ...] = CHANNELS_3D
    ambient_temp: float = AMBIENT_TEMP
    step: int = 0
    solver_version: str = "unknown"

    @staticmethod
    def for_res(res: int, dt: float = SIM_DT) -> "FieldMeta3D":
        return FieldMeta3D(res=(res, res, res), dx=DOMAIN_SIZE / res, dt=dt)


@dataclass
class FieldState3D:
    """data: (6, D, H, W) float32 cuda. solid: (D, H, W) float32 cuda, 1 inside obstacles."""
    data: torch.Tensor
    solid: torch.Tensor
    meta: FieldMeta3D

    def ch(self, name: str) -> torch.Tensor:
        return self.data[CH3[name]]

    @property
    def velocity(self) -> torch.Tensor:          # (3, D, H, W): vel_x, vel_y, vel_z
        return self.data[CH3["vel_x"]:CH3["vel_z"] + 1]


@dataclass
class Volume:
    """What the renderer draws. density (D,H,W) >= 0, temperature (D,H,W) kelvin; any resolution,
    float16 or float32, cuda. Produced by the solver directly (raw), by a classical upscale, or
    by a neural upscaler."""
    density: torch.Tensor
    temperature: torch.Tensor


@dataclass
class Emitter3D:
    pos: Vec3
    radius: float
    density_rate: float = 1.0
    temperature: float = 0.0            # K target inside the emitter; 0 = no heat
    fuel_rate: float = 0.0
    velocity: Vec3 = (0.0, 0.0, 0.0)


@dataclass
class Impulse3D:
    pos: Vec3
    radius: float
    velocity: Vec3                      # m/s added, Gaussian falloff, one step


@dataclass
class Forces3D:
    emitters: List[Emitter3D] = field(default_factory=list)
    impulses: List[Impulse3D] = field(default_factory=list)
    wind: Vec3 = (0.0, 0.0, 0.0)


@dataclass
class Obstacle3D:
    """kind: 'sphere' (size[0] = radius) or 'box' (size = half-extents)."""
    kind: str
    pos: Vec3
    size: Vec3


def rasterize_obstacles_3d(obstacles: List[Obstacle3D], res: int, device: str = "cuda") -> torch.Tensor:
    dx = DOMAIN_SIZE / res
    c = (torch.arange(res, device=device, dtype=torch.float32) + 0.5) * dx
    zz, yy, xx = torch.meshgrid(c, c, c, indexing="ij")
    solid = torch.zeros(res, res, res, device=device, dtype=torch.float32)
    for o in obstacles:
        if o.kind == "sphere":
            m = (xx - o.pos[0]) ** 2 + (yy - o.pos[1]) ** 2 + (zz - o.pos[2]) ** 2 <= o.size[0] ** 2
        elif o.kind == "box":
            m = (((xx - o.pos[0]).abs() <= o.size[0]) & ((yy - o.pos[1]).abs() <= o.size[1])
                 & ((zz - o.pos[2]).abs() <= o.size[2]))
        else:
            raise ValueError(o.kind)
        solid[m] = 1.0
    return solid


@dataclass
class Camera:
    """Perspective camera in world space; the domain cube spans [0, DOMAIN_SIZE]^3."""
    position: Vec3 = (0.5, 0.55, -1.4)
    target: Vec3 = (0.5, 0.45, 0.5)
    up: Vec3 = (0.0, 1.0, 0.0)
    fov_y_deg: float = 40.0
