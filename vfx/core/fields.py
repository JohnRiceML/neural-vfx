"""Shared schema: every tensor that crosses a module boundary is defined here.

OWNER: lead. Changing this file requires a SPEC.md update and lead approval.

Demo (v0) scope: 2D only. Conventions:
  * Layout (C, H, W). Index [c, y, x]. y = 0 is the BOTTOM row (gravity points to -y,
    buoyancy pushes +y). The renderer flips vertically for display.
  * The domain is a square box of side DOMAIN_SIZE meters. dx = DOMAIN_SIZE / W.
    All world-space quantities (emitter positions, radii, speeds) are in meters / m/s,
    so the same Forces drive a 128^2 and a 512^2 solver identically.
  * Velocity is collocated (cell-centered) at the boundary, in m/s.
  * Temperature is in kelvin. AMBIENT_TEMP is the "cold" value.
  * float32 inside the solver, float16 at the network boundary.
  * Tensors live on the GPU ("cuda"). A CPU copy inside the runtime frame loop is a bug,
    except the single final display blit in app/ (documented demo exception).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

# ----------------------------------------------------------------------------- constants
DOMAIN_SIZE: float = 1.0          # meters, square box
AMBIENT_TEMP: float = 293.0       # K
LR_RES: int = 128                 # low-res grid (runtime classical solver)
UPSCALE: int = 4                  # neural upscale factor
HR_RES: int = LR_RES * UPSCALE    # 512, high-res reference / network output
SIM_DT: float = 1.0 / 30.0        # s, solver step (sim at 30 Hz, render at 60)

# Fixed channel order. The demo uses ONE layout everywhere (fire layout); smoke scenes
# simply have fuel == 0. vel_z is absent in 2D (absent, not zero-filled).
CHANNELS: Tuple[str, ...] = ("density", "vel_x", "vel_y", "temperature", "fuel")
CH = {name: i for i, name in enumerate(CHANNELS)}
NUM_CHANNELS = len(CHANNELS)

# What the detail network predicts at HR.
HR_CHANNELS: Tuple[str, ...] = ("density", "temperature")
HR_CH = {name: i for i, name in enumerate(HR_CHANNELS)}


# ----------------------------------------------------------------------------- field state
@dataclass
class FieldMeta:
    res: Tuple[int, int]                 # (H, W)
    dx: float                            # meters per cell
    dt: float                            # seconds per step
    channels: Tuple[str, ...] = CHANNELS
    ambient_temp: float = AMBIENT_TEMP
    step: int = 0                        # simulation step counter
    solver_version: str = "unknown"

    @staticmethod
    def for_res(res: int, dt: float = SIM_DT) -> "FieldMeta":
        return FieldMeta(res=(res, res), dx=DOMAIN_SIZE / res, dt=dt)


@dataclass
class FieldState:
    """The core contract. data: (C, H, W) float32 on cuda, channel order = CHANNELS.

    solid: (H, W) float32 on cuda, 1.0 inside obstacles / walls-as-obstacles, 0.0 fluid.
    The box boundary itself is always a wall (no-slip normal, free-slip tangential);
    it is NOT marked in `solid`.
    """
    data: torch.Tensor
    solid: torch.Tensor
    meta: FieldMeta

    def ch(self, name: str) -> torch.Tensor:
        return self.data[CH[name]]

    @property
    def velocity(self) -> torch.Tensor:          # (2, H, W) view: vel_x, vel_y
        return self.data[CH["vel_x"]:CH["vel_y"] + 1]

    def validate(self) -> None:
        H, W = self.meta.res
        assert self.data.shape == (NUM_CHANNELS, H, W), self.data.shape
        assert self.solid.shape == (H, W), self.solid.shape
        assert self.data.dtype == torch.float32
        assert self.data.is_cuda and self.solid.is_cuda


@dataclass
class HRFields:
    """Output of the upscaler (neural or classical baseline); input to the renderer.

    density:     (H, W) on cuda, >= 0, nominally in [0, ~2]
    temperature: (H, W) on cuda, kelvin
    Either float16 or float32. Resolution is usually HR_RES but the renderer must accept
    any resolution (it also draws raw LR fields for the "classical low-res" view).
    """
    density: torch.Tensor
    temperature: torch.Tensor


# ----------------------------------------------------------------------------- forces
@dataclass
class Emitter:
    """A circular source, world units. Rates are per second."""
    pos: Tuple[float, float]            # (x, y) meters, y up
    radius: float                       # meters
    density_rate: float = 1.0           # density units / s added at center
    temperature: float = 0.0            # target temperature (K) inside emitter; 0 = no heat
    fuel_rate: float = 0.0              # fuel units / s (fire); 0 = smoke only
    velocity: Tuple[float, float] = (0.0, 0.0)   # m/s imposed inside the emitter


@dataclass
class Impulse:
    """A localized velocity push (e.g. mouse drag). Applied for one step."""
    pos: Tuple[float, float]            # meters
    radius: float                       # meters
    velocity: Tuple[float, float]       # m/s added (Gaussian falloff)


@dataclass
class Forces:
    emitters: List[Emitter] = field(default_factory=list)
    impulses: List[Impulse] = field(default_factory=list)
    wind: Tuple[float, float] = (0.0, 0.0)       # m/s^2 global acceleration


@dataclass
class Obstacle:
    """Static obstacle, world units. kind: 'circle' (size = radius) or 'box' (size = half-extents)."""
    kind: str
    pos: Tuple[float, float]
    size: Tuple[float, float]


def rasterize_obstacles(obstacles: List[Obstacle], res: int, device: str = "cuda") -> torch.Tensor:
    """(res, res) float32 solid mask. Shared so LR and HR grids agree."""
    dx = DOMAIN_SIZE / res
    c = (torch.arange(res, device=device, dtype=torch.float32) + 0.5) * dx
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    solid = torch.zeros(res, res, device=device, dtype=torch.float32)
    for o in obstacles:
        if o.kind == "circle":
            m = (xx - o.pos[0]) ** 2 + (yy - o.pos[1]) ** 2 <= o.size[0] ** 2
        elif o.kind == "box":
            m = ((xx - o.pos[0]).abs() <= o.size[0]) & ((yy - o.pos[1]).abs() <= o.size[1])
        else:
            raise ValueError(o.kind)
        solid[m] = 1.0
    return solid


@dataclass
class Scenario:
    """A reproducible scene. datagen samples these; the app has a few presets."""
    seed: int
    emitters: List[Emitter]
    obstacles: List[Obstacle] = field(default_factory=list)
    wind: Tuple[float, float] = (0.0, 0.0)
    num_steps: int = 120
    tags: List[str] = field(default_factory=list)   # e.g. ["fire"], ["ood", "box_obstacle"]
