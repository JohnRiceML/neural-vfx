"""Randomized 3D scenarios. Deterministic from (seed, split). World units (1 m cube, y up).

The distribution mirrors the sandbox (vfx/app3d/presets.py): the four preset KINDS (campfire, smoke
plume past a sphere, twin torches in wind, smoke + fire mix) plus generic random scenes, all with the
sandbox's game dissipation (density 1.2 /s, cooling 5 /s) varied around it. The exact presets are never
sampled (continuous parameters; `is_preset_like` guards it) - they are the live test.

Train / test: 1-3 emitters, fire / smoke / mixed, mild wind (|w_xz| <= 0.5 m/s^2), 0-2 spheres,
optional floor "hearth" slab under a campfire (a floor-level box like preset 1's; it sits under the
emitter, not in the flow).
OOD: in-flow box obstacles (preset 4 style, mid-air slabs / blocks), strong wind, large emitters.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from vfx.core.fields3d import Emitter3D, Impulse3D, Obstacle3D

SPLITS = ("train", "test", "ood")
DEFAULT_FRAMES = 16
WARMUP_RANGE = (30, 150)              # steps before the first saved frame (1.0 - 5.0 s)

KINDS = ("campfire", "plume_sphere", "torches_wind", "mixed", "random")
KIND_PROBS = (0.2, 0.2, 0.15, 0.15, 0.3)

RANGES = {
    "density_dissipation": (0.8, 1.8),        # sandbox 1.2 /s
    "temperature_cooling": (3.5, 7.0),        # sandbox 5 /s
    "emitter_xz": (0.25, 0.75),
    "emitter_y": (0.06, 0.14),
    "fire": {"radius": (0.04, 0.075), "temperature": (1150.0, 1400.0), "fuel_rate": (5.0, 12.0),
             "density_rate": (0.0, 0.3), "vel_y": (0.2, 0.5)},
    "smoke": {"radius": (0.05, 0.09), "temperature": (520.0, 700.0), "density_rate": (1.8, 4.5),
              "vel_y": (0.35, 0.8)},
    "emitter_vel_xz": (-0.1, 0.1),
    "wind_mild": (0.0, 0.5),                  # |wind| m/s^2 (preset 3: 0.46), horizontal, random direction
    "wind_prob": 0.5,                         # random / mixed / campfire kinds: P(wind)
    "sphere": {"num": (0, 2), "radius": (0.06, 0.14), "y": (0.3, 0.65), "xz": (0.25, 0.75)},
    "hearth_prob": 0.5,                       # campfire: floor slab half-extents
    "hearth": {"half_xz": (0.07, 0.13), "half_y": (0.015, 0.035)},
    "impulse_prob": 0.25,                     # sandbox 'push': a gust every few steps
    "impulse": {"speed": (0.5, 2.0), "radius": (0.08, 0.15), "every": (6, 20)},
}

OOD_RANGES = {
    "factor_prob": 0.6,
    "box_obstacle": {"num": (1, 2), "half_xz": (0.05, 0.22), "half_y": (0.02, 0.08), "y": (0.3, 0.7)},
    "strong_wind": {"abs": (0.8, 1.6), "y": (-0.2, 0.2)},
    "large_emitter": {"radius": (0.1, 0.15), "rate_scale": (1.3, 2.0), "vel_y": (0.6, 1.1)},
}


@dataclass
class Scenario3D:
    seed: int
    split: str
    kind: str
    emitters: list
    obstacles: list
    wind: tuple
    sim: dict                                  # SolverConfig3D overrides (both resolutions)
    warmup: int
    frames: int
    impulses: list = field(default_factory=list)   # [(every_n_steps, phase, Impulse3D)]
    tags: list = field(default_factory=list)

    @property
    def num_steps(self) -> int:
        return self.warmup + self.frames

    def forces(self, step: int):
        """Forces3D for solver step `step` (0-based)."""
        from vfx.core.fields3d import Forces3D
        imps = [imp for every, phase, imp in self.impulses if step % every == phase]
        return Forces3D(emitters=list(self.emitters), impulses=imps, wind=tuple(self.wind))

    def asdict(self) -> dict:
        d = asdict(self)
        d["impulses"] = [{"every": e, "phase": p, **asdict(i)} for e, p, i in self.impulses]
        return d


def _u(rng, lo_hi) -> float:
    return float(rng.uniform(*lo_hi))


def _fire(rng, pos, radius=None, scale=1.0, vel_y=None) -> Emitter3D:
    R = RANGES["fire"]
    return Emitter3D(pos=pos, radius=radius or _u(rng, R["radius"]), density_rate=_u(rng, R["density_rate"]),
                     temperature=_u(rng, R["temperature"]), fuel_rate=scale * _u(rng, R["fuel_rate"]),
                     velocity=(_u(rng, RANGES["emitter_vel_xz"]), vel_y or _u(rng, R["vel_y"]),
                               _u(rng, RANGES["emitter_vel_xz"])))


def _smoke(rng, pos, radius=None, scale=1.0, vel_y=None) -> Emitter3D:
    R = RANGES["smoke"]
    return Emitter3D(pos=pos, radius=radius or _u(rng, R["radius"]), density_rate=scale * _u(rng, R["density_rate"]),
                     temperature=_u(rng, R["temperature"]), fuel_rate=0.0,
                     velocity=(_u(rng, RANGES["emitter_vel_xz"]), vel_y or _u(rng, R["vel_y"]),
                               _u(rng, RANGES["emitter_vel_xz"])))


def _pos(rng, xz=None, y=None):
    xz = xz or RANGES["emitter_xz"]
    return (_u(rng, xz), _u(rng, y or RANGES["emitter_y"]), _u(rng, xz))


def _far(p, others, d):
    return all(math.dist((p[0], p[2]), (q[0], q[2])) > d for q in others)


def _wind(rng, lo_hi, y=(0.0, 0.0)):
    a = _u(rng, (0, 2 * math.pi))
    m = _u(rng, lo_hi)
    return (m * math.cos(a), _u(rng, y), m * math.sin(a))


def sample_scenario(seed: int, split: str, frames: int = DEFAULT_FRAMES) -> Scenario3D:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    rng = np.random.default_rng([int(seed), SPLITS.index(split), 0x3D5EED])
    ood = split == "ood"
    factors: list[str] = []
    if ood:
        while not factors:
            factors = [n for n in ("box_obstacle", "strong_wind", "large_emitter")
                       if rng.random() < OOD_RANGES["factor_prob"]]
    kind = str(rng.choice(KINDS, p=KIND_PROBS))
    L = OOD_RANGES["large_emitter"]
    big = "large_emitter" in factors

    def em(make, p):
        if big:
            return make(rng, p, radius=_u(rng, L["radius"]), scale=_u(rng, L["rate_scale"]), vel_y=_u(rng, L["vel_y"]))
        return make(rng, p)

    emitters, obstacles, wind = [], [], (0.0, 0.0, 0.0)
    tags = [split, kind]
    if kind == "campfire":
        p = _pos(rng, xz=(0.35, 0.65))
        emitters.append(em(_fire, p))
        if rng.random() < RANGES["hearth_prob"]:
            H = RANGES["hearth"]
            hy = _u(rng, H["half_y"])
            obstacles.append(Obstacle3D("box", (p[0], hy, p[2]), (_u(rng, H["half_xz"]), hy, _u(rng, H["half_xz"]))))
            emitters[0].pos = (p[0], max(p[1], 2 * hy + 0.04), p[2])
            tags.append("hearth")
        if rng.random() < RANGES["wind_prob"]:
            wind = _wind(rng, (0.0, 0.25))
    elif kind == "plume_sphere":
        p = _pos(rng, xz=(0.35, 0.65))
        emitters.append(em(_smoke, p))
        S = RANGES["sphere"]
        r = _u(rng, S["radius"])
        c = (p[0] + _u(rng, (-0.08, 0.08)), _u(rng, (0.35, 0.6)), p[2] + _u(rng, (-0.08, 0.08)))
        obstacles.append(Obstacle3D("sphere", c, (r, r, r)))
    elif kind == "torches_wind":
        for _ in range(2):
            for _try in range(30):
                p = _pos(rng, xz=(0.22, 0.72))
                if _far(p, [e.pos for e in emitters], 0.2):
                    break
            emitters.append(em(_fire, p))
        wind = _wind(rng, (0.2, 0.5), y=(0.0, 0.0))
    else:   # mixed / random
        n = 2 if kind == "mixed" else int(rng.integers(1, 4))
        style = "mixed" if kind == "mixed" else str(rng.choice(["fire", "smoke", "mixed"]))
        tags.append(style)
        for i in range(n):
            for _try in range(30):
                p = _pos(rng, y=(0.06, 0.3) if kind == "random" else None)
                if _far(p, [e.pos for e in emitters], 0.15):
                    break
            make = {"fire": _fire, "smoke": _smoke}.get(style, _fire if i % 2 else _smoke)
            emitters.append(em(make, p))
        if rng.random() < RANGES["wind_prob"]:
            wind = _wind(rng, RANGES["wind_mild"])
    # spheres (any kind except plume_sphere which has its own; 0-2)
    if kind in ("random", "mixed", "torches_wind"):
        S = RANGES["sphere"]
        for _ in range(int(rng.integers(S["num"][0], S["num"][1] + 1))):
            for _try in range(30):
                r = _u(rng, S["radius"])
                c = (_u(rng, S["xz"]), _u(rng, S["y"]), _u(rng, S["xz"]))
                ok = all(math.dist(c, e.pos) > r + e.radius + 0.08 for e in emitters) and \
                    all(math.dist(c, o.pos) > r + o.size[0] + 0.03 for o in obstacles)
                if ok:
                    obstacles.append(Obstacle3D("sphere", c, (r, r, r)))
                    break
    if any(o.kind == "sphere" for o in obstacles):
        tags.append("sphere")
    # OOD factors
    if "box_obstacle" in factors:
        B = OOD_RANGES["box_obstacle"]
        for _ in range(int(rng.integers(B["num"][0], B["num"][1] + 1))):
            for _try in range(30):
                h = (_u(rng, B["half_xz"]), _u(rng, B["half_y"]), _u(rng, B["half_xz"]))
                c = (_u(rng, (0.25, 0.75)), _u(rng, B["y"]), _u(rng, (0.25, 0.75)))
                ok = all(c[1] - h[1] > e.pos[1] + e.radius + 0.08 for e in emitters) and \
                    all(math.dist(c, o.pos) > max(h) + max(o.size) + 0.03 for o in obstacles)
                if ok:
                    obstacles.append(Obstacle3D("box", c, h))
                    break
    if "strong_wind" in factors:
        W = OOD_RANGES["strong_wind"]
        wind = _wind(rng, W["abs"], y=W["y"])
    tags += factors
    has_fire = any(e.fuel_rate > 0 for e in emitters)
    has_smoke = any(e.fuel_rate == 0 for e in emitters)
    tags.append("fire+smoke" if has_fire and has_smoke else ("fire" if has_fire else "smoke"))

    impulses = []
    if rng.random() < RANGES["impulse_prob"]:
        I = RANGES["impulse"]
        every = int(rng.integers(I["every"][0], I["every"][1] + 1))
        a = _u(rng, (0, 2 * math.pi))
        s = _u(rng, I["speed"])
        c = (_u(rng, (0.3, 0.7)), _u(rng, (0.15, 0.5)), _u(rng, (0.3, 0.7)))
        impulses.append((every, int(rng.integers(0, every)),
                         Impulse3D(pos=c, radius=_u(rng, I["radius"]), velocity=(s * math.cos(a), 0.0, s * math.sin(a)))))
        tags.append("impulses")

    sim = {"density_dissipation": round(_u(rng, RANGES["density_dissipation"]), 4),
           "temperature_cooling": round(_u(rng, RANGES["temperature_cooling"]), 4)}
    warmup = int(rng.integers(WARMUP_RANGE[0], WARMUP_RANGE[1] + 1))
    tags.append("early" if warmup < 60 else "late")
    for e in emitters:
        e.pos = tuple(float(v) for v in e.pos)
        e.velocity = tuple(float(v) for v in e.velocity)
    sc = Scenario3D(seed=int(seed), split=split, kind=kind, emitters=emitters, obstacles=obstacles,
                    wind=tuple(float(v) for v in wind), sim=sim, warmup=warmup, frames=int(frames),
                    impulses=impulses, tags=tags)
    assert not is_preset_like(sc)
    return sc


def is_preset_like(sc: Scenario3D, tol: float = 1e-3) -> bool:
    """True if the scenario reproduces a sandbox preset (emitters + obstacles + wind all equal)."""
    from vfx.app3d.presets import PRESETS
    for p in PRESETS.values():
        if len(p.emitters) != len(sc.emitters) or len(p.obstacles) != len(sc.obstacles):
            continue
        same = all(np.allclose(a.pos, b.pos, atol=tol) and abs(a.radius - b.radius) < tol
                   for a, b in zip(p.emitters, sc.emitters))
        if same and np.allclose(p.wind, sc.wind, atol=tol):
            return True
    return False
