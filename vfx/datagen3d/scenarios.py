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


def sample_scenario(seed: int, split: str, frames: int = DEFAULT_FRAMES, sampler: str = "v1") -> Scenario3D:
    """sampler "v1": the s3d_v1 distribution (unchanged); "v2": the s3d_v2 distribution (see sample_scenario_v2)."""
    if sampler == "v2":
        return sample_scenario_v2(seed, split, frames)
    if sampler != "v1":
        raise ValueError(f"unknown sampler {sampler!r}")
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


# ============================================================================ s3d_v2 distribution
# Motivation (vfx/train3d/EXPERIMENTS.md, runs 1-3 on s3d_v1): the screen-space net is data-limited and weakest
# on LARGE fires seen CLOSE UP. v2 shifts the mix toward large / hot / multiple fires, flame-dominated scenes and
# fire behind or around obstacles, keeps smoke-only scenes (~20 %), and covers all four sandbox preset kinds
# (incl. preset 4's mid-air slab, as a thin "slab"); the exact presets stay held out (`is_preset_like`).
V2_KINDS = ("campfire", "big_fire", "multi_fire", "fire_obstacle", "torches_wind", "mixed", "plume_sphere", "random")
V2_KIND_PROBS = (0.16, 0.16, 0.12, 0.12, 0.07, 0.11, 0.11, 0.15)
V2_RANGES = {
    "fire_size": {                     # radius (m), fuel_rate (/s), vel_y (m/s) per size class
        "small": {"radius": (0.03, 0.05), "fuel_rate": (4.0, 9.0), "vel_y": (0.15, 0.5)},
        "medium": {"radius": (0.05, 0.08), "fuel_rate": (6.0, 14.0), "vel_y": (0.2, 0.55)},
        "large": {"radius": (0.08, 0.115), "fuel_rate": (10.0, 22.0), "vel_y": (0.25, 0.75)},
    },
    "fire_size_probs": (0.2, 0.4, 0.4),                 # small / medium / large (generic fires)
    "fire_temperature": (1100.0, 1550.0),              # K (v1 1150-1400; sandbox 1300)
    "fire_density_rate": (0.0, 0.3), "sooty_prob": 0.2, "sooty_density_rate": (0.3, 0.8),
    "smoke": {"radius": (0.04, 0.11), "temperature": (500.0, 750.0), "density_rate": (1.5, 5.5), "vel_y": (0.3, 0.9)},
    "occluder": {"num": (1, 3), "half_long": (0.04, 0.15), "half_thin": (0.02, 0.06), "half_y": (0.06, 0.22),
                 "gap": (0.05, 0.2)},           # floor-standing walls / pillars / crates beside a fire
    "slab_prob": 0.35,                          # mixed kind: thin mid-air slab above the emitters (preset-4 kind)
    "slab": {"half_xz": (0.08, 0.18), "half_y": (0.02, 0.04), "y": (0.4, 0.65)},
}
V2_OOD_RANGES = {                               # beyond every v2 train range (tags keep the v1 OOD names)
    "factor_prob": 0.6,
    "box_obstacle": {"num": (1, 2), "half_xz": (0.05, 0.22), "half_y": (0.045, 0.09), "y": (0.3, 0.7)},
    "strong_wind": {"abs": (0.8, 1.6), "y": (-0.2, 0.2)},
    "large_emitter": {"radius": (0.12, 0.16), "fuel_rate": (18.0, 30.0), "smoke_rate": (5.5, 9.0), "vel_y": (0.6, 1.1)},
}
LARGE_FIRE_RADIUS = 0.08                        # tag "large_fire": a fire emitter with radius >= this (v1 train max 0.075)


def _fire2(rng, pos, size=None, big_ood=False) -> Emitter3D:
    R = V2_RANGES
    if big_ood:
        L = V2_OOD_RANGES["large_emitter"]
        rad, fuel, vy = _u(rng, L["radius"]), _u(rng, L["fuel_rate"]), _u(rng, L["vel_y"])
    else:
        size = size or str(rng.choice(("small", "medium", "large"), p=R["fire_size_probs"]))
        S = R["fire_size"][size]
        rad, fuel, vy = _u(rng, S["radius"]), _u(rng, S["fuel_rate"]), _u(rng, S["vel_y"])
    dens = _u(rng, R["sooty_density_rate"]) if rng.random() < R["sooty_prob"] else _u(rng, R["fire_density_rate"])
    return Emitter3D(pos=pos, radius=rad, density_rate=dens, temperature=_u(rng, R["fire_temperature"]),
                     fuel_rate=fuel, velocity=(_u(rng, RANGES["emitter_vel_xz"]), vy, _u(rng, RANGES["emitter_vel_xz"])))


def _smoke2(rng, pos, big_ood=False) -> Emitter3D:
    S = V2_RANGES["smoke"]
    if big_ood:
        L = V2_OOD_RANGES["large_emitter"]
        rad, rate, vy = _u(rng, L["radius"]), _u(rng, L["smoke_rate"]), _u(rng, L["vel_y"])
    else:
        rad, rate, vy = _u(rng, S["radius"]), _u(rng, S["density_rate"]), _u(rng, S["vel_y"])
    return Emitter3D(pos=pos, radius=rad, density_rate=rate, temperature=_u(rng, S["temperature"]), fuel_rate=0.0,
                     velocity=(_u(rng, RANGES["emitter_vel_xz"]), vy, _u(rng, RANGES["emitter_vel_xz"])))


def _box_point_dist_xz(c, h, p) -> float:
    dx = max(abs(p[0] - c[0]) - h[0], 0.0)
    dz = max(abs(p[2] - c[2]) - h[2], 0.0)
    return math.hypot(dx, dz)


def _place(rng, emitters, xz, y, min_gap, radius_of_new=0.0):
    p = None
    for _try in range(40):
        p = _pos(rng, xz=xz, y=y)
        if all(math.dist((p[0], p[2]), (e.pos[0], e.pos[2])) > min_gap + 0.5 * (e.radius + radius_of_new)
               for e in emitters):
            break
    return p


def scene_tags(sc) -> list[str]:
    """Derived content tags (also computed for imported v1 scenarios). `sc` is a Scenario3D or its asdict()."""
    d = sc if isinstance(sc, dict) else sc.asdict()
    em = d["emitters"]
    fires = [e for e in em if e["fuel_rate"] > 0]
    t = []
    if any(e["radius"] >= LARGE_FIRE_RADIUS for e in fires):
        t.append("large_fire")
    if len(fires) >= 2:
        t.append("multi_fire")
    if fires and not any(e["fuel_rate"] == 0 for e in em):
        t.append("fire_only")
    if not fires:
        t.append("smoke_only")
    if any(e["temperature"] >= 1450 for e in fires):
        t.append("hot_fire")
    if any(e["density_rate"] > 0.3 for e in fires):
        t.append("sooty_fire")
    if any(o["kind"] == "box" and o["pos"][1] - o["size"][1] < 1e-3 and o["size"][1] >= 0.05 for o in d["obstacles"]):
        t.append("occluder")
    return t


def sample_scenario_v2(seed: int, split: str, frames: int = DEFAULT_FRAMES) -> Scenario3D:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    rng = np.random.default_rng([int(seed), SPLITS.index(split), 0x3D5EED, 2])
    R = V2_RANGES
    ood = split == "ood"
    factors: list[str] = []
    if ood:
        while not factors:
            factors = [n for n in ("box_obstacle", "strong_wind", "large_emitter")
                       if rng.random() < V2_OOD_RANGES["factor_prob"]]
    big = "large_emitter" in factors
    kind = str(rng.choice(V2_KINDS, p=V2_KIND_PROBS))
    fire = lambda p, size=None: _fire2(rng, p, size, big_ood=big)
    smoke = lambda p: _smoke2(rng, p, big_ood=big)
    emitters, obstacles, wind = [], [], (0.0, 0.0, 0.0)
    tags = [split, kind]

    def hearth(e, prob):
        if rng.random() < prob:
            H = RANGES["hearth"]
            hy = _u(rng, H["half_y"])
            hx = max(_u(rng, H["half_xz"]), e.radius + 0.02)
            hz = max(_u(rng, H["half_xz"]), e.radius + 0.02)
            obstacles.append(Obstacle3D("box", (e.pos[0], hy, e.pos[2]), (hx, hy, hz)))
            e.pos = (e.pos[0], max(e.pos[1], 2 * hy + 0.04), e.pos[2])
            tags.append("hearth")

    if kind == "campfire":
        emitters.append(fire(_pos(rng, xz=(0.35, 0.65)), str(rng.choice(("small", "medium", "large"), p=(0.1, 0.4, 0.5)))))
        hearth(emitters[0], 0.5)
        if rng.random() < 0.5:
            wind = _wind(rng, (0.0, 0.3))
    elif kind == "big_fire":
        emitters.append(fire(_pos(rng, xz=(0.4, 0.6)), "large"))
        hearth(emitters[0], 0.3)
        if rng.random() < 0.4:
            wind = _wind(rng, (0.0, 0.35))
    elif kind == "multi_fire":
        for _ in range(int(rng.integers(2, 5))):
            size = str(rng.choice(("small", "medium", "large"), p=(0.3, 0.4, 0.3)))
            emitters.append(fire(_place(rng, emitters, (0.25, 0.75), None, 0.07, 0.07), size))
        if rng.random() < 0.5:
            wind = _wind(rng, RANGES["wind_mild"])
    elif kind == "fire_obstacle":
        for _ in range(int(rng.integers(1, 3))):
            emitters.append(fire(_place(rng, emitters, (0.3, 0.7), (0.06, 0.12), 0.15, 0.08),
                                 str(rng.choice(("medium", "large"), p=(0.4, 0.6)))))
        O = R["occluder"]
        for _ in range(int(rng.integers(O["num"][0], O["num"][1] + 1))):
            for _try in range(40):
                e = emitters[int(rng.integers(0, len(emitters)))]
                hl, ht, hy = _u(rng, O["half_long"]), _u(rng, O["half_thin"]), _u(rng, O["half_y"])
                h = (hl, hy, ht) if rng.random() < 0.5 else (ht, hy, hl)
                a = _u(rng, (0, 2 * math.pi))
                dd = e.radius + _u(rng, O["gap"]) + 0.5 * (h[0] + h[2])
                c = (e.pos[0] + dd * math.cos(a), hy, e.pos[2] + dd * math.sin(a))
                if not (0.06 + h[0] < c[0] < 0.94 - h[0] and 0.06 + h[2] < c[2] < 0.94 - h[2]):
                    continue
                if all(_box_point_dist_xz(c, h, q.pos) > q.radius + 0.03 for q in emitters) and \
                        all(_box_point_dist_xz(c, h, o.pos) > max(o.size[0], o.size[2]) + 0.02 for o in obstacles):
                    obstacles.append(Obstacle3D("box", c, h))
                    break
        if rng.random() < 0.4:            # flow around an obstacle above / beside the fire
            r = _u(rng, RANGES["sphere"]["radius"])
            e = emitters[0]
            c = (e.pos[0] + _u(rng, (-0.1, 0.1)), _u(rng, (0.32, 0.55)), e.pos[2] + _u(rng, (-0.1, 0.1)))
            if all(math.dist(c, q.pos) > r + q.radius + 0.08 for q in emitters):
                obstacles.append(Obstacle3D("sphere", c, (r, r, r)))
        if rng.random() < 0.4:
            wind = _wind(rng, (0.0, 0.35))
    elif kind == "torches_wind":
        for _ in range(2):
            emitters.append(fire(_place(rng, emitters, (0.22, 0.72), None, 0.17),
                                 str(rng.choice(("small", "medium", "large"), p=(0.3, 0.5, 0.2)))))
        wind = _wind(rng, (0.2, 0.5))
    elif kind == "mixed":
        emitters.append(smoke(_place(rng, emitters, (0.3, 0.7), None, 0.12)))
        emitters.append(fire(_place(rng, emitters, (0.3, 0.7), None, 0.12)))
        if rng.random() < 0.5:
            emitters.reverse()
        if rng.random() < R["slab_prob"]:
            S = R["slab"]
            for _try in range(30):
                h = (_u(rng, S["half_xz"]), _u(rng, S["half_y"]), _u(rng, S["half_xz"]))
                cx = float(np.mean([e.pos[0] for e in emitters])) + _u(rng, (-0.08, 0.08))
                cz = float(np.mean([e.pos[2] for e in emitters])) + _u(rng, (-0.08, 0.08))
                c = (cx, _u(rng, S["y"]), cz)
                if all(c[1] - h[1] > e.pos[1] + e.radius + 0.12 for e in emitters):
                    obstacles.append(Obstacle3D("box", c, h))
                    tags.append("slab")
                    break
        if rng.random() < RANGES["wind_prob"]:
            wind = _wind(rng, RANGES["wind_mild"])
    elif kind == "plume_sphere":
        e = smoke(_pos(rng, xz=(0.35, 0.65)))
        emitters.append(e)
        r = _u(rng, RANGES["sphere"]["radius"])
        c = (e.pos[0] + _u(rng, (-0.08, 0.08)), _u(rng, (0.35, 0.6)), e.pos[2] + _u(rng, (-0.08, 0.08)))
        obstacles.append(Obstacle3D("sphere", c, (r, r, r)))
    else:   # random
        n = int(rng.integers(1, 4))
        style = str(rng.choice(["fire", "smoke", "mixed"], p=(0.35, 0.4, 0.25)))
        tags.append(style)
        for i in range(n):
            p = _place(rng, emitters, None, (0.06, 0.3), 0.12)
            emitters.append(fire(p) if style == "fire" or (style == "mixed" and i % 2) else smoke(p))
        if rng.random() < RANGES["wind_prob"]:
            wind = _wind(rng, RANGES["wind_mild"])
    if kind in ("random", "mixed", "torches_wind", "multi_fire"):
        S = RANGES["sphere"]
        for _ in range(int(rng.integers(0, 3 if kind == "random" else 2))):
            for _try in range(30):
                r = _u(rng, S["radius"])
                c = (_u(rng, S["xz"]), _u(rng, S["y"]), _u(rng, S["xz"]))
                if all(math.dist(c, e.pos) > r + e.radius + 0.08 for e in emitters) and \
                        all(math.dist(c, o.pos) > r + max(o.size) + 0.03 for o in obstacles):
                    obstacles.append(Obstacle3D("sphere", c, (r, r, r)))
                    break
    if any(o.kind == "sphere" for o in obstacles):
        tags.append("sphere")
    if "box_obstacle" in factors:
        B = V2_OOD_RANGES["box_obstacle"]
        for _ in range(int(rng.integers(B["num"][0], B["num"][1] + 1))):
            for _try in range(30):
                h = (_u(rng, B["half_xz"]), _u(rng, B["half_y"]), _u(rng, B["half_xz"]))
                c = (_u(rng, (0.25, 0.75)), _u(rng, B["y"]), _u(rng, (0.25, 0.75)))
                if all(c[1] - h[1] > e.pos[1] + e.radius + 0.08 for e in emitters) and \
                        all(math.dist(c, o.pos) > max(h) + max(o.size) + 0.03 for o in obstacles):
                    obstacles.append(Obstacle3D("box", c, h))
                    break
    if "strong_wind" in factors:
        wind = _wind(rng, V2_OOD_RANGES["strong_wind"]["abs"], y=V2_OOD_RANGES["strong_wind"]["y"])
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
    sc.tags += [t for t in scene_tags(sc) if t not in sc.tags]
    assert not is_preset_like(sc)
    return sc
