"""Randomized scenario sampling. Deterministic from (seed, split).

All quantities are world units (meters, m/s, m/s^2, K), see vfx.core.fields.
The ranges below are the single source of truth; build.py copies them into card.json
and README.md documents them.
"""
from __future__ import annotations

import math

import numpy as np

from vfx.core.fields import AMBIENT_TEMP, DOMAIN_SIZE, Emitter, Obstacle, Scenario

SPLITS = ("train", "test", "ood")
DEFAULT_FRAMES = 8

# Warm-up (steps simulated before the first saved frame). Spread uniformly so that both
# young plumes (just leaving the emitter) and developed ones (reaching the top) appear.
WARMUP_RANGE = (20, 140)

# --------------------------------------------------------------------- in-distribution
TRAIN_RANGES = {
    "num_emitters": (1, 3),
    "emitter_x": (0.2, 0.8),
    "emitter_y": (0.08, 0.30),
    "emitter_radius": (0.025, 0.06),
    "emitter_vel_x": (-0.2, 0.2),
    "emitter_vel_y": (0.0, 0.6),
    "fire_prob": 0.5,
    # fire: hot, burns fuel (the solver turns fuel into heat + smoke)
    "fire_temperature": (900.0, 1500.0),
    "fire_fuel_rate": (1.0, 4.0),
    "fire_density_rate": (0.1, 0.6),
    # smoke: warm, no fuel
    "smoke_temperature": (AMBIENT_TEMP + 30.0, AMBIENT_TEMP + 200.0),
    "smoke_density_rate": (0.5, 2.0),
    "wind_x": (-0.3, 0.3),              # m/s^2, mild
    "wind_y": (0.0, 0.0),
    "num_circle_obstacles": (0, 2),
    "circle_radius": (0.04, 0.10),
    "obstacle_x": (0.15, 0.85),
    "obstacle_y": (0.40, 0.80),
}

# --------------------------------------------------------------------- held-out (OOD)
# Scene types never seen in train/test. Each OOD scenario enables a random non-empty
# subset of these factors (each with prob 0.6), tagged by name.
OOD_RANGES = {
    "factor_prob": 0.6,
    "box_obstacle": {"num": (1, 2), "half_w": (0.04, 0.15), "half_h": (0.02, 0.08)},
    "strong_wind": {"wind_abs_x": (0.6, 1.2), "wind_y": (-0.3, 0.3)},
    "large_emitter": {"radius": (0.07, 0.11), "vel_y": (0.4, 1.0)},
}


def _u(rng: np.random.Generator, lo_hi) -> float:
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))


def _clear(pos, r, items, margin=0.03) -> bool:
    """True if a disk (pos, r) does not overlap any (pos, r) in items."""
    return all(math.hypot(pos[0] - p[0], pos[1] - p[1]) > r + q + margin for p, q in items)


def sample_scenario(seed: int, split: str, frames: int = DEFAULT_FRAMES) -> Scenario:
    """Deterministic random scene. num_steps = warm-up + frames (run() saves the last `frames`)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    rng = np.random.default_rng([int(seed), SPLITS.index(split), 0x5EED])
    R = TRAIN_RANGES
    ood = split == "ood"

    factors: list[str] = []
    if ood:
        names = ("box_obstacle", "strong_wind", "large_emitter")
        while not factors:
            factors = [n for n in names if rng.random() < OOD_RANGES["factor_prob"]]

    is_fire = bool(rng.random() < R["fire_prob"])
    radius_range = OOD_RANGES["large_emitter"]["radius"] if "large_emitter" in factors else R["emitter_radius"]
    vel_y_range = OOD_RANGES["large_emitter"]["vel_y"] if "large_emitter" in factors else R["emitter_vel_y"]

    emitters: list[Emitter] = []
    placed: list[tuple] = []
    n_em = int(rng.integers(R["num_emitters"][0], R["num_emitters"][1] + 1))
    for _ in range(n_em):
        for _try in range(30):
            r = _u(rng, radius_range)
            pos = (_u(rng, R["emitter_x"]), _u(rng, R["emitter_y"]))
            pos = (pos[0], max(pos[1], r + 0.02))               # keep inside the box
            if _clear(pos, r, placed, margin=0.0):
                break
        placed.append((pos, r))
        vel = (_u(rng, R["emitter_vel_x"]), _u(rng, vel_y_range))
        if is_fire:
            em = Emitter(pos=pos, radius=r, density_rate=_u(rng, R["fire_density_rate"]),
                         temperature=_u(rng, R["fire_temperature"]),
                         fuel_rate=_u(rng, R["fire_fuel_rate"]), velocity=vel)
        else:
            em = Emitter(pos=pos, radius=r, density_rate=_u(rng, R["smoke_density_rate"]),
                         temperature=_u(rng, R["smoke_temperature"]), fuel_rate=0.0, velocity=vel)
        emitters.append(em)

    obstacles: list[Obstacle] = []
    blockers = list(placed)
    n_circ = int(rng.integers(R["num_circle_obstacles"][0], R["num_circle_obstacles"][1] + 1))
    for _ in range(n_circ):
        for _try in range(30):
            r = _u(rng, R["circle_radius"])
            pos = (_u(rng, R["obstacle_x"]), _u(rng, R["obstacle_y"]))
            if _clear(pos, r, blockers):
                obstacles.append(Obstacle("circle", pos, (r, r)))
                blockers.append((pos, r))
                break
    if "box_obstacle" in factors:
        B = OOD_RANGES["box_obstacle"]
        for _ in range(int(rng.integers(B["num"][0], B["num"][1] + 1))):
            for _try in range(30):
                hw, hh = _u(rng, B["half_w"]), _u(rng, B["half_h"])
                pos = (_u(rng, R["obstacle_x"]), _u(rng, R["obstacle_y"]))
                if _clear(pos, math.hypot(hw, hh), blockers):
                    obstacles.append(Obstacle("box", pos, (hw, hh)))
                    blockers.append((pos, math.hypot(hw, hh)))
                    break

    if "strong_wind" in factors:
        W = OOD_RANGES["strong_wind"]
        wind = (float(rng.choice([-1.0, 1.0])) * _u(rng, W["wind_abs_x"]), _u(rng, W["wind_y"]))
    else:
        wind = (_u(rng, R["wind_x"]), _u(rng, R["wind_y"]))

    warmup = int(rng.integers(WARMUP_RANGE[0], WARMUP_RANGE[1] + 1))
    mid = 0.5 * (WARMUP_RANGE[0] + WARMUP_RANGE[1])
    tags = [split, "fire" if is_fire else "smoke", "early" if warmup < mid else "late"]
    if any(o.kind == "circle" for o in obstacles):
        tags.append("circle_obstacle")
    if ood:
        tags += factors                      # tags[0] == "ood" already

    assert all(0.0 < e.pos[0] < DOMAIN_SIZE and 0.0 < e.pos[1] < DOMAIN_SIZE for e in emitters)
    return Scenario(seed=int(seed), emitters=emitters, obstacles=obstacles, wind=wind,
                    num_steps=warmup + int(frames), tags=tags)


def scenario_split(scenario: Scenario) -> str:
    for s in SPLITS[::-1]:                 # "ood" wins over anything else
        if s in scenario.tags:
            return s
    return "train"
