"""3D Layer 1: solver correctness (SPEC "Testing and validation"), via the vfx.sim3d public interface only.

Written from SPEC.md + CLAUDE.md ("3D phase") + vfx/core/fields3d.py, not from the solver code.
Spec thresholds (unchanged from 2D):
  * normalized divergence after projection < 1e-4 everywhere (fluid cells)
  * density mass conserved within 0.1% over 1000 steps, closed box, no sources, dissipation 0
  * pure advection matches the analytic result; buoyancy lifts hot fluid; walls/obstacles hold
  * determinism: bit-identical; convergence: error vs a fine reference shrinks with resolution
Tolerances this file states itself (no spec number exists):
  * analytic translation: centroid < 0.5 cell, rel L2 < 0.15 (as the 2D gate); the blob
    (sigma ~6 cells) is kept clear of the walls so the unbounded-domain analytic answer applies
  * buoyancy "not sideways": mean |vx|,|vz| in the hot parcel < 10% of mean vy
  * 64^3 vs 128^3 large-scale agreement: smoke mass and heat within 10%, plume centroid height within
    0.04 m, plume top within 0.0625 m (4 cells at 64^3)
  * 1000-step stress: finite, |u| < 100 m/s, density < 1e3, T < 1e4 K
Base resolution 64^3 (the spec's runtime grid) to keep the file fast.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from vfx.core.fields3d import (
    AMBIENT_TEMP, DOMAIN_SIZE, NUM_CHANNELS_3D, SIM_DT, Emitter3D, Forces3D, Impulse3D,
    Obstacle3D, rasterize_obstacles_3d,
)
from tests3d.helpers3d import (
    DEV, blob, check_finite, fluid_mask, import_or_skip, make_config, mass, requires_cuda,
    smooth_noise, world_grid,
)

pytestmark = requires_cuda
RES = 64
DIV_TOL = 1e-4          # SPEC Layer 1
MASS_TOL = 1e-3         # SPEC: 0.1%


@pytest.fixture(scope="module")
def sim():
    return import_or_skip("vfx.sim3d", ("Solver3D", "SolverConfig3D", "SOLVER3D_VERSION"))


def no_obstacles():
    return []


def sphere_obstacle():
    return [Obstacle3D(kind="sphere", pos=(0.5, 0.45, 0.5), size=(0.14, 0.14, 0.14))]


def box_obstacle():
    return [Obstacle3D(kind="box", pos=(0.3, 0.7, 0.6), size=(0.1, 0.05, 0.08))]


# ----------------------------------------------------------------------------- interface
def test_interface_reset_contract(sim):
    s = sim.Solver3D(RES)
    st = s.reset()
    assert st.data.shape == (NUM_CHANNELS_3D, RES, RES, RES)
    assert st.data.dtype == torch.float32 and st.data.is_cuda
    assert st.solid.shape == (RES, RES, RES)
    assert tuple(st.meta.res) == (RES, RES, RES)
    assert math.isclose(st.meta.dx, DOMAIN_SIZE / RES)
    assert torch.all(st.ch("density") == 0)
    assert torch.all(st.velocity == 0)
    assert torch.all(st.ch("fuel") == 0)
    assert torch.allclose(st.ch("temperature"), torch.full_like(st.ch("temperature"), AMBIENT_TEMP))
    assert torch.all(st.solid == 0), "box walls must NOT be marked in solid (fields3d.py)"
    assert isinstance(sim.SOLVER3D_VERSION, str) and sim.SOLVER3D_VERSION
    cfg = sim.SolverConfig3D()
    for name in ("buoyancy_alpha", "buoyancy_beta", "vorticity_eps", "density_dissipation",
                 "temperature_cooling", "burn_rate", "ignition_temp", "burn_heat",
                 "smoke_from_fuel", "pressure_tol", "pressure_max_iters",
                 "heat_diffusion", "confinement_floor"):
        assert hasattr(cfg, name), f"SolverConfig3D missing {name}"


def test_reset_rasterizes_obstacles(sim):
    obs = sphere_obstacle() + box_obstacle()
    st = sim.Solver3D(RES).reset(obs)
    assert torch.equal(st.solid, rasterize_obstacles_3d(obs, RES, DEV))


def test_step_contract(sim):
    s = sim.Solver3D(RES)
    st = s.reset()
    f = Forces3D(emitters=[Emitter3D(pos=(0.5, 0.2, 0.5), radius=0.08, density_rate=1.0, temperature=800.0)])
    out = s.step(st, f, SIM_DT)
    assert out.data.shape == (NUM_CHANNELS_3D, RES, RES, RES) and out.data.is_cuda
    assert out.ch("density").sum().item() > 0, "emitter added no smoke"
    assert out.ch("density").min().item() >= -1e-6, "negative density"
    check_finite(out)


# ----------------------------------------------------------------------------- divergence
TIGHT = dict(pressure_tol=1e-7, pressure_max_iters=5000)


def _project_noise(sim, res, obstacles, **cfg):
    s = sim.Solver3D(res, make_config(sim, **cfg) if cfg else None)
    st = s.reset(obstacles)
    v = smooth_noise(res, 3, seed=1, amp=2.0) + torch.tensor([0.8, -0.3, 0.4], device=DEV)[:, None, None, None]
    st.velocity.copy_(v * fluid_mask(st))
    before = s.divergence(st).max().item()
    st = s.project(st)
    return s, st, before


@pytest.mark.parametrize("obstacles", [no_obstacles, sphere_obstacle], ids=["open", "sphere"])
def test_divergence_after_projection(sim, obstacles):
    """SPEC: < 1e-4 after projection, with the DEFAULT config (what the runtime uses)."""
    s, st, before = _project_noise(sim, RES, obstacles())
    div = s.divergence(st)
    assert div.shape == (RES, RES, RES)
    assert torch.all(div[st.solid > 0.5] == 0), "divergence() must be 0 in solid cells"
    check_finite(st, "after project")
    m = div.max().item()
    print(f"64^3 {obstacles.__name__}: normalized div before {before:.3e} after {m:.3e}")
    assert before > 1e-2, "test field should start strongly divergent"
    assert m < DIV_TOL, f"max normalized divergence {m:.3e} >= {DIV_TOL} (default pressure_tol/max_iters)"


@pytest.mark.parametrize("obstacles", [no_obstacles, sphere_obstacle], ids=["open", "sphere"])
def test_divergence_converged_solve(sim, obstacles):
    """Same threshold with a tight solve: separates the discretization from the iteration budget."""
    s, st, _ = _project_noise(sim, RES, obstacles(), **TIGHT)
    m = s.divergence(st).max().item()
    print(f"64^3 {obstacles.__name__} tight solve: max normalized div {m:.3e}")
    assert m < DIV_TOL


@pytest.mark.parametrize("obstacles", [no_obstacles, sphere_obstacle], ids=["open", "sphere"])
def test_divergence_after_full_steps(sim, obstacles):
    s = sim.Solver3D(RES)
    st = s.reset(obstacles())
    f = Forces3D(emitters=[Emitter3D(pos=(0.5, 0.15, 0.5), radius=0.07, density_rate=2.0,
                                     temperature=900.0, velocity=(0.0, 1.0, 0.0))])
    for _ in range(60):
        st = s.step(st, f, SIM_DT)
    m = s.divergence(st).max().item()
    print(f"{obstacles.__name__}: normalized div after 60 steps {m:.3e}")
    assert m < DIV_TOL


def test_projection_removes_flow_into_walls(sim):
    """u = grad(x + 0.5y - 0.3z) is a pure gradient: its no-penetration projection in the closed box is
    zero. Residual |u| and the net flux through every x/y/z slab must be < 1% of the input speed."""
    s = sim.Solver3D(RES)
    st = s.reset()
    st.velocity[0].fill_(1.0)
    st.velocity[1].fill_(0.5)
    st.velocity[2].fill_(-0.3)
    st = s.project(st)
    check_finite(st, "after project")
    u = st.velocity
    resid = u.abs().max().item()
    fx = u[0].mean(dim=(0, 1)).abs().max().item()     # per x-slab (mean over z,y)
    fy = u[1].mean(dim=(0, 2)).abs().max().item()
    fz = u[2].mean(dim=(1, 2)).abs().max().item()
    print(f"max|u| after {resid:.3e}, net slab flux x {fx:.3e} y {fy:.3e} z {fz:.3e}")
    assert resid < 1e-2, f"projection left |u| = {resid:.3e} of a pure-gradient (into-wall) flow"
    assert max(fx, fy, fz) < 1e-2


# ----------------------------------------------------------------------------- mass
def test_mass_conserved_closed_box(sim):
    s = sim.Solver3D(RES, make_config(sim, density_dissipation=0.0))
    st = s.reset()
    st.ch("density").copy_(blob(RES, 0.5, 0.35, 0.5, 0.08) + 0.5 * blob(RES, 0.3, 0.7, 0.6, 0.06))
    st.ch("temperature").copy_(AMBIENT_TEMP + 400.0 * blob(RES, 0.5, 0.35, 0.5, 0.08))
    st.velocity.copy_(smooth_noise(RES, 3, seed=3, amp=0.5))          # stir it, including into walls
    m0 = mass(st)
    for _ in range(1000):
        st = s.step(st, Forces3D(), SIM_DT)
    check_finite(st, "after 1000 steps")
    rel = abs(mass(st) - m0) / m0
    print(f"64^3 closed box: mass {m0:.5f} -> {mass(st):.5f} (rel change {rel:.3e})")
    assert rel < MASS_TOL, f"mass changed by {rel * 100:.3f}% (> 0.1%)"


def test_mass_conserved_with_obstacle(sim):
    s = sim.Solver3D(RES, make_config(sim, density_dissipation=0.0))
    st = s.reset(sphere_obstacle())
    st.ch("density").copy_(blob(RES, 0.5, 0.15, 0.5, 0.07) * fluid_mask(st))
    st.ch("temperature").copy_(AMBIENT_TEMP + 500.0 * blob(RES, 0.5, 0.15, 0.5, 0.07))
    m0 = mass(st)
    for _ in range(1000):
        st = s.step(st, Forces3D(), SIM_DT)
    check_finite(st, "after 1000 steps")
    rel = abs(mass(st) - m0) / m0
    print(f"64^3 sphere: rel mass change {rel:.3e}")
    assert rel < MASS_TOL, f"mass changed by {rel * 100:.3f}% (> 0.1%)"


# ----------------------------------------------------------------------------- advection
def test_advect_integer_shift_is_exact(sim):
    """u*dt = whole cells -> any consistent semi-Lagrangian/MacCormack scheme is exact in the interior."""
    s = sim.Solver3D(RES)
    dx = DOMAIN_SIZE / RES
    f0 = blob(RES, 0.3, 0.35, 0.55, 0.07)                 # stays >= 4 sigma clear of every wall
    vel = torch.zeros(3, RES, RES, RES, device=DEV)
    vel[0], vel[1], vel[2] = 2 * dx / SIM_DT, 1 * dx / SIM_DT, -1 * dx / SIM_DT
    f = f0.clone()
    n = 10
    for _ in range(n):
        f = s.advect(f, vel, SIM_DT)
    assert f.shape == f0.shape
    expect = blob(RES, 0.3 + 2 * n * dx, 0.35 + n * dx, 0.55 - n * dx, 0.07)
    # "interior": cells whose back-traced origin after n steps is still inside the box (cells near the
    # upstream walls are filled by the inflow boundary value, which an unbounded analytic shift does not see)
    err = (f - expect)[: RES - n - 1, n + 1:, 2 * n + 1:].abs().max().item()
    print(f"integer-shift advection max err {err:.2e}")
    assert err < 1e-4


def test_advect_matches_analytic_translation(sim):
    """Fractional-cell constant velocity (0.31/0.17/0.12 cells per step): the blob centroid follows
    x0 + u t (< 0.5 cell) and the shape stays within rel L2 0.15 of the analytic blob."""
    s = sim.Solver3D(RES)
    dx = DOMAIN_SIZE / RES
    u = (0.15, 0.08, 0.06)
    sig = 0.09                                      # ~6 cells; >= 3.9 sigma from every wall throughout
    c0 = (0.35, 0.38, 0.40)
    f = blob(RES, *c0, sig)
    vel = torch.zeros(3, RES, RES, RES, device=DEV)
    vel[0], vel[1], vel[2] = u
    n = 45                                          # 1.5 s
    for _ in range(n):
        f = s.advect(f, vel, SIM_DT)
    T = n * SIM_DT
    ce = [c0[i] + u[i] * T for i in range(3)]
    ex = blob(RES, *ce, sig)
    xx, yy, zz = world_grid(RES)
    m = f.sum()
    cen = [((f * g).sum() / m).item() for g in (xx, yy, zz)]
    # the analytic blob, truncated by the box, also has a centroid != its centre: compare like with like
    cen_ex = [((ex * g).sum() / ex.sum()).item() for g in (xx, yy, zz)]
    cerr = math.dist(cen, cen_ex) / dx
    rel_l2 = ((f - ex).norm() / ex.norm()).item()
    print(f"centroid err {cerr:.3f} cells, rel L2 {rel_l2:.3f}, peak {f.max().item():.3f} (analytic 1.0)")
    assert cerr < 0.5
    assert rel_l2 < 0.15


def test_advect_multichannel_consistent(sim):
    s = sim.Solver3D(32)
    f = torch.stack([blob(32, 0.4, 0.5, 0.5, 0.1), 2 * blob(32, 0.6, 0.5, 0.4, 0.08)])
    vel = smooth_noise(32, 3, seed=5, amp=0.3)
    out = s.advect(f, vel, SIM_DT)
    assert out.shape == f.shape
    for c in range(2):
        assert torch.allclose(out[c], s.advect(f[c].contiguous(), vel, SIM_DT), atol=1e-6)


def test_advection_converges_with_resolution(sim):
    """Analytic translation error shrinks as resolution increases (same world-space problem)."""
    errs = {}
    u = (0.11, 0.05, 0.07)
    for res in (32, 48, 64, 96):
        s = sim.Solver3D(res)
        vel = torch.zeros(3, res, res, res, device=DEV)
        vel[0], vel[1], vel[2] = u
        f = blob(res, 0.35, 0.35, 0.35, 0.1)
        n = 30
        for _ in range(n):
            f = s.advect(f, vel, SIM_DT)
        ex = blob(res, *(0.35 + u[i] * n * SIM_DT for i in range(3)), 0.1)
        errs[res] = ((f - ex).abs().mean() / ex.abs().mean()).item()
    print("advection rel L1 vs analytic:", {k: round(v, 4) for k, v in errs.items()})
    r = sorted(errs)
    for a, b in zip(r, r[1:]):
        assert errs[b] < errs[a], errs


# ----------------------------------------------------------------------------- buoyancy
def test_buoyancy_lifts_hot_fluid_straight_up(sim):
    cfg = make_config(sim, buoyancy_alpha=0.0, vorticity_eps=0.0, burn_rate=0.0)
    if getattr(cfg, "buoyancy_beta", 0) <= 0:
        pytest.fail("default buoyancy_beta must be > 0 (heat lifts)")
    s = sim.Solver3D(RES, cfg)
    st = s.reset()
    hot = blob(RES, 0.5, 0.3, 0.5, 0.08)
    st.ch("temperature").copy_(AMBIENT_TEMP + 500.0 * hot)
    st.ch("density").copy_(hot)                              # passive tracer of the hot parcel
    xx, yy, zz = world_grid(RES)

    def centroid():
        d = st.ch("density")
        return [((d * g).sum() / d.sum()).item() for g in (xx, yy, zz)]

    c0 = centroid()
    for _ in range(20):
        st = s.step(st, Forces3D(), SIM_DT)
    c1 = centroid()
    w = hot / hot.sum()
    vx, vy, vz = [(st.velocity[i] * w).sum().item() for i in range(3)]
    print(f"hot parcel centroid {['%.3f' % c for c in c0]} -> {['%.3f' % c for c in c1]}; "
          f"mean v in parcel ({vx:.4f}, {vy:.4f}, {vz:.4f}) m/s")
    assert vy > 0, "hot fluid must accelerate up (+y)"
    assert c1[1] > c0[1] + 0.01, "hot parcel did not rise"
    assert abs(vx) < 0.1 * vy and abs(vz) < 0.1 * vy, "symmetric hot parcel drifted sideways"
    assert abs(c1[0] - c0[0]) < 0.1 * (c1[1] - c0[1]) + 1e-3
    assert abs(c1[2] - c0[2]) < 0.1 * (c1[1] - c0[1]) + 1e-3


def test_dense_cold_fluid_sinks(sim):
    base = sim.SolverConfig3D()
    alpha = base.buoyancy_alpha if base.buoyancy_alpha > 0 else 1.0
    s = sim.Solver3D(RES, make_config(sim, buoyancy_beta=0.0, buoyancy_alpha=alpha, vorticity_eps=0.0,
                                      burn_rate=0.0))
    st = s.reset()
    d0 = 2.0 * blob(RES, 0.5, 0.7, 0.5, 0.08)
    st.ch("density").copy_(d0)
    for _ in range(20):
        st = s.step(st, Forces3D(), SIM_DT)
    vy = (st.velocity[1] * d0).sum() / d0.sum()
    print(f"dense parcel mean vy {vy.item():.4f} m/s (alpha={alpha})")
    assert vy.item() < 0


def test_ambient_fluid_stays_at_rest(sim):
    s = sim.Solver3D(RES)
    st = s.reset([Obstacle3D("box", (0.5, 0.5, 0.5), (0.1, 0.1, 0.1))])
    for _ in range(50):
        st = s.step(st, Forces3D(), SIM_DT)
    assert st.velocity.abs().max().item() < 1e-5
    assert st.ch("density").abs().max().item() < 1e-6


# ----------------------------------------------------------------------------- boundaries
def test_no_flow_into_obstacles(sim):
    """Jets blown at a box and a sphere: nothing enters them (density/fuel 0, velocity ~0 inside)."""
    s = sim.Solver3D(RES)
    obs = [Obstacle3D("box", (0.5, 0.55, 0.5), (0.18, 0.06, 0.18)),
           Obstacle3D("sphere", (0.8, 0.3, 0.5), (0.1, 0.1, 0.1))]
    st = s.reset(obs)
    f = Forces3D(emitters=[Emitter3D(pos=(0.5, 0.2, 0.5), radius=0.07, density_rate=3.0, temperature=1200.0,
                                     fuel_rate=1.0, velocity=(0.0, 2.0, 0.0)),
                           Emitter3D(pos=(0.5, 0.3, 0.5), radius=0.05, density_rate=2.0, velocity=(3.0, 0.0, 0.0))])
    # cells at least one cell deep inside the solid (the first layer may hold interpolated BC values)
    deep = (-F.max_pool3d(-st.solid[None, None], 3, 1, 1)[0, 0]) > 0.5
    assert deep.sum().item() > 0
    for _ in range(150):
        st = s.step(st, f, SIM_DT)
    umax = st.velocity.abs().max().item()
    d_in = st.ch("density")[deep].abs().max().item()
    fuel_in = st.ch("fuel")[deep].abs().max().item()
    u_in = st.velocity[:, deep].abs().max().item()
    print(f"umax {umax:.3f}; inside solid: |u| {u_in:.2e}, density {d_in:.2e}, fuel {fuel_in:.2e}")
    assert umax > 0.1, "jet should be moving"
    assert u_in <= 1e-3 * umax, "velocity inside a static obstacle must be ~0"
    assert d_in < 1e-3 and fuel_in < 1e-3, "smoke/fuel leaked into the obstacle"


def test_no_leak_through_box_walls(sim):
    """Strong flow aimed at walls and corners, dissipation 0, no sources: no density lost."""
    s = sim.Solver3D(RES, make_config(sim, density_dissipation=0.0))
    st = s.reset()
    st.ch("density").copy_(blob(RES, 0.85, 0.85, 0.85, 0.06) + blob(RES, 0.12, 0.1, 0.15, 0.05))
    m0 = mass(st)
    for i in range(150):
        imp = [Impulse3D(pos=(0.85, 0.85, 0.85), radius=0.1, velocity=(3.0, 3.0, 3.0)),
               Impulse3D(pos=(0.12, 0.1, 0.15), radius=0.1, velocity=(-3.0, -3.0, -3.0))] if i % 10 == 0 else []
        st = s.step(st, Forces3D(impulses=imp), SIM_DT)
    check_finite(st)
    rel = abs(mass(st) - m0) / m0
    print(f"wall-jet rel mass change {rel:.3e}")
    assert rel < MASS_TOL


# ----------------------------------------------------------------------------- determinism
def _scripted_run(sim, steps=120, res=RES):
    s = sim.Solver3D(res)
    st = s.reset(sphere_obstacle())
    for i in range(steps):
        f = Forces3D(emitters=[Emitter3D(pos=(0.5, 0.12, 0.5), radius=0.06, density_rate=2.0, temperature=1300.0,
                                         fuel_rate=1.5, velocity=(0.0, 1.0, 0.0))],
                     impulses=[Impulse3D(pos=(0.3, 0.5, 0.4), radius=0.1, velocity=(2.0, 0.5, -1.0))]
                     if i % 17 == 0 else [],
                     wind=(0.2, 0.0, 0.1))
        st = s.step(st, f, SIM_DT)
    return st.data.clone()


def test_determinism_bit_identical(sim):
    a = _scripted_run(sim)
    b = _scripted_run(sim)
    assert a.abs().sum().item() > 0
    diff = (a != b).sum().item()
    assert diff == 0, f"{diff} values differ between identical runs (max abs {(a - b).abs().max().item():.3e})"


# ----------------------------------------------------------------------------- convergence
PLUME = Forces3D(emitters=[Emitter3D(pos=(0.5, 0.2, 0.5), radius=0.1, density_rate=1.0, temperature=600.0)])


def _plume(sim, res, steps, forces=PLUME):
    s = sim.Solver3D(res)
    st = s.reset()
    for _ in range(steps):
        st = s.step(st, forces, SIM_DT)
    check_finite(st, f"res {res}")
    return st


def test_convergence_against_fine_reference(sim):
    """Short smooth plume: error vs a 96^3 reference (compared on a common 16^3 grid) shrinks with res."""
    steps = 20
    pool = lambda st: F.adaptive_avg_pool3d(st.ch("density")[None, None], 16)[0, 0]
    ref = pool(_plume(sim, 96, steps))
    errs = {}
    for res in (32, 48, 64):
        errs[res] = ((pool(_plume(sim, res, steps)) - ref).abs().mean() / ref.abs().mean()).item()
    print("plume rel L1 vs 96^3 reference:", {k: round(v, 4) for k, v in errs.items()})
    assert errs[48] < errs[32] and errs[64] < errs[48], errs


def _plume_stats(st):
    d = st.ch("density") * fluid_mask(st)
    dv = st.meta.dx ** 3
    _, yy, _ = world_grid(d.shape[0])
    smoke = d.double().sum().item() * dv
    heat = ((st.ch("temperature") - AMBIENT_TEMP) * fluid_mask(st)).double().sum().item() * dv
    cy = ((d * yy).sum() / d.sum()).item()
    slab = d.sum(dim=(0, 2))                                  # per-y slab mass profile
    rows = torch.nonzero(slab > 0.02 * slab.max()).flatten()
    top = (rows.max().item() + 1) * st.meta.dx                # upper edge of the highest significant slab
    return {"smoke": smoke, "heat": heat, "centroid_y": cy, "top_y": top}


def test_large_scale_agreement_64_vs_128(sim):
    """World units: a 64^3 and a 128^3 run of the same campfire scene match at large scales (fields3d.py).
    Tolerances stated in the module docstring."""
    scene = Forces3D(emitters=[Emitter3D(pos=(0.5, 0.15, 0.5), radius=0.08, density_rate=1.0, temperature=700.0,
                                         velocity=(0.0, 0.5, 0.0))])
    steps = 60                                                # 2 s
    a = _plume_stats(_plume(sim, 64, steps, scene))
    b = _plume_stats(_plume(sim, 128, steps, scene))
    print("64^3 :", {k: round(v, 4) for k, v in a.items()})
    print("128^3:", {k: round(v, 4) for k, v in b.items()})
    rs = abs(a["smoke"] - b["smoke"]) / b["smoke"]
    rh = abs(a["heat"] - b["heat"]) / max(abs(b["heat"]), 1e-12)
    dc = abs(a["centroid_y"] - b["centroid_y"])
    dt_ = abs(a["top_y"] - b["top_y"])
    print(f"smoke rel diff {rs:.3f}, heat rel diff {rh:.3f}, centroid dy {dc:.4f} m, top dy {dt_:.4f} m")
    assert b["smoke"] > 0 and b["centroid_y"] > 0.15, "plume did not develop"
    assert rs < 0.10, f"smoke mass differs {rs * 100:.1f}% (> 10%)"
    assert rh < 0.10, f"heat differs {rh * 100:.1f}% (> 10%)"
    assert dc < 0.04, f"plume centroid height differs {dc:.3f} m (> 0.04)"
    assert dt_ < 0.0625, f"plume top differs {dt_:.3f} m (> 0.0625)"


# ----------------------------------------------------------------------------- stress
def test_stress_1000_steps_fire_and_impulses(sim):
    s = sim.Solver3D(RES)
    st = s.reset([Obstacle3D("sphere", (0.35, 0.6, 0.5), (0.09, 0.09, 0.09)),
                  Obstacle3D("box", (0.7, 0.45, 0.4), (0.06, 0.14, 0.06))])
    g = torch.Generator().manual_seed(0)
    umax = dmax = tmax = float("nan")
    for i in range(1000):
        imp = []
        if i % 5 == 0:
            p = torch.rand(3, generator=g).tolist()
            v = ((torch.rand(3, generator=g) * 2 - 1) * 8.0).tolist()
            imp = [Impulse3D(pos=tuple(p), radius=0.12, velocity=tuple(v))]
        f = Forces3D(emitters=[Emitter3D(pos=(0.5, 0.1, 0.5), radius=0.09, density_rate=5.0, temperature=2000.0,
                                         fuel_rate=5.0, velocity=(0.0, 3.0, 0.0)),
                               Emitter3D(pos=(0.2, 0.15, 0.6), radius=0.06, density_rate=3.0, temperature=1800.0,
                                         fuel_rate=4.0, velocity=(1.5, 2.0, -0.5))],
                     impulses=imp, wind=(0.5 * math.sin(i * 0.01), 0.0, 0.3 * math.cos(i * 0.013)))
        st = s.step(st, f, SIM_DT)
        if i % 100 == 99:
            check_finite(st, f"at step {i + 1}")
            umax = st.velocity.abs().max().item()
            dmax = st.ch("density").max().item()
            tmax = st.ch("temperature").max().item()
            assert umax < 100.0, f"velocity blow-up {umax:.1f} m/s at step {i + 1}"
            assert dmax < 1e3, f"density blow-up {dmax:.1f} at step {i + 1}"
            assert tmax < 1e4, f"temperature blow-up {tmax:.0f} K at step {i + 1}"
    print(f"after 1000 steps: umax {umax:.2f} m/s, dmax {dmax:.2f}, Tmax {tmax:.0f} K")


# ============================================================================= sim-carried detail phase
# CLAUDE.md "Sim-carried detail phase": opt-in solver paths, defaults unchanged and BIT-IDENTICAL.
#   Solver3D(res, config, device, scalar_res=None|128); solver.hires_volume() -> Volume | None
#   SolverConfig3D.pressure_mode: "pcg" (default) | "fused"
#
# Default-path hashes: sha256 of state.data (float32 bytes) after 1 / 30 / 120 steps of `_scripted_run`
# (64^3, sphere obstacle, fire emitter, impulses every 17 steps, wind). Recorded by the test engineer on
# 2026-11-09 BEFORE any sim-carried-detail solver change (vfx/sim3d files last modified 16:10, version
# "sim3d-warp-0.1.0"), on the environment below. Bit-exact hashes are only meaningful on the same GPU /
# torch / warp build, so the test skips (with the reason) on a different environment.
DEFAULT_PATH_HASHES = {
    1: "06d785b40a77148a783adcda59510246c6874cc5c5adb906103888e02f4df1fc",
    30: "99e634ff247cd9b468875d85822dee02c9ee740ad1ebe1686d8cba5a0d51999a",
    120: "940e2b56268e8ff3a83eb1643731afd4207741726397eb20d757fef3a9d64e77",
}
# Same run with vfx.sim3d.runtime_config() (the sandbox's / run_pair's 64^3 setting), recorded at the same time.
RUNTIME_PATH_HASHES = {
    1: "79b3e8a646ab1a568a0fbb34c84ecfa1aa613ee3ac6a7d4918ccbec6c67acab6",
    30: "f25a1f5ae0217a56907d438398ee8fdd1fcb68453e7f371482c8928c06a021d1",
    120: "8a4e6e9796563d518dea3bb864eb3de1221ac885198b815e78d94ab991caf85d",
}
HASH_ENV = {"gpu":"NVIDIA GeForce RTX 5080", "torch": "2.11.0+cu128", "warp": "1.19.0"}
FINE_MASS_TOL = 1e-3            # contract: fine-field mass within 0.1% over 1000 steps
RESTRICT_TOL = 0.05             # this file: rel L1 of avgpool(fine) vs the 64^3 density (stated, no spec number)


def _hash_env_or_skip():
    import warp as wp
    env = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "warp": wp.config.version}
    if env != HASH_ENV:
        pytest.skip(f"stored default-path hash was recorded on {HASH_ENV}; this machine is {env}")


def _hashes_of_run(make_solver, steps=120, at=(1, 30, 120)):
    import hashlib
    s = make_solver()
    st = s.reset(sphere_obstacle())
    out = {}
    for i in range(steps):
        f = Forces3D(emitters=[Emitter3D(pos=(0.5, 0.12, 0.5), radius=0.06, density_rate=2.0, temperature=1300.0,
                                         fuel_rate=1.5, velocity=(0.0, 1.0, 0.0))],
                     impulses=[Impulse3D(pos=(0.3, 0.5, 0.4), radius=0.1, velocity=(2.0, 0.5, -1.0))]
                     if i % 17 == 0 else [],
                     wind=(0.2, 0.0, 0.1))
        st = s.step(st, f, SIM_DT)
        if i + 1 in at:
            out[i + 1] = hashlib.sha256(st.data.contiguous().cpu().numpy().tobytes()).hexdigest()
    return out


def test_default_path_bit_identical_to_recorded_hash(sim):
    """The default Solver3D(64) must reproduce the hashes recorded before the opt-in paths were added."""
    _hash_env_or_skip()
    got = _hashes_of_run(lambda: sim.Solver3D(RES))
    bad = {k: got[k][:12] for k in DEFAULT_PATH_HASHES if got[k] != DEFAULT_PATH_HASHES[k]}
    assert not bad, f"default solver path changed: steps {sorted(bad)} differ from the recorded hashes ({bad})"


def test_runtime_config_path_bit_identical_to_recorded_hash(sim):
    _hash_env_or_skip()
    if not hasattr(sim, "runtime_config"):
        pytest.skip("vfx.sim3d.runtime_config not exported")
    got = _hashes_of_run(lambda: sim.Solver3D(RES, sim.runtime_config()))
    bad = {k: got[k][:12] for k in RUNTIME_PATH_HASHES if got[k] != RUNTIME_PATH_HASHES[k]}
    assert not bad, f"runtime_config() solver path changed: steps {sorted(bad)} differ from the recorded hashes"


def _accepts_kw(fn, name):
    import inspect
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def test_explicit_defaults_bit_identical(sim):
    """Passing the documented defaults explicitly (scalar_res=None, pressure_mode='pcg') == the default path."""
    _hash_env_or_skip()
    kw = {}
    if _accepts_kw(sim.Solver3D.__init__, "scalar_res"):
        kw["scalar_res"] = None
    has_mode = hasattr(sim.SolverConfig3D(), "pressure_mode")
    if not kw and not has_mode:
        pytest.skip("neither scalar_res nor pressure_mode exists yet")
    cfg = make_config(sim, pressure_mode="pcg") if has_mode else None
    got = _hashes_of_run(lambda: sim.Solver3D(RES, cfg, **kw))
    assert got == DEFAULT_PATH_HASHES, "explicit default options do not reproduce the recorded default path"
    if _accepts_kw(sim.Solver3D.__init__, "scalar_res"):
        assert sim.Solver3D(RES).hires_volume() is None, "hires_volume() must be None without scalar_res"


# ----------------------------------------------------------------------------- pressure_mode="fused"
def _fused_or_skip(sim):
    if not hasattr(sim.SolverConfig3D(), "pressure_mode"):
        pytest.skip("SolverConfig3D.pressure_mode not implemented yet (sim-carried detail phase)")
    return make_config(sim, pressure_mode="fused")


def _fused_label():
    """Contract: fused either meets div < 1e-4 or is REPORTED as a labelled lower-quality level with its
    number. The label is looked for in vfx/sim3d/README.md (the solver owner's documentation)."""
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / "vfx" / "sim3d" / "README.md"
    txt = p.read_text(encoding="utf-8").lower() if p.exists() else ""
    return "fused" in txt and ("lower-quality" in txt or "lower quality" in txt or "quality level" in txt)


def _check_fused_div(m, what):
    print(f"pressure_mode='fused' {what}: max normalized div {m:.3e} (spec {DIV_TOL})")
    if m < DIV_TOL:
        return
    if _fused_label():
        pytest.xfail(f"fused {what}: div {m:.3e} >= {DIV_TOL}; documented as a lower-quality level in sim3d README")
    pytest.fail(f"fused {what}: max normalized div {m:.3e} >= {DIV_TOL} and vfx/sim3d/README.md does not label "
                f"fused as a lower-quality level")


@pytest.mark.parametrize("obstacles", [no_obstacles, sphere_obstacle], ids=["open", "sphere"])
def test_fused_divergence_after_projection(sim, obstacles):
    cfg = _fused_or_skip(sim)
    s = sim.Solver3D(RES, cfg)
    st = s.reset(obstacles())
    v = smooth_noise(RES, 3, seed=1, amp=2.0) + torch.tensor([0.8, -0.3, 0.4], device=DEV)[:, None, None, None]
    st.velocity.copy_(v * fluid_mask(st))
    st = s.project(st)
    check_finite(st, "fused project")
    div = s.divergence(st)
    assert torch.all(div[st.solid > 0.5] == 0)
    _check_fused_div(div.max().item(), f"project ({obstacles.__name__})")


@pytest.mark.parametrize("obstacles", [no_obstacles, sphere_obstacle], ids=["open", "sphere"])
def test_fused_divergence_after_full_steps(sim, obstacles):
    cfg = _fused_or_skip(sim)
    s = sim.Solver3D(RES, cfg)
    st = s.reset(obstacles())
    f = Forces3D(emitters=[Emitter3D(pos=(0.5, 0.15, 0.5), radius=0.07, density_rate=2.0,
                                     temperature=900.0, velocity=(0.0, 1.0, 0.0))])
    for _ in range(60):
        st = s.step(st, f, SIM_DT)
    check_finite(st, "fused 60 steps")
    _check_fused_div(s.divergence(st).max().item(), f"60 steps ({obstacles.__name__})")


def test_fused_deterministic_and_finite(sim):
    cfg = _fused_or_skip(sim)
    a = _hashes_of_run(lambda: sim.Solver3D(RES, cfg), steps=120, at=(120,))
    b = _hashes_of_run(lambda: sim.Solver3D(RES, cfg), steps=120, at=(120,))
    assert a == b, "fused pressure path is not bit-identical between identical runs"


# ----------------------------------------------------------------------------- scalar_res=128
FINE = 128


def _fine_or_skip(sim):
    if not _accepts_kw(sim.Solver3D.__init__, "scalar_res") or not hasattr(sim.Solver3D, "hires_volume"):
        pytest.skip("Solver3D(scalar_res=...) / hires_volume() not implemented yet (sim-carried detail phase)")


def _fine_mass(vol, solid_fine=None):
    d = vol.density.float()
    if solid_fine is not None:
        d = d * (solid_fine < 0.5)
    return float(d.double().sum().item()) * (DOMAIN_SIZE / d.shape[0]) ** 3


def _fine_closed_box_run(sim, obstacles, steps):
    """Fill the fine field through the public interface (an emitter), stop the sources, then run `steps`
    sourceless steps with dissipation 0. Returns (fine mass before, fine mass after, solver, state)."""
    s = sim.Solver3D(RES, make_config(sim, density_dissipation=0.0), scalar_res=FINE)
    st = s.reset(obstacles)
    fill = Forces3D(emitters=[Emitter3D(pos=(0.5, 0.2, 0.5), radius=0.08, density_rate=2.0, temperature=700.0,
                                        velocity=(0.0, 0.6, 0.0))])
    for _ in range(30):
        st = s.step(st, fill, SIM_DT)
    vol = s.hires_volume()
    assert vol is not None, "hires_volume() returned None with scalar_res set"
    assert tuple(vol.density.shape) == (FINE,) * 3 and tuple(vol.temperature.shape) == (FINE,) * 3
    solid_f = rasterize_obstacles_3d(obstacles, FINE, DEV)
    m0 = _fine_mass(vol, solid_f)
    assert m0 > 0, "emitter put no smoke in the fine field"
    for _ in range(steps):
        st = s.step(st, Forces3D(), SIM_DT)
    m1 = _fine_mass(s.hires_volume(), solid_f)
    return m0, m1, s, st


@pytest.mark.parametrize("obstacles", [no_obstacles, sphere_obstacle], ids=["open", "sphere"])
def test_scalar_res_fine_mass_conserved(sim, obstacles):
    _fine_or_skip(sim)
    m0, m1, s, st = _fine_closed_box_run(sim, obstacles(), 1000)
    vol = s.hires_volume()
    assert torch.isfinite(vol.density).all() and torch.isfinite(vol.temperature).all()
    check_finite(st)
    rel = abs(m1 - m0) / m0
    print(f"scalar_res={FINE} {obstacles.__name__}: fine mass {m0:.5f} -> {m1:.5f} (rel {rel:.3e})")
    assert rel < FINE_MASS_TOL, f"fine-field mass changed {rel * 100:.3f}% over 1000 steps (> 0.1%)"


def test_scalar_res_restriction_consistent(sim):
    """The 64^3 density/temperature the solver keeps (and uses for buoyancy) must be consistent with the
    fine field: avgpool2(fine) vs coarse, rel L1 < RESTRICT_TOL, and mass equal within 0.1%."""
    _fine_or_skip(sim)
    s = sim.Solver3D(RES, scalar_res=FINE)
    st = s.reset(sphere_obstacle())
    f = Forces3D(emitters=[Emitter3D(pos=(0.5, 0.12, 0.5), radius=0.07, density_rate=2.0, temperature=1000.0,
                                     velocity=(0.0, 1.0, 0.0))])
    for _ in range(90):
        st = s.step(st, f, SIM_DT)
    vol = s.hires_volume()
    fluid = fluid_mask(st)
    rd = F.avg_pool3d(vol.density.float()[None, None], 2)[0, 0] * fluid
    rt = F.avg_pool3d(vol.temperature.float()[None, None], 2)[0, 0]
    cd = st.ch("density") * fluid
    l1d = ((rd - cd).abs().sum() / cd.abs().sum().clamp(min=1e-12)).item()
    ex_c = (st.ch("temperature") - AMBIENT_TEMP) * fluid
    l1t = (((rt - AMBIENT_TEMP) * fluid - ex_c).abs().sum() / ex_c.abs().sum().clamp(min=1e-12)).item()
    mrel = abs(rd.double().sum().item() - cd.double().sum().item()) / max(cd.double().sum().item(), 1e-12)
    print(f"restriction vs 64^3: density rel L1 {l1d:.4f}, excess-T rel L1 {l1t:.4f}, mass rel {mrel:.3e}")
    assert l1d < RESTRICT_TOL, f"avgpool(fine density) differs from 64^3 density by rel L1 {l1d:.3f}"
    assert l1t < RESTRICT_TOL, f"avgpool(fine temperature) differs from 64^3 temperature by rel L1 {l1t:.3f}"
    assert mrel < MASS_TOL, f"fine and coarse smoke mass differ by {mrel * 100:.3f}%"


def test_scalar_res_no_smoke_inside_obstacles(sim):
    _fine_or_skip(sim)
    s = sim.Solver3D(RES, scalar_res=FINE)
    obs = [Obstacle3D("box", (0.5, 0.55, 0.5), (0.18, 0.06, 0.18))]
    st = s.reset(obs)
    f = Forces3D(emitters=[Emitter3D(pos=(0.5, 0.2, 0.5), radius=0.07, density_rate=3.0, temperature=1200.0,
                                     velocity=(0.0, 2.0, 0.0))])
    for _ in range(120):
        st = s.step(st, f, SIM_DT)
    sf = rasterize_obstacles_3d(obs, FINE, DEV)
    deep = (-F.max_pool3d(-sf[None, None], 5, 1, 2)[0, 0]) > 0.5
    d_in = s.hires_volume().density.float()[deep].abs().max().item()
    print(f"fine density inside obstacle (2+ fine cells deep): {d_in:.2e}")
    assert d_in < 1e-3


def test_scalar_res_deterministic(sim):
    _fine_or_skip(sim)
    import hashlib

    def run():
        s = sim.Solver3D(RES, scalar_res=FINE)
        _hashes_of_run(lambda: s, steps=60, at=(60,))
        v = s.hires_volume()
        return hashlib.sha256(torch.cat([v.density.float().flatten(), v.temperature.float().flatten()])
                              .cpu().numpy().tobytes()).hexdigest()
    assert run() == run(), "scalar_res fine field differs between identical runs"


def test_scalar_res_stress_1000_steps_finite(sim):
    _fine_or_skip(sim)
    s = sim.Solver3D(RES, scalar_res=FINE)
    st = s.reset([Obstacle3D("sphere", (0.35, 0.6, 0.5), (0.09, 0.09, 0.09))])
    g = torch.Generator().manual_seed(0)
    for i in range(1000):
        imp = []
        if i % 5 == 0:
            p = torch.rand(3, generator=g).tolist()
            v = ((torch.rand(3, generator=g) * 2 - 1) * 8.0).tolist()
            imp = [Impulse3D(pos=tuple(p), radius=0.12, velocity=tuple(v))]
        f = Forces3D(emitters=[Emitter3D(pos=(0.5, 0.1, 0.5), radius=0.09, density_rate=5.0, temperature=2000.0,
                                         fuel_rate=5.0, velocity=(0.0, 3.0, 0.0))], impulses=imp)
        st = s.step(st, f, SIM_DT)
        if i % 100 == 99:
            check_finite(st, f"at step {i + 1}")
            v = s.hires_volume()
            nb = (~torch.isfinite(v.density)).sum().item() + (~torch.isfinite(v.temperature)).sum().item()
            assert nb == 0, f"{nb} non-finite fine values at step {i + 1}"
            assert v.density.float().min().item() >= -1e-4, "negative fine density"
            assert v.density.float().max().item() < 1e3
