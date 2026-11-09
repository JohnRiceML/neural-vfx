"""Layer 1: solver correctness (SPEC "Testing and validation"), via the vfx.sim public interface only.

Written from SPEC.md + CLAUDE.md, not from the solver implementation. Thresholds are the spec's:
  * normalized divergence after projection < 1e-4 everywhere (fluid cells)
  * density mass conserved within 0.1% over 1000 steps, closed box, no sources, dissipation 0
  * pure advection matches the analytic result
  * buoyancy lifts hot fluid; walls / obstacles are not penetrated
  * determinism: bit-identical
  * convergence: error vs a fine reference shrinks with resolution
plus a 2000-step no-NaN stress run (strong fire + impulses).
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from vfx.core.fields import (
    AMBIENT_TEMP, CH, DOMAIN_SIZE, SIM_DT, Emitter, FieldState, Forces, Impulse, Obstacle,
)
from tests.helpers import import_or_skip, make_config, requires_cuda

pytestmark = requires_cuda
DEV = "cuda"
DIV_TOL = 1e-4          # SPEC Layer 1
MASS_TOL = 1e-3         # 0.1%


@pytest.fixture(scope="module")
def sim():
    return import_or_skip("vfx.sim", ("Solver", "SolverConfig", "SOLVER_VERSION"))


def world_grid(res):
    c = (torch.arange(res, device=DEV, dtype=torch.float32) + 0.5) * (DOMAIN_SIZE / res)
    yy, xx = torch.meshgrid(c, c, indexing="ij")     # [y, x], y = 0 bottom
    return xx, yy


def blob(res, cx, cy, sigma):
    xx, yy = world_grid(res)
    return torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))


def smooth_noise(res, channels, seed, coarse=12, amp=1.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    z = torch.randn(1, channels, coarse, coarse, device=DEV, generator=g)
    return amp * F.interpolate(z, size=(res, res), mode="bicubic", align_corners=False)[0]


def fluid_mask(state: FieldState) -> torch.Tensor:
    return state.solid < 0.5


def mass(state: FieldState) -> float:
    d = state.ch("density")
    return float((d * fluid_mask(state)).double().sum().item())


def check_finite(state: FieldState, where=""):
    bad = (~torch.isfinite(state.data)).sum().item()
    assert bad == 0, f"{bad} non-finite values {where}"


def no_obstacles():
    return []


def circle_obstacle():
    return [Obstacle(kind="circle", pos=(0.5, 0.45), size=(0.12, 0.12))]


def box_obstacle():
    return [Obstacle(kind="box", pos=(0.3, 0.7), size=(0.08, 0.05))]


# ----------------------------------------------------------------------------- interface
def test_interface_reset_contract(sim):
    s = sim.Solver(128)
    st = s.reset()
    st.validate()
    assert st.meta.res == (128, 128)
    assert math.isclose(st.meta.dx, DOMAIN_SIZE / 128)
    assert torch.all(st.ch("density") == 0)
    assert torch.all(st.velocity == 0)
    assert torch.allclose(st.ch("temperature"), torch.full_like(st.ch("temperature"), AMBIENT_TEMP))
    assert torch.all(st.solid == 0), "box walls must NOT be marked in solid (fields.py)"
    assert isinstance(sim.SOLVER_VERSION, str) and sim.SOLVER_VERSION
    for name in ("buoyancy_alpha", "buoyancy_beta", "vorticity_eps", "density_dissipation",
                 "temperature_cooling", "burn_rate", "ignition_temp", "burn_heat",
                 "smoke_from_fuel", "pressure_tol", "pressure_max_iters"):
        assert hasattr(sim.SolverConfig(), name), f"SolverConfig missing {name}"


def test_reset_rasterizes_obstacles(sim):
    from vfx.core.fields import rasterize_obstacles
    obs = circle_obstacle() + box_obstacle()
    st = sim.Solver(128).reset(obs)
    assert torch.equal(st.solid, rasterize_obstacles(obs, 128, DEV))


# ----------------------------------------------------------------------------- divergence
TIGHT = dict(pressure_tol=1e-7, pressure_max_iters=5000)       # "converged" solve
TIGHTER = dict(pressure_tol=1e-8, pressure_max_iters=20000)   # below fp32 eps: must stall, not diverge


def _project_noise(sim, res, obstacles, **cfg):
    s = sim.Solver(res, make_config(sim, **cfg) if cfg else None)
    st = s.reset(obstacles)
    v = smooth_noise(res, 2, seed=1, amp=2.0) + torch.tensor([0.8, -0.3], device=DEV)[:, None, None]
    st.velocity.copy_(v * fluid_mask(st))
    before = s.divergence(st).max().item()
    st = s.project(st)
    return s, st, before


@pytest.mark.parametrize("obstacles", [no_obstacles, circle_obstacle], ids=["open", "obstacle"])
@pytest.mark.parametrize("res", [128, 256])
def test_divergence_after_projection(sim, obstacles, res):
    """SPEC: < 1e-4 after projection, with the DEFAULT config (what datagen and the runtime use)."""
    s, st, before = _project_noise(sim, res, obstacles())
    div = s.divergence(st)
    assert div.shape == (res, res)
    assert torch.all(div[st.solid > 0.5] == 0), "divergence() must be 0 in solid cells"
    check_finite(st, "after project")
    m = div.max().item()
    print(f"res {res} {obstacles.__name__}: normalized div before {before:.3e} after {m:.3e}")
    assert before > 1e-2, "test field should start strongly divergent"
    assert m < DIV_TOL, f"max normalized divergence {m:.3e} >= {DIV_TOL} (default pressure_tol/max_iters)"


@pytest.mark.parametrize("obstacles", [no_obstacles, circle_obstacle], ids=["open", "obstacle"])
@pytest.mark.parametrize("res", [128, 256])
def test_divergence_converged_solve(sim, obstacles, res):
    """Same threshold with a tight pressure solve: isolates the discretization from the iteration budget."""
    s, st, _ = _project_noise(sim, res, obstacles(), **TIGHT)
    m = s.divergence(st).max().item()
    print(f"res {res} {obstacles.__name__} tight solve: max normalized div {m:.3e}")
    assert m < DIV_TOL


@pytest.mark.parametrize("obstacles", [no_obstacles, circle_obstacle], ids=["open", "obstacle"])
def test_tighter_pressure_tol_is_never_worse(sim, obstacles):
    """Asking the pressure solver for more accuracy must not make the result worse (solver stability)."""
    s1, st1, _ = _project_noise(sim, 128, obstacles(), **TIGHT)
    s2, st2, _ = _project_noise(sim, 128, obstacles(), **TIGHTER)
    check_finite(st2, "after tighter project")
    m1, m2 = s1.divergence(st1).max().item(), s2.divergence(st2).max().item()
    print(f"{obstacles.__name__}: max div tol 1e-7 {m1:.3e}, tol 1e-8 {m2:.3e}")
    assert m2 <= max(2 * m1, DIV_TOL), f"tol=1e-8 gives {m2:.3e}, worse than tol=1e-7 ({m1:.3e})"


@pytest.mark.parametrize("obstacles", [no_obstacles, circle_obstacle], ids=["open", "obstacle"])
def test_divergence_after_full_steps(sim, obstacles):
    """step() ends with a projection, so a running plume must stay divergence-free."""
    s = sim.Solver(128)
    st = s.reset(obstacles())
    f = Forces(emitters=[Emitter(pos=(0.5, 0.15), radius=0.06, density_rate=2.0,
                                 temperature=900.0, velocity=(0.0, 1.0))])
    for _ in range(60):
        st = s.step(st, f, SIM_DT)
    m = s.divergence(st).max().item()
    print(f"{obstacles.__name__}: normalized div after 60 steps {m:.3e}")
    assert m < DIV_TOL


@pytest.mark.parametrize("cfg", [{}, TIGHT], ids=["default", "tight"])
def test_projection_removes_flow_into_walls(sim, cfg):
    """A uniform flow u = grad(x + 0.5 y) is a pure gradient: its divergence-free, no-penetration
    projection in the closed box is zero, so the flow into the walls must be removed, and the net
    flux through every column/row is ~0. Tolerance: 1% of the input speed. (Not run with an obstacle:
    a masked uniform field violates the obstacle BC, so the discrete answer is scheme-dependent;
    obstacles are covered by test_no_flow_into_obstacles.)"""
    res = 128
    s = sim.Solver(res, make_config(sim, **cfg) if cfg else None)
    st = s.reset()
    st.velocity[0].fill_(1.0)
    st.velocity[1].fill_(0.5)
    st.velocity.mul_(fluid_mask(st))
    st = s.project(st)
    check_finite(st, "after project")
    u = st.velocity
    resid = u.abs().max().item()                                  # input speed scale is 1.0
    col_flux = u[0].mean(dim=0).abs().max().item()
    row_flux = u[1].mean(dim=1).abs().max().item()
    print(f"{cfg or 'default'}: max|u| after {resid:.3e}, net col flux {col_flux:.3e}, "
          f"row {row_flux:.3e}")
    assert resid < 1e-2, f"projection left |u| = {resid:.3e} of a pure-gradient (into-wall) flow"
    assert col_flux < 1e-2 and row_flux < 1e-2


# ----------------------------------------------------------------------------- mass
def test_mass_conserved_closed_box(sim):
    cfg = make_config(sim, density_dissipation=0.0)
    s = sim.Solver(128, cfg)
    st = s.reset()
    st.ch("density").copy_(blob(128, 0.5, 0.35, 0.08) + 0.5 * blob(128, 0.3, 0.7, 0.05))
    st.ch("temperature").copy_(AMBIENT_TEMP + 400.0 * blob(128, 0.5, 0.35, 0.08))
    st.velocity.copy_(smooth_noise(128, 2, seed=3, amp=0.5))       # stir it, including into walls
    m0 = mass(st)
    for i in range(1000):
        st = s.step(st, Forces(), SIM_DT)
    check_finite(st, "after 1000 steps")
    m1 = mass(st)
    rel = abs(m1 - m0) / m0
    print(f"mass {m0:.4f} -> {m1:.4f} (rel change {rel:.3e})")
    assert rel < MASS_TOL, f"mass changed by {rel*100:.3f}% (> 0.1%)"


def test_mass_conserved_with_obstacle(sim):
    cfg = make_config(sim, density_dissipation=0.0)
    s = sim.Solver(128, cfg)
    st = s.reset(circle_obstacle())
    st.ch("density").copy_(blob(128, 0.5, 0.15, 0.07) * fluid_mask(st))
    st.ch("temperature").copy_(AMBIENT_TEMP + 500.0 * blob(128, 0.5, 0.15, 0.07))
    m0 = mass(st)
    for _ in range(1000):
        st = s.step(st, Forces(), SIM_DT)
    rel = abs(mass(st) - m0) / m0
    print(f"obstacle: rel mass change {rel:.3e}")
    assert rel < MASS_TOL


# ----------------------------------------------------------------------------- advection
def test_advect_integer_shift_is_exact(sim):
    """u*dt = whole cells -> any consistent semi-Lagrangian/MacCormack scheme is exact in the interior."""
    res = 128
    s = sim.Solver(res)
    dx = DOMAIN_SIZE / res
    f0 = blob(res, 0.35, 0.4, 0.06)
    vel = torch.zeros(2, res, res, device=DEV)
    vel[0] = 2 * dx / SIM_DT
    vel[1] = 1 * dx / SIM_DT
    f = f0.clone()
    n = 20
    for _ in range(n):
        f = s.advect(f, vel, SIM_DT)
    expect = blob(res, 0.35 + 2 * n * dx, 0.4 + n * dx, 0.06)
    err = (f - expect).abs().max().item()
    print(f"integer-shift advection max err {err:.2e}")
    assert err < 1e-4


def test_advect_matches_analytic_translation(sim):
    """Fractional-cell constant velocity: blob centroid must follow x0 + u t; shape within tolerance."""
    res = 128
    s = sim.Solver(res)
    dx = DOMAIN_SIZE / res
    u = (0.13, 0.07)                                 # m/s, 0.55 / 0.30 cells per step
    sig = 0.07
    f = blob(res, 0.3, 0.3, sig)
    vel = torch.zeros(2, res, res, device=DEV)
    vel[0], vel[1] = u
    n = 60                                           # 2 s
    for _ in range(n):
        f = s.advect(f, vel, SIM_DT)
    T = n * SIM_DT
    ex = blob(res, 0.3 + u[0] * T, 0.3 + u[1] * T, sig)
    xx, yy = world_grid(res)
    cx = (f * xx).sum() / f.sum()
    cy = (f * yy).sum() / f.sum()
    cerr = math.hypot(cx.item() - (0.3 + u[0] * T), cy.item() - (0.3 + u[1] * T)) / dx
    rel_l2 = ((f - ex).norm() / ex.norm()).item()
    print(f"centroid err {cerr:.3f} cells, rel L2 {rel_l2:.3f}, peak {f.max().item():.3f}")
    assert cerr < 0.5
    assert rel_l2 < 0.15


def test_advect_multichannel_consistent(sim):
    s = sim.Solver(64)
    f = torch.stack([blob(64, 0.4, 0.5, 0.08), 2 * blob(64, 0.6, 0.5, 0.05)])
    vel = smooth_noise(64, 2, seed=5, amp=0.3)
    out = s.advect(f, vel, SIM_DT)
    assert out.shape == f.shape
    for c in range(2):
        assert torch.allclose(out[c], s.advect(f[c].contiguous(), vel, SIM_DT), atol=1e-6)


def test_advection_converges_with_resolution(sim):
    """Analytic translation error shrinks as resolution increases (same world-space problem)."""
    errs = {}
    for res in (32, 64, 128, 256):
        s = sim.Solver(res)
        vel = torch.zeros(2, res, res, device=DEV)
        vel[0], vel[1] = 0.11, 0.05
        f = blob(res, 0.3, 0.3, 0.08)
        n = 30
        for _ in range(n):
            f = s.advect(f, vel, SIM_DT)
        ex = blob(res, 0.3 + 0.11 * n * SIM_DT, 0.3 + 0.05 * n * SIM_DT, 0.08)
        errs[res] = ((f - ex).abs().mean() / ex.abs().mean()).item()
    print("advection rel L1 vs analytic:", errs)
    r = sorted(errs)
    for a, b in zip(r, r[1:]):
        assert errs[b] < errs[a], errs


# ----------------------------------------------------------------------------- buoyancy
def test_buoyancy_lifts_hot_fluid(sim):
    cfg = make_config(sim, buoyancy_alpha=0.0, vorticity_eps=0.0, burn_rate=0.0)
    if getattr(cfg, "buoyancy_beta", 0) <= 0:
        pytest.fail("default buoyancy_beta must be > 0 (heat lifts)")
    s = sim.Solver(128, cfg)
    st = s.reset()
    hot = blob(128, 0.5, 0.3, 0.07)
    st.ch("temperature").copy_(AMBIENT_TEMP + 500.0 * hot)
    st.ch("density").copy_(hot)                          # passive tracer of the hot parcel
    xx, yy = world_grid(128)
    cy0 = ((st.ch("density") * yy).sum() / st.ch("density").sum()).item()
    for _ in range(20):
        st = s.step(st, Forces(), SIM_DT)
    d = st.ch("density")
    cy1 = ((d * yy).sum() / d.sum()).item()
    vy_hot = (st.velocity[1] * hot).sum() / hot.sum()
    print(f"hot parcel centroid y {cy0:.3f} -> {cy1:.3f}, mean vy in parcel {vy_hot.item():.3f} m/s")
    assert vy_hot.item() > 0
    assert cy1 > cy0 + 0.01


def test_dense_cold_fluid_sinks(sim):
    base = sim.SolverConfig()
    alpha = base.buoyancy_alpha if base.buoyancy_alpha > 0 else 1.0
    cfg = make_config(sim, buoyancy_beta=0.0, buoyancy_alpha=alpha, vorticity_eps=0.0, burn_rate=0.0)
    s = sim.Solver(128, cfg)
    st = s.reset()
    d0 = 2.0 * blob(128, 0.5, 0.7, 0.07)
    st.ch("density").copy_(d0)
    for _ in range(20):
        st = s.step(st, Forces(), SIM_DT)
    vy = (st.velocity[1] * d0).sum() / d0.sum()
    print(f"dense parcel mean vy {vy.item():.3f} m/s (alpha={alpha})")
    assert vy.item() < 0


def test_ambient_fluid_stays_at_rest(sim):
    s = sim.Solver(128)
    st = s.reset([Obstacle("box", (0.5, 0.5), (0.1, 0.1))])
    for _ in range(50):
        st = s.step(st, Forces(), SIM_DT)
    assert st.velocity.abs().max().item() < 1e-5
    assert st.ch("density").abs().max().item() < 1e-6


# ----------------------------------------------------------------------------- boundaries
def test_no_flow_into_obstacles(sim):
    """A jet blown at an obstacle: nothing enters it (density/fuel stay 0, velocity 0 inside)."""
    s = sim.Solver(128)
    obs = [Obstacle("box", (0.5, 0.55), (0.15, 0.06)), Obstacle("circle", (0.8, 0.3), (0.08, 0.08))]
    st = s.reset(obs)
    f = Forces(emitters=[Emitter(pos=(0.5, 0.2), radius=0.06, density_rate=3.0, temperature=1200.0,
                                 fuel_rate=1.0, velocity=(0.0, 2.0)),
                         Emitter(pos=(0.45, 0.3), radius=0.04, density_rate=2.0, velocity=(3.0, 0.0))])
    inside = st.solid > 0.5
    # cells at least 1 cell deep inside the solid (the first layer may hold interpolated BC values)
    deep = (-F.max_pool2d(-st.solid[None, None], 3, 1, 1)[0, 0]) > 0.5
    for i in range(200):
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
    """Strong flow aimed at every wall, dissipation 0, no sources: no density lost through walls."""
    cfg = make_config(sim, density_dissipation=0.0)
    s = sim.Solver(128, cfg)
    st = s.reset()
    st.ch("density").copy_(blob(128, 0.85, 0.85, 0.06) + blob(128, 0.12, 0.1, 0.05))
    m0 = mass(st)
    for i in range(150):
        imp = [Impulse(pos=(0.85, 0.85), radius=0.1, velocity=(3.0, 3.0)),
               Impulse(pos=(0.12, 0.1), radius=0.1, velocity=(-3.0, -3.0))] if i % 10 == 0 else []
        st = s.step(st, Forces(impulses=imp), SIM_DT)
    rel = abs(mass(st) - m0) / m0
    print(f"wall-jet rel mass change {rel:.3e}")
    assert rel < MASS_TOL


# ----------------------------------------------------------------------------- determinism
def _scripted_run(sim, steps=120, res=128):
    s = sim.Solver(res)
    st = s.reset(circle_obstacle())
    for i in range(steps):
        f = Forces(emitters=[Emitter(pos=(0.5, 0.12), radius=0.05, density_rate=2.0, temperature=1300.0,
                                     fuel_rate=1.5, velocity=(0.0, 1.0))],
                   impulses=[Impulse(pos=(0.3, 0.5), radius=0.08, velocity=(2.0, 0.5))] if i % 17 == 0 else [],
                   wind=(0.2, 0.0))
        st = s.step(st, f, SIM_DT)
    return st.data.clone()


def test_determinism_bit_identical(sim):
    a = _scripted_run(sim)
    b = _scripted_run(sim)
    assert a.abs().sum().item() > 0
    diff = (a != b).sum().item()
    assert diff == 0, f"{diff} values differ between identical runs (max abs {(a - b).abs().max().item():.3e})"


# ----------------------------------------------------------------------------- convergence
def _plume_density(sim, res, steps):
    s = sim.Solver(res)
    st = s.reset()
    f = Forces(emitters=[Emitter(pos=(0.5, 0.2), radius=0.08, density_rate=1.0, temperature=600.0)])
    for _ in range(steps):
        st = s.step(st, f, SIM_DT)
    check_finite(st, f"res {res}")
    return st.ch("density").clone()


def test_convergence_against_fine_reference(sim):
    """Short smooth plume: error vs a 512^2 reference (compared on a common 32^2 grid) shrinks with res."""
    steps = 20
    ref = F.adaptive_avg_pool2d(_plume_density(sim, 512, steps)[None, None], 32)[0, 0]
    errs = {}
    for res in (64, 128, 256):
        d = F.adaptive_avg_pool2d(_plume_density(sim, res, steps)[None, None], 32)[0, 0]
        errs[res] = ((d - ref).abs().mean() / ref.abs().mean()).item()
    print("plume rel L1 vs 512^2 reference:", errs)
    assert errs[128] < errs[64] and errs[256] < errs[128], errs


# ----------------------------------------------------------------------------- stress
def test_stress_2000_steps_fire_and_impulses(sim):
    s = sim.Solver(128)
    st = s.reset([Obstacle("circle", (0.35, 0.6), (0.07, 0.07)), Obstacle("box", (0.7, 0.45), (0.05, 0.12))])
    g = torch.Generator().manual_seed(0)
    for i in range(2000):
        imp = []
        if i % 5 == 0:
            p = torch.rand(2, generator=g).tolist()
            v = ((torch.rand(2, generator=g) * 2 - 1) * 8.0).tolist()
            imp = [Impulse(pos=(p[0], p[1]), radius=0.1, velocity=(v[0], v[1]))]
        f = Forces(emitters=[Emitter(pos=(0.5, 0.1), radius=0.08, density_rate=5.0, temperature=2000.0,
                                     fuel_rate=5.0, velocity=(0.0, 3.0)),
                             Emitter(pos=(0.2, 0.15), radius=0.05, density_rate=3.0, temperature=1800.0,
                                     fuel_rate=4.0, velocity=(1.5, 2.0))],
                   impulses=imp, wind=(0.5 * math.sin(i * 0.01), 0.0))
        st = s.step(st, f, SIM_DT)
        if i % 100 == 99:
            check_finite(st, f"at step {i + 1}")
            umax = st.velocity.abs().max().item()
            dmax = st.ch("density").max().item()
            tmax = st.ch("temperature").max().item()
            assert umax < 100.0, f"velocity blow-up {umax:.1f} m/s at step {i + 1}"
            assert dmax < 1e3, f"density blow-up {dmax:.1f} at step {i + 1}"
            assert tmax < 1e4, f"temperature blow-up {tmax:.0f} K at step {i + 1}"
    print(f"after 2000 steps: umax {umax:.2f} m/s, dmax {dmax:.2f}, Tmax {tmax:.0f} K")
