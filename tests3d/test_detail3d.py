"""Contract tests for vfx.detail3d.AdvectedDetail + VolumeRenderer.draw(detail=...) (CLAUDE.md "Sim-carried detail
phase"). Written from the contract and the research report (shortlist #1: object-space detail carried by the 64^3
velocity, sampled in the march), not from the implementation. Everything skips until the module / kwarg exists.

Interface under test:
    AdvectedDetail(res=64, device="cuda", **params); reset(); step(state, dt); fields() -> dict; timings
    VolumeRenderer.draw(volume, camera, solid=None, ..., detail=None)

States are built directly as FieldState3D (no vfx.sim3d needed), with analytic velocity fields, so every expected
answer is known. Tolerances stated here (no spec number exists):
  * detail=None: bit-identical to a draw() without the kwarg, and to the hash recorded below
  * mean opacity (rendered alpha over the smoke footprint) changes < 5%; mean luminance there < 5%
  * empty space / obstacles: outside the no-detail footprint dilated by DIL px, frames differ by <= 1 code
  * finite over 2000 steps of a swirling flow (fields and renders), mean alpha stays within 25% of the no-detail one
  * moves with the flow: the detail layer (with - without detail) of a translating blob shifts by the projected
    translation within 25% (and along the motion direction); a screen-stuck layer would measure ~0
  * no periodic pop: frame-to-frame change of the detail layer in a quasi-steady plume (static density, gas flowing
    through it), burst = max_t / median_t <= POP_MAX = 2.0 over 150 steps (>= 2 regeneration periods of any
    period <= 2.5 s)
  * cost at 2560x1440, march_scale 0.5 (sandbox default): step() p50 + (render with detail - render without) p50
    <= 2.5 ms (the detail budget). Only asserted on an idle GPU (nvidia-smi); on a shared GPU the test skips with
    the measured numbers in the reason.
"""
from __future__ import annotations

import hashlib
import inspect
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vfx.core.fields3d import (
    AMBIENT_TEMP, DOMAIN_SIZE, NUM_CHANNELS_3D, SIM_DT, CH3, Camera, FieldMeta3D, FieldState3D, Obstacle3D, Volume,
    rasterize_obstacles_3d,
)
from tests3d.helpers3d import DEV, blob, dilate2d, import_or_skip, requires_cuda, smooth_noise, world_grid

pytestmark = requires_cuda
RES = 64
OUT = (360, 640)
DIL = 6
POP_MAX = 2.0
DETAIL_BUDGET_MS = 2.5

# draw(detail=None) of `_hash_scene()` at OUT, Camera(), sphere obstacle (0.75,0.6,0.5) r 0.1 rasterized at 128.
# Recorded 2026-11-09 by the test engineer AFTER the render engineer's 20:34 edit that added `detail=` (the
# pre-phase renderer could not be hashed without git: the lead can confirm it against tag v0.3-screen-space-baseline).
RENDER_HASH = {1.0: "b8328ba14455eb3f570499a75e767b3880dd66f2ebe6b46b66d17b9fcb1c11d2",
               0.5: "66a92025a588bb4ab50adaa51e44633cf66784062d013bd6f037ae8ef9bd08dc"}
RENDER_HASH_ENV = {"gpu": "NVIDIA GeForce RTX 5080", "torch": "2.11.0+cu128"}


# ----------------------------------------------------------------------------- fixtures / helpers
@pytest.fixture(scope="module")
def det_mod():
    return import_or_skip("vfx.detail3d", ("AdvectedDetail",))


@pytest.fixture(scope="module")
def rd():
    mod = import_or_skip("vfx.render3d", ("VolumeRenderer",))
    if "detail" not in inspect.signature(mod.VolumeRenderer.draw).parameters:
        pytest.skip("VolumeRenderer.draw has no `detail` parameter yet")
    return mod


def make_state(density: torch.Tensor, velocity=None, temperature=None, obstacles=()) -> FieldState3D:
    data = torch.zeros(NUM_CHANNELS_3D, RES, RES, RES, device=DEV)
    data[CH3["density"]] = density
    data[CH3["temperature"]] = AMBIENT_TEMP if temperature is None else temperature
    if velocity is not None:
        data[CH3["vel_x"]:CH3["vel_z"] + 1] = velocity
    solid = rasterize_obstacles_3d(list(obstacles), RES, DEV) if obstacles else torch.zeros(RES, RES, RES, device=DEV)
    data[CH3["density"]] *= (solid < 0.5)
    return FieldState3D(data=data, solid=solid, meta=FieldMeta3D.for_res(RES))


def uniform_velocity(u) -> torch.Tensor:
    v = torch.zeros(3, RES, RES, RES, device=DEV)
    for i in range(3):
        v[i] = u[i]
    return v


def swirl_velocity(omega: float = 2.0, up: float = 0.3) -> torch.Tensor:
    """Rotation about the vertical axis through the domain centre + an updraft (m/s)."""
    xx, yy, zz = world_grid(RES)
    v = torch.zeros(3, RES, RES, RES, device=DEV)
    v[0] = -omega * (zz - 0.5)
    v[2] = omega * (xx - 0.5)
    v[1] = up * torch.exp(-((xx - 0.5) ** 2 + (zz - 0.5) ** 2) / 0.05)
    return v


def plume_density() -> torch.Tensor:
    d = blob(RES, 0.5, 0.45, 0.5, 0.11) * 1.5
    return d.clamp(min=0)


def volume_of(st: FieldState3D) -> Volume:
    return Volume(st.ch("density"), st.ch("temperature"))


def new_detail(det_mod):
    torch.manual_seed(0)
    d = det_mod.AdvectedDetail(RES, device=DEV)
    d.reset()
    return d


def draw(r, st, cam, detail=None, solid=None, aux=False):
    kw = {"detail": detail} if detail is not None else {}
    if aux:
        f, a = r.draw(volume_of(st), cam, solid, return_aux=True, **kw)
        return f.clone(), {k: v.clone() for k, v in a.items()}
    return r.draw(volume_of(st), cam, solid, **kw).clone()


def lum(frame_hwc_u8) -> torch.Tensor:
    x = frame_hwc_u8.float() / 255.0
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def _hash_scene():
    d = (blob(64, 0.5, 0.4, 0.5, 0.12) * (1 + 0.5 * smooth_noise(64, 1, seed=7)[0])).clamp(min=0)
    t = AMBIENT_TEMP + 900 * blob(64, 0.5, 0.3, 0.5, 0.08)
    return Volume(d.contiguous(), t.contiguous())


def _iter_tensors(obj):
    """Yield torch tensors (and warp arrays converted with wp.to_torch) found in fields()."""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_tensors(v)
    else:
        try:
            import warp as wp
            if isinstance(obj, wp.array):
                yield wp.to_torch(obj)
        except Exception:  # noqa: BLE001
            pass


def fields_finite(d) -> int:
    bad = 0
    for t in _iter_tensors(d.fields()):
        if t.is_floating_point():
            bad += int((~torch.isfinite(t.float())).sum().item())
    return bad


# ----------------------------------------------------------------------------- interface
def test_interface(det_mod):
    d = new_detail(det_mod)
    st = make_state(plume_density(), swirl_velocity())
    d.step(st, SIM_DT)
    f = d.fields()
    assert isinstance(f, dict) and f, "fields() must return a non-empty dict of GPU resources"
    ts = list(_iter_tensors(f))
    assert ts and all(t.is_cuda for t in ts), "fields() tensors must live on the GPU"
    assert isinstance(d.timings, dict)
    assert fields_finite(d) == 0


# ----------------------------------------------------------------------------- detail=None
def test_detail_none_bit_identical_to_no_kwarg(rd):
    vol = _hash_scene()
    solid = rasterize_obstacles_3d([Obstacle3D("sphere", (0.75, 0.6, 0.5), (0.1, 0.1, 0.1))], 128)
    for ms in (1.0, 0.5):
        r = rd.VolumeRenderer(OUT, march_scale=ms)
        a = r.draw(vol, Camera(), solid).clone()
        b = r.draw(vol, Camera(), solid, detail=None).clone()
        assert torch.equal(a, b), f"march_scale {ms}: draw(detail=None) != draw()"
        fa, xa = r.draw(vol, Camera(), solid, return_aux=True)
        fa, xa = fa.clone(), {k: v.clone() for k, v in xa.items()}
        fb, xb = r.draw(vol, Camera(), solid, return_aux=True, detail=None)
        assert torch.equal(fa, fb) and all(torch.equal(xa[k], xb[k]) for k in xa)


def test_detail_none_matches_recorded_render_hash(rd):
    env = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
    if env != RENDER_HASH_ENV:
        pytest.skip(f"render hash recorded on {RENDER_HASH_ENV}, this machine {env}")
    vol = _hash_scene()
    solid = rasterize_obstacles_3d([Obstacle3D("sphere", (0.75, 0.6, 0.5), (0.1, 0.1, 0.1))], 128)
    for ms, h in RENDER_HASH.items():
        f = rd.VolumeRenderer(OUT, march_scale=ms).draw(vol, Camera(), solid, detail=None)
        got = hashlib.sha256(f.contiguous().cpu().numpy().tobytes()).hexdigest()
        assert got == h, f"march_scale {ms}: render without detail changed since it was recorded"


def test_detail_after_reset_with_detail_drawn_then_none(det_mod, rd):
    """Drawing with detail must not leave state behind: a later detail=None draw equals a fresh renderer's."""
    st = make_state(plume_density(), swirl_velocity())
    d = new_detail(det_mod)
    for _ in range(5):
        d.step(st, SIM_DT)
    r = rd.VolumeRenderer(OUT, march_scale=0.5)
    draw(r, st, Camera(), d)
    after = draw(r, st, Camera())
    fresh = draw(rd.VolumeRenderer(OUT, march_scale=0.5), st, Camera())
    assert torch.equal(after, fresh)


# ----------------------------------------------------------------------------- conservation / locality
def _detail_pair(det_mod, rd, st, steps=30, solid=None, cam=None):
    cam = cam or Camera()
    d = new_detail(det_mod)
    for _ in range(steps):
        d.step(st, SIM_DT)
    r = rd.VolumeRenderer(OUT, march_scale=1.0)
    f0, a0 = draw(r, st, cam, None, solid, aux=True)
    f1, a1 = draw(r, st, cam, d, solid, aux=True)
    return f0, a0, f1, a1


def test_detail_visible(det_mod, rd):
    st = make_state(plume_density(), swirl_velocity())
    f0, a0, f1, a1 = _detail_pair(det_mod, rd, st)
    fp = a0["alpha"].float() > 0.02
    diff = (f1.float() - f0.float()).abs().amax(-1)[fp]
    print(f"detail changes {(diff > 2).float().mean().item() * 100:.1f}% of footprint pixels by > 2 codes, "
          f"mean |diff| {diff.mean().item():.2f} codes")
    assert (diff > 2).float().mean().item() > 0.05, "detail draws are indistinguishable from no-detail draws"


def test_detail_conserves_mean_opacity_and_brightness(det_mod, rd):
    st = make_state(plume_density(), swirl_velocity(), AMBIENT_TEMP + 500 * blob(RES, 0.5, 0.35, 0.5, 0.07))
    f0, a0, f1, a1 = _detail_pair(det_mod, rd, st)
    fp = dilate2d(a0["alpha"].float() > 0.02, DIL)
    ma0, ma1 = a0["alpha"].float()[fp].mean().item(), a1["alpha"].float()[fp].mean().item()
    ml0, ml1 = lum(f0)[fp].mean().item(), lum(f1)[fp].mean().item()
    print(f"mean alpha {ma0:.4f} -> {ma1:.4f} ({(ma1 / ma0 - 1) * 100:+.2f}%), "
          f"mean luminance {ml0:.4f} -> {ml1:.4f} ({(ml1 / ml0 - 1) * 100:+.2f}%)")
    assert abs(ma1 / ma0 - 1) < 0.05, "detail changed the mean opacity of the plume by >= 5%"
    assert abs(ml1 / ml0 - 1) < 0.05, "detail changed the mean brightness of the plume by >= 5%"


def test_no_detail_in_empty_space_or_obstacles(det_mod, rd):
    """Two compact puffs, a sphere obstacle in the empty space between them: outside the dilated no-detail
    footprint (empty space, the obstacle, its shadow-free surroundings) the detail draw equals the plain draw."""
    d0 = blob(RES, 0.28, 0.5, 0.5, 0.05) + blob(RES, 0.75, 0.35, 0.5, 0.05)
    obs = [Obstacle3D("sphere", (0.52, 0.62, 0.5), (0.08, 0.08, 0.08))]
    st = make_state(d0, swirl_velocity(), obstacles=obs)
    solid = rasterize_obstacles_3d(obs, 128)
    f0, a0, f1, a1 = _detail_pair(det_mod, rd, st, solid=solid)
    fp = dilate2d(a0["alpha"].float() > 1e-3, DIL)
    out = ~fp
    # Shadow receivers (the ground plane y = 0 and the obstacle) may legitimately change: the detailed smoke
    # casts a detailed shadow. Opacity (alpha) may not change anywhere outside the smoke.
    from tests3d.helpers3d import camera_rays, rays_hit_sphere
    _, dirs = camera_rays(Camera(), *OUT)
    recv = (dirs[..., 1] < 0) | dilate2d(rays_hit_sphere(Camera(), *OUT, (0.52, 0.62, 0.5), 0.08), 2)
    d = (f1.float() - f0.float()).abs().amax(-1)[out & ~recv]
    da = (a1["alpha"].float() - a0["alpha"].float()).abs()[out]
    dob = (a1["alpha"].float() - a0["alpha"].float()).abs()[rays_hit_sphere(Camera(), *OUT, (0.52, 0.62, 0.5), 0.08) & out]
    d_recv = (f1.float() - f0.float()).abs().amax(-1)[out & recv]
    print(f"outside footprint: max frame diff {d.max().item():.0f} codes (excl. shadow receivers; receivers "
          f"{d_recv.max().item():.0f}), max alpha diff {da.max().item():.2e}, over the obstacle {dob.max().item():.2e}")
    assert da.max().item() < 2e-3, "detail added opacity where the 64^3 field is empty"
    assert dob.numel() == 0 or dob.max().item() < 2e-3, "detail added opacity in front of / inside the obstacle"
    assert d.max().item() <= 1, "detail painted light where the 64^3 field is empty"


def test_empty_volume_unchanged(det_mod, rd):
    st = make_state(torch.zeros(RES, RES, RES, device=DEV), swirl_velocity())
    f0, a0, f1, a1 = _detail_pair(det_mod, rd, st)
    assert (f1.float() - f0.float()).abs().max().item() <= 1
    assert a1["alpha"].float().max().item() < 1e-3


# ----------------------------------------------------------------------------- determinism / stability
def test_deterministic(det_mod, rd):
    st = make_state(plume_density(), swirl_velocity())

    def run():
        d = new_detail(det_mod)
        for _ in range(40):
            d.step(st, SIM_DT)
        ts = [t.clone() for t in _iter_tensors(d.fields())]
        return ts, draw(rd.VolumeRenderer(OUT, march_scale=0.5), st, Camera(), d)
    ta, fa = run()
    tb, fb = run()
    assert len(ta) == len(tb) and all(torch.equal(x, y) for x, y in zip(ta, tb)), "fields() differ between runs"
    assert torch.equal(fa, fb), "detail renders differ between identical runs"


def test_finite_over_2000_steps(det_mod, rd):
    st = make_state(plume_density(), swirl_velocity(omega=3.0, up=0.6))
    d = new_detail(det_mod)
    r = rd.VolumeRenderer(OUT, march_scale=0.5)
    _, a0 = draw(r, st, Camera(), None, aux=True)
    fp = a0["alpha"].float() > 0.02
    m0 = a0["alpha"].float()[fp].mean().item()
    worst = 0.0
    for i in range(2000):
        d.step(st, SIM_DT)
        if i % 250 == 249:
            assert fields_finite(d) == 0, f"non-finite detail fields at step {i + 1}"
            _, a = draw(r, st, Camera(), d, aux=True)
            assert torch.isfinite(a["alpha"].float()).all()
            m = a["alpha"].float()[fp].mean().item()
            worst = max(worst, abs(m / m0 - 1))
    print(f"2000 steps: worst mean-alpha change vs no detail {worst * 100:.1f}%")
    assert worst < 0.25, f"detail drifted: mean alpha off by {worst * 100:.1f}% after long advection"


# ----------------------------------------------------------------------------- moves with the flow
def _project_x(cam: Camera, p, h, w):
    pos = np.array(cam.position, np.float64)
    fwd = np.array(cam.target, np.float64) - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array(cam.up, np.float64))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    v = np.array(p, np.float64) - pos
    z = v @ fwd
    f = (h / 2) / math.tan(math.radians(cam.fov_y_deg) / 2)
    return np.array([f * (v @ right) / z, -f * (v @ up) / z])


def _best_shift(a: torch.Tensor, b: torch.Tensor, maxs: int):
    """Integer shift s (dx, dy) maximizing the normalized correlation of b(p) with a(p - s), |s| <= maxs."""
    best, arg = -2.0, (0, 0)
    h, w = a.shape
    for dy in range(-maxs // 3, maxs // 3 + 1):
        for dx in range(-maxs, maxs + 1):
            ya, yb = slice(max(0, -dy), h - max(0, dy)), slice(max(0, dy), h - max(0, -dy))
            xa, xb = slice(max(0, -dx), w - max(0, dx)), slice(max(0, dx), w - max(0, -dx))
            p, q = a[ya, xa].flatten(), b[yb, xb].flatten()
            p, q = p - p.mean(), q - q.mean()
            c = (p @ q / (p.norm() * q.norm() + 1e-12)).item()
            if c > best:
                best, arg = c, (dx, dy)
    return arg, best


def translating_eddy(cx: float, cy: float, u: float, omega: float) -> torch.Tensor:
    """u e_x + rigid rotation omega about the z axis (the view axis of Camera()) through the moving blob centre:
    the blob translates at u, its gas also turns in the image plane (vorticity 2 omega, so turbulence-keyed detail
    has something to key on). Rotation averages out of the mean image shift."""
    xx, yy, _ = world_grid(RES)
    v = torch.zeros(3, RES, RES, RES, device=DEV)
    v[0] = u - omega * (yy - cy)
    v[1] = omega * (xx - cx)
    return v


def moving_curl_noise(shift_x: float, vmax: float = 0.05, seed: int = 3, modes: int = 16) -> torch.Tensor:
    """Divergence-free small-scale velocity (curl of 16 random Fourier modes, wavelengths 0.08-0.2 m) that
    translates with the blob (evaluated at x - shift_x): co-moving 'sub-grid turbulence' for detail keyed on the
    coarse turbulence. Fixed normalization so |v| <= ~vmax."""
    g = np.random.default_rng(seed)
    xx, yy, zz = world_grid(RES)
    X = xx - shift_x
    A = torch.zeros(3, RES, RES, RES, device=DEV)
    for _ in range(modes):
        lam = g.uniform(0.08, 0.2)
        dvec = g.normal(size=3)
        k = 2 * math.pi / lam * dvec / np.linalg.norm(dvec)
        s = torch.sin(float(k[0]) * X + float(k[1]) * yy + float(k[2]) * zz + g.uniform(0, 2 * math.pi))
        a = g.normal(size=3)
        for i in range(3):
            A[i] += float(a[i]) * s
    dA = [torch.gradient(A[i]) for i in range(3)]                  # (d/dz, d/dy, d/dx) per component
    v = torch.stack([dA[2][1] - dA[1][0], dA[0][0] - dA[2][2], dA[1][2] - dA[0][1]])
    return v * (vmax / 0.2033)                                     # 0.2033 = max|curl| of this seed at vmax 1


def _blur2(x: torch.Tensor, s: float = 1.0) -> torch.Tensor:
    from vfx.eval3d.metrics import gaussian_blur
    return gaussian_blur(x[None].expand(3, -1, -1), s)[0]


@pytest.mark.parametrize("n", [4, 8])
def test_detail_moves_with_the_flow(det_mod, rd, n):
    """A blob translating in +x at u = 0.3 m/s (density moved analytically every step) carrying co-moving sub-grid
    turbulence (|v'| <= 0.05 m/s, so the detail keyed on turbulence has something to key on and local displacements
    stay < 1 px per step). Detail layer D = alpha(draw with detail) - alpha(draw without) (float16 opacity: no 8-bit
    quantization), Gaussian sigma 1 px against march sampling noise. After n more steps D must be D shifted by the
    projected translation (25%), along x, with correlation > 0.3. A screen/world-stuck layer peaks at (0, 0)."""
    u, sig = 0.3, 0.1
    cam = Camera()
    r = rd.VolumeRenderer(OUT, march_scale=1.0)
    d = new_detail(det_mod)
    c0 = 0.38
    layers = {}
    for i in range(30 + n + 1):
        cx = c0 + u * i * SIM_DT
        vel = uniform_velocity((u, 0.0, 0.0)) + moving_curl_noise(u * i * SIM_DT)
        st = make_state(blob(RES, cx, 0.45, 0.5, sig) * 1.5, vel)
        d.step(st, SIM_DT)
        if i in (30, 30 + n):
            _, a0 = draw(r, st, cam, None, aux=True)
            _, a1 = draw(r, st, cam, d, aux=True)
            layers[i] = _blur2(a1["alpha"].float() - a0["alpha"].float())
    a, b = layers[30], layers[30 + n]
    e = _project_x(cam, (c0 + u * (30 + n) * SIM_DT, 0.45, 0.5), *OUT) - _project_x(cam, (c0 + u * 30 * SIM_DT, 0.45, 0.5), *OUT)
    exp = abs(e[0])
    (dx, dy), corr = _best_shift(a, b, int(exp * 1.6) + 4)
    _, c_stuck = _best_shift(a, b, 0)
    print(f"n={n}: expected detail shift {exp:.1f} px (x), measured ({dx}, {dy}) px, corr {corr:.3f}; "
          f"corr at zero shift (stuck component) {c_stuck:.3f}; D std {a.std().item():.2e}")
    assert a.std().item() > 1e-4, "detail layer too weak to measure"
    assert corr > 0.3, f"detail layer is not a shifted copy of itself (corr {corr:.2f}): re-rolled or boiling"
    assert abs(abs(dx) - exp) <= 0.25 * exp, (f"detail moved ({dx}, {dy}) px, the flow moved {exp:.1f} px "
                                              f"(zero-shift corr {c_stuck:.2f}: part of the detail is world/screen-fixed)")
    assert abs(dy) <= max(2, 0.15 * exp), f"detail moved vertically ({dy} px) under a horizontal flow"


def test_no_periodic_pop(det_mod, rd):
    """Quasi-steady plume: static density, gas swirling and rising through it (vorticity present, so detail keyed on
    the coarse turbulence is non-zero). The detail layer must change smoothly:
    burst = max_t / median_t of mean |D_t - D_{t-1}| on the footprint <= POP_MAX."""
    st = make_state(plume_density(), swirl_velocity(omega=2.0, up=0.4))
    d = new_detail(det_mod)
    r = rd.VolumeRenderer(OUT, march_scale=0.5)
    base = lum(draw(r, st, Camera()))
    _, a0 = draw(r, st, Camera(), None, aux=True)
    fp = a0["alpha"].float() > 0.05
    for _ in range(30):
        d.step(st, SIM_DT)
    prev, ch = None, []
    for _ in range(150):
        d.step(st, SIM_DT)
        D = lum(draw(r, st, Camera(), d)) - base
        if prev is not None:
            ch.append((D - prev).abs()[fp].mean().item())
        prev = D
    ch = np.array(ch)
    burst = ch.max() / max(np.median(ch), 1e-12)
    spec = np.abs(np.fft.rfft(ch - ch.mean()))
    print(f"detail-layer change per step: median {np.median(ch):.2e}, max {ch.max():.2e}, burst {burst:.2f} "
          f"(<= {POP_MAX}); strongest periodic component at {np.argmax(spec[1:]) + 1} cycles / {len(ch)} steps")
    assert np.median(ch) > 0, "detail is frozen in a flowing plume"
    assert burst <= POP_MAX, f"detail pops: a step changes {burst:.2f}x the median step (> {POP_MAX})"


# ----------------------------------------------------------------------------- cost
def test_cost_within_detail_budget_1440p(det_mod, rd):
    st = make_state(plume_density(), swirl_velocity(), AMBIENT_TEMP + 500 * blob(RES, 0.5, 0.35, 0.5, 0.07))
    d = new_detail(det_mod)
    r = rd.VolumeRenderer((1440, 2560), march_scale=0.5)
    cam = Camera()
    vol = volume_of(st)

    def timed(fn, n=40, warm=10):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]
        for a, b in ev:
            a.record()
            fn()
            b.record()
        torch.cuda.synchronize()
        return float(np.median([a.elapsed_time(b) for a, b in ev]))
    t_step = timed(lambda: d.step(st, SIM_DT))
    t_plain = timed(lambda: r.draw(vol, cam))

    def with_detail():
        d.step(st, SIM_DT)
        r.draw(vol, cam, detail=d)
    t_both = timed(with_detail)
    extra = t_both - t_step - t_plain
    total = t_step + max(extra, 0.0)
    msg = (f"1440p ms0.5: step {t_step:.3f} ms, render plain {t_plain:.3f}, render+step with detail {t_both:.3f} "
           f"-> detail cost {total:.3f} ms (budget {DETAIL_BUDGET_MS}); timings {d.timings}")
    print(msg)
    try:
        from bench3d.run import gpu_check
        g = gpu_check()
    except Exception as e:  # noqa: BLE001
        g = {"shared": None, "reasons": [repr(e)]}
    if g.get("shared") is not False:
        pytest.skip(f"GPU shared/unknown ({'; '.join(g.get('reasons', []))}); measured: {msg}")
    assert total <= DETAIL_BUDGET_MS, msg
