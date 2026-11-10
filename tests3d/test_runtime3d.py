"""vfx.runtime3d contract (CLAUDE.md "3D phase"), from the interface only. Untrained weights in this
phase, so these are contract tests (shapes, dtypes, finiteness, sparsity semantics, no host syncs),
not quality tests.

    classical_upscale_3d(state, factor=4, mode="trilinear") -> Volume
    VolumeUpscaler(checkpoint_path=None, device="cuda", **arch).infer(state) -> Volume  (empty bricks skipped)
    ScreenDetail(checkpoint_path=None, device="cuda", **arch).infer(frame, aux=None) -> (h,w,3) uint8 cuda
Inputs are built directly as FieldState3D, so nothing here needs vfx.sim3d.
Tolerances stated here: trilinear of an affine field exact in the interior (1e-4 rel for float32 output,
2e-3 rel for float16); sparse vs dense within 1% of the max density in and around active bricks;
regions > 0.28 m from any smoke stay empty (density <= 1e-3, |T - ambient| <= 1 K).
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from vfx.core.fields3d import (
    AMBIENT_TEMP, CH3, DOMAIN_SIZE, NUM_CHANNELS_3D, FieldMeta3D, FieldState3D, Volume,
)
from tests3d.helpers3d import DEV, SyncCounter, compact_blob, import_or_skip, requires_cuda, smooth_noise, world_grid

pytestmark = requires_cuda
LR = 64


@pytest.fixture(scope="module")
def rt():
    return import_or_skip("vfx.runtime3d")


def state_from(density, temperature, velocity=None, fuel=None) -> FieldState3D:
    res = density.shape[-1]
    data = torch.zeros(NUM_CHANNELS_3D, res, res, res, device=DEV)
    data[CH3["density"]] = density
    data[CH3["temperature"]] = temperature
    if velocity is not None:
        data[CH3["vel_x"]:CH3["vel_z"] + 1] = velocity
    if fuel is not None:
        data[CH3["fuel"]] = fuel
    return FieldState3D(data=data, solid=torch.zeros(res, res, res, device=DEV), meta=FieldMeta3D.for_res(res))


def plume_state(res=LR):
    env = compact_blob(res, 0.5, 0.4, 0.5, 0.3)
    n = smooth_noise(res, 5, seed=7, coarse=8)
    d = (env * (0.7 + 0.3 * n[0])).clamp(min=0)
    t = AMBIENT_TEMP + 800.0 * env ** 2 * (0.5 + 0.5 * n[1]).clamp(0, 1)
    v = 0.5 * n[2:5] * env
    v[1] += 1.0 * env
    return state_from(d, t, v, fuel=0.2 * env ** 3)


def empty_state(res=LR):
    z = torch.zeros(res, res, res, device=DEV)
    return state_from(z, torch.full_like(z, AMBIENT_TEMP))


def corner_state(res=LR):
    env = compact_blob(res, 0.2, 0.2, 0.2, 0.12)
    return state_from(env, AMBIENT_TEMP + 600.0 * env)


def check_volume(v, res):
    assert isinstance(v, Volume), f"expected Volume, got {type(v)}"
    for name in ("density", "temperature"):
        t = getattr(v, name)
        assert tuple(t.shape) == (res, res, res), f"{name} shape {tuple(t.shape)} != {(res,) * 3}"
        assert t.is_cuda and t.dtype in (torch.float16, torch.float32), f"{name}: {t.dtype} {t.device}"
        bad = (~torch.isfinite(t)).sum().item()
        assert bad == 0, f"{bad} non-finite values in {name}"
    assert v.density.min().item() >= 0, f"Volume.density must be >= 0 (min {v.density.min().item():.3e})"


def timings_ok(obj):
    t = getattr(obj, "timings", None)
    assert isinstance(t, dict) and t, f"{type(obj).__name__}.timings must be a non-empty dict after infer, got {t!r}"
    assert all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in t.values()), t


# ----------------------------------------------------------------------------- classical
def _affine(res):
    xx, yy, zz = world_grid(res)
    return 1.0 + 0.5 * xx + 0.3 * yy + 0.2 * zz, 400.0 + 200.0 * xx + 50.0 * yy - 100.0 * zz


@pytest.mark.parametrize("factor", [4, 2])
def test_classical_upscale_affine_exact_in_interior(rt, factor):
    if not hasattr(rt, "classical_upscale_3d"):
        pytest.skip("vfx.runtime3d does not export classical_upscale_3d yet")
    d, t = _affine(LR)
    v = rt.classical_upscale_3d(state_from(d, t), factor=factor, mode="trilinear")
    hr = LR * factor
    check_volume(v, hr)
    ed, et = _affine(hr)
    m = slice(factor, hr - factor)                      # interior: HR centres strictly inside LR centres
    rel = 2e-3 if v.density.dtype == torch.float16 else 1e-4
    err_d = ((v.density.float() - ed).abs() / ed.abs())[m, m, m].max().item()
    err_t = ((v.temperature.float() - et).abs() / et.abs())[m, m, m].max().item()
    print(f"factor {factor} ({v.density.dtype}): interior max rel err density {err_d:.2e}, temperature {err_t:.2e}")
    assert err_d < rel and err_t < rel


def test_classical_upscale_default_is_4x_trilinear(rt):
    if not hasattr(rt, "classical_upscale_3d"):
        pytest.skip("vfx.runtime3d does not export classical_upscale_3d yet")
    s = plume_state()
    v = rt.classical_upscale_3d(s)
    check_volume(v, LR * 4)
    ref = F.interpolate(s.ch("density")[None, None], scale_factor=4, mode="trilinear", align_corners=False)[0, 0]
    err = (v.density.float() - ref).abs().max().item()
    print(f"default classical_upscale_3d vs F.interpolate trilinear: max abs err {err:.2e}")
    assert err < 2e-3 * max(ref.max().item(), 1.0)


def test_classical_upscale_no_host_sync(rt):
    if not hasattr(rt, "classical_upscale_3d"):
        pytest.skip("vfx.runtime3d does not export classical_upscale_3d yet")
    s = plume_state()
    rt.classical_upscale_3d(s)
    with SyncCounter() as c:
        rt.classical_upscale_3d(s)
    assert not c.syncs, f"{len(c.syncs)} host syncs: {c.syncs[:3]}"


# ----------------------------------------------------------------------------- VolumeUpscaler (route A)
@pytest.fixture(scope="module")
def vup(rt):
    if not hasattr(rt, "VolumeUpscaler"):
        pytest.skip("vfx.runtime3d does not export VolumeUpscaler yet")
    torch.manual_seed(0)
    return rt.VolumeUpscaler(None, device="cuda")


def _reset(u):
    if hasattr(u, "reset"):
        u.reset()


def test_volume_upscaler_contract(vup):
    _reset(vup)
    v = vup.infer(plume_state())
    torch.cuda.synchronize()
    check_volume(v, LR * 4)
    v2 = vup.infer(plume_state())                    # second frame: temporal state path
    check_volume(v2, LR * 4)
    timings_ok(vup)
    print("VolumeUpscaler timings:", {k: round(x, 3) for k, x in vup.timings.items()})


def test_volume_upscaler_empty_state_is_empty(vup):
    """Empty bricks are skipped: an empty LR state gives an empty HR volume (density 0, T ambient)."""
    _reset(vup)
    v = vup.infer(empty_state())
    check_volume(v, LR * 4)
    dmax = v.density.float().abs().max().item()
    tdev = (v.temperature.float() - AMBIENT_TEMP).abs().max().item()
    print(f"empty state: max density {dmax:.2e}, max |T - ambient| {tdev:.3f} K")
    assert dmax <= 1e-3 and tdev <= 1.0


def test_volume_upscaler_far_field_untouched(vup):
    """Smoke only in one corner: HR voxels > 0.28 m away (x, y, z > 0.6) stay empty."""
    _reset(vup)
    v = vup.infer(corner_state())
    check_volume(v, LR * 4)
    hr = LR * 4
    k = int(0.6 * hr)
    far_d = v.density[k:, k:, k:].float().abs().max().item()
    far_t = (v.temperature[k:, k:, k:].float() - AMBIENT_TEMP).abs().max().item()
    near = v.density[: hr // 2, : hr // 2, : hr // 2].float().sum().item()
    print(f"corner smoke: far-field max density {far_d:.2e}, max |T - ambient| {far_t:.3f}; near sum {near:.1f}")
    assert near > 0, "smoke vanished in the active corner"
    assert far_d <= 1e-3 and far_t <= 1.0


def _dense_switch(u):
    """(attribute, dense value) if the instance exposes a sparse/dense switch, else None."""
    for attr, dense_val in (("sparse", False), ("dense", True), ("skip_empty", False), ("use_bricks", False)):
        if isinstance(getattr(u, attr, None), bool):
            return attr, dense_val
    return None


def test_volume_upscaler_sparse_equals_dense(rt):
    if not hasattr(rt, "VolumeUpscaler"):
        pytest.skip("vfx.runtime3d does not export VolumeUpscaler yet")
    torch.manual_seed(0)
    sp = rt.VolumeUpscaler(None, device="cuda")
    sw = _dense_switch(sp)
    if sw is None:
        pytest.skip("VolumeUpscaler exposes no bool sparse/dense switch (sparse/dense/skip_empty/use_bricks)")
    torch.manual_seed(0)
    de = rt.VolumeUpscaler(None, device="cuda")
    setattr(de, sw[0], sw[1])
    s = plume_state()
    _reset(sp), _reset(de)
    a, b = sp.infer(s), de.infer(s)
    torch.cuda.synchronize()
    # compare in and around active LR bricks (4^3 LR = 16^3 HR), dilated by one brick
    act = ((s.ch("density") > 1e-3) | (s.ch("temperature") > AMBIENT_TEMP + 1.0)).float()[None, None]
    act = F.max_pool3d(F.max_pool3d(act, 4, 4), 3, 1, 1)
    m = F.interpolate(act, scale_factor=16, mode="nearest")[0, 0] > 0.5
    scale = max(b.density.float()[m].abs().max().item(), 1e-6)
    err = (a.density.float() - b.density.float())[m].abs().max().item()
    terr = (a.temperature.float() - b.temperature.float())[m].abs().max().item()
    print(f"sparse vs dense ({sw[0]}={sw[1]}): max |d| diff {err:.3e} (max d {scale:.3e}), max |T| diff {terr:.3f} K; "
          f"active region {m.float().mean().item() * 100:.1f}% of voxels")
    assert err <= 1e-2 * scale, f"sparse and dense differ by {err / scale * 100:.2f}% of max density"
    assert terr <= 1e-2 * max(b.temperature.float()[m].max().item() - AMBIENT_TEMP, 1.0)


def test_volume_upscaler_deterministic(rt):
    if not hasattr(rt, "VolumeUpscaler"):
        pytest.skip("vfx.runtime3d does not export VolumeUpscaler yet")
    outs = []
    for _ in range(2):
        torch.manual_seed(0)
        u = rt.VolumeUpscaler(None, device="cuda")
        _reset(u)
        outs.append(u.infer(plume_state()).density.clone())
    assert torch.equal(outs[0], outs[1]), "same seed + input gave different HR volumes"


def test_volume_upscaler_no_host_sync(vup):
    s = plume_state()
    _reset(vup)
    vup.infer(s)
    vup.infer(s)
    with SyncCounter() as c:
        vup.infer(s)
    print(f"VolumeUpscaler.infer host syncs seen by torch: {len(c.syncs)}")
    assert not c.syncs, f"{len(c.syncs)} host syncs in infer (spec: none in the frame loop): {c.syncs[:3]}"


# ----------------------------------------------------------------------------- ScreenDetail (route B)
@pytest.fixture(scope="module")
def sdet(rt):
    if not hasattr(rt, "ScreenDetail"):
        pytest.skip("vfx.runtime3d does not export ScreenDetail yet")
    torch.manual_seed(0)
    return rt.ScreenDetail(None, device="cuda")


def make_frame(h, w, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    base = F.interpolate(torch.rand(1, 3, 18, 32, device=DEV, generator=g), size=(h, w), mode="bilinear")[0]
    return (base.permute(1, 2, 0) * 255).to(torch.uint8).contiguous()


@pytest.mark.parametrize("hw", [(1440, 2560), (720, 1280), (721, 1283)], ids=["1440p", "720p", "odd"])
def test_screen_detail_contract(sdet, hw):
    f = make_frame(*hw)
    out = sdet.infer(f, None)
    torch.cuda.synchronize()
    assert tuple(out.shape) == (*hw, 3), f"expected {(*hw, 3)}, got {tuple(out.shape)}"
    assert out.dtype == torch.uint8 and out.is_cuda
    timings_ok(sdet)


def test_screen_detail_deterministic(rt):
    if not hasattr(rt, "ScreenDetail"):
        pytest.skip("vfx.runtime3d does not export ScreenDetail yet")
    outs = []
    for _ in range(2):
        torch.manual_seed(0)
        sd = rt.ScreenDetail(None, device="cuda")
        if hasattr(sd, "reset"):
            sd.reset()
        outs.append(sd.infer(make_frame(720, 1280)).clone())
    assert torch.equal(outs[0], outs[1])


def test_screen_detail_no_host_sync(sdet):
    f = make_frame(1440, 2560)
    sdet.infer(f)
    sdet.infer(f)
    with SyncCounter() as c:
        sdet.infer(f)
    print(f"ScreenDetail.infer host syncs seen by torch: {len(c.syncs)}")
    assert not c.syncs, f"{len(c.syncs)} host syncs in infer: {c.syncs[:3]}"
