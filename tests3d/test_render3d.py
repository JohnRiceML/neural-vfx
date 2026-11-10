"""VolumeRenderer contract (CLAUDE.md "3D phase"), from the interface only.

    VolumeRenderer(out_size, device="cuda", march_scale=1.0).draw(volume, camera, solid=None)
        -> (out_h, out_w, 3) uint8 cuda, row 0 = top of screen;  .timings: dict[str, float]

out_size is taken as (h, w), the 2D Renderer's convention; a non-square size checks it.
Tolerances stated here (no spec number exists):
  * outside the projected footprint of a compact smoke blob the image equals the empty render (<= 2 levels)
  * march_scale 0.5 vs 1.0 PSNR >= 30 dB, 0.25 vs 1.0 >= 25 dB (on a smooth plume volume)
"""
from __future__ import annotations

import math

import pytest
import torch

from vfx.core.fields3d import AMBIENT_TEMP, Camera, Volume
from tests3d.helpers3d import (
    DEV, ambient_like, camera_rays, compact_blob, dilate2d, import_or_skip, psnr, rays_hit_box, rays_hit_sphere, requires_cuda,
    smooth_noise,
)

pytestmark = requires_cuda
H, W = 360, 640              # small, non-square, 16:9 like 1440p
CAM = Camera()
BLOB_C = (0.5, 0.45, 0.5)    # on the mirror plane x = 0.5: horizontal handedness does not matter
BLOB_R = 0.2


@pytest.fixture(scope="module")
def rd():
    return import_or_skip("vfx.render3d", ("VolumeRenderer",))


def empty_volume(res=64):
    z = torch.zeros(res, res, res, device=DEV)
    return Volume(density=z, temperature=ambient_like(z))


def smoke_volume(res=64, scale=1.0, temp=None, center=BLOB_C, radius=BLOB_R):
    d = scale * compact_blob(res, *center, radius)
    t = ambient_like(d) if temp is None else AMBIENT_TEMP + (temp - AMBIENT_TEMP) * (d / max(scale, 1e-9))
    return Volume(density=d, temperature=t)


def draw(r, vol, cam=CAM, solid=None):
    """draw() may return a reused output buffer (allowed: no per-frame allocation), so copy it."""
    img = r.draw(vol, cam, solid).clone()
    torch.cuda.synchronize()
    return img


def lum(img):
    f = img.float()
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def footprint(res=64, center=BLOB_C, radius=BLOB_R, h=H, w=W, pad_px=3):
    """Pixels whose ray passes within (radius + 2 voxels) of the blob, dilated by pad_px."""
    m = rays_hit_sphere(CAM, h, w, center, radius + 2.0 / res)
    return dilate2d(m, pad_px)


# ----------------------------------------------------------------------------- contract
def test_output_contract(rd):
    r = rd.VolumeRenderer((H, W), device="cuda")
    img = draw(r, smoke_volume())
    assert img.shape == (H, W, 3), f"expected (out_h, out_w, 3) = {(H, W, 3)} for out_size=(h, w), got {tuple(img.shape)}"
    assert img.dtype == torch.uint8 and img.is_cuda
    t = getattr(r, "timings", None)
    assert isinstance(t, dict), "VolumeRenderer.timings must be a dict"
    assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in t.values()), t


@pytest.mark.parametrize("res", [32, 64, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_any_resolution_and_dtype(rd, res, dtype):
    """Volume: any resolution, float16 or float32 (fields3d.py)."""
    r = rd.VolumeRenderer((H, W))
    v = smoke_volume(res, temp=1200.0)
    img = draw(r, Volume(density=v.density.to(dtype), temperature=v.temperature.to(dtype)))
    assert img.shape == (H, W, 3) and img.dtype == torch.uint8


def test_solid_argument_accepted(rd):
    from vfx.core.fields3d import Obstacle3D, rasterize_obstacles_3d
    r = rd.VolumeRenderer((H, W))
    solid = rasterize_obstacles_3d([Obstacle3D("sphere", (0.5, 0.3, 0.5), (0.1, 0.1, 0.1))], 64)
    img = draw(r, smoke_volume(), solid=solid)
    assert img.shape == (H, W, 3)


# ----------------------------------------------------------------------------- background
def test_empty_volume_renders_background(rd):
    """An empty volume is invisible: the image does not depend on the (empty) volume's resolution, and
    there is no step in brightness where rays start / stop crossing the domain box (checked on the rows
    through the box, 6 px either side of each box edge)."""
    r = rd.VolumeRenderer((H, W))
    a = draw(r, empty_volume(64))
    b = draw(r, empty_volume(16))
    assert torch.equal(a, b), "empty volumes of different resolution render differently"
    hit = rays_hit_box(CAM, H, W)
    L = lum(a)
    steps = []
    for row in range(H // 4, 3 * H // 4, 8):
        h = hit[row].int()
        edges = torch.nonzero(h[1:] != h[:-1]).flatten().tolist()
        for e in edges:
            if 6 <= e < W - 7:
                steps.append(abs(L[row, e - 5:e + 1].mean().item() - L[row, e + 1:e + 7].mean().item()))
    assert steps, "camera framing: the box edges should be on screen"
    print(f"empty volume: {len(steps)} box-edge samples, max brightness step {max(steps):.2f} levels")
    assert max(steps) <= 3.0, f"empty domain box is visible (brightness step {max(steps):.2f} at its edge)"


def test_smoke_only_changes_its_footprint(rd):
    r = rd.VolumeRenderer((H, W))
    bg = draw(r, empty_volume())
    img = draw(r, smoke_volume(scale=4.0))
    fp = footprint()
    # the renderer may draw a ground plane at y = 0 that receives the smoke's shadow: that is scene, not
    # background, so pixels whose rays reach the plane y = 0 are excluded
    o, dirs = camera_rays(CAM, H, W)
    floor = dirs[..., 1] < 0
    d = (img.int() - bg.int()).abs().amax(-1)
    sel = ~fp & ~floor
    out_max = d[sel].max().item()
    in_mean = d[fp].float().mean().item()
    floor_changed = ((d > 2) & floor & ~fp).float().mean().item()
    print(f"blob footprint {fp.float().mean().item() * 100:.1f}% of the image; mean change inside "
          f"{in_mean:.1f} levels, max change outside (excl. ground plane) {out_max}; "
          f"ground-plane pixels changed (shadow) {floor_changed * 100:.2f}% of the image")
    assert in_mean > 5, "smoke blob is not visible"
    assert out_max <= 2, f"pixels whose rays miss the smoke changed by up to {out_max} levels"


# ----------------------------------------------------------------------------- opacity / emission
def test_more_density_more_opacity(rd):
    r = rd.VolumeRenderer((H, W))
    bg = draw(r, empty_volume()).float()
    fp = footprint(pad_px=0)
    devs = {}
    for s in (0.25, 1.0, 4.0, 16.0):
        img = draw(r, smoke_volume(scale=s)).float()
        devs[s] = (img - bg).abs().mean(-1)[fp].mean().item()
    print("mean |img - background| in footprint by density scale:", {k: round(v, 2) for k, v in devs.items()})
    ks = sorted(devs)
    for a, b in zip(ks, ks[1:]):
        assert devs[b] > devs[a], devs


def test_hot_volume_emits(rd):
    """Same density, hot (1500 K core) vs ambient: hot is brighter and warmer-coloured (R-B larger)."""
    r = rd.VolumeRenderer((H, W))
    fp = footprint(pad_px=0)
    cold = draw(r, smoke_volume(scale=1.0)).float()
    hot = draw(r, smoke_volume(scale=1.0, temp=1500.0)).float()
    lc, lh = lum(cold)[fp].mean().item(), lum(hot)[fp].mean().item()
    wc = (cold[..., 0] - cold[..., 2])[fp].mean().item()
    wh = (hot[..., 0] - hot[..., 2])[fp].mean().item()
    print(f"footprint luminance cold {lc:.1f} hot {lh:.1f}; R-B cold {wc:.1f} hot {wh:.1f}")
    assert lh > lc + 5, "hot smoke is not brighter than cold smoke"
    assert wh > wc + 5, "hot smoke is not warmer-coloured than cold smoke"


# ----------------------------------------------------------------------------- camera
def orbit(cam: Camera, deg: float) -> Camera:
    tx, ty, tz = cam.target
    px, py, pz = cam.position
    a = math.radians(deg)
    dx, dz = px - tx, pz - tz
    return Camera(position=(tx + dx * math.cos(a) - dz * math.sin(a), py, tz + dx * math.sin(a) + dz * math.cos(a)),
                  target=cam.target, up=cam.up, fov_y_deg=cam.fov_y_deg)


def test_camera_orbit_changes_image(rd):
    r = rd.VolumeRenderer((H, W))
    a = smoke_volume(scale=4.0, center=(0.3, 0.45, 0.3), radius=0.12)
    b = smoke_volume(scale=4.0, center=(0.7, 0.55, 0.65), radius=0.15, temp=1300.0)
    vol = Volume(density=a.density + b.density, temperature=torch.maximum(a.temperature, b.temperature))
    i0 = draw(r, vol, CAM)
    i1 = draw(r, vol, orbit(CAM, 60.0))
    i0b = draw(r, vol, CAM)
    assert torch.equal(i0, i0b), "same camera must give the same image (deterministic draw)"
    diff = (i0.float() - i1.float()).abs().mean().item()
    print(f"orbit 60 deg: mean abs pixel change {diff:.2f}")
    assert diff > 1.0


# ----------------------------------------------------------------------------- march_scale
def plume_volume(res=64):
    n = smooth_noise(res, 2, seed=11, coarse=8)
    env = compact_blob(res, 0.5, 0.45, 0.5, 0.35)
    d = (2.0 * env * (0.6 + 0.4 * n[0])).clamp(min=0)
    t = AMBIENT_TEMP + 900.0 * env ** 2 * (0.5 + 0.5 * n[1]).clamp(0, 1)
    return Volume(density=d, temperature=t)


@pytest.mark.parametrize("scale,min_psnr", [(0.5, 30.0), (0.25, 25.0)])
def test_march_scale_close_to_full_res(rd, scale, min_psnr):
    h, w = 720, 1280
    vol = plume_volume()
    full = draw(rd.VolumeRenderer((h, w), march_scale=1.0), vol)
    r = rd.VolumeRenderer((h, w), march_scale=scale)
    low = draw(r, vol)
    assert low.shape == (h, w, 3) and low.dtype == torch.uint8, "march_scale must still return out_size"
    p = psnr(low, full)
    print(f"march_scale {scale}: PSNR vs 1.0 = {p:.2f} dB")
    assert p >= min_psnr, f"march_scale {scale} PSNR {p:.2f} dB < {min_psnr}"


# ----------------------------------------------------------------------------- robustness
def test_extreme_values_render_validly(rd):
    """uint8 cannot hold NaN, so check the float path does not saturate into garbage: a very dense /
    very hot volume and NaN-free inputs at the extremes still give a valid, deterministic image."""
    r = rd.VolumeRenderer((H, W))
    d = 1e3 * compact_blob(64, *BLOB_C, BLOB_R)
    t = AMBIENT_TEMP + 5000.0 * (d > 0)
    a = draw(r, Volume(density=d, temperature=t))
    b = draw(r, Volume(density=d, temperature=t))
    assert a.shape == (H, W, 3) and torch.equal(a, b)
    bg = draw(r, empty_volume())
    fp = footprint(pad_px=0)
    assert (a.int() - bg.int()).abs().amax(-1)[fp].float().mean().item() > 5, "extreme volume rendered as empty"


def test_draw_no_host_sync(rd):
    """CLAUDE.md: no host syncs / CPU copies in the frame loop (only torch-visible syncs are detectable)."""
    from tests3d.helpers3d import SyncCounter
    r = rd.VolumeRenderer((H, W))
    vol = plume_volume()
    r.draw(vol, CAM)
    r.draw(vol, CAM)
    with SyncCounter() as c:
        r.draw(vol, orbit(CAM, 10.0))
    print(f"VolumeRenderer.draw host syncs seen by torch: {len(c.syncs)}")
    assert not c.syncs, f"{len(c.syncs)} host syncs in draw: {c.syncs[:3]}"
