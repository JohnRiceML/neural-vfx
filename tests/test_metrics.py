"""Unit tests for vfx.eval.metrics (the metrics themselves must be trustworthy before they gate anything)."""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vfx.core.fields import CH, FieldMeta, FieldState, NUM_CHANNELS
from vfx.eval import metrics as M

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def turbulent_field(n=256, seed=0, slope=-3.0):
    """Random field with a power-law spectrum (smoke-like), >= 0."""
    g = torch.Generator().manual_seed(seed)
    f = torch.fft.fftfreq(n, d=1.0 / n)
    ky, kx = torch.meshgrid(f, f, indexing="ij")
    k = torch.sqrt(kx ** 2 + ky ** 2).clamp_min(1.0)
    amp = k ** (slope / 2)
    phase = torch.randn(n, n, generator=g) + 1j * torch.randn(n, n, generator=g)
    x = torch.fft.ifft2(amp * phase).real
    x = (x - x.min()) / (x.max() - x.min())
    return x.to(DEV)


def blur(x, sigma):
    r = int(3 * sigma) + 1
    t = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    y = F.conv2d(x[None, None], k.view(1, 1, 1, -1), padding=(0, r))
    y = F.conv2d(y, k.view(1, 1, -1, 1), padding=(r, 0))
    return y[0, 0]


def gaussian_blob(n, cx, cy, sigma):
    c = (torch.arange(n, device=DEV, dtype=torch.float32) + 0.5) / n
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    return torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))


# ----------------------------------------------------------------------------- spectrum
def test_spectrum_identity_is_zero():
    x = turbulent_field()
    assert M.spectrum_error(x, x) < 1e-9
    assert M.spectrum_error(x.cpu().numpy(), x.cpu().numpy()) < 1e-9


def test_spectrum_parseval():
    x = turbulent_field(64)
    e = M.radial_power_spectrum(x)
    # total power over all shells within the inscribed circle <= mean(x^2); DC == mean^2
    assert math.isclose(e[0].item(), x.mean().item() ** 2, rel_tol=1e-4)
    assert e.sum().item() <= (x ** 2).mean().item() * (1 + 1e-4)


def test_spectrum_blur_scores_worse_monotonic():
    x = turbulent_field()
    e1 = M.spectrum_error(blur(x, 1.0), x)
    e2 = M.spectrum_error(blur(x, 2.0), x)
    e4 = M.spectrum_error(blur(x, 4.0), x)
    assert 0 < e1 < e2 < e4, (e1, e2, e4)


def test_spectrum_bicubic_of_pooled_worse_than_truth():
    x = turbulent_field(256)
    lr = F.avg_pool2d(x[None, None], 4)
    up = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)[0, 0]
    assert M.spectrum_error(up, x) > 0.3        # a smooth upscale misses a lot of high-k energy


def test_spectrum_shift_invariant_and_noise_penalized():
    x = turbulent_field()
    assert M.spectrum_error(torch.roll(x, (17, -5), (0, 1)), x) < 1e-5   # spectrum ignores phase
    noisy = x + 0.2 * torch.randn_like(x)
    assert M.spectrum_error(noisy, x) > 0.1


def test_spectrum_batched_and_empty():
    x = torch.stack([turbulent_field(seed=i) for i in range(3)])
    assert M.spectrum_error(x, x) < 1e-9
    z = torch.zeros(64, 64, device=DEV)
    assert math.isfinite(M.spectrum_error(z, z))
    with pytest.raises(ValueError):
        M.spectrum_error(torch.zeros(4, 8), torch.zeros(4, 8))


# ----------------------------------------------------------------------------- l1
def test_l1_basic():
    a = torch.zeros(4, 4)
    b = torch.full((4, 4), 0.5)
    assert M.l1(a, a) == 0.0
    assert math.isclose(M.l1(a, b), 0.5)
    with pytest.raises(ValueError):
        M.l1(a, torch.zeros(3, 3))


# ----------------------------------------------------------------------------- warp / flicker
def test_warp_integer_shift_exact():
    n = 64
    x = gaussian_blob(n, 0.4, 0.5, 0.06)
    dx = 1.0 / n
    dt = 0.1
    vel = torch.zeros(2, n, n, device=DEV)
    vel[0] = 3 * dx / dt               # +3 cells in x
    vel[1] = -2 * dx / dt              # -2 cells in y (down)
    w = M.semi_lagrangian_warp(x, vel, dx, dt)
    expect = torch.roll(x, shifts=(-2, 3), dims=(0, 1))
    assert (w - expect)[8:-8, 8:-8].abs().max().item() < 1e-5


def test_warp_moves_toward_plus_y_is_up():
    n = 64
    x = gaussian_blob(n, 0.5, 0.3, 0.05)
    vel = torch.zeros(2, n, n, device=DEV)
    vel[1] = 1.0
    w = M.semi_lagrangian_warp(x, vel, 1.0 / n, 0.1)
    rows = torch.arange(n, device=DEV, dtype=torch.float32)
    cy0 = (x.sum(1) * rows).sum() / x.sum()
    cy1 = (w.sum(1) * rows).sum() / w.sum()
    assert abs((cy1 - cy0).item() - 0.1 * n) < 0.25     # moved +6.4 rows (index y grows upward)


def test_flicker_zero_for_pure_transport_and_ratio_detects_noise():
    n, T = 64, 8
    dx, dt = 1.0 / n, 0.05
    vel = torch.zeros(T, 2, n, n, device=DEV)
    vel[:, 0] = 1 * dx / dt
    frames = [gaussian_blob(n, 0.3, 0.5, 0.07)]
    for t in range(1, T):
        frames.append(torch.roll(frames[-1], 1, dims=1))
    seq = torch.stack(frames)
    ref = M.flicker(seq, vel, dx, dt)
    assert ref < 1e-4
    # static (non-advected) prediction flickers relative to the flow
    static = seq[:1].expand(T, n, n)
    assert M.flicker(static, vel, dx, dt) > 10 * max(ref, 1e-6)
    noisy = seq + 0.05 * torch.randn_like(seq)
    assert M.flicker(noisy, vel, dx, dt) > M.flicker(seq, vel, dx, dt) + 0.02


def test_flicker_resamples_lr_velocity():
    n, T = 64, 4
    seq = torch.rand(T, n, n, device=DEV)
    vel_lr = torch.zeros(T, 2, 16, 16, device=DEV)
    assert math.isclose(M.flicker(seq, vel_lr, 1.0 / n, 0.03),
                        M.flicker(seq, torch.zeros(T, 2, n, n, device=DEV), 1.0 / n, 0.03), rel_tol=1e-6)


# ----------------------------------------------------------------------------- baseline
def test_bicubic_upscale_shapes_and_constant():
    data = torch.zeros(NUM_CHANNELS, 32, 32, device=DEV)
    data[CH["density"]] = 0.7
    data[CH["temperature"]] = 400.0
    st = FieldState(data=data, solid=torch.zeros(32, 32, device=DEV), meta=FieldMeta.for_res(32))
    out = M.bicubic_upscale(st)
    assert out.density.shape == (128, 128) and out.temperature.shape == (128, 128)
    assert torch.allclose(out.density, torch.full_like(out.density, 0.7), atol=1e-5)
    assert torch.allclose(out.temperature, torch.full_like(out.temperature, 400.0), rtol=1e-6)


def test_flicker_responds_to_known_noise_amplitude():
    """Known-flicker injection: per-frame iid Gaussian noise of std s, warped by an integer-cell shift
    (exact warp): x_t - warp(x_{t-1}) = n_t - n'_{t-1} ~ N(0, 2 s^2), so flicker = 2 s / sqrt(pi)."""
    n, T = 128, 8
    dx, dt = 1.0 / n, 0.05
    vel = torch.zeros(T, 2, n, n, device=DEV)
    vel[:, 1] = 2 * dx / dt
    # a static clean field: with an integer-cell shift every warped sample is exactly one noise value
    # of the previous frame (also at the clamped border), so the expectation is exact everywhere.
    seq = torch.zeros(T, n, n, device=DEV)
    base = M.flicker(seq, vel, dx, dt)
    g = torch.Generator(device=DEV).manual_seed(0)
    for s in (0.002, 0.01, 0.05):
        noisy = seq + s * torch.randn(seq.shape, device=DEV, generator=g)
        f = M.flicker(noisy, vel, dx, dt)
        assert abs(f - 2 * s / math.sqrt(math.pi)) < 0.1 * (2 * s / math.sqrt(math.pi)) + base, (s, f, base)


def _write_scn(root, split, seed, hr, lr, lr_sim, dt):
    d = root / split / f"scn_{seed:06d}"
    d.mkdir(parents=True)
    np.save(d / "hr.npy", hr.astype(np.float16))
    np.save(d / "lr.npy", lr.astype(np.float16))
    np.save(d / "lr_sim.npy", lr_sim.astype(np.float16))
    np.save(d / "solid_lr.npy", np.zeros(lr.shape[-2:], np.uint8))
    np.save(d / "solid_hr.npy", np.zeros(hr.shape[-2:], np.uint8))
    (d / "meta.json").write_text('{"dt": %r, "start_step": 1, "tags": [], "scenario": {"seed": %d}}' % (dt, seed))
    return d


def test_evaluate_upscaler_warps_pred_with_its_driving_velocity(tmp_path, monkeypatch):
    """Audit regression test. HR/lr: a blob moving +x with lr velocity +x. lr_sim (an independent run):
    a blob moving +y with lr_sim velocity +y. Bicubic of lr_sim is perfectly transported by ITS OWN flow,
    so the audited flicker is ~ the reference; warping it with the HR run's velocity (legacy) is large.
    Also: unfinished scenario dirs (no meta.json) are skipped and counted."""
    from vfx.core.fields import UPSCALE
    h, T, dt = 32, 6, 0.05
    H = h * UPSCALE
    dx_lr = 1.0 / h
    c = (torch.arange(H, dtype=torch.float32) + 0.5) / H
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    shift = 1 * dx_lr                       # one LR cell per frame
    hr = np.zeros((T, 2, H, H), np.float32)
    lr = np.zeros((T, NUM_CHANNELS, h, h), np.float32)
    ls = np.zeros((T, NUM_CHANNELS, h, h), np.float32)
    for t in range(T):
        b = torch.exp(-((xx - 0.3 - t * shift) ** 2 + (yy - 0.5) ** 2) / (2 * 0.08 ** 2))
        hr[t, 0] = b.numpy()
        hr[t, 1] = 300.0
        lr[t, CH["density"]] = F.avg_pool2d(b[None, None], UPSCALE)[0, 0].numpy()
        lr[t, CH["temperature"]] = 300.0
        lr[t, CH["vel_x"]] = shift / dt
        bs = torch.exp(-((xx - 0.5) ** 2 + (yy - 0.3 - t * shift) ** 2) / (2 * 0.08 ** 2))
        ls[t, CH["density"]] = F.avg_pool2d(bs[None, None], UPSCALE)[0, 0].numpy()
        ls[t, CH["temperature"]] = 300.0
        ls[t, CH["vel_y"]] = shift / dt
    d = _write_scn(tmp_path, "test", 1, hr, lr, ls, dt)
    (tmp_path / "test" / "scn_000002").mkdir()       # in progress: no meta.json
    monkeypatch.setattr(M, "scenario_dirs", lambda v, s: [d, tmp_path / "test" / "scn_000002"])
    r = M.evaluate_upscaler(M.bicubic_upscale, "fake", "test", "lr_sim", device=DEV)
    assert r.n_scenarios == 1 and r.n_skipped_incomplete == 1 and r.n_frames == T
    assert r.flicker_ref < 0.01 * hr[:, 0].mean() + 2e-3
    assert r.flicker < 0.25 * r.flicker_legacy_hrvel, (r.flicker, r.flicker_legacy_hrvel)
    assert r.flicker_ratio < r.flicker_ratio_legacy_hrvel
    # source="lr": audited and legacy definitions coincide
    q = M.evaluate_upscaler(M.bicubic_upscale, "fake", "test", "lr", device=DEV)
    assert math.isclose(q.flicker, q.flicker_legacy_hrvel, rel_tol=1e-9)
