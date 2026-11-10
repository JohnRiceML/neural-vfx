"""Unit tests of vfx/eval3d/metrics.py on synthetic images (no dataset, no other 3D module).

The metrics are the measuring instrument for route B, so each is checked against a known answer:
identity -> 0, monotone in the corruption it is meant to detect, blind to what it must ignore (background
outside the effect region), and calibrated where a closed form exists (white-noise power, injected flicker).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from vfx.eval3d import metrics as M

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
DEV = "cuda"
H, W = 144, 208


# ----------------------------------------------------------------------------- synthetic content
def _waves(seed: int, n: int = 14, lam=(5.0, 40.0)):
    g = np.random.default_rng(seed)
    k = []
    for _ in range(n):
        L = g.uniform(*lam)
        th = g.uniform(0, 2 * math.pi)
        k.append((2 * math.pi / L * math.cos(th), 2 * math.pi / L * math.sin(th), g.uniform(0, 2 * math.pi),
                  g.uniform(0.3, 1.0), g.integers(0, 3)))
    return k


def texture(t: float = 0.0, v=(0.0, 0.0), seed: int = 0, h: int = H, w: int = W, envelope: bool = True,
            lam=(5.0, 40.0)):
    """Analytic RGB texture translated by v * t pixels (exact for fractional shifts). Returns (3,h,w) float
    in [0,1] and alpha (h,w) (a soft blob that moves with the content)."""
    ys = torch.arange(h, device=DEV, dtype=torch.float64)[:, None]
    xs = torch.arange(w, device=DEV, dtype=torch.float64)[None, :]
    X, Y = xs - v[0] * t, ys - v[1] * t
    img = torch.zeros(3, h, w, device=DEV, dtype=torch.float64)
    for kx, ky, ph, a, c in _waves(seed, lam=lam):
        s = a * torch.sin(kx * X + ky * Y + ph)
        img[int(c)] += s
        img[(int(c) + 1) % 3] += 0.4 * s
    img = 0.5 + 0.08 * img
    cx, cy = w * 0.5, h * 0.5
    env = torch.exp(-(((X - cx) / (0.28 * w)) ** 2 + ((Y - cy) / (0.3 * h)) ** 2))
    if envelope:
        img = 0.1 + (img - 0.1) * env[None]
    return img.clamp(0, 1).float(), env.float()


def seq(L: int = 8, v=(2.0, -1.0), seed: int = 0, lam=(5.0, 40.0)):
    frames, alphas = zip(*[texture(t, v, seed, lam=lam) for t in range(L)])
    motion = torch.zeros(L, 2, H, W, device=DEV)
    motion[:, 0], motion[:, 1] = v
    return torch.stack(frames), torch.stack(alphas), motion


def u8(x):
    return (x.clamp(0, 1) * 255 + 0.5).to(torch.uint8)


# ----------------------------------------------------------------------------- basics / masks
def test_to_float_and_luminance():
    x = torch.full((3, 4, 5), 255, dtype=torch.uint8, device=DEV)
    assert torch.allclose(M.to_float(x), torch.ones(3, 4, 5, device=DEV))
    assert torch.allclose(M.luminance(x), torch.ones(4, 5, device=DEV))      # weights sum to 1


def test_effect_mask_threshold_and_dilation():
    a = torch.zeros(1, 40, 40, device=DEV)
    a[0, 20, 20] = 0.5
    a[0, 5, 5] = 0.01                       # below threshold
    m = M.effect_mask(a, dilate_px=3)
    assert m.dtype == torch.bool and m.shape == a.shape
    assert int(m.sum()) == 49 and bool(m[0, 17, 23]) and not bool(m[0, 16, 20]) and not bool(m[0, 5, 5])
    assert M.default_dilate_px(1440, 2560) == 20 and M.default_dilate_px(64, 64) == 2


def test_l1_identity_offset_and_mask():
    x, a = texture()
    assert M.l1(x, x) == 0.0
    assert M.l1(x, (x + 0.1)) == pytest.approx(0.1, abs=1e-6)
    m = torch.zeros(H, W, dtype=torch.bool, device=DEV)
    m[:, : W // 2] = True
    y = x.clone()
    y[:, :, W // 2:] += 0.5                 # change only outside the mask
    assert M.l1(y, x, m) == 0.0
    assert M.l1(u8(x), x) < 0.5 / 255 + 1e-6  # uint8 accepted (quantization only)


# ----------------------------------------------------------------------------- spectrum
def test_spectrum_identity_zero():
    x, a = texture()
    assert M.spectrum_error(x, x, M.effect_mask(a)) == pytest.approx(0.0, abs=1e-9)


def test_spectrum_monotone_in_blur_and_signed():
    x, a = texture(envelope=False)
    m = M.effect_mask(a)
    errs = [M.spectrum_error(M.gaussian_blur(x, s), x, m) for s in (0.7, 1.5, 3.0)]
    assert errs[0] < errs[1] < errs[2], errs
    assert M.spectrum_log_ratio(M.gaussian_blur(x, 1.5), x, m) < 0


def test_spectrum_penalizes_noise_and_signed_positive():
    x, a = texture(envelope=False)
    m = M.effect_mask(a)
    g = torch.Generator(device=DEV).manual_seed(1)
    errs = [M.spectrum_error(x + s * torch.randn(x.shape, device=DEV, generator=g), x, m) for s in (0.005, 0.02, 0.05)]
    assert errs[0] < errs[1] < errs[2], errs
    assert M.spectrum_log_ratio(x + 0.02 * torch.randn(x.shape, device=DEV, generator=g), x, m) > 0


def test_spectrum_ignores_content_outside_effect_bbox():
    x, a = texture()
    m = torch.zeros(H, W, dtype=torch.bool, device=DEV)
    m[30:110, 40:150] = True
    y = x.clone()
    y[:, :20, :] = torch.rand(3, 20, W, device=DEV)     # garbage outside the bbox
    y[:, :, 170:] = 0.0
    assert M.spectrum_error(y, x, m) == pytest.approx(0.0, abs=1e-9)


def test_spectrum_shift_invariant():
    x0, _ = texture(0.0, (3.0, 2.0), envelope=False)
    x1, _ = texture(1.0, (3.0, 2.0), envelope=False)    # same texture translated by (3, 2) px
    assert M.spectrum_error(x1, x0) < 0.05


def test_spectrum_white_noise_calibration():
    """White noise of variance s^2 has power s^2 in every bin (window-normalized)."""
    g = torch.Generator(device=DEV).manual_seed(3)
    s = 0.05
    n = s * torch.randn(16, 3, 128, 128, device=DEV, generator=g)
    lum_n = n[:, :1].expand(-1, 3, -1, -1)              # identical channels -> luminance = that noise
    p, cnt = M.radial_power_spectrum(lum_n)
    ok = np.isfinite(p) & (cnt > 50)
    assert ok.sum() > 20
    assert np.median(p[ok]) == pytest.approx(s * s, rel=0.1)


def test_spectrum_quantization_floor():
    """FLOOR = uint8 quantization noise power: quantizing a frame costs at most log10(1 + 0.56) = 0.19 decade in a
    bin where the reference has NO power (0.56 = sum of squared luma weights); 0 where the reference has content."""
    x, a = texture()
    assert M.spectrum_error(u8(x), x, M.effect_mask(a)) < math.log10(1 + 0.56) + 0.01
    g = torch.Generator(device=DEV).manual_seed(9)
    xn = (x + 0.01 * torch.randn(x.shape, device=DEV, generator=g)).clamp(0, 1)    # broadband content
    assert M.spectrum_error(u8(xn), xn, M.effect_mask(a)) < 0.03


def test_band_power_detail_higher():
    x, a = texture(envelope=False)
    b = M.gaussian_blur(x, 2.0)
    assert M.band_power(x, None, 0.15, 0.5) > 3 * M.band_power(b, None, 0.15, 0.5)
    assert M.band_power(x, None, 0.0, 0.03) == pytest.approx(M.band_power(b, None, 0.0, 0.03), rel=0.3)


# ----------------------------------------------------------------------------- structural
def test_ms_dssim_identity_monotone_masked():
    x, a = texture()
    m = M.effect_mask(a)
    assert M.ms_dssim(x, x, m) == pytest.approx(0.0, abs=1e-6)
    b = [M.ms_dssim(M.gaussian_blur(x, s), x, m) for s in (0.7, 1.5, 3.0)]
    assert 0 < b[0] < b[1] < b[2] <= 2.0, b
    g = torch.Generator(device=DEV).manual_seed(4)
    n = [M.ms_dssim(x + s * torch.randn(x.shape, device=DEV, generator=g), x, m) for s in (0.01, 0.05)]
    assert 0 < n[0] < n[1], n
    mm = torch.zeros(H, W, dtype=torch.bool, device=DEV)
    mm[40:100, 50:150] = True
    y = x.clone()
    y[:, :20] = torch.rand(3, 20, W, device=DEV)
    assert M.ms_dssim(y, x, mm) < 0.01                  # far outside the mask: (almost) no effect


# ----------------------------------------------------------------------------- warp / flicker
def test_warp_sign_convention_integer_shift():
    """motion (+3, -2): content moved 3 px right and 2 px up -> W(prev)(p) = prev(p - m) equals cur exactly."""
    prev, _ = texture(0.0, (3.0, -2.0))
    cur, _ = texture(1.0, (3.0, -2.0))
    m = torch.zeros(1, 2, H, W, device=DEV)
    m[:, 0], m[:, 1] = 3.0, -2.0
    w, valid = M.warp_backward(prev[None], m)
    assert valid.shape == (1, H, W) and not bool(valid[0, :, :3].any()) and bool(valid[0, 5:-5, 5:-5].all())
    assert (w[0] - cur)[:, valid[0]].abs().max().item() < 1e-4
    w2, _ = M.warp_backward(prev[None], -m)               # wrong sign is clearly wrong
    assert (w2[0] - cur)[:, 5:-5, 5:-5].abs().mean().item() > 0.01


def test_flicker_known_translation():
    """Pure transport: flicker ~ 0 with the true motion (integer), small for fractional, and far below the
    unwarped frame difference."""
    for v, tol in (((2.0, -1.0), 1e-4), ((1.5, 0.75), 4e-3)):
        x, a, mo = seq(8, v)
        mk = M.effect_mask(a)
        f = M.flicker(x, mo, mk)
        nowarp = M.warp_error(x, mo, mk, use_motion=False)
        assert f < tol, (v, f)
        assert f < 0.1 * nowarp, (v, f, nowarp)


def test_flicker_known_injection_amplitude():
    """iid per-frame noise of std s on an exactly transported sequence adds E|n_t - n_{t-1}| = 2 s / sqrt(pi)."""
    x, a, mo = seq(10, (2.0, -1.0))
    g = torch.Generator(device=DEV).manual_seed(5)
    for s in (0.004, 0.02):
        xn = x + s * torch.randn(x.shape, device=DEV, generator=g)
        f = M.flicker(xn, mo, None)
        assert f == pytest.approx(2 * s / math.sqrt(math.pi), rel=0.08), (s, f)


def test_flicker_ratio_vs_reference():
    """The gate quantity: noisy prediction / clean reference > 1.2; clean prediction == reference."""
    x, a, mo = seq(8, (1.5, 0.75))
    mk = M.effect_mask(a)
    ref = M.flicker(x, mo, mk)
    g = torch.Generator(device=DEV).manual_seed(6)
    noisy = x + 0.01 * torch.randn(x.shape, device=DEV, generator=g) * a[:, None]
    assert M.flicker(x, mo, mk) / ref == pytest.approx(1.0)
    assert M.flicker(noisy, mo, mk) / ref > 1.2


def test_flicker_with_another_sequences_motion_is_wrong():
    """The 2D lesson: warping with the wrong sequence's motion measures motion mismatch, not flicker."""
    x, a, mo = seq(8, (2.0, -1.0))
    _, _, other = seq(8, (-1.0, 2.0), seed=1)
    mk = M.effect_mask(a)
    assert M.flicker(x, other, mk) > 20 * max(M.flicker(x, mo, mk), 1e-5)


def test_flicker_uint8_and_motion_offset_diagnostic():
    x, a, mo = seq(6, (2.0, 0.0))
    f8 = M.flicker(u8(x), mo, M.effect_mask(a))
    assert f8 < 1.0 / 255
    mo2 = mo.clone()
    mo2[0] = 0.0
    assert M.warp_error(x, mo2, None, motion_offset=1) == pytest.approx(M.warp_error(x, mo, None, motion_offset=1), abs=1e-6)


def test_estimate_flow_recovers_translation():
    """Own-flow estimator: integer and fractional translations recovered inside the content (contract sign)."""
    for v in ((2.0, -1.0), (0.6, 1.4), (-4.0, 0.0)):
        x, a, _ = seq(3, v)
        f = M.estimate_flow(x[:-1], x[1:])
        core = a[1:] > 0.5
        ex, ey = (f[:, 0][core] - v[0]).abs().median().item(), (f[:, 1][core] - v[1]).abs().median().item()
        assert ex < 0.1 and ey < 0.1, (v, ex, ey)


def test_own_flow_flicker_translation_and_injection():
    """Default flicker (own estimated flow): ~0 for pure translation, noise injection ~ 2s/sqrt(pi) on top, and a
    flow from another sequence is wrong (the 2D lesson)."""
    x, a, mo = seq(8, (1.5, -0.75))
    mk = M.effect_mask(a)
    f0 = M.flicker(x, None, mk)
    assert f0 < 0.15 * M.warp_error(x, mo, mk, use_motion=False), f0
    g = torch.Generator(device=DEV).manual_seed(13)
    s = 0.01
    xn = x + s * torch.randn(x.shape, device=DEV, generator=g)
    fn = M.flicker(xn, None, mk)
    assert fn == pytest.approx(2 * s / math.sqrt(math.pi), rel=0.25), fn
    assert fn / f0 > 1.2
    y, _, _ = seq(8, (-1.0, 1.0), seed=3)
    assert M.flicker(x, M.sequence_flow(y), mk) > 5 * f0


def test_own_flow_swim_screen_fixed_detail():
    v = (1.5, -0.75)
    base, a, _ = seq(8, v)
    det, _, _ = seq(8, v, seed=7)
    glued = base + 0.1 * (det - 0.5) * a[:, None]
    fixed = base + 0.1 * (det[:1] - 0.5) * a[:1, None]
    mk = M.effect_mask(a)
    ref = M.detail_swim(glued, base, None, mk)
    assert M.detail_swim(fixed, base, None, mk) > 2 * ref, (M.detail_swim(fixed, base, None, mk), ref)


def test_temporal_jerk():
    """Linear-in-time change cancels exactly, slow translation (k v << 1) mostly; iid noise gives 2 s sqrt(3/pi)."""
    x, a, mo = seq(10, (0.25, 0.0))
    fade = torch.stack([x[0] * (1 - 0.05 * t) for t in range(10)])
    assert M.temporal_jerk(fade) < 1e-6
    assert M.temporal_jerk(x, a) < 0.4 * M.warp_error(x, mo, a, use_motion=False)
    g = torch.Generator(device=DEV).manual_seed(12)
    s = 0.01
    still = x[:1].expand(10, -1, -1, -1)
    n = still + s * torch.randn(still.shape, device=DEV, generator=g)
    assert M.temporal_jerk(n) == pytest.approx(2 * s * math.sqrt(3 / math.pi), rel=0.05)


def test_alpha_weighted_flicker_ignores_uncompensated_background():
    """Renderer motion is 0 on background pixels even when the camera moves; opacity weighting keeps that
    un-compensated background change out of the temporal metrics (item_masks / score_sequence default)."""
    x, a, mo = seq(6, (2.0, 0.0))
    alpha = (a > 0.3).float() * a                                # medium only in the core
    g = torch.Generator(device=DEV).manual_seed(11)
    bg = torch.rand(6, 3, H, W, device=DEV, generator=g)          # background changes every frame
    xs = torch.where(alpha[:, None] > 0, x, bg)
    w_alpha = M.flicker(xs, mo, alpha)
    w_dil = M.flicker(xs, mo, M.effect_mask(alpha))
    assert w_alpha < 1e-3 and w_dil > 20 * w_alpha, (w_alpha, w_dil)
    mask, tw = M.item_masks({"hr_alpha": (a * 255).to(torch.uint8)}, torch.cat([alpha[:, None], alpha[:, None], mo], 1))
    assert torch.equal(tw, alpha) and mask.sum() > M.effect_mask(alpha).sum()     # spatial: union with hr alpha


def test_detail_swim():
    """Detail glued to the content swims like the reference; screen-fixed or re-rolled detail swims a lot."""
    v = (2.0, -1.0)
    base, a, mo = seq(8, v)
    det, _, _ = seq(8, v, seed=7)                    # a second texture moving WITH the content
    glued = base + 0.2 * (det - 0.5) * a[:, None]
    hr = glued.clone()
    mk = M.effect_mask(a)
    ref = M.detail_swim(hr, base, mo, mk)
    assert M.detail_swim(glued, base, mo, mk) / max(ref, 1e-9) == pytest.approx(1.0)
    assert M.detail_swim(base, base, mo, mk) == 0.0
    fixed_det = 0.2 * (det[0:1] - 0.5) * a[0:1, None]            # stuck to the screen
    screen_fixed = base + fixed_det
    g = torch.Generator(device=DEV).manual_seed(8)
    reroll = base + 0.02 * torch.randn(base.shape, device=DEV, generator=g) * a[:, None]
    hr_real = base + 0.02 * (det - 0.5) * a[:, None]
    ref_real = M.detail_swim(hr_real, base, mo, mk)
    assert M.detail_swim(screen_fixed, base, mo, mk) > 2 * M.detail_swim(base + 0.2 * (det - 0.5) * a[:, None], base, mo, mk) + 1e-4
    assert M.detail_swim(reroll, base, mo, mk) > 3 * ref_real


# ----------------------------------------------------------------------------- baselines
def test_unsharp():
    x, a = texture()
    assert torch.equal(M.unsharp(u8(x), 1.0, 0.0), u8(x))
    s = M.unsharp(x, 1.0, 1.5)
    assert M.band_power(s, None, 0.08, 0.2) > 1.5 * M.band_power(x, None, 0.08, 0.2)
    hwc = u8(x).permute(1, 2, 0).contiguous()
    o = M.unsharp(hwc, 1.0, 1.0)
    assert o.shape == hwc.shape and o.dtype == torch.uint8
    assert torch.equal(o, M.unsharp(u8(x), 1.0, 1.0).permute(1, 2, 0))
    assert M.identity(hwc) is hwc


# ----------------------------------------------------------------------------- runtime-style runner + evaluate
class _Recorder:
    """ScreenDetail-like stub: records calls, returns a REUSED buffer (like the runtime)."""

    def __init__(self, add: int = 0):
        self.resets, self.calls, self.add, self.buf = 0, [], add, None

    def reset(self):
        self.resets += 1

    def infer(self, frame, aux=None):
        assert frame.dtype == torch.uint8 and frame.dim() == 3 and frame.shape[-1] == 3 and frame.is_cuda
        assert set(aux) >= {"alpha", "depth", "motion"}
        h, w = frame.shape[:2]
        assert aux["alpha"].shape == (h, w) and aux["depth"].shape == (h, w) and aux["motion"].shape == (2, h, w)
        self.calls.append(1)
        if self.buf is None or self.buf.shape != frame.shape:
            self.buf = torch.empty_like(frame)
        self.buf.copy_((frame.int() + self.add).clamp(0, 255).to(torch.uint8))
        return self.buf


def _fake_dataset(n=3, L=6):
    items = []
    for i in range(n):
        v = (1.0 + i, -1.0)
        hr, a, mo = seq(L, v, seed=10 + i, lam=(2.2, 30.0))      # content up to near Nyquist, like a real render
        lr = M.gaussian_blur(hr, 1.5)
        aux = torch.cat([a[:, None], torch.ones_like(a)[:, None], mo], 1).half()
        items.append({"lr": u8(lr), "hr": u8(hr), "aux": aux, "seed": 100 + i, "cam": 0, "t0": 0})
    return items


def test_run_sequence_autoregressive_contract():
    ds = _fake_dataset(1, 5)
    rec = _Recorder(add=3)
    out = M.run_sequence(rec, ds[0]["lr"], ds[0]["aux"])
    assert rec.resets == 1 and len(rec.calls) == 5
    assert out.shape == ds[0]["lr"].shape and out.dtype == torch.uint8
    # every frame kept its own value although the stub reuses one buffer
    exp = (ds[0]["lr"].int() + 3).clamp(0, 255).to(torch.uint8)
    assert torch.equal(out, exp)


def test_evaluate_screen_detail_oracle_and_identity(monkeypatch):
    ds = _fake_dataset()
    monkeypatch.setattr(M, "_load", lambda version, split, length, stride=None: ds)

    class Oracle:                                 # returns hr: perfect model
        def __init__(self):
            self.seq, self.t = -1, 0

        def reset(self):
            self.seq += 1
            self.t = 0

        def infer(self, frame, aux):
            o = ds[self.seq]["hr"][self.t].permute(1, 2, 0).contiguous()
            self.t += 1
            return o

    r = M.evaluate_screen_detail(Oracle(), "fake", "test", sharpen={"sigma": 1.0, "amount": 1.0})
    assert r["n_sequences"] == 3 and r["n_frames"] == 18
    mo, lr, sh = r["methods"]["model"], r["methods"]["lr"], r["methods"]["sharpen"]
    assert mo["l1"] == 0 and mo["spectrum_error"] == pytest.approx(0, abs=1e-9) and mo["ms_dssim"] == pytest.approx(0, abs=1e-6)
    assert mo["flicker_ratio"] == pytest.approx(1.0) and mo["swim_ratio"] == pytest.approx(1.0)
    for k in ("l1", "spectrum_error", "ms_dssim"):
        assert lr[k] > 0 and sh[k] > 0
    assert lr["swim"] == 0.0 and lr["spectrum_log_ratio"] < 0
    # identity model scores exactly like the lr baseline
    r2 = M.evaluate_screen_detail(M.identity, "fake", "test", sharpen={"sigma": 1.0, "amount": 1.0})
    for k in ("l1", "spectrum_error", "ms_dssim", "flicker"):
        assert r2["methods"]["model"][k] == pytest.approx(r2["methods"]["lr"][k])


def test_tune_unsharp_prefers_sharpening_on_blurred_data(monkeypatch, tmp_path):
    ds = _fake_dataset()
    monkeypatch.setattr(M, "_load", lambda version, split, length, stride=None: ds)
    monkeypatch.setattr(M, "_cache_dir", lambda: tmp_path)
    t = M.tune_unsharp("fake", use_cache=False, sigmas=(0.75, 1.5), amounts=(0.25, 1.0, 2.0))
    assert t["amount"] > 0 and t["objective"] < 1.0
    assert (tmp_path / "unsharp_fake_train.json").exists()


def test_importable_without_dataset_or_3d_modules():
    """Training imports these before the dataset exists: importing must not pull in datagen3d/runtime3d/render3d."""
    import subprocess
    import sys

    code = ("import sys, vfx.eval3d.metrics as m; bad=[k for k in sys.modules if k.startswith(('vfx.datagen3d',"
            "'vfx.runtime3d','vfx.render3d','vfx.sim3d','vfx.model3d'))]; assert not bad, bad; print('ok')")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and "ok" in p.stdout, p.stderr[-2000:]
