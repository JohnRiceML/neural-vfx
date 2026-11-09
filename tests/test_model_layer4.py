"""Layer 4: model quality, on checkpoint vfx.core.paths.default_checkpoint() and dataset $VFX_DATASET.

Gates (v0 = M3 baseline):
  * overfit-single-batch sanity on a fresh DetailNet (no checkpoint/dataset needed)
  * beats bicubic ('tricubic' in 2D) on spectrum error and L1 on the test split
  * long rollout: >= 1000 frames of Upscaler on a live 128^2 Solver, no NaN / blow-up / drift
Informational (xfail, strict=False, numbers printed): the v1 gates (spectrum >= 50% below
bicubic, flicker within 1.2x of reference, OOD within 20% of in-distribution).
The baseline is vfx.eval.metrics.bicubic_upscale (independent of vfx.runtime.classical_upscale).
A JSON report is written to DATA_ROOT/runs/eval/layer4_latest.json.
"""
from __future__ import annotations

import json
import math
import time

import pytest
import torch
import torch.nn.functional as F

from vfx.core.fields import AMBIENT_TEMP, CH, LR_RES, HR_RES, NUM_CHANNELS, SIM_DT, Emitter, Forces, Impulse, Obstacle
from vfx.core.paths import RUNS, dataset_dir, default_checkpoint
from vfx.eval import metrics as M
from tests.helpers import dataset_version, import_or_skip, requires_cuda

pytestmark = requires_cuda
DEV = "cuda"
VERSION = dataset_version()
CKPT = default_checkpoint()


def _need_ckpt():
    if not CKPT.exists():
        pytest.skip(f"no checkpoint at {CKPT}")


def _need_dataset():
    if not (dataset_dir(VERSION) / "splits.json").exists():
        pytest.skip(f"dataset '{VERSION}' not built")


# ----------------------------------------------------------------------------- sanity: overfit
def _synthetic_batch(B=4, h=32, seed=0):
    """A batch whose HR target is a fixed, learnable function of the LR input (sharpened bicubic +
    density-modulated ripples), in physical units."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    H = 4 * h
    z = torch.randn(B, 1, 8, 8, device=DEV, generator=g)
    d_hr = F.interpolate(z, size=(H, H), mode="bicubic", align_corners=False).relu() * 0.8
    t_hr = AMBIENT_TEMP + 700.0 * d_hr
    vel = 0.5 * F.interpolate(torch.randn(B, 2, 8, 8, device=DEV, generator=g), size=(h, h), mode="bicubic",
                              align_corners=False)
    lr = torch.zeros(B, NUM_CHANNELS, h, h, device=DEV)
    lr[:, CH["density"]] = F.avg_pool2d(d_hr, 4)[:, 0]
    lr[:, CH["temperature"]] = F.avg_pool2d(t_hr, 4)[:, 0]
    lr[:, CH["vel_x"]:CH["vel_y"] + 1] = vel
    up = F.interpolate(lr[:, [CH["density"], CH["temperature"]]], scale_factor=4, mode="bicubic",
                       align_corners=False)
    yy, xx = torch.meshgrid(torch.arange(H, device=DEV), torch.arange(H, device=DEV), indexing="ij")
    ripple = torch.sin(xx * 0.9) * torch.cos(yy * 0.7)
    tgt = up.clone()
    tgt[:, 0] = (up[:, 0] * (1 + 0.5 * ripple)).clamp_min(0)
    tgt[:, 1] = AMBIENT_TEMP + (up[:, 1] - AMBIENT_TEMP) * (1 + 0.3 * ripple)
    prev = up.detach().clone()                       # uninformative "previous HR"
    return lr, prev, tgt, up


def _loss(pred, tgt):
    return (pred[:, 0] - tgt[:, 0]).abs().mean() + (pred[:, 1] - tgt[:, 1]).abs().mean() / 500.0


def test_overfit_single_batch():
    model_mod = import_or_skip("vfx.model", ("DetailNet",))
    torch.manual_seed(0)
    net = model_mod.DetailNet().to(DEV).train()
    lr, prev, tgt, up = _synthetic_batch()
    out = net(lr, prev)
    assert out.shape == tgt.shape, f"DetailNet output {tuple(out.shape)} != {tuple(tgt.shape)}"
    base = _loss(up, tgt).item()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    l0 = _loss(net(lr, prev), tgt).item()
    t0 = time.time()
    for i in range(600):
        opt.zero_grad(set_to_none=True)
        loss = _loss(net(lr, prev), tgt)
        loss.backward()
        opt.step()
    lf = loss.item()
    print(f"overfit: bicubic loss {base:.4f}, initial {l0:.4f}, final {lf:.4f} ({time.time() - t0:.1f}s)")
    assert math.isfinite(lf)
    assert lf < 0.2 * l0, "loss did not drop 5x on a single batch"
    assert lf < 0.5 * base, "model did not learn detail beyond bicubic on a single batch"


# ----------------------------------------------------------------------------- dataset evaluation
@pytest.fixture(scope="module")
def report():
    _need_ckpt()
    _need_dataset()
    rt = import_or_skip("vfx.runtime", ("Upscaler",))
    up = rt.Upscaler(str(CKPT), device=DEV)
    root = dataset_dir(VERSION)
    splits = json.loads((root / "splits.json").read_text())
    rep = {"checkpoint": str(CKPT), "dataset": VERSION, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "flicker_definition": "audited-2026-11-09: pred warped with its driving (source) velocity, "
                                 "ref = HR warped with pooled-HR velocity; legacy key *_legacy_hrvel",
           "results": {}}
    for split in ("test", "ood"):
        if not splits.get(split):
            continue
        for source in ("lr", "lr_sim"):
            key = f"{split}/{source}"
            try:
                n = M.evaluate_upscaler(up, VERSION, split, source)
                b = M.evaluate_upscaler(M.bicubic_upscale, VERSION, split, source)
            except FileNotFoundError as e:
                rep["results"][key] = {"error": str(e)}
                continue
            if n.n_scenarios == 0:     # dataset still being written (e.g. v2 under construction)
                rep["results"][key] = {"skip": f"no finished scenarios ({n.n_skipped_incomplete} incomplete)"}
                continue
            rep["results"][key] = {"neural": n.as_dict(), "bicubic": b.as_dict()}
            fpf = n.n_frames / n.n_scenarios
            print(f"[{key}] {n.n_scenarios} scn x {fpf:.0f} frames ({n.n_skipped_incomplete} incomplete skipped) | "
                  f"spectrum neural {n.spectrum_error:.4f} vs bicubic {b.spectrum_error:.4f} | "
                  f"L1 dens {n.l1_density:.5f} vs {b.l1_density:.5f} | L1 T {n.l1_temperature:.3f} vs "
                  f"{b.l1_temperature:.3f} K | flicker ratio neural {n.flicker_ratio:.3f}, bicubic {b.flicker_ratio:.3f} "
                  f"(legacy HR-velocity: {n.flicker_ratio_legacy_hrvel:.3f} / {b.flicker_ratio_legacy_hrvel:.3f}; "
                  f"detail-normalized: {n.flicker_ratio_detail_norm:.3f} / {b.flicker_ratio_detail_norm:.3f})")
    out = RUNS / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "layer4_latest.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep["results"]


def _get(report, key):
    r = report.get(key)
    if r is None:
        pytest.skip(f"{key} not evaluated (split empty)")
    if "skip" in r:
        pytest.skip(f"{key}: {r['skip']}")
    if "error" in r:
        pytest.fail(f"{key}: {r['error']}")
    return r["neural"], r["bicubic"]


@pytest.mark.parametrize("source", ["lr", "lr_sim"])
def test_beats_bicubic_spectrum(report, source):
    n, b = _get(report, f"test/{source}")
    print(f"test/{source}: spectrum error neural {n['spectrum_error']:.4f} vs bicubic {b['spectrum_error']:.4f}")
    assert n["spectrum_error"] < b["spectrum_error"]


def test_beats_bicubic_l1(report):
    """L1 on the pooled-HR input (pairs are aligned; the independent LR sim diverges from HR, so
    its L1 is reported in the JSON but not gated)."""
    n, b = _get(report, "test/lr")
    print(f"test/lr: L1 density neural {n['l1_density']:.5f} vs bicubic {b['l1_density']:.5f}; "
          f"temperature {n['l1_temperature']:.3f} vs {b['l1_temperature']:.3f} K")
    assert n["l1_density"] < b["l1_density"]


@pytest.mark.xfail(strict=False, reason="v1 gate (M4); v0 is the M3 baseline - informational")
@pytest.mark.parametrize("source", ["lr", "lr_sim"])
def test_v1_spectrum_50pct_below_bicubic(report, source):
    n, b = _get(report, f"test/{source}")
    red = 1 - n["spectrum_error"] / max(b["spectrum_error"], 1e-12)
    print(f"test/{source}: spectrum error reduction vs bicubic {red * 100:.1f}% (v1 needs >= 50%)")
    assert n["spectrum_error"] <= 0.5 * b["spectrum_error"]


@pytest.mark.xfail(strict=False, reason="v1 gate (M4); v0 is the M3 baseline - informational")
@pytest.mark.parametrize("source", ["lr", "lr_sim"])
def test_v1_flicker_within_1p2_of_reference(report, source):
    """Audited definition (metrics.evaluate_upscaler): pred warped with the velocity of the LR input it was
    fed, reference = HR warped with its own pooled velocity. The pre-audit number (pred warped with the HR
    run's velocity even when fed the independent LR sim) is printed as 'legacy' for history only."""
    n, b = _get(report, f"test/{source}")
    print(f"test/{source}: flicker ratio pred/ref neural {n['flicker_ratio']:.3f} (bicubic {b['flicker_ratio']:.3f}); "
          f"legacy HR-velocity {n['flicker_ratio_legacy_hrvel']:.3f} (bicubic {b['flicker_ratio_legacy_hrvel']:.3f}); "
          "v1 needs <= 1.2")
    assert n["flicker_ratio"] <= 1.2


def test_ood_reported_separately(report):
    if "ood/lr" not in report:
        pytest.skip("no OOD split in dataset")
    n, b = _get(report, "ood/lr")
    ni, _ = _get(report, "test/lr")
    print(f"OOD/lr: spectrum neural {n['spectrum_error']:.4f} (in-dist {ni['spectrum_error']:.4f}), "
          f"bicubic {b['spectrum_error']:.4f}; L1 {n['l1_density']:.5f} (in-dist {ni['l1_density']:.5f})")
    for row in n["per_scenario"]:
        print(f"  {row['dir']} tags={row['tags']} spec {row['spectrum_error']:.4f} l1 {row['l1_density']:.5f}")
    assert math.isfinite(n["spectrum_error"]) and math.isfinite(n["l1_density"])


@pytest.mark.xfail(strict=False, reason="v1 gate (generalization); informational for v0")
def test_v1_ood_within_20pct(report):
    if "ood/lr_sim" not in report:
        pytest.skip("no OOD split in dataset")
    n, _ = _get(report, "ood/lr_sim")
    ni, _ = _get(report, "test/lr_sim")
    ratio = n["spectrum_error"] / max(ni["spectrum_error"], 1e-12)
    print(f"OOD/in-dist spectrum ratio {ratio:.3f} (v1 needs <= 1.2)")
    assert ratio <= 1.2


# ----------------------------------------------------------------------------- long rollout
def test_long_rollout_live_solver():
    _need_ckpt()
    sim = import_or_skip("vfx.sim", ("Solver",))
    rt = import_or_skip("vfx.runtime", ("Upscaler",))
    solver = sim.Solver(LR_RES)
    st = solver.reset([Obstacle("circle", (0.6, 0.6), (0.08, 0.08))])
    up = rt.Upscaler(str(CKPT), device=DEV)
    up.reset()
    g = torch.Generator().manual_seed(1)
    frames = 1200
    stats = []
    for i in range(frames):
        imp = []
        if i % 30 == 0:
            p = torch.rand(2, generator=g).tolist()
            v = ((torch.rand(2, generator=g) * 2 - 1) * 4).tolist()
            imp = [Impulse(pos=(p[0], p[1]), radius=0.08, velocity=(v[0], v[1]))]
        f = Forces(emitters=[Emitter(pos=(0.45, 0.12), radius=0.05, density_rate=1.0, temperature=1300.0,
                                     fuel_rate=2.5, velocity=(0.0, 0.5))], impulses=imp)
        st = solver.step(st, f, SIM_DT)
        out = up.infer(st)
        if i % 100 == 99:
            d, T = out.density.float(), out.temperature.float()
            assert d.shape == (HR_RES, HR_RES) and T.shape == (HR_RES, HR_RES)
            assert torch.isfinite(d).all() and torch.isfinite(T).all(), f"non-finite output at frame {i + 1}"
            dmax, tmin, tmax = d.max().item(), T.min().item(), T.max().item()
            lr_mean = st.ch("density").mean().item()
            hr_mean = d.mean().item()
            stats.append((i + 1, dmax, tmin, tmax, lr_mean, hr_mean))
            assert dmax < 50.0, f"density blow-up {dmax:.2f} at frame {i + 1}"
            assert 0.0 < tmin and tmax < 1e4, f"temperature out of range [{tmin:.0f}, {tmax:.0f}] at frame {i + 1}"
            if lr_mean > 1e-3:
                drift = abs(hr_mean - lr_mean) / lr_mean
                assert drift < 0.25, f"HR mean density drifted {drift * 100:.1f}% from LR at frame {i + 1}"
    for s in stats:
        print("frame %5d  dmax %.3f  T [%.0f, %.0f]  mean density LR %.4f HR %.4f" % s)


# ----------------------------------------------------------------------------- live-app flicker (no dataset)
@pytest.fixture(scope="module")
def live():
    """vfx.eval.live_flicker: 128^2 Solver on fixed scenes (fire; smoke + circle), 300 steps, fed to the
    Upscaler and classical_upscale; flicker with the 128^2 run's own velocity; reference = a 512^2 Solver
    run of the same scene (a different flow realization: indicative, not pixel-matched)."""
    _need_ckpt()
    import_or_skip("vfx.sim", ("Solver",))
    import_or_skip("vfx.runtime", ("Upscaler", "classical_upscale"))
    from vfx.eval.live_flicker import live_flicker
    res = live_flicker(steps=300, warmup=30, checkpoint=str(CKPT))
    out = RUNS / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "live_flicker_latest.json").write_text(json.dumps(
        {"checkpoint": str(CKPT), "steps": 300, "warmup": 30, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
         "results": res}, indent=1))
    for k, r in res.items():
        print(f"live/{k}: flicker neural {r['neural']:.6f} bicubic {r['bicubic']:.6f} ref(512^2) {r['ref']:.6f} | "
              f"ratio neural {r['ratio_neural']:.3f} bicubic {r['ratio_bicubic']:.3f} neural/bicubic "
              f"{r['neural_over_bicubic']:.3f}")
    return res


@pytest.mark.parametrize("scene", ["fire", "smoke_obstacle"])
def test_live_flicker_measured(live, scene):
    r = live[scene]
    assert all(math.isfinite(r[k]) and r[k] > 0 for k in ("neural", "bicubic", "ref")), r


@pytest.mark.xfail(strict=False, reason="v1 gate (M4) on the live app situation; reference is a 512^2 run of the "
                                        "same scene (different realization) - informational")
@pytest.mark.parametrize("scene", ["fire", "smoke_obstacle"])
def test_v1_live_flicker_within_1p2_of_reference(live, scene):
    r = live[scene]
    print(f"live/{scene}: flicker ratio neural {r['ratio_neural']:.3f} (bicubic {r['ratio_bicubic']:.3f}); v1 needs <= 1.2")
    assert r["ratio_neural"] <= 1.2
