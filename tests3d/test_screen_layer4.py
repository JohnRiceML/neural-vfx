"""Layer 4 (model quality) for route B: vfx.runtime3d.ScreenDetail with its trained checkpoint.

Checkpoint: vfx.eval3d.checkpoint.screen_checkpoint() (env VFX_SCREEN_CKPT, else a default exported by the
owning modules / the sandbox's load_backends(checkpoint_b=...) default, else CHECKPOINTS/screen3d/latest.pt ...).
Dataset: $VFX_DATASET3D (default s3d_v1). Everything skips cleanly until its inputs exist.

Gates (SPEC Layer 4 + "Route B training phase"; numbers are never relaxed):
* overfit-one-batch on a FRESH ScreenNet (no checkpoint, no dataset needed): L1 to the target drops to <= 0.5x
  the identity L1 within 400 Adam steps.
* the runtime reads the CLAUDE.md aux dict {"alpha", "depth", "motion"}: same frames, aux zeroed -> output differs.
* on the test split, the model beats BOTH baselines (the unmodified lr frame and the unsharp mask tuned on train)
  on l1, spectrum_error and ms_dssim (vfx/eval3d/metrics.py).
* OOD reported separately; spec: quality within 20% of in-distribution. Gated on the score RELATIVE TO THE lr
  BASELINE of the same split, (model/lr)_ood <= 1.2 x (model/lr)_test per metric: the absolute ood/test ratio is
  dominated by how hard the OOD scenes are for any method (on s3d_evalcheck the unmodified lr frame alone has an
  ood/test spectrum-error ratio of 3.3), so it would not measure the model's generalization. The absolute ratio
  is printed too.
* flicker (opacity-weighted, own motion) ratio to the hr reference <= 1.2 (spec) and not worse than the lr input's
  own ratio unless still at or under the reference: ratio_model <= max(ratio_lr, 1.0).
  Flicker warps every sequence with its OWN estimated apparent flow (metrics.sequence_flow), not the stored
  fluid motion (which is not the image motion; with it the gate was blind). Also temporal_jerk (unwarped 2nd
  temporal difference) ratio to hr <= 1.2. On s3d_v1, 2-code iid shimmer moves flicker x1.35 and jerk x1.33.
* view consistency (detail swim on moving-camera sequences, metrics.detail_swim): swim(model - lr) /
  swim(hr - lr) <= 1.2. The spec has no separate number; 1.2 is the spec's temporal tolerance applied to the
  detail layer alone (stated here, not tuned).
* live (no dataset): 64^3 Solver3D on the four sandbox presets (their dissipating SolverConfig3D overrides), orbiting
  camera (0.35 rad/s, the sandbox auto-orbit), 600 frames at 2560x1440 (march_scale 0.5, sandbox default) through
  the real VolumeRenderer (aux + motion incl. camera motion when supported) + ScreenDetail (domain bbox, as the
  sandbox passes it). After 60 warm-up frames: everything finite, no frame > 50 ms (host wall incl. sync),
  mean ScreenDetail GPU ms <= 2.5 (spec detail budget), flicker(output) / flicker(lr render) (each warped with its
  OWN estimated flow, effect region, on 2x2-pooled frames = the stored scale) and the same for temporal_jerk,
  <= 1.2 x max(1, flicker(hr)/flicker(lr) measured on the test split).
  Runs with the untrained network when no checkpoint exists (timing only; marked "untrained").
Reports: DATA_ROOT/runs/eval3d/layer4_latest.json and live_latest.json.
"""
from __future__ import annotations

import inspect
import json
import math
import os
import time

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vfx.eval3d import metrics as M
from vfx.eval3d.checkpoint import screen_checkpoint

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
DEV = "cuda"
VERSION = os.environ.get("VFX_DATASET3D", "s3d_v1")
CKPT, CKPT_SRC = screen_checkpoint()
METRICS = ("l1", "spectrum_error", "ms_dssim")
FLICKER_SPEC = 1.2
SWIM_LIMIT = 1.2
OOD_LIMIT = 1.2
DETAIL_BUDGET_MS = 2.5
HITCH_MS = 50.0
LIVE_FRAMES, LIVE_WARMUP, LIVE_SIZE = 600, 60, (1440, 2560)
ORBIT_RATE = 0.35          # rad/s, sandbox auto-orbit


def _report(name: str, obj: dict):
    from vfx.core.paths import RUNS

    d = RUNS / "eval3d"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(obj, indent=1, default=str))


def _dataset_ready(split="test") -> bool:
    from vfx.core.paths import dataset_dir

    root = dataset_dir(VERSION)
    if not (root / "splits.json").exists():
        return False
    try:
        return len(M._load(VERSION, split, _length(), stride=_length())) > 0
    except Exception as e:  # noqa: BLE001
        print(f"[layer4] {VERSION}/{split} not loadable: {e!r}")
        return False


def _length() -> int:
    from vfx.core.paths import dataset_dir

    p = dataset_dir(VERSION) / "card.json"
    return int(json.loads(p.read_text()).get("frames_per_sequence", 16)) if p.exists() else 16


def need_ckpt():
    if CKPT is None:
        pytest.skip(f"route B checkpoint absent: {CKPT_SRC}")


def need_data(split="test"):
    if not _dataset_ready(split):
        pytest.skip(f"dataset {VERSION} split {split} not available yet")


# ============================================================================ overfit one batch (fresh net)
def _arch():
    if CKPT is not None:
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "config" in ck:
            return dict(ck["config"])
    try:
        from vfx.runtime3d.screen_detail import DEFAULT_ARCH
        return dict(DEFAULT_ARCH)
    except Exception:  # noqa: BLE001
        return {}


def _one_batch():
    """2 frames of 128^2: a real train crop if the dataset exists, else synthetic (lr = blurred hr)."""
    if _dataset_ready("train"):
        ds = M._load(VERSION, "train", 2, stride=2)
        it = ds[0]
        lr = M.to_float(torch.as_tensor(it["lr"])[:, :, 64:192, 64:192].to(DEV))
        hr = M.to_float(torch.as_tensor(it["hr"])[:, :, 64:192, 64:192].to(DEV))
        aux = torch.as_tensor(it["aux"])[:, :, 64:192, 64:192].to(DEV).float()
        return lr, hr, aux, "train crop"
    g = torch.Generator(device=DEV).manual_seed(0)
    base = F.interpolate(torch.rand(2, 3, 32, 32, device=DEV, generator=g), size=(128, 128), mode="bilinear")
    det = 0.15 * (torch.rand(2, 3, 128, 128, device=DEV, generator=g) - 0.5)
    hr = (base + det).clamp(0, 1)
    lr = M.gaussian_blur(hr, 1.5)
    aux = torch.zeros(2, 4, 128, 128, device=DEV)
    aux[:, 0] = 1.0
    return lr, hr, aux, "synthetic"


def test_overfit_one_batch_fresh_net():
    mm = pytest.importorskip("vfx.model3d")
    if not hasattr(mm, "ScreenNet"):
        pytest.skip("vfx.model3d has no ScreenNet")
    torch.manual_seed(0)
    arch = _arch()
    cfg_t = getattr(mm, "ScreenNetConfig", None)
    if cfg_t is not None:
        names = set(inspect.signature(cfg_t).parameters)
        net = mm.ScreenNet(cfg_t(**{k: v for k, v in arch.items() if k in names})).to(DEV)
    else:
        net = mm.ScreenNet(**arch).to(DEV)
    lr, hr, aux, src = _one_batch()
    a = aux.clone()
    a[:, 2:4] = a[:, 2:4] / 16.0
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    use_aux = getattr(getattr(net, "cfg", None), "use_aux", True)
    use_prev = getattr(getattr(net, "cfg", None), "use_prev", True)
    l0 = (lr - hr).abs().mean().item()
    loss = None
    for _ in range(400):
        d = net(lr, a if use_aux else None, lr if use_prev else None)
        loss = ((lr + d) - hr).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    lf = loss.item()
    print(f"\n[layer4] overfit ({src}, arch {arch}): L1 identity {l0:.5f} -> {lf:.5f} ({lf / l0:.3f}x) in 400 steps")
    assert math.isfinite(lf) and lf <= 0.5 * l0, f"fresh ScreenNet cannot overfit one batch: {l0:.5f} -> {lf:.5f}"


# ============================================================================ dataset evaluation
@pytest.fixture(scope="module")
def baselines_eval():
    need_data("test")
    L = _length()
    sh = M.tune_unsharp(VERSION, "train")
    return {sp: (M.evaluate_screen_detail(None, VERSION, sp, length=L, sharpen=sh) if _dataset_ready(sp) else None)
            for sp in ("test", "ood")} | {"sharpen": sh}


@pytest.fixture(scope="module")
def model_eval(baselines_eval):
    need_ckpt()
    L = _length()
    model = M.load_screen_detail_for_dataset(CKPT, DEV)
    out = {"checkpoint": str(CKPT), "checkpoint_source": CKPT_SRC, "version": VERSION,
           "created": time.strftime("%Y-%m-%d %H:%M:%S"), "sharpen": baselines_eval["sharpen"]}
    for sp in ("test", "ood"):
        out[sp] = M.evaluate_screen_detail(model, VERSION, sp, length=L, sharpen=baselines_eval["sharpen"]) \
            if _dataset_ready(sp) else None
    _report("layer4_latest.json", out)
    for sp in ("test", "ood"):
        if out[sp]:
            print(f"\n[layer4] {sp} ({out[sp]['n_sequences']} seqs):")
            for name, m in out[sp]["methods"].items():
                print(f"   {name:8s} " + "  ".join(f"{k} {v:.5g}" for k, v in m.items()))
    return out


def test_baselines_report(baselines_eval):
    """Numbers the ML engineer trains against (no checkpoint needed): lr and sharpen vs the hr reference."""
    sh = baselines_eval["sharpen"]
    print(f"\n[layer4] unsharp tuned on train: sigma {sh['sigma']} amount {sh['amount']} objective {sh['objective']:.4f}")
    for sp in ("test", "ood"):
        r = baselines_eval[sp]
        if not r:
            continue
        print(f"[layer4] {sp} ({r['n_sequences']} seqs) reference flicker_hr {r['reference']['flicker_hr']:.5g} "
              f"swim_hr {r['reference']['swim_hr']:.5g}")
        for name, m in r["methods"].items():
            print(f"   {name:8s} " + "  ".join(f"{k} {v:.5g}" for k, v in m.items()))
    _report("layer4_baselines_latest.json", baselines_eval)
    assert sh["objective"] <= 1.0


def test_runtime_reads_contract_aux(baselines_eval):
    """ScreenDetail.infer must consume the CLAUDE.md aux keys: zeroing alpha/depth/motion must change the output."""
    need_ckpt()
    ds = M._load(VERSION, "test", _length(), stride=_length())
    it = ds[0]
    lr = torch.as_tensor(it["lr"]).to(DEV)
    aux = torch.as_tensor(it["aux"]).to(DEV)
    sd = M.load_screen_detail_for_dataset(CKPT, DEV)
    a = M.run_sequence(sd, lr, aux)
    b = M.run_sequence(sd, lr, torch.zeros_like(aux))
    diff = (a.int() - b.int()).abs().float().mean().item()
    assert not torch.equal(a, lr), "model output equals its input on this sequence (identity / untrained weights)"
    print(f"\n[layer4] mean |out(aux) - out(aux=0)| = {diff:.4f} codes")
    assert diff > 0.0, ("output does not depend on aux {'alpha','depth','motion'}: the runtime ignores the "
                        "CLAUDE.md aux dict (check key names / layout)")


@pytest.mark.parametrize("metric", METRICS)
def test_beats_lr_and_sharpen_on_test(model_eval, metric):
    m = model_eval["test"]["methods"]
    print(f"\n[layer4] test {metric}: model {m['model'][metric]:.5g}  lr {m['lr'][metric]:.5g}  "
          f"sharpen {m['sharpen'][metric]:.5g}")
    assert m["model"][metric] < m["lr"][metric], f"{metric}: model {m['model'][metric]:.5g} >= lr {m['lr'][metric]:.5g}"
    assert m["model"][metric] < m["sharpen"][metric], \
        f"{metric}: model {m['model'][metric]:.5g} >= sharpen {m['sharpen'][metric]:.5g}"


@pytest.mark.parametrize("metric", METRICS)
def test_ood_within_20pct(model_eval, metric):
    if not model_eval.get("ood"):
        pytest.skip("no OOD split")
    t, o = model_eval["test"]["methods"], model_eval["ood"]["methods"]
    ra = o["model"][metric] / max(t["model"][metric], 1e-12)
    rl = o["lr"][metric] / max(t["lr"][metric], 1e-12)
    qt = t["model"][metric] / max(t["lr"][metric], 1e-12)
    qo = o["model"][metric] / max(o["lr"][metric], 1e-12)
    r = qo / max(qt, 1e-12)
    print(f"\n[layer4] OOD {metric}: model/lr ood {qo:.3f} vs test {qt:.3f} -> ratio {r:.3f} (gate <= {OOD_LIMIT}); "
          f"absolute model ood/test {ra:.3f}, lr ood/test {rl:.3f}")
    for p in model_eval["ood"]["per_sequence"]:
        print(f"   ood seed {p['seed']} cam {p['cam']}: model {p['model'][metric]:.5g} lr {p['lr'][metric]:.5g}")
    assert r <= OOD_LIMIT, f"OOD {metric}: model/lr {qo:.3f} vs {qt:.3f} in-distribution = {r:.3f}x (spec: within 20%)"


def test_flicker_ratio(model_eval):
    m = model_eval["test"]["methods"]
    rm, rl, rs = m["model"]["flicker_ratio"], m["lr"]["flicker_ratio"], m["sharpen"]["flicker_ratio"]
    print(f"\n[layer4] flicker ratio vs hr: model {rm:.3f} (static {m['model']['flicker_ratio_static']:.3f}, "
          f"moving {m['model']['flicker_ratio_moving']:.3f}); lr {rl:.3f}; sharpen {rs:.3f}")
    assert rm <= FLICKER_SPEC, f"flicker ratio {rm:.3f} > {FLICKER_SPEC} (spec)"
    assert rm <= max(rl, 1.0), f"model flickers more than its lr input ({rm:.3f} vs lr {rl:.3f}) and more than hr"


def test_temporal_jerk_ratio(model_eval):
    m = model_eval["test"]["methods"]
    rj = m["model"]["jerk_ratio"]
    print(f"\n[layer4] temporal jerk ratio vs hr: model {rj:.3f}; lr {m['lr']['jerk_ratio']:.3f}; "
          f"sharpen {m['sharpen']['jerk_ratio']:.3f}")
    assert rj <= FLICKER_SPEC, f"temporal jerk ratio {rj:.3f} > {FLICKER_SPEC}: per-frame shimmer"


def test_view_consistency(model_eval):
    m = model_eval["test"]["methods"]["model"]
    r = m["swim_ratio_moving"]
    which = "moving-camera"
    if not np.isfinite(r):
        r, which = m["swim_ratio"], "all (no moving-camera sequences)"
    print(f"\n[layer4] detail swim ratio ({which}) model {r:.3f}; sharpen "
          f"{model_eval['test']['methods']['sharpen']['swim_ratio']:.3f}")
    assert r <= SWIM_LIMIT, f"added detail swims {r:.3f}x the reference's (limit {SWIM_LIMIT})"


# ============================================================================ live (no dataset)
def _domain_bbox(cam, h, w):
    from vfx.render3d import camera_basis

    b = camera_basis(cam, w / h)
    o, fw, rt, up, th = b[0:3], b[3:6], b[6:9], b[9:12], b[12]
    c = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], np.float32) - o
    zc = c @ fw
    if (zc <= 1e-3).any():
        return (0, 0, w, h)
    sx = (c @ rt) / zc / (th * w / h)
    sy = (c @ up) / zc / th
    px, py = (sx + 1) * 0.5 * w, (1 - sy) * 0.5 * h
    return (int(np.clip(px.min(), 0, w)), int(np.clip(py.min(), 0, h)),
            int(np.ceil(np.clip(px.max(), 0, w))), int(np.ceil(np.clip(py.max(), 0, h))))


def _orbit(t):
    from vfx.core.fields3d import Camera

    yaw, pitch, dist, tg = ORBIT_RATE * t, 0.20, 2.15, np.array([0.5, 0.42, 0.5])    # sandbox Orbit defaults
    cp = math.cos(pitch)
    off = np.array([cp * math.sin(yaw), math.sin(pitch), -cp * math.cos(yaw)]) * dist
    return Camera(position=tuple((tg + off).tolist()), target=tuple(tg.tolist()), fov_y_deg=40.0)


def _hr_over_lr_flicker():
    """flicker(hr) / flicker(lr) on the test split, if measured (makes the live limit lenient only if hr itself
    flickers more than the lr render)."""
    try:
        from vfx.core.paths import RUNS
        p = RUNS / "eval3d" / "layer4_baselines_latest.json"
        r = json.loads(p.read_text())["test"]["methods"]["lr"]["flicker_ratio"]
        return 1.0 / r if r and r > 0 else None
    except Exception:  # noqa: BLE001
        return None


@pytest.fixture(scope="module")
def live():
    sim = pytest.importorskip("vfx.sim3d")
    rd = pytest.importorskip("vfx.render3d")
    rt = pytest.importorskip("vfx.runtime3d")
    try:
        from vfx.app3d.presets import PRESETS
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"sandbox presets not importable: {e!r}")
    from vfx.core.fields3d import CH3, LR_RES_3D, SIM_DT, Forces3D, Volume, rasterize_obstacles_3d
    from bench3d.run import gpu_check

    gpu = gpu_check(samples=2)
    H, W = LIVE_SIZE
    torch.manual_seed(0)
    sd = rt.ScreenDetail(str(CKPT) if CKPT else None, device=DEV)
    renderer = rd.VolumeRenderer((H, W), device=DEV, march_scale=0.5)
    params = inspect.signature(renderer.draw).parameters
    has_aux = "return_aux" in params
    prev_kw = next((k for k in ("prev_camera", "camera_prev", "prev_cam") if k in params), None)
    rcfg = getattr(sim, "runtime_config", None) or sim.SolverConfig3D
    out = {"checkpoint": str(CKPT) if CKPT else None, "untrained": CKPT is None, "renderer_aux": has_aux,
           "prev_camera_kw": prev_kw, "gpu": {k: gpu.get(k) for k in ("shared", "reasons", "utilization_pct")},
           "frames": LIVE_FRAMES, "warmup": LIVE_WARMUP, "size": [H, W], "presets": {}}
    for pid, preset in sorted(PRESETS.items()):
        try:
            cfg = rcfg(**preset.sim)
        except TypeError:
            cfg = rcfg()
        solver = sim.Solver3D(LR_RES_3D, cfg, device=DEV)
        st = solver.reset(list(preset.obstacles))
        solid = rasterize_obstacles_3d(list(preset.obstacles), 128, DEV) if preset.obstacles else None
        forces = Forces3D(emitters=list(preset.emitters), wind=preset.wind)
        sd.reset()
        wall, det, nonfinite = [], [], 0
        fl_out = fl_lr = jk_out = jk_lr = 0.0
        n_fl = n_jk = 0
        prev_lr = prev_out = prev2_lr = prev2_out = None
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        prev_cam = _orbit(-SIM_DT)
        for f in range(LIVE_FRAMES):
            cam = _orbit(f * SIM_DT)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            st = solver.step(st, forces, SIM_DT)
            vol = Volume(st.data[CH3["density"]], st.data[CH3["temperature"]])
            aux = None
            if has_aux:
                kw = {"return_aux": True, "velocity": st.velocity}
                if prev_kw:
                    kw[prev_kw] = prev_cam
                img, aux = renderer.draw(vol, cam, solid, **kw)
            else:
                img = renderer.draw(vol, cam, solid)
            ad = dict(aux) if aux else {}
            ad["bbox"] = _domain_bbox(cam, H, W)
            ev0.record()
            o = sd.infer(img, ad)
            ev1.record()
            torch.cuda.synchronize()
            wall.append((time.perf_counter() - t0) * 1000)
            det.append(ev0.elapsed_time(ev1))
            prev_cam = cam
            if f % 50 == 0 or f == LIVE_FRAMES - 1:
                nonfinite += int((~torch.isfinite(st.data)).sum())
                if aux:
                    nonfinite += sum(int((~torch.isfinite(v.float())).sum()) for v in aux.values()
                                     if isinstance(v, torch.Tensor))
            # judged at the stored / processed scale (2x2 pool of 1440p), like the dataset metrics
            cur_lr = F.avg_pool2d(M.to_float(img.permute(2, 0, 1))[None], 2)[0]
            cur_out = F.avg_pool2d(M.to_float(o.permute(2, 0, 1))[None], 2)[0]
            if f >= LIVE_WARMUP and prev_lr is not None:
                if aux and "alpha" in aux:
                    wm = M.effect_mask(F.avg_pool2d(aux["alpha"].float()[None, None], 2)[0, 0])
                else:
                    wm = torch.ones(cur_lr.shape[-2:], dtype=torch.bool, device=DEV)
                w2 = wm[None].expand(2, -1, -1)
                for seq_pair, key in ((torch.stack([prev_lr, cur_lr]), "lr"), (torch.stack([prev_out, cur_out]), "out")):
                    v = M.flicker(seq_pair, None, w2)          # each sequence with ITS OWN estimated flow
                    if key == "lr":
                        fl_lr += v
                    else:
                        fl_out += v
                n_fl += 1
                if prev2_lr is not None:
                    w3 = wm[None].expand(3, -1, -1)
                    jk_lr += M.temporal_jerk(torch.stack([prev2_lr, prev_lr, cur_lr]), w3)
                    jk_out += M.temporal_jerk(torch.stack([prev2_out, prev_out, cur_out]), w3)
                    n_jk += 1
            prev2_lr, prev2_out = prev_lr, prev_out
            prev_lr, prev_out = cur_lr, cur_out
        w, d = np.array(wall[LIVE_WARMUP:]), np.array(det[LIVE_WARMUP:])
        out["presets"][preset.name] = {
            "wall_ms_mean": float(w.mean()), "wall_ms_p95": float(np.percentile(w, 95)), "wall_ms_max": float(w.max()),
            "frames_over_50ms": int((w > HITCH_MS).sum()), "detail_ms_mean": float(d.mean()),
            "detail_ms_p95": float(np.percentile(d, 95)), "nonfinite": nonfinite,
            "warmup_wall_ms_max": float(np.max(wall[:LIVE_WARMUP])),
            "flicker_out": fl_out / n_fl if n_fl else float("nan"), "flicker_lr": fl_lr / n_fl if n_fl else float("nan"),
            "flicker_ratio_vs_lr": (fl_out / fl_lr) if n_fl and fl_lr > 0 else float("nan"),
            "jerk_ratio_vs_lr": (jk_out / jk_lr) if n_jk and jk_lr > 0 else float("nan"),
            "processed_fraction": float(sd.timings.get("processed_fraction", float("nan"))),
            "density_max_end": float(st.ch("density").max())}
        del solver, st
    out["hr_over_lr_flicker_test"] = _hr_over_lr_flicker()
    _report("live_latest.json", out)
    print(f"\n[layer4 live] {'UNTRAINED' if CKPT is None else CKPT} renderer aux={has_aux} prev_camera={prev_kw} "
          f"GPU shared={gpu.get('shared')} {gpu.get('reasons')}")
    for k, v in out["presets"].items():
        print(f"   {k:28s} " + "  ".join(f"{a} {b:.4g}" if isinstance(b, float) else f"{a} {b}" for a, b in v.items()))
    return out


def test_live_finite_and_no_hitches(live):
    for name, r in live["presets"].items():
        assert r["nonfinite"] == 0, f"{name}: {r['nonfinite']} non-finite values"
        assert r["frames_over_50ms"] == 0, f"{name}: {r['frames_over_50ms']} frames > 50 ms (max {r['wall_ms_max']:.1f})"


def test_live_detail_budget_1440p(live):
    worst = max(r["detail_ms_mean"] for r in live["presets"].values())
    tag = "untrained" if live["untrained"] else "trained"
    print(f"\n[layer4 live] mean ScreenDetail ms at 2560x1440 ({tag}): " +
          ", ".join(f"{k} {r['detail_ms_mean']:.3f}" for k, r in live["presets"].items()) +
          f"; GPU shared={live['gpu'].get('shared')}")
    assert worst <= DETAIL_BUDGET_MS, (f"detail stage {worst:.3f} ms > {DETAIL_BUDGET_MS} ms budget "
                                       f"(GPU shared: {live['gpu'].get('shared')} {live['gpu'].get('reasons')})")


def test_live_flicker_vs_lr_render(live):
    f = live.get("hr_over_lr_flicker_test")
    limit = FLICKER_SPEC * max(1.0, f or 1.0)
    rs = {k: r["flicker_ratio_vs_lr"] for k, r in live["presets"].items()}
    rj = {k: r["jerk_ratio_vs_lr"] for k, r in live["presets"].items()}
    print(f"\n[layer4 live] temporal jerk(output)/jerk(lr render) {rj}")
    print(f"\n[layer4 live] flicker(output)/flicker(lr render) {rs}; limit {limit:.3f} (hr/lr on test: {f}); "
          f"{'UNTRAINED: identity, ratio is trivially ~1' if live['untrained'] else ''}")
    bad = {k: v for k, v in list(rs.items()) + [(k + " (jerk)", v) for k, v in rj.items()] if not (v <= limit)}
    assert not bad, f"live flicker over limit {limit:.3f}: {bad}"
