"""Band-power calibration of the advected detail against the steered 256³ reference (the bench3d MODES.md
metric: signed log10 power ratio per octave band, eval3d.band_spectrum, same camera path / size / mask as
bench3d.compare_modes).

Two phases, so the expensive part runs once:

    # 1. cache (one GPU session): per preset run vfx.datagen3d.sim.run_pair, store every 64³ state (fp16) and the
    #    reference + raw renders / alphas at the spatial frames under DATA_ROOT/runs/detail3d_tune/p<K>/
    python -m vfx.detail3d.tune_bands --cache [--presets 1 2 3 4] [--frames 48 --warmup 90 --spatial 8]

    # 2. evaluate variants (seconds per variant and preset): replays the cached 64³ states (detail coordinates and
    #    particles advance from step 0 exactly as in the harness), renders the spatial frames, prints bands
    python -m vfx.detail3d.tune_bands --variants "medium:base=detail;quality=medium" \\
        "cal:base=detail;octave_amps=0.012,0.004" "pd:base=particles_detail;base_res=128;cov_damp=0.8" ...
    python -m vfx.detail3d.tune_bands --grid          # the built-in candidate set (see GRID below)

Variant syntax: name:key=val;key=val. `base` in raw | detail | particles | particles_detail | hires | hires_detail
(hires runs its own scalar_res=128 solver live from step 0 with the same forces: own 64³ flow, "unmatched" as in
the harness). `quality` = low|medium|high; any other key is a DetailParams field (tuples comma-separated).
Output: per preset and mean, the five bands 32-64 / 16-32 / 8-16 / 4-8 / 2-4 px (0 = like the reference, > 0 too
much power, < 0 too little) and the band distance; also written to DATA_ROOT/runs/detail3d_tune/results.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

GRID = [
    "raw:base=raw",
    "low:base=detail;quality=low",
    "medium:base=detail;quality=medium",
    # calibrated candidates: less displacement overall, top octave much weaker than wavelength-proportional
    "c1:base=detail;octave_amps=0.016,0.004",
    "c2:base=detail;octave_amps=0.012,0.003",
    "c3:base=detail;octave_amps=0.010,0.002",
    "c4:base=detail;octave_amps=0.014",
    "c5:base=detail;octave_amps=0.010",
    "c6:base=detail;freq0=8.0;octave_amps=0.016,0.004",
    "c7:base=detail;freq0=8.0;octave_amps=0.012",
    # stacking, base-aware
    "particles:base=particles",
    "pd_old:base=particles_detail;quality=medium",
    "pd_b128:base=particles_detail;octave_amps=0.012,0.003;base_res=128",
    "pd_cov:base=particles_detail;octave_amps=0.012,0.003;cov_damp=0.9",
    "pd_both:base=particles_detail;octave_amps=0.012,0.003;base_res=128;cov_damp=0.7",
    "hires:base=hires",
    "hd_old:base=hires_detail;quality=medium",
    "hd_b128:base=hires_detail;octave_amps=0.012,0.003;base_res=128",
    "hd_b128s:base=hires_detail;octave_amps=0.008,0.002;base_res=128",
]


def search_specs() -> list[str]:
    """--search: a full grid over the octave-0 displacement, the top/octave-0 ratio and the base frequency for
    the plain detail (36 variants), plus raw / medium for reference. Use --search-stacked afterwards with the
    winner's amplitudes to tune base_res / cov_damp for the stacked modes."""
    out = ["raw:base=raw", "medium:base=detail;quality=medium"]
    for f0 in (8.0, 12.0):
        for a0 in (0.006, 0.008, 0.010, 0.012, 0.016, 0.020):
            for r in (0.0, 0.15, 0.3):
                amps = f"{a0}" if r == 0 else f"{a0},{a0 * r:.4f}"
                out.append(f"s_f{f0:.0f}_a{a0 * 1000:.0f}_r{int(r * 100)}:base=detail;freq0={f0};octave_amps={amps}")
    return out


def stacked_specs(amps: str, freq0: float) -> list[str]:
    out = ["particles:base=particles", "hires:base=hires"]
    for b in (64, 128, 256):
        out.append(f"hd_b{b}:base=hires_detail;freq0={freq0};octave_amps={amps};base_res={b}")
        for cd in (0.0, 0.5, 0.9):
            out.append(f"pd_b{b}_c{int(cd * 10)}:base=particles_detail;freq0={freq0};octave_amps={amps};base_res={b};cov_damp={cd}")
    return out


def _root(presets_dir=None) -> Path:
    from vfx.core.paths import RUNS
    return Path(presets_dir) if presets_dir else RUNS / "detail3d_tune"


def _pool2(img_hwc):
    import torch.nn.functional as F
    x = img_hwc.permute(2, 0, 1)[None].float()
    return (F.avg_pool2d(x, 2)[0] + 0.5).clamp(0, 255).to(torch.uint8).cpu()


def _harness_consts():
    import bench3d.compare_modes as CM
    return CM.cam_at, CM.preset_forces


# ============================================================================ phase 1: cache
@torch.no_grad()
def cache(presets, frames, warmup, spatial, size, march_scale, root: Path):
    from vfx.app3d.compare import PresetScenario
    from vfx.app3d.presets import PRESETS
    from vfx.core.fields3d import CH3, Volume, rasterize_obstacles_3d
    from vfx.datagen3d.guide3d import DEFAULT_GUIDE
    from vfx.datagen3d.sim import free_cache, run_pair
    from vfx.render3d import VolumeRenderer
    from vfx.render3d.bench import gpu_note
    cam_at, _ = _harness_consts()
    H, W = size
    print("GPU:", gpu_note())
    for pid in presets:
        d = root / f"p{pid}"
        d.mkdir(parents=True, exist_ok=True)
        pre = PRESETS[pid]
        solid = rasterize_obstacles_3d(pre.obstacles, 128) if pre.obstacles else None
        r_ref = VolumeRenderer((H, W), march_scale=march_scale)
        r_raw = VolumeRenderer((H, W), march_scale=march_scale)
        sub = sorted(set(np.linspace(0, frames - 1, min(spatial, frames)).round().astype(int).tolist()))
        states, ref, alpha = [], {}, {}
        ref_seq, alpha_half = [], []                     # every recorded frame, 2x2-pooled (as the harness)
        t0 = time.time()
        solid_lr = None
        for step, lr, hr, _ in run_pair(PresetScenario(pre, warmup + frames), DEFAULT_GUIDE):
            states.append(lr.data.half().cpu())
            solid_lr = lr.solid
            k = step - warmup
            if k >= 0:
                cam = cam_at(k)
                fr, ax = r_ref.draw(Volume(hr.data[CH3["density"]], hr.data[CH3["temperature"]]), cam, solid, return_aux=True)
                ref_seq.append(_pool2(fr))
                a_ref = ax["alpha"].float().clone()
                _, ax2 = r_raw.draw(Volume(lr.data[CH3["density"]], lr.data[CH3["temperature"]]), cam, solid, return_aux=True)
                alpha_half.append(torch.nn.functional.avg_pool2d(torch.maximum(a_ref, ax2["alpha"].float())[None, None], 2)[0, 0].cpu())
            if k in sub:
                cam = cam_at(k)
                fr, ax = r_ref.draw(Volume(hr.data[CH3["density"]], hr.data[CH3["temperature"]]), cam, solid, return_aux=True)
                ref[k] = fr.clone().cpu()
                a_ref = ax["alpha"].float().clone()
                _, ax2 = r_raw.draw(Volume(lr.data[CH3["density"]], lr.data[CH3["temperature"]]), cam, solid, return_aux=True)
                alpha[k] = torch.maximum(a_ref, ax2["alpha"].float()).cpu()
        torch.save({"states": torch.stack(states), "solid": solid_lr.cpu(), "warmup": warmup, "frames": frames,
                    "sub": sub, "ref": ref, "alpha": alpha, "size": [H, W], "march_scale": march_scale,
                    "ref_seq": torch.stack(ref_seq), "alpha_half": torch.stack(alpha_half)}, d / "cache.pt")
        print(f"preset {pid}: cached {len(states)} states + {len(ref)} reference frames in {time.time() - t0:.0f} s -> {d}")
        free_cache()
        torch.cuda.empty_cache()


# ============================================================================ phase 2: variants
def parse_variant(spec: str) -> tuple[str, dict]:
    name, _, kv = spec.partition(":")
    out = {}
    for t in [t for t in kv.split(";") if t]:
        k, v = t.split("=", 1)
        if "," in v:
            out[k] = tuple(float(x) for x in v.split(",") if x)
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    if k_ := out.get("octave_amps"):
        out["octave_amps"] = tuple(k_) if isinstance(k_, tuple) else (float(k_),)
    return name, out


@torch.no_grad()
def evaluate(presets, variants, root: Path, out_json: Path | None = None, save_pngs: bool = True,
             temporal: bool = False):
    from vfx.app3d.presets import PRESETS
    from vfx.core.fields3d import CH3, SIM_DT, FieldMeta3D, FieldState3D, Volume, rasterize_obstacles_3d
    from vfx.detail3d import AdvectedDetail
    from vfx.eval3d import metrics as M
    from vfx.render3d import VolumeRenderer
    cam_at, preset_forces = _harness_consts()
    results = {}
    for pid in presets:
        c = torch.load(root / f"p{pid}" / "cache.pt")
        H, W = c["size"]
        pre = PRESETS[pid]
        forces = preset_forces(pre)
        solid = rasterize_obstacles_3d(pre.obstacles, 128) if pre.obstacles else None
        sub = c["sub"]
        ref = torch.stack([c["ref"][k].permute(2, 0, 1) for k in sub]).cuda()
        mask = M.effect_mask(torch.stack([c["alpha"][k] for k in sub]).cuda())
        lr = FieldState3D(data=torch.zeros(c["states"].shape[1:], device="cuda"), solid=c["solid"].cuda(),
                          meta=FieldMeta3D.for_res(64))
        for name, kv in variants:
            kv = dict(kv)
            base = kv.pop("base", "detail")
            q = kv.pop("quality", "medium")
            det = AdvectedDetail(64, quality=q, **kv) if "detail" in base else None
            dp = hs = hst = None
            if base.startswith("particles"):
                from vfx.particles3d import DetailParticles
                from vfx.sim3d import runtime_config
                dp = DetailParticles(1_000_000, 128, config=runtime_config(**pre.sim))
            if base.startswith("hires"):
                from vfx.sim3d import Solver3D, runtime_config
                hs = Solver3D(64, runtime_config(**pre.sim), scalar_res=128)
                hst = hs.reset(pre.obstacles)
            rend = VolumeRenderer((H, W), march_scale=c["march_scale"])
            frames, seq = {}, []
            want_t = temporal and "ref_seq" in c
            for step in range(c["states"].shape[0]):
                lr.data.copy_(c["states"][step].cuda().float())
                st = lr
                if hs is not None:
                    hst = hs.step(hst, forces, SIM_DT)
                    st = hst
                if dp is not None:
                    dp.step(lr, forces, SIM_DT)
                if det is not None:
                    cov = dp.dep[2] if (dp is not None and hasattr(dp, "dep")) else None
                    det.step(st, SIM_DT, coverage=cov)
                k = step - c["warmup"]
                if want_t and k >= 0:
                    if dp is not None:
                        vol_t = dp.volume()
                    elif hs is not None:
                        vol_t = hs.hires_volume()
                    else:
                        vol_t = Volume(st.data[CH3["density"]], st.data[CH3["temperature"]])
                    seq.append(_pool2(rend.draw(vol_t, cam_at(k), solid, detail=det)))
                if k in sub:
                    if dp is not None:
                        vol = dp.volume()
                    elif hs is not None:
                        vol = hs.hires_volume()
                    else:
                        vol = Volume(st.data[CH3["density"]], st.data[CH3["temperature"]])
                    frames[k] = rend.draw(vol, cam_at(k), solid, detail=det).permute(2, 0, 1).clone()
            fr = torch.stack([frames[k] for k in sub])
            bs = M.band_spectrum(fr, ref, mask)
            swd = M.patch_swd(fr, ref, mask)
            results.setdefault(name, {})[pid] = {"bands": bs["bands"], "distance": bs["distance"], "patch_swd": swd,
                                                 "params": {"base": base, "quality": q, **{k: v for k, v in kv.items()}}}
            if want_t:
                tl = _tlp(torch.stack(seq), c["ref_seq"], c["alpha_half"])
                results[name][pid].update(tl)
            if save_pngs:
                _save_crop(root / f"p{pid}" / f"{name}.png", frames[sub[len(sub) // 2]])
            del det, dp, hs, rend
            torch.cuda.empty_cache()
        print(f"preset {pid} done")
    _print(results, presets)
    if out_json:
        out_json.write_text(json.dumps(results, indent=1, default=str))
        print("->", out_json)
    return results


_REF_FLOW: dict = {}


def _tlp(seq_hwc, ref_hwc, alpha_half) -> dict:
    """tLP / log ratio / pop exactly as bench3d.compare_modes (2x2-pooled frames, each sequence's own flow)."""
    from vfx.eval3d import metrics as M
    dev = "cuda"
    mask = M.effect_mask(alpha_half.to(dev))
    def flow_of(seq):
        x = M.to_float(seq.to(dev))
        fl = torch.zeros(x.shape[0], 2, *x.shape[-2:], device=dev)
        for i in range(1, x.shape[0], 8):
            j = min(i + 8, x.shape[0])
            fl[i:j] = M.estimate_flow(x[i - 1:j - 1], x[i:j])
        return fl
    out = seq_hwc.contiguous().to(dev)            # (N, 3, h, w) uint8 (as _pool2 returns, = the harness layout)
    ref = ref_hwc.contiguous().to(dev)
    key = (ref_hwc.data_ptr(), tuple(ref_hwc.shape))
    if key not in _REF_FLOW:
        _REF_FLOW.clear()
        _REF_FLOW[key] = flow_of(ref)
    r = M.tlp(out, ref, mask, flow_out=flow_of(out), flow_ref=_REF_FLOW[key])
    return {"tlp": r["tlp"], "tlp_log": r["tlp_log_ratio"], "pop": r["pop"]}


def _save_crop(path: Path, img_chw):
    import os
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    a = img_chw.permute(1, 2, 0).contiguous().cpu().numpy()
    pygame.image.save(pygame.image.frombuffer(a.tobytes(), (a.shape[1], a.shape[0]), "RGB"), str(path))


def _print(results, presets):
    from vfx.eval3d.metrics import OCTAVE_NAMES
    print("\nsigned band log10 power ratio vs steered 256³ (0 = like the reference); mean over presets "
          + str(presets))
    print(f"{'variant':12s} " + " ".join(f"{n:>8s}" for n in OCTAVE_NAMES) + f" {'dist':>6s} {'swd':>6s}   per-preset dist")
    for name, per in results.items():
        b = np.array([[per[p]["bands"][n] for n in OCTAVE_NAMES] for p in presets if p in per])
        dist = np.mean([per[p]["distance"] for p in presets if p in per])
        swd = np.mean([per[p]["patch_swd"] for p in presets if p in per])
        pp = " ".join(f"{per[p]['distance']:.3f}" for p in presets if p in per)
        t = ""
        if all("tlp" in per[p] for p in presets if p in per):
            t = (f"   tLP {np.mean([per[p]['tlp'] for p in presets]):.3f} log {np.mean([per[p]['tlp_log'] for p in presets]):+.2f}"
                 f" pop {np.mean([per[p]['pop'] for p in presets]):.2f}")
        print(f"{name:12s} " + " ".join(f"{v:+8.2f}" for v in b.mean(0)) + f" {dist:6.3f} {swd:6.3f}   {pp}{t}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--presets", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--warmup", type=int, default=90)
    ap.add_argument("--spatial", type=int, default=8)
    ap.add_argument("--size", default="2560x1440")
    ap.add_argument("--march-scale", type=float, default=0.5)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--search", action="store_true", help="36-variant grid for the plain detail (see search_specs)")
    ap.add_argument("--search-stacked", default=None, metavar="AMPS@FREQ0",
                    help="e.g. 0.010,0.0015@12: base_res / cov_damp grid for hires_detail and particles_detail")
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--temporal", action="store_true", help="also tLP / log ratio / pop on all recorded frames "
                    "(needs a cache made by this version: re-run --cache)")
    ap.add_argument("--root", default=None)
    a = ap.parse_args(argv)
    root = _root(a.root)
    W, H = (int(v) for v in a.size.lower().split("x"))
    if a.cache:
        cache(a.presets, a.frames, a.warmup, a.spatial, (H, W), a.march_scale, root)
    specs = (GRID if a.grid else []) + (search_specs() if a.search else []) + (a.variants or [])
    if a.search_stacked:
        amps, f0 = a.search_stacked.split("@")
        specs += stacked_specs(amps, float(f0))
    if specs:
        res = evaluate(a.presets, [parse_variant(s) for s in specs], root,
                       root / f"results_{time.strftime('%H%M%S')}.json", save_pngs=not a.no_png, temporal=a.temporal)
        rank = sorted(res, key=lambda n: (np.mean([v["distance"] for v in res[n].values()]),
                                          np.mean([v["patch_swd"] for v in res[n].values()])))
        print("\nbest by mean band distance (then patch SWD):", ", ".join(rank[:6]))


if __name__ == "__main__":
    main()
