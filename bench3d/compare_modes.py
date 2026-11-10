"""One comparison harness for every smoke/fire approach (owner: 3D test/eval engineer).

    python -m bench3d.compare_modes [--presets 1 2 3 4] [--frames N] [--out DIR] [--size 2560x1440]
           [--warmup 90] [--march-scale 0.5] [--screen-ckpts PATH ...] [--no-cost] [--no-quality]

For each approach that is AVAILABLE right now (everything else is listed with the reason it was skipped), on the
four sandbox presets, with one fixed orbiting camera (vfx.app3d.compare.orbit_cam: start -30 deg, 1 deg per sim
step, distance 1.9) and the matched steered-256^3 reference (vfx.datagen3d.sim.run_pair with DEFAULT_GUIDE, exactly
as vfx/app3d/compare.py gets it), all rendered by VolumeRenderer at the same size and march_scale:

  QUALITY  (one process; every frame = one sim step after `--warmup` steps)
    legacy, pixel-aligned  : l1, ms_dssim, spectrum_error (+ log ratio)       on K full-res frames
    distribution-level     : band_spectrum (per octave + distance), gradient / local-contrast histogram W1,
                             patch SWD                                         on the same K full-res frames
    temporal, vs reference : tLP (+ log ratio, pop), tOF                       on all frames, 2x2-pooled
    legacy temporal gates  : flicker ratio, jerk ratio, detail swim ratio     on all frames, 2x2-pooled
    All on the effect region = dilated union of the raw-64^3 render alpha and the reference render alpha, shared by
    every approach (vfx/eval3d/metrics.py; definitions in tests3d/README.md).
  COST     (one subprocess per approach, all presets; CUDA events; nvidia-smi read first)
    sim / detail / render / frame GPU ms (p50, mean, worst) at --size vs the 2.0 / 2.5 / 1.5 / 6.0 ms budget,
    torch peak allocated MB and device-level used delta MB (Warp allocations are only in the latter).

Approaches (rows): reference (steered 256^3, anchor), raw 64^3 (anchor), trilinear 256^3, unsharp (classical
screen-space baseline, sigma/amount tuned on s3d_v1 train at the 2x-pooled scale, applied x2 at full res), route B
ScreenDetail per checkpoint (latest.pt, v1_reg_backup.pt, adversarial.pt / adv_latest.pt if present, the newest
gan_*.pt, plus --screen-ckpts), AdvectedDetail (vfx.detail3d, rendered with VolumeRenderer.draw(detail=...)),
Solver3D(scalar_res=128) rendering hires_volume(), pressure_mode="fused", fused + scalar_res + AdvectedDetail
("combo"), and a direct 128^3 sim (full-quality config, as the sandbox's ref128 mode).
Approaches built on the raw 64^3 sim (trilinear, unsharp, route B, AdvectedDetail) use the SAME 64^3 states the
reference is steered to follow. scalar_res / fused / combo / direct 128^3 run their own sim from the same forces
(their large scale drifts from the reference's guide; recorded per row).

Writes bench3d/results/MODES.md and bench3d/results/modes.json (always), and per preset a contact sheet of labelled
1:1 crops of every approach side by side to --out (default DATA_ROOT/runs/compare_modes/latest, outside the repo).
Evaluation code: host syncs are fine here; the frame loop being timed is the cost worker.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BUDGET = {"sim": 2.0, "detail": 2.5, "render": 1.5, "frame": 6.0}
RESULTS = Path(__file__).resolve().parent / "results"
START_DEG, ORBIT_DEG, DIST = -30.0, 1.0, 1.9
SOLID_RES = 128
MARK = "@@COST_JSON@@"


# ============================================================================ discovery
def _try(fn):
    try:
        return fn(), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"


def _ckpt_dir():
    from vfx.core.paths import CHECKPOINTS
    return CHECKPOINTS / "screen3d"


def screen_ckpts(extra=()) -> list[tuple[str, Path, str, bool]]:
    """[(label, path, why, report_only)] route B checkpoints to compare (lead's designation 2026-11-09)."""
    d = _ckpt_dir()
    out = []
    for name, why, ro in (("final1_v2_reg_20261109_210028.pt", "route B regression (designated; s3d_v2)", False),
                          ("v1_reg_backup.pt", "route B regression v1 (old, kept for history)", False),
                          ("final2_v2_gan_20261109_212544.pt", "route B adversarial (designated)", False)):
        if (d / name).is_file():
            out.append((name, d / name, why, ro))
    for p in sorted(d.glob("teacher*.pt"), key=lambda p: p.stat().st_mtime):
        if not p.name.startswith("teacher_probe"):
            out.append((p.name, p, "teacher (REPORT-ONLY, not a budget candidate)", True))
    for p in extra:
        p = Path(p)
        if p.is_file() and all(p != q for _, q, _, _ in out):
            out.append((p.name, p, "--screen-ckpts", False))
    return out


def discover(extra_ckpts=()) -> list[dict]:
    """Every approach with availability and the reason when unavailable. Row spec keys:
    sim: "lr" (the 64^3 runtime states the reference is steered to follow) | "hr" | {"res", "cfg", "extra",
    "scalar_res"} (own sim); vol: "state" | "tri" | "hires" | "particles"; detail_q; post: None|"unsharp"|"routeB";
    matched: whether the reference is a matched twin of this row's 64^3 flow."""
    rows = []
    sim, e_sim = _try(lambda: __import__("vfx.sim3d", fromlist=["_"]))
    rd, e_rd = _try(lambda: __import__("vfx.render3d", fromlist=["_"]))
    rt, e_rt = _try(lambda: __import__("vfx.runtime3d", fromlist=["_"]))
    rp, e_rp = _try(lambda: __import__("vfx.datagen3d.sim", fromlist=["run_pair"]).run_pair)
    det, e_det = _try(lambda: __import__("vfx.detail3d", fromlist=["AdvectedDetail"]).AdvectedDetail)
    par, e_par = _try(lambda: __import__("vfx.particles3d", fromlist=["DetailParticles"]).DetailParticles)
    base_ok = sim is not None and rd is not None
    base_reason = e_sim or e_rd

    def add(key, label, ok, reason, sim_="lr", vol="state", detail_q=None, post=None, **kw):
        own = isinstance(sim_, dict)
        rows.append({"key": key, "label": label, "available": bool(ok), "reason": None if ok else reason,
                     "sim": sim_, "vol": vol, "detail_q": detail_q, "post": post,
                     "base": "own" if own else ("hr" if sim_ == "hr" else "raw64"),
                     "matched": not own, **kw})       # own sim (any solver option) = not the reference's twin

    add("reference", "steered 256^3 (reference)", base_ok and rp is not None, base_reason or e_rp, sim_="hr")
    add("raw", "raw 64^3", base_ok, base_reason)
    add("trilinear", "trilinear 256^3", base_ok and rt is not None, base_reason or e_rt, vol="tri")
    add("unsharp", "unsharp mask (screen)", base_ok, base_reason, post="unsharp")
    for label, path, why, ro in screen_ckpts(extra_ckpts):
        add(f"routeB:{path}", f"route B {label}" + (" [report-only]" if ro else ""),
            base_ok and rt is not None and hasattr(rt, "ScreenDetail"), base_reason or e_rt, post="routeB",
            checkpoint=str(path), checkpoint_note=why, report_only=ro)
    has_detail_kw = base_ok and "detail" in inspect.signature(rd.VolumeRenderer.draw).parameters
    det_ok = det is not None and has_detail_kw
    det_reason = e_det if det is None else ("VolumeRenderer.draw has no `detail` parameter yet" if not has_detail_kw else None)
    for q in ("medium", "low", "high"):
        add(f"detail_{q}", f"detail (AdvectedDetail {q})", base_ok and det_ok, base_reason or det_reason, detail_q=q)
    scal_ok = base_ok and "scalar_res" in inspect.signature(sim.Solver3D.__init__).parameters \
        and hasattr(sim.Solver3D, "hires_volume")
    fused_ok = base_ok and hasattr(sim, "fused_config")
    conf_ok = base_ok and hasattr(sim.SolverConfig3D(), "confinement_ref_res")
    for tag, cfg, extra, ok, why in (("", "runtime", {}, True, None),
                                     ("_fused", "fused", {}, fused_ok, "vfx.sim3d.fused_config missing"),
                                     ("_confref", "runtime", {"confinement_ref_res": 128}, conf_ok,
                                      "SolverConfig3D.confinement_ref_res missing")):
        spec = {"res": 64, "cfg": cfg, "extra": extra, "scalar_res": 128}
        nm = {"": "", "_fused": " + fused_config()", "_confref": " + confinement_ref_res=128"}[tag]
        add(f"hires{tag}", f"hires (scalar_res=128){nm}", scal_ok and ok, base_reason or why or
            "Solver3D(scalar_res=...) missing", sim_=spec, vol="hires")
        add(f"hires_detail{tag}", f"hires_detail (scalar_res=128 + detail medium){nm}", scal_ok and ok and det_ok,
            base_reason or why or det_reason or "scalar_res missing", sim_=spec, vol="hires", detail_q="medium")
    add("fused", "raw 64^3 + fused_config()", fused_ok, base_reason or "fused_config missing",
        sim_={"res": 64, "cfg": "fused", "extra": {}, "scalar_res": None})
    add("particles", "particles DetailParticles(1M, 128)", base_ok and par is not None, base_reason or e_par,
        vol="particles")
    add("particles_detail", "particles 1M@128 + detail medium", base_ok and par is not None and det_ok,
        base_reason or e_par or det_reason, vol="particles", detail_q="medium")
    add("direct128", "direct 128^3 sim", base_ok, base_reason, sim_={"res": 128, "cfg": "full", "extra": {}, "scalar_res": None})
    try:
        from vfx.app3d.pipeline import MODES
        known = {"raw", "trilinear", "neural_vol", "neural_screen", "ref128"}
        rows.append({"key": "_sandbox_modes", "sandbox_modes": list(MODES), "new_sandbox_modes": [m for m in MODES if m not in known]})
    except Exception:  # noqa: BLE001
        pass
    return rows


# ============================================================================ runners
def _overrides(pre) -> dict:
    return dict(pre.sim) if pre.sim else {}


def _cfg(sim, kind: str, ov: dict, extra: dict):
    from bench3d.run import make_config
    fac = {"runtime": sim.runtime_config, "fused": getattr(sim, "fused_config", None), "full": sim.SolverConfig3D}[kind]
    return make_config(fac, {**ov, **extra})


class Runner:
    """One approach on one preset. own_sim=False: states come from `set_state` (quality pass shares the 64^3 run
    the reference is steered to follow, or the reference itself). Stages: step (sim), detail (pre-render volume work:
    upscale, detail advection, particles), render, post (screen-space work on the rendered frame)."""

    def __init__(self, row: dict, pre, size, march_scale: float, own_sim: bool, want_aux: bool = False):
        import torch
        from vfx.core.fields3d import rasterize_obstacles_3d
        from vfx.render3d import VolumeRenderer
        import vfx.sim3d as sim

        self.row, self.key, self.pre = row, row["key"], pre
        self.H, self.W = size
        self.rend = VolumeRenderer((self.H, self.W), march_scale=march_scale)
        self.solid = rasterize_obstacles_3d(pre.obstacles, SOLID_RES) if pre.obstacles else None
        self.want_aux = want_aux
        self.st = self.vol = self.frame = self.aux = None
        self.solver = self.det = self.sd = self.dp = None
        self.prev_cam = None
        self.forces = preset_forces(pre)
        ov = _overrides(pre)
        rec = {"preset_overrides": ov, "solver_version": getattr(sim, "SOLVER3D_VERSION", None),
               "solver_tag": _try(lambda: sim.solver_version_tag())[0] if hasattr(sim, "solver_version_tag") else None,
               "matched_reference": row.get("matched", True)}
        spec = row["sim"]
        if isinstance(spec, dict) or own_sim:
            if not isinstance(spec, dict):
                spec = ({"res": 256, "cfg": "full", "extra": {}, "scalar_res": None} if spec == "hr"
                        else {"res": 64, "cfg": "runtime", "extra": {}, "scalar_res": None})
            cfg = _cfg(sim, spec["cfg"], ov, spec["extra"])
            kw = {"scalar_res": spec["scalar_res"]} if spec.get("scalar_res") else {}
            self.solver = sim.Solver3D(spec["res"], cfg, **kw)
            self.st = self.solver.reset(pre.obstacles)
            rec["sim"] = (f"Solver3D({spec['res']}, {spec['cfg']}_config(**preset, **{spec['extra']})"
                          f"{', scalar_res=%d' % spec['scalar_res'] if spec.get('scalar_res') else ''})"
                          + (" UNGUIDED (cost proxy of the reference; guide excluded)" if row["sim"] == "hr" else ""))
            rec["config"] = {f: getattr(cfg, f) for f in ("pressure_tol", "pressure_max_iters", "pressure_mode",
                                                         "confinement_ref_res", "vorticity_eps") if hasattr(cfg, f)}
        else:
            rec["sim"] = ("vfx.datagen3d.sim.run_pair: guided Solver3D(256, SolverConfig3D(**preset)), DEFAULT_GUIDE"
                          if spec == "hr" else "vfx.datagen3d.sim.run_pair LR: Solver3D(64, runtime_config(**preset))")
        if row.get("post") == "unsharp":
            sh = _unsharp_params()
            self.us = (2.0 * sh["sigma"], sh["amount"])
            rec["unsharp"] = {"sigma_fullres": self.us[0], "amount": self.us[1], "source": sh["source"]}
        if row.get("post") == "routeB":
            from vfx.runtime3d import ScreenDetail
            torch.manual_seed(0)
            self.sd = ScreenDetail(row["checkpoint"], device="cuda")
            if getattr(self.sd, "untrained", False):
                raise RuntimeError(f"checkpoint {row['checkpoint']} loaded as UNTRAINED")
            p = Path(row["checkpoint"])
            rec["checkpoint"] = {"path": str(p), "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)),
                                 "note": row.get("checkpoint_note"), "process_scale": getattr(self.sd, "scale", None)}
            self.want_aux = True
        if row.get("detail_q"):
            from vfx.detail3d import AdvectedDetail
            self.det = AdvectedDetail(64, device="cuda", quality=row["detail_q"])
            self.det.reset()
            rec["advected_detail"] = {"quality": row["detail_q"], "params": _jsonable(getattr(self.det, "params", None))}
        if row["vol"] == "particles":
            from vfx.particles3d import DetailParticles
            self.dp = DetailParticles(1_000_000, 128, config=_cfg(sim, "runtime", ov, {}))
            self.dp.reset()
            rec["particles"] = "DetailParticles(1_000_000, 128, config=runtime_config(**preset)); defaults"
        if row["vol"] == "tri":
            from vfx.runtime3d import classical_upscale_3d
            self._up = classical_upscale_3d
            rec["upscale"] = "runtime3d.classical_upscale_3d(state, 4, 'trilinear')"
        rec["renderer"] = {"size": [self.W, self.H], "march_scale": march_scale}
        self.record = rec

    # -------------------------------------------------------------- stages
    def set_state(self, st):
        self.st = st

    def step(self, forces):
        from vfx.core.fields3d import SIM_DT
        self.st = self.solver.step(self.st, forces, SIM_DT)

    @property
    def ticks(self) -> bool:
        """Per-sim-step work that must run every step (detail coordinates, particles), even during warm-up."""
        return self.det is not None or self.dp is not None

    def detail(self):
        from vfx.core.fields3d import SIM_DT, Volume
        v = self.row["vol"]
        if v == "tri":
            self.vol = self._up(self.st, 4, "trilinear")
            return
        if self.det is not None:
            self.det.step(self.st, SIM_DT)
        if self.dp is not None:
            self.dp.step(self.st, self.forces, SIM_DT)
            self.vol = self.dp.volume()
        elif v == "hires":
            self.vol = self.solver.hires_volume()
        else:
            self.vol = Volume(self.st.ch("density"), self.st.ch("temperature"))

    def render(self, cam):
        kw = {}
        if self.det is not None:
            kw["detail"] = self.det
        if self.want_aux:
            self.frame, self.aux = self.rend.draw(self.vol, cam, self.solid, return_aux=True,
                                                  velocity=self.st.velocity, prev_camera=self.prev_cam, **kw)
        else:
            self.frame = self.rend.draw(self.vol, cam, self.solid, **kw)
        self.prev_cam = cam

    def post(self, cam):
        if self.sd is not None:
            import torch.nn.functional as F
            from bench3d.run import domain_bbox
            a = dict(self.aux or {})
            hdr = getattr(self.rend, "hdr", None)
            if hdr is not None:                                  # as vfx/app3d/compare.py passes it
                tr = hdr[..., 3]
                if tr.shape != (self.H, self.W):
                    tr = F.interpolate(tr[None, None], size=(self.H, self.W), mode="bilinear", align_corners=False)[0, 0]
                a["transmittance"] = tr
            a["bbox"] = domain_bbox(cam, self.H, self.W)
            self.frame = self.sd.infer(self.frame, a)
        elif self.row.get("post") == "unsharp":
            from vfx.eval3d.metrics import unsharp
            self.frame = unsharp(self.frame, *self.us)


def _jsonable(x):
    try:
        json.dumps(x)
        return x
    except Exception:  # noqa: BLE001
        return repr(x)[:300]


def _unsharp_params() -> dict:
    try:
        from vfx.core.paths import RUNS
        p = RUNS / "eval3d" / "unsharp_s3d_v1_train.json"
        d = json.loads(p.read_text())
        return {"sigma": float(d["sigma"]), "amount": float(d["amount"]), "source": str(p)}
    except Exception:  # noqa: BLE001
        return {"sigma": 2.0, "amount": 0.25, "source": "fallback (route B latest.json sharpen values)"}


def cam_at(k: int):
    from vfx.app3d.compare import orbit_cam
    return orbit_cam(START_DEG + ORBIT_DEG * k, DIST)


def preset_forces(pre):
    from vfx.core.fields3d import Forces3D
    return Forces3D(emitters=list(pre.emitters), wind=tuple(pre.wind))


# ============================================================================ quality pass
def _pool2_u8(img_hwc):
    import torch.nn.functional as F
    x = img_hwc.permute(2, 0, 1)[None].float()
    return (F.avg_pool2d(x, 2)[0] + 0.5).clamp(0, 255).to(dtype=img_hwc.dtype).cpu()


def quality_preset(pid: int, rows: list[dict], frames: int, warmup: int, size, march_scale: float, k_spatial: int,
                   log, ref_res: int = 256) -> tuple[dict, dict]:
    """Render every available approach on preset pid; returns (per-approach metrics, contact-sheet material)."""
    import torch
    import torch.nn.functional as F
    from vfx.app3d.compare import PresetScenario
    from vfx.app3d.presets import PRESETS
    from vfx.datagen3d.guide3d import DEFAULT_GUIDE
    from vfx.datagen3d.sim import run_pair
    from vfx.eval3d import metrics as M

    pre = PRESETS[pid]
    H, W = size
    avail = [r for r in rows if r.get("available") and not r["key"].startswith("_")]
    runners, records, failed = {}, {}, {}
    for r in avail:
        rr, err = _try(lambda: Runner(r, pre, size, march_scale, own_sim=False,
                                      want_aux=r["key"] in ("reference", "raw")))
        if rr is None:
            failed[r["key"]] = err
            log(f"  {r['key']}: construction failed: {err}")
        else:
            runners[r["key"]] = rr
            records[r["key"]] = rr.record
    forces = preset_forces(pre)
    sub_idx = set(np.linspace(0, frames - 1, min(k_spatial, frames)).round().astype(int).tolist())
    seq_half = {k: [] for k in runners}
    full = {k: {} for k in runners}
    alpha_half, alpha_full = [], {}
    t0 = time.time()
    gen = run_pair(PresetScenario(pre, warmup + frames), DEFAULT_GUIDE, hr_res=ref_res)
    with torch.no_grad():
        for step, lr, hr, _ in gen:
            k = step - warmup
            for key, rn in runners.items():
                if rn.solver is not None:
                    rn.step(forces)
                elif key == "reference":
                    rn.set_state(hr)
                else:
                    rn.set_state(lr)
                if rn.ticks:
                    rn.detail()                      # detail coordinates / particles advance from step 0
            if k < -10:
                continue
            cam = cam_at(k)
            for key, rn in runners.items():
                if not rn.ticks:
                    rn.detail()
                rn.render(cam)
                rn.post(cam)
            if k < 0:                                 # route B temporal state / camera history warm-up
                continue
            a_raw = runners["raw"].aux["alpha"].float() if "raw" in runners else None
            a_ref = runners["reference"].aux["alpha"].float() if "reference" in runners else None
            a = a_raw if a_ref is None else (a_ref if a_raw is None else torch.maximum(a_raw, a_ref))
            alpha_half.append(F.avg_pool2d(a[None, None], 2)[0, 0].cpu())
            if k in sub_idx:
                alpha_full[k] = a.cpu()
            for key, rn in runners.items():
                seq_half[key].append(_pool2_u8(rn.frame))
                if k in sub_idx:
                    full[key][k] = rn.frame.permute(2, 0, 1).contiguous().cpu()
            if k % 16 == 0:
                log(f"  preset {pid} frame {k}/{frames}  ({time.time() - t0:.0f} s)")
    torch.cuda.synchronize()
    del gen
    # ------------------------------------------------------------ metrics
    dev = "cuda"
    ks = sorted(alpha_full)
    mask_full = M.effect_mask(torch.stack([alpha_full[i] for i in ks]).to(dev))
    mask_half = M.effect_mask(torch.stack(alpha_half).to(dev))

    def flow_of(seq):
        x = M.to_float(seq.to(dev))
        fl = torch.zeros(x.shape[0], 2, *x.shape[-2:], device=dev)
        for i in range(1, x.shape[0], 8):
            j = min(i + 8, x.shape[0])
            fl[i:j] = M.estimate_flow(x[i - 1:j - 1], x[i:j])
        return fl

    out = {}
    ref_key = "reference" if "reference" in runners else None
    if ref_key is None:
        log("  no reference available: quality metrics skipped")
        return {k: {"record": records[k]} for k in runners} | {k: {"error": v} for k, v in failed.items()}, {}
    ref_seq = torch.stack(seq_half[ref_key])
    ref_full = torch.stack([full[ref_key][i] for i in ks]).to(dev)
    ref_flow = flow_of(ref_seq)
    ref_fl_err = M.warp_error(ref_seq.to(dev), ref_flow, mask_half)
    ref_jerk = M.temporal_jerk(ref_seq.to(dev), mask_half)
    raw_seq = torch.stack(seq_half["raw"]).to(dev) if "raw" in seq_half else None
    ref_swim = M.detail_swim(ref_seq.to(dev), raw_seq, ref_flow, mask_half) if raw_seq is not None else float("nan")
    for key in runners:
        seq = torch.stack(seq_half[key]).to(dev)
        fr = torch.stack([full[key][i] for i in ks]).to(dev)
        fo = ref_flow if key == ref_key else flow_of(seq)
        r = {"record": records[key]}
        r["l1"] = M.l1(fr, ref_full, mask_full)
        r["ms_dssim"] = M.ms_dssim(fr, ref_full, mask_full)
        r["spectrum_error"] = M.spectrum_error(fr, ref_full, mask_full)
        r["spectrum_log_ratio"] = M.spectrum_log_ratio(fr, ref_full, mask_full)
        bs = M.band_spectrum(fr, ref_full, mask_full)
        r["band_distance"] = bs["distance"]
        r.update({f"band_{n}": v for n, v in bs["bands"].items()})
        r["grad_hist"] = M.gradient_hist_distance(fr, ref_full, mask_full)
        r["contrast_hist"] = M.local_contrast_hist_distance(fr, ref_full, mask_full)
        r["patch_swd"] = M.patch_swd(fr, ref_full, mask_full)
        r.update(M.tlp(seq, ref_seq.to(dev), mask_half, flow_out=fo, flow_ref=ref_flow))
        r["tof"] = M.tof(seq, ref_seq.to(dev), mask_half, flow_out=fo, flow_ref=ref_flow)
        r["flicker_ratio"] = M.warp_error(seq, fo, mask_half) / max(ref_fl_err, 1e-12)
        r["jerk_ratio"] = M.temporal_jerk(seq, mask_half) / max(ref_jerk, 1e-12)
        if runners[key].row["base"] == "raw64" and raw_seq is not None and key != "raw":
            r["swim_ratio"] = M.detail_swim(seq, raw_seq, fo, mask_half) / max(ref_swim, 1e-12)
        out[key] = r
        del seq, fr
        torch.cuda.empty_cache()
    for k, v in failed.items():
        out[k] = {"error": v}
    sheet = {"frame": ks[len(ks) // 2], "alpha": alpha_full[ks[len(ks) // 2]],
             "frames": {k: full[k][ks[len(ks) // 2]] for k in runners}}
    return out, sheet


def contact_sheet(pid: int, sheet: dict, rows: list[dict], path: Path, crop: int = 360):
    """Two 1:1 crops (plume body at the reference/raw alpha centroid, plume top) of every approach side by side."""
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    import torch
    pygame.font.init()
    font = pygame.font.SysFont("consolas", 15, bold=True)
    a = sheet["alpha"].float()
    H, W = a.shape
    ys = torch.arange(H, dtype=torch.float32)[:, None]
    xs = torch.arange(W, dtype=torch.float32)[None, :]
    wsum = a.sum().clamp(min=1e-6)
    cy, cx = float((a * ys).sum() / wsum), float((a * xs).sum() / wsum)
    rows_on = torch.nonzero((a > 0.05).any(1)).flatten()
    top = float(rows_on[0]) if rows_on.numel() else cy - crop
    band = a[int(top):int(top) + crop // 2]
    tx = float((band * xs).sum() / band.sum().clamp(min=1e-6)) if band.numel() else cx
    centres = [(cx, cy), (tx, top + crop / 2 - 20)]
    labels = {r["key"]: r["label"] for r in rows if "label" in r}
    keys = list(sheet["frames"])
    lab_h = 22
    surf = pygame.Surface((len(keys) * (crop + 4), len(centres) * (crop + lab_h + 4)))
    surf.fill((30, 30, 34))
    for j, (x, y) in enumerate(centres):
        x0 = int(np.clip(x - crop / 2, 0, W - crop))
        y0 = int(np.clip(y - crop / 2, 0, H - crop))
        for i, k in enumerate(keys):
            img = sheet["frames"][k][:, y0:y0 + crop, x0:x0 + crop].permute(1, 2, 0).contiguous().numpy()
            s = pygame.image.frombuffer(img.tobytes(), (img.shape[1], img.shape[0]), "RGB")
            px, py = i * (crop + 4), j * (crop + lab_h + 4)
            surf.blit(s, (px, py + lab_h))
            t = font.render(labels.get(k, k)[:44], True, (255, 230, 150))
            surf.blit(t, (px + 4, py + 3))
    path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surf, str(path))
    return {"frame": sheet["frame"], "crops_xy": [[int(np.clip(x - crop / 2, 0, W - crop)),
                                                   int(np.clip(y - crop / 2, 0, H - crop))] for x, y in centres],
            "crop": crop, "path": str(path)}


# ============================================================================ cost pass
def cost_worker(key: str, presets: list[int], size, march_scale: float, warmup: int, iters: int) -> dict:
    import torch
    try:
        import warp as wp
        wp.config.quiet = True
    except Exception:  # noqa: BLE001
        pass
    from vfx.app3d.presets import PRESETS
    from bench3d.run import stats

    row = next((r for r in discover() if r.get("key") == key), None)
    if row is None and key.startswith("routeB:"):
        p = Path(key.split(":", 1)[1])
        row = {"key": key, "label": f"route B {p.name}", "base": "raw64", "available": p.is_file(), "checkpoint": str(p),
               "sim": "lr", "vol": "state", "detail_q": None, "post": "routeB"}
    if row is None or not row.get("available"):
        return {"key": key, "skipped": (row or {}).get("reason", "unknown approach")}
    torch.cuda.init()
    f0, tot = torch.cuda.mem_get_info()
    used0 = (tot - f0) / 2 ** 20
    torch.cuda.reset_peak_memory_stats()
    res = {"key": key, "presets": {}, "record": None}
    peak_dev = 0.0
    for pid in presets:
        pre = PRESETS[pid]
        rn = Runner(row, pre, size, march_scale, own_sim=True)
        res["record"] = rn.record
        forces = preset_forces(pre)
        offline = key == "reference"
        n_warm = 30 if offline else warmup
        n_it = min(iters, 20) if offline else iters
        with torch.no_grad():
            for i in range(n_warm):
                rn.step(forces)
                if rn.ticks:
                    rn.detail()
            for i in range(10):                                   # renderer / graph capture, camera history
                cam = cam_at(i)
                rn.step(forces)
                rn.detail()
                rn.render(cam)
                rn.post(cam)
            torch.cuda.synchronize()
            evs = [[torch.cuda.Event(enable_timing=True) for _ in range(4)] for _ in range(n_it)]
            for i, e in enumerate(evs):
                cam = cam_at(10 + i)
                e[0].record()
                rn.step(forces)
                e[1].record()
                rn.detail()
                e[2].record()
                rn.render(cam)
                rn.post(cam)
                e[3].record()
            torch.cuda.synchronize()
        sim = [e[0].elapsed_time(e[1]) for e in evs]
        pre_render = [e[1].elapsed_time(e[2]) for e in evs]
        render_post = [e[2].elapsed_time(e[3]) for e in evs]
        frame = [e[0].elapsed_time(e[3]) for e in evs]
        # render vs post split: time post alone on the last frame's input (screen-space work)
        post_ms = []
        if rn.sd is not None or rn.row.get("post") == "unsharp":
            for i in range(min(20, n_it)):
                cam = cam_at(10 + n_it + i)
                rn.render(cam)
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                rn.post(cam)
                b.record()
                torch.cuda.synchronize()
                post_ms.append(a.elapsed_time(b))
        post_med = float(np.median(post_ms)) if post_ms else 0.0
        render = [max(x - post_med, 0.0) for x in render_post]
        detail = [x + post_med for x in pre_render]
        f1, _ = torch.cuda.mem_get_info()
        peak_dev = max(peak_dev, (tot - f1) / 2 ** 20 - used0)
        res["presets"][pid] = {"sim": stats(sim), "detail": stats(detail), "render": stats(render),
                               "frame": stats(frame), "post_ms_median": post_med, "offline": offline,
                               "detail_timings": _jsonable(dict(getattr(rn.det, "timings", {}) or {})) if rn.det else None,
                               "particle_timings": _jsonable(dict(getattr(rn.dp, "timings", {}) or {})) if rn.dp else None}
        del rn
        torch.cuda.empty_cache()
    res["vram"] = {"torch_peak_allocated_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
                   "device_used_delta_mb": peak_dev}
    return res


def run_cost(rows, presets, size, march_scale, warmup, iters, log) -> dict:
    out = {}
    for r in rows:
        if r["key"].startswith("_") or not r.get("available"):
            continue
        cmd = [sys.executable, "-m", "bench3d.compare_modes", "--cost-worker", r["key"], "--presets",
               *[str(p) for p in presets], "--size", f"{size[1]}x{size[0]}", "--march-scale", str(march_scale),
               "--cost-warmup", str(warmup), "--cost-iters", str(iters)]
        log(f"cost: {r['key']}")
        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                               cwd=str(Path(__file__).resolve().parents[1]))
            line = next((l for l in p.stdout.splitlines() if l.startswith(MARK)), None)
            out[r["key"]] = json.loads(line[len(MARK):]) if line else {
                "key": r["key"], "error": f"exit {p.returncode}: {(p.stderr or p.stdout)[-800:]}"}
        except Exception as e:  # noqa: BLE001
            out[r["key"]] = {"key": r["key"], "error": repr(e)}
        log(f"   {time.time() - t0:.0f} s  {'ERROR ' + out[r['key']]['error'][:200] if 'error' in out[r['key']] else ''}")
    return out


# ============================================================================ report
Q_COLS = [("l1", "L1", 4), ("ms_dssim", "MS-DSSIM", 3), ("spectrum_error", "spec err", 3),
          ("band_distance", "band dist", 3), ("band_2-4px", "2-4px", 2), ("band_8-16px", "8-16px", 2),
          ("grad_hist", "grad W1", 3), ("contrast_hist", "contr W1", 3), ("patch_swd", "patch SWD", 3)]
T_COLS = [("tlp", "tLP", 3), ("tlp_log_ratio", "tLP log", 2), ("pop", "pop", 2), ("tof", "tOF px", 2),
          ("flicker_ratio", "flicker r", 2), ("jerk_ratio", "jerk r", 2), ("swim_ratio", "swim r", 2)]


def _mean(vals):
    v = [x for x in vals if isinstance(x, (int, float)) and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def summarize(rows, quality, cost, presets) -> list[dict]:
    table = []
    for r in rows:
        if r["key"].startswith("_"):
            continue
        k = r["key"]
        t = {"key": k, "label": r["label"], "available": r["available"], "reason": r.get("reason"),
             "matched": r.get("matched", True), "report_only": r.get("report_only", False)}
        if r["available"]:
            for c, _, _ in Q_COLS + T_COLS:
                t[c] = _mean([quality.get(p, {}).get(k, {}).get(c) for p in presets])
            errs = {p: quality.get(p, {}).get(k, {}).get("error") for p in presets}
            t["quality_errors"] = {p: e for p, e in errs.items() if e}
            c = cost.get(k, {})
            if "presets" in c:
                for s in ("sim", "detail", "render", "frame"):
                    t[f"{s}_ms"] = _mean([c["presets"][p][s]["p50_ms"] for p in c["presets"]])
                t["worst_frame_ms"] = max(c["presets"][p]["frame"]["max_ms"] for p in c["presets"])
                t["vram_mb"] = c["vram"]["torch_peak_allocated_mb"]
                t["vram_dev_mb"] = c["vram"]["device_used_delta_mb"]
                t["offline"] = any(c["presets"][p].get("offline") for p in c["presets"])
                t["within_budget"] = all(t[f"{s}_ms"] <= BUDGET[s] for s in BUDGET) and not t["offline"]
                t["over_budget_stages"] = [s for s in BUDGET if t[f"{s}_ms"] > BUDGET[s]]
            elif c:
                t["cost_error"] = c.get("error") or c.get("skipped")
            rec = (c.get("record") or next((quality.get(p, {}).get(k, {}).get("record") for p in presets
                                            if quality.get(p, {}).get(k, {}).get("record")), None))
            t["record"] = rec
        table.append(t)
    return table


def _f(x, nd):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def write_report(meta, rows, table, quality, cost, sheets, presets):
    RESULTS.mkdir(parents=True, exist_ok=True)
    L = ["# Approach comparison - `python -m bench3d.compare_modes`", "",
         f"Generated {meta['created']}. Presets {presets}, {meta['frames']} recorded frames after {meta['warmup']} "
         f"warm-up steps, {meta['size'][0]}x{meta['size'][1]}, march_scale {meta['march_scale']}, orbiting camera "
         f"(start {START_DEG} deg, {ORBIT_DEG} deg/step, distance {DIST}).", ""]
    if meta.get("ref_res", 256) != 256:
        L += [f"> **REFERENCE IS A STEERED {meta['ref_res']}^3 SIM, NOT THE 256^3 REFERENCE** (--ref-res smoke test). "
              "Do not use these quality numbers for decisions.", ""]
    g = meta.get("gpu", {})
    if g.get("shared"):
        L += [f"> **GPU SHARED during this run** ({'; '.join(g.get('reasons', []))}). All ms numbers are NOT "
              "representative; the lead re-runs on an idle GPU.", ""]
    else:
        L += [f"GPU: {'idle' if g.get('shared') is False else 'unknown'} at start ({g.get('utilization_pct')}%).", ""]
    L += ["Lower is better everywhere except the signed columns (2-4px / 8-16px band log10 power ratio to the "
          "reference, tLP log ratio: 0 = like the reference, < 0 missing detail / too static, > 0 too much / boiling). "
          "`pop` 1 = as bursty as the reference. Ratios (flicker r, jerk r, swim r) are to the reference; they are "
          "legacy upper-bound gates that REWARD blur (r < 1 is not better). Quality = mean over presets; cost = "
          "mean of per-preset p50 GPU ms, worst = max single frame. Budget sim 2.0 / detail 2.5 / render 1.5 / "
          "frame 6.0 ms.", "",
          "Pixel-aligned (legacy) | distribution-level (fair to unaligned detail) | temporal vs reference | cost", ""]
    hdr = ["approach"] + [h for _, h, _ in Q_COLS] + [h for _, h, _ in T_COLS] + \
          ["sim ms", "detail ms", "render ms", "frame ms", "worst ms", "VRAM MB", "budget"]
    L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for t in table:
        if not t["available"]:
            continue
        lab = t["label"] + ("" if t.get("matched", True) else " (unmatched: own 64^3 flow)")
        cells = [f"**{lab}**" if t["key"] in ("reference", "raw") else lab]
        cells += [_f(t.get(c), nd) for c, _, nd in Q_COLS + T_COLS]
        cells += [_f(t.get(f"{s}_ms"), 2) for s in ("sim", "detail", "render", "frame")]
        cells += [_f(t.get("worst_frame_ms"), 1), _f(t.get("vram_mb"), 0)]
        if "within_budget" in t:
            verdict = "yes" if t["within_budget"] else "over: " + ",".join(t["over_budget_stages"])
            if t.get("offline"):
                verdict = "offline"
            elif t.get("report_only"):
                verdict = f"report-only ({verdict})"
            cells.append(verdict)
        else:
            cells.append(t.get("cost_error", "-")[:40] if t.get("cost_error") else "-")
        L.append("| " + " | ".join(cells) + " |")
    L += ["", "## Not available (re-run when they land)", ""]
    for t in table:
        if not t["available"]:
            L.append(f"- {t['label']}: {t['reason']}")
    sm = next((r for r in rows if r["key"] == "_sandbox_modes"), None)
    if sm:
        L.append(f"- sandbox modes: {sm['sandbox_modes']}; new since the screen-space baseline: "
                 f"{sm['new_sandbox_modes'] or 'none'} (harness rows are built from the module interfaces, not the "
                 "sandbox modes)")
    L += ["", "## What produced each row", ""]
    for t in table:
        if t["available"]:
            L.append(f"- **{t['label']}** - `{json.dumps(t.get('record'), default=str)}`")
    L += ["", "## Per preset", ""]
    for p in presets:
        q = quality.get(p, {})
        L.append(f"### preset {p}" + (f" - contact sheet `{sheets[p]['path']}` (frame {sheets[p]['frame']}, "
                                      f"1:1 crops {sheets[p]['crop']} px at {sheets[p]['crops_xy']})" if p in sheets else ""))
        L += ["", "| approach | L1 | MS-DSSIM | band dist | patch SWD | tLP | tLP log | pop | tOF | frame ms |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for t in table:
            r = q.get(t["key"])
            if not r:
                continue
            if "error" in r:
                L.append(f"| {t['label']} | error: {r['error'][:80]} |||||||||")
                continue
            cp = cost.get(t["key"], {}).get("presets", {})
            fm = cp.get(str(p), cp.get(p, {})).get("frame", {}).get("p50_ms")
            L.append("| " + " | ".join([t["label"]] + [_f(r.get(c), n) for c, n in (
                ("l1", 4), ("ms_dssim", 3), ("band_distance", 3), ("patch_swd", 3), ("tlp", 3),
                ("tlp_log_ratio", 2), ("pop", 2), ("tof", 2))] + [_f(fm, 2)]) + " |")
        L.append("")
    L += ["## Notes", "",
          "- Spatial metrics: K evenly spaced full-resolution frames; temporal metrics: all frames 2x2-pooled "
          "(the route B processed scale). Effect region = dilated union of raw-64^3 and reference alpha.",
          "- Rows on their own sim (scalar_res, fused, combo, direct 128^3) are NOT steered like the reference, so "
          "their pixel-aligned columns (L1, MS-DSSIM) include large-scale drift; compare them on the distribution and "
          "temporal columns.",
          "- swim r is only defined for approaches that add detail to the raw 64^3 render (detail layer = out - raw).",
          "- reference cost row = UNGUIDED 256^3 step + render (offline; guide cost excluded).",
          "- route B cost = raw render with aux (render) + ScreenDetail.infer (detail).",
          "- AdvectedDetail cost: detail = AdvectedDetail.step(); its per-sample cost in the march is inside render.",
          "- Identical rows mean identical outputs (e.g. two checkpoint files holding the same weights)."]
    (RESULTS / "MODES.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (RESULTS / "modes.json").write_text(json.dumps({"meta": meta, "rows": rows, "table": table, "quality": quality,
                                                    "cost": cost, "sheets": sheets}, indent=1, default=str))


# ============================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--presets", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--warmup", type=int, default=90)
    ap.add_argument("--out", default=None, help="contact sheets dir (default DATA_ROOT/runs/compare_modes/latest)")
    ap.add_argument("--size", default="2560x1440")
    ap.add_argument("--march-scale", type=float, default=0.5)
    ap.add_argument("--spatial-frames", type=int, default=8)
    ap.add_argument("--screen-ckpts", nargs="*", default=[])
    ap.add_argument("--ref-res", type=int, default=256, help="steered reference resolution (256 = THE reference; " "lower only for smoke-testing the harness, labelled in the report)")
    ap.add_argument("--no-cost", action="store_true")
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--cost-iters", type=int, default=60)
    ap.add_argument("--cost-warmup", type=int, default=60)
    ap.add_argument("--cost-worker", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    W, H = (int(v) for v in a.size.lower().split("x"))
    size = (H, W)
    if a.cost_worker:
        r = cost_worker(a.cost_worker, a.presets, size, a.march_scale, a.cost_warmup, a.cost_iters)
        print(MARK + json.dumps(r, default=str))
        return 0
    try:
        import warp as wp
        wp.config.quiet = True
    except Exception:  # noqa: BLE001
        pass
    from bench3d.run import gpu_check, loud_warning
    log = lambda s: print(s, flush=True)
    gpu = gpu_check()
    log("nvidia-smi: " + json.dumps({k: gpu.get(k) for k in ("utilization_pct", "memory_used_mb", "shared", "reasons")}))
    if gpu.get("shared"):
        loud_warning(gpu.get("reasons", []))
    rows = discover(a.screen_ckpts)
    for r in rows:
        if not r["key"].startswith("_"):
            log(f"{'+' if r['available'] else '-'} {r['label']:44s} {'' if r['available'] else r['reason']}")
    out_dir = Path(a.out) if a.out else None
    if out_dir is None:
        from vfx.core.paths import RUNS
        out_dir = RUNS / "compare_modes" / "latest"
    quality, sheets = {}, {}
    if not a.no_quality:
        for p in a.presets:
            log(f"quality: preset {p}")
            q, sh = quality_preset(p, rows, a.frames, a.warmup, size, a.march_scale, a.spatial_frames, log, a.ref_res)
            quality[p] = q
            if sh:
                sheets[p] = contact_sheet(p, sh, rows, out_dir / f"contact_p{p}.png")
            try:
                from vfx.datagen3d.sim import free_cache
                free_cache()
            except Exception:  # noqa: BLE001
                pass
    cost = {} if a.no_cost else run_cost(rows, a.presets, size, a.march_scale, a.cost_warmup, a.cost_iters, log)
    gpu_after = gpu_check()
    meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "presets": a.presets, "frames": a.frames,
            "warmup": a.warmup, "size": [W, H], "march_scale": a.march_scale, "spatial_frames": a.spatial_frames,
            "gpu": gpu, "gpu_after": gpu_after, "gpu_shared": bool(gpu.get("shared") or gpu_after.get("shared")),
            "budget_ms": BUDGET, "contact_sheets_dir": str(out_dir), "ref_res": a.ref_res}
    table = summarize(rows, quality, cost, a.presets)
    for t in table:
        t["gpu_shared"] = meta["gpu_shared"]
    write_report(meta, rows, table, quality, cost, sheets, a.presets)
    log(f"-> {RESULTS / 'MODES.md'}, {RESULTS / 'modes.json'}, sheets in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
