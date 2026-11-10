"""Feasibility bench for the 3D detail routes (untrained weights, timing only).

  .venv/Scripts/python -m vfx.runtime3d.bench_selfcheck [--quick] [--out path.json]

Prints markdown tables and writes them as JSON (default vfx/runtime3d/bench_results.json, small).
Every number is the median per-frame GPU time (CUDA events around the whole infer(), so eager-mode
launch gaps count) and the allocator peak (torch.cuda.max_memory_allocated) above the pre-run baseline.
Records nvidia-smi state before/after: if another job shares the GPU, numbers are pessimistic.
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vfx.core.fields import AMBIENT_TEMP
from vfx.core.fields3d import CH3, FieldMeta3D, FieldState3D
from vfx.runtime3d import ScreenDetail, VolumeUpscaler, classical_upscale_3d, sample_bricks, warp_prev_3d
from vfx.runtime3d.warp3d import BrickGrid

DEV = "cuda"
OCC = [1.0, 0.5, 0.25, 0.10, 0.05]
BUDGET_MS = 2.5


# ---------------------------------------------------------------------------- utils
def gpu_state() -> dict:
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        u, m, t = (int(v) for v in q.strip().split(","))
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout
        return {"util_pct": u, "mem_used_mib": m, "mem_total_mib": t,
                "compute_apps": [a.strip() for a in apps.strip().splitlines() if a.strip()]}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def clean():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


LAST_MEDIAN: list = []      # median of the last measurement (the tables show p10; JSON keeps both)


def time_frames(fn, warm=4, iters=30, max_s=1.5) -> float:
    """Returns the 10th-percentile per-frame ms (robust to bursty work from other processes on a shared
    GPU); the median is appended to LAST_MEDIAN."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts, t0 = [], time.perf_counter()
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
        if time.perf_counter() - t0 > max_s and len(ts) >= 8:
            break
    LAST_MEDIAN.append(round(float(np.median(ts)), 3))
    return float(np.percentile(ts, 10))


def measure(make, frame, **kw):
    """make() -> object; frame(obj) runs one frame. Returns p10 ms, peak MiB above baseline, obj params."""
    clean()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    obj = make()
    ms = time_frames(lambda: frame(obj), **kw)
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    p = getattr(obj, "params", None)
    del obj
    clean()
    return round(ms, 3), round(peak, 1), p


def plume_state(R=64) -> FieldState3D:
    c = torch.arange(R, device=DEV, dtype=torch.float32)
    z, y, x = torch.meshgrid(c, c, c, indexing="ij")
    blob = torch.exp(-((x - R / 2) ** 2 + ((y - R / 3) / 2) ** 2 + (z - R / 2) ** 2) / (R * 1.5))
    d = torch.zeros(6, R, R, R, device=DEV)
    d[CH3["density"]] = blob
    d[CH3["temperature"]] = AMBIENT_TEMP + 600 * blob
    d[CH3["vel_y"]] = 1.0 * blob
    return FieldState3D(d, torch.zeros(R, R, R, device=DEV), FieldMeta3D.for_res(R))


def localized_ids(n_active: int, nb: int) -> np.ndarray:
    """The n bricks nearest a plume axis (tall in y) — a compact, realistic active set."""
    g = np.stack(np.meshgrid(*[np.arange(nb)] * 3, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    c = (nb - 1) / 2
    dist = (g[:, 0] - c) ** 2 + ((g[:, 1] - nb / 3) * 0.5) ** 2 + (g[:, 2] - c) ** 2
    return np.sort(np.argsort(dist, kind="stable")[:n_active])


def macs(model, lr_mods, hr_mods, s, dims):
    """MACs per HR output element: LR-resolution convs / s^dims + HR-resolution convs."""
    lr = sum(m.weight.numel() for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d))
             and m not in hr_mods)
    hr = sum(m.weight.numel() for m in hr_mods)
    return lr / s ** dims + hr


def put(r, key, ms, mib):
    r[key], r[key + "_mib"], r[key + "_med"] = ms, mib, LAST_MEDIAN[-1]


# ---------------------------------------------------------------------------- route A
def vol_run(state, occ, graph, sparse=True, **kw):
    def make():
        return VolumeUpscaler(cuda_graph=graph, sparse=sparse, dense_above=1.01 if sparse else 0.0, **kw)
    ids = None
    if sparse:
        tmp_nb = 64 // kw.get("brick", 8)
        ids = localized_ids(max(1, round(occ * tmp_nb ** 3)), tmp_nb)
    return measure(make, lambda u: u.infer(state, active=ids))


def bench_floor(state, rows):
    R = 64
    out = torch.zeros(1, 2, 256, 256, 256, device=DEV, dtype=torch.half)
    o2 = torch.empty_like(out)
    x16 = state.data[[0, 4]].unsqueeze(0).half()
    v16 = state.data[1:4].unsqueeze(0).half()
    def add(name, fn, note=""):
        clean()
        torch.cuda.reset_peak_memory_stats()
        b = torch.cuda.memory_allocated()
        with torch.inference_mode():
            ms = time_frames(fn, iters=30)
            med = LAST_MEDIAN[-1]
            try:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    fn()
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fn()
                ms_g = round(time_frames(g.replay, iters=30), 3)
            except RuntimeError as e:
                print("  (no graph for", name, ":", str(e)[:80], ")")
                ms_g = ""
        rows.append({"op": name, "ms_graph": ms_g, "ms": round(ms, 3), "median_ms": med,
                     "peak_mib": round((torch.cuda.max_memory_allocated() - b) / 2 ** 20, 1), "note": note})
    add("fill 256^3 x2 fp16", lambda: out.fill_(0), "pure write, 64 MiB")
    add("copy 256^3 x2 fp16", lambda: o2.copy_(out), "read+write")
    add("classical_upscale_3d trilinear x4 fp16", lambda: classical_upscale_3d(state), "separable (D then HW)")
    add("classical_upscale_3d trilinear x4 fp32", lambda: classical_upscale_3d(state, dtype=None))
    add("classical_upscale_3d nearest x4 fp16", lambda: classical_upscale_3d(state, mode="nearest"))
    add("F.interpolate trilinear x4 fp16 (stock)",
        lambda: F.interpolate(x16, scale_factor=4, mode="trilinear", align_corners=False), "reference only")
    add("classical_upscale_3d trilinear x2 fp16 (128^3)", lambda: classical_upscale_3d(state, factor=2))
    add("dense warp_prev_3d 256^3 fp16", lambda: warp_prev_3d(out, v16, 1 / 256, 1 / 30))
    out128 = torch.zeros(1, 2, 128, 128, 128, device=DEV, dtype=torch.half)
    add("dense warp_prev_3d 128^3 fp16", lambda: warp_prev_3d(out128, v16, 1 / 128, 1 / 30))
    samp = torch.cat([x16, v16], 1)
    bg = BrickGrid(256, 32, DEV)
    for occ in OCC:
        ids = torch.from_numpy(localized_ids(max(1, round(occ * 512)), 8)).to(DEV)
        g = bg.grid(ids)
        add(f"sparse warp+base, {occ:.0%} bricks", lambda: sample_bricks(samp, out, g, len(ids), 32, 1 / 256, 1 / 30))
    fr, _ = screen_inputs()
    fo = torch.empty_like(fr)
    add("route B: 1440p RGB8 frame copy", lambda: fo.copy_(fr), "output = input outside the bbox")
    add("route B: 1440p u8 -> fp16 -> u8 round trip",
        lambda: fo.copy_((fr.permute(2, 0, 1).half() * (1 / 255.0) * 255.0 + 0.5).permute(1, 2, 0).to(torch.uint8)))
    ms, mib, _ = measure(lambda: VolumeUpscaler(), lambda u: u.infer(state, active=np.zeros(0, np.int64)),
                         iters=30)
    rows.append({"op": "VolumeUpscaler pre-stage only (0 active bricks, graph)", "ms_graph": ms, "ms": "",
                 "median_ms": LAST_MEDIAN[-1], "peak_mib": mib,
                 "note": "cast + clamp + canvas + mask + async mask copy"})


def bench_volume(state, quick):
    widths = [16, 32] if quick else [16, 32, 48, 64]
    depths = [2, 4] if quick else [2, 4, 8]
    occs = [0.25, 0.10] if quick else OCC
    rows = []
    for c in widths:
        for n in depths:
            arch = dict(width=c, n_blocks=n)
            r = {"width": c, "depth": n}
            for graph in (True, False):
                tag = "g" if graph else "e"
                ms, mib, p = vol_run(state, 1.0, graph, sparse=False, **arch)
                r["params"] = p
                put(r, f"dense_{tag}", ms, mib)
                for occ in occs:
                    ms, mib, _ = vol_run(state, occ, graph, **arch)
                    put(r, f"{occ:.2f}_{tag}", ms, mib)
            rows.append(r)
            print(f"  A {c}x{n}: {r}", flush=True)
    return rows


def bench_volume_variants(state, quick):
    rows = []
    variants = [
        ("32x2 full", dict(width=32, n_blocks=2)),
        ("32x2 unfused (pure PyTorch HR stages)", dict(width=32, n_blocks=2, fused=False)),
        ("32x4 full (default)", dict(width=32, n_blocks=4)),
        ("32x4 unfused (pure PyTorch HR stages)", dict(width=32, n_blocks=4, fused=False)),
        ("32x4 sep (2+1)D", dict(width=32, n_blocks=4, block="sep")),
        ("32x4 mp (2D slices, cycling axes)", dict(width=32, n_blocks=4, block="mp")),
        ("32x4 dw (depthwise+pointwise)", dict(width=32, n_blocks=4, block="dw")),
        ("64x4 sep (2+1)D", dict(width=64, n_blocks=4, block="sep")),
        ("64x4 mp", dict(width=64, n_blocks=4, block="mp")),
        ("64x8 dw", dict(width=64, n_blocks=8, block="dw")),
        ("32x2 no channels_last", dict(width=32, n_blocks=2, channels_last=False)),
        ("32x2 x2 upscale (128^3)", dict(width=32, n_blocks=2, scale=2)),
        ("32x4 x2 upscale (128^3)", dict(width=32, n_blocks=4, scale=2)),
        ("64x4 x2 upscale (128^3)", dict(width=64, n_blocks=4, scale=2)),
        ("32x2 HR head 3x3x3 (unfused)", dict(width=32, n_blocks=2, hr_kernel=3)),
        ("32x2 HR feats 8 / hidden 16", dict(width=32, n_blocks=2, hr_feats=8, hr_hidden=16)),
        ("32x2 brick 8 halo 1", dict(width=32, n_blocks=2, brick=8, halo=1)),
        ("32x2 brick 8 halo 4", dict(width=32, n_blocks=2, brick=8, halo=4)),
        ("32x2 brick 16 halo 4", dict(width=32, n_blocks=2, brick=16, halo=4)),
        ("32x2 brick 4 halo 2", dict(width=32, n_blocks=2, brick=4, halo=2)),
    ]
    if quick:
        variants = variants[:4]
    for name, kw in variants:
        r = {"variant": name}
        for occ in (0.25, 0.10):
            ms, mib, p = vol_run(state, occ, True, **kw)
            put(r, f"{occ:.2f}_g", ms, mib)
            r["params"] = p
        rows.append(r)
        print(f"  A variant {r}", flush=True)
    return rows


# ---------------------------------------------------------------------------- route B
SCREEN = (1440, 2560)
BBOX = (512, 300, 1990, 1132)       # ~1/3 of the screen (quantized to 1536 x 832 = 35%)


def screen_inputs():
    H, W = SCREEN
    g = torch.Generator(device=DEV).manual_seed(0)
    fr = torch.randint(0, 256, (H, W, 3), device=DEV, dtype=torch.uint8, generator=g)
    aux = {"velocity": torch.randn(H, W, 2, device=DEV, generator=g),
           "depth": torch.rand(H, W, device=DEV, generator=g),
           "transmittance": torch.rand(H, W, device=DEV, generator=g)}
    return fr, aux


def scr_run(fr, aux, scale, bbox, graph, **kw):
    a = dict(aux, bbox=bbox) if bbox else aux
    return measure(lambda: ScreenDetail(process_scale=scale, cuda_graph=graph, **kw), lambda s: s.infer(fr, a))


def bench_screen(quick):
    fr, aux = screen_inputs()
    widths = [16, 32] if quick else [16, 32, 48, 64]
    depths = [2, 4] if quick else [2, 4, 8]
    modes = [("full", 1.0, None), ("half", 0.5, None), ("quarter", 0.25, None),
             ("bbox", 1.0, BBOX), ("bbox_half", 0.5, BBOX), ("bbox_quarter", 0.25, BBOX)]
    rows = []
    for c in widths:
        for n in depths:
            r = {"width": c, "depth": n}
            for name, sc, bb in modes:
                ms, mib, p = scr_run(fr, aux, sc, bb, True, width=c, n_blocks=n)
                put(r, name, ms, mib)
                r["params"] = p
            rows.append(r)
            print(f"  B {c}x{n}: {r}", flush=True)
    extra = []
    ex = [("64x4 (default arch), eager (no graph)", dict(width=64, n_blocks=4), False),
          ("32x4, eager (no graph)", dict(width=32, n_blocks=4), False),
          ("32x4 trunk stride 2", dict(width=32, n_blocks=4, trunk_stride=2), True),
          ("32x4 trunk stride 8", dict(width=32, n_blocks=4, trunk_stride=8), True),
          ("64x8 trunk stride 8", dict(width=64, n_blocks=8, trunk_stride=8), True),
          ("32x4 no aux, no prev", dict(width=32, n_blocks=4, use_aux=False, use_prev=False), True),
          ("32x4 HR head 3x3", dict(width=32, n_blocks=4, hr_kernel=3), True),
          ("32x4 no channels_last", dict(width=32, n_blocks=4, channels_last=False), True)]
    if quick:
        ex = ex[:2]
    for name, kw, graph in ex:
        r = {"variant": name}
        for mname, sc, bb in modes:
            ms, mib, p = scr_run(fr, aux, sc, bb, graph, **kw)
            put(r, mname, ms, mib)
            r["params"] = p
        extra.append(r)
        print(f"  B variant {r}", flush=True)
    return rows, extra


def bench_recommended(state):
    """The shipped defaults, exactly as the sandbox constructs them."""
    rows = []
    for occ in (0.05, 0.10, 0.20, 0.25):
        ids = localized_ids(round(occ * 512), 8)
        ms, mib, p = measure(lambda: VolumeUpscaler(), lambda u: u.infer(state, active=ids))
        rows.append({"config": "VolumeUpscaler() default", "case": f"{occ:.0%} bricks active", "ms": ms,
                     "median_ms": LAST_MEDIAN[-1], "peak_mib": mib, "params": p})
    fr, aux = screen_inputs()
    for case, bb in (("effect bbox ~1/3 screen", BBOX), ("whole 1440p frame", None)):
        a = dict(aux, bbox=bb) if bb else aux
        ms, mib, p = measure(lambda: ScreenDetail(), lambda s: s.infer(fr, a))
        rows.append({"config": "ScreenDetail() default (process_scale 0.5)", "case": case, "ms": ms,
                     "median_ms": LAST_MEDIAN[-1], "peak_mib": mib, "params": p})
    return rows


# ---------------------------------------------------------------------------- correctness
def selfcheck() -> dict:
    """Correctness (not timing): sparse warp == dense warp at brick voxels; fused Warp HR path ==
    PyTorch VolumeNet with random (non-zero) head weights; untrained outputs == classical / input."""
    from vfx.runtime3d.fused import FusedHR
    res = {}
    st = plume_state()
    g = torch.Generator(device=DEV).manual_seed(0)
    st.data[1:4] += 0.3 * torch.randn(st.data[1:4].shape, device=DEV, generator=g)
    prev = torch.rand(1, 2, 256, 256, 256, device=DEV, generator=g)
    vel = torch.randn(1, 3, 64, 64, 64, device=DEV, generator=g)
    dense = warp_prev_3d(prev, vel, 1 / 256, 1 / 30)
    ids = torch.tensor([0, 77, 300, 511], device=DEV)
    samp = torch.cat([st.data[[0, 4]].unsqueeze(0), vel], 1)
    _, pw = sample_bricks(samp, prev, BrickGrid(256, 32, DEV, torch.float32).grid(ids), 4, 32, 1 / 256, 1 / 30)
    err = 0.0
    for i, b in enumerate(ids.tolist()):
        z, y, x = (b // 64) * 32, ((b // 8) % 8) * 32, (b % 8) * 32
        err = max(err, (pw[i] - dense[0, :, z:z + 32, y:y + 32, x:x + 32]).abs().max().item())
    res["sparse_vs_dense_warp_maxerr_fp32"] = err
    for name, active in (("sparse", localized_ids(100, 8)), ("dense", None)):
        kw = dict(sparse=active is not None, dense_above=1.01 if active is not None else 0.0)
        ut = VolumeUpscaler(fused=False, **kw)
        torch.manual_seed(1)
        torch.nn.init.normal_(ut.model.hr2.weight, std=0.05)
        uf = VolumeUpscaler(fused=True, **kw)
        uf.model.load_state_dict(ut.model.state_dict())
        uf.fused = FusedHR(uf.model, DEV)
        d = c = 0.0
        for _ in range(3):
            a, b = ut.infer(st, active=active), uf.infer(st, active=active)
            d = max(d, (a.density.float() - b.density.float()).abs().max().item())
            c = max(c, (a.density.float() - classical_upscale_3d(st).density.float()).abs().max().item())
        res[f"fused_vs_torch_{name}_density_maxerr"] = d
        res[f"fused_vs_torch_{name}_net_change"] = c
    up = VolumeUpscaler()
    for _ in range(3):
        v = up.infer(st, active=localized_ids(50, 8))
    ref = classical_upscale_3d(st)
    res["untrained_volume_vs_trilinear_density_maxerr"] = (v.density.float() - ref.density.float()).abs().max().item()
    fr, aux = screen_inputs()
    sd = ScreenDetail(process_scale=0.5)
    for _ in range(3):
        o = sd.infer(fr, dict(aux, bbox=BBOX))
    res["untrained_screen_vs_input_maxerr"] = (o.int() - fr.int()).abs().max().item()
    ok = (res["sparse_vs_dense_warp_maxerr_fp32"] < 1e-3 and res["untrained_screen_vs_input_maxerr"] == 0
          and res["untrained_volume_vs_trilinear_density_maxerr"] < 2e-3
          and all(res[f"fused_vs_torch_{n}_density_maxerr"] < 4e-3 for n in ("sparse", "dense")))
    res["ok"] = ok
    del ut, uf, up, sd
    clean()
    return res


# ---------------------------------------------------------------------------- reference + report
def reference_2d():
    from vfx.model.detailnet import DetailNet
    m = DetailNet(lr_channels=64, n_blocks=8)
    hr = [m.hr1, m.hr2]
    return {"params": sum(p.numel() for p in m.parameters()), "macs_per_out": macs(m, None, hr, 4, 2)}


def capacity(kind, **kw):
    if kind == "vol":
        from vfx.model3d import VolumeNet, VolumeNetConfig
        from vfx.runtime3d.volume_upscaler import DEFAULT_ARCH
        m = VolumeNet(VolumeNetConfig(**{**DEFAULT_ARCH, **kw}))
        return macs(m, None, [m.hr1, m.hr2], m.cfg.scale, 3)
    from vfx.model3d import ScreenNet, ScreenNetConfig
    from vfx.runtime3d.screen_detail import DEFAULT_ARCH
    m = ScreenNet(ScreenNetConfig(**{**DEFAULT_ARCH, **kw}))
    return macs(m, None, [m.hr1, m.hr2], m.cfg.trunk_stride, 2)


def md_table(rows, cols, headers=None):
    headers = headers or cols
    s = "| " + " | ".join(headers) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n"
    return s


def report(res) -> str:
    ref = res["reference_2d"]
    out = [f"GPU before: {res['gpu_before']}", f"GPU after: {res['gpu_after']}",
           f"2D reference DetailNet 64x8: {ref['params']} params, {ref['macs_per_out']:.0f} MACs / HR pixel",
           f"selfcheck: {res.get('selfcheck', 'skipped')}",
           "Times: 10th-percentile ms per frame (medians in the JSON as *_med); MiB = allocator peak.", ""]
    out.append("### Recommended defaults\n" + md_table(res.get("recommended", []),
               ["config", "case", "params", "ms", "median_ms", "peak_mib"]))
    out.append("### Floor costs\n" + md_table(res["floor"], ["op", "ms_graph", "ms", "peak_mib", "note"],
                                                  ["op", "ms (graph)", "ms (eager)", "peak_mib", "note"]))
    occs = [k for k in ("1.00", "0.50", "0.25", "0.10", "0.05") if f"{k}_g" in res["volume"][0]]
    for r in res["volume"]:
        r["macs/vox"] = round(capacity("vol", width=r["width"], n_blocks=r["depth"]))
    cols = ["width", "depth", "params", "macs/vox", "dense_g"] + [f"{o}_g" for o in occs] + \
           ["dense_e"] + [f"{o}_e" for o in occs] + ["0.25_g_mib", "1.00_g_mib" if "1.00_g_mib" in res["volume"][0] else "0.10_g_mib"]
    out.append("### Route A: VolumeUpscaler 64^3 -> 256^3, ms/frame (g = CUDA graph, e = eager)\n"
               + md_table(res["volume"], cols))
    out.append("### Route A variants (CUDA graph)\n" + md_table(
        res["volume_variants"], ["variant", "params", "0.25_g", "0.10_g", "0.25_g_mib"]))
    for r in res["screen"]:
        r["macs/px"] = round(capacity("scr", width=r["width"], n_blocks=r["depth"]))
    m = ["full", "half", "quarter", "bbox", "bbox_half", "bbox_quarter"]
    out.append("### Route B: ScreenDetail 2560x1440, ms/frame (CUDA graph)\n" + md_table(
        res["screen"], ["width", "depth", "params", "macs/px"] + m + ["full_mib", "bbox_mib"]))
    out.append("### Route B variants\n" + md_table(res["screen_variants"], ["variant", "params"] + m))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).with_name("bench_results.json")))
    ap.add_argument("--only", default="check,floor,recommended,volume,variants,screen")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    only = set(a.only.split(","))
    res = {"gpu_before": gpu_state(), "torch": torch.__version__, "device": torch.cuda.get_device_name(),
           "quick": a.quick, "reference_2d": reference_2d(), "floor": [], "volume": [], "volume_variants": [],
           "screen": [], "screen_variants": []}
    state = plume_state()
    t0 = time.time()
    if "check" in only:
        res["selfcheck"] = selfcheck()
        print("selfcheck:", res["selfcheck"], flush=True)
    if "floor" in only:
        bench_floor(state, res["floor"])
    if "recommended" in only:
        res["recommended"] = bench_recommended(state)
        print("recommended:", res["recommended"], flush=True)
    if "volume" in only:
        res["volume"] = bench_volume(state, a.quick)
    if "variants" in only:
        res["volume_variants"] = bench_volume_variants(state, a.quick)
    if "screen" in only:
        res["screen"], res["screen_variants"] = bench_screen(a.quick)
    res["gpu_after"] = gpu_state()
    res["wall_s"] = round(time.time() - t0, 1)
    md = report(res) if res["volume"] and res["screen"] else json.dumps(
        {k: v for k, v in res.items() if k in ("selfcheck", "floor", "gpu_before", "gpu_after")}, indent=1)
    print(md)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"\nwrote {a.out} ({res['wall_s']} s)")


if __name__ == "__main__":
    main()
