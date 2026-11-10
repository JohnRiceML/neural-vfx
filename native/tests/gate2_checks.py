"""Gate 2 checks (native/README.md, "Gate 2 - native detail inference"): D1, D2, D3, D4, T3, negative controls.

OWNER: native test engineer. The native runner is a black box reached through nvfx_detail.h (+ the proposed texture
helpers, see nvfx_detail_ctypes.py). The PyTorch reference, vfx.runtime3d.ScreenDetail(latest.pt).infer(frame, aux),
runs in-process in the same session on the same inputs (native/tools/detail_frames.py), so nothing large is stored.

Thresholds (quoted from native/README.md, not to be changed here):
  D1  max abs <= 2 (8-bit units); pixels differing by more than 1: <= 0.01 %; processed region identical to Python's;
      pixels outside it identical to the input; >= 5 different crop buckets incl. 1536x832 -> 1600x960 and full frame,
      both scenes.
  D2  same inputs twice -> bit-identical output.
  D3  200 random bboxes (seed 0) in sequence after creation: no call > 10 ms CPU wall time; a sample of 10 meets D1.
  D4  nvfx_detail_run with >= 20 ms of GPU work already queued returns in < 1 ms.
  T3  GPU ms (timestamps, pre + net + post), 2560x1440, 1/3-screen bbox, FP16, median and p10 of >= 300 frames <= 2.5 ms;
      full frame reported; PyTorch in the same sitting alongside.
Interpretations made here (also listed in native/tests/README.md, for the lead):
  - "pixels differing by more than 1": a pixel counts when any of R, G, B differs by > 1; the denominator is the
    processed region (the stricter choice); the whole-frame fraction is reported too.
  - D1 compares RGB only (Python returns RGB). The header's "A is passed through" is checked and reported
    (`a_passthrough`) but is not part of D1's pass.
  - T3 is gated on `total_ms` (first -> last timestamp, >= pre + net + post); the sum is reported alongside.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

D1_MAX_ABS = 2
D1_FRAC_GT1 = 1e-4          # 0.01 %
D1_MIN_BUCKETS = 5
ONE_THIRD_WH = (1536, 832)
ONE_THIRD_CROP = (1600, 960)
D3_N, D3_SAMPLE, D3_CPU_MS, D3_SEED = 200, 10, 10.0, 0
D4_RETURN_MS, D4_QUEUED_MS, D4_TRIALS = 1.0, 20.0, 5
D4_MAX_STEPS, D4_MAX_ATTEMPTS = 2000, 4     # bounds on the queued solver load (precondition only, not a threshold)
T3_BUDGET_MS, T3_N, T3_WARM = 2.5, 300, 30
NC_SHIFT_PX = 64


# ================================================================== comparator (pure numpy, unit-tested without a GPU)
def region_mask(H: int, W: int, r: dict) -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[r["y0"]:r["y0"] + r["h"], r["x0"]:r["x0"] + r["w"]] = True
    return m


def _maxabs_rgb(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """per-pixel max over R, G, B of |a - b| (uint8 inputs, (H,W,>=3)) -> (H,W) int16, low memory."""
    d = None
    for c in range(3):
        x = np.abs(a[..., c].astype(np.int16) - b[..., c].astype(np.int16))
        d = x if d is None else np.maximum(d, x)
    return d


def compare_d1(nat_rgba: np.ndarray, py_rgb: np.ndarray, inp_rgba: np.ndarray, py_region: dict,
               nat_regions: list[dict]) -> dict:
    """D1 for one frame. py_region / nat_regions: dicts x0, y0, w, h, crop_w, crop_h. nat_regions holds every region the
    native side reports for this frame (nvfx_detail_region and the frame's timings); all must equal Python's.
    Pixel statistics use Python's region (the reference)."""
    H, W = py_rgb.shape[:2]
    keys = ("x0", "y0", "w", "h", "crop_w", "crop_h")
    region_ok = all(all(int(r[k]) == int(py_region[k]) for k in keys) for r in nat_regions) and len(nat_regions) > 0
    m = region_mask(H, W, py_region)
    d = _maxabs_rgb(nat_rgba, py_rgb)
    din = d[m]
    n_in = int(din.size)
    max_abs = int(din.max()) if n_in else 0
    n_gt1 = int((din > 1).sum())
    frac_gt1 = n_gt1 / max(n_in, 1)
    hist = {str(v): int((din == v).sum()) for v in (0, 1, 2)}
    hist[">2"] = int((din > 2).sum())
    out_diff = _maxabs_rgb(nat_rgba, inp_rgba)[~m]
    n_out_diff = int((out_diff > 0).sum())
    outside_ok = n_out_diff == 0
    a_pass = bool(np.array_equal(nat_rgba[..., 3], inp_rgba[..., 3]))
    py_out = _maxabs_rgb(py_rgb, inp_rgba)
    sig = py_out[m]
    pixel_ok = max_abs <= D1_MAX_ABS and frac_gt1 <= D1_FRAC_GT1
    ok = bool(region_ok and pixel_ok and outside_ok)
    return dict(
        pass_=ok, region_ok=bool(region_ok), pixel_ok=bool(pixel_ok), outside_ok=bool(outside_ok),
        py_region=py_region, nat_regions=nat_regions,
        max_abs=max_abs, n_gt1=n_gt1, frac_gt1=frac_gt1, region_px=n_in, hist=hist,
        frame_max_abs=int(d.max()), frame_frac_gt1=float((d > 1).sum() / d.size),
        outside_px_differing=n_out_diff, a_passthrough=a_pass,
        py_outside_equals_input=bool(int(py_out[~m].max(initial=0)) == 0),
        # how much the network itself changes the frame in the region: if this were <= 2 everywhere, D1 could not
        # tell a working runner from an identity copy
        signal_max_abs=int(sig.max()) if sig.size else 0,
        signal_frac_gt2=float((sig > 2).sum() / max(sig.size, 1)),
        signal_mean_abs=float(sig.mean()) if sig.size else 0.0)


def d1_line(tag: str, c: dict) -> str:
    return (f"{tag}: {'ok' if c['pass_'] else 'FAIL'} max {c['max_abs']}, >1 {c['n_gt1']} px "
            f"({100 * c['frac_gt1']:.4f} %), region {'=' if c['region_ok'] else '!='} "
            f"{c['py_region']['w']}x{c['py_region']['h']}@({c['py_region']['x0']},{c['py_region']['y0']}) "
            f"crop {c['py_region']['crop_w']}x{c['py_region']['crop_h']}, outside diff {c['outside_px_differing']} px, "
            f"A {'kept' if c['a_passthrough'] else 'CHANGED'}, net signal max {c['signal_max_abs']}")


def _stats(xs) -> dict:
    a = np.asarray(xs, np.float64)
    if a.size == 0:
        return dict(n=0)
    return dict(median=float(np.median(a)), p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)),
                min=float(a.min()), max=float(a.max()), n=int(a.size))


# ================================================================== bbox plans
def centred(cx: int, cy: int, w: int, h: int, W: int, H: int) -> tuple:
    w, h = min(w, W), min(h, H)
    x0 = min(max(cx - w // 2, 0), W - w)
    y0 = min(max(cy - h // 2, 0), H - h)
    return (x0, y0, x0 + w, y0 + h)


def effect_centre(inp) -> tuple[int, int]:
    H, W = inp.hw
    if inp.effect_bbox is None:
        return W // 2, H // 2
    x0, y0, x1, y1 = inp.effect_bbox
    return (x0 + x1) // 2, (y0 + y1) // 2


def one_third_bbox(inp) -> tuple:
    H, W = inp.hw
    cx, cy = effect_centre(inp)
    return centred(cx, cy, ONE_THIRD_WH[0], ONE_THIRD_WH[1], W, H)


def d1_plan(inp) -> list[tuple[str, tuple | None]]:
    """bboxes for D1: one per crop-bucket class around the effect, the 1/3-screen one, borders, full frame."""
    H, W = inp.hw
    cx, cy = effect_centre(inp)
    c = lambda w, h: centred(cx, cy, w, h, W, H)
    plan = [("effect_alpha_bbox", tuple(inp.effect_bbox) if inp.effect_bbox else None),
            ("600x340", c(600, 340)), ("900x500", c(900, 500)), ("1200x700", c(1200, 700)),
            ("one_third_1536x832", c(*ONE_THIRD_WH)), ("1800x1000", c(1800, 1000)),
            ("narrow_tall_500x1400", c(500, 1400)), ("wide_flat_2400x300", c(2400, 300)),
            ("corner_top_left_700x400", (0, 0, 700, 400)), ("corner_bottom_right_700x400", (W - 700, H - 400, W, H)),
            ("full_explicit", (0, 0, W, H)), ("full_frame_none", None)]
    return plan


def d3_bboxes(W: int, H: int, n: int = D3_N, seed: int = D3_SEED) -> list[tuple]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        w = int(rng.integers(32, W + 1))
        h = int(rng.integers(32, H + 1))
        x0 = int(rng.integers(0, W - w + 1))
        y0 = int(rng.integers(0, H - h + 1))
        out.append((x0, y0, x0 + w, y0 + h))
    return out


def d3_sample_indices(n: int = D3_N, k: int = D3_SAMPLE) -> list[int]:
    return sorted({int(round(v)) for v in np.linspace(0, n - 1, k)})


def garbage(H: int, W: int, seed: int) -> np.ndarray:
    return np.random.default_rng(1000 + seed).integers(0, 256, (H, W, 4), dtype=np.uint8)


# ================================================================== Python reference
class PyRef:
    """vfx.runtime3d.ScreenDetail(checkpoint).infer(frame, aux) at the frame size of the inputs, FP16 (as shipped)."""

    def __init__(self, checkpoint: str):
        import torch
        from vfx.runtime3d.screen_detail import ScreenDetail
        self.torch = torch
        self.checkpoint = str(checkpoint)
        self.sd = ScreenDetail(self.checkpoint)
        cfg = self.sd.model.cfg
        if getattr(cfg, "use_prev", False):
            raise RuntimeError("checkpoint uses use_prev: nvfx_detail has no previous-output input; D1 is undefined")
        self._gpu: dict = {}

    def _tensors(self, inp, zero_alpha=False):
        key = (id(inp), zero_alpha)
        if key not in self._gpu:
            t = self.torch
            fr = t.from_numpy(np.ascontiguousarray(inp.rgb)).cuda()
            al = t.from_numpy(np.zeros_like(inp.alpha) if zero_alpha else inp.alpha).cuda()
            aux = {"alpha": al, "depth": t.from_numpy(inp.depth).cuda(),
                   "motion": t.from_numpy(np.ascontiguousarray(inp.motion)).cuda()}
            self._gpu[key] = (fr, aux)
        return self._gpu[key]

    def drop(self):
        self._gpu.clear()
        self.torch.cuda.empty_cache()

    def region(self, H: int, W: int, bbox) -> dict:
        y0, x0, ch, cw, vh, vw = self.sd._region(H, W, bbox)
        return dict(x0=int(x0), y0=int(y0), w=int(vw), h=int(vh), crop_w=int(cw), crop_h=int(ch))

    def run(self, inp, bbox, zero_alpha=False) -> tuple[np.ndarray, dict]:
        fr, aux = self._tensors(inp, zero_alpha)
        a = dict(aux)
        if bbox is not None:
            a["bbox"] = tuple(int(v) for v in bbox)
        out = self.sd.infer(fr, a).clone()
        self.torch.cuda.synchronize()
        H, W = inp.hw
        r = self.region(H, W, bbox)
        assert self.sd._last_region == (r["y0"], r["x0"], r["h"], r["w"]), "ScreenDetail region bookkeeping changed"
        return out.cpu().numpy(), r

    def timing(self, inp, bbox, n=T3_N, warm=T3_WARM) -> dict:
        fr, aux = self._tensors(inp)
        a = dict(aux)
        if bbox is not None:
            a["bbox"] = tuple(int(v) for v in bbox)
        for _ in range(warm):
            self.sd.infer(fr, a)
        self.torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            self.sd.infer(fr, a)
            ts.append(self.sd.timings)       # synchronizes on the end event
        out = {k: _stats([t[k] for t in ts]) for k in ("net", "post", "total")}
        out["processed_fraction"] = ts[-1]["processed_fraction"]
        out["what"] = ("ScreenDetail.timings (CUDA events): net = copies in + graph, post = paste, total = whole infer "
                       "incl. the full-frame out.copy_(frame)")
        return out


def gpu_snapshot() -> dict:
    from gate1_checks import gpu_snapshot as g
    s = g()
    apps = s.get("compute_apps", [])
    me = str(os.getpid())
    s["other_compute_apps"] = [a for a in apps if not a.split(",")[0].strip() == me]
    return s


# ================================================================== native runs
def native_d1_frame(nd, inset, out_tex, bbox, seed: int) -> tuple[np.ndarray, list[dict], dict | None]:
    """One native frame for D1: pre-fill the output with garbage, run, download; regions from nvfx_detail_region and
    from the frame's own timings (frame_index checked)."""
    nd.fill_output(out_tex, garbage(nd.H, nd.W, seed))
    r_api = nd.region(bbox)
    nd.run(inset, out_tex, bbox)
    img = nd.download(out_tex)          # waits for the queue
    t = nd.timings()
    regs = [dict(r_api, src="nvfx_detail_region")]
    if t is not None and t["frame_index"] == nd.n_runs:
        regs.append(dict(t["region"], src="timings"))
    else:
        regs.append(dict(x0=-1, y0=-1, w=-1, h=-1, crop_w=-1, crop_h=-1,
                         src=f"timings missing/stale: {None if t is None else t['frame_index']} != {nd.n_runs}"))
    return img, regs, t


def _strip(c: dict) -> dict:
    c = dict(c)
    c["pass"] = c.pop("pass_")
    return c


def check_d1(nd, py, inp, inset_name: str, out_tex) -> dict:
    H, W = inp.hw
    inset = nd.inputs[inset_name]
    rows, buckets = [], set()
    full_crop = (py.region(H, W, None)["crop_w"], py.region(H, W, None)["crop_h"])
    for i, (label, bbox) in enumerate(d1_plan(inp)):
        py_img, py_r = py.run(inp, bbox)
        nat_img, regs, t = native_d1_frame(nd, inset, out_tex, bbox, seed=i)
        c = compare_d1(nat_img, py_img, inset.rgba, py_r, regs)
        c.update(label=label, bbox=None if bbox is None else list(bbox), gpu_total_ms=None if t is None else t["total"])
        rows.append(_strip(c))
        buckets.add((py_r["crop_w"], py_r["crop_h"]))
    has_third = ONE_THIRD_CROP in buckets
    has_full = full_crop in buckets
    third_row = next(r for r in rows if r["label"] == "one_third_1536x832")
    third_maps = (third_row["py_region"]["crop_w"], third_row["py_region"]["crop_h"]) == ONE_THIRD_CROP
    ok = all(r["pass"] for r in rows) and len(buckets) >= D1_MIN_BUCKETS and has_third and has_full and third_maps
    worst = max(rows, key=lambda r: (r["max_abs"], r["n_gt1"]))
    summ = (f"D1 {inp.scene}: {sum(r['pass'] for r in rows)}/{len(rows)} bboxes pass; {len(buckets)} buckets "
            f"(1600x960 {'yes' if has_third and third_maps else 'NO'}, full {full_crop[0]}x{full_crop[1]} "
            f"{'yes' if has_full else 'NO'}); worst: {d1_line(worst['label'], {**worst, 'pass_': worst['pass']})}")
    return {"pass": bool(ok), "summary": summ, "rows": rows, "buckets": sorted(buckets), "full_crop": full_crop,
            "inputs": inp.meta}


def check_d1_region_sweep(nd, py, H: int, W: int, n: int = 3000, seed: int = 7) -> dict:
    """'processed region identical to Python's' over many bboxes: nvfx_detail_region (pure CPU) vs ScreenDetail._region.
    Includes the D3 bboxes, small / degenerate-but-valid boxes, boxes partly outside the frame."""
    rng = np.random.default_rng(seed)
    boxes = list(d3_bboxes(W, H)) + [(0, 0, W, H), (0, 0, 1, 1), (W - 1, H - 1, W, H), (W // 2, H // 2, W // 2 + 1, H // 2 + 1)]
    for _ in range(n):
        x0, x1 = sorted(int(v) for v in rng.integers(-200, W + 200, 2))
        y0, y1 = sorted(int(v) for v in rng.integers(-200, H + 200, 2))
        if x1 > x0 and y1 > y0:
            boxes.append((x0, y0, x1, y1))
    bad = []
    keys = ("x0", "y0", "w", "h", "crop_w", "crop_h")
    for b in boxes:
        p, q = py.region(H, W, b), nd.region(b)
        if any(int(p[k]) != int(q[k]) for k in keys):
            bad.append(dict(bbox=list(b), python=p, native=q))
    ok = not bad
    return {"pass": ok, "summary": f"D1 region sweep: {len(boxes) - len(bad)}/{len(boxes)} bboxes give Python's region "
                                   f"and crop (nvfx_detail_region vs ScreenDetail._region)", "n": len(boxes),
            "mismatches": bad[:50], "n_mismatch": len(bad)}


def check_d1_negative(nd, py, inp, kind: str, out_tex) -> dict:
    """Negative controls: the native side gets a wrong input; D1 must FAIL.
    kind 'alpha0': the native alpha texture is all zeros (Python gets the real alpha); 'shift64': the native bbox is the
    1/3-screen bbox shifted right by 64 px (Python gets the unshifted one)."""
    H, W = inp.hw
    bbox = one_third_bbox(inp)
    py_img, py_r = py.run(inp, bbox)
    if kind == "alpha0":
        inset, nb = nd.inputs[f"{inp.scene}_alpha0"], bbox
    elif kind == "shift64":
        inset = nd.inputs[inp.scene]
        nb = (bbox[0] + NC_SHIFT_PX, bbox[1], bbox[2] + NC_SHIFT_PX, bbox[3])
        if nb[2] > W:
            nb = (bbox[0] - NC_SHIFT_PX, bbox[1], bbox[2] - NC_SHIFT_PX, bbox[3])
    else:
        raise KeyError(kind)
    nat_img, regs, _ = native_d1_frame(nd, inset, out_tex, nb, seed=99)
    c = _strip(compare_d1(nat_img, py_img, inset.rgba, py_r, regs))
    failed = not c["pass"]
    # for alpha0 the region is right, so the failure has to come from the pixel comparison itself
    meaningful = failed and (kind != "alpha0" or not c["pixel_ok"])
    summ = (f"negative control {kind} ({inp.scene}): D1 {'FAILS as expected' if failed else 'PASSES (control broken!)'}"
            f"; pixel part {'fails' if not c['pixel_ok'] else 'passes'}, region part "
            f"{'fails' if not c['region_ok'] else 'passes'}; " + d1_line("vs Python", {**c, 'pass_': c['pass']}))
    return {"pass": bool(meaningful), "expected": "FAIL", "d1": c, "summary": summ, "native_bbox": list(nb),
            "python_bbox": list(bbox)}


def check_py_control(py, py2, inp, kind: str) -> dict:
    """Harness controls that need no DLL: Python vs Python through the same comparator.
    'self' (a second, independent ScreenDetail instance) must PASS; 'alpha0' / 'shift64' must FAIL."""
    bbox = one_third_bbox(inp)
    ref, r = py.run(inp, bbox)
    if kind == "self":
        other, r2 = py2.run(inp, bbox)
    elif kind == "alpha0":
        other, r2 = py.run(inp, bbox, zero_alpha=True)
    elif kind == "shift64":
        sb = (bbox[0] + NC_SHIFT_PX, bbox[1], bbox[2] + NC_SHIFT_PX, bbox[3])
        other, r2 = py.run(inp, sb)
    else:
        raise KeyError(kind)
    H, W = inp.hw
    rgba = inp.rgba()
    other_rgba = np.concatenate([other, rgba[..., 3:]], -1)
    c = _strip(compare_d1(other_rgba, ref, rgba, r, [r2]))
    expected = "PASS" if kind == "self" else "FAIL"
    ok = c["pass"] if expected == "PASS" else (not c["pass"] and (kind != "alpha0" or not c["pixel_ok"]))
    return {"pass": bool(ok), "expected": expected, "d1": c,
            "summary": f"HC2 python-vs-python {kind} ({inp.scene}): D1 {'PASS' if c['pass'] else 'FAIL'} "
                       f"(expected {expected}); " + d1_line("", {**c, 'pass_': c['pass']})}


def check_d2(nd, inp, out_a, out_b) -> dict:
    inset = nd.inputs[inp.scene]
    rows = []
    for label, bbox in (("one_third_1536x832", one_third_bbox(inp)), ("full_frame_none", None)):
        nd.fill_output(out_a, garbage(nd.H, nd.W, 1))
        nd.run(inset, out_a, bbox)
        a = nd.download(out_a)
        nd.fill_output(out_b, garbage(nd.H, nd.W, 2))
        nd.run(inset, out_b, bbox)
        b = nd.download(out_b)
        nd.run(inset, out_a, centred(*effect_centre(inp), 600, 340, nd.W, nd.H))     # something else in between
        nd.run(inset, out_a, bbox)
        c = nd.download(out_a)
        ab = int((a != b).any(-1).sum())
        ac = int((a != c).any(-1).sum())
        rows.append(dict(label=label, bbox=None if bbox is None else list(bbox), px_differ_run1_run2=ab,
                         px_differ_run1_run3_after_other_bbox=ac, identical=ab == 0 and ac == 0))
    ok = all(r["identical"] for r in rows)
    return {"pass": ok, "rows": rows,
            "summary": f"D2 {inp.scene}: " + "; ".join(f"{r['label']}: run2 {r['px_differ_run1_run2']} px differ, run3 "
                                                         f"(after another bbox) {r['px_differ_run1_run3_after_other_bbox']}"
                                                         for r in rows)}


def check_d3(make_fresh, py, inputs: dict) -> dict:
    """make_fresh() -> a NEW NativeDetail; the 200 calls are the first calls after its creation (inputs uploaded
    first; uploads synchronize and are not timed)."""
    nd = make_fresh()
    try:
        scenes = list(inputs)
        for s in scenes:
            nd.upload_inputs(s, inputs[s])
        out = nd.make_output()
        H, W = nd.H, nd.W
        boxes = d3_bboxes(W, H)
        sample = set(d3_sample_indices())
        wall, rows, buckets = [], [], {}
        for i, b in enumerate(boxes):
            scene = scenes[i % len(scenes)]
            ms = nd.run(nd.inputs[scene], out, b)
            wall.append(ms)
            if i in sample:
                img = nd.download(out)
                t = nd.timings()
                regs = [dict(nd.region(b), src="nvfx_detail_region")]
                if t is not None and t["frame_index"] == nd.n_runs:
                    regs.append(dict(t["region"], src="timings"))
                py_img, py_r = py.run(inputs[scene], b)
                c = _strip(compare_d1(img, py_img, nd.inputs[scene].rgba, py_r, regs))
                c.update(index=i, scene=scene, bbox=list(b), api_cpu_ms=None if t is None else t["cpu"])
                rows.append(c)
                key = f"{py_r['crop_w']}x{py_r['crop_h']}"
                buckets[key] = buckets.get(key, 0) + 1
        nd.wait_idle()
        create_s, vram = nd.create_s, nd.vram_bytes()
    finally:
        nd.close()
    worst = int(np.argmax(wall))
    ok_cpu = max(wall) <= D3_CPU_MS
    ok_d1 = all(r["pass"] for r in rows) and len(rows) == len(sample)
    summ = (f"D3: max CPU wall {max(wall):.3f} ms (call #{worst}), median {float(np.median(wall)):.3f} ms, "
            f"{sum(w > D3_CPU_MS for w in wall)} calls > {D3_CPU_MS} ms; sample D1 {sum(r['pass'] for r in rows)}/"
            f"{len(rows)} pass (worst max abs {max(r['max_abs'] for r in rows)}, worst >1 frac "
            f"{100 * max(r['frac_gt1'] for r in rows):.4f} %); create {create_s:.2f} s, vram {vram / 2**20:.0f} MiB")
    return {"pass": bool(ok_cpu and ok_d1), "summary": summ, "cpu_wall_ms": wall, "cpu_wall": _stats(wall),
            "first5_ms": wall[:5], "sample": rows, "sample_buckets": buckets, "create_s": create_s, "vram_bytes": vram}


def check_d4(nd, inp) -> dict:
    from nvfx_detail_ctypes import QueueLoad
    inset = nd.inputs[inp.scene]
    out = nd.make_output()
    q = QueueLoad(nd)
    try:
        nd.run(inset, out, None)
        q.submit(4)
        nd.wait_idle()
        t0 = time.perf_counter()
        q.submit(8)
        nd.wait_idle()
        per_step = (time.perf_counter() - t0) * 1e3 / 8
        # Target >= 2x the requirement (incl. the detail run's own budget). The per-step cost drifts with load on a
        # shared card, so each call's queued-ahead work is measured after the fact; a call that fell short is re-run
        # with the step count rescaled from what was actually observed (bounded), and n carries forward.
        target = 2.0 * (D4_QUEUED_MS + T3_BUDGET_MS)
        n = int(min(D4_MAX_STEPS, max(10, math.ceil(target / max(per_step, 1e-3)))))
        n_first = n
        trials, attempts = [], []
        for k in range(D4_TRIALS):
            for label, bbox in (("one_third_1536x832", one_third_bbox(inp)), ("full_frame_none", None)):
                for a in range(D4_MAX_ATTEMPTS):
                    nd.wait_idle()
                    ts = time.perf_counter()
                    q.submit(n)
                    t0 = time.perf_counter()
                    ms = nd.run(inset, out, bbox)
                    t1 = time.perf_counter()
                    nd.wait_idle()
                    t2 = time.perf_counter()
                    t = nd.timings()
                    det = t["total"] if t is not None and t["frame_index"] == nd.n_runs else float("nan")
                    backlog = (t2 - t1) * 1e3
                    queued_ahead = backlog - (det if math.isfinite(det) else T3_BUDGET_MS)
                    row = dict(trial=k, label=label, attempt=a, steps_queued=n, submit_ms=(t0 - ts) * 1e3,
                               run_return_ms=ms, backlog_after_return_ms=backlog, detail_gpu_ms=det,
                               queued_ahead_ms=queued_ahead, valid=queued_ahead >= D4_QUEUED_MS)
                    attempts.append(row)
                    if queued_ahead < target:   # re-aim the next call from the observed queue depth
                        # at most 2x per re-aim: an oversized queue can fill the driver's launch queue and block
                        n = int(min(D4_MAX_STEPS, 2 * n, max(n + 1, math.ceil(n * target / max(queued_ahead, 1.0)))))
                    if row["valid"]:
                        break
                trials.append(row)
    finally:
        q.close()
    valid = all(t["valid"] for t in trials)
    mx = max(t["run_return_ms"] for t in attempts)      # every call counts toward the < 1 ms bound
    ok = valid and mx < D4_RETURN_MS
    retries = len(attempts) - len(trials)
    summ = (f"D4: nvfx_detail_run returned in max {mx:.3f} ms (median "
            f"{float(np.median([t['run_return_ms'] for t in attempts])):.3f}) over {len(attempts)} calls; "
            f"{sum(t['valid'] for t in trials)}/{len(trials)} trials with >= {D4_QUEUED_MS:.0f} ms queued ahead "
            f"(min {min(t['queued_ahead_ms'] for t in trials):.1f}, median "
            f"{float(np.median([t['queued_ahead_ms'] for t in trials])):.1f} ms; target {target:.0f} ms; "
            f"{min(t['steps_queued'] for t in trials)}-{max(t['steps_queued'] for t in trials)} solver steps, "
            f"calibrated ~{per_step:.2f} ms/step -> {n_first}; {retries} re-aimed call(s))"
            f"{'' if valid else '; PRECONDITION NOT MET (< 20 ms queued)'}")
    return {"pass": bool(ok), "summary": summ, "per_step_ms": per_step, "target_queued_ms": target,
            "trials": trials, "attempts": attempts, "precondition_met": valid}


def native_timing(nd, inset, out, bbox, n=T3_N, warm=T3_WARM) -> dict:
    for _ in range(warm):
        nd.run(inset, out, bbox)
    nd.wait_idle()
    base = nd.n_runs
    samples: dict = {}
    method = "non-blocking get_timings after each run"

    def grab():
        t = nd.timings()
        if t is not None and t["frame_index"] > base:
            samples.setdefault(t["frame_index"], t)

    k = 0
    while len(samples) < n and k < 20 * n:
        nd.run(inset, out, bbox)
        k += 1
        grab()
    if len(samples) < n:
        method = "wait_idle + get_timings after each run (non-blocking reads gave too few distinct frames)"
        while len(samples) < n and k < 40 * n:
            nd.run(inset, out, bbox)
            k += 1
            nd.wait_idle()
            grab()
    nd.wait_idle()
    grab()
    ts = list(samples.values())
    out_ = {s: _stats([t[s] for t in ts]) for s in ("pre", "net", "post", "sum", "total", "cpu")}
    regs = {(t["region"]["crop_w"], t["region"]["crop_h"]) for t in ts}
    out_.update(method=method, runs=k, crops=sorted(regs))
    return out_


def check_t3(nd, py, inp) -> dict:
    inset = nd.inputs[inp.scene]
    out = nd.make_output()
    bb = one_third_bbox(inp)
    g0 = gpu_snapshot()
    nat = {"one_third": native_timing(nd, inset, out, bb), "full_frame": native_timing(nd, inset, out, None)}
    pyt = {"one_third": py.timing(inp, bb), "full_frame": py.timing(inp, None)}
    g1 = gpu_snapshot()
    t = nat["one_third"]
    ok = t["total"].get("n", 0) >= T3_N and t["total"]["median"] <= T3_BUDGET_MS and t["crops"] == [ONE_THIRD_CROP]
    idle = (not g0.get("other_compute_apps")) and (not g1.get("other_compute_apps")) and (g0.get("util_pct", 100) <= 5)
    f = lambda d: f"{d['median']:.3f} (p10 {d['p10']:.3f})"
    summ = (f"T3 {inp.scene} 1/3-screen: native total {f(t['total'])} ms, pre+net+post {f(t['sum'])}, n {t['total']['n']} "
            f"<= {T3_BUDGET_MS} ms; full frame {f(nat['full_frame']['total'])}; PyTorch total {f(pyt['one_third']['total'])}"
            f" / full {f(pyt['full_frame']['total'])}; GPU util {g0.get('util_pct')}/{g1.get('util_pct')} %, mem "
            f"{g0.get('mem_used_mib')}/{g1.get('mem_used_mib')} MiB; "
            f"{'idle GPU (figure of record)' if idle else 'GPU SHARED (not the figure of record)'}")
    return {"pass": bool(ok), "summary": summ, "native": nat, "python": pyt, "gpu_before": g0, "gpu_after": g1,
            "idle_gpu": bool(idle), "bbox": list(bb)}
