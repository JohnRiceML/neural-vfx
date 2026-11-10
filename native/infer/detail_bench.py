"""Route B native-vs-PyTorch driver (parity + timing). Run from the worktree root with the export venv:

    native/.venv-export/Scripts/python native/infer/detail_bench.py make-frame   # render campfire 2560x1440 + aux
    native/.venv-export/Scripts/python native/infer/detail_bench.py parity       # PyTorch vs nvfx_detail_bench.exe
    native/.venv-export/Scripts/python native/infer/detail_bench.py time         # timing table, both sides
    native/.venv-export/Scripts/python native/infer/detail_bench.py all

Inputs live outside the repo in %LOCALAPPDATA%/neural-vfx/native_ref/detail/campfire_1440p:
  frame_rgb.u8 (H,W,3) uint8, alpha.f16 (H,W), depth.f16 (H,W), motion.f16 (2,H,W) planar, meta.json.
The native side is native/build/bin/nvfx_detail_bench.exe (built by native/build.ps1); it reads the same files,
uploads them once into D3D12 textures (RGBA8 / R16F / R16F / RG16F), runs nvfx_detail_* and writes the RGBA8
output back for comparison. Nothing per frame touches the CPU in the timed loop.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NATIVE_REF = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "neural-vfx" / "native_ref"
FRAME_DIR = NATIVE_REF / "detail" / "campfire_1440p"
ONNX_DIR = NATIVE_REF / "onnx"
BIN = ROOT / "native" / "build" / "bin"
H, W = 1440, 2560


def gpu_note() -> str:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"nvidia-smi failed: {e}"


# ------------------------------------------------------------------ inputs
def make_frame(out: Path = FRAME_DIR, steps: int = 150):
    import torch
    from vfx.app3d.backends import load_backends
    from vfx.app3d.presets import PRESETS
    from vfx.core.fields import SIM_DT
    from vfx.core.fields3d import CH3, Forces3D, Volume, rasterize_obstacles_3d
    from vfx.render3d import VolumeRenderer
    from vfx.render3d.bench import camera_at

    pre = PRESETS[1]   # campfire
    b = load_backends(warmup=False)
    s = b.Solver(64, b.runtime_config(), device="cuda")
    st = s.reset(pre.obstacles)
    for _ in range(steps):
        st = s.step(st, Forces3D(emitters=list(pre.emitters), wind=pre.wind), SIM_DT)
    vol = Volume(st.data[CH3["density"]].clone(), st.data[CH3["temperature"]].clone())
    solid = rasterize_obstacles_3d(pre.obstacles, 128)
    cam, cam0 = camera_at(1.3), camera_at(1.32)
    r = VolumeRenderer((H, W), march_scale=0.5)
    frame, aux = r.draw(vol, cam, solid, return_aux=True, velocity=st.velocity.clone(), prev_camera=cam0)
    torch.cuda.synchronize()
    out.mkdir(parents=True, exist_ok=True)
    fr = frame.cpu().numpy()
    al = aux["alpha"].cpu().numpy()
    fr.tofile(out / "frame_rgb.u8")
    al.astype(np.float16).tofile(out / "alpha.f16")
    aux["depth"].cpu().numpy().astype(np.float16).tofile(out / "depth.f16")
    aux["motion"].cpu().numpy().astype(np.float16).tofile(out / "motion.f16")
    ys, xs = np.nonzero(al > 0.02)
    eff = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else [0, 0, W, H]
    cx, cy = (eff[0] + eff[2]) // 2, (eff[1] + eff[3]) // 2
    bx0 = int(np.clip(cx - 768, 0, W - 1536))
    by0 = int(np.clip(cy - 416, 0, H - 832))
    meta = {"H": H, "W": W, "source": f"campfire preset, {steps} steps runtime_config, camera_at(1.3), prev camera_at(1.32)",
            "bbox_effect": eff, "bbox_third": [bx0, by0, bx0 + 1536, by0 + 832],
            "alpha_coverage": float((al > 0.02).mean())}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    try:
        from PIL import Image
        Image.fromarray(fr).save(out / "frame.png")
    except Exception:
        pass


# ------------------------------------------------------------------ helpers
def load_inputs(d: Path = FRAME_DIR):
    import torch
    meta = json.loads((d / "meta.json").read_text())
    h, w = meta["H"], meta["W"]
    t = lambda a: torch.from_numpy(a.copy()).cuda()
    fr = t(np.fromfile(d / "frame_rgb.u8", np.uint8).reshape(h, w, 3))
    aux = {"alpha": t(np.fromfile(d / "alpha.f16", np.float16).reshape(h, w)),
           "depth": t(np.fromfile(d / "depth.f16", np.float16).reshape(h, w)),
           "motion": t(np.fromfile(d / "motion.f16", np.float16).reshape(2, h, w))}
    return fr, aux, meta


def bboxes(meta) -> dict:
    """name -> bbox (x0,y0,x1,y1). 'third' = the 1536x832 (35 % of 1440p) effect bbox -> 1600x960 bucket."""
    t = meta["bbox_third"]
    cx, cy = (t[0] + t[2]) // 2, (t[1] + t[3]) // 2
    small = [cx - 300, cy - 180, cx + 300, cy + 180]          # 600x360 -> 640x384 (smallest bucket)
    mid = [cx - 600, cy - 330, cx + 600, cy + 330]            # 1200x660 -> 1280x768
    wide = [cx - 900, cy - 500, cx + 900, cy + 500]           # 1800x1000 -> 1920x1088
    return {"third": t, "small": small, "mid": mid, "wide": wide, "effect": meta["bbox_effect"], "full": [0, 0, 0, 0]}


FLAG_NAMES = {0: "DirectML (raw graph, shipping)", 1: "ORT DML EP", 2: "DirectML half-precision compute",
              4: "DirectML no metacommands", 8: "DirectML NHWC intermediates"}


def spec(fl) -> tuple[list, str]:
    """native config spec: an int flag set (nvfx_detail.h), optionally suffixed 'r' = hosted nvfx_detail_record path."""
    s = str(fl)
    rec = s.endswith("r")
    v = int(s.rstrip("r"))
    return (["--flags", str(v)] + (["--record", "1"] if rec else []),
            FLAG_NAMES.get(v, f"flags {v}") + (", nvfx_detail_record into a host list" if rec else ""))


def native(mode: str, boxes: list, out: Path | None = None, model: Path | None = None, extra=()) -> list[dict]:
    exe = BIN / "nvfx_detail_bench.exe"
    cmd = [str(exe), "--frame-dir", str(FRAME_DIR), "--model", str(model or ONNX_DIR / "screennet_fp16.onnx"),
           "--mode", mode, "--bbox", ";".join(",".join(str(int(v)) for v in b) for b in boxes)]
    if out:
        cmd += ["--out", out.as_posix()]   # forward slashes: the exe echoes it inside JSON
    cmd += list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{exe} failed ({r.returncode}): {r.stderr}\n{r.stdout}")
    return [json.loads(line) for line in r.stdout.splitlines() if line.startswith("{")]


def screen_detail():
    from vfx.runtime3d.screen_detail import ScreenDetail
    return ScreenDetail()    # the promoted checkpoint, FP16, CUDA graphs: exactly what the sandbox runs


# ------------------------------------------------------------------ parity
PARITY_BOXES = ("third", "small", "mid", "wide", "full")


def parity(flag_sets=(0, 1, 2, 8), names=PARITY_BOXES) -> list[dict]:
    """Output difference native vs PyTorch ScreenDetail.infer, 8-bit units, inside the processed region
    (outside it both are the input frame; checked too). pct_gt1 / pct_ne = % of region pixels with any channel
    differing by > 1 / by >= 1."""
    fr, aux, meta = load_inputs()
    bb = bboxes(meta)
    sd = screen_detail()
    inp = fr.cpu().numpy().astype(np.int16)
    refs = {}
    for n in names:
        refs[n] = (sd.infer(fr, {**aux, "bbox": None if n == "full" else tuple(bb[n])}).cpu().numpy().astype(np.int16),
                   sd._last_region)
    rows = []
    for fl in flag_sets:
        out = ROOT / "native" / "build" / "detail_out" / f"flags{fl}"   # ~75 MB per set: git-ignored build dir
        out.mkdir(parents=True, exist_ok=True)
        extra, name = spec(fl)
        ev = native("parity", [bb[n] for n in names], out, extra=extra)
        for n, e in zip(names, [x for x in ev if x["event"] == "parity"]):
            ref, (y0, x0, vh, vw) = refs[n]
            nat = np.fromfile(e["file"], np.uint8).reshape(H, W, 4)
            assert (nat[..., 3] == 255).all(), "alpha channel not passed through"
            nat = nat[..., :3].astype(np.int16)
            d = np.abs(nat - ref)
            reg = d[y0:y0 + vh, x0:x0 + vw]
            outside = d.copy()
            outside[y0:y0 + vh, x0:x0 + vw] = 0
            changed = np.abs(ref - inp)[y0:y0 + vh, x0:x0 + vw]
            row = {"flags": fl, "backend": name, "bbox": n, "bbox_px": bb[n], "crop": e["crop"],
                   "region_native": e["region"], "region_python": [x0, y0, vw, vh],
                   "region_match": e["region"] == [x0, y0, vw, vh], "max": int(reg.max()), "mean": float(reg.mean()),
                   "pct_gt1": float((reg > 1).any(-1).mean() * 100), "pct_ne": float((reg > 0).any(-1).mean() * 100),
                   "n_gt1": int((reg > 1).any(-1).sum()), "n_px": int(vh * vw),
                   "outside_max": int(outside.max()),
                   "net_change_max": int(changed.max()), "net_change_mean": float(changed.mean())}
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


# ------------------------------------------------------------------ timing
def time_torch(names=("third", "full"), frames=300, warmup=60) -> list[dict]:
    """PyTorch ScreenDetail.infer, same inputs: GPU ms from its own CUDA events (start -> end of infer, i.e. copies in,
    graph, paste) and CPU wall ms of the infer() call (no sync inside)."""
    import torch
    fr, aux, meta = load_inputs()
    bb = bboxes(meta)
    sd = screen_detail()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    rows = []
    for n in names:
        a = {**aux, "bbox": None if n == "full" else tuple(bb[n])}
        for _ in range(warmup):
            sd.infer(fr, a)
        torch.cuda.synchronize()
        g, c = [], []
        for _ in range(frames):
            t0 = time.perf_counter()
            sd.infer(fr, a)
            c.append((time.perf_counter() - t0) * 1e3)
            g.append(sd.timings["total"])       # synchronizes on the end event
        g, c = np.array(g), np.array(c)
        row = {"side": "pytorch", "bbox": n, "gpu_total": stats(g), "cpu_call": stats(c),
               "peak_alloc_mb": (torch.cuda.max_memory_allocated() - base) / 2 ** 20}
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def stats(v) -> dict:
    v = np.asarray(v, np.float64)
    return {"med": float(np.median(v)), "p10": float(np.percentile(v, 10)), "min": float(v.min()),
            "p90": float(np.percentile(v, 90)), "mean": float(v.mean())}


def time_native(flag_sets=(0, 1), names=("third", "full"), frames=300, warmup=60) -> list[dict]:
    fr, aux, meta = load_inputs()
    bb = bboxes(meta)
    rows = []
    for fl in flag_sets:
        extra, name = spec(fl)
        ev = native("time", [bb[n] for n in names], extra=extra + ["--frames", str(frames), "--warmup", str(warmup)])
        create = next(e for e in ev if e["event"] == "create")
        for n, e in zip(names, [x for x in ev if x["event"] == "time"]):
            row = {"side": "native", "flags": fl, "backend": name, **e, "bbox": n, "bbox_px": e["bbox"],
                   "create_s": create["create_s"], "vram_mb_detail": create["vram_mb_detail"],
                   "owned_mb": create.get("owned_mb")}
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("backend", "bbox", "total", "net", "cpu_run", "vram_mb_detail")}), flush=True)
    return rows


def run_time(native_flags=(0, 1), rounds=2):
    """Alternates native and PyTorch runs (contention from other GPU users hits both alike)."""
    res = {"gpu_before": gpu_note(), "rounds": []}
    for r in range(rounds):
        rd = {"gpu_start": gpu_note()}
        rd["native"] = time_native(native_flags)
        rd["gpu_mid"] = gpu_note()
        rd["pytorch"] = time_torch()
        rd["gpu_end"] = gpu_note()
        res["rounds"].append(rd)
        print("nvidia-smi (util, mem):", rd["gpu_start"], "|", rd["gpu_mid"], "|", rd["gpu_end"], flush=True)
    return res


def table(res: dict) -> str:
    """Markdown tables from results.json (README)."""
    out = []
    if "time" in res:
        out.append("| round | side / backend | bbox | pre | net | post | **total GPU ms** med / p10 / min | CPU ms per call med / p10 | VRAM MB |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        f = lambda s: f"{s['med']:.3f} / {s['p10']:.3f} / {s['min']:.3f}"
        for i, rd in enumerate(res["time"]["rounds"]):
            for r in rd["native"]:
                bn = r["bbox"] if isinstance(r["bbox"], str) else ("full" if r["bbox"][2] <= r["bbox"][0] else "third")
                out.append(f"| {i + 1} | native: {r['backend']} | {bn} |{r['pre']['med']:.3f} | {r['net']['med']:.3f} | "
                           f"{r['post']['med']:.3f} | **{f(r['total'])}** | {r['cpu_run']['med']:.3f} / {r['cpu_run']['p10']:.3f} | "
                           f"{r['vram_mb_detail']:.0f} |")
            for r in rd["pytorch"]:
                out.append(f"| {i + 1} | PyTorch ScreenDetail (CUDA graphs) | {r['bbox']} | | | | **{f(r['gpu_total'])}** | "
                           f"{r['cpu_call']['med']:.3f} / {r['cpu_call']['p10']:.3f} | {r['peak_alloc_mb']:.0f} (alloc peak) |")
            out.append(f"| {i + 1} | nvidia-smi util, mem: start {rd['gpu_start']} / mid {rd['gpu_mid']} / end {rd['gpu_end']} | | | | | | | |")
    if "parity" in res:
        out.append("")
        out.append("| backend | bbox -> crop | max | mean | px > 1 LSB | px != | outside region max | net change max / mean |")
        out.append("|---|---|---|---|---|---|---|---|")
        for r in res["parity"]:
            out.append(f"| {r['backend']} | {r['bbox']} -> {r['crop'][0]}x{r['crop'][1]} | {r['max']} | {r['mean']:.4f} | "
                       f"{r.get('n_gt1', '?')} / {r.get('n_px', '?')} | {r['pct_ne']:.2f} % | {r['outside_max']} | "
                       f"{r['net_change_max']} / {r['net_change_mean']:.2f} |")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make-frame", "parity", "time", "all", "table"])
    ap.add_argument("--flags", default=None, help="comma-separated native flag sets (see nvfx_detail.h)")
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    fl = tuple(a.flags.split(",")) if a.flags else None
    out = NATIVE_REF / "detail" / "results.json"
    res = json.loads(out.read_text()) if out.is_file() else {}
    if a.cmd == "make-frame":
        make_frame()
    if a.cmd in ("parity", "all"):
        res["parity"] = parity(*( (fl,) if fl else ()))
    if a.cmd in ("time", "all"):
        res["time"] = run_time(*((fl,) if fl else ()), rounds=a.rounds)
    if a.cmd == "table":
        print(table(res))
    elif a.cmd != "make-frame":
        out.write_text(json.dumps(res, indent=1))
        print("wrote", out)
