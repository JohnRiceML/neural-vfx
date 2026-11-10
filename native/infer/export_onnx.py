"""Export route B's ScreenNet (vfx.model3d) to ONNX for the native DirectML runner, and verify it.

    native/.venv-export/Scripts/python native/infer/export_onnx.py            # export + verify (run from the worktree root)
    ... export_onnx.py --checkpoint PATH --out DIR --no-verify --static 960x1600,1472x2560

What is exported is exactly `ScreenNet.forward(rgb, aux)` of the checkpoint (use_prev is False in the shipped
checkpoint): processed-resolution rgb (1,3,h,w) in [0,1] + aux (1,4,h,w) alpha / depth m / motion processed px
-> delta rgb (1,3,h,w). Everything inside forward is in the graph (aux normalisation + clip, PixelUnshuffle,
trunk, PixelShuffle, HR head, grey_lowfreq, alpha_gate). The pooling / upsample / add / clamp / uint8 glue of
ScreenDetail._body is NOT in the graph: the native runner does it in its HLSL pre and post passes.

Files written to --out (default %LOCALAPPDATA%/neural-vfx/native_ref/onnx):
  screennet_fp16.onnx / screennet_fp32.onnx      dynamic H, W (symbolic dims "H", "W"; the runner fixes them per
                                                  crop bucket with ORT free-dimension overrides)
  screennet_fp16_<h>x<w>.onnx                     static-shape variants (--static, processed h x w) for timing
  screennet.json                                  checkpoint sha256 / run / config / opset / buckets / verification
Re-export after a new checkpoint is promoted: run this script again (one command; the json records the hash).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vfx.model3d import load_screen_net  # noqa: E402

NATIVE_REF = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "neural-vfx" / "native_ref"
DEFAULT_OUT = NATIVE_REF / "onnx"
OPSET = 17
CROP_FRACS = (0.25, 0.375, 0.5, 0.625, 0.75, 1.0)   # ScreenDetail defaults
PROCESS_SCALE = 0.5
QUANTUM = 64


def default_checkpoint() -> Path:
    from vfx.runtime3d.screen_detail import default_screen_checkpoint
    return default_screen_checkpoint()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def quantum_for(cfg, process_scale=PROCESS_SCALE, quantum=QUANTUM) -> int:
    """ScreenDetail.__init__'s crop quantum."""
    r = math.lcm(cfg.trunk_stride, getattr(cfg, "grey_lowfreq", 0) or 1)
    q = max(quantum, int(round(r / process_scale)))
    while (q * process_scale) % r:
        q *= 2
    return q


def sizes(L: int, q: int, fracs=CROP_FRACS) -> list[int]:
    """ScreenDetail.sizes."""
    return sorted({-(-max(int(L * f), 1) // q) * q for f in fracs})


class Net(nn.Module):
    """ScreenNet.forward(rgb, aux) without the optional prev input (the shipped net has use_prev False)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, rgb, aux):
        return self.m(rgb, aux, None)


def export(model, dtype, path: Path, hw=None):
    net = Net(model).to(dtype).eval()
    h, w = hw or (240, 320)
    rgb = torch.rand(1, 3, h, w, dtype=dtype)
    aux = torch.rand(1, 4, h, w, dtype=dtype)
    dyn = None if hw else {"rgb": {2: "H", 3: "W"}, "aux": {2: "H", 3: "W"}, "delta": {2: "H", 3: "W"}}
    torch.onnx.export(net, (rgb, aux), str(path), input_names=["rgb", "aux"], output_names=["delta"],
                      opset_version=OPSET, dynamic_axes=dyn, do_constant_folding=True, dynamo=False)
    import onnx
    m = onnx.load(str(path))
    # scalar config for the raw-DirectML runner (it builds the graph from the weights + these, see detail.cpp)
    cfg = model.cfg
    meta = {"nvfx.arch": "screennet-v1", "nvfx.res_scale": cfg.res_scale, "nvfx.depth_scale": cfg.depth_scale,
            "nvfx.motion_scale": cfg.motion_scale, "nvfx.input_clip": cfg.input_clip,
            "nvfx.alpha_gate": cfg.alpha_gate, "nvfx.grey_lowfreq": cfg.grey_lowfreq,
            "nvfx.trunk_stride": cfg.trunk_stride, "nvfx.use_aux": int(cfg.use_aux), "nvfx.use_prev": int(cfg.use_prev)}
    for k, v in meta.items():
        p = m.metadata_props.add()
        p.key, p.value = k, repr(v) if isinstance(v, float) else str(v)
    onnx.save(m, str(path))
    onnx.checker.check_model(m)
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    return ops


# ------------------------------------------------------------------ verification inputs
def pooled_inputs(frame_u8: torch.Tensor, aux: dict, y0, x0, ch, cw, vh, vw, scale=PROCESS_SCALE):
    """Processed-resolution net inputs of one crop, computed exactly as ScreenDetail._body (fp16 path)."""
    dev = frame_u8.device
    rgb_b = torch.zeros(ch, cw, 3, dtype=torch.uint8, device=dev)
    aux_b = torch.zeros(4, ch, cw, dtype=torch.half, device=dev)
    rgb_b[:vh, :vw] = frame_u8[y0:y0 + vh, x0:x0 + vw]
    aux_b[0, :vh, :vw] = aux["alpha"][y0:y0 + vh, x0:x0 + vw]
    aux_b[1, :vh, :vw] = aux["depth"][y0:y0 + vh, x0:x0 + vw]
    aux_b[2:4, :vh, :vw] = aux["motion"][:, y0:y0 + vh, x0:x0 + vw]
    rgb = rgb_b.permute(2, 0, 1).unsqueeze(0).half() * (1.0 / 255.0)
    k = int(round(1.0 / scale))
    x = F.avg_pool2d(rgb, k)
    a = aux_b.unsqueeze(0)
    al = a[:, 0:1].float().clamp(0.0, 1.0)
    w = F.avg_pool2d(al, k)
    rest = F.avg_pool2d(a[:, 1:4].float() * al, k) / w.clamp_min(1e-4)
    rest = rest * (w > 1e-4).float()
    xa = torch.cat([w, rest[:, 0:1], rest[:, 1:3] * scale], 1).half()
    return x, xa


def load_frame(path: Path, device="cuda"):
    meta = json.loads((path / "meta.json").read_text())
    H, W = meta["H"], meta["W"]
    fr = np.fromfile(path / "frame_rgb.u8", np.uint8).reshape(H, W, 3)
    al = np.fromfile(path / "alpha.f16", np.float16).reshape(H, W)
    de = np.fromfile(path / "depth.f16", np.float16).reshape(H, W)
    mo = np.fromfile(path / "motion.f16", np.float16).reshape(2, H, W)
    t = lambda a: torch.from_numpy(a.copy()).to(device)
    return t(fr), {"alpha": t(al), "depth": t(de), "motion": t(mo)}, meta


def synthetic_frame(H=1440, W=2560, device="cuda"):
    """Structured stand-in when no rendered frame exists: a soft plume with noise, matching aux."""
    g = torch.Generator(device="cpu").manual_seed(0)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
    blob = torch.exp(-(((xx - 0.5) / 0.18) ** 2 + ((yy - 0.55) / 0.3) ** 2))
    noise = F.interpolate(torch.rand(1, 1, H // 16, W // 16, generator=g), size=(H, W), mode="bilinear")[0, 0]
    al = (blob * (0.6 + 0.4 * noise)).clamp(0, 1)
    rgb = torch.stack([0.9 * al + 0.05, 0.5 * al ** 2 + 0.05, 0.2 * al ** 3 + 0.06], -1)
    fr = (rgb.clamp(0, 1) * 255 + 0.5).to(torch.uint8)
    de = torch.where(al > 1e-3, 1.5 + 0.3 * noise, torch.zeros_like(al))
    mo = torch.stack([3 * (noise - 0.5), -6 * blob]) * (al > 1e-3)
    return (fr.to(device), {"alpha": al.half().to(device), "depth": de.half().to(device),
                            "motion": mo.half().to(device)}, {"H": H, "W": W, "source": "synthetic"})


def region(H, W, bbox, q):
    """ScreenDetail._region."""
    x0, y0, x1, y1 = bbox

    def fit(a, b, L):
        a, b = max(a, 0), min(b, L)
        sz = sizes(L, q)
        n = next((v for v in sz if v >= b - a), sz[-1])
        if n >= L:
            return 0, n, L
        a = min(max(a - (n - (b - a)) // 2, 0), L - n)
        return a, n, n
    x0, cw, vw = fit(x0, x1, W)
    y0, ch, vh = fit(y0, y1, H)
    return y0, x0, ch, cw, vh, vw


def err255(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a.astype(np.float64) - b.astype(np.float64)) * 255.0
    return {"max": float(d.max()), "mean": float(d.mean()), "p99_9": float(np.percentile(d, 99.9))}


def verify(model_fp32, model_fp16_cuda, files: dict, frame_dir: Path | None, q: int, out: dict):
    import onnxruntime as ort
    if frame_dir and (frame_dir / "meta.json").is_file():
        frame, aux, meta = load_frame(frame_dir)
    else:
        frame, aux, meta = synthetic_frame()
    H, W = meta["H"], meta["W"]
    bbox = meta.get("bbox_third", [(W - 1536) // 2, (H - 832) // 2, (W - 1536) // 2 + 1536, (H - 832) // 2 + 832])
    y0, x0, ch, cw, vh, vw = region(H, W, bbox, q)
    x, xa = pooled_inputs(frame, aux, y0, x0, ch, cw, vh, vw)
    with torch.inference_mode():
        # fp32 reference on the CPU: true fp32 (CUDA fp32 convs would use TF32)
        ref32 = model_fp32.cpu().float()(x.float().cpu(), xa.float().cpu(), None).numpy()
        fmt = torch.channels_last
        ref16 = model_fp16_cuda(x.contiguous(memory_format=fmt), xa.contiguous(memory_format=fmt), None).float().cpu().numpy()
    xn, xan = x.cpu().numpy(), xa.cpu().numpy()
    res = {"input": meta.get("source", "rendered"), "crop": [ch, cw], "processed": list(x.shape[2:]),
           "delta_range_255": [float(ref32.min() * 255), float(ref32.max() * 255)],
           "torch_fp16_cuda_vs_fp32": err255(ref16, ref32)}
    so = ort.SessionOptions()
    so.log_severity_level = 3
    runs = [("ort_cpu_fp32", files["fp32"], ["CPUExecutionProvider"], np.float32),
            ("ort_dml_fp32", files["fp32"], ["DmlExecutionProvider"], np.float32),
            ("ort_dml_fp16", files["fp16"], ["DmlExecutionProvider"], np.float16)]
    for name, path, prov, dt in runs:
        s = ort.InferenceSession(str(path), so, providers=prov)
        y = s.run(None, {"rgb": xn.astype(dt), "aux": xan.astype(dt)})[0].astype(np.float32)
        res[name + "_vs_torch_fp32"] = err255(y, ref32)
        res[name + "_vs_torch_fp16_cuda"] = err255(y, ref16)
    out["verification"] = res
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--frame-dir", default=str(NATIVE_REF / "detail" / "campfire_1440p"),
                    help="rendered frame + aux from detail_bench.py make-frame (synthetic if missing)")
    ap.add_argument("--static", default="960x1600,1472x2560",
                    help="crop sizes (h x w, output px) to also export with static shapes; 'all' = every 1440p bucket")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args(argv)
    ck = Path(a.checkpoint) if a.checkpoint else default_checkpoint()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = load_screen_net(str(ck), "cpu").float().eval()
    cfg = model.cfg
    if cfg.use_prev:
        raise SystemExit("checkpoint uses use_prev: the native runner does not implement the previous-output input")
    raw = torch.load(str(ck), map_location="cpu", weights_only=False)
    q = quantum_for(cfg)
    meta = {"checkpoint": str(ck), "checkpoint_sha256": sha256(ck), "run": (raw.get("extra") or {}).get("run"),
            "promoted_run": None, "config": raw["config"], "opset": OPSET, "torch": torch.__version__,
            "process_scale": PROCESS_SCALE, "crop_fracs": list(CROP_FRACS), "quantum": q,
            "inputs": {"rgb": "(1,3,H,W) [0,1]", "aux": "(1,4,H,W) alpha, depth m, motion_x, motion_y processed px/SIM_DT"},
            "outputs": {"delta": "(1,3,H,W) delta rgb"},
            "buckets_1440p": {"h": sizes(1440, q), "w": sizes(2560, q)}}
    lj = ck.with_suffix(".json")
    if lj.is_file():
        try:
            meta["promoted_run"] = json.loads(lj.read_text()).get("run")
        except Exception:
            pass
    files = {}
    for tag, dt in (("fp16", torch.float16), ("fp32", torch.float32)):
        p = out_dir / f"screennet_{tag}.onnx"
        meta[f"ops_{tag}"] = export(model, dt, p)
        files[tag] = p
    stat = []
    if a.static == "all":
        stat = [(h, w) for h in sizes(1440, q) for w in sizes(2560, q)]
    elif a.static:
        stat = [tuple(int(v) for v in s.split("x")) for s in a.static.split(",")]
    meta["static"] = []
    for (h, w) in stat:
        ph, pw = int(h * PROCESS_SCALE), int(w * PROCESS_SCALE)
        p = out_dir / f"screennet_fp16_{ph}x{pw}.onnx"
        export(model, torch.float16, p, (ph, pw))
        meta["static"].append({"crop": [h, w], "processed": [ph, pw], "file": p.name})
    meta["files"] = {k: v.name for k, v in files.items()}
    print(f"exported in {time.time() - t0:.1f} s -> {out_dir}")
    print("ops fp16:", meta["ops_fp16"])
    if not a.no_verify:
        from vfx.runtime3d.screen_detail import ScreenDetail
        sd = ScreenDetail(str(ck), prewarm=False, cuda_graph=False)
        fd = Path(a.frame_dir) if a.frame_dir else None
        res = verify(load_screen_net(str(ck), "cpu"), sd.model, files, fd, q, meta)
        print(json.dumps(res, indent=1))
    (out_dir / "screennet.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out_dir / "screennet.json")


if __name__ == "__main__":
    main()
