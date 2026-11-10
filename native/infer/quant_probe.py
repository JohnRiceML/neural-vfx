"""Gate 2 item 4: is an 8-bit (QDQ INT8) ScreenNet realistic on DirectML? Measures it instead of guessing.

    native/.venv-export/Scripts/python native/infer/quant_probe.py

1. static QDQ quantization (onnxruntime.quantization, per-channel weights, activations uint8/int8 QDQ) of the fp32
   ONNX export, calibrated on pooled crops of the rendered campfire frame;
2. which ops the DML EP actually runs (session.disable_cpu_ep_fallback must hold);
3. error vs PyTorch fp32 (delta in 8-bit units) and DML EP timing next to the fp16 model, same sitting
   (Python IO binding on DML device memory, one sync per run).
Writes %LOCALAPPDATA%/neural-vfx/native_ref/onnx/screennet_int8_qdq.onnx and prints a json summary.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import export_onnx as ex  # noqa: E402

ONNX = ex.DEFAULT_OUT


def calib_inputs(n_crops=8):
    import torch
    frame, aux, meta = ex.load_frame(ex.NATIVE_REF / "detail" / "campfire_1440p")
    H, W = meta["H"], meta["W"]
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n_crops):
        x0 = int(rng.integers(0, W - 1600)); y0 = int(rng.integers(0, H - 960))
        x, xa = ex.pooled_inputs(frame, aux, y0, x0, 960, 1600, 960, 1600)
        out.append({"rgb": x.float().cpu().numpy(), "aux": xa.float().cpu().numpy()})
    bb = meta["bbox_third"]
    y0, x0, ch, cw, vh, vw = ex.region(H, W, bb, 64)
    x, xa = ex.pooled_inputs(frame, aux, y0, x0, ch, cw, vh, vw)
    return out, {"rgb": x.float().cpu().numpy(), "aux": xa.float().cpu().numpy()}


def time_dml(path, feeds, dtype, n=200):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    s = ort.InferenceSession(str(path), so, providers=["DmlExecutionProvider"])
    b = s.io_binding()
    vals = {k: ort.OrtValue.ortvalue_from_numpy(v.astype(dtype), "dml", 0) for k, v in feeds.items()}
    for k, v in vals.items():
        b.bind_ortvalue_input(k, v)
    y = ort.OrtValue.ortvalue_from_shape_and_type(list(feeds["rgb"].shape), dtype, "dml", 0)
    b.bind_ortvalue_output("delta", y)
    for _ in range(30):
        s.run_with_iobinding(b)
    b.synchronize_outputs()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        s.run_with_iobinding(b)
        b.synchronize_outputs()
        ts.append((time.perf_counter() - t) * 1e3)
    ts = np.array(ts)
    return {"med": float(np.median(ts)), "p10": float(np.percentile(ts, 10)), "min": float(ts.min())}, \
        y.numpy().astype(np.float32)


def main():
    import onnx
    import torch
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    calib, feed = calib_inputs()

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib)

        def get_next(self):
            return next(self.it, None)

    src = ONNX / "screennet_fp32.onnx"
    pre = ONNX / "screennet_fp32_static480x800.onnx"
    q = ONNX / "screennet_int8_qdq.onnx"
    # static shapes (480x800 processed = the 1600x960 bucket) so shape inference / fusion see a fixed graph
    m = onnx.load(str(src))
    for t in list(m.graph.input) + list(m.graph.output):
        d = t.type.tensor_type.shape.dim
        d[2].dim_value, d[3].dim_value = 480, 800
    onnx.save(m, str(pre))
    quant_pre_process(str(pre), str(pre), skip_symbolic_shape=True)
    qop = ONNX / "screennet_int8_qop.onnx"
    t0 = time.time()
    quantize_static(str(pre), str(q), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    op_types_to_quantize=["Conv"])
    calib[:] = calib_inputs()[0]
    # QOperator form (QLinearConv): what DML_OPERATOR_QUANTIZED_LINEAR_CONVOLUTION implements
    quantize_static(str(pre), str(qop), Reader(), quant_format=QuantFormat.QOperator, per_channel=False,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QUInt8,
                    op_types_to_quantize=["Conv"])
    res = {"quantize_s": round(time.time() - t0, 1), "gpu": subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()}
    from vfx.model3d import load_screen_net
    from vfx.runtime3d.screen_detail import default_screen_checkpoint
    net = load_screen_net(str(default_screen_checkpoint()), "cpu").float().eval()
    with torch.inference_mode():
        ref = net(torch.from_numpy(feed["rgb"]), torch.from_numpy(feed["aux"]), None).numpy()
    fp16 = ONNX / "screennet_fp16_480x800.onnx"
    if not fp16.is_file():
        fp16 = ONNX / "screennet_fp16.onnx"
    for name, path, dt in (("fp16", fp16, np.float16), ("int8_qdq", q, np.float32), ("int8_qlinearconv", qop, np.float32)):
        try:
            t, y = time_dml(path, feed, dt)
            res[name] = {"dml_ms": t, "err_vs_torch_fp32_255": ex.err255(y, ref)}
        except Exception as e:  # report, do not hide
            res[name] = {"error": str(e)[:400]}
    ops = {}
    for n in onnx.load(str(q)).graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    res["int8_ops"] = ops
    print(json.dumps(res, indent=1))
    (ONNX / "quant_probe.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
