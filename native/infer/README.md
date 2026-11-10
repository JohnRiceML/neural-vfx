# native/infer — route B ScreenDetail on D3D12 + DirectML (owner: native inference engineer)

Gate 2: the screen-space detail network (`vfx/runtime3d/screen_detail.py` `ScreenDetail`, `vfx/model3d/screennet.py`
`ScreenNet`) running natively in C++ on a shipping inference runtime, on the GPU, sharing D3D12 resources with the
renderer, at 2560x1440 on a ~1/3-screen effect within 2.5 ms, matching PyTorch within FP16 tolerance.

**Result: met.** The shipping path (raw DirectML graph + HLSL pre/post) runs the 1/3-screen bbox in
**0.72 ms GPU** and the whole 1440p frame in **1.25 ms** (median; p10 within 0.01 ms; GPU shared with a training job,
see "Timing"), matches PyTorch to **max 1 LSB** (2 LSB on 5 of 2.1 M pixels in one bucket), never touches the CPU with
frame data, never waits for the GPU, and uses **~270 MB** of VRAM for all 36 crop buckets.

## Files
| file | what |
|---|---|
| `export_onnx.py` | checkpoint -> ONNX (fp16 + fp32, dynamic H/W, optional static per bucket) + `screennet.json` metadata; verifies with ORT CPU / DML against PyTorch fp32 (CPU) and fp16 (CUDA) |
| `nvfx_detail.h` | **proposed C API** (to be merged into `include/nvfx.h` by the lead) |
| `detail.cpp` | the runner: ONNX weight reader, DirectML graph per bucket, ORT DML EP backend, frame recording, timings |
| `shaders/detail.hlsl` | HLSL passes: `pre_dml_s2/s1`, `pre_s2/s1` (ORT inputs), `lowfreq`, `fin`, `post_s2/s1` |
| `bench.cpp` | `nvfx_detail_bench.exe`: parity dumps, timing, the forced-sync test (C API + plain D3D12 for test I/O) |
| `detail_bench.py` | driver: renders the input frame, runs PyTorch and the exe on the same inputs, parity + timing tables |
| `quant_probe.py` | INT8 (QDQ and QLinearConv) on the DML EP: error and speed vs fp16 |
| `CMakeLists.txt` | pinned NuGet fetch (SHA-256 checked), dxc shader embedding, the bench exe |

## How to run (from the worktree root)
```
powershell -File native/build.ps1                                     # fetches _deps on first configure, builds nvfx.dll + exe
native/.venv-export/Scripts/python native/infer/detail_bench.py make-frame   # campfire 2560x1440 + aux -> native_ref/detail
native/.venv-export/Scripts/python native/infer/export_onnx.py              # checkpoint -> native_ref/onnx (+ verify)
native/.venv-export/Scripts/python native/infer/detail_bench.py parity --flags 0,0r,1
native/.venv-export/Scripts/python native/infer/detail_bench.py time --flags 0,0r,1 --rounds 2
native/.venv-export/Scripts/python native/infer/quant_probe.py
```
Re-export after a new checkpoint is promoted = run `export_onnx.py` again (one command). `screennet.json` records the
checkpoint path, its SHA-256, the training run name (`extra.run` in the .pt, and `run` of `latest.json`), the config,
opset 17, torch version and the 1440p buckets. Everything large lives in `%LOCALAPPDATA%/neural-vfx/native_ref/`
(`onnx/`, `detail/`), nothing in the repo. Flag sets for the drivers: see `nvfx_detail.h` (`0` = shipping DirectML,
`1` = ORT DML EP, `2` half-precision DML compute, `4` no metacommands, `8` NHWC intermediates; suffix `r` = drive the
hosted `nvfx_detail_record` path through a host command list instead of `nvfx_detail_run`).

## Dependencies (fetched by CMake into `native/build/_deps`, pinned + SHA-256 checked, copied next to nvfx.dll)
| package | version | .nupkg | used |
|---|---|---|---|
| `Microsoft.AI.DirectML` | 1.15.4 | 193 MiB download (mostly PDBs / Xbox / ARM; only `DirectML.dll` 17.7 MB + headers are extracted) | always |
| `Microsoft.ML.OnnxRuntime.DirectML` | 1.24.4 | 11.9 MiB (`onnxruntime.dll` 16.5 MB + headers) | ORT backend only |

`_deps` totals 240 MB (the two cached .nupkg + extracted files). The fetch is verified by SHA-256, not by TLS: the
CMake on this machine (WinLibs/MinGW 4.4.3) has no CA bundle and fails `file(DOWNLOAD)` with "SSL peer certificate
... not OK", so `CMakeLists.txt` falls back to Windows' `curl.exe` (Schannel); a wrong file fails the configure. Checked
with a clean `_deps` (download + extract + build, exit 0).

nvfx.dll does not import either DLL: `detail.cpp` loads them **by full path from nvfx.dll's directory** on the first
`nvfx_detail_create` (System32 has its own older `DirectML.dll` and `onnxruntime.dll`, which a plain `LoadLibrary`
could pick up), so the Gate 1 solver keeps working without them. The DirectML backend needs only `DirectML.dll`.

## Design
Per frame (`nvfx_detail_run` = record + submit; `nvfx_detail_record` = record into the host's list):
1. **Region** (CPU, no GPU work): bbox -> crop bucket exactly as `ScreenDetail._region` / `sizes` (6 sizes per axis,
   64 px quantum = `ScreenDetail`'s rule from `trunk_stride` and `grey_lowfreq`, centred and clamped; whole axis padded
   to the bucket with zeros when it does not fit, e.g. 1440 rows -> 1472).
2. **pre** (HLSL, one thread per 2 processed pixels): crop, uint8 -> fp16, 2x2 average pool exactly as `_body`
   (rgb mean; alpha mean; depth and motion alpha-weighted with the 1e-4 clamp/mask; motion x 0.5), `.half()`, then the
   first ops of `ScreenNet.forward` — aux x (1, depth_scale, motion_scale, motion_scale) in fp16, clip +-8, concat and
   **PixelUnshuffle(4)** — written straight into the network's NCHW fp16 input buffer `X0 (1,112,h/4,w/4)`; also the
   pooled rgb (HR-head input) and the pooled alpha (alpha gate).
3. **network** (one compiled DirectML graph per bucket, static shapes): head conv+ReLU, 8 residual blocks
   (conv+ReLU, conv, x0.1, add), tail conv+ReLU (ReLU commutes with the shuffle), DepthToSpace(CRD, 4) = PixelShuffle,
   join with rgb, hr1 3x3 + ReLU, hr2 3x3 -> raw delta. Bound directly to the D3D12 buffers, recorded with
   `IDMLCommandRecorder::RecordDispatch` into the same command list.
4. **lowfreq + fin** (HLSL): the rest of `forward` — `grey_lowfreq` (avg_pool 8, bilinear x8, `d - lo + mean_c(lo)`)
   and `alpha_gate` (clamp to +-0.5*alpha) — with the same fp16 rounding at every step as the PyTorch graph.
5. **post** (HLSL, full frame): bilinear upsample of delta (`F.interpolate(align_corners=False)` formula), add to the
   fp16 rgb, clamp, `*255` and `+0.5` each rounded to fp16, truncation to uint8; pixels outside the region are copied
   from the input (A passes through). The output texture is fully written every frame.

Every fp16 rounding is done in integer arithmetic (round-to-nearest-even) so the result does not depend on the
driver's float->half conversion. UNORM->float input is re-quantized with `round(c*255)` for the same reason.

Everything is created in `nvfx_detail_create`: 36 compiled graphs at 1440p (~1.1-2.4 s total), each with its own small
persistent resource (weights in DML's layout); **one temporary resource shared by all buckets** (only one bucket runs at
a time on the queue) and one set of network buffers sized for the largest bucket. A bbox change only selects another
pre-built graph: no compile, no allocation, no hitch. Create also records and executes every bucket's graph once:
DirectML does 0.2-2.6 ms of lazy CPU work inside the FIRST `RecordDispatch` of each compiled graph (found by the
independent Gate 2 suite, D4; attributed with `NVFX_DETAIL_TRACE=1`, which prints the per-call CPU split
slot_wait / prepare / begin / record to stderr). After the warm-up, first use of each of the 36 buckets costs
0.18 ms median / 0.38 ms max of CPU vs 0.09 / 0.16 ms for the second call (`nvfx_detail_bench --mode firstuse`).
`nvfx_detail_run` blocks only when 8 of its frames are still in flight (back-pressure, a GPU wait by design): a caller
submitting hundreds of frames back to back on a GPU slowed by another process will see ~10 ms waits. Descriptors for the caller's textures are written per frame into an
8-slot ring; timings use timestamp queries in the frame's own list, resolved to a readback ring with a
`WriteBufferImmediate` marker, read without blocking (also for host-recorded frames, which the runner cannot fence).

### ONNX Runtime vs raw DirectML — decision: **raw DirectML** (ORT kept as `NVFX_DETAIL_BACKEND_ORT`)
Both were built and measured (tables below). The ORT path is the textbook one: `OrtDmlApi::SessionOptionsAppendExecutionProvider_DML1`
with the context's `IDMLDevice` + `ID3D12CommandQueue`, `CreateGPUAllocationFromD3DResource` on our D3D12 buffers, IO
binding, `disable_synchronize_execution_providers=1`, one session per bucket (free-dimension overrides `H`, `W`, so DML
graph fusion sees static shapes). It works, is exact, and **does not force a sync** (measured, see "Forced syncs").
It loses on three counts:
- **VRAM: 2.2 GB** for the 36 sessions (60-135 MB each: every session owns its DML allocator and graph temporaries;
  ORT has no way to share them across sessions on the DML EP). Raw DML: **270 MB** total, because the temporary
  resource is shared. One dynamic-shape session instead (no overrides) costs 320 MB but runs **8-14 ms** with **8 ms of
  CPU per call** (no graph fusion, per-op dispatch): unusable.
- **Hosting**: ORT executes its own command lists on the queue, so it cannot record into an engine's command list
  (RDG pass). Raw DML records into any list: `nvfx_detail_record` exists only for this backend.
- **Speed**: 0.62 vs 0.73 ms for the network at the 1/3 bucket (ORT runs the unshuffle as Reshape/Transpose copies and
  the output constraints as separate DML element-wise ops; the raw path moves them into the HLSL passes).
The ONNX file stays the interchange format: the raw backend reads the initializers with a ~100-line protobuf reader and
builds the graph from the weight shapes + scalar metadata (`nvfx.*` keys) that `export_onnx.py` writes. The cost: the
graph builder is specific to the ScreenNet family (width / depth / stride / HR head are read from the weights; a
different layer type needs code). Unreal NNE's RDG DirectML runtime makes the same trade.

### Forced syncs
None in either backend. `nvfx_detail_bench --mode sync` queues ~23-35 ms of GPU copies on the queue and then calls
`nvfx_detail_run`: the call returns in ~0.1 ms (ORT) — long before the queued work finishes — 6/6 times. In the raw
DML path there is nothing to wait on by construction (record + `ExecuteCommandLists`). The only CPU wait is the 8-frame
ring (descriptor / timestamp slots), i.e. back-pressure when the CPU is 8 frames ahead.

## Parity (`detail_bench.py parity`)
Inputs: one real frame — the campfire preset (`vfx.app3d.presets` #1), 150 steps of the 64³ `runtime_config` solver,
rendered by `vfx.render3d` at 2560x1440 (`march_scale` 0.5, camera_at(1.3), previous camera camera_at(1.32), aux from
the sim velocity), saved raw under `native_ref/detail/campfire_1440p`. Both sides get identical uint8 colour and fp16
alpha / depth / motion. Reference = `ScreenDetail()` (promoted checkpoint, fp16, CUDA graphs, channels_last) —
exactly what the sandbox runs. Native = `nvfx_detail_bench --mode parity` (each frame is preceded, back to back, by a
frame with a different bbox, so a queue-ordering bug would show). Differences in 8-bit units inside the processed region;
outside it both sides must equal the input frame (they do: max 0). "net change" = |PyTorch output - input| for scale.

| backend | bbox -> crop | max | mean | px > 1 LSB | px != | outside region max | net change max / mean |
|---|---|---|---|---|---|---|---|
| DirectML (raw graph, shipping) | third -> 1600x960 | 1 | 0.0103 | 0 / 1536000 | 2.90 % | 0 | 53 / 1.86 |
| DirectML (raw graph, shipping) | small -> 640x384 | 1 | 0.0200 | 0 / 245760 | 5.68 % | 0 | 40 / 3.68 |
| DirectML (raw graph, shipping) | mid -> 1280x768 | 1 | 0.0116 | 0 / 983040 | 3.30 % | 0 | 53 / 2.27 |
| DirectML (raw graph, shipping) | wide -> 1920x1088 | 2 | 0.0089 | 5 / 2088960 | 2.51 % | 0 | 53 / 1.61 |
| DirectML (raw graph, shipping) | full -> 2560x1472 | 1 | 0.0064 | 0 / 3686400 | 1.80 % | 0 | 48 / 1.20 |
| DirectML (raw graph, shipping), nvfx_detail_record into a host list | third -> 1600x960 | 1 | 0.0103 | 0 / 1536000 | 2.90 % | 0 | 53 / 1.86 |
| DirectML (raw graph, shipping), nvfx_detail_record into a host list | small -> 640x384 | 1 | 0.0200 | 0 / 245760 | 5.68 % | 0 | 40 / 3.68 |
| DirectML (raw graph, shipping), nvfx_detail_record into a host list | mid -> 1280x768 | 1 | 0.0116 | 0 / 983040 | 3.30 % | 0 | 53 / 2.27 |
| DirectML (raw graph, shipping), nvfx_detail_record into a host list | wide -> 1920x1088 | 2 | 0.0089 | 5 / 2088960 | 2.51 % | 0 | 53 / 1.61 |
| DirectML (raw graph, shipping), nvfx_detail_record into a host list | full -> 2560x1472 | 1 | 0.0064 | 0 / 3686400 | 1.80 % | 0 | 48 / 1.20 |
| ORT DML EP | third -> 1600x960 | 1 | 0.0105 | 0 / 1536000 | 2.94 % | 0 | 53 / 1.86 |
| ORT DML EP | small -> 640x384 | 1 | 0.0200 | 0 / 245760 | 5.68 % | 0 | 40 / 3.68 |
| ORT DML EP | mid -> 1280x768 | 1 | 0.0117 | 0 / 983040 | 3.33 % | 0 | 53 / 2.27 |
| ORT DML EP | wide -> 1920x1088 | 2 | 0.0091 | 1 / 2088960 | 2.54 % | 0 | 53 / 1.61 |
| ORT DML EP | full -> 2560x1472 | 1 | 0.0065 | 0 / 3686400 | 1.83 % | 0 | 48 / 1.20 |

Half-precision-compute and NHWC variants: bit-identical to the shipping rows (results.json).

Region / bucket selection matches Python on every case (`region_match`). fp16 tolerance is met: max 1 LSB, except 5
pixels at 2 LSB out of 2.1 M in the 1920x1088 bucket. The residual 1-LSB pixels (2-6 %) come from the convolutions
(cuDNN vs DirectML metacommand accumulation order); the HLSL pre/post maths reproduce PyTorch's fp16 rounding steps
exactly. `DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION` and NHWC intermediates give bit-identical outputs
to the default (the NVIDIA metacommand ignores both).

ONNX-level check (`export_onnx.py`, network only, 1600x960 bucket of the same frame, delta in 8-bit units, against
PyTorch fp32 on the CPU / fp16 on CUDA): ORT CPU fp32 max 0.028 / mean 0.0016; ORT DML fp32 max 0.028; ORT DML fp16
max 0.87 / mean 0.011 (vs PyTorch fp16-CUDA max 0.90); PyTorch fp16-CUDA vs fp32 max 0.17. Numbers in `screennet.json`.

## Timing (`detail_bench.py time --flags 0,0r,1 --rounds 2`)
2560x1440 frame, "third" = the 1536x832 effect bbox -> 1600x960 crop (processed 800x480, trunk 200x120), "full" =
whole frame -> 2560x1472 crop. FP16. 300 frames after 60 warm-up frames per row. Native GPU ms = D3D12 timestamps
on the queue the work runs on, one frame at a time (run, wait, read) so the split is exact; CPU ms = wall time inside
`nvfx_detail_run` / `nvfx_detail_record` (descriptor writes + recording + submit, incl. DirectML's / ORT's CPU work).
PyTorch = `ScreenDetail.infer` with its own CUDA events (start -> end of `infer`: copies in, graph, paste) and CPU wall
of the `infer` call, measured in the same sitting, alternating with the native runs.

**The GPU was shared**: a training job held it at 96-99 % utilization and 12.8-15.0 GB for the whole sitting
(nvidia-smi in the table). Medians are inflated whenever the other process's time slices land inside our interval
(p90 up to 10-200 ms); **min and p10 are the best estimate of our own cost**, the median is what we got. An idle-GPU
re-run is the lead's.

﻿| round | side / backend | bbox | pre | net | post | **total GPU ms** med / p10 / min | CPU ms per call med / p10 | VRAM MB |
|---|---|---|---|---|---|---|---|---|
| 1 | native: DirectML (raw graph, shipping) | third |0.048 | 0.621 | 0.055 | **0.724 / 0.722 / 0.718** | 0.198 / 0.158 | 269 |
| 1 | native: DirectML (raw graph, shipping) | full |0.077 | 1.092 | 0.090 | **1.260 / 1.256 / 1.251** | 0.170 / 0.155 | 269 |
| 1 | native: DirectML (raw graph, shipping), nvfx_detail_record into a host list | third |0.047 | 162.304 | 0.058 | **162.409 / 123.721 / 52.691** | 0.278 / 0.221 | 269 |
| 1 | native: DirectML (raw graph, shipping), nvfx_detail_record into a host list | full |0.076 | 1.084 | 0.089 | **1.250 / 1.245 / 1.240** | 0.106 / 0.101 | 269 |
| 1 | native: ORT DML EP | third |0.042 | 0.726 | 0.041 | **0.808 / 0.804 / 0.795** | 0.142 / 0.125 | 2223 |
| 1 | native: ORT DML EP | full |0.082 | 1.305 | 0.067 | **1.461 / 1.441 / 1.433** | 0.162 / 0.122 | 2223 |
| 1 | PyTorch ScreenDetail (CUDA graphs) | third | | | | **1.300 / 1.294 / 1.284** | 0.182 / 0.098 | 278 (alloc peak) |
| 1 | PyTorch ScreenDetail (CUDA graphs) | full | | | | **10.680 / 3.207 / 2.930** | 0.218 / 0.197 | 278 (alloc peak) |
| 1 | nvidia-smi util, mem: start 96 %, 14258 MiB / mid 99 %, 13569 MiB / end 99 %, 15000 MiB | | | | | | | |
| 2 | native: DirectML (raw graph, shipping) | third |0.047 | 0.624 | 0.055 | **0.732 / 0.718 / 0.714** | 0.212 / 0.167 | 269 |
| 2 | native: DirectML (raw graph, shipping) | full |0.080 | 1.084 | 0.089 | **1.254 / 1.248 / 1.242** | 0.182 / 0.162 | 269 |
| 2 | native: DirectML (raw graph, shipping), nvfx_detail_record into a host list | third |0.047 | 12.643 | 0.057 | **12.798 / 0.719 / 0.714** | 0.192 / 0.105 | 269 |
| 2 | native: DirectML (raw graph, shipping), nvfx_detail_record into a host list | full |0.080 | 1.085 | 0.089 | **1.255 / 1.249 / 1.241** | 0.110 / 0.103 | 269 |
| 2 | native: ORT DML EP | third |0.043 | 0.723 | 0.041 | **0.807 / 0.802 / 0.785** | 0.158 / 0.135 | 2223 |
| 2 | native: ORT DML EP | full |0.068 | 1.298 | 0.067 | **1.434 / 1.427 / 1.422** | 0.146 / 0.130 | 2223 |
| 2 | PyTorch ScreenDetail (CUDA graphs) | third | | | | **1.301 / 1.294 / 1.282** | 0.182 / 0.097 | 278 (alloc peak) |
| 2 | PyTorch ScreenDetail (CUDA graphs) | full | | | | **7.735 / 3.201 / 2.954** | 0.207 / 0.148 | 278 (alloc peak) |
| 2 | nvidia-smi util, mem: start 99 %, 15000 MiB / mid 99 %, 12804 MiB / end 98 %, 14210 MiB | | | | | | | |

Anomaly, unresolved: the hosted-record row "third" in round 1 measured median 162 ms / min 53 ms, while the same
config gave p10 0.719 in round 2 and every neighbouring row (same process, same bucket via `run`) was normal; the
GPU was at 99 % with a training job. It was GPU-side, not a block in the API: `nvfx_detail_record` has
no fence wait, no allocator acquire and no `ExecuteCommandLists` (CPU per call in that same round: median 0.28 ms),
and a later 100-frame check (GPU 99 % busy) gave network median 0.626 / p10 0.624 ms. An idle-GPU timing re-run is
still owed.

Summary (p10, GPU ms): **1/3 screen 0.72 native vs 1.29 PyTorch; full frame 1.25 native vs 3.20 PyTorch**.
The 2.5 ms budget is met at the 1/3 bbox by 3.5x, and the whole 1440p frame fits too. Peak VRAM: native 269 MB
(DXGI `QueryVideoMemoryInfo` process usage delta over the create call; `nvfx_detail_vram_bytes` = 181 MB owned buffers
+ DML persistent/temporary resources, the rest is DirectML's own device objects); ORT backend 2223 MB; PyTorch 278 MB
allocator peak (excludes the CUDA context).

Where the native time goes (1/3 bucket): pre 0.05 ms, network 0.62 ms, post (lowfreq + fin + upsample/add/round +
full-frame copy) 0.055 ms. DirectML does not expose per-operator GPU times inside a compiled graph; by FLOPs the trunk
is 96 % of the network (35 GFLOP at 200x120 vs 1.5 GFLOP for the 3x3 HR head at 800x480), and 36.5 GFLOP in 0.62 ms
= 59 TFLOPS, about half of the 5080's fp16-with-fp32-accumulate tensor peak. Experiments (same bucket, same sitting):
| variant | network ms |
|---|---|
| shipping (metacommands on, NCHW, fp32 accumulation allowed) | 0.62 |
| `ALLOW_HALF_PRECISION_COMPUTATION` | 0.62 (bit-identical output) |
| NHWC-strided intermediates | 0.62 (bit-identical output) |
| metacommands disabled (DirectML's own HLSL convs) | 2.18-2.35 |
| ORT DML EP, static per bucket (unshuffle as Reshape/Transpose, output constraints as DML ops) | 0.72 |
| ORT DML EP, one dynamic-shape session | 6.4-9.6 (plus 8 ms CPU per call) |
| ORT DML EP, graph fusion off | 1.24-1.30 |
| Python ORT DML fp16, dynamic file + free-dim override vs static-shape file (wall incl. sync) | 0.91 vs 0.92 min |
Static shapes matter (or overrides, which are equivalent); layout and precision flags do not; metacommands are
everything. Nothing here needed fixing for the gate.

## Quantization (`quant_probe.py`): INT8 is not worth it on DirectML for this net
Static post-training quantization with onnxruntime (calibration: 8 crops of the rendered frame, Conv only), 1600x960
bucket, DML EP, Python IO binding (wall incl. one sync, same sitting, GPU 66-96 % busy):
| model | DML ms min / p10 | delta error vs PyTorch fp32, 8-bit units max / mean |
|---|---|---|
| fp16 | 0.94-1.13 / 1.07-1.27 | 0.87 / 0.011 |
| INT8 QDQ (per-channel s8 weights, u8 activations) | 2.71-2.86 / 3.24 | 29.0 / 0.55 |
| INT8 QOperator (`QLinearConv`) | 2.81 / 3.79 | 28.3 / 0.30 |
Both INT8 forms run **~2.5-3x slower** than fp16: the DML EP has no INT8 convolution metacommand on this GPU/driver,
so QDQ runs as dequantize + fp16/fp32 conv and `QLinearConv` as DirectML's generic HLSL quantized convolution. And naive
PTQ is far outside the parity tolerance (up to 29 LSB in delta; the residual trunk accumulates the activation
quantization error). A useful 8-bit path would need quantization-aware training **and** a runtime with INT8/FP8 tensor
core kernels (TensorRT; possibly a future DirectML metacommand). FP16 stays the gate path.

## TensorRT (not installed; not attempted)
Where it would likely differ: (1) speed — TensorRT fuses conv+bias+ReLU+residual-add and picks tensor-core tactics
per shape, and offers INT8/FP8 with calibration or QAT; the trunk is ~50 % of fp16 peak here, so a 1.3-2x gain on the
network is plausible, more with FP8. (2) integration — TensorRT runs on CUDA streams, not on a D3D12 queue: the frame
would cross APIs every frame (`cudaImportExternalMemory` on the D3D12 buffers, `cudaImportExternalSemaphore` on a D3D12
fence, wait/signal around the network), i.e. zero-copy but with a cross-API fence pair per frame and no recording into
the engine's command list. (3) one engine per crop bucket or optimization profiles, like the per-bucket DML graphs.

## The C API (`nvfx_detail.h`, proposed for `nvfx.h`)
```c
NvfxResult nvfx_detail_create(NvfxContext*, const NvfxDetailDesc* {model_path, width, height, process_scale, flags}, NvfxDetail**);
void       nvfx_detail_destroy(NvfxDetail*);
NvfxResult nvfx_detail_run(NvfxDetail*, const NvfxDetailFrame* {color, alpha, depth, motion, output, bbox[4]});   // submit
NvfxResult nvfx_detail_record(NvfxDetail*, void* d3d12_command_list, const NvfxDetailFrame*);                    // hosted
NvfxResult nvfx_detail_get_timings(NvfxDetail*, NvfxDetailTimings* {pre, net, post, total ms, cpu_ms, region, crop, frame_index});
NvfxResult nvfx_detail_region(NvfxDetail*, const int32_t bbox[4], int32_t out[6]);
uint64_t   nvfx_detail_vram_bytes(NvfxDetail*);
```
Changes from the brief and why: `record` added (the raw-DML backend can record into the host's list, which is how an
Unreal RDG pass would call it; `run` is the stand-alone path); the output is a separate texture that is fully written
(region + copy of the rest), not an in-place edit, because the network reads the unmodified frame around the region;
aux textures may be NULL (= zeros, like missing aux keys in Python); textures are passed per frame (descriptors are
rewritten each frame, no registration step), so the renderer may ping-pong. Required states are listed in the header
and nothing is transitioned by the runner.

## What does not work / is missing
- **Validation**: the D3D12 debug layer is not installed on this machine (see `core/README.md`), so resource states,
  barriers and DirectML bindings were checked only by the parity tests, never by the debug / DML debug layers.
- `nvfx_detail_record`: the runner cannot see when a host list executes, so the 8-frame descriptor / timestamp ring is
  a contract (at most 8 recorded frames outstanding), not enforced. Detail frames must execute on one queue (shared DML
  temporary resource).
- Previous-output input (`use_prev`) and `prev_warp` are not implemented (the shipped net has `use_prev=False`;
  `nvfx_detail_create` rejects such a model).
- The raw-DML graph builder covers the ScreenNet family (any width / depth / stride / HR head size / k); another
  architecture needs builder code (the ORT backend runs any ONNX, at 2.2 GB VRAM for 36 buckets).
- Only `process_scale` 0.5 and 1.0 (Python allows any 1/k); only RGBA8 colour in/out.
- Per-operator GPU times inside the DML graph are not available (DirectML compiled graphs are opaque); the split
  above is pre / network / post.
- Create time 1.1-2.4 s at 1440p (36 graph compiles + initializations, measured on the shared GPU); a frame-size change
  means a new `NvfxDetail`.

## Measurements still owed (GPU paused by the lead mid-session)
- `pytest native/tests` (Gate 1) after the infer build changes: started, then stopped unfinished when the lead paused
  GPU work. My changes add sources/targets only (nothing in core/ or sim/ changed), but the run is owed.
- An idle-GPU timing run (`detail_bench.py time`).

## Contract notes for the lead
- `vfx/runtime3d/README.md` "How route B works" still describes the old net (1x1 HR head, previous output warped);
  the shipped checkpoint is a 3x3 HR head with `use_prev=False`, `grey_lowfreq` 8, `alpha_gate` 0.5. The code
  (`screen_detail.py`, `screennet.py`) is right; only that README paragraph is stale (vfx/ is read-only here).
- `ScreenDetail` pads the 1440-row axis to a 1472-row crop for the whole-frame bucket (zero rows the net sees); the native
  runner reproduces it, so the bottom rows match, but it costs ~2 % extra work on full-frame runs. A 1440-friendly
  quantum would remove it — that is a Python-side decision.
- `nvfx_context_create` makes a DIRECT queue; the runner works on DIRECT or COMPUTE queues (DirectML records into
  either). The README's "renderer reads solver textures directly" model extends naturally: the renderer's colour + aux
  textures go straight into `nvfx_detail_run/record`.
