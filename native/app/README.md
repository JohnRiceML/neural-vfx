# native/app: `nvfx_demo.exe`, the Gate 3 stand-alone demo (owner: native app engineer)

This is a small Win32 + D3D12 application with no Python and no third-party code (Windows SDK only). Every frame it
runs the whole pipeline on the GPU: the native 64³ smoke/fire solver, the half-resolution ray-march with
alpha/depth/motion, the screen-space neural detail (DirectML), and present. It hosts the three libraries the way an
engine would: **the app creates its own `ID3D12Device`, direct queue and flip-model swap chain**, hands the device and
queue to `nvfx_context_create_from_device`, and records all library work into **its own command list** through the
`record_*` entry points.

> **Status (2026-11-09): written and compiled (no warnings), NOT RUN.** A GPU pause was in effect, so nothing here
> has created a D3D12 device yet. The renderer and the detail runner have not run on the GPU in their current state
> either. See "Not yet run" below.

## Files
| file | what |
|---|---|
| `main.cpp` | window, D3D12 setup, input, scene presets, frame loop, HUD text, bench / screenshot |
| `present.hlsl` | the app's only GPU pass: a full-screen triangle that copies the frame into the back buffer and draws the HUD |
| `hud_font.h` | a 5x7 bitmap font (ASCII 32..95, original glyph art), packed into the HUD constant buffer |
| `CMakeLists.txt` | `nvfx_demo` target; dxc compiles `present.hlsl` (vs_6_0 / ps_6_0) into embedded headers |

## Build and run
```
powershell -File native/build.ps1 -BuildDir build-app       # -> native/build-app/bin/nvfx_demo.exe
native\build-app\bin\nvfx_demo.exe                          # interactive, 2560x1440, neural mode if the model exists
native\build-app\bin\nvfx_demo.exe --bench 20 --mode neural --preset 1 --out D:\nvfx_demo\bench_neural.json
native\build-app\bin\nvfx_demo.exe --bench 20 --mode raw    --preset 1 --out D:\nvfx_demo\bench_raw.json
native\build-app\bin\nvfx_demo.exe --frames 300 --screenshot D:\nvfx_demo\shot.bmp      # or .ppm
```
`nvfx_demo.exe` sits in `<builddir>/bin` next to `nvfx.dll`. `native/infer` copies `DirectML.dll` (and the ONNX
Runtime DLLs) there at configure time, and nvfx.dll loads `DirectML.dll` from its own directory.

Options:
- Window:
  - `--width N --height N`: default 2560x1440 client area. If the window frame does not fit the work area but the size
    fits the monitor, the window is borderless. Otherwise it is shrunk to fit, keeping the aspect ratio.
  - `--fullscreen`: a borderless window covering the primary monitor. There is no exclusive mode.
- Scene and mode:
  - `--preset 1-4`, `--mode raw|neural`.
  - `--model PATH`: the default is `%LOCALAPPDATA%\neural-vfx\native_ref\onnx\screennet_fp16.onnx`, written by
    `native/infer/export_onnx.py`.
  - If the model is missing, or `nvfx_detail_create` fails, the app runs raw-only. It prints the reason on the
    console and shows "NEURAL UNAVAILABLE: ..." in the HUD.
- Bench and checks:
  - `--bench S --out JSON`, `--warmup S` (default 3).
  - `--bench-dist M` (default 2.15, the sandbox's orbit distance). At this distance the domain cube covers about 1/3
    of the frame; the JSON reports the measured `bbox_frac_mean`.
  - `--frames N`, `--screenshot PATH`.
- Bisection switches:
  - `--no-detail`: never create the detail runner.
  - `--no-render-aux`: draw without AUX; detail then gets NULL alpha/depth/motion, i.e. zeros.
  - `--solver-only`: no renderer and no detail; the HUD shows over a flat background.
  - `--no-hud`.
  - `--sim-every-frame`.
- Other:
  - `--novsync`, `--orbit` (start with auto-orbit), `--march-scale S` (default 0.5).
  - `--adapter N`: DXGI adapter index. The default is the high-performance adapter.
  - `--debug-layer`: the D3D12 debug layer, if Graphics Tools is installed.
- Exit codes:
  - 0: OK.
  - 2: bad arguments.
  - 1: start-up failed.
  - 3: a library call or Present failed mid-run.
  - 4: the screenshot failed.

## Controls (as in `vfx/app3d`, adapted to the mouse)
| input | action |
|---|---|
| RMB drag / wheel | orbit / zoom (pitch -0.15..1.35 rad, distance 0.9..4.5 m) |
| LMB hold | fire at the cursor: `fire_emitter(cursor, r=0.04, fuel 7, T 1300 K)`, the sandbox's held-F torch |
| LMB drag | also pushes the smoke: an impulse along the drag (r 0.07, v = Δp/dt × 0.6, capped at 2 m/s), last 8 per tick |
| Shift+LMB or G | smoke puff at the cursor (density 8, 520 K, 8 ticks) |
| N | toggle RAW 64³ / NEURAL detail |
| 1-4 | presets from `vfx/app3d/presets.py`: campfire, plume past a sphere, twin torches + wind, box obstacle (GAME_SIM dissipation 1.2/s, cooling 5/s) |
| C / R / Space / V / H / Esc | auto-orbit / reset preset / pause sim / vsync on-off / HUD / quit |

The cursor point is `Orbit.cursor_point` from the sandbox. It casts a ray from the camera through the pixel and
intersects it with the plane through the orbit target that faces the camera. The result is clamped to
[0.03, 0.97]³, and nothing is read back from the GPU.

## Frame structure
There is one direct queue, and each frame is one command list:
```
WaitForSingleObjectEx(frame-latency waitable)            <- THE frame-pacing wait (CPU, top of frame)
harvest slot (frame N-3): read own timestamps            (fence check; never blocks after the wait above; counted)
poll nvfx_*_get_timings                                  (non-blocking, lagged)
input -> orbit camera, forces
list.Reset
  [ts0]
  nvfx_solver_record_step × 0..2                         30 Hz accumulator (Python main.py); --sim-every-frame: 1
  nvfx_renderer_record_draw(AUX [| REUSE_VOLUME])        REUSE on frames without a step (and not right after a reset)
  nvfx_detail_record(color, alpha, depth, motion -> app's RGBA8 UAV texture, bbox = nvfx_render_screen_bbox)   NEURAL
  [ts1]
  present pass: SetDescriptorHeaps/root sig of the app (the libraries set their own), barriers, 1 full-screen triangle
                (copy the frame + HUD) into the back buffer
  [ts2] ResolveQueryData -> readback ring
list.Close, ExecuteCommandLists, Signal(fence)
Present(1, 0) | Present(0, ALLOW_TEARING)
```
- **Pacing.** There is a single pacing wait: the swap chain's frame-latency waitable object
  (`SetMaximumFrameLatency(2)`), waited at the top of the frame before input is sampled.
- **Frame slots.** There are 3 per-frame resource slots (allocator, HUD constants, timestamps). Before a slot is
  reused, the app checks its fence. That check cannot block once the latency wait has returned. If it ever does
  block, the event is counted: the HUD shows "FENCE WAITS n" and the JSON reports `slot_fence_waits_blocked`.
- **No other waits.** Nothing else in the loop waits on the GPU. The libraries' own rings (solver 16, renderer 16,
  detail 8 outstanding) are never reached with 2 frames in flight.
- **Upload data.** The CPU→GPU data per frame is small constants only:
  - the forces, camera and parameter blocks the libraries write into their upload rings;
  - the app's 1.7 KB HUD constant buffer.

  No image or volume data moves between CPU and GPU.
- **Prev camera for MOTION.** This is `Pipeline.prev_camera`: the camera of the draw at the same position within the
  previous sim tick, so motion is in pixels per SIM_DT on every frame.
- **Raw mode.** The present pass reads the renderer's COLOR texture directly. Transitions:
  NON_PIXEL_SHADER_RESOURCE → ALL_SHADER_RESOURCE → back.
- **Neural mode.** The present pass reads the app's detail output. Transitions: UNORDERED_ACCESS →
  PIXEL_SHADER_RESOURCE → back.
- **Error handling.** Every `NvfxResult` is checked, and failures print the call name and `nvfx_last_error()`.
  - A failed solver or renderer record is fatal (exit 3).
  - A failed detail record switches the app to raw for the rest of the run.

  In both cases the list is still executed: the libraries' rings expect their recorded work to run.
- **Synchronizing calls.** These happen outside the frame loop:
  - `nvfx_solver_reset` and `set_config` on R and 1-4 submit stand-alone on the same queue. They are ordered before
    the frame's list, which then does a full draw.
  - `nvfx_texture2d_download`, used for `--screenshot`, waits for idle. It runs once, on the last frame, before
    Present.

## HUD
The HUD is drawn in the frame by `present.hlsl`, not in the window title. The text grid is 64x16 cells of 1 byte
each: a glyph plus a 2-bit colour. Together with the 5x7 font, it lives in the root CBV, which is rewritten each
frame from a persistently mapped upload ring.

The HUD adds no texture, no vertex buffer, no extra pass and no readback. The copy into the back buffer has to happen
anyway, and the text is a few ALU ops per pixel inside the HUD rectangle.

The cost is measured: `PRESENT+HUD` = timestamps ts1→ts2, which is the copy plus the HUD. For the HUD's own share,
compare against a `--no-hud` run. It is unmeasured yet; I expect ~0.05 ms for the 1440p copy, with the HUD itself
within noise.

Contents:
- Mode, preset, fps and 1% low over the last 3 s (the mean of the slowest 1 % of present-to-present times, as in
  the sandbox), vsync.
- Per-stage GPU ms, each the median of the last 90 values from the libraries' timestamp timings, against its budget
  (green within, red over):
  - sim step (2.0), with PCG iterations;
  - render new-volume (1.5) and reuse;
  - detail (2.5), with the processed region;
  - TOTAL: the app's own ts0→ts2 timestamps on sim frames (6.0), and the same on frames without a sim step.
- Present+HUD, VRAM, step and frame counters, the key help.

## Bench (`--bench S --mode raw|neural --preset N --out JSON`)
- **Determinism.** The run is deterministic:
  - Vsync is off (tearing if supported).
  - Time is a virtual 60 Hz timeline: the camera auto-orbits at 0.35 rad/s × 1/60 per frame at `--bench-dist`.
  - There is a sim step on every second frame (every frame with `--sim-every-frame`), plus a fire held at
    (0.40, 0.14, 0.50).
  - Wall-clock time is used only for the warm-up cut and the duration.
- **Recorded after warm-up:**
  - every QPC present-to-present CPU frame time;
  - the per-frame GPU time from the app's timestamps, split into sim frames and frames without a sim step;
  - the libraries' per-stage GPU ms: sim, render new-volume, render reuse, detail;
  - the present pass;
  - the projected-bbox fraction;
  - VRAM (DXGI `QueryVideoMemoryInfo`, process local: current, peak, budget).
- **JSON output:**
  - `fps_avg` and `fps_1pct_low`.
  - n/mean/p10/median/p95/p99/max for every stage.
  - `total_gpu_ms.measured_sim_frame_median` against 6 ms. This is the gate figure: a whole sim frame, measured.
  - `stage_sum_sim_frame` (sim + render new + detail medians) and `amortized_per_frame` for reference.
  - Per-stage budget flags.
  - The detail region/crop and VRAM.
  - Both frame-time series.
- A summary is printed to the console.
- **Caveat.** The library timings are polled once per frame and each poll returns only the newest finished item. A
  stage's samples are therefore a subset of its executions, with no frame attribution (see API notes).

## API notes (requests to the lead / library owners)
1. **Timings are "newest finished" only.** `nvfx_{solver,renderer,detail}_get_timings` return only the newest
   finished item. A host polling once per frame can miss items and cannot attribute a value to a frame. Request:
   `get_timings_history(out[], max, *count)` (everything finished since the last call), or a lookup by
   step/draw/frame index.
2. **No record variant of reset.** `nvfx_solver_reset` / `set_config` submit stand-alone on the host's queue in the
   hosted path. An engine wants `nvfx_solver_record_reset(s, list, obstacles, n)`, plus a documented rule for when
   `set_config` takes effect for a recorded step.
3. **Headers in three places.** `nvfx_render.h` and `nvfx_detail.h` still live in `render/` and `infer/`. Once merged
   into `include/nvfx.h`, the app's extra include directories go away.
4. **REUSE after a reset is the caller's job.** `REUSE_VOLUME` after a reset is a caller-side contract. The renderer
   could detect it itself by comparing the solver's step counter or texture pointers with its last full draw.
5. **No cheap model validation.** `nvfx_detail_create` is the only way to check a model, and it takes 1-3 s. A cheap
   `nvfx_detail_probe(path)` would let the HUD say why a model is unusable without paying for the full create.

## Not yet run (GPU pause)
Nothing has been executed. Everything below is unverified:
- Device, swap chain and waitable-object creation. The hosted `nvfx_context_create_from_device` on the app's
  device: its feature checks need SM 6.6, binding tier 3 and typed UAV loads.
- The present PSO and root signature. The HUD layout: font bit order, cell maths, colours.
- The DPI and window-size logic on the actual 1440p display.
- Every library record path inside a foreign command list:
  - solver `record_step`;
  - renderer `record_draw` with AUX/REUSE and the prev camera;
  - detail `record` with the renderer's textures and bbox.

  Also untested: the state contracts between them, and the app re-binding its heap after them.
- Timestamps:
  - the app's own ts0/ts1/ts2 and the readback ring;
  - the lagged polling of the library timings.
- Input mapping (cursor ray, torch, push, puff), presets, reset, vsync toggle with tearing, minimise handling.
- Bench JSON contents, `--screenshot` (`nvfx_texture2d_download` of a swap-chain buffer in PRESENT state), and every
  number: fps, GPU ms, VRAM and the HUD cost.

First run, when the GPU is free (bisect in this order):
```
nvfx_demo.exe --solver-only --frames 120 --screenshot D:\nvfx_demo\s0.bmp
nvfx_demo.exe --no-detail --no-render-aux --frames 120 --screenshot D:\nvfx_demo\s1.bmp
nvfx_demo.exe --no-detail --frames 120 --screenshot D:\nvfx_demo\s2.bmp
nvfx_demo.exe --mode neural --frames 300 --screenshot D:\nvfx_demo\s3.bmp
nvfx_demo.exe                                                     # interactive
```
