# native/ — the runtime outside Python (D3D12 + HLSL)

Goal: run the 64³ solver, the half-resolution volume ray-marcher and the route B screen-space detail
network natively, the way a game engine would host them, with no Python, PyTorch or Warp in the loop.
Decisions (user, 2026-11-09): first engine target **Unreal 5**; graphics API **D3D12 compute shaders in
HLSL**. Inference runtime: **DirectML** first (shares D3D12 resources, and is what Unreal's NNE uses on
D3D12); TensorRT only as a timing comparison.

The Python code under `vfx/` is the reference and is not edited by this workstream.

## Contract
`include/nvfx.h` is the contract (C API, structs, constants). Read it first. Summary:

- **Coordinates** as `vfx/core/fields3d.py`: cube of side 1 m, x right, y up, z depth, world units.
- **CPU layout** `(C, D, H, W)`, index `[c, z, y, x]`, float32, channel order density, vel_x, vel_y, vel_z,
  temperature, fuel. A downloaded state is byte-compatible with `FieldState3D.data.cpu().numpy()`.
- **GPU layout**: `Texture3D`, texel `(x, y, z)` = cell `(x, y, z)`.
  `NVFX_TEX_VEL_DENSITY` RGBA32F (vel_x, vel_y, vel_z, density), `NVFX_TEX_TEMP_FUEL` RG32F
  (temperature, fuel), `NVFX_TEX_SOLID` R32F. The renderer and the engine read these directly: no copy,
  no CPU round-trip. How the solver stores its scratch fields is the solver owner's business.
- **Scheme**: `nvfx_solver_step` is `Solver3D.step` — same stages, same order, same constants
  (`vfx/sim3d/README.md`, `kernels.py`, `pressure.py`). Interpolation in the advection must be
  full-precision float (hardware trilinear filtering has 8-bit weights and is not parity-safe; it may be
  offered as a separately measured fast mode, never as the default).
- **No host sync in a step**: `nvfx_solver_step` records and submits, it never waits for the GPU. Only the
  functions marked "synchronizes" (download/upload, `wait_idle`) block.
- **Determinism**: identical inputs give bit-identical state on the same machine and build.
- **Timing**: D3D12 timestamp queries inside the step's command list, split advection / forces /
  projection, read back without blocking (`nvfx_solver_get_timings`).
- **Hosted use**: `nvfx_context_create_from_device` + `nvfx_solver_record_step` record into the host's
  command list. Nothing in the library may assume it owns the device, the queue or the swap chain.

## Layout and ownership (one owner per directory)
| Path | Owner | Gate |
|---|---|---|
| `include/nvfx.h`, `README.md`, `CMakeLists.txt` (top level), `build.ps1` | lead | all |
| `core/` (device, queue, resources, shader loading, timestamp queries) | native solver engineer | 1 |
| `sim/` (solver: C++ + HLSL) | native solver engineer | 1 |
| `tools/` (Python reference dumps), `tests/` (Python-vs-native harness, native gates) | native test engineer | 1–3 |
| `infer/` (ONNX export glue, DirectML / TensorRT runners) | native inference engineer | 2 |
| `render/` (ray-marcher: C++ + HLSL) | native render engineer | 3 |
| `app/` (windowed demo) | native app engineer | 3 |

Rules, as in the rest of the repo: do not edit another owner's files; if an interface is wrong, report it
to the lead; the test engineer never tests code it wrote and never weakens a threshold below; owners may
keep quick self-checks in their own directory; report numbers you measured, and say whether the GPU was
shared (`nvidia-smi`); no git commands — the lead commits.

## Build
Toolchain on the target machine: VS Build Tools 2022 (MSVC 14.44), Windows SDK 10.0.26100 (D3D12 headers,
`dxc.exe`), CMake 4.4 + Ninja. No Vulkan SDK, no CUDA toolkit, no TensorRT SDK — do not depend on them.
```
powershell -File native/build.ps1            # configures + builds Release into native/build/
```
Output: `native/build/bin/nvfx.dll` (+ `nvfx.lib`, shaders embedded in the DLL), test and tool
executables beside it. `native/build/` is git-ignored. Third-party code: none vendored without the lead's
approval; small NuGet packages (DirectML, ONNX Runtime) are fetched into `native/build/_deps`.

## Gate 1 — native solver: what "passes" means
Reference: the Python `Solver3D` from this checkout, run through the main checkout's venv
(`C:\Users\ricej\OneDrive\Documents\RustGames\neural-vfx\.venv\Scripts\python`), same inputs.
Scenes: `campfire` and `plume` (plume past a sphere) from `vfx/sim3d/selfcheck.py`, 64³, dt = 1/30 s.
Error measures per channel c at one step: `rel_l2 = ‖n − p‖₂ / (‖p − ambient_c‖₂ + ε)` and
`abs/peak = |n − p| / max|p − ambient_c|` (ambient 293 K for temperature, 0 otherwise).

| # | Check | Threshold |
|---|---|---|
| P1 | **Single-step parity** (teacher-forced): upload the Python state of step k, step both once with a fixed iteration count (`pressure_tol 0`, `pressure_max_iters 8`, warm start off); k = 0…119 on both scenes | every channel, every step: rel_l2 ≤ 1e-4, 99.9th-percentile abs/peak ≤ 1e-3, max abs/peak ≤ 5e-2 |
| P2 | **Per-stage parity** where a stage can be isolated through the public API: projection only (`nvfx_solver_project` vs `Solver3D.project` on the same field), and a step with every source/force term zeroed (pure advection + mass correction + projection) | velocity rel_l2 ≤ 1e-4 |
| P3 | **Free-running parity**: 120 steps from reset with default config, no teacher forcing | at step 120: total smoke mass and total heat (Σ(T − ambient)) within 1 %, density centroid within 0.01 m; the rel_l2 curve is reported, not gated (the flow is chaotic) |
| S1 | Normalized divergence after a default-config step, both scenes, steps 1…120 | max < 1e-4 |
| S2 | Mass in a closed box with a sphere, no sources, 1000 steps | within 0.1 % |
| S3 | Determinism: two runs of 200 steps with emitters and impulses | bit-identical download |
| S4 | Stress: 1000 steps of the `selfcheck.py` stress scenario | no NaN/Inf, max\|u\| and Tmax bounded as in Python |
| T1 | Step time at 64³, GPU timestamp queries, both scenes, default and runtime config, median of ≥ 200 steps after warm-up, split by stage, Python `step_profiled` measured in the same session | ≤ 2.0 ms (budget); the Python figure is reported alongside, not gated |

Clarifications (lead, after the test engineer's contract review; no threshold changed):
- ε in rel_l2 is 1e-12. Where the reference channel is exactly ambient everywhere (e.g. fuel in `plume`),
  abs/peak is undefined and the native channel must match exactly.
- P1 trajectory states x_k come from a default-config Python run; the one-step comparison uses the P1 config.
  P2 `project` uses the P1 pressure config.
- `pressure_tol 0` is "run `pressure_max_iters`" **with Python's remaining stops still active** (1e-6 float32
  floor, stagnation at residual > 4× the best seen, rz = 0). The native loop implements the same stops.
- S4 bound: Tmax ≤ `max_temperature`, and max|u| ≤ 1.5× the Python max|u| measured in the same session.
- `step_index` counts completed `nvfx_solver_step` calls since create/reset (first step = 1).
  `nvfx_solver_get_timings` before any step has finished returns zeros with `step_index` 0.

The thresholds are set before any native code exists. If one turns out to be wrong for a reason that is
not a bug (for example a discontinuous limiter amplifying rounding), the test engineer reports the numbers
and the lead decides; nobody edits a threshold to make a test pass.

## Gates 2 and 3 (interfaces fixed when Gate 1 passes)
- Gate 2: `ScreenDetail` exported to ONNX, run with DirectML on D3D12 buffers shared with the renderer,
  FP16, outputs compared with PyTorch, timed at 2560×1440 on a ~1/3-screen effect against 2.5 ms.
- Gate 3: windowed demo — solver → half-resolution march (colour + alpha/depth/motion) → detail →
  present; orbit camera, fire/smoke at the cursor, raw/neural toggle, per-stage GPU ms against 6 ms.
