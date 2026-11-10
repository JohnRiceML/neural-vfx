# native/tests — Gate 1 harness (Python reference vs nvfx.dll)

Owner: native test engineer. This harness was written from `native/include/nvfx.h` and `native/README.md` alone.
The native solver (`native/core`, `native/sim`) is a black box here and is reached only through its C API.

## Files
| file | what |
|---|---|
| `nvfx_ctypes.py` | ctypes binding of all of `nvfx.h`: constants, structs (x64 layouts asserted), every prototype, and `nvfx_last_error` checking. Also the `NativeSolver` wrapper: `reset(obstacles)`, `step(Forces3D, dt)`, `project`, `download` → (6,D,H,W) f32, `download_solid`, `upload(data, solid)`, `divergence`, `get_timings`, `wait_idle`. |
| `backends.py` | `PyFakeSolver`: the same interface built on `vfx.sim3d.Solver3D`. `upload`/`reset` zero the warm-start φ, as the contract requires. `Perturbed`: deliberately broken fakes for the negative controls. |
| `gate1_checks.py` | P1, P2, P3, S1–S4 and T1 as backend-agnostic functions. The Python reference runs in-process in the same session, so nothing large is stored. |
| `test_gate1.py` | the gate tests (`--backend` selects the backend) |
| `test_p3_noise_floor.py` | P3 noise-floor **measurement** (not a gate): Python rounding-perturbation ensemble vs the backend (section below) |
| `test_harness_controls.py` | struct layouts, conversions, the memorylessness of the P1 reference, the skip path, negative controls, `dump_reference` replay |
| `../tools/nvfx_scenes.py` | scenes from `vfx/sim3d/selfcheck.py`: `campfire`, `plume` (+SPHERE), `determinism`, `stress`, `mass_box` |
| `test_gate2.py`, `gate2_checks.py`, `nvfx_detail_ctypes.py` | Gate 2 (native detail inference), section below |
| `../tools/detail_frames.py` | Gate 2 inputs: real 2560×1440 frames + aux rendered by the Python reference |
| `../tools/dump_reference.py` | stored references for non-Python consumers (format in its docstring; output under `%LOCALAPPDATA%\neural-vfx\native_ref`) |

## Run (from the worktree root)
```
set PY=C:\Users\ricej\OneDrive\Documents\RustGames\neural-vfx\.venv\Scripts\python
%PY% -m pytest native/tests -q -s                       # native backend (env NVFX_DLL overrides the DLL path)
%PY% -m pytest native/tests -q -s --backend=pyfake      # harness validation on the reference-backed fake
%PY% -m pytest native/tests/test_harness_controls.py -q -s   # controls only (never needs the DLL)
%PY% native/tools/dump_reference.py --scene plume --steps 120 --with-onestep --every 10
```
The backend can also be set with the env var `NVFX_BACKEND=native|pyfake`. `NVFX_DEBUG_LAYER=1` turns on the D3D12 debug layer.
When the DLL is missing, every gate test skips with the message `native backend: nvfx.dll not found at … (build with …)`.

Results are written to `native/build/test_results/` (git-ignored):
- `gate1_<backend>.json`: every measured number and the per-step curves. Partial runs made with `-k` are merged into it.
- `gate1_<backend>.md`: a pass/fail table plus the T1 stage table, ready to paste into a report.
- `harness_controls.json` / `.md`

## What each test measures (thresholds quoted from native/README.md)
Error measures, per channel c, computed in float64 over all cells: `rel_l2 = ‖n − p‖₂ / (‖p − ambient_c‖₂ + ε)` and
`abs/peak = |n − p| / max|p − ambient_c|`. Ambient is 293 K for temperature and 0 for the other channels. **ε = 1e-12** (the contract does not give a value).
When `max|p − ambient_c| = 0`, abs/peak is 0 where n == p and +inf elsewhere (strict).

- **P1**: teacher-forced single step, k = 0…119, campfire and plume. A Python trajectory with the *default* config supplies x_k.
  x_k is stepped once by (a) a separate `Solver3D` configured with `pressure_tol 0, pressure_max_iters 8, pressure_warm_start False`, and
  (b) the backend after `upload(x_k)` with the same config. Pass: for every channel and every step, rel_l2 ≤ 1e-4, the 99.9th percentile of abs/peak ≤ 1e-3, and the max of abs/peak ≤ 5e-2.
  P1 also asserts that the rasterized `solid` is identical to Python's.
- **P2**: (i) `project()` against `Solver3D.project` on three fields: selfcheck's smoothed random field, the same field with the sphere, and the plume state at step 60 plus a random field.
  (ii) One step with every source and force term zeroed (`buoyancy_alpha/beta, vorticity_eps, density_dissipation, temperature_cooling, burn_rate, heat_diffusion = 0`; no emitters, impulses or wind), starting from step 60 of each scene.
  Both use the P1 pressure config. Pass: velocity rel_l2 ≤ 1e-4, checked per velocity channel.
- **P3**: 120 free steps with the default config. At step 120: total smoke mass (Σ density) and total heat (Σ(T − 293), not clamped) within 1 %, and the 3D density centroid within 0.01 m.
  The rel_l2 curve is recorded every 10 steps and is not gated.
- **S1**: the max normalized divergence after every default-config step, steps 1…120, both scenes, must be < 1e-4.
- **S2**: `selfcheck.mass_drift`: closed box, sphere at (0.6,0.6,0.55) with r 0.1, density blob and shear velocity, `density_dissipation 0`, projected once, then 1000 steps. Mass must stay within 0.1 %.
  Python's `mass_drift` is run in the same session for the side-by-side figure.
- **S3**: two fresh solvers each run 200 steps of campfire, the sphere, and an impulse every 10 steps. The two downloads must be bit-identical (uint32 compare).
  S3b, an extra contract check: one solver, `reset`, then the same 200 steps again, must also be bit-identical.
- **S4**: `selfcheck.stress` replayed for 1000 steps (default config). max|u| is sampled as selfcheck does it (`velocity.abs().max()` every 100 steps).
  The gate requires no NaN/Inf and Tmax ≤ `max_temperature`. "max|u| bounded as in Python" has no number in the contract, so this harness uses a **provisional** bound: max|u| ≤ 1.5 × Python's max|u| in the same session. **The lead must decide this bound.**
- **T1**: both scenes, in the default and the runtime config. 60 warm-up steps, then ≥ 200 distinct `nvfx_solver_get_timings` samples, read without blocking after each step and de-duplicated by `step_index`.
  If those reads give fewer than 200 distinct samples, the harness falls back to `wait_idle` before each read, and the result records which method was used.
  The test reports the median and p95 per stage. Pass: total median ≤ 2.0 ms. In the same session the harness also times Python: the median and p95 per stage of 200 `step_profiled` calls, and 200 single-graph `step`s timed with CUDA events.
  It records `nvidia-smi` utilization, memory and compute apps before and after.

## How P1's Python side is memoryless
With `pressure_warm_start=False`, `k_rhs` in `pressure.py` sets φ = 0 before it builds r. The `init=1` pass re-initialises the PCG scalars and the iteration counter.
The multigrid operators depend only on `solid`, and every other buffer (hat/adv/omega/force, the mass ratio) is fully overwritten each step.
`test_python_one_step_is_memoryless` checks this directly: stepping x₁₀, then x₃₉, then x₁₀ again gives bits identical to the first x₁₀ step and to a fresh solver.
The same test shows that a warm-started solver is *not* memoryless.
The harness also does not rely on that property alone. P1 uses a separate one-step `Solver3D` (P1 config) whose tensors are overwritten with x_k before each step, so the trajectory solver's warm-started φ never enters the reference.
The fake zeroes `pressure.phi` on `upload`/`reset`, mirroring the contract's "upload clears the warm start".

## Measured: harness validation on `--backend=pyfake` (2026-11-09, RTX 5080, GPU shared with a training job at 81–96 % util, ~12.1 GB used)
- P1 campfire and plume, k = 0…119: every measure is exactly **0** in every channel and every step. Solid mismatch: 0 cells.
- P2: project (3 fields) and the zero-sources step (both scenes): velocity rel_l2 **0**.
- P3: mass, heat and centroid differences are **0**, and the rel_l2 curve is 0.
- S1: max div 2.21e-5 for campfire (step 6) and 1.99e-5 for plume (step 31). At step 120: 5.1e-6 / 1.4e-5.
- S2: **+0.00016 %**, equal to `selfcheck.mass_drift` in the same session and to vfx/sim3d/README (+0.00016 %).
- S3 / S3b: bit-identical.
- S4: finite, max|u| **3.742 m/s**, Tmax **2664 K**, div **1.01e-5**, solid clean 0. These are identical to `selfcheck.stress` in the same session; vfx/sim3d/README gives 3.7 m/s, 2664 K, 1.0e-5.
- T1 (fake = Python step_profiled), total median (p95) in ms:

  | scene | config | total median (p95) |
  |---|---|---|
  | campfire | default | 1.315 (9.32) |
  | campfire | runtime | 0.795 (1.57) |
  | plume | default | 1.402 (3.34) |
  | plume | runtime | 0.795 (1.61) |

  Python single-graph step: 1.036, 0.787, 1.300 and 0.787 ms. The p95 values are inflated by the shared GPU.

## Measured: negative controls (each must make its check FAIL; the 1e-6 control must PASS)
| broken fake | check | result |
|---|---|---|
| density × (1+1e-3) on download | P1 | FAIL as expected: density rel_l2 1.0e-3 (10× threshold) |
| density × (1+1e-6) on download | P1 | PASS as expected: rel_l2 1.0e-6 (0.01× threshold) |
| vel_x and vel_z swapped | P1 | FAIL as expected: vel_x rel_l2 1.33 |
| one-cell roll along x | P1 | FAIL as expected: density rel_l2 0.61 |
| `project()` is a no-op | P2 | FAIL as expected: velocity rel_l2 0.74 |
| density × 1.02 after each step | P3 | FAIL as expected: mass +300 %, centroid 0.041 m |
| 2 PCG iterations, tol 0 | S1 (plume) | FAIL as expected: div 1.6e-2 |
| 1-ulp nudge to one random cell per step | S3 (20 steps) | FAIL as expected: 722551 values differ |

## P3 noise floor (a MEASUREMENT, not a gate): `test_p3_noise_floor.py`, `gate1_checks.measure_p3_noise_floor`
This answers one question: is the backend-vs-Python P3 difference larger than the spread Python shows against itself?
For each scene the harness replays the P3 run in Python (fresh `Solver3D`, default config, 120 steps). Each ensemble member gets one perturbation, injected after step 1:
- `vel_scale`: velocity × (1 + k·1e-7) for k = ±1…8, computed in f64 and rounded to f32 (16 members).
- `ulp_nudge`: vx, vy, vz and T nudged by ±1 ulp at 256 random fluid cells, seeds 0…15 (16 members).
- `matched_amp`: multiplicative Gaussian noise per channel, with amplitude equal to the backend's own step-1 rel_l2 (8 members, reported separately).

Every member is compared with the unperturbed run at steps 1, 2, 3, 5, 10, …, 120. The measured quantities are the signed heat %, signed mass %, centroid distance, and velocity, T and density rel_l2.
A fresh backend run in the same session is compared with the same unperturbed run. At step 120 the backend's position in the ensemble is reported as z = (x − mean)/std and as k/N, the number of members with |x| ≥ |backend|.
Control: a zero perturbation (k = 0, through the same code path) must reproduce the unperturbed run bit for bit. The test also asserts that every real perturbation changed the step-1 state.
These are the only assertions. P3 and its thresholds are unchanged. Run: `%PY% -m pytest native/tests/test_p3_noise_floor.py -q -s` (about 65 s).
Output: `native/build/test_results/p3_noise_floor.json` (every member's curve) and `.md`.

Measured 2026-11-09 against nvfx 0.1, with the GPU shared with a training job at 97 % util and 11.8 GB used (this does not affect the numbers; nothing here is timed):

| scene | Python rounding ensemble (N 32), heat % at 120: min / median / max / std | backend heat % | z | k/N | Python members failing P3's 1 % heat check |
|---|---|---|---|---|---|
| campfire | −1.19 / −0.07 / +1.60 / 0.71 | +0.42 | +0.7 | 17/32 | 6/32 |
| plume | −1.52 / +1.48 / +4.14 / 1.41 | +2.38 | +0.7 | 7/32 | 25/32 |

The zero-perturbation control was bit-identical. The backend's step-1 difference is rounding-level: rel_l2 is 0.9–2.7e-7 per channel. Backend trajectories do not separate earlier than perturbed Python ones.
For plume, the backend's velocity rel_l2 is 1.2e-6 at step 20, while the ensemble median is 1.3e-4. The matched-amplitude members track the backend almost exactly (1.8e-4 at step 30 for both).
Mass is far inside the noise in both scenes, and the centroid distance is z ≤ 1.3 and below 0.01 m in every member.

## Gate 2 — native detail inference (`test_gate2.py`, `gate2_checks.py`, `nvfx_detail_ctypes.py`, `../tools/detail_frames.py`)
The runner is a black box here. The harness was written from `infer/nvfx_detail.h`, `include/nvfx.h` and the "how to run" part of `infer/README.md`.
The reference is `vfx.runtime3d.ScreenDetail(latest.pt).infer(frame, aux)`, run in-process in the same session.

**Contract gap.** `nvfx_detail_run` takes `ID3D12Resource*` textures, and no public function creates a Texture2D or copies CPU data into or out of one.
The tests use four proposed helpers: `nvfx_texture2d_create`, `nvfx_texture2d_release`, `nvfx_texture2d_upload` and `nvfx_texture2d_download`. Their exact signatures are in the `nvfx_detail_ctypes.py` docstring.
Every native test skips with that message until `nvfx.dll` exports them.

Inputs: `detail_frames.py` runs `Solver3D` with the default config for 150 steps (campfire, plume + sphere) and renders with `VolumeRenderer((1440, 2560), march_scale=0.5).draw(..., return_aux=True, velocity, prev_camera)`.
The camera is `camera_at(1.3)` and the previous camera is `camera_at(1.32)`. The frames are cached under `native/build/gate2_cache/`.
The aux data is float16 on both sides. The native colour texture carries a non-constant A pattern, so the "A passed through" behaviour can be observed.
The ONNX model's `screennet.json` `checkpoint_sha256` must equal the SHA-256 of the checkpoint that Python loads, or the native tests error.

| test | what |
|---|---|
| `test_unit_*` | struct layouts, D1 comparator on synthetic data (including the 0.01 % boundary at 100 and 101 px out of 1e6), bbox plans. Needs no GPU and no DLL. |
| `HC2_py_{self,alpha0,shift64}` | harness controls with no DLL. Python is compared with Python through the D1 comparator: a second instance must PASS; alpha zeroed and the bbox shifted by 64 px must FAIL. |
| `D1_{campfire,plume}` | 12 bboxes per scene: the effect bbox, 5 centred sizes covering every width bucket including 1536×832 → 1600×960, tall/flat cases, two corners, explicit full frame and `None`/(0,0,0,0). The output is pre-filled with garbage first. |
| `D1_region_sweep` | `nvfx_detail_region` against `ScreenDetail._region` on about 3000 bboxes, including partly off-screen ones |
| `D1_NC_{alpha0,shift64}` | negative controls: the native side gets alpha = 0 or the bbox shifted by 64 px, and D1 must FAIL. For alpha0 the pixel part must fail by itself. |
| `D2_*` | run → A, run → B (pre-filled with different garbage), another bbox, then the same frame → A again. All three outputs must be bit-identical. Done for the 1/3 bbox and the full frame. |
| `D3` | a fresh `NvfxDetail`, then the 200 seed-0 random bboxes (alternating scenes). Each call's wall time is measured in Python. Calls 0, 22, …, 199 are downloaded and must pass D1. |
| `D4` | `QueueLoad` queues 64³ solver steps (tol 0, 256 iteration slots) on the same context, then `nvfx_detail_run` is timed. The test checks that ≥ 20 ms of work was still queued ahead of the call (wall time to idle after the return, minus the frame's own GPU ms). Otherwise it reports PRECONDITION NOT MET. |
| `T3` | 30 warm-up runs, then ≥ 300 distinct `get_timings` frames, read without blocking. Reports the median, p10 and p90 of pre, net, post, their sum, total and cpu. The 1/3 bbox is gated; the full frame is reported. PyTorch `ScreenDetail.timings` is measured in the same session, and `nvidia-smi` is read before and after. |

Run: `%PY% -m pytest native/tests/test_gate2.py -q -s` (`-k unit` runs without a GPU). The env vars are listed in the `test_gate2.py` docstring.
Output: `native/build/test_results/gate2_native.json` and `.md`.

Gate 2 interpretations made by this harness (for the lead; no threshold changed):
1. D1 "pixels differing by more than 1": a pixel counts if any of R, G, B differs by more than 1. The denominator is the processed region, which is stricter; the whole-frame fraction is reported too.
2. D1 compares RGB only, because Python returns RGB. A pass-through (stated in the header) is reported as `a_passthrough` and is not gated.
3. T3 "pre + net + post" is gated on `total_ms` (first to last timestamp, which is ≥ the sum). The sum is reported.
   The table asks for the median and **p10**. p10 is the fast tail, so p90 is reported as well.
4. Degenerate bboxes differ by design: in native, `x1 <= x0` means the whole frame; Python's `ScreenDetail` uses the smallest bucket. Full-frame parity therefore uses `None` in Python against (0,0,0,0) in native, plus an explicit (0,0,W,H) on both sides.
5. D3 "after creation" means the first 200 `nvfx_detail_run` calls of a fresh object, with input uploads (synchronizing) before them. The CPU time is wall time around the ctypes call, which is stricter than `cpu_ms`.
6. D4 does not say how to queue 20 ms of GPU work. The harness uses the public solver API on the same context and verifies the backlog.
7. The table names no camera or step count for the D1 frames. This harness uses 150 steps with `camera_at(1.3)`, as `infer/` did for export verification.
8. The PyTorch `total` includes the full-frame `out.copy_(frame)`, so it is not the same quantity as the native `total_ms`.

## Contract ambiguities found (for the lead)
1. **ε in rel_l2 is unspecified.** This harness uses 1e-12.
2. **abs/peak is undefined when the reference channel is exactly ambient.** In plume, fuel is 0 in all 120 P1 steps. The harness then requires n == p exactly.
3. **P1 does not say which config generates the states x_k.** The harness uses the default config for the trajectory and the P1 config for the single step.
4. **`pressure_tol 0` is not a strict fixed iteration count in Python.** `fin_rz` still stops at the 1e-6·max|u| float32 floor, on stagnation (residual > 4× the best seen) or when rz = 0.
   Parity in P1 and P2 therefore requires the native loop to apply the same stopping rules, not just to run 8 iterations.
5. **P2 does not give a config for `project`.** The harness uses the P1 pressure config.
6. **S4 "bounded as in Python" has no number.** The provisional bound is 1.5× Python's max|u|, plus Tmax ≤ max_temperature.
7. **`NvfxSolverTimings.step_index`** has no stated base (0 or 1), and nothing says what `get_timings` returns before the first step completes.
   The harness accepts only `total_ms > 0` and `step_index > warm-up count`.

## Gate 3a — native ray-marcher (`test_gate3a.py`, `gate3a_checks.py`, `nvfx_render_ctypes.py`, `../tools/gate3a_dither_probe.py`)
**Status (2026-11-09): written, collected (77 tests), CPU-only unit tests pass (6). NOTHING on the GPU has run (GPU pause).**
Black box: written from `render/nvfx_render.h`, `include/nvfx.h` and the API part of `render/README.md` only.
Reference: `VolumeRenderer((1440, 2560), march_scale=0.5).draw(volume, cam, solid=state.solid, return_aux=True,
velocity=state.velocity, prev_camera=prev)` on the Python `Solver3D` state after 150 steps; that state (data + solid)
is uploaded to a native solver (`nvfx_solver_upload_state`, round-trip checked bit-exact: `PRE_upload_*`) and drawn natively.

Cameras: `Camera()`; `Camera()` orbited 60° about the vertical axis through the domain centre; close-up = `bench.camera_at(d)`
with the largest d in 1.3…0.6 whose Python alpha > 0.05 coverage is ≥ 1/3 (`PRE_closeup_coverage_*`).
prev_camera = camera orbited −3° about the domain centre (moving camera). R4 also runs with prev_camera None (moving volume only).

| test | what |
|---|---|
| `test_unit_*` | struct layouts; `NvfxRenderParams` field order = `RenderParams3D` (AST); `camera_at` = bench.py (executed from source); orbit/yaw geometry; R1–R4 comparators incl. threshold boundaries; PNG round-trip. No GPU, no DLL, no torch. |
| `HC3_*`, `PRE_closeup_*` | Python only (CUDA, no DLL): Python vs Python must PASS; Python yawed 1° vs Python must FAIL R1; motion swapped / negated must FAIL R4 |
| `N_api_defaults`, `N_march_size` | `nvfx_render_params_default` / `nvfx_camera_default` = Python defaults (f32); march buffer size = Python's |
| `R1`–`R6` | per (scene, camera), thresholds quoted from native/README.md. R4 gates on the mean end-point error (stricter than the per-component mean, which is reported). R5: AUX off vs on colour bytes. R6: same draw again after two other draws, all 4 outputs bit-identical. |
| `NC_R1_yaw1_*`, `NC_R4_{swapxy,negate}_*` | negative controls vs native; must FAIL |
| `R1_dither_modes` | REPORT: R1 in a fresh process per `NVFX_RENDER_DITHER_MODE` (unset, 0, 1, 2) and which mode the default is byte-identical to. The gate R1 is the in-process default run (the test fails R1 if the env var is set). |
| `T2_{full,view}` | ≥ 300 distinct `nvfx_renderer_get_timings` draws after 30 warm-up, aux on: (full) native solver stepped between draws, (view) `REUSE_VOLUME` with the camera orbiting 0.25°/frame; `flags` checked. Python `VolumeRenderer.timings` (`last_timings(sync=True)`) in the same session; nvidia-smi before/after. |

R1's "no structured error visible" is not automated. The gate is the two numbers. Reported proxy: `struct_energy` = share of the
signed error energy that survives 8×8 block averaging (≈ 0.016 for uncorrelated dither, ≈ 1 for smooth error), and the same
vs the error with its pixels shuffled (`struct_ratio`). The images are flagged for the lead's visual review.
Images: `native/build/test_results/gate3a_images/<scene>_<camera>_{python,native,diff_x8,nc_yaw1_diff_x8}.png` + `contact_sheet.png`
(rows = cases; columns Python | native | diff ×8, ¼ size, diff max-pooled). Dither-probe inputs: `native/build/gate3a_cache/`.
Run (its own pytest process: one nvfx.dll per process; DLL = env NVFX_DLL, else `native/build-render/bin/nvfx.dll`):
`%PY% -m pytest native/tests/test_gate3a.py -q -s`. Output: `native/build/test_results/gate3a_native.json` / `.md`.
