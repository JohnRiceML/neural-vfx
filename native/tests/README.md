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
| `test_harness_controls.py` | struct layouts, conversions, the memorylessness of the P1 reference, the skip path, negative controls, `dump_reference` replay |
| `../tools/nvfx_scenes.py` | scenes from `vfx/sim3d/selfcheck.py`: `campfire`, `plume` (+SPHERE), `determinism`, `stress`, `mass_box` |
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
