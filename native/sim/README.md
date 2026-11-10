# native/sim — the 64³ smoke + fire solver on D3D12 (owner: native solver engineer)

A port of `vfx/sim3d` `Solver3D` (sim3d-warp-0.1.0) to C++20 and HLSL compute shaders. It has the same stages, the
same order, the same constants and the same operation order inside each kernel. Files:
- `solver.cpp`: the C API in `include/nvfx.h`, resources and command recording.
- `shaders/*.hlsl`: `advect`, `forces`, `pressure`, `setup`. They are compiled by dxc at build time (`cs_6_6 -O3 -Gis
  -denorm preserve`) and embedded in `nvfx.dll`.
- `slots.h`: descriptor and scalar slots, shared by C++ and HLSL.
- `selfcheck.cpp` (`nvfx_selfcheck.exe`): the owner self-check, which uses the C API only.
- `parity.py`: the owner's ctypes diff against the Python `Solver3D`.

## Design
- **State** lives in the public textures and ping-pongs between two sets: `VEL_DENSITY` RGBA32F and `TEMP_FUEL`
  RG32F. `SOLID` R32F is a single texture. Between calls they are in NON_PIXEL_SHADER_RESOURCE.
  Scratch fields are internal RGBA32F/RG32F textures (always UAV) and flat structured buffers (pressure).
- **Binding**: there is one shader-visible heap per solver, bound once per list. One root signature holds 16 root
  constants and one root CBV with the per-step parameter block (the Python `prm`). Every resource is reached
  through `ResourceDescriptorHeap[SL_*]`, so there are no descriptor tables. PSOs are prebuilt. Only global UAV
  barriers are used, and only between dependent dispatches. State transitions are batched into those barriers.
- **Interpolation** is manual trilinear from `Load`s in full fp32. Hardware filtering is never used.
- **Reductions** (mass, p·q, r·z, max|r|, max|u|): one partial per group, fixed-order groupshared trees. No
  atomics anywhere, no host readback in a step, and reruns are bit-identical.
- **GPU-side PCG stop**: the list records `pressure_max_iters` iterations. Iteration k is wrapped in
  `SetPredication` (EQUAL_ZERO) on its own 8-byte predicate buffer, which iteration k−1's stopping test writes.
  Predicated-off dispatches do not run, but their barriers still execute (see the costs below).
  `ExecuteIndirect` with zeroed arguments would keep the same barriers, so predication was used.
- **Timings**: 4 timestamps per step go into a 64-slot readback ring tagged by a `WriteBufferImmediate`
  MARKER_OUT. `get_timings` returns the newest tagged step and never blocks. `pcg_iters` comes from the same
  ring. The parameter upload ring has 16 slots, so a step can block only when the GPU is 16 steps behind
  (like Python's HOST_RING).
- **Fused path (default)**: the same arithmetic as the reference sequence, with 14 instead of 25 dependent
  dispatches per PCG iteration. It adds `p_cg_update` (fin_rz + upd_p + the next matvec, with p ping-pong and
  banked scalars), `p_resid_fwd`, and the coarse V-cycle from m ≤ 8 in one 1024-thread group with groupshared
  x/r. `NVFX_SIM_FUSED=0` records the reference (Python) sequence, and `NVFX_SIM_BLOCK_M=2|4|8` sets the coarse
  level. Stencil kernels issue all of their loads before the mask tests (clamped indices), which halved
  matvec / resid / post_rz.
- `NVFX_SIM_PROFILE=1`: a timestamp after every dispatch, the step waits for the GPU, and a per-kernel table is
  printed to stderr at `nvfx_solver_destroy`. This is for debugging only.

## Measured (RTX 5080; the GPU was SHARED: another process held 84-99 % utilization and 10-13 GB the whole time)
GPU timestamps, 64³, median (and p10) of 240 steps after a 60-step warm-up. Python is `selfcheck.profile()` (what
`--bench` uses), measured in the same sitting:

| config | scene | native total | adv / forces / proj | PCG iters | Python step (adv / forces / proj) |
|---|---|---|---|---|---|
| full | campfire | 1.18 (p10 0.99) | 0.038 / 0.058 / 1.07 | 5 | 1.02 (0.049 / 0.072 / 1.41) |
| full | plume + sphere | 1.29 (p10 1.09) | 0.038 / 0.056 / 1.17 | 6 | 2.26 (0.049 / 0.070 / 1.71) |
| runtime | campfire | 0.57 (p10 0.56) | 0.038 / 0.056 / 0.475 | 4 | 0.93 (0.049 / 0.072 / 0.68) |
| runtime | plume + sphere | 0.56 (p10 0.56) | 0.038 / 0.055 / 0.472 | 4 | 0.85 (0.049 / 0.070 / 0.68) |

- **Executed PCG iteration: 0.095 ms** (fixed 0/1/2/4 iterations: 0.093 / 0.195 / 0.287 / 0.471 ms projection). The
  reference sequence costs 0.127 ms; Python's README gives 0.125 ms.
- **Skipped (predicated-off) iteration: 10.3 µs**, which is about 0.6 µs per dispatch+barrier plus about 2 µs for
  the predicate fetch (40 skipped: +0.40 ms; reference sequence: 13 µs). At the default `pressure_max_iters 40`,
  plume runs 6 iterations and pays about 0.35 ms for the 34 unused slots. **This is the main gap to Warp's CUDA-graph
  conditional, whose skipped iterations cost nothing.** A lower cap removes most of it (12 would cost about 0.06 ms).
- What limits a step: dependent-dispatch latency. Small multigrid dispatches cost 2.4-4 µs each, full-grid
  stencil kernels 5-7 µs. The single-group coarse kernel (24 µs) is throughput-bound on one SM, and it costs the
  same as the 4 dispatches it replaces at m = 8. Advection is 36 µs, mostly dependent texture-fetch chains of
  the RK2 traces.
- Per-kernel medians (`NVFX_SIM_PROFILE`, plume, 8 fixed iterations): cg_update 14.2, pre 6.8, resid_fwd 7.4,
  smooth_h 4.7 / 2.8 (m = 32 / 16), restrict 5.3 / 4.1, coarse_lds 24.5, prolong_black 5.3 / 3.5, inv_transform
  4.1, post_rz 6.1 µs.

## Self-check (`nvfx_selfcheck.exe`, all through the C API)
- Divergence at the default config, steps 1..120: 2.2e-5 (campfire), 2.0e-5 (plume). Smoothed random field:
  5.6e-6 / 1.2e-5 (without / with sphere).
- Mass over 1000 closed-box steps with a sphere: +0.001 %.
- Rerun of 200 steps (campfire + impulses + sphere): bit-identical. The hosted path (adopted device,
  `record_step` into a host list) is bit-identical to `nvfx_solver_step`.
- Stress run of 1000 steps (own RNG for the impulses, not numpy's): finite, max|u| 3.5 m/s, Tmax 2658 K, solid
  cells exactly 0 / ambient. Divergence: 1.6e-5 full, 1.5e-3 runtime (Python reports runtime misses the gate
  near obstacles).

## Parity with Python (`parity.py`, same sitting)
- P2 projection only (8 iterations, cold): velocity rel_l2 1.3e-7 (open box) / 1.7e-7 (sphere).
- P1 teacher-forced, k = 0..119, worst over steps and channels: rel_l2 3.3e-5 (temperature; velocity ≤ 4.4e-6),
  p99.9 abs/peak ≤ 2.6e-7, max abs/peak ≤ 5.0e-4 (gate 1e-4 / 1e-3 / 5e-2).
- P3 free-running 120 steps: mass −0.15 % / 0.00 %, heat +0.42 % / **+2.4 %** (campfire / plume), density
  centroid ≤ 0.006 m. The rel_l2 curve goes from 2e-6 at step 20 to 3e-2 at step 40 to 0.7 at step 120. This is
  chaos, not a porting error: **Python against Python**, with a ±1e-7 relative velocity perturbation after step 1,
  differs at step 120 by plume heat +1.0 / +1.5 %, campfire heat −0.4 / −0.7 %, and vel_y rel_l2 0.75. PCG
  iteration counts then differ at 6-8 of the 120 steps. The 1 % heat gate of P3 is therefore not reachable on
  plume, even Python against itself.

## Deviations from the Python scheme
- None in the maths. Differences are rounding only: FMA contraction and exp/div accuracy of the driver against
  nvcc, and reduction tree shapes (mass, dot products), which change the mass ratio and CG scalars at the 1e-7 level.
- `upload_state` and `reset` clear the warm start (φ := 0). Python's `reset()` keeps φ.
- Obstacles are rasterized on the CPU with the same float32 operations as torch. Sphere r² is computed in double
  from the float32 radius, so a cell exactly on the boundary could differ from Python's Python-double radius.
- Solid changes are detected by API (reset or upload with `solid`), not by a per-step GPU compare. The MG
  operators are rebuilt only then.
- `pressure_max_iters` is capped at 256 (one predicate buffer per iteration).

## Caveats / known problems
- Not validated by the D3D12 debug layer (not installed). The states assumed are: public textures NPSR between
  calls; scratch buffers rely on implicit promotion and decay; params, scalars and predicates are explicitly
  COMMON at list boundaries.
- `record_step` sets its own descriptor heap, root signature and pipeline on the host list. The host must
  re-bind its own afterwards. The parameter ring assumes the host executes recorded steps: if 16 are recorded
  without being executed, the 17th polls for up to 20 s and then fails.
- Timings measure GPU wall time between timestamps. On a shared GPU this includes time-slices of other
  processes, so use the p10 numbers to compare.
- Only 64³ has been measured. Other even sizes build the same level structure as Python, with the fast V-cycle
  for powers of two and the reference V-cycle otherwise; 96³ is untested.
