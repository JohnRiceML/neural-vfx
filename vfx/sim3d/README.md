# vfx/sim3d — 3D smoke + fire solver (Warp)

Interface as in `CLAUDE.md` (3D phase): `Solver3D`, `SolverConfig3D`, `SOLVER3D_VERSION`
(`reset / step / divergence / advect / project`), plus `runtime_config()` (the game-budget setting)
and `Solver3D.step_profiled()` (the same step, run as three graphs with CUDA events between them; returns
ms per stage). `FieldState3D.data` stays a torch cuda tensor that Warp reads and writes in place (`wp.from_torch`, no copy).
Files: `config.py`, `kernels.py` (advection, sources, diffusion, forces), `pressure.py`
(projection), `solver.py` (driver), `selfcheck.py`.

## Scheme (a 3D port of vfx/sim 0.2.0, same physics, same fixes)
1. **Advect** all 6 channels by uⁿ: MacCormack, RK2 traces, clamped to the 8 source values
   (falls back to semi-Lagrangian where it would overshoot).
2. **Density mass correction**: a global multiplicative rescale back to the pre-advection fluid
   total. It uses a deterministic block-tile reduction with no atomics and no host sync.
3. **Sources**: emitters (falloff `(1-r²/R²)²`), impulses (Gaussian), combustion (100 K ignition
   ramp, heat release saturating at `max_temperature`), exponential cooling and dissipation. Then
   **heat + fuel diffusion** (explicit 7-point, zero flux at walls and solids, λ capped at 0.16; 1 substep up to 256³).
4. **Forces**: buoyancy; vorticity confinement `ε·dx·(N×ω_s)` with ω_s the [1,2,1]³-smoothed
   vorticity and the floored N; wind weighted by the gas marker. The summed force is applied
   [1,2,1]³-smoothed. These are the 2D 0.2.0 anti-speckle fixes: smoothed confinement and force, plus heat diffusion.
   The smoothing runs as 3 separable passes of `B(w·f)/B(w)`, which is the same as the 27-tap version.
5. **Projection** is exact for the face-averaged central divergence that `divergence()` reports. It solves
   `K φ = -d(u*)` with K = d dᵀ, the 2dx "wide" Laplacian. PCG with a V(1,1) preconditioner in a 2×2×2
   Hadamard basis: 8 components, which block-diagonalises K exactly in a box. Level 1 is probed exactly
   from K; coarser levels are Galerkin with constant prolongation (scale 1/16); smoothing is red-black GS.
   There is one damped-Jacobi sweep (ω = 0.8) on the exact K before and after. The loop is a
   `wp.capture_while` conditional, so there is no host sync. It stops at `max|div|·dx/max|u*| < pressure_tol`,
   at `pressure_max_iters`, at a 1e-6 float32 floor, or on stagnation. φ is warm-started from the previous call.
   Solid cells get u = 0, d = 0, T = ambient, fuel = 0.

The whole step is one CUDA graph per (data, solid) buffer pair (LRU of 4). Per-step inputs (dt, forces,
config) go through a pinned 4-slot host ring into a device buffer, so config edits take effect
without recapture. Masks and MG operators are rebuilt only when `solid` changes: a GPU-side compare
plus `wp.capture_if`. A step makes no allocations. No atomics anywhere, so reruns are bit-identical.

## Defaults
Same values as 2D: `buoyancy_alpha 0.6, buoyancy_beta 0.004, vorticity_eps 6, density_dissipation 0.05,
temperature_cooling 4, burn_rate 8, ignition_temp 550, burn_heat 4000, smoke_from_fuel 0.35,
pressure_tol 2e-5, pressure_max_iters 40, max_temperature 3000, wind_marker_temp 300,
heat_diffusion 5e-5, confinement_floor 0.5`, plus `pressure_warm_start True`.
**Runtime quality** `runtime_config()`: `pressure_tol 0, pressure_max_iters 4`. This is a fixed
4 PCG iterations, so the cost is constant.

## Measured (RTX 5080, CUDA events; GPU SHARED: another job held 93-99 % utilization during most runs)
`selfcheck --bench`: median of 60 steps after a 60-step warm-up; the stage split is the median of 30
`step_profiled` calls. Divergence is the normalized max at the end. The VRAM column is the bytes held by
the solver's own buffers plus the state.

| res | quality | scene | ms/step | advection | forces | projection | PCG iters | div | VRAM |
|---|---|---|---|---|---|---|---|---|---|
| 64³ | full | campfire | **0.88** | 0.045 | 0.070 | 0.77 | 5 | 6.6e-6 | 54 MiB |
| 64³ | full | plume + sphere | **1.01** | 0.045 | 0.068 | 0.89 | 6 | 8.7e-6 | 54 MiB |
| 64³ | runtime | campfire | 0.76 | 0.045 | 0.069 | 0.64 | 4 | 2.8e-5 | 54 MiB |
| 64³ | runtime | plume + sphere | 0.76 | 0.046 | 0.069 | 0.65 | 4 | **1.7e-4** | 54 MiB |
| 96³ | full | campfire / plume | 2.29 / 2.61 | 0.14 | 0.20 | 1.96 / 2.27 | 5 / 6 | ≤1.7e-5 | 182 MiB |
| 96³ | runtime | campfire / plume | 1.98 / 1.97 | 0.14 | 0.20 | 1.64 | 4 | ≤1.3e-4 | 182 MiB |
| 128³ | full | campfire / plume | 4.85 / 5.75 | 0.40 | 0.68 | 3.37 / 4.70 | 6 / 7 | ≤1.5e-5 | 431 MiB |
| 128³ | runtime | campfire / plume | 3.90 / 3.91 | 0.40 | 0.68 | 2.82 | 4 | ≤6.8e-4 | 431 MiB |
| 256³ | full | campfire / plume | 54 / 72 | 3.4 | 7.9 | 43 / 60 | 6 / 8 | ≤1.7e-5 | 3.4 GiB |
| 256³ | runtime | campfire / plume | 42 / 42 | 3.4 | 7.9 | 31 | 4 | ≤3.0e-3 | 3.4 GiB |

At 64³ the projection takes 75-88 % of the step. It costs about **0.125 ms per PCG iteration**, plus about
0.15 ms fixed (rhs, the initial V-cycle, apply, the solid-change check). One iteration is about 25
dependent launches. At 64³ every kernel is latency- or launch-bound: about 2-3 µs is the floor for a dependent launch in a graph,
and 6-12 µs for the full-grid kernels. At 128³ and above the kernels are bandwidth-bound.

**Divergence gate (< 1e-4) inside 2 ms: yes at full quality** (0.88-1.01 ms at 64³). The runtime setting
saves about 0.25 ms but misses the gate next to obstacles (plume + sphere 1.7e-4; 5.5e-4 in the
stress run). Without obstacles it meets the gate (campfire 2.8e-5). Divergence by fixed iteration count at 64³ with warm start
(campfire / plume + sphere): 2 its 1.2e-3 / 1.4e-2 (0.51 ms); 3 its 2.6e-4 / 4.4e-3 (0.64 ms);
4 its 5.8e-5 / 1.3e-3 (0.76 ms). Adaptive `pressure_tol=1e-4`: 4 / 6 its, 0.76 / 1.02 ms (p10).

### What each optimization bought (64³ plume + sphere unless noted; interleaved A/B, min of 200 steps)
Every version, including the baseline, ran as a single graph with no per-step allocations. The pressure loop ran as a graph conditional,
with fused dot products and max reductions (per-thread values reduced as a block tile).
| change | effect |
|---|---|
| launch-reduced V-cycle (half-size red/black launches, zero-guess red sweep fused into transform/restriction, prolongation fused into the black sweep, m ≤ 2 levels in one single-block kernel) | 1.20 → 1.06 ms at 6 fixed iters (−12 %); 128³ 5.68 → 4.99 ms |
| warm-started φ | −1 PCG iteration (plume 7.9 → 6.8, campfire 5.9 → 5.0 mean): plume 1.46 → 1.32 ms p10 |
| outer Jacobi ω 0.6 → 0.8 (swept 0-0.9; ω = 0 doubles iterations with obstacles) | plume 7.2 → 6.0 iters, 1.14 → 1.01 ms p10 |
| fused α finalize into the Jacobi-front kernel | −1 launch per iteration (not separately measured) |
| MG setup only when `solid` changes (GPU compare + `capture_if`) | removes the probe and Galerkin launches from every step |
| separable [1,2,1]³ smoothing | per kernel 22 → 19 µs at 64³ and 140 → 90 µs at 128³; **no measurable end-to-end change** |
| tried and reverted: serial per-component coarse solve (33 µs), single-block kernel from m ≤ 8 (67 µs vs about 20 µs of launches), Jacobi residual computed on the fly (+0.1 ms), 1 instead of 2 cells/thread | slower |

Headroom at 64³: about 1.0 ms of the 2.0 ms budget at full quality, and 1.25 ms in runtime mode. Further gains would need fewer
dependent launches per iteration. Warp exposes no block barrier other than tile ops, and one block has
too little latency hiding to run the m = 8 level. Fewer iterations would need a stronger preconditioner per launch.

## Self-check (`python -m vfx.sim3d.selfcheck --bench --out DIR`)
Projection of a smoothed random field: div 2.3e-5 / 1.1e-5 (64³, without / with sphere), 1.3e-5 / 1.8e-5 (128³).
Mass over 1000 steps in a closed box with a sphere: +0.00016 % (64³), −0.00001 % (128³). Hot blob heat centroid
0.300 → 0.346 m after 1 s. Solid cells exactly 0 / ambient. Bit-identical reruns (64³, 128³, with
impulses). 1000-step stress run (4× fuel, a 1500 K fire, an extra 1200 K jet, an impulse of up to 6 m/s every 7 steps, rotating wind, sphere + box): finite,
max|u| 3.7 m/s, Tmax 2664 K, div 1.0e-5 (full quality) / 5.5e-4 (runtime).
At step 150 (5 s), 64³ and 128³ agree on the budgets: smoke mass within 6 %, heat within 5 % (plume) and 15 % (campfire),
and the heat centroid within 0.02 m. But 64³ plumes rise more slowly: smoke centroid 0.40 vs 0.58 m for the plume and 0.68 vs
0.76 m for the campfire. A 0.06 m emitter is only about 4 cells at 64³, so numerical
diffusion of momentum is large there.

## Caveats
- Collocated wide-stencil projection: odd-even velocity modes are invisible to the gated divergence, as in 2D.
  The smoothed forces keep them small.
- Emitter temperature is stamped once per step. A fast jet (u·dt ≈ 5 cells at 128³) shows faint bands
  spaced u·dt apart just above the emitter, as in 2D.
- `advect()` does not see obstacles and has no mass correction. Obstacles are read from `state.solid` every step.
- φ persists in the solver between `step`/`project` calls (warm start). It only changes the initial guess.
- Grid: cubic, even size; powers of two use the fast V-cycle (96³ works too, with an odd coarsest level of 3).
  At most 32 emitters and 32 impulses.
