# vfx/sim — 2D smoke + fire solver (Warp)

Interface exactly as in `CLAUDE.md`: `Solver`, `SolverConfig`, `SOLVER_VERSION`
(`reset / step / divergence / advect / project`). `FieldState.data` stays a torch cuda tensor;
Warp views it zero-copy (`wp.from_torch`). Files: `config.py` (params), `kernels.py`
(advection, sources, forces), `pressure.py` (projection), `solver.py` (driver), `selfcheck.py`.

## Step (`step()` ends divergence-free)
1. **Advect** all 5 channels by uⁿ: **MacCormack** (Selle et al. 2008) with RK2 traces and
   min/max clamping to the 4 source values (falls back to semi-Lagrangian where clamped).
2. **Density mass correction**: global multiplicative renormalisation restoring the
   pre-advection fluid-cell total (deterministic tile reduction, no atomics, no host sync).
   SL/MacCormack are not conservative; the loss happens mostly at walls / obstacles, so
   a global correction is the cheapest scheme that meets the 0.1 % gate exactly (measured drift
   ≤ 6e-6 relative over 1000 steps). Fuel and heat are not renormalised.
3. **Sources**: emitters (smooth falloff `(1-r²/R²)²`; density/fuel rates per second;
   temperature = `max(T, T_amb + (T_e - T_amb)·w)`; velocity blended toward the emitter velocity
   only if it is nonzero), impulses (Gaussian `exp(-r²/R²)`), **combustion** (first order,
   `burn_rate` /s above `ignition_temp` with a 100 K ramp; heat `burn_heat` per fuel unit,
   scaled by headroom to `max_temperature` so a stagnant fed cell cannot run away; smoke
   `smoke_from_fuel`), exponential cooling and dissipation (`exp(-k dt)`, dt-consistent).
   Then **heat + fuel diffusion** (`heat_diffusion` m²/s, Lewis number 1, explicit, zero flux at
   walls/solids, `2·ceil(N/256)` substeps, per-substep λ capped at 0.24).
4. **Forces**: buoyancy `f_y = -α·d + β·(T - T_amb)`; vorticity confinement `ε·dx·(N×ω_s)`
   (`ε` in 1/s, the dx factor keeps the per-step effect on grid-scale vortices resolution
   independent) with ω_s = [1,2,1]²-smoothed vorticity and
   `N = ∇|ω_s| / sqrt(|∇|ω_s||² + (confinement_floor·|ω_s|)²)`; wind (see interface notes).
   The summed force is applied [1,2,1]²-smoothed (fluid-masked), i.e. resolved over ≥ 2 cells.
5. **Projection** (below). Solid cells get u = 0, d = 0, T = ambient, fuel = 0.

## Projection (exact for the collocated divergence)
`divergence()` uses face velocity = mean of the two adjacent cells (0 on walls and solid faces),
i.e. central differences. A MAC projection followed by averaging back to cell centres does NOT
make that zero, so the solver projects exactly against it: solve `K φ = -d(u*)`, `K = d dᵀ`
(the 2dx "wide" Laplacian), `u = u* - g(φ)` with `g = -dᵀ`. Solver: **PCG** with a multigrid
preconditioner. A 2x2-block Hadamard transform block-diagonalises K *exactly* in a box (sum
component: Neumann 5-point Laplacian; parity-difference components: Dirichlet along that axis).
Level-1 stencils are probed exactly from K (component-diagonal part); coarser levels are
Galerkin with constant prolongation ×½; red-black GS V(1,1); plus one damped-Jacobi sweep on the
exact K before/after (fixes obstacle boundaries). The loop is a CUDA-graph conditional
(`wp.capture_while`), so there is no host sync: it stops at `max|div|·dx/max|u*| < pressure_tol`,
after `pressure_max_iters`, at a 1e-6 float32 floor, or when the residual climbs 4× above its
best. Iterations do not grow with resolution: 5–7 in a box, 7–13 with obstacles (20 worst case,
white-noise velocity with a thin wall).

## Determinism / performance
No atomics; all reductions are block-tile reductions with a fixed order, so outputs are bit-identical.
The whole step is captured once per (`state.data`, `state.solid`) buffer pair as a CUDA graph
(LRU of 4). Per-step inputs (dt, forces, the config itself) go through a pinned 4-slot host
ring into a device buffer, so config edits take effect without recapture. Max 32 emitters / 32 impulses.

## Defaults (`SolverConfig`)
`buoyancy_alpha 0.6`, `buoyancy_beta 0.004`, `vorticity_eps 6`, `density_dissipation 0.05`,
`temperature_cooling 4`, `burn_rate 8`, `ignition_temp 550`, `burn_heat 4000`,
`smoke_from_fuel 0.35`, `pressure_tol 2e-5` (5× margin under the 1e-4 gate), `pressure_max_iters 40`,
plus `max_temperature 3000`, `wind_marker_temp 300`, `heat_diffusion 5e-5`, `confinement_floor 0.5`.

## 0.2.0: saw-tooth / speckle fix (fire at 512²)
Cause (isolated by ablation, campfire + app preset 1, 200 steps): (a) combustion is bistable and
had no diffusion, so the ignition front was a 1-cell temperature step (hundreds of K of heat release
per step; widening the ignition ramp to 400–1000 K did not help) — with burning off, or buoyancy
off, the fields are smooth; (b) vorticity confinement pumped grid-scale velocity (unit N flips sign
on every |ω| ridge and points in a noise-driven direction where ∇|ω| ≈ 0, at full strength);
(c) the collocated wide-stencil projection answers grid-scale divergence of such forcing with
row/column-alternating velocity that central curl/divergence cannot see (stair-steps; walls and
obstacles also emit it). MacCormack's limiter was a minor contributor (pure semi-Lagrangian was
still saw-toothed). Fix: heat/fuel diffusion, smoothed + floored confinement, smoothed force.
Interior (12-cell margin) power fraction in the top quarter band (max(|kx|,|ky|) ≥ 0.75 k_Nyq),
mean over steps 120–200, ×1e-3, 512² campfire / preset 1: density 6.7→2.7 / 13.1→6.1,
temperature 4.4→0.01 / 10.2→0.05, u_x 2.9→2.3 / 4.8→3.4, u_y 0.45→0.16 / 1.34→0.52.
128²: density 16.0→7.0 / 11.4→6.5, T 12.5→7.0 / 11.3→6.6, u_x 12.3→6.8 / 11.5→6.4.
Residual: MacCormack keeps 1-cell smoke fronts, a faint 1-cell seam can appear on a symmetric
convergence line (preset 1 centre), and walls still emit weak odd-even velocity in the first ~10 cells.

## Measured (RTX 5080, CUDA events, 200-step average after warm-up)
| scene | 128² ms/step | 512² ms/step |
|---|---|---|
| plume, no obstacle | 0.59 | 1.18 |
| plume + circle | 0.80 | 1.80 |
| plume + thin wall + circle, with impulse | 1.08 | 2.45 |
| selfcheck campfire / plume-past-cylinder | 0.57 / 0.75 (median) | 1.06 / 1.85 (median) |
| same, 0.2.0 | 0.58 / 0.85 (median) | 1.37 / 2.18 (median) |

Selfcheck: normalized div 4e-6 to 1.3e-5; mass drift over 1000 steps −0.0004 % (128²) and
+0.0006 % (512²); with `density_dissipation=0.5` the mass follows `exp(-k t)` within 0.002 %;
bit-identical reruns. The 128 vs 512 comparison of the same scenario gives smoke mass within 1 %,
heat within 3 %, smoke and heat centroid heights within 0.03 m. Pointwise blurred L2 diverges
after about 2 s (0.17 to 0.44 at step 50, about 0.5 later) because buoyant plumes are chaotic.
`tests/test_sim_layer1.py`: 30/30 pass.
0.2.0 selfcheck: campfire LR-vs-pooled-HR blurred L2 0.68 / 0.79 / 0.60 at steps 50/100/150 (0.1.0: 0.44 /
0.61 / 0.52; chaotic, one run), budgets still agree (smoke mass < 1 %, heat < 5 %, centroids < 0.03 m);
mass drift +0.0002 % (both res); bit-identical reruns; Layer 1 30/30.

## Interface notes / caveats
- **Wind**: a uniform acceleration in a closed box is a pure gradient, so the projection would
  cancel it exactly. It is applied weighted by a gas marker `clamp(d + fuel + ΔT/300, 0, 1)`.
- `advect()` is the MacCormack scheme without the density renormalisation (that step is
  density-only, in `step()`). It does not see obstacles.
- Obstacles are taken from `state.solid` every step, not from `reset()`. The MG operators are rebuilt inside the graph.
- Collocated wide-stencil projection leaves odd-even velocity modes (zero central divergence)
  untouched and answers grid-scale divergence with them; 0.2.0 keeps the forcing smooth at grid
  scale instead. (Any projection orthogonal w.r.t. the gated central divergence returns the same
  field, so adding a compact-Laplacian pre-stage changes nothing — tested.)
- Grid must be square with even size (power of two recommended for the MG hierarchy).
