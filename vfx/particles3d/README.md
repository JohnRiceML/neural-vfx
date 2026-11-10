# vfx/particles3d — sub-grid smoke detail carried on particles (feasibility study)

```python
from vfx.particles3d import DetailParticles
dp = DetailParticles(count=1_000_000, out_res=128, config=runtime_config(**preset.sim))   # same SolverConfig3D as the sim
dp.step(state, forces, SIM_DT)       # after solver.step: seed at emitters, advect, age, cull, deposit
renderer.draw(dp.volume(), cam, solid)   # Volume(density, temperature) at out_res^3, existing renderer unchanged
dp.timings                           # {'advect+deposit', 'deposit', 'compose', 'total'} ms (CUDA events, resolved lazily)
```
Scripts: `python -m vfx.particles3d.compare --preset K --frames N --out DIR` (side-by-side vs the steered 256³
reference), `python -m vfx.particles3d.bench --out DIR` (cost table), `python -m vfx.particles3d.selfcheck`.
Files: `kernels.py` (Warp), `particles.py` (driver), `compare.py`, `bench.py`, `selfcheck.py`.

## Method (one sim tick, all GPU, no host sync, ~7 launches)
1. **Velocity grid** (`vel_res`³, default 128): trilinear 64³ velocity + a sub-grid curl-noise velocity
   `u_sg = sg_amp · |∇×u|₆₄ · dx₆₄ · curlnoise(x·f₀, t·f_t) / rms`, 3 octaves, lacunarity 2, gain 2^(-1/3)
   (Kolmogorov: velocity at scale ℓ ∝ ℓ^(1/3)), f₀ = 1/(2 dx₆₄) (the finest scale the grid cannot carry), capped
   at `sg_max`. This is the "simplest thing" of the brief: a Smagorinsky-style sub-grid speed (|ω|·Δ) driving a
   divergence-free noise field, in the spirit of Pfaff et al.'s turbulence particles / Bridson's curl noise, without
   their energy transport. Evaluating the noise once per grid cell (not per particle and RK stage) makes the
   noise cost independent of the particle count. It also kept Warp's compile time sane: per-particle 4D curl noise
   inlined in RK stages plus Warp's default adjoint code took minutes to compile, so the module is forward-only.
2. **Advect** every live particle (RK2 midpoint, optional Ralston RK3) through that field. Particles that would
   enter a solid cell of the sim grid stay put (the wall velocity is 0 anyway); a particle that starts in a solid is killed.
3. **Payload** per particle: smoke mass and fuel mass (extensive), fuel concentration and temperature (intensive).
   They follow the solver's per-cell source ODE along the path (`sim3d/kernels.sources`): the emitter temperature
   stamp `T = max(T, T_e w)`, first-order combustion above ignition with the 100 K ramp and saturating heat release
   (`burn_heat` × burned concentration × room), smoke from fuel (`smoke_from_fuel` × burned fuel mass),
   exponential cooling and dissipation, with the rates of the SolverConfig3D the sim uses. Total particle smoke mass
   therefore tracks the grid's (heat and fuel diffusion are not modelled on particles).
4. **Seed** into a fixed-capacity ring: `k = count · dt / life` new particles per tick, `life = ln(1/life_eps) /
   density_dissipation` (≤ `max_life`), so the oldest slot is reused exactly when its smoke has decayed to 2 %.
   Each emitter gets new particles in proportion to the smoke it injects (density + smoke_from_fuel · fuel, × the
   discrete integral of its falloff), positions drawn with pdf ∝ (1-r²/R²)² so equal-mass particles reproduce the
   injected profile; temperature and fuel concentration are read from the grid at the seed point; each new particle
   is advanced by a random fraction of the tick (no seeding bands).
5. **Deposit** (fused into 2-4): trilinear (tent) scatter with float atomics of smoke mass, heat (T-T_amb) and count
   into `out_res`³ grids.
6. **Compose** (`out_res`³): `ρ = up(G) · (1 - b + b · clamp(P / up(R(P)), mod_max))`, same for heat with H.
   G = the solver's 64³ density, up = trilinear, R = box restriction to 64³. P / up(R(P)) is the particle detail
   as a modulation with mean ≈ 1 per sim cell, so the restriction of the volume stays close to the solver's density
   (measured below), and b = strength · clamp(particles per fine cell / n_full) fades the modulation out where too
   few particles make it noisy, leaving the plain upsampled solver field (no holes). A first version divided by the
   per-cell ratio G / R(P) and interpolated that ratio: cells with few particles have huge ratios that leak into
   their neighbours (volume mass 2-3x the grid's). Dividing smooth by smooth fixed it.

**Determinism**: particle positions and payload are bit-identical on rerun (no atomics on the particle path, counter
RNG per slot and tick). The deposit uses float atomics, so the volume differs between runs at rounding level.

A particle that has faded below 1 % of its seed mass and is within 1 K of ambient is culled. The first version
used an absolute mass threshold, which killed the small particles of large rings early (a 4M ring had only 1.4M
alive). The threshold is now relative to the seed mass, and the whole ring stays alive (1.00M / 4.00M).

## Parameters (`DEFAULTS` in particles.py)
| name | default | meaning |
|---|---|---|
| count, out_res | (ctor) | ring capacity; deposit grid (multiple of 64) |
| sg_amp | 0.6 | sub-grid speed / (curl magnitude × dx), at 64³. 0 = pure coarse advection |
| sg_freq, sg_oct, sg_lac, sg_gain | 1/(2dx), 3, 2, 2^(-1/3) | curl-noise octaves (Kolmogorov falloff) |
| sg_tfreq, sg_max | 1.5 /s, 0.6 m/s | noise evolution rate; speed cap |
| vel_res | 128 | grid the per-tick particle velocity is evaluated on |
| rk3 | False | RK2 midpoint (default) or Ralston RK3 |
| life_eps, max_life | 0.02, 6 s | ring lifetime (smoke decayed to 2 %) |
| n_full | 4 | particles per fine cell (averaged over a sim cell) for full modulation weight |
| strength, mod_max | 1.0, 8 | density modulation weight / cap |
| heat_strength, heat_mod_max | 0.5, 2 | temperature modulation weight (× strength) / cap. Emission is very nonlinear in T |
| blur | 0 | [1,2,1]³ passes on the deposit (tested: removes the detail along with the grain; not useful) |
| fused | True | deposit inside the advect kernel; False = separate kernels (stage timing) |

Recommended: **128³, 1M particles, defaults**. For a sharper look: **192³, 4M particles, strength 0.6**.

## Results (2026-11-09; GPU shared at 85-98 % by training jobs throughout, so no timings are reported here)
Script: `compare --frames 16 --warmup 90 --orbit-deg 3 --dist 1.3 --size 480x540` (16 consecutive sim ticks with a
45° orbit). The reference is the steered 256³ from `vfx.datagen3d.sim.run_pair`, cached as rendered frames under
`D:/neural-vfx-data/scratch/particles3d`. Metrics come from `vfx.eval3d.metrics`, on luminance in the reference's
effect mask, over the 16 frames:
* spec err: mean abs log10 P/P_ref of the radial spectrum (decades; 0 = matches the reference)
* hf: power above 1/16 cycles/px relative to the reference (1 = the same amount of fine detail)
* tLP: frame-to-frame change relative to the reference's (0 = changes as much as the reference); the log ratio is
  < 0 for too static and > 0 for boiling
* pop: bursts of non-motion change (1 = like the reference)

| preset | metric | raw 64³ | trilinear 256³ | advected noise (detail3d medium) | **particles 128³ 1M** | **particles 192³ 4M str 0.6** |
|---|---|---|---|---|---|---|
| 1 campfire | spec err / hf | 0.54 / 0.31 | 0.60 / 0.26 | 0.50 / 2.07 | 0.25 / 1.13 | **0.19** / 1.04 |
| | tLP (log ratio) / pop | 0.42 (-0.55) / 1.03 | 0.44 / 1.02 | 0.43 (+0.36) / 1.03 | **0.05** (-0.05) / 0.99 | 0.10 (-0.11) / 1.00 |
| 2 plume + sphere | spec err / hf | 0.13 / 0.44 | 0.14 / 0.41 | 0.15 / 1.17 | 0.04 / 0.99 | **0.017** / 0.94 |
| | tLP (log ratio) / pop | 0.39 (-0.50) / 1.01 | 0.41 / 1.02 | **0.06** (-0.05) / 1.04 | 0.12 (-0.13) / 0.99 | 0.13 (-0.14) / 0.99 |
| 3 twin torches | spec err / hf | 0.36 / 0.22 | 0.38 / 0.19 | 0.18 / 1.08 | 0.16 / 0.99 | **0.11** / 0.85 |
| | tLP (log ratio) / pop | 0.37 (-0.46) / 0.96 | 0.39 / 0.97 | 0.12 (-0.12) / 0.97 | **0.08** (-0.08) / 0.96 | 0.12 (-0.12) / 0.96 |
| 4 box obstacle | spec err / hf | 0.21 / 0.35 | 0.23 / 0.32 | 0.22 / 1.38 | 0.09 / 0.90 | **0.066** / 0.84 |
| | tLP (log ratio) / pop | 0.32 (-0.38) / 1.03 | 0.33 / 1.04 | **0.02** (-0.02) / 1.03 | 0.09 (-0.09) / 1.02 | 0.10 (-0.11) / 1.02 |

Mass and restriction (end of each run): particle smoke mass / grid mass is 1.04 (smoke plume), 0.96 (campfire),
0.91 (torches) and 1.00 (box). On fire presets part of the smoke is still unburnt fuel on the particles; the grid
burns slightly earlier because heat diffuses there. The restriction of the volume to 64³ differs from the solver
density by rel L1 0.21-0.32. **Most of that is the trilinear upsample itself**: the same measure for plain trilinear
is 0.19-0.26, and the particle modulation adds only +0.02 to +0.06.

### Sweep (preset 2, `p2_sweep1` and `p2_sweep2`, same reference)
| variant | spec err | hf | tLP log ratio | look |
|---|---|---|---|---|
| 0.25M 128³ (with the old cull bug) | 0.091 | 0.78 | -0.31 | coarser clumps, fades to soft grid early |
| 1M 128³, sg 0 (no sub-grid; old cull bug) | 0.084 | 0.83 | -0.34 | already textured: Lagrangian clumping + shot noise |
| 1M 128³, sg 0.6 | **0.057** | 1.11 | -0.12 | billowy texture, not grainy |
| 1M 128³, sg 1.2 (old cull bug) | 0.071 | 0.92 | -0.23 | same; stronger noise does not make filaments |
| 4M 128³ | 0.053 | 1.18 | -0.10 | indistinguishable from 1M: 128³ is not particle-limited at 1M |
| 1M 192³ | 0.071 | 1.60 | -0.01 | grainy speckle (too few particles per 192³ cell) |
| 4M 192³ | 0.137 | 2.28 | +0.19 | crisp but over-textured ("sandy"), slight boiling |
| **4M 192³ strength 0.6** | **0.024** | 1.03 | -0.16 | sharpest believable result |
| 4M 192³ blur 1 | 0.089 | 1.03 | -0.21 | blur removes the grain and the detail with it |
| 1M 128³ n_full 8 (old cull bug) | 0.083 | 0.86 | -0.34 | more grid fallback, softer |

Particle count for a non-grainy result: about **1M at 128³** (≈ 4-8 particles per occupied fine cell for these
plume sizes; 0.25M is visibly coarser) and **≥ 4M at 192³** (with strength 0.6).

### What the pictures show (`frame_*.png`, `orbit_and_consecutive.png`, `spectrum.png` per run)
* **Wisps / filaments: no.** Particles add a billowy, clumpy sub-grid texture (cauliflower / cotton) that is close
  in spectrum and in the look of *dense* regions to the 256³ reference. They do not produce the long, thin, curling
  sheets and strands of the reference. Stronger curl noise did not change that. Pure coarse advection
  (sg 0) already gives most of the texture: it is particle clustering in the interpolated 64³ flow (the
  collocated, wide-stencil-projected velocity is not divergence-free at sub-cell scale) plus Monte-Carlo noise,
  not resolved filament stretching.
* **Grain**: none at 128³ / 1M. Visible speckle at 192³ unless there are ≥ 4M particles *and* the modulation is
  damped (strength 0.6).
* **Temporal stability**: good. In 3+ consecutive frames the texture moves with the flow and does not boil or
  pop (pop 0.96-1.02, tLP log ratio about -0.1: slightly calmer than the reference). It is volumetric, so it is
  orbit-consistent by construction.
* **Lighting / self-shadowing**: correct. The renderer is unchanged and the light volume sees the detailed density.
* **Fire: weak.** With full temperature modulation, flames break into glowing beads/sparks and run too high
  (blackbody emission amplifies the particle noise; `p1_heat`). This is an "embers" look, possibly useful as an
  effect. With the default heat_strength 0.5 / cap 2, flames stay mostly whole but are still mottled compared
  with the reference's smooth tongues. Particles add nothing good to flames; they help the smoke above them.
* **vs advected noise (vfx/detail3d)**: complementary. Advected noise gives the thin strands and crisp folds on
  thin smoke (torches, the plume's top), but makes dense columns stringy/vertically streaked. Particles give the
  right billowy, dense-column texture (box obstacle, plume, campfire smoke) and the best spectra on all four
  presets, but no wisps and worse flames. A plausible combination: particle volume for the smoke density plus
  advected-noise displacement at render time on thin, low-density regions. `draw(dp.volume(), ..., detail=det)`
  should work as is, since the detail samples any volume, but it is untested.

### Cost
Not measured yet: the lead will provide a quiet GPU window. Run `python -m vfx.particles3d.bench --out DIR`
(0.25M / 1M / 4M × 128³ / 192³: advect, deposit, compose, total per tick, render of the volume at 1440p,
VRAM). Per-panel ms printed in the frames were taken at 85-98 % foreign GPU load and are not budget numbers.
Memory: 28 B per particle (112 MB at 4M), plus about 6 grids of out_res³ float32 (50 MB at 128³, 170 MB at 192³).

### Integration into the sandbox (after review)
`app3d` would add a mode `particles`: construct `DetailParticles(1_000_000, 128, config=solver.config)`, call
`dp.step(state, forces, SIM_DT)` after `solver.step` (same forces), and draw `dp.volume()` instead of the raw volume.
On reset or preset change, call `dp.reset()` and set `dp.cfg` to the new SolverConfig3D. The step is about 7 launches
plus a tiny pinned H2D copy of the emitters, with no host sync. It could be captured in a CUDA graph like the
solver. The Volume buffers are reused every tick (the renderer copies them in `pack`).
