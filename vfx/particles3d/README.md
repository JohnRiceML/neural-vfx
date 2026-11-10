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

## Parameters (`DEFAULTS` in particles.py)
See the table in particles.py; the ones that matter: `count`, `out_res`, `sg_amp`, `sg_freq`, `n_full`, `strength`,
`blur`.

## Results (interim — GPU runs on hold at the lead's request)
First look, preset 2 (plume past a sphere), 120 ticks, 1M particles, sg_amp 0.6, no reference panel yet:
* Mass: particle smoke mass / grid mass 0.89-1.04 over 150 ticks (presets 1 and 2); ~0.49-0.55M of the 1M ring alive.
* Restriction of the composed volume to 64³ vs the solver density: rel L1 0.25 (part of it is the trilinear
  upsample itself; to be measured separately).
* Look: the particle volumes add sub-grid texture the 64³ and trilinear renders lack — small curled clumps and
  crisper edges at 128³. 192³ with the same 1M particles is visibly grainier (fewer particles per fine cell).
  The texture reads more as fine turbulent clumping than as long thin filaments so far. Lighting and
  self-shadowing come out consistent because the renderer is unchanged.
* Timings so far are meaningless: the GPU was at 99-100 % from other jobs (ms inflated 10-60x).

Pending: steered 256³ reference per preset (rendered frames cached on D:), parameter sweep, temporal stability,
spectra, cost table, selfcheck.
