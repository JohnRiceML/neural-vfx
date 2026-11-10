# tests3d/ + bench3d/ — 3D phase gate tests and feasibility benchmark (owner: 3D test/eval engineer)

```
.venv/Scripts/python -m pytest tests3d -q              # ~1 min; modules not landed yet SKIP with a reason
.venv/Scripts/python -m pytest tests3d -q -s -rs       # with the measured numbers
.venv/Scripts/python -m bench3d.run [--update-baseline] [--quick] [--strict]
```

Written from SPEC.md, the CLAUDE.md "3D phase" interface and `vfx/core/fields3d.py`, never from the
implementations. Spec thresholds are never relaxed; where the spec gives no number, the tolerance is
stated in the test module docstring. Each test only needs the module it tests (runtime tests build
`FieldState3D` directly, so they run without `vfx.sim3d`).

| File | What it checks |
|---|---|
| `test_sim3d_layer1.py` | `vfx.sim3d` public API, 64³. Reset contract (shape, ambient, walls not in `solid`, `SolverConfig3D` field names incl. `heat_diffusion`, `confinement_floor`), obstacles rasterized with `rasterize_obstacles_3d`. Normalized divergence `< 1e-4` after `project()` (default config and tight solve, open and sphere) and after 60 full `step()`s; pure-gradient flow into the walls removed (< 1% residual, zero net slab flux in x/y/z). Mass within 0.1% over 1000 steps (closed box, `density_dissipation=0`, with and without a sphere) and under wall/corner jets. `advect()`: whole-cell shifts exact in the Lagrangian interior, fractional translation vs analytic (centroid < 0.5 cell, rel L2 < 0.15), multi-channel consistency, error shrinks 32→48→64→96. Hot parcel rises (+y) with no sideways drift (|vx|,|vz| < 10% of vy); dense cold parcel sinks; ambient stays at rest. Nothing enters box/sphere obstacles. Bit-identical determinism (120 steps, fire, impulses, wind, sphere). Plume convergence 32/48/64 vs a 96³ reference. 64³ vs 128³ large-scale agreement (smoke mass, heat within 10%, centroid height within 0.04 m, plume top within 0.0625 m). 1000-step stress (strong fire, impulses every 5 steps, two obstacles): finite and bounded. |
| `test_render3d.py` | `VolumeRenderer`: `(out_h, out_w, 3)` uint8 cuda for `out_size=(h, w)` (non-square), `timings` dict, any volume resolution and fp16/fp32, `solid` accepted. Empty volume invisible (independent of its resolution, no brightness step at the domain-box edges); a compact smoke blob changes only its own screen footprint (ground-plane pixels excluded: the renderer draws a floor that receives the smoke's shadow). More density → more deviation from background; hot smoke brighter and warmer (R−B) than cold; orbiting the camera changes the image, same camera is deterministic; `march_scale` 0.5 / 0.25 PSNR vs 1.0 ≥ 30 / 25 dB; extreme values render validly; no torch-visible host sync in `draw`. Note: `draw()` returns a reused buffer — callers must copy if they keep a frame. |
| `test_runtime3d.py` | `classical_upscale_3d`: affine field exact in the interior (×4, ×2), default = 4× trilinear (vs `F.interpolate`), no host sync. `VolumeUpscaler` (untrained): 256³ `Volume`, finite, density ≥ 0, `timings`; empty state → empty volume; far field (> 0.28 m from smoke) untouched; sparse == dense in and around active bricks when the instance exposes a bool `sparse`/`dense`/`skip_empty`/`use_bricks` switch; seeded determinism; no host sync in `infer`. `ScreenDetail` (untrained): shape/dtype at 1440p, 720p and an odd size, `timings`, seeded determinism, no host sync. |

"No host sync" uses `torch.cuda.set_sync_debug_mode("warn")`: it sees torch-initiated syncs and H2D/D2H
copies only, not Warp's own; it is the "where feasible" check.

## bench3d (the 3D-phase deliverable)
`bench3d/run.py` (see its docstring) measures complete pipelines — raw 64³, 64³→trilinear 256³, route A
(VolumeUpscaler), route B (ScreenDetail), brute-force 128³ and 256³, plus full-quality-config step-only
lines — on two fixed scenes (campfire; plume past a sphere) at 2560×1440 with `march_scale` 0.5 and 1.0:
per-stage and whole-frame CUDA-event times (mean/p50/p95, hitches > 50 ms), host wall time per frame,
peak VRAM (torch + device-level), occupancy, and speed-up vs brute-force 256³. Each (scene, pipeline)
runs in its own subprocess. nvidia-smi is read first; a shared GPU is recorded and loudly flagged.
Outputs `bench3d/results/latest.json` and `bench3d/results/REPORT.md`; `--update-baseline` writes
`bench3d/baseline.json`; a later run fails on any p50 / VRAM regression > 10% (not gated on a shared GPU
unless `--strict`).

## Deferred
- Layer 2 physics benchmarks in 3D (Taylor–Green, lid-driven cavity vs Ghia, vortex rings), PhiFlow cross-check.
- Quality gates for routes A/B (spectrum error vs tricubic, flicker, OOD): no 3D training in this phase.
  The 3D "tricubic" baseline itself is not implemented yet.
- 256³ Layer 1 runs (all Layer 1 tests are 64³, plus 96³/128³ references) and 10,000-frame 3D rollouts.
- Renderer quality vs a reference renderer, LPIPS, Layer 6 visual review page; `vfx.app3d` sandbox is not tested.
- Warp-internal host syncs (not visible to torch's sync debug mode); CUDA-graph capture of the whole frame.
- Idle-GPU numbers: the lead re-runs `bench3d` on an idle GPU.
