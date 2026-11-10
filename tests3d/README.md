# tests3d/ + bench3d/ — 3D phase gate tests and feasibility benchmark (owner: 3D test/eval engineer)

```
.venv/Scripts/python -m pytest tests3d -q              # ~1 min; modules not landed yet SKIP with a reason
.venv/Scripts/python -m pytest tests3d -q -s -rs       # with the measured numbers
.venv/Scripts/python -m bench3d.run [--update-baseline] [--quick] [--strict] [--scene-set closed|presets|all]
VFX_DATASET3D=s3d_v1 .venv/Scripts/python -m pytest tests3d/test_data3d_layer3.py -q -s -rs
VFX_SCREEN_CKPT=<path> .venv/Scripts/python -m pytest tests3d/test_screen_layer4.py -q -s -rs   # default: screen_checkpoint()
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
| `test_eval3d_metrics.py` | The route B metrics themselves (`vfx/eval3d/metrics.py`) on analytic synthetic images: identity → 0; L1/MS-DSSIM/spectrum blind to content outside the effect region; spectrum error monotone in blur and in noise, signed log-ratio < 0 for blur and > 0 for noise, shift-invariant, white noise of variance s² → power s² per bin, uint8 quantization costs ≤ log10(1.56); warp sign convention (content at p came from p − motion) exact for integer shifts; **known translation** → flicker ≈ 0 (integer) / < 0.1× unwarped (fractional); **known flicker injection** → flicker = 2s/√π; flicker ratio = 1 for the reference itself, > 1.2 with noise; warping with another sequence's motion is ≫ own motion; opacity weighting ignores un-compensated background; temporal jerk: 0 for fades, iid noise → 2s√(3/π); detail swim: glued detail = reference, screen-fixed or re-rolled detail ≫; unsharp; `run_sequence` is autoregressive (one reset, one infer per frame with the CLAUDE.md aux dict, outputs copied); `evaluate_screen_detail` oracle → 0 / ratio 1, identity == lr baseline; `tune_unsharp`; importing the metrics pulls in no 3D module. |
| `test_data3d_layer3.py` | Layer 3 for `$VFX_DATASET3D` (default `s3d_v1`) through `vfx.datagen3d.load_sequences`: size ≤ 4 GB, card fields, solver version current, seed splits disjoint (and items come from their split), every OOD scenario has a scene-type tag absent from train (split names excluded), item shapes/dtypes, aux finite / alpha ∈ [0,1] / depth and motion 0 where alpha = 0 / plausible ranges, per-sequence stats (not black/saturated, effect visible, not frozen, hr/lr luminance ratio) with 8-MAD outlier report (failures must be flagged in the card), **lr/hr alignment** of static geometry (best integer shift (0,0), median diff ≤ 1 code), **hr has ≥ 1.5× lr's fine-scale energy while tracking it at coarse scale** (blurred correlation ≥ 0.9; scales from the projected 64³ voxel size), **alignment table** (own estimated flow must reduce warp error ≤ 0.9× for lr and hr; stored motion reducing it is an xfail), and a sensitivity check of the temporal metrics on the real data (2-code iid noise injected into hr). |
| `test_screen_layer4.py` | Layer 4 for route B: overfit one batch on a fresh `ScreenNet`; baseline report (lr, tuned unsharp, hr reference numbers, no checkpoint needed); with the checkpoint (`vfx.eval3d.checkpoint.screen_checkpoint`): runtime reads the CLAUDE.md aux keys, beats lr AND unsharp on L1 / spectrum / MS-DSSIM (test), OOD relative-to-lr score within 20%, flicker ratio ≤ 1.2 and ≤ max(lr's, 1), temporal-jerk ratio ≤ 1.2, detail swim (moving cameras) ≤ 1.2× reference. **Live** (no dataset): 64³ `Solver3D` on the 4 sandbox presets, orbiting camera, 600 frames at 2560×1440 through `VolumeRenderer(return_aux)` + `ScreenDetail` — finite, no frame > 50 ms after 60 warm-up frames, mean detail ms ≤ 2.5, flicker and jerk vs the lr render. Reports in `DATA_ROOT/runs/eval3d/`. |

## Route B metrics (`vfx/eval3d/metrics.py`; exact formulas in its docstring)
All lower-is-better, on frames `(…,3,h,w)`, scored on the **effect region** (lr alpha > 0.02, dilated
`max(2, max(h,w)/128)` px; union with the hr alpha when the dataset has it) shared by every method compared.
- **l1** masked mean |pred − hr| (0..1 units).
- **spectrum_error** (the detail metric): luminance, effect bbox, Hann window, |FFT|²/Σw², 32 linear radial bins in
  cycles/pixel over [1/64, 0.5], mean |log10(P_pred+F) − log10(P_hr+F)|, F = uint8 quantization noise power.
- **ms_dssim** 1 − Σ w_s·SSIM_s (luminance, σ 1.5, scales 1, ½, ¼, w 0.5/0.3/0.2). No downloaded weights.
- **flicker** mean_t |x_t − x_{t−1}(p − m_t)|, m = the sequence's own estimated flow; ratio to hr (with hr's own flow).
- **temporal_jerk** mean_t |x_{t+1} − 2x_t + x_{t−1}| (unwarped), ratio to hr.
- **detail swim** flicker of the detail layer (out − lr, warped with flow(out)) / same for (hr − lr, flow(hr)).
- Baselines: the unmodified lr frame and an unsharp mask whose σ, amount are grid-searched on train
  (objective: mean of l1, spectrum, ms_dssim relative to lr; cached in `runs/eval3d/unsharp_<version>_train.json`).
- `evaluate_screen_detail(model, version, split)` runs the model autoregressively per sequence (reset, then one
  `infer(frame (h,w,3) uint8, {"alpha","depth","motion"})` per frame), non-overlapping whole sequences; a
  checkpoint path is loaded as `ScreenDetail(path, process_scale=1.0)` because the stored crops already are the
  runtime's processed resolution (2×2-pooled 1440p).

Which motion is legitimate (the 2D lesson: never warp a sequence with another sequence's motion; revised
2026-11-09 with the lead after measuring s3d_v1). The stored aux `motion` is the opacity-weighted *fluid* velocity,
not the apparent image motion: near the source a plume's image is quasi-steady while gas flows through it, only
the camera part is exact, and background carries 0. Warp error relative to no warp, region-weighted, pooled
(`test_data3d_layer3.py` alignment table):

| alignment | static lr | static hr | moving lr | moving hr |
|---|---|---|---|---|
| stored motion | 1.67 | 1.36 | 1.26 | 1.12 |
| stored × 0.5 / × 0.25 | 1.16 / 1.02 | 1.05 / 0.98 | 0.96 / 0.92 | 0.92 / 0.92 |
| **own estimated flow** | **0.37** | **0.50** | **0.31** | **0.42** |
| the other sequence's estimated flow | 0.77 | 0.80 | 0.62 | 0.70 |

So every temporal metric warps each sequence with its **own** apparent motion, `metrics.sequence_flow`
(pyramidal Lucas-Kanade on luminance, smooth: σ 4 px windows, 2 px smoothing): prediction with the prediction's
flow, hr with hr's. A shared lr-derived flow would favour lr-like (blurred) output. With the stored motion the
gate was blind (2-code iid shimmer in hr: ratio 1.06); with own flow it is 1.35, temporal jerk 1.33, detail swim
1.22. A 1-px blur of hr lowers flicker (0.75) — less detail changes less — so flicker, jerk and swim are
**upper-bound gates (≤ 1.2× hr), never training objectives**. Temporal metrics use the same dilated effect region
as the spatial ones (own flow also compensates the camera motion of the background). The Layer 3 requirement
"stored motion vectors reduce warp error" is an `xfail` with its numbers (lead-acknowledged).

"No host sync" uses `torch.cuda.set_sync_debug_mode("warn")`: it sees torch-initiated syncs and H2D/D2H
copies only, not Warp's own; it is the "where feasible" check.

## bench3d (the 3D-phase deliverable)
`bench3d/run.py` (see its docstring) measures complete pipelines — raw 64³, 64³→trilinear 256³, route A
(VolumeUpscaler), route B (ScreenDetail: the trained checkpoint when `screen_checkpoint()` finds one, else
"untrained"; renders with aux and passes aux + the projected domain bbox), brute-force 128³ and 256³, plus
full-quality-config step-only lines — on two scene sets (`--scene-set closed|presets|all`, default all): the two
original closed-box scenes (campfire; plume past a sphere; kept for baseline continuity) and the four sandbox
presets `p1..p4` with their dissipating solver overrides on every solver, pre-simulated 150 steps (route A at
realistic occupancy) — at 2560×1440 with `march_scale` 0.5 and 1.0:
per-stage and whole-frame CUDA-event times (mean/p50/p95, hitches > 50 ms), host wall time per frame,
peak VRAM (torch + device-level), occupancy, and speed-up vs brute-force 256³. Each (scene, pipeline)
runs in its own subprocess. nvidia-smi is read first; a shared GPU is recorded and loudly flagged.
Outputs `bench3d/results/latest.json` and `bench3d/results/REPORT.md`; `--update-baseline` writes
`bench3d/baseline.json`; a later run fails on any p50 / VRAM regression > 10% (not gated on a shared GPU
unless `--strict`).

## Deferred
- Layer 2 physics benchmarks in 3D (Taylor–Green, lid-driven cavity vs Ghia, vortex rings), PhiFlow cross-check.
- Quality gates for route A (spectrum error vs tricubic, flicker, OOD): route A stays untrained. The 3D
  "tricubic" baseline itself is not implemented yet. (Route B gates: `test_screen_layer4.py`.)
- LPIPS (needs downloaded weights): MS-DSSIM is used as the structural metric instead.
- 256³ Layer 1 runs (all Layer 1 tests are 64³, plus 96³/128³ references) and 10,000-frame 3D rollouts.
- Renderer quality vs a reference renderer, LPIPS, Layer 6 visual review page; `vfx.app3d` sandbox is not tested.
- Warp-internal host syncs (not visible to torch's sync debug mode); CUDA-graph capture of the whole frame.
- Idle-GPU numbers: the lead re-runs `bench3d` on an idle GPU.
