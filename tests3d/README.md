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
| `test_eval3d_fair_metrics.py` | The distribution-level and reference-relative temporal metrics (band spectrum, gradient / local-contrast histogram W1, patch SWD, tLP, tOF, pop) on a synthetic moving plume with five controls (see "Metrics fair to non-pixel-aligned detail"). |
| `test_detail3d.py` | `vfx.detail3d.AdvectedDetail` + `VolumeRenderer.draw(detail=)` contract (see "Sim-carried detail phase"). |
| `test_data3d_layer3.py` | Layer 3 for `$VFX_DATASET3D` (default `s3d_v1`) through `vfx.datagen3d.load_sequences`: size ≤ 4 GB, card fields, solver version current, seed splits disjoint (and items come from their split), every OOD scenario has a scene-type tag absent from train (split names excluded), item shapes/dtypes, aux finite / alpha ∈ [0,1] / depth and motion 0 where alpha = 0 / plausible ranges, per-sequence stats (not black/saturated, effect visible, not frozen, hr/lr luminance ratio) with 8-MAD outlier report (failures must be flagged in the card), **lr/hr alignment** of static geometry (best integer shift (0,0), median diff ≤ 1 code), **hr has ≥ 1.5× lr's fine-scale energy while tracking it at coarse scale** (blurred correlation ≥ 0.9; scales from the projected 64³ voxel size), **alignment table** (own estimated flow must reduce warp error ≤ 0.9× for lr and hr; stored motion reducing it is an xfail), and a sensitivity check of the temporal metrics on the real data (2-code iid noise injected into hr). |
| `test_screen_layer4.py` | Layer 4 for route B: overfit one batch on a fresh `ScreenNet`; baseline report (lr, tuned unsharp, hr reference numbers, no checkpoint needed); with the checkpoint (`vfx.eval3d.checkpoint.screen_checkpoint`): runtime reads the CLAUDE.md aux keys, beats lr AND unsharp on the shift-tolerant detail metrics (spectrum_error_bl, band distance, gradient W1, patch SWD, tLP; L1 / spectrum / MS-DSSIM reported only since 2026-11-09, see "Detail-oracle controls"), OOD relative-to-lr score within 20%, flicker ratio ≤ 1.2 and ≤ max(lr's, 1), temporal-jerk ratio ≤ 1.2, detail swim (moving cameras) ≤ 1.2× reference. **Live** (no dataset): 64³ `Solver3D` on the 4 sandbox presets, orbiting camera, 600 frames at 2560×1440 through `VolumeRenderer(return_aux)` + `ScreenDetail` — finite, no frame > 50 ms after 60 warm-up frames, mean detail ms ≤ 2.5, flicker and jerk vs the lr render. Reports in `DATA_ROOT/runs/eval3d/`. |

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

## Metrics fair to non-pixel-aligned detail (sim-carried detail phase; `metrics.py` last section)
Procedural / advected / adversarial detail never matches the steered 256³ reference pixel for pixel: l1 and
MS-DSSIM against it punish exactly the wisps we want, and warp-error flicker rewards blur (1-px blur of hr: 0.75).
The legacy metrics and gates above are unchanged (names, formulas, thresholds) so old results stay comparable;
these are added next to them. All 0 for the reference itself, lower is better unless marked signed.
- **band_spectrum** — the `spectrum_error` power spectrum (same crop, Hann window, FLOOR) averaged per OCTAVE band
  (wavelength 32–64, 16–32, 8–16, 4–8, 2–4 px): per band signed log10(P_pred/P_ref) (< 0 missing detail at that
  scale, > 0 excess) and `distance` = mean |·|. Translation invariant: "right amount at each scale" without "right place".
- **gradient_hist_distance** — W1 between the luminance gradient-magnitude distributions on the effect region (all
  frames pooled) / the reference's mean gradient magnitude.
- **local_contrast_hist_distance** — same for local std of luminance (Gaussian windows σ 1 and 3 px), mean of scales.
- **patch_swd** — sliced Wasserstein distance between sets of 4096 random 7×7 patches (fixed seed, centres in the
  effect region) of 3 Laplacian-pyramid levels, normalized by the reference set's RMS, 128 fixed random projections,
  mean over levels. No downloaded weights.
- **tLP** (TecoGAN-style, weight-free) — d(t) = `lap_distance`(x_{t−1}, x_t) (mean over 4 Laplacian levels of the
  masked mean |Δband|, brightness residual excluded); `tlp` = mean_t |d_out − d_ref| / mean_t d_ref; `tlp_log_ratio`
  = mean_t log(d_out/d_ref) (signed: < 0 too static / blurred, > 0 boiling); `pop` = burst(c_out)/burst(c_ref),
  burst = max_t/median_t of the change left after warping each sequence with its OWN flow (1 = as bursty as the reference).
- **tOF** — mean endpoint distance |F_out − F_ref| (px) between each sequence's own smooth Lucas-Kanade flow.
- `fair_detail_scores(out, ref, mask)` returns all of them.

Controls that validate them (`test_eval3d_fair_metrics.py`, synthetic plume translating (1.5, −1) px/frame with
band-pass detail glued to it and slowly evolving; numbers measured):

| control | L1 | MS-DSSIM | band dist | 2–4 px band | grad W1 | contrast W1 | patch SWD | tLP | tLP log | pop | tOF px | legacy flicker r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| independent realization (procedural done right) | 0.022 | 0.202 | 0.032 | −0.03 | 0.010 | 0.015 | 0.036 | 0.020 | 0.01 | 0.98 | 0.05 | 1.00 |
| blur σ 1.5 | **0.010** | **0.062** | 0.774 | −2.35 | 0.501 | 0.359 | 0.238 | 0.636 | −1.01 | 0.52 | 0.04 | **0.24** |
| + iid noise 0.03 | 0.024 | 0.143 | 0.085 | +0.36 | 0.704 | 0.579 | 0.248 | 0.825 | +0.60 | 0.28 | 0.58 | 13.6 |
| detail stuck to screen | 0.022 | 0.195 | 0.040 | +0.07 | 0.058 | 0.043 | 0.039 | 0.860 | −1.96 | 0.28 | 1.42 | 2.60 |
| detail re-rolled every 6 frames | 0.023 | 0.208 | 0.036 | −0.03 | 0.006 | 0.007 | 0.034 | 0.056 | +0.03 | **5.75** | 0.19 | 1.90 |

L1, MS-DSSIM and the legacy flicker ratio all prefer the blur over correct-but-unaligned detail; every new
distribution metric and tLP reverse that (asserted: control ≥ 1.5× the independent realization on the metric meant
to catch it). Blur and boiling both raise tLP with opposite tLP log-ratio signs; screen-stuck detail is caught by
tOF (and tLP); periodic regeneration is caught by `pop` (the unwarped tLP is nearly blind to it at this motion speed).

## Detail-oracle controls and the Layer 4 gate change (2026-11-09, lead request)
The ML engineer reported that pixel-aligned metrics cannot see displaced detail; reproduced independently
(`test_eval3d_oracle.py`, s3d_v1 test, 12 whole sequences). Oracle: lr + D shifted s px, D = hr − G_σ2(hr) (the
reference's own detail: plausible by construction).

| control | l1 | ms_dssim | spectrum_error | band dist | grad W1 | patch SWD | tLP | tOF px |
|---|---|---|---|---|---|---|---|---|
| lr | 0.0393 | 0.198 | 0.266 | 0.323 | 0.228 | 0.250 | 0.515 | 1.53 |
| lr + D, 0 px | **0.0361** | **0.141** | 0.366 (worse than lr) | 0.262 | 0.148 | 0.162 | 0.310 | 1.30 |
| lr + D, 1 px | 0.0372 | 0.162 | 0.186 | 0.187 | 0.136 | 0.145 | 0.302 | 1.30 |
| lr + D, 2 px | 0.0389 (≈ lr) | 0.195 (≈ lr) | 0.222 | 0.190 | 0.138 | 0.146 | 0.302 | 1.31 |
| lr + D, 4 px | 0.0410 (worse) | 0.228 (worse) | 0.231 | 0.209 | 0.140 | 0.156 | 0.307 | 1.37 |
| lr blurred σ 1 | 0.0395 | 0.201 | 0.898 | 0.617 | 0.278 | 0.310 | 0.569 | 1.54 |
| unsharp (tuned) | 0.0396 | 0.200 | 0.253 | 0.292 | 0.208 | 0.227 | 0.483 | 1.53 |

1. l1 / ms_dssim reward detail only when registered to ~1 px; 2 px is a tie with lr, 4 px loses. A gate "beat lr on
   l1 and ms_dssim" therefore fails real, plausible filaments in slightly different places.
2. Full-band `spectrum_error` ranks lr + exactly registered detail (0.366) below lr (0.266). Per bin, the excess is in
   bins above 0.25 cycles/px (reference power only ~15–200× the uint8 quantization floor), where the error depends on
   how the added detail's phase lines up with the high frequencies lr already has (the 1-px shift has the identical
   detail power spectrum and scores 0.186). Below 0.25 cycles/px the ordering is the expected one.

Changes (legacy names and formulas untouched; history stays comparable):
- new `spectrum_error_bl` = `spectrum_error` restricted to bins with centre < 0.25 cycles/px (`F_MAX_BL`).
- `score_sequence` / `evaluate_screen_detail` now ALSO report `fair_detail_scores` (spectrum_error_bl, band_*,
  grad_hist, contrast_hist, patch_swd, tlp, tlp_log_ratio, pop, tof) per method.
- `test_screen_layer4.py`: `test_beats_lr_and_sharpen_on_test` now gates on `metrics.DETAIL_GATE_METRICS` =
  spectrum_error_bl, band_distance, grad_hist, patch_swd, tlp (model < lr AND < tuned unsharp, unchanged rule);
  `test_ood_within_20pct` uses the same metrics (unchanged 1.2 rule). l1, ms_dssim and spectrum_error are printed by
  `test_pixel_metrics_report` with what the old gate would have said, and no longer fail. Flicker / jerk / swim
  upper-bound gates, overfit, aux and live gates are unchanged.
- Validation, `test_eval3d_oracle.py`: every shifted oracle (0/1/2/4 px) must beat lr, blur and unsharp on every gate
  metric, and the 2 / 4 px oracles must score ≤ 1.25× the registered one. (From the measured table all hold; the
  file has not been re-run after the code change because the GPU was reserved by the lead.)

## Sim-carried detail phase: contract tests and the comparison harness
- `test_sim3d_layer1.py` (additions, all skip until the paths exist): **default path bit-identical** — sha256 of the
  state after 1/30/120 steps of the scripted fire+impulse+wind+sphere run, for `Solver3D(64)` and
  `Solver3D(64, runtime_config())`, recorded 2026-11-09 before any solver change (sim3d files of 16:10, version
  `sim3d-warp-0.1.0`, RTX 5080 / torch 2.11.0+cu128 / warp 1.19.0; skips on another environment); explicit
  `scalar_res=None`, `pressure_mode="pcg"` reproduce it and `hires_volume()` is None. `pressure_mode="fused"`:
  divergence < 1e-4 after `project()` and after 60 steps (open, sphere), else an xfail ONLY if vfx/sim3d/README.md
  labels fused as a lower-quality level (the number is printed either way); bit-identical repeat. `scalar_res=128`:
  fine-field mass within 0.1% over 1000 sourceless steps (open, sphere; filled by an emitter first), avgpool2(fine)
  vs the 64³ density / excess temperature rel L1 < 5% and equal mass within 0.1%, no fine smoke ≥ 2 fine cells inside
  an obstacle, deterministic, 1000-step fire+impulse stress finite and non-negative.
- `test_detail3d.py` (skips until `vfx.detail3d.AdvectedDetail` and `draw(detail=)` exist): interface; `detail=None`
  bit-identical to no kwarg (frame and aux) and to a recorded hash (recorded AFTER the render engineer's 20:34 edit —
  the lead can confirm it against tag v0.3); drawing with detail leaves no state behind; visible; mean opacity and
  brightness within 5%; no opacity outside the dilated smoke footprint or over an obstacle, no light change outside
  it except on shadow receivers (floor, obstacle); empty volume unchanged; deterministic; finite over 2000 steps of a
  swirl; **moves with the flow** (translating blob + co-moving sub-grid turbulence; the float alpha detail layer must
  shift by the projected translation ±25% with correlation > 0.3 after 4 and 8 steps); **no periodic pop**
  (quasi-steady swirling plume, max/median per-step change of the detail layer ≤ 2.0 over 150 steps); cost at 1440p
  (march_scale 0.5) step + extra march ≤ 2.5 ms, asserted only on an idle GPU (skips with the numbers otherwise).
- `python -m bench3d.compare_modes` — see its docstring; writes `bench3d/results/MODES.md` + `modes.json` and per-preset
  contact sheets of 1:1 crops (default `DATA_ROOT/runs/compare_modes/latest`). `--ref-res` < 256 is only for
  smoke-testing the harness on a full GPU and is recorded in the report.

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
