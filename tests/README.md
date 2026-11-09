# tests/ — gate tests (owner: test/eval engineer)

```
.venv/Scripts/python -m pytest tests -q                 # all layers; missing modules/data/checkpoints SKIP
.venv/Scripts/python -m pytest tests -q -s -rsx         # with measured numbers and skip/xfail reasons
VFX_DATASET=v1 .venv/Scripts/python -m pytest tests/test_data_layer3.py -q
VFX_DATASET=v2 .venv/Scripts/python -m pytest tests/test_model_layer4.py -q -s -rsx
.venv/Scripts/python -m vfx.eval.live_flicker               # live-app flicker, no dataset
.venv/Scripts/python -m bench.run [--update-baseline]   # Layer 5 (not part of pytest)
```

Tests are written from SPEC.md "Testing and validation" and the CLAUDE.md interfaces, never from
the implementation under test. Spec thresholds are never relaxed to make something pass; a v1 gate
that v0 is not expected to meet is an `xfail(strict=False)` that prints the measured number.

| File | Layer | What it checks |
|---|---|---|
| `test_metrics.py` | — | The metrics themselves (`vfx/eval/metrics.py`): spectrum error 0 on identity, monotone in blur, shift-invariant, noise-penalized; warp exact for integer shifts and +y = up; flicker 0 for pure transport; flicker equals 2s/√π for injected per-frame noise of std s; `evaluate_upscaler` warps the prediction with its driving (`lr_sim`) velocity, keeps the legacy HR-velocity number, and skips unfinished scenario dirs; bicubic baseline. |
| `test_sim_layer1.py` | 1 | Via `vfx.sim` public API only. Interface/reset contract; normalized divergence `< 1e-4` after `project()` (128²/256², with and without an obstacle) and after full `step()`s; zero net flux through every row/column after projection (no-penetration walls); density mass within 0.1% over 1000 steps (closed box, no sources, `density_dissipation=0`, also with an obstacle and with jets aimed at the walls); `advect()` exact for whole-cell shifts, centroid/shape vs analytic translation, multi-channel consistency, error shrinks 32→256; hot fluid rises, dense fluid sinks, ambient stays at rest; nothing enters obstacles; bit-identical determinism; plume convergence vs a 512² reference; 2000-step no-NaN/no-blow-up stress (strong fire, impulses every 5 steps). |
| `test_data_layer3.py` | 3 | Dataset `$VFX_DATASET` (default `v1`) read directly from disk: card/splits/counts, size < ~6 GB, shapes/dtypes, meta fields, solver version == current `vfx.sim.SOLVER_VERSION`, no NaN/Inf, channel order (lr density/temperature == 4× avg-pool of hr), solids == `rasterize_obstacles(meta)`, no density inside solids, frames not frozen, physical ranges (out-of-range/empty scenarios must be in `card["flagged"]`), statistical outlier report (median + 8·MAD), disjoint split seeds, no near-duplicate frames between train and test/ood, every OOD scene has a tag never seen in train, `load_dataset` item contract (`hr_prev` = frame t−1). M2 count gate (≥ 500 train) is informational xfail. |
| `test_model_layer4.py` | 4 | Overfit-single-batch on a fresh `DetailNet` (no data needed). On `default_checkpoint()` + test split of `$VFX_DATASET` (v1: 8 frames/scenario; v2: 16 frames, all stored frames are used; unfinished scenario dirs without `meta.json` are skipped, a dataset with no finished scenarios or no `splits.json` skips): beats bicubic on spectrum error (input = pooled LR and LR sim) and on density L1 (pooled LR only). v1 gates as xfail with numbers: spectrum ≥ 50% below bicubic, flicker ratio ≤ 1.2 (audited definition below), OOD within 20%. OOD scores printed per scenario. **Live flicker** (no dataset, `vfx/eval/live_flicker.py`): 128² `Solver` on fixed scenes (fire; smoke + circle obstacle), 300 steps, fed to `Upscaler` and `classical_upscale`, flicker with the 128² run's own velocity vs a 512² `Solver` run of the same scene (indicative reference); v1 gate as xfail. Long rollout: 1200 frames of `Upscaler` on a live 128² `Solver` with a fire emitter + impulses — finite, bounded, HR mean density within 25% of LR (drift). Reports: `DATA_ROOT/runs/eval/layer4_latest.json`, `live_flicker_latest.json`. |
| `test_bench_m0.py` | 5 / M0 | GPU detected; harness times a trivial torch and Warp kernel; regression comparator flags > 10%. |

Metric definitions (exact formulas in `vfx/eval/metrics.py` docstrings):
- **spectrum_error**: mean over wavenumber shells k = 1..N/2 of |log10 E_pred(k) − log10 E_ref(k)| of
  density (decades; frames averaged in spectral space; DC excluded).
- **flicker**: mean |x_t − warp(x_{t−1}, v_{t−1})| with our own first-order semi-Lagrangian warp, where v is
  the velocity of the flow that actually transports x (the LR input the upscaler was fed). Judged as the ratio
  `flicker(pred, v_source) / flicker(HR, v_lr)`: the reference is the HR run warped with its own pooled velocity.
  Both velocities are 128², so both sides are warped with equally coarse flow. Response to real flicker: iid
  per-frame noise of std s adds ≈ 2s/√π (exact for an exact warp, `test_flicker_responds_to_known_noise_amplitude`);
  on v1 HR density, noise of 0.2% of the frame max doubles the value.
  - **Audit 2026-11-09 (why the lr_sim number changed).** Before, the prediction was always warped with the pooled-HR
    velocity (`lr.npy`), also when it was driven by `lr_sim`. In v1, `lr_sim` is an independent run whose flow has
    diverged from the HR run (mean |v_lr_sim − v_lr| ≈ 56% of mean |v_lr| on test), so that warp moved the frame
    along the wrong flow: bicubic(lr_sim) warped with HR velocity gave 0.00126, barely below not warping at all
    (0.00133), while with its own velocity it gives 0.00060 (reference 0.00058). Symmetrically, HR warped with
    the lr_sim velocity is 3.7× its own-velocity value. The old lr_sim ratios (neural 2.78, bicubic 2.20) therefore
    measured run divergence, not temporal instability. The time index (v_{t−1}) was verified against the solver
    (advection is the first kernel of `step()`, using the velocity stored with frame t−1).
  - The pre-audit number is kept as `flicker_ratio_legacy_hrvel` (identical to the audited value for source `lr`).
  - `flicker_ratio_detail_norm` (flicker / mean |∇x|, pred over HR) is a diagnostic only: injected noise also raises
    |∇x|, so it saturates (2% noise: raw flicker ×10.6, normalized ×1.5) and would hide real flicker.
  - v2: `lr_sim` and `hr` are a matched pair, so the reference follows the same flow; `flicker_ref_srcvel`
    (HR warped with the lr_sim velocity) then also becomes meaningful and is reported.
  - Live (`vfx.eval.live_flicker`): the reference is a 512² run of the same scene — a different realization, so
    it is indicative (statistically comparable sequence, not pixel-matched).
- **Baseline** is our own bicubic (`metrics.bicubic_upscale`, the 2D "tricubic"), not `runtime.classical_upscale`.

`bench/run.py` (Layer 5): CUDA-event timing per iteration of `Solver.step` 128², `Upscaler.infer`
(if a checkpoint exists), `Renderer.draw` 1024², a whole frame, plus peak VRAM (torch allocator and
device-level). Writes `bench/results/latest.json`; fails (exit 1) if any stage's median or peak VRAM
regresses > 10% vs `bench/baseline.json`. The SPEC budgets are printed for information.

## Deferred from SPEC (not in v0)
- **3D**: every test is 2D (v0 demo scope); 64³ solver budget not measured.
- **Layer 2 physics benchmarks** (nightly): Taylor–Green decay, lid-driven cavity vs Ghia 1982,
  vortex rings, dam break.
- **PhiFlow cross-check** of smoke scenes.
- **LPIPS** on fixed renders and **Layer 6** visual review page.
- Divergence of predicted velocity (v0 network does not predict velocity).
- 10,000-frame rollout and 10-minute interactive session (v0 runs 1200 frames); 60 fps / 1% lows
  and input latency are app-level measurements not yet automated.
- CI wiring (tests run manually on the target machine).
