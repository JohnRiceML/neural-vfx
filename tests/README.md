# tests/ — gate tests (owner: test/eval engineer)

```
.venv/Scripts/python -m pytest tests -q                 # all layers; missing modules/data/checkpoints SKIP
.venv/Scripts/python -m pytest tests -q -s -rsx         # with measured numbers and skip/xfail reasons
VFX_DATASET=v1 .venv/Scripts/python -m pytest tests/test_data_layer3.py -q
.venv/Scripts/python -m bench.run [--update-baseline]   # Layer 5 (not part of pytest)
```

Tests are written from SPEC.md "Testing and validation" and the CLAUDE.md interfaces, never from
the implementation under test. Spec thresholds are never relaxed to make something pass; a v1 gate
that v0 is not expected to meet is an `xfail(strict=False)` that prints the measured number.

| File | Layer | What it checks |
|---|---|---|
| `test_metrics.py` | — | The metrics themselves (`vfx/eval/metrics.py`): spectrum error 0 on identity, monotone in blur, shift-invariant, noise-penalized; warp exact for integer shifts and +y = up; flicker 0 for pure transport; bicubic baseline. |
| `test_sim_layer1.py` | 1 | Via `vfx.sim` public API only. Interface/reset contract; normalized divergence `< 1e-4` after `project()` (128²/256², with and without an obstacle) and after full `step()`s; zero net flux through every row/column after projection (no-penetration walls); density mass within 0.1% over 1000 steps (closed box, no sources, `density_dissipation=0`, also with an obstacle and with jets aimed at the walls); `advect()` exact for whole-cell shifts, centroid/shape vs analytic translation, multi-channel consistency, error shrinks 32→256; hot fluid rises, dense fluid sinks, ambient stays at rest; nothing enters obstacles; bit-identical determinism; plume convergence vs a 512² reference; 2000-step no-NaN/no-blow-up stress (strong fire, impulses every 5 steps). |
| `test_data_layer3.py` | 3 | Dataset `$VFX_DATASET` (default `v1`) read directly from disk: card/splits/counts, size < ~6 GB, shapes/dtypes, meta fields, solver version == current `vfx.sim.SOLVER_VERSION`, no NaN/Inf, channel order (lr density/temperature == 4× avg-pool of hr), solids == `rasterize_obstacles(meta)`, no density inside solids, frames not frozen, physical ranges (out-of-range/empty scenarios must be in `card["flagged"]`), statistical outlier report (median + 8·MAD), disjoint split seeds, no near-duplicate frames between train and test/ood, every OOD scene has a tag never seen in train, `load_dataset` item contract (`hr_prev` = frame t−1). M2 count gate (≥ 500 train) is informational xfail. |
| `test_model_layer4.py` | 4 | Overfit-single-batch on a fresh `DetailNet` (no data needed). On `default_checkpoint()` + test split: beats bicubic on spectrum error (input = pooled LR and independent LR sim) and on density L1 (pooled LR only: the independent sim is not pixel-aligned with HR). v1 gates as xfail with numbers: spectrum ≥ 50% below bicubic, flicker ratio ≤ 1.2, OOD within 20%. OOD scores printed per scenario. Long rollout: 1200 frames of `Upscaler` on a live 128² `Solver` with a fire emitter + impulses — finite, bounded, HR mean density within 25% of LR (drift). Report: `DATA_ROOT/runs/eval/layer4_latest.json`. |
| `test_bench_m0.py` | 5 / M0 | GPU detected; harness times a trivial torch and Warp kernel; regression comparator flags > 10%. |

Metric definitions (exact formulas in `vfx/eval/metrics.py` docstrings):
- **spectrum_error**: mean over wavenumber shells k = 1..N/2 of |log10 E_pred(k) − log10 E_ref(k)| of
  density (decades; frames averaged in spectral space; DC excluded).
- **flicker**: mean |x_t − warp(x_{t−1}, v_{t−1})| with our own first-order semi-Lagrangian warp;
  judged as the ratio pred / HR reference on the same (pooled-LR) velocities.
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
