# vfx/correct3d — learned large-scale velocity correction for the 64³ solver (feasibility study)

Shortlist item 5 (solver-in-the-loop). The question is whether a cheap correction applied inside the 64³ solver loop can
make its large-scale motion track a finer sim (restricted 128³), within about 0.5 ms per step, stably.
**Status: in progress.** The characterization (E0) and the classical baseline (E1) are done. The learned corrector
(E2/E3) is written but has not been trained yet, because GPU work is on hold. See `EXPERIMENTS.md`.

This module does not edit `vfx/sim3d`. It uses only the public `Solver3D` interface and edits `state.data` in place
between steps, as `datagen3d/guide3d` does. It reads `datagen3d` (the scenario sampler and the guide) without modifying it.

## Interface
```python
from vfx.correct3d import VelocityCorrector
corr = VelocityCorrector(checkpoint_path)    # None -> identity (apply() is a no-op)
corr.apply(state, dt)                        # BEFORE solver.step(state, forces, dt): u += c(state), in place
solver.step(state, forces, dt)               # that step's projection removes the divergent part of c
corr.timings, corr.last_stats                # last_stats: GPU (3,) [rms c, rms u, fallback flag]
```
The corrector expects the 64³ solver to run with the tuned-classical config it was trained on (`vorticity_eps=3.0`).
Limiter (on the GPU, no host sync):
- each cell's correction is capped at `cell_cap` m/s;
- if rms(c) is greater than `max_ratio`·rms(u) + `abs_floor`, or c is not finite, the whole step's correction is dropped (identity fallback).

The call is captured as one CUDA graph per state buffer.

## Files
| file | content |
|---|---|
| `common.py` | scenes (6 named and held out; sampler scenes), restriction, diagnostics (centroid, 95 % height, KE, …), KE spectrum, MIP, PNG writer, `CORR_DATA` (D:) |
| `rollout.py` | instrumented rollout (any resolution, optional corrector, perturbation events) and `summarize` against a reference |
| `characterize.py` | E0: 64³ vs 128³ (vs 256³) on the named scenes |
| `tune_classical.py` | E1: 64³ config sweep against 128³ on train scenes; `--noise_floor` gives the 128³-vs-perturbed-128³ spread |
| `model.py` | `CorrNet` (37k params, works at 32³ and is upsampled ×2), `features`, symmetry augmentation |
| `collect.py` | training pairs: `pre` (one-step from the restricted fine state) and `dagger` (on the corrected coarse trajectory, with 128³ nudged to follow it) |
| `train.py` | training of at most 15 min (MSE; R² on validation scenes) |
| `evaluate.py` | `compare` (held-out scenes, images), `stability` (3000 steps of user perturbations), `timing` |

Datasets and checkpoints are kept on D: (`D:\neural-vfx-data\datasets\corr3d_*`, `...\checkpoints\corr3d`), as the lead instructed, because C: is full.

## Findings so far
- **The gap is not missing energy.** Mass and heat match within 5 %, and the 64³ plume sits 0.05–0.14 m lower. But 64³
  carries 1.5–3× the kinetic energy of 128³, with 4–8× the power at k = 5–16 cycles/domain. The coarse column breaks
  up into a wide, bushy cloud near the source, and that stirring slows the rise.
- **Most of the gap comes from one parameter.** With `vorticity_eps` 6 → 3 (the same eps·dx as 128³), the combined score drops 63 %,
  the centroid error falls from 0.069 to 0.022 m, and total KE comes to 0.98×. The learned model has to beat this baseline.
- **What is left is structural and depends on scale.** k5–8 stays 1.6× too energetic at every eps, while k1–4 swings from
  too weak to too strong as eps changes. A scale-selective forcing could fix this; a global parameter cannot.
- **128³ is not converged either.** At 256³ the plume rises another 0.04–0.07 m.
