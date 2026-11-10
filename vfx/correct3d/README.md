# vfx/correct3d — learned large-scale velocity correction for the 64³ solver (feasibility study)

Shortlist item 5 (solver-in-the-loop). The question: can a cheap correction applied inside the 64³ solver loop
make the 64³ sim's large-scale motion track a finer sim (128³ shrunk to 64³ by box-averaging), in about
0.5 ms/step, stably?

**Answer: feasible on cost and stability, but the gain over the classical fix is small. Not worth adopting now.**
The full log is in `EXPERIMENTS.md`. Every timing was taken on a shared RTX 5080.

This module does not modify `vfx/sim3d`. It uses only the public `Solver3D` interface and edits `state.data` in place
between steps, the same way `datagen3d/guide3d` does. It reads `datagen3d` (scenario sampler, guide) without changing it.

## Interface
```python
from vfx.correct3d import VelocityCorrector
corr = VelocityCorrector(ckpt, gain=0.5)     # None -> identity (apply() is a no-op)
corr.apply(state, dt)                        # BEFORE solver.step(state, forces, dt): u += c(state), in place
solver.step(state, forces, dt)               # that step's projection removes the divergent part of c
corr.timings, corr.last_stats                # last_stats: GPU (3,) [rms c, rms u, fallback flag]
```
- **Checkpoint:** `D:\neural-vfx-data\checkpoints\corr3d\e3.pt`. It was trained for a 64³ solver running
  `SolverConfig3D(confinement_ref_res=128)` (PCG). It also works with `fused_config(confinement_ref_res=128)`.
  Use gain 0.5.
- **Limiter (GPU only, no host sync):**
  - each cell is capped at |c| ≤ `cell_cap` = 0.15 m/s;
  - if rms(c) > 0.2·rms(u) + 0.01, or the value is non-finite, that step falls back to identity.
- The whole call is one captured CUDA graph per state buffer.

## Results (13 held-out scenes: 6 hand-made including an OOD box + strong-wind scene, 4 test, 3 OOD; 300 steps; against free-running 128³)
| | combined score ↓ | plume centroid err | KE-spectrum err k1–8 | time-mean velocity err | cost/step |
|---|---|---|---|---|---|
| 64³ default | 4.61 | 0.056 m | 0.73 | 0.91 | 0.88–1.01 ms (PCG) |
| tuned classical (`confinement_ref_res=128`) | 1.75 | 0.020 m | 0.34 | 0.51 | same |
| + learned corrector (E3, gain 0.5) | **1.56** | 0.019 m | **0.20** | **0.44** | **+0.22 ms** |
| chaos floor (128³ vs a perturbed 128³) | 0.45 | 0.004 m | 0.03 | 0.25 | — |

- **Stability:** 3000 steps with pushes, emitters switched off/on, a box obstacle toggled and a wind gust. 5/5 scenes stayed
  finite. Kinetic energy shows no drift, divergence is ≤ 2.6e-5 (≤ 7.0e-5 with the fused solver), and the limiter never triggered.
- **The gap:** the 64³ sim is over-stirred, not under-energetic. Vorticity confinement scales with ε·dx, so 64³
  confines twice as hard as 128³. That gives 4–8× the restricted fine power at k = 5–16, a bushy column, and a lower plume.

## What adoption would cost downstream
These costs apply to the classical `confinement_ref_res` fix as well as the learned corrector, because both change the 64³ flow.
- **Route-B dataset and ScreenDetail:** `s3d_v1/v2` pair renders of the default 64³ sim with a 256³ sim steered to follow it. They
  would have to be regenerated with the new 64³ flow (the guide would steer toward it), and ScreenDetail retrained. That
  steered reference is also why corrected runs score slightly worse there: it is not a matched twin for a different flow.
- **Other tuning against the 64³ velocity:** detail3d / particles3d parameters tuned on it should be re-checked.
- **The corrector's own dependencies:** its checkpoint is tied to the solver config and version (`solver_version_tag` + corrector id). Any solver change
  other than the fused pressure solve means re-collecting data (about 3 min of GPU) and retraining (≤ 20 min).

## Files
| file | content |
|---|---|
| `common.py` | scenes, restriction, diagnostics, KE spectrum, MIP / PNG, data paths (`D:\neural-vfx-data`, set by the lead because C: is full) |
| `rollout.py` | instrumented rollout (any resolution, optional corrector, perturbation events) and `summarize` |
| `characterize.py` | E0: 64³ vs 128³ / 256³ |
| `tune_classical.py` | E1 parameter sweep; `--noise_floor` |
| `model.py` | `CorrNet` (37k params, works at 32³, ×2 trilinear), `features`, symmetry augmentation |
| `collect.py` | `pre` (one-step) and `dagger` (on the corrected trajectory, with 128³ nudged via guide3d) pairs |
| `train.py` | ≤ 15 min training, R² validation |
| `evaluate.py` | `compare` (images `cmp_*.png`: raw \| classical \| learned \| ref128), `stability`, `timing` |

```
python -m vfx.correct3d.collect --name r0 --mode pre --seeds 0-23 --coarse_cfg '{"confinement_ref_res":128}'
python -m vfx.correct3d.train --data r0 --out D:\neural-vfx-data\checkpoints\corr3d\e2.pt --minutes 10
python -m vfx.correct3d.collect --name r1 --mode dagger --ckpt ...e2.pt --seeds 0-23 --coarse_cfg '{"confinement_ref_res":128}'
python -m vfx.correct3d.train --data r0,r1 --init ...e2.pt --out ...e3.pt --minutes 10
python -m vfx.correct3d.evaluate compare|stability|timing --ckpt ...e3.pt --gain 0.5 --classical '{"confinement_ref_res":128}' --learned_cfg '{"confinement_ref_res":128}'
```
