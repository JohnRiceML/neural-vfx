# correct3d experiment log

All timings and runs on a SHARED RTX 5080 (other training / dataset jobs held 93-99 % utilization and up
to 14.7 GB of VRAM; RAM 1.4-2 GB free). Fine references: 128^3, full-quality solver; 64^3 also
full-quality (`SolverConfig3D()`, the 0.8-1.0 ms setting), sandbox dissipation (density 1.2 /s, cooling 5 /s).
"Restricted" = box average to 64^3. Scenes: `common.named_scenes()` (hand-made, never trained on),
train = `datagen3d` sampler train split seeds 0.., held-out = test 100000.. / ood 200000.. seeds.

## E0 - characterization, no learning (`characterize.py`, 300 steps = 10 s)
Hypothesis (from the brief): the 64^3 plume is slower because it is less energetic (numerical dissipation).

| scene | smoke centroid end 64 / 128 / 256 (m) | mass, heat 64/128 | KE 64/128 | KE band ratio 64/128 restricted k1-2, 3-4, 5-8, 9-16, 17-31 | density-weighted v_y 64/128 |
|---|---|---|---|---|---|
| fire | 0.418 / 0.562 / 0.633 | 1.04, 1.04 | **2.21** | 1.59, 2.05, 3.57, 4.40, 3.41 | 0.77 |
| smoke | 0.303 / 0.352 / 0.396 | 1.00, 1.01 | **3.18** | 1.71, 2.27, 5.79, 8.18, 4.37 | 0.88 |
| smoke_sphere | 0.256 / 0.306 | 1.00, 1.01 | 2.50 | 1.35, 2.37, 4.13, 7.89, 3.39 | 0.89 |
| fire_wind | 0.420 / 0.561 | 1.04, 1.05 | 1.84 | 0.65, 1.26, 2.86, 4.07, 3.03 | 0.77 |
| smoke_push | 0.218 / 0.207 | 1.00, 1.04 | 1.10 | 0.67, 1.05, 1.73, 1.97, 1.75 | 1.04 |
| ood_box_wind | 0.294 / 0.341 | 1.01, 1.05 | 1.54 | 0.61, 1.06, 2.23, 3.46, 2.56 | 0.92 |

256^3 vs 128^3 (fire / smoke): KE 0.98 / 0.89, bands 1.68,1.49,1.18,0.81,0.57 / 1.79,1.59,1.03,0.54,0.43,
time-mean density rel. error 0.20 / 0.14 (64^3 vs 128^3: 0.44 / 0.21).

Result: the hypothesis is WRONG in its mechanism. Mass and heat agree within 5 %, the 64^3 plume does sit
lower (centroid -0.05 to -0.14 m, mean upward velocity of the smoke 0.77-0.89x), but the 64^3 velocity field
carries 1.5-3x MORE kinetic energy than 128^3, concentrated at k = 5-16 cycles/domain (4-8x the
restricted fine power), i.e. at 4-13 cells. MIPs (char_*.png): the 64^3 column breaks up within ~0.15 m
of the source into a bushy, wide cloud with lateral streaks; 128^3 and 256^3 keep a thin coherent column
that rises higher. So the coarse error is excess mid-scale stirring that mixes momentum sideways and
slows the plume, not missing energy. Prime suspect: vorticity confinement, whose force is eps*dx*(N x w)
- with eps fixed in "world units" the confining force per unit vorticity doubles from 128^3 to 64^3.
The trend continues 128 -> 256 (plume higher again, more large-scale and less mid-scale energy), so
128^3 is not converged; it is used as the reference because 256^3 does not fit the shared GPU for training.
Decision: the cheap classical fix to test first is the confinement strength (and buoyancy), E1.

## E1 - tuned-classical baseline (`tune_classical.py`, 8 train scenes, 300 steps, vs restricted 128^3)
Hypothesis: if the gap is mostly the confinement scaling, halving eps at 64^3 (eps*dx matched to 128^3)
closes most of it. Grid: vorticity_eps {6,4,3,2,1,0} x buoyancy (alpha and beta) x{1,1.2,1.4}.
Score = |centroid err|/0.05 + |95%-height err|/0.05 + |log KE ratio| + mean|log spectrum ratio| k1-8
+ rel. error of the time-mean density at 16^3 (lower = better; mean over scenes).
Run stopped by the lead's GPU hold after 16 of 18 configs (eps 1.0 x1.4 done, eps 0 x1.2 / x1.4 missing;
no JSON written - numbers from the log).

| eps | buoy | score | centroid err (m) | h95 err (m) | KE 64/128 | spec err k1-8 | mean-density err | mean-velocity err | band ratios k1-2, 3-4, 5-8, 9-16, 17-31 |
|---|---|---|---|---|---|---|---|---|---|
| 6 (default) | 1.0 | 4.93 | 0.069 | 0.112 | 1.43 | 0.66 | 0.310 | 0.731 | 0.50, 1.08, 2.22, 3.11, 2.37 |
| 6 | 1.4 | 3.32 | 0.035 | 0.061 | 1.58 | 0.68 | 0.280 | 0.719 | 0.57, 1.20, 2.43, 3.36, 2.54 |
| 4 | 1.2 | 1.97 | 0.020 | 0.041 | 1.11 | 0.42 | 0.225 | 0.509 | 0.72, 1.17, 1.79, 1.84, 1.34 |
| **3** | **1.0** | **1.84** | 0.022 | 0.037 | 0.98 | 0.35 | 0.232 | **0.456** | 0.82, 1.20, 1.64, 1.41, 1.04 |
| 3 | 1.2 | 1.87 | **0.014** | 0.042 | 1.08 | 0.39 | **0.218** | 0.480 | 0.93, 1.31, 1.79, 1.53, 1.12 |
| 2 | 1.0 | 2.00 | 0.017 | 0.048 | 1.00 | 0.38 | 0.247 | 0.490 | 1.13, 1.40, 1.66, 1.23, 0.91 |
| 1 | 1.0 | 2.40 | 0.019 | 0.059 | 1.04 | 0.49 | 0.249 | 0.602 | 1.53, 1.65, 1.69, 1.08, 0.83 |
| 0 | 1.0 | 3.05 | 0.025 | 0.076 | 1.09 | 0.58 | 0.291 | 0.734 | 2.04, 1.93, 1.65, 0.92, 0.72 |

Result: one parameter does most of the work. eps 6 -> 3 (exactly the dx scaling: same eps*dx as the
128^3 reference) cuts the score 63 %, the centroid error 3x (0.069 -> 0.022 m), the time-mean velocity
error 0.73 -> 0.46, and brings total KE to 0.98x. Extra buoyancy only helps when confinement is too strong
(it compensates the slowdown, it does not remove the cause). What is left at eps 3 is structural and
scale-dependent, and no single eps fixes it: k5-8 stays 1.6-1.7x too energetic for every eps (even
eps 0), while k1-4 goes from too weak (eps >= 4) to too strong (eps <= 2). Lowering eps removes
mid-scale stirring but also lets the large scales run away; the coarse grid's own transport (MacCormack
at 4-cell emitters, smoothed forces) is a second source of the k5-8 excess.
Decision: tuned-classical baseline = `vorticity_eps=3.0` (score-best; x1.2 buoyancy is a tie). The learned
corrector is trained and evaluated ON TOP of it (residual over the best classical), so any gain it shows is
gain over the baseline. Its job: remove the k5-8 excess without draining k1-4, i.e. a scale-selective
forcing, which a single global parameter cannot do.

Update (after the hold): the solver engineer added the opt-in `SolverConfig3D.confinement_ref_res`
(eps_used = eps * res / ref). The baseline is now `confinement_ref_res=128` (= eps 3 at 64^3; checked:
same rollout as `vorticity_eps=3` up to float rounding / chaos, max diag diff 0.016 after 120 steps).

## E1b - noise floor (`tune_classical --noise_floor`, same 8 train scenes)
128^3 vs the same 128^3 run nudged by 1 mm/s at step 5: score 0.45 (centroid err 0.004 m, h95 0.013 m,
KE 1.00, spectrum err 0.026, mean-density err 0.080, mean-velocity err 0.246). So of the tuned baseline's
1.84, about 1.4 is reducible in principle; 0.45 is chaos over a 10 s window (probably an
underestimate: one perturbation may not fully decorrelate the run).

## E2 - one-step supervised ("pre-computed") corrector  [training run 1 of 4]
Hypothesis: a local net can learn R f_{t+1} - step64(P R f_t) (low-passed at 10 cycles/domain,
projected) and, applied before each step on top of the tuned baseline, removes the k5-8 excess.
Data `corr3d_r0` (D:): 24 train scenes, 240 steps, every 4th step, 1103 pairs, 0.6 GB. Target rms only
0.0006 m/s/step (domain rms; |u| ~ 0.1-1 m/s). CorrNet width 16 (37k params), 10 min, 134k iters:
**train R^2 0.41, val R^2 0.37** (unseen train-split scenes; 0 = identity).
Held-out compare (6 named + 4 test + 3 OOD scenes, 300 steps, vs restricted 128^3):

| | score | centroid err | h95 err | KE ratio | spec err k1-8 | mean-D err | mean-U err | bands k1-2..17-31 |
|---|---|---|---|---|---|---|---|---|
| raw 64^3 | 4.61 | 0.056 | 0.098 | 1.83 | 0.735 | 0.247 | 0.909 | 0.73, 1.30, 2.61, 3.70, 2.66 |
| tuned classical (confref 128) | **1.727** | 0.020 | 0.038 | 0.99 | **0.327** | **0.155** | **0.499** | 0.91, 1.19, 1.56, 1.39, 1.00 |
| learned E2 on top | 1.731 | **0.018** | **0.034** | 1.15 | 0.397 | 0.157 | 0.598 | 1.15, 1.36, 1.70, 1.81, 1.26 |

Result: no net benefit. The learned run has the lower score on 8/13 scenes (fire 1.57 vs 1.91,
smoke_sphere 1.90 vs 2.59) but loses clearly on 5 (smoke_push, test_100000, ood_200002 2.71 -> 3.29), and on average puts ENERGY BACK
(KE 0.99 -> 1.15, k9-16 1.39 -> 1.81, mean-velocity error worse), with a marginal plume-height gain.
Diagnosis = the textbook distribution shift: the one-step target from a restricted fine state is mostly
"the coarse step blurs the sharp restricted fine structure, put it back" (an energising correction),
while at runtime the coarse state is already over-stirred at k5-16. Decision: E3 DAgger (targets on the
corrector's own trajectory, tendency difference vs a 128^3 sim nudged to follow it).

## Planned (on hold: lead paused GPU work at this point)
* E1b noise floor: two 128^3 runs of the same scene with a 1e-4 perturbation -> how much of the remaining
  mean-density / velocity error is irreducible chaos (the 256 vs 128 numbers above suggest 0.14-0.20).
* E2 one-step supervised ("pre"): `collect --mode pre --coarse_cfg '{"vorticity_eps":3}' --seeds 0-23
  --steps 240 --every 4` (~1000 pairs, ~0.6 GB fp16 at 32^3, on D:), `train --minutes 10`, then
  `evaluate compare / stability / timing`.
* E3 DAgger: `collect --mode dagger --ckpt E2` (128^3 nudged to follow the corrected 64^3, tendency-
  difference targets on the coarse trajectory's own states), retrain from E2, evaluate.
* E4 (only if E2/E3 show benefit but drift): gain / limiter sweep at evaluation time, no new training.
