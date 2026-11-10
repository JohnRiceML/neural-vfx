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

## E3 - one DAgger round  [training run 2 of 4]
Hypothesis: targets collected on the corrected coarse trajectory (inputs = the coarse states the corrector
really sees; target = low-passed, projected [128^3 tendency] - [uncorrected 64^3 tendency], with the 128^3
sim nudged by guide3d (k_vel 1, k_scalar 0.7, cutoff 8) to share the coarse large scales) remove the
energising bias of E2.
Data `corr3d_r1` (D:): 24 train scenes rolled out with E2, 1103 pairs, 0.6 GB; target rms 0.0004-0.0043
(about 3x E2's). Trained on r0 + r1 from E2's weights, 10 min, 180k iters: **train R^2 0.14, val 0.12**.
The DAgger targets are mostly unpredictable from the local coarse state (they include the chaotic mismatch
between the nudged fine sim and the coarse one), so the net learns only their conditional mean.

| (13 held-out scenes) | score | centroid err | h95 err | KE ratio | spec err k1-8 | mean-D err | mean-U err | bands k1-2..17-31 |
|---|---|---|---|---|---|---|---|---|
| tuned classical (this eval) | 1.709 | 0.020 | 0.037 | 0.99 | 0.333 | 0.160 | 0.502 | 0.94, 1.22, 1.56, 1.41, 1.01 |
| learned E3, gain 1 | 1.652 | 0.021 | 0.037 | **0.83** | **0.147** | 0.171 | **0.427** | 0.79, 0.91, 1.07, 1.19, 0.90 |

Result: the sign of the correction flipped, as intended. It now DAMPS the over-stirred band: k5-8 goes from
1.56x to 1.07x, the spectrum error halves, and the time-mean velocity error falls 15 %. But it overshoots.
Large scales are drained (k1-2 0.79, KE 0.83), plume height is unchanged, the mean density is slightly
worse, and it wins only 6/13 scenes. Net: -3 %, which is inside the evaluation noise. Note that the same
classical runs scored 1.709 / 1.727 / 1.752 in three evaluations: the cached solver's warm-started
pressure carries over between scenes, so scene order perturbs the chaotic runs, giving about +-0.3 per scene
and about +-0.03 on the mean.

## E4 - correction gain 0.5 (no training; evaluation-time knob)
Hypothesis: E3 is directionally right but too strong; halving its output balances damping vs drain.
Caveat: the gain was picked after looking at the held-out set (two values tried), so this is mildly
optimistic.

| (13 held-out scenes) | score | centroid err | h95 err | KE ratio | spec err k1-8 | mean-D err | mean-U err | bands |
|---|---|---|---|---|---|---|---|---|
| raw 64^3 | 4.61 | 0.056 | 0.097 | 1.83 | 0.73 | 0.245 | 0.91 | 0.73, 1.30, 2.61, 3.70, 2.66 |
| tuned classical (this eval) | 1.752 | 0.020 | 0.039 | 0.99 | 0.335 | 0.161 | 0.508 | 0.92, 1.19, 1.57, 1.41, 1.01 |
| **learned E3 x0.5** | **1.563** | **0.019** | **0.034** | 0.89 | **0.195** | 0.162 | **0.439** | 0.84, 1.02, 1.26, 1.29, 0.93 |
| noise floor (128 vs 128') | 0.45 | 0.004 | 0.013 | 1.00 | 0.026 | 0.080 | 0.246 | |

Result: -11 % score vs the tuned baseline, with the lower score on 10/13 scenes. It wins on all 3 OOD
scenes and on ood_box_wind (1.84 -> 1.47); it loses slightly on fire_wind, smoke_push and test_100000. The
gain comes almost entirely from the velocity spectrum (k5-8 excess 1.57 -> 1.26) and the mean velocity
field. The plume-height and mean-density errors, the visible symptom, are no better than the
one-parameter fix. Of the reducible gap (1.75 - 0.45 = 1.30), the corrector closes about 0.19, or 15 %.


## Stability - E3 at gain 0.5, 3000 steps (100 s) with user perturbations (`evaluate stability`)
Events: a random push of 1-4 m/s every 37 steps; all emitters off for steps 900-1290; a 0.4 x 0.06 x 0.3 m
box toggled into the flow for steps 1590-2190; a 1.7 m/s^2 wind gust for steps 2400-2610.
Scenes: fire, smoke_sphere, ood_box_wind, test_100001, ood_200002. The learned runs use confref 128 + the
corrector; the "raw" runs are the default 64^3.

| | PCG (full quality) | fused pressure (`fused_config` settings) |
|---|---|---|
| finite / blow-up | 5/5 finite, max |u| at end 1.1-1.6 m/s (raw 1.2-1.5) | 2/2 finite, 1.2-1.4 m/s |
| energy drift: KE learned / raw per 500-step sixth | flat at 0.74-0.86 in every scene, no trend | flat at 0.77-0.82 |
| max normalized divergence after projection | <= 2.6e-5 (raw <= 2.7e-5) | <= 7.0e-5 (gate 1e-4 met) |
| limiter fallback triggered | 0 % of steps | 0 % |
| correction rms / velocity rms | 0.4-0.5 % (max rms 7e-4 m/s/step) | same |

Smoke mass stays within 3 % of the raw run in every sixth (0-3 %: the flow differs, not the budget). The constant KE offset of about 0.8 is the
corrector (plus confref) removing the 64^3 over-stirring; it does not grow over time. **The corrector
composes with the fused pressure solve**: it was trained with PCG, and with fused it still meets the
divergence gate with the same damping.

## Cost (`evaluate timing`, 64^3 plume + sphere, CUDA events, median of 200 steps)
GPU almost idle at the start (6 % utilization, 0.6 GB; one other light job shared the card), so this is
close to the real cost; the lead should re-run it on an idle card.
**Corrector 0.22 ms/step (p90 0.22) as one captured CUDA graph, FP16**; 0.57 ms eager (launch-bound).
The PCG step in the same run took 1.03 ms. So PCG + corrector = 1.25 ms, and fused (0.48-0.62 ms, solver
README) + corrector is about 0.7-0.85 ms, both inside the 2.0 ms sim budget. The pieces: 2x pool to 32^3,
5 convolutions at 32^3 / 16^3 (37k params), trilinear x2, limiter (two global reductions, no host sync).

## Verdict
Feasible on cost and stability. Benefit over the cheap classical fix is real but small. The measured gap is
mainly the confinement scaling (eps*dx), and one opt-in parameter (`confinement_ref_res=128`) removes
about 63 % of it. The learned corrector on top removes another 11 % of the score (15 % of what remains
above the chaos floor). That comes from the velocity spectrum and the mean flow, NOT the plume height or
the visible density, and the images show no difference a player would notice between tuned-classical and
learned. One-step supervised training made things worse (energising, E2). One DAgger round fixed the sign
but leaves a weakly predictable target (R^2 0.12), so the net learns a blunt "damp k5-16" filter that
needs a hand-tuned gain. Not worth adopting now. A real step beyond classical would need training through
the unrolled solver (multi-step loss on time-averaged statistics), which the Warp solver does not support
today, or a 256^3 target, since 128^3 is not converged at large scales either.
Training runs used: 2 of 4 (E2, E3). Datasets on D: corr3d_r0 + corr3d_r1 = 1.2 GB (kept for
reproducibility; delete `D:\neural-vfx-data\datasets\corr3d_*` if not needed).
