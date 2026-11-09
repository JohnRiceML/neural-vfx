# vfx/datagen — paired LR/HR dataset factory

Owner: data engineer. Randomized scenarios -> 512² solver runs (+ independent 128² runs for
eval splits) -> versioned float16 `.npy` datasets under `DATA_ROOT/datasets/<version>/`.

```
.venv/Scripts/python -m vfx.datagen.build --version v1                 # defaults below
.venv/Scripts/python -m vfx.datagen.build --version v2                 # guided, 16 frames (preset)
.venv/Scripts/python -m vfx.datagen.build --version vX --guided        # guided HR, any version name
.venv/Scripts/python -m vfx.datagen.build --version v1 --plan          # size + time estimate only
.venv/Scripts/python -m vfx.datagen.build --version v1 --reflag        # recompute card flags, no sim
.venv/Scripts/python -m vfx.datagen.selfcheck [--fake] [--keep] [--png-dir DIR]
```

| Module | Content |
|---|---|
| `scenarios.py` | `sample_scenario(seed, split, frames=8)`; all randomization ranges |
| `runner.py` | `run(scenario, out_dir=None, frames=None, with_lr_sim=None, guide=None)` -> meta dict |
| `guide.py` | `GuideConfig`, `simulate_guided`: 512² run relaxed toward the independent 128² run (v2) |
| `build.py` | CLI, seed splits, version presets, card + outlier flags, `--reflag`, resumable loop with ETA |
| `dataset.py` | `load_dataset(version, split)` -> `PairedDataset`; `load_sequences(version, split, length)` -> `SequenceDataset` |
| `selfcheck.py` | 3-scenario end-to-end check (fake solver if `vfx.sim` is not importable) |

## Defaults (v1)
500 train / 32 test / 32 ood scenarios, 8 frames each (7 training pairs per scenario).
Size: 1.21 MB per train frame, 1.37 MB per test/ood frame -> **~5.6 GB**. The build refuses
anything estimated over 6 GB unless `--force-size`. Build time: `--plan` times the solver
(measured 2026-11-09, sim2d-warp-0.1.0, RTX 5080: 13.5 ms/step at 512², 8.2 ms at 128²;
50k HR + 5.6k LR steps -> ~12 min). The build prints progress and ETA.

Measured LR(pooled HR) vs independent LR sim disagreement (relative L2 after a further 4x pool
to 32²): ~0.05–0.7 (median ~0.35) over 12 test/ood scenes, worst for fire (0.5–0.75) — the two sims drift apart (chaotic
flow), so use `lr_sim` as the honest eval input, not as a ground-truth pair.

Resumable: a scenario dir counts as done when `meta.json` and all its `.npy` exist
(`meta.json` is written last, files are written via temp + rename). Rerun the same command.

## v2 — guided HR (input = independent 128² sim)
Why: v1 pairs are (avgpool4(HR) -> HR), but the app feeds an independent 128² sim that drifts from
the 512² run (rel L2 0.05–0.9 at 32²), so a v1-trained net sees a different input distribution
live (flicker 2.8x HR ref on lr_sim). v2 runs the 128² sim freely (`lr_sim`, the runtime input)
and makes the 512² run follow it at large scales, so (lr_sim -> hr) is a consistent pair.

Scheme (`guide.py`, `GuideConfig()` defaults = v2): every step
- before the HR step: `vel_hr += k_vel * L(vel_lr - avgpool4(vel_hr))`; the solver's projection
  then removes the divergence, so saved velocities are solver output;
- after the HR step: `s *= exp(k_scalar * L(log((lp(s_lr)+eps)/(lp(avgpool4(s_hr))+eps))))` for
  s = density and T - ambient (multiplicative: rescales existing HR structure, never paints new
  smoke); fuel not guided;
- `L` = spectral low-pass `exp(-(k/cutoff)^8)` (k in cycles/domain) on the mirror-extended 128²
  grid, then bilinear x4. Solids untouched.
- **v2: k_vel 1.0, k_scalar 0.7, cutoff 32, eps 0.01 / 5 K.** Recorded in card.json and every
  meta.json (`guidance`), plus per-frame tracking error in meta.json (`tracking`).

How k was chosen (22 scenes = 16 test + 6 ood, 16 frames, rel L2 of avgpool4(lr) vs avgpool4(lr_sim)
at 32²; spectrum = radial power of HR density, geo-mean ratio guided/unguided per band):

| setting | density max (median) | T max | vel max | fire k 64–128 / 128–256 | smoke k 64–128 / 128–256 |
|---|---|---|---|---|---|
| unguided | 0.92 (0.52) | 0.55 | 0.91 | 1 / 1 | 1 / 1 |
| additive all, k 0.5/0.2, gauss σ1 | 0.14 (0.07) | 0.19 | 0.09 | 0.75 / 0.70 | 1.03 / 1.03 |
| additive scalars only, k 0.3 | 0.39 (0.15) | 0.38 | 0.94 | 0.40 / 0.49 | 0.60 / 0.66 |
| velocity only, k 0.8, c24 | 0.29 (0.19) | 0.24 | 0.09 | 1.06 / 0.93 | 1.37 / 1.47 |
| vel 0.8 + mult 0.5 pre-step, c24 | 0.14 (0.06) | 0.15 | 0.09 | 0.94 / 0.90 | 1.19 / 1.36 |
| **vel 1.0 + mult 0.7 post-step, c32 (v2)** | **0.073 (0.041)** | **0.086** | **0.065** | **0.88 / 0.92** | **1.19 / 1.49** |

Additive scalar nudging paints smooth haze over the HR and removes 25–60 % of its fine-scale power
(seen in PNGs); multiplicative scalar + velocity guidance keeps the detail. Expected losses: the
16–64 band drops (fire 0.62, smoke 0.79) because those scales now follow the smoother 128² sim;
smoke gains fine-scale power (forcing drives extra roll-ups; looks like plausible turbulence in
PNGs, no grid artifacts). Stability: 450-step runs of 4 scenes (fire/smoke, ood) stay finite,
solver divergence <= 2e-5, tracking 0.015–0.05. The jagged speckled flame column seen in fire
scenes is in the unguided 512² run too (solver MacCormack/combustion), not caused by guidance.

v2 build (2026-11-09, sim2d-warp, RTX 5080): 200 train / 32 test / 32 ood (same seeds,
splits and OOD definition as v1), 16 frames each (15 pairs; 3000 train pairs), **5.89 GB, 98 s**.
Tracking over all 264 scenes (per-scene max over frames): density median 0.036, p95 0.062, max
0.112 (train/scn_000113, a flagged dense-smoke outlier; fire max 0.080); T max 0.077; vel max 0.064.

## Format — `DATA_ROOT/datasets/<version>/`
```
card.json        version, created, solver_version, counts, seed_ranges, frames_per_scenario,
                 param_ranges, flagged {"split/scn_x": [flags]}, bytes, build_seconds
splits.json      {"train": [seeds], "test": [seeds], "ood": [seeds]}
<split>/scn_<seed:06d>/
  meta.json      scenario (dataclasses.asdict), split, solver_version, lr_res, hr_res, dt, frames,
                 start_step, tags, channels, has_lr_sim, stats (per-channel min/max/mean), flags
  hr.npy         (T, 2, 512, 512) float16   density, temperature (K)  — target
  lr.npy         (T, 5, 128, 128) float16   HR full state 4x average-pooled — training input
  solid_hr.npy   (512, 512) uint8           solver's solid mask
  solid_lr.npy   (128, 128) uint8           rasterize_obstacles(obstacles, 128)
  lr_sim.npy     (T, 5, 128, 128) float16   independent 128² run, same steps (v1: test/ood only; v2: all)
```
v2 differences: T = 16; `lr_sim.npy` in every split; `hr.npy` is the GUIDED 512² run and `lr.npy`
is still avgpool4 of it (so `lr` ~ `lr_sim` at large scales but not equal); meta.json has
`guidance` and `tracking` {density, temperature, velocity: per-frame rel L2 at 32²}; card.json
has `guidance`, `lr_sim_splits`, `tracking_rel_l2_32`.
Channel order is `vfx.core.fields.CHANNELS` (density, vel_x, vel_y, temperature, fuel); y = 0 is
the bottom row. Frame i is the solver state after step `start_step + i` (1 step = `SIM_DT`), so
frame t-1 -> t is exactly one step. `lr.npy` pools *all* cells including solids.

Flags (listed in card.json `flagged`; flagged scenarios are kept but listed, never silently dropped):
`nonfinite`, `empty` (HR density max < 1e-3), `out_of_range` (density > 20, T > 5000 K or
T < ambient - 50 K) — these are also in meta.json. Statistical outliers (card only, rule in
card `outliers`): `outlier_max_density` / `outlier_max_T` / `outlier_max_speed` when HR max density,
HR max T or LR max |vel_x| exceeds median + 8 MAD over all scenarios of the version (the
Layer 3 rule). These are real (dense smoke, hottest fires), not broken: do not drop them blindly.
`--reflag` recomputes the card flags from meta.json stats without simulating. v1 after reflag:
40 flagged (30 density, 10 T); v2: 26 (11 density, 15 T).

## Loader
`load_dataset(version, split)` returns a `torch.utils.data.Dataset`; item = dict of float32
tensors `lr (5,128,128)`, `hr_prev (2,512,512)`, `hr (2,512,512)`, `solid_lr`, `solid_hr`, plus
ints `seed`, `t` (t >= 1). test/ood items also have `lr_sim (5,128,128)` (pass
`include_lr_sim=False` to drop it); in v2 every split has `lr_sim`.
`load_sequences(version, split, length)` yields windows of `length` consecutive frames (all
offsets): `lr`, `hr`, `lr_sim` as (L, C, H, W), `solid_lr`, `solid_hr`, `seed`, `t0` — for
multi-step self-feedback rollouts (v2: length <= 16). Arrays are `np.load(mmap_mode="r")`, opened lazily per
worker, so it works with `DataLoader(num_workers>0)` on Windows. Incomplete scenario dirs are
skipped.

## Randomization (all world units; exact numbers in `scenarios.py`, copied into card.json)
Seeds: train `0..`, test `100000..`, ood `200000..` (disjoint by construction). The scenario is a
pure function of `(seed, split)`.

Train and test (same distribution):
- 1–3 emitters, x in [0.2, 0.8] m, y in [0.08, 0.30] m (lower part of the 1 m box), radius
  0.025–0.06 m, imposed velocity x ±0.2, y 0–0.6 m/s.
- 50% **fire** scenes (all emitters): T 900–1500 K, fuel_rate 1–4 /s, density_rate 0.1–0.6 /s.
  50% **smoke** scenes: T ambient+30..+200 K, no fuel, density_rate 0.5–2 /s.
- Mild wind: x in ±0.3 m/s², y 0.
- 0–2 **circle** obstacles, radius 0.04–0.10 m, centered in x [0.15, 0.85], y [0.40, 0.80],
  never overlapping emitters.
- Warm-up 20–140 steps (0.7–4.7 s) before the first saved frame, uniform, so young and developed
  plumes are both represented (tag `early` / `late`).

**OOD ("held-out scene types")** — each OOD scene enables a non-empty random subset of three
factors (each p = 0.6), tagged by name; everything else as train:
- `box_obstacle`: 1–2 axis-aligned boxes (half-size 0.04–0.15 × 0.02–0.08 m). Boxes never occur
  in train/test.
- `strong_wind`: |wind_x| 0.6–1.2 m/s², wind_y ±0.3 m/s² (train max |wind_x| is 0.3).
- `large_emitter`: radius 0.07–0.11 m, upward velocity 0.4–1.0 m/s.

Tags: `[split, fire|smoke, early|late, circle_obstacle?, <ood factors>...]`.
