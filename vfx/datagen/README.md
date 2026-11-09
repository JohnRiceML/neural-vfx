# vfx/datagen — paired LR/HR dataset factory

Owner: data engineer. Randomized scenarios -> 512² solver runs (+ independent 128² runs for
eval splits) -> versioned float16 `.npy` datasets under `DATA_ROOT/datasets/<version>/`.

```
.venv/Scripts/python -m vfx.datagen.build --version v1                 # defaults below
.venv/Scripts/python -m vfx.datagen.build --version v1 --plan          # size + time estimate only
.venv/Scripts/python -m vfx.datagen.selfcheck [--fake] [--keep] [--png-dir DIR]
```

| Module | Content |
|---|---|
| `scenarios.py` | `sample_scenario(seed, split, frames=8)`; all randomization ranges |
| `runner.py` | `run(scenario, out_dir=None, frames=None, with_lr_sim=None)` -> meta dict |
| `build.py` | CLI, seed splits, card, resumable loop with ETA |
| `dataset.py` | `load_dataset(version, split="train")` -> `PairedDataset` |
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
  lr_sim.npy     (T, 5, 128, 128) float16   independent 128² run, same steps (test/ood only)
```
Channel order is `vfx.core.fields.CHANNELS` (density, vel_x, vel_y, temperature, fuel); y = 0 is
the bottom row. Frame i is the solver state after step `start_step + i` (1 step = `SIM_DT`), so
frame t-1 -> t is exactly one step. `lr.npy` pools *all* cells including solids.

Flags (also listed in card.json; flagged scenarios are kept but listed, never silently dropped):
`nonfinite`, `empty` (HR density max < 1e-3), `out_of_range` (density > 20, T > 5000 K or
T < ambient - 50 K).

## Loader
`load_dataset(version, split)` returns a `torch.utils.data.Dataset`; item = dict of float32
tensors `lr (5,128,128)`, `hr_prev (2,512,512)`, `hr (2,512,512)`, `solid_lr`, `solid_hr`, plus
ints `seed`, `t` (t >= 1). test/ood items also have `lr_sim (5,128,128)` (pass
`include_lr_sim=False` to drop it). Arrays are `np.load(mmap_mode="r")`, opened lazily per
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
