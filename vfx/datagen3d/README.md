# vfx/datagen3d — route-B (screen-space detail) dataset factory

Owner: 3D data engineer. Each scenario runs the **64³ runtime sim** (exactly the sandbox's solver
setting) and a **256³ sim guided to follow it**, then renders both with the sandbox renderer from the same
cameras. Output: versioned uint8 image sequences + LR aux buffers under `DATA_ROOT/datasets/<version>/`.

```
.venv/Scripts/python -m vfx.datagen3d.build --version s3d_v1 [--plan]     # resumable, prints ETA
.venv/Scripts/python -m vfx.datagen3d.build --version s3d_v1 --card       # rewrite card.json only
.venv/Scripts/python -m vfx.datagen3d.selfcheck --version s3d_v1 [--png-dir DIR]
.venv/Scripts/python -m vfx.datagen3d.selfcheck --mini                    # tiny build (s3d_selfcheck) + checks
.venv/Scripts/python -m vfx.datagen3d.tune_guide --out DIR                # guidance sweep (evidence below)
.venv/Scripts/python -m vfx.datagen3d.build --version s3d_v2              # v2 (see "s3d_v2" below), resumable
.venv/Scripts/python -m vfx.datagen3d.tune_guide --out DIR --sampler v2 --configs unguided,c16 --scenes train:17,...
```

| Module | Content |
|---|---|
| `scenarios.py` | `Scenario3D`, `sample_scenario(seed, split, frames=16, sampler="v1"\|"v2")`, `scene_tags`; all ranges |
| `guide3d.py` | `GuideConfig3D`, `Guide3D` (velocity + multiplicative scalar relaxation), `tracking_error` |
| `sim.py` | `run_pair`: LR 64³ (runtime config) + HR 256³ (full quality, guided), solver/buffer cache |
| `render.py` | runtime constants, orbit cameras (`sample_cameras` v1, `sample_cameras_v2`), crop policy, `PairRenderer` |
| `runner.py` | `run(scenario, out_dir, cam_policy, n_spare, storage)` writes one scenario dir (meta.json last); camera validity + tags; `load_arrays` |
| `pack.py` | lossless per-frame compressed scenario container `data.pack` (stdlib zlib + numpy) |
| `build.py` | CLI, seed splits, v1-scenario import, size / free-disk guards, card.json |
| `dataset.py` | `load_sequences(version, split, length, stride=1)` (npy or pack), `ds.tags(i)`, `ds.meta(i)` |
| `selfcheck.py`, `tune_guide.py`, `png.py` | checks, guidance sweep, tiny PNG writer (no PIL in the venv) |

## Guidance (`guide3d.py`, port of `vfx/datagen/guide.py`)
Every step: before the HR step `vel += k_vel·L(vel_lr − pool4(vel_hr))` (the HR projection removes the
introduced divergence); after it `density, T−amb *= exp(k_s·L(clamp(log ratio, ±1)))` (multiplicative:
rescales the HR structure, never paints haze); fuel unguided. `L` = `exp(-(k/cutoff)^8)` spectral low-pass
on the mirror-extended 64³ grid, then trilinear ×4. **Default (s3d_v1): k_vel 1.0, k_scalar 0.7, cutoff 16
cycles/domain** (64³ Nyquist 32), eps 0.01 / 5 K.

Sweep (`tune_guide`, 6 scenes: 2 campfire, 2 torches+wind+sphere, 1 smoke+fire mix, 1 OOD strong wind +
large emitters; 150 steps; sandbox-like dissipation). Tracking = rel L2 at 16³ of HR vs the 64³ run, over
all steps ≥ 10. Spectrum = 256³ density power per band (cycles/domain), guided / unguided, geo-mean over
scenes, mean of the last 30 steps:

| setting | worst track max d / T / u | median track mean d / T / u | power 4-8 / 8-16 / 16-32 / 32-64 / 64-128 |
|---|---|---|---|
| unguided 256³ | 0.71 / 0.49 / 1.24 | 0.46 / 0.23 / 0.79 | 1 / 1 / 1 / 1 / 1 |
| vel only, c12 | 0.49 / 0.28 / 0.15 | 0.29 / 0.16 / 0.10 | 0.87 / 0.85 / 0.92 / 0.95 / 1.05 |
| weak (0.5 / 0.35), c12 | 0.26 / 0.17 / 0.21 | 0.15 / 0.10 / 0.13 | 1.05 / 0.99 / 1.08 / 1.03 / 1.11 |
| c8 | 0.34 / 0.24 / 0.40 | 0.18 / 0.10 / 0.25 | 1.08 / 1.09 / 1.15 / 1.09 / 1.14 |
| c12 | 0.18 / 0.12 / 0.16 | 0.10 / 0.07 / 0.10 | 1.07 / 0.98 / 1.08 / 1.07 / 1.19 |
| k_scalar 1.0, c12 | 0.15 / 0.11 / 0.16 | 0.09 / 0.05 / 0.10 | 1.10 / 1.00 / 1.14 / 1.14 / 1.30 |
| **c16 (default)** | **0.12 / 0.09 / 0.11** | **0.07 / 0.04 / 0.07** | **1.09 / 0.96 / 1.07 / 1.13 / 1.28** |

Unguided, the 256³ plume rises faster and thinner than the 64³ one (as the solver engineer measured) and
drifts (density error 0.5–0.7 at 16³). Velocity guidance alone fixes velocity but not density/heat (the
64³ transport is more diffusive). With multiplicative scalars + velocity at cutoff 16, HR follows the 64³
shape (renders: same plume outline, height and lean) while keeping — slightly increasing — fine-scale
power (no band below 0.82 in any scene; forcing drives extra roll-ups, as in 2D smoke). No grid artifacts
in slices or renders. Stable: all runs finite, max |u| within 20 % of unguided. Cost: +5–10 ms/step on
256³ (~70–80 ms/step idle-ish GPU). 192³ fallback was not needed.

## Rendering, resolution and crop policy (`render.py`)
Runtime footprint matched: the sandbox renders **2560×1440, march_scale 0.5** (default `RenderParams3D`,
192 steps/m) and ScreenDetail processes the effect box at **process_scale 0.5** (2×2 avg-pool). We render the
full 1440p frame with those exact settings (obstacles rasterized at 128³ as in the sandbox, same solid
for LR and HR), cut a fixed **512×512 output-px window** per camera sequence around the effect and
**2×2-average-pool it to 256×256**. So one stored pixel = one runtime processed pixel = one march ray, with
the exact runtime perspective; a fully convolutional net trained on these crops sees the runtime pixel
scale. The window is fixed over the sequence (motion vectors stay in one pixel frame); its centre is the
mean projection of the orbit target (effect centroid ± 5 cm) ± 20 % jitter, clamped to the frame.

Cameras (sandbox `Orbit` parametrization, fov 40°): 2 per scenario, yaw spread around the effect, pitch
−0.1..1.1 rad, distance log-uniform 0.9–4.0 m (a 256-px crop spans ~0.23 m close up, ~1 m = whole box
far away). 40 % of sequences move: yaw rate ±0.15–0.7 rad/s, pitch rate ±0.15 rad/s, dolly
±0.25 /s (log-distance), per sim step.

## Format — `DATA_ROOT/datasets/<version>/`
```
card.json      counts, seed ranges, param ranges, guidance, renderer settings (all RenderParams3D), solver
               version, tracking summary, list of moving-camera sequences, bytes, build_seconds
splits.json    {"train": [seeds], "test": [...], "ood": [...]}    train 0.., test 100000.., ood 200000..
<split>/scn_<seed:06d>/
  lr.npy        (K,T,3,256,256) uint8   render of the raw 64^3 sim (sRGB, tone-mapped, as displayed)
  hr.npy        (K,T,3,256,256) uint8   render of the guided 256^3 sim, identical camera/crop/settings
  aux.npy       (K,T,4,256,256) float16 LR render: alpha 0..1, depth (m from camera along the ray),
                                        motion_x, motion_y in STORED px per SIM_DT (x right, y down)
  hr_alpha.npy  (K,T,256,256)   uint8   HR render alpha x 255 (effect masks for eval)
  meta.json     scenario, tags, solver version, guidance, tracking_16 (per step), per camera: orbit params,
                moving flag, crop_xy_out, per-frame camera (index 0 = one step BEFORE frame 0),
                domain_bbox_out (what the sandbox passes ScreenDetail), alpha / motion stats
                (v2 adds: per-camera "stats", "tags", "flagged"; scenario "rejected_cams", "storage",
                "cam_policy", "imported_from" for scenarios copied from v1)
  data.pack     (s3d_v2: REPLACES the four .npy) the same four arrays, lossless, see "s3d_v2 storage"
```
K = 2 cameras, T = 16 consecutive sim steps (frame t−1 → t is one SIM_DT; frame i = state after solver
step `start_step + i`). aux is pooled from the renderer's output-res aux: alpha = mean, depth and motion =
alpha-weighted mean (0 where alpha = 0), motion × 0.5 (output px → stored px).
`load_sequences(version, split, length, stride=1)` items: `lr`, `hr` (L,3,256,256) uint8, `aux`
(L,4,256,256) fp16, `hr_alpha` (L,256,256) uint8, `seed`, `cam`, `t0`, `moving`. mmap'ed, lazily per
worker. `ds.meta(i)` returns the scenario's meta.json.

## Scenario distribution (`scenarios.py`)
Kinds (p): campfire 0.2 (1 fire, optional floor "hearth" slab like preset 1), plume_sphere 0.2 (smoke
plume + sphere above it, preset 2), torches_wind 0.15 (2 fires + wind 0.2–0.5 m/s², preset 3),
mixed 0.15 (smoke + fire, preset 4 without its box), random 0.3 (1–3 emitters, fire/smoke/mixed,
emitters up to 0.3 m high). 0–2 spheres (r 0.06–0.14), mild horizontal wind ≤ 0.5 m/s², 25 % with a
periodic impulse "push" (0.5–2 m/s every 6–20 steps). Sim: density_dissipation 0.8–1.8 /s, cooling
3.5–7 /s (sandbox 1.2 / 5), applied to both resolutions. Warm-up 30–150 steps. Fire emitters 1150–1400 K,
fuel 5–12 /s; smoke 520–700 K, 1.8–4.5 /s (preset magnitudes). The exact presets are never sampled
(continuous draws + an `is_preset_like` assert).
OOD (each factor p 0.6, ≥ 1): in-flow box obstacles (half-extents 0.05–0.22 × 0.02–0.08, centre y
0.3–0.7), strong wind 0.8–1.6 m/s² (± 0.2 vertical), large emitters (r 0.10–0.15, rates ×1.3–2).
Deviation from the brief: a floor-level hearth slab (box) is allowed in train under campfires because
preset 1 has one; in-flow boxes stay OOD.

## s3d_v1 build (2026-11-09, sim3d-warp-0.1.0, RTX 5080, GPU shared with other jobs)
96 train / 12 test / 12 ood scenarios × 2 cameras × 16 frames = 240 sequences (98 moving), 1728 train
windows of 8. **3.77 GB, 1259 s** (~85 ms per paired step incl. guidance and rendering). No non-finite runs.
Tracking (rel L2 at 16³, per-scenario max over all steps ≥ 10): density median 0.069 / max 0.160,
T median 0.047 / max 0.147, velocity median 0.094 / max 0.169. selfcheck PASS: lr/hr best shift (0,0) in
696/696 frames; camera-only motion warp gain 5.9x; fluid motion: see below.
Size lever for v2: zlib halves the arrays (lr 1.9x, hr 1.7x, aux 2.2x; ~10–17 ms to inflate one
scenario), at the cost of mmap.

## What the training side must know
* Frames are final display images (ACES + sRGB + dither, bloom); `lr` is what ScreenDetail's
  `pool(rgb)` sees at process_scale 0.5. At runtime, the aux from the 1440p render must be pooled the same
  way and motion halved to match the stored units.
* **Motion vectors**: camera motion is exact (selfcheck: frozen volume + moving camera, warp improves MSE
  ~6x). The fluid part (opacity-weighted gas velocity) does NOT predict how the rendered image changes:
  a fed plume's shape is quasi-steady near the source (any warp hurts there), and in the turbulent upper
  plume the pattern moves at ~0.25–0.5x the stored motion. Warping lr[t−1] by the full motion is worse
  than no warp on static cameras. Feed motion as an input feature / use learned or partial warping; do not
  hard-warp the previous output by it. Background and floor pixels have motion 0 even when the camera moves
  (renderer contract); camera-only reprojection can be computed from the per-frame cameras + depth in meta.
* lr/hr differ only by the volume: obstacles, floor and background line up to the pixel; floor shadows
  and fire glow differ (they come from the volume).
