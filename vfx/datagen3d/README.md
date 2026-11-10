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
(Size lever noted at the time:) zlib halves the arrays (lr 1.9x, hr 1.7x, aux 2.2x; ~10–17 ms to inflate one
scenario), at the cost of mmap.

## s3d_v2 (2026-11-09) — 4x data, large fires close up, lossless storage
Why: on s3d_v1 the screen-space net is data-limited (hold-out error bottoms out after 3–5 min) and weakest
on large fires seen close up (vfx/train3d/EXPERIMENTS.md runs 1–3).

**Splits** (432 scenarios, 1272 sequences × 16 frames, all 256² stored px):
train 384 (seeds 0–383, v2 sampler, 3 cameras = 1152 seqs); test 24 / ood 24 = seeds 100000–100011 /
200000–200011 **imported bit-identically from s3d_v1** (same arrays, 2 cameras each, meta gains v2 tags;
`meta["imported_from"]`) + 100012–100023 / 200012–200023 new (v2 sampler, 3 cameras). Seed ranges are
disjoint per split, so no v1 test/ood scenario can be in v2 train. Slice v1-comparable results with
`imported_from == "s3d_v1"`.

**Distribution** (`sample_scenario(seed, split, sampler="v2")`): kinds campfire .16 / big_fire .16 /
multi_fire .12 / fire_obstacle .12 / torches_wind .07 / mixed .11 (35 % with a thin mid-air "slab", the
preset-4 kind) / plume_sphere .11 / random .15. Fire size classes small r .03–.05 / medium .05–.08 /
large .08–.115 m with fuel 4–9 / 6–14 / 10–22 /s, 1100–1550 K, 20 % sooty (density_rate .3–.8). Smoke
r .04–.11, 1.5–5.5 /s. fire_obstacle: 1–2 fires + 1–3 floor-standing walls / pillars / crates (0.12–0.44 m
tall) beside them, 40 % a sphere above. OOD (same three factor names as v1, ranges beyond every v2 train
range): mid-air blocks half-height .045–.09 (train slabs ≤ .04), wind .8–1.6 m/s², emitters r .12–.16
with fuel 18–30 (train ≤ .115 / 22). Exact presets never sampled (`is_preset_like`).
Note: v2 train now contains thin mid-air slabs and fires up to r .115, so v1's OOD "box_obstacle" and
"large_emitter" scenes are partly in-distribution for a v2-trained model (their tags still differ).

**Cameras** (`render.sample_cameras_v2`): 55 % from a close-up band (orbit distance log-uniform 0.8–1.35 m:
the 512-px window spans 0.21–0.35 m, so a large fire fills it), else 1.35–4.0 m; 50 % (65 % with fire) aimed
at an emitter at a height inside the flame instead of the centroid; in fire_obstacle scenes 50 % look past a
floor occluder at its fire; positions kept outside the 1 m domain; v1 motion model (38 % moving).
**Spare cameras**: 5 candidates are rendered per scenario (rendering is ~1 % of the cost) and the first 3
that pass `runner.VALID` are kept (effect visible in every frame ≥ 1 % of px and mean ≥ 2 %, frame mean
4–246 /255, hr/lr effect luminance 0.6–1.67); rejects are listed in meta `rejected_cams`. 0 sequences had
to be kept invalid (card `flagged` is empty).

**Tags** (card `tag_definitions`, counts per split in `tag_counts` / `camera_tag_counts`; `ds.tags(i)` gives
scenario + camera tags of an item):

| share | v1 train | v2 train |
|---|---|---|
| scenarios with a large fire (r ≥ 0.08 m) | 0 % | 49 % |
| cameras closeup (orbit ≤ 1.35 m) | 23 % | 54 % |
| scenarios large_fire with a closeup camera | 0 % | 44 % |
| cameras flame_dominated (≥ 25 % of effect px are flame) | — | 55 % |
| cameras "fill" (≥ 50 % of the window covered) | — | 56 % |
| multi_fire / smoke_only / occluder scenes | 21 / 30 / 0 % | 24 / 22 / 10 % |

**Storage** (`pack.py`): one `data.pack` per scenario: per frame and array, an integer predictor (lr:
x − left − up + upleft mod 256; hr: same on hr − lr; aux: same on the fp16 bit patterns mod 2¹⁶ then byte
planes split; hr_alpha: same) + zlib level 6. Lossless (round trip verified bit-exact), stdlib only.
Chunks are ordered (camera, frame, array), so a window is one contiguous read. **5.79 GB for 20.0 GB of
raw arrays (3.46x)**; plain zlib would give ~1.9–2x, lzma-1 / bz2-9 with simple filters were at
most ~10 % smaller and 5–6x slower to decode. Loading (one process, warm OS cache, CPU shared with the build + 2 training jobs):

| loader | items/s, windows of 8 |
|---|---|
| s3d_v1 npy (mmap), serial | 4180 (memcpy from page cache); 359 through a DataLoader with collate + pin |
| s3d_v2 pack, serial | 26 (≈ 4 ms per frame: inflate 1.8 ms, predictor cumsums 2.4 ms) |
| s3d_v2 pack, 4 / 8 threads (zlib + numpy release the GIL) | 81 / 147 (≈ 1200 frames/s) |

DataLoader **process** workers on this machine failed with WinError 1455 ("paging file too small") while the
other jobs ran: each spawned worker re-imports torch/CUDA DLLs. Use threads (above) or 1–2 workers.

**Guidance on large fires** (unchanged settings, `tune_guide --sampler v2 --configs unguided,c16`, 150 steps,
5 large-fire train scenes + 1 OOD triple-large-fire; evidence in the scratchpad `data3d_v2/guide_large`):
tracking max d / T / u 0.037–0.065 / 0.026–0.036 / 0.047–0.080 (unguided 0.42–0.62 / 0.27–0.50 / 0.66–1.03).
Fine-scale power vs unguided 256³, bands 16-32 / 32-64 / 64-128: density 0.64–0.89 / 0.65–0.96 /
0.74–1.31, temperature 0.84–1.19 / 0.91–1.36 / 0.91–1.53. Density detail is a bit lower than on v1's scenes
(min 0.82) in the hottest single fire (scene 31: 0.64), temperature (what makes the flame) is retained.
Renders: guided HR keeps the LR plume's shape and the unguided run's filaments. Build tracking over all 432
scenarios (rel L2 at 16³, max over steps ≥ 10): density median 0.061 / p95 0.104 / max 0.247 (6 campfires
> 0.16; the worst, a thin r 0.033 m fire, checked visually: aligned), T 0.036 / 0.081 / 0.188, velocity 0.074 / 0.150 / 0.270.

**Bug found and fixed here** (`PairRenderer._fence`): `VolumeRenderer.draw` writes the camera into one of
4 pinned host buffers and uploads it with `non_blocking=True`; with more than 4 draws queued ahead of the
GPU the CPU overwrites a slot before its copy runs and a draw renders another camera (v2 pilot: hr of one
camera showed another camera's view, hr/lr luminance 0.4–5x). We now sync before every draw. s3d_v1
(4 draws per frame) re-rendered for 2 scenarios: same views (background diff ≤ 1 code) — not affected
visibly. The render3d owner should make the ring safe for other callers.

**Build**: 2026-11-09, sim3d-warp-0.1.0, GPU shared with 2–4 training / eval jobs the whole time;
~97 min wall over 3 sessions (stopped twice by the free-disk guard while C: free space swung by ~15 GB:
the Windows pagefile grew under RAM pressure; the build now pauses below 4 GB free and resumes at 7 GB).
6–90 s per scenario (~100–200 ms per paired step shared, 85 ms on v1's quieter GPU). No non-finite runs.

## s3d_hi (2026-11-09) — sharper target on a v2 subset
`build --version s3d_hi`: the first 48 train seeds + all 24 test seeds of s3d_v2 with the **same scenarios,
cameras and crop windows** (read from v2's meta.json, `runner.run(fixed_cams=...)`), hr from a guided
**320³** sim (5x the 64³ grid per axis). lr is re-simulated in the same run: equal to v2's lr up to sim
nondeterminism (test 100000: mean |Δ| 0.03 codes). Same item format; card `hr_res` / `source`. Lives on D:
(`datasets/s3d_hi` is a junction).

**Resolution reached: 320³.** Guided pair (64³ + HR + guidance), one process, big-fire scene:
256³ 3.8 GiB / 81 ms/step, 320³ 6.9 GiB / 204 ms/step (GPU shared), 384³ ran out of device memory with
another job holding 3.6 GB (needs ~12 GiB by N³ scaling; measured 1014 ms/step while spilling into shared
system memory). 384³ would only fit on an otherwise idle card and would OOM anyone else; 352³ is not a
multiple of 64 (the guide needs an integer factor). 512³ (~27 GiB) does not fit.

**Guidance at 5x** (tune_guide `--hr-res 320`, 150 steps, unchanged settings): tracking max d/T/u big fire
0.064 / 0.037 / 0.066, smoke plume 0.078 / 0.051 / 0.152 (unguided 0.53 / 0.28 / 1.08 and 0.38 / 0.14 / 1.60);
fine power vs unguided 320³, bands 4-8 … 64-128: density 0.80–1.10 (fire), 0.88–1.88 (plume); temperature
0.89–1.46 / 0.99–2.39. Finite; max |u| 27–39 % above unguided (the run follows the LR flow).

**Does 320³ show more on screen than 256³?** Same scenario + camera, luminance high-pass energy in the
effect region (frames 0/8/15), hr320 / hr256: big-fire close-up 1.56 (σ 1 px, output px) / 1.49 (σ 1,
stored px); smoke close-up 1.14 / 1.13; far smoke 1.43 / 1.37. Visible in 1:1 crops (more filaments at the
plume edge). The gain is about the same at output and stored pixels, so the stored 256² footprint is not
the limit at these distances; no larger window stored. Over the whole built set (same windows, mask = v2
effect region, σ 1 stored px): hr320 / hr256 = 1.17 (train) / 1.22 (test), finer in 97 % of 204
sequences; hr/lr = 2.0 (vs 1.6–1.7 for 256³).

**Build**: 48 train (144 seqs) + 24 test (60 seqs), no ood split (empty list); **0.89 GB, 2035 s** (GPU
shared; ~200–300 ms per paired step). Layer 3 (`VFX_DATASET3D=s3d_hi`): 9 passed, 1 skipped (no OOD),
1 xfailed, 1 failed = the same close-up motion bound as v2 (identical cameras).

## What the training side must know
* **s3d_v2**: decompressed train is 18.1 GB (1152 seqs × 15.7 MB) — it does not fit on the GPU like
  `GPUWindows` loads v1 (2.9 GB). Stream with threads, or hold a rotating subset. Cameras per scenario vary
  (3 new, 2 imported): use `ds.sequences` / `ds.seq_scenario` instead of assuming 2 per scenario (the
  `seq_range` hold-out in `vfx/train3d/data.py` assumes 2). Hold-out by scenario seed, not by sequence index.
* **s3d_v2 close-ups carry large stored motion**: p99 |motion| (alpha > 0.05) exceeds 50 stored px/step in
  118 / 1272 sequences, max 86 (v1 max 62). It is real: hot gas at 2–3 m/s seen from 0.8–0.9 m is ~1100
  stored px/m. Normalize motion inputs accordingly (do not assume |motion| < 64).
* Layer 3 (`VFX_DATASET3D=s3d_v2 pytest tests3d/test_data3d_layer3.py`): 9 passed, 1 xfailed (stored
  motion, known), 2 failed: `test_size_and_card` (5.79 GB vs the v1-era 4 GiB limit; v2's budget is 6 GB)
  and `test_no_nan_inf_and_aux_ranges` only on its motion bound (p99 < 0.25 x 256 px/step; the close-up
  sequences above). Alignment 103/103 frames (0,0); coarse corr median 0.997; hr/lr fine energy 1.93x.
* selfcheck (`--max-scn 60`: all 48 test/ood + 60 train scenarios): splits, ranges, size, loader, motion
  sign (static 91 % / moving 92 %), camera-only motion warp gain 5.9x all ok. Alignment first showed 4/781
  frames with a ±3 px "best shift": close-ups whose only medium-free pixels are plain background (no
  geometry; checked visually, same view). The check now needs a shift to beat no-shift by ≥ 10 % (a real
  2-px shift is still caught) -> 781/781.
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
