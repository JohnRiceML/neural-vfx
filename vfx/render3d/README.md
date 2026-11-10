# vfx/render3d — volume ray-marcher (smoke + fire)

```python
from vfx.render3d import VolumeRenderer
r = VolumeRenderer((1440, 2560), march_scale=0.5)          # out_size = (h, w)
img = r.draw(volume, camera, solid)                         # (h, w, 3) uint8 cuda, row 0 = top
img = r.draw(volume, camera2, solid, reuse_volume=True)    # camera-only redraw of the same volume
r.timings                                                   # {'prep','view','total'} ms of the last
                                                            # finished draw (resolved lazily, no sync)
img, aux = r.draw(volume, camera, solid, return_aux=True,   # route B aux buffers (see "Aux buffers")
                  velocity=state.velocity, prev_camera=cam_of_previous_sim_step)
```
`volume`: `fields3d.Volume`, any resolution, fp32 or fp16. `solid`: optional (D,H,W) mask at any
resolution (the app passes obstacles rasterized at 128³). `RenderParams3D` holds the look
(`steps` = ray samples per meter, `light_res`, extinction, emission, colours); `set_params(**kw)`.

## Pipeline (all GPU, no host sync; one tiny pinned H2D copy of the camera)
Volume prep (steps 1-2a..c) is skipped when `reuse_volume=True`; the view (march, bloom, composite)
runs every draw. Each part is one CUDA-graph replay.
1. **pack** (direct launch): density/temperature → half2 volume + 32³ occupancy bricks (1-voxel apron).
2. **one CUDA-graph replay** per (volume shape, has-solid):
   * upload the half2 volume into a `wp.Texture3D` → hardware trilinear filtering. Measured ~1.8x
     faster than manual trilinear on array loads for the same march (256³, 720p, 256 steps:
     1.51 vs 2.68 ms). torch `grid_sample` was ruled out without a full kernel: it would have to
     materialize every sample position (3.7 M rays × hundreds of steps).
   * **light volume** (64³): transmittance toward the key light (march to the cube exit, smoke +
     obstacles), sky occlusion (short march up), local blackbody emission.
   * **fire glow** (32³ + 8³ blurred emission pyramid): lights smoke, obstacles and the floor.
   * **march** at `march_scale` × output res: ray/box intersection with the unit cube, jittered
     (interleaved-gradient-noise) start, empty-brick skipping on the step lattice, front-to-back
     compositing, early termination (T < 0.004). Smoke: absorbing/scattering medium,
     Henyey-Greenstein phase (g=0.55 mixed with a back lobe), shadowed key light + sky ambient +
     fire glow, albedo darkened by soot where hot. Fire: Planck blackbody LUT (shared with the 2D
     renderer, `vfx.render.blackbody`) × emissivity, integrated along the ray. Obstacles: opaque
     iso-surface of the (pre-smoothed) solid mask, bisection-refined hit, gradient normals, Lambert +
     spec + hemisphere ambient, smoke shadows and fire glow. Floor plane y=0 with a grid, receiving
     smoke/obstacle shadows (via the light volume, also outside the cube) and fire glow.
   * **bloom** (2 levels from the HDR buffer) + **composite** at output res: bilinear upsample of
     the march buffer, dark vertical-gradient background with vignette, ACES, sRGB, dither → uint8.

`march_scale < 1` marches fewer rays; the only visual cost is softer edges (obstacle silhouettes,
thin wisps, grid lines), since the volume itself is smooth at that scale anyway.

## Feasibility table (2560x1440)
See below; produced by `python -m vfx.render3d.bench --out DIR` (writes `table.md`,
`results.json`, per-config PNGs and two contact sheets).

Measured 2026-11-09, RTX 5080, **GPU shared**: nvidia-smi showed 96-97 % utilization from other
processes (a game, other python jobs) for the whole run, so every number below is inflated. A lighter-
contention run earlier the same day (before the prep/view split) measured ~1.7x lower totals, e.g. 64³
scale 0.5: 0.77 ms (64 steps/m), 1.09 ms (128 steps/m). The lead should re-run on an idle GPU.

Content: campfire preset, real `vfx.sim3d` 64³ after 150 steps, trilinearly resampled to each
resolution; camera chosen so the effect covers 33 % of the frame. ms = min over 9 batches of 20
back-to-back draws, best of 2 runs. **full** = new volume (pack + texture upload + light/glow volumes +
march + composite); **view** = `reuse_volume=True` (camera-only redraw: the 60 Hz frames between 30 Hz
sim ticks); **60 Hz avg** = (full + view)/2. VRAM = torch peak + textures for the renderer alone.

| volume | march_scale | steps/m | march px | full ms | view ms | 60 Hz avg | VRAM MB |
|---|---|---|---|---|---|---|---|
| 64³ | 1 | 64 | 2560x1440 | 3.77 | 3.39 | 3.58 | 161 |
| 64³ | 1 | 128 | 2560x1440 | 5.16 | 4.80 | 4.98 | 161 |
| 64³ | 1 | 256 | 2560x1440 | 7.70 | 7.28 | 7.49 | 161 |
| 64³ | 0.5 | 64 | 1280x720 | **1.28** | **1.06** | **1.17** | 57 |
| 64³ | 0.5 | 128 | 1280x720 | 1.88 | 1.54 | 1.71 | 57 |
| 64³ | 0.5 | 256 | 1280x720 | 2.84 | 2.21 | 2.52 | 57 |
| 64³ | 0.25 | 64 | 640x360 | **0.82** | **0.48** | **0.65** | 46 |
| 64³ | 0.25 | 128 | 640x360 | **0.94** | **0.69** | **0.81** | 46 |
| 64³ | 0.25 | 256 | 640x360 | **1.22** | **0.77** | **1.00** | 46 |
| 128³ | 1 | 64 | 2560x1440 | 3.69 | 3.43 | 3.56 | 175 |
| 128³ | 1 | 128 | 2560x1440 | 5.22 | 4.76 | 4.99 | 175 |
| 128³ | 1 | 256 | 2560x1440 | 7.44 | 7.12 | 7.28 | 175 |
| 128³ | 0.5 | 64 | 1280x720 | **1.37** | **1.07** | **1.22** | 71 |
| 128³ | 0.5 | 128 | 1280x720 | 1.93 | **1.42** | 1.67 | 71 |
| 128³ | 0.5 | 256 | 1280x720 | 2.57 | 1.71 | 2.14 | 71 |
| 128³ | 0.25 | 64 | 640x360 | **0.86** | **0.56** | **0.71** | 60 |
| 128³ | 0.25 | 128 | 640x360 | **1.00** | **0.66** | **0.83** | 60 |
| 128³ | 0.25 | 256 | 640x360 | **1.33** | **0.89** | **1.11** | 60 |
| 256³ | 1 | 64 | 2560x1440 | 4.63 | 3.35 | 3.99 | 287 |
| 256³ | 1 | 128 | 2560x1440 | 6.14 | 4.66 | 5.40 | 287 |
| 256³ | 1 | 256 | 2560x1440 | 8.84 | 7.53 | 8.19 | 288 |
| 256³ | 0.5 | 64 | 1280x720 | 2.53 | **1.06** | 1.80 | 184 |
| 256³ | 0.5 | 128 | 1280x720 | 2.96 | 1.52 | 2.24 | 184 |
| 256³ | 0.5 | 256 | 1280x720 | 3.52 | 2.35 | 2.94 | 184 |
| 256³ | 0.25 | 64 | 640x360 | 1.91 | **0.55** | **1.23** | 174 |
| 256³ | 0.25 | 128 | 640x360 | 2.13 | **0.76** | **1.44** | 174 |
| 256³ | 0.25 | 256 | 640x360 | 2.24 | **0.90** | 1.57 | 174 |

Bold = within 1.5 ms. Stage profile (lower quartile, no graphs, same contention): pack 0.01 / 0.04 / 0.23 ms
(64/128/256³), texture upload 0.18 ms at 256³, light + glow volumes ~0.45 ms (less contended run), march
scales with rays ³ steps (~1.3 ms at full 1440p ³ 64 steps/m), bloom + composite ~0.25-0.45 ms.

**What fits 1.5 ms at 1440p (on this shared GPU):**
* 64³ / 128³ volumes: march_scale 0.5 at 64 steps/m (full 1.28-1.37 ms), and every 0.25 setting.
* 256³ (route A output): only as a 60 Hz average at march_scale 0.25 (64-128 steps/m, 1.23-1.44 ms);
  the full new-volume frame costs 1.9-2.5 ms because of the fixed per-volume work (pack 0.23 +
  64 MB texture upload 0.18 + light volume). View-only redraws at scale 0.5 cost 1.06 ms.
* march_scale 1.0 never fits (3.4 ms+ just to march 3.7 M rays).

**Visual cost** (1:1 crops: `crops_1to1.png` in the bench output): with trilinear content, scale 0.5
and 0.25 are hard to tell from 1.0; only the floor grid lines and obstacle silhouettes soften.
64 steps/m shows no banding (jittered start). The real cost appears with detailed 256³ volumes: at
1440p a 1/3-screen plume is ~500 px across, so scale 0.5 is ~1 ray per voxel and 0.25 discards about
half the neural detail. Recommendation: scale 0.5, 64-128 steps/m, with the volume prep at 30 Hz.
Cheapest remaining win for route A: have the upscaler write the packed half2 volume (or the texture)
directly, removing pack + upload (~0.4 ms per new 256³ volume).

## Look pass (2026-11-09, after the table above)
No procedural detail is added: every change shades the given volume, so mode comparisons stay honest.
* Lighting: key light low from the left (`light_dir (-0.80, 0.50, -0.20)`), stronger self-shadowing
  (`k_shadow` 40 -> 120, `k_abs` 26 -> 40), darker ambient (0.05, 0.065, 0.10).
* Multiple-scattering approximation (Wrenninge octaves): key = Σ_i a^i T^(b^i) phase(g c^i), 3 octaves,
  `ms_a 0.3, ms_b 0.5, ms_c 0.5`. Deep shadows stay readable without a flat ambient fill.
* Dual-lobe phase: forward lobe g 0.6 weighted by `rim` 0.75 (silver lining when the camera faces the
  light) + back lobe.
* Light volume 64³ -> 80³ with a jittered start and 1.5-voxel steps (no shadow banding).
* Composite: Catmull-Rom upsample of the half-res march buffer (clamped to the 2x2 neighbourhood, no
  ringing) instead of bilinear; contrast power curve around mid-grey (`contrast` 1.15) before ACES.
Cost at 2560x1440, march_scale 0.5, 192 steps/m (CUDA events, min of 7x20 draws; GPU mostly idle, 2 %
util before, another engineer's python job present, i.e. possibly shared):

| volume | before full / view | after full / view | after 60 Hz avg |
|---|---|---|---|
| 64³ (campfire) | 0.62 / 0.50 | 0.79 / 0.60 | 0.69 |
| trilinear 256³ (campfire) | 1.10 / 0.49 | 1.41 / 0.58 | 1.00 |

## Aux buffers (`return_aux=True`, route B) — added 2026-11-09
`draw(..., return_aux=True, velocity=None, prev_camera=None)` returns `(frame, aux)`. The frame is
**bit-identical** to `return_aux=False` (checked). `aux` = dict of fp16 cuda tensors at **out_size**
(not march res), internal buffers overwritten by the next aux draw — `.clone()` to keep them:

| key | shape | meaning |
|---|---|---|
| `alpha` | (h,w) | 1 − transmittance of the **smoke/fire medium only** along the ray, 0 = empty. Obstacles and the floor do not count (a ray that hits an obstacle has alpha = the smoke in front of it). Early ray termination (T < 0.004) gives alpha = 1. |
| `depth` | (h,w) | opacity-weighted (weight T·a per sample) mean **distance from the camera position along the ray** (not view-z), meters (domain = 1 m cube). 0 where alpha < 1e-3. |
| `motion` | (2,h,w) | (x right, y down) in **output pixels per SIM_DT** (1/30 s). Content at pixel p was at **p − motion** one sim step ago (backward-warp: `prev.sample(p - motion)` ≈ current). 0 where alpha < 1e-3. |

How motion is computed (same march, same samples, same jitter as the colour): accumulate Σw, Σw·t,
Σw·v (v = trilinear `velocity` at the sample, w = T·a), take the mean point on the ray
q̄ = o + d·depth and mean velocity v̄, then `motion = pixel_centre − project_prev(q̄ − v̄·SIM_DT)`.
`velocity`: (3,D,H,W) m/s (vel_x, vel_y, vel_z, `FieldState3D.velocity`), any resolution, uploaded to
a 4-ch fp16 texture. `prev_camera`: the camera one sim step ago; None = same camera. With neither,
motion = 0. Emission-only samples (hot, density ≈ 0) have no opacity weight. Background/floor pixels
have motion 0 even if the camera moves (floor is not part of the medium; the sky is screen-fixed).
Aux is accumulated premultiplied at march res and resolved to output res by a bilinear upsample +
un-premultiply (`kernels.aux_resolve`), so silhouettes do not bleed depth/motion into empty pixels.
With `reuse_volume=True` the velocity texture is reused if the same tensor (data_ptr) is passed.

Implementation: `kernels._make_march(AUX)` builds two variants of one kernel source; the aux code is
inside `wp.static(AUX)` branches, so the colour-only variant compiles to the old march (zero cost).

**Cost** (`python -m vfx.render3d.aux_check --time`; 2560x1440, march_scale 0.5, 192 steps/m,
campfire 64³ + its velocity, prev camera; trials interleaved, min of 25x30 draws; GPU idle at
start (2 % util) and end of that run): full frame 0.996 → 1.150 ms (**+0.155 ms**), view-only redraw
0.805 → 0.946 ms (**+0.142 ms**). A second, contended run (another python job, 98 % util) gave
+0.155 to +0.33 ms.

**Validation** (`python -m vfx.render3d.aux_check`, 1280x720, scale 0.5, textured Gaussian blob,
floor off): frame t−1 backward-warped by `motion` vs frame t, MSE(no warp)/MSE(warp):

| case | median motion | ratio whole frame | ratio alpha > 0.5 |
|---|---|---|---|
| blob translated by v=(1.2, 0.6, −0.9) m/s, fixed camera | 23 px | 26.7 | 40.9 |
| static blob, camera orbit 4° + pan (0.05, −0.03) m | 34 px | 73.3 | 73.6 |
| both | 26 px | 49.0 | 32.4 |

Sign check: v = +x, +y gives motion_x > 0, motion_y < 0 (screen y down). The remaining warp error is
mostly disocclusion (pixels the blob left: alpha_t = 0 so motion = 0) and lighting change as the
blob moves through the light volume.

Files: `renderer.py` (pipeline, params, camera), `kernels.py` (Warp kernels), `bench.py`,
`aux_check.py` (aux self-check + cost).
