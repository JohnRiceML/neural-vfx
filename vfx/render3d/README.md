# vfx/render3d â€” volume ray-marcher (smoke + fire)

```python
from vfx.render3d import VolumeRenderer
r = VolumeRenderer((1440, 2560), march_scale=0.5)          # out_size = (h, w)
img = r.draw(volume, camera, solid)                         # (h, w, 3) uint8 cuda, row 0 = top
img = r.draw(volume, camera2, solid, reuse_volume=True)    # camera-only redraw of the same volume
r.timings                                                   # {'prep','view','total'} ms of the last
                                                            # finished draw (resolved lazily, no sync)
```
`volume`: `fields3d.Volume`, any resolution, fp32 or fp16. `solid`: optional (D,H,W) mask at any
resolution (the app passes obstacles rasterized at 128Â³). `RenderParams3D` holds the look
(`steps` = ray samples per meter, `light_res`, extinction, emission, colours); `set_params(**kw)`.

## Pipeline (all GPU, no host sync; one tiny pinned H2D copy of the camera)
Volume prep (steps 1-2a..c) is skipped when `reuse_volume=True`; the view (march, bloom, composite)
runs every draw. Each part is one CUDA-graph replay.
1. **pack** (direct launch): density/temperature â†’ half2 volume + 32Â³ occupancy bricks (1-voxel apron).
2. **one CUDA-graph replay** per (volume shape, has-solid):
   * upload the half2 volume into a `wp.Texture3D` â†’ hardware trilinear filtering. Measured ~1.8x
     faster than manual trilinear on array loads for the same march (256Â³, 720p, 256 steps:
     1.51 vs 2.68 ms). torch `grid_sample` was ruled out without a full kernel: it would have to
     materialize every sample position (3.7 M rays Ã— hundreds of steps).
   * **light volume** (64Â³): transmittance toward the key light (march to the cube exit, smoke +
     obstacles), sky occlusion (short march up), local blackbody emission.
   * **fire glow** (32Â³ + 8Â³ blurred emission pyramid): lights smoke, obstacles and the floor.
   * **march** at `march_scale` Ã— output res: ray/box intersection with the unit cube, jittered
     (interleaved-gradient-noise) start, empty-brick skipping on the step lattice, front-to-back
     compositing, early termination (T < 0.004). Smoke: absorbing/scattering medium,
     Henyey-Greenstein phase (g=0.55 mixed with a back lobe), shadowed key light + sky ambient +
     fire glow, albedo darkened by soot where hot. Fire: Planck blackbody LUT (shared with the 2D
     renderer, `vfx.render.blackbody`) Ã— emissivity, integrated along the ray. Obstacles: opaque
     iso-surface of the (pre-smoothed) solid mask, bisection-refined hit, gradient normals, Lambert +
     spec + hemisphere ambient, smoke shadows and fire glow. Floor plane y=0 with a grid, receiving
     smoke/obstacle shadows (via the light volume, also outside the cube) and fire glow.
   * **bloom** (2 levels from the HDR buffer) + **composite** at output res: bilinear upsample of
     the march buffer, dark vertical-gradient background with vignette, ACES, sRGB, dither â†’ uint8.

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
scales with rays × steps (~1.3 ms at full 1440p × 64 steps/m), bloom + composite ~0.25-0.45 ms.

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

Files: `renderer.py` (pipeline, params, camera), `kernels.py` (Warp kernels), `bench.py`.
