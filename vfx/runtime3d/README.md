# vfx/runtime3d — 3D detail runtime + feasibility bench (owner: 3D ML/runtime engineer)

Phase question: can neural detail for one 3D effect fit **≤ 2.5 ms/frame and ≤ 4 GB** on the RTX 5080?
Everything here is **untrained (timing only)**. Untrained outputs are harmless: `VolumeUpscaler` returns the
trilinear upscale, `ScreenDetail` returns the input frame (the last conv is zero-initialised).

## API (CLAUDE.md "3D phase" contract)
- `classical_upscale_3d(state, factor=4, mode="trilinear"|"nearest", dtype=torch.half) -> Volume` — baseline.
  Trilinear runs as a 1D pass over D plus a 2D bilinear pass over HW. The result is identical to
  `F.interpolate(trilinear)` up to fp16 rounding and 3.4x faster (0.16 vs 0.54 ms at 256³).
- `warp_prev_3d(prev_hr, lr_velocity, dx_hr, dt)` — dense semi-Lagrangian warp, the training reference
  (3D twin of `vfx.runtime.warp_prev`). `sample_bricks(...)` does the same warp only at active-brick voxels.
- `VolumeUpscaler(checkpoint_path=None, device="cuda", **arch)` — route A. `infer(state, active=None) -> Volume`
  (fp16 256³), `reset()`, `warmup(dt)`, `timings` (pre / net / post / total / processed_fraction).
  `active=` forces a brick-id set (bench/sandbox); otherwise the lagged GPU activity mask is used, as in the 2D Upscaler.
- `ScreenDetail(checkpoint_path=None, device="cuda", process_scale=0.5, **arch)` — route B.
  `infer(frame (h,w,3) uint8, aux=None) -> (h,w,3) uint8`. Optional `aux` keys: `bbox` (x0,y0,x1,y1 px of the
  effect), `transmittance`, `depth` (h,w), `velocity` (h,w,2) px/frame. `timings`.
- `bench_selfcheck.py` — `.venv/Scripts/python -m vfx.runtime3d.bench_selfcheck [--quick] [--only ...]`.
  Runs correctness checks, then regenerates every table below. Writes `bench_results.json` (small) and prints markdown.
  About 5 min. `--only check,floor,recommended` takes about 30 s.

## How route A works (`volume_upscaler.py`, `fused.py`, `vfx/model3d/volumenet.py`)
64³ LR → 256³ HR. Bricks are 8³ LR cells (32³ HR voxels), giving 512 bricks, with a 2-cell halo, so the
trunk sees 12³ per brick.
1. **pre graph** (fixed shape, 0.27 ms): fp16 cast, clamp, trilinear canvas for the whole 256³ (this canvas is the output
   outside active bricks), activity mask (density/T threshold, dilated 2 LR cells, pooled to bricks) copied async to pinned memory.
2. **net graph**, one per active-count bucket (multiples of 8 bricks):
   - gather the active LR bricks with their halo;
   - **Warp kernel 1** does the sparse warp of the previous HR output (back-trace along trilinear LR velocity of frame t-1),
     normalises it and writes it directly in `pixel_unshuffle3d` channels-last order;
   - the cuDNN trunk runs, conv3d fp16 channels_last_3d;
   - **Warp kernel 2** does the 3D pixel shuffle, trilinear base, 1x1x1 HR head, residual, clamp and denorm, and scatters into the canvas.
3. The two canvases alternate between frames (output t is prev t+1), so there is no 256³ copy.
   Above 50% occupancy the net runs once on the whole volume instead (`dense_above`).

The Warp kernels exist because profiling the pure-PyTorch path showed **~50% of GPU time in strided copies**
(pixel-shuffle permutes, layout conversions, concat) at HR resolution and only ~25% in convs. Fusing them is
what an engine compute shader would do. It is **1.5–2x faster** (variants table) and is checked against the PyTorch
`VolumeNet.forward` by `selfcheck()`: the max density difference is 1e-3, which is fp16 rounding, against a net change of 0.04–0.2.

## How route B works (`screen_detail.py`, `vfx/model3d/screennet.py`)
The processed region is the effect's screen bbox, rounded up to 64 px (or the whole frame). In it:
- inputs are uint8 → fp16, then avg-pooled to `process_scale`;
- the previous output is warped by screen velocity at that resolution;
- the ScreenNet trunk runs at 1/4 of that resolution (PixelUnshuffle), then PixelShuffle, a 1x1 HR head, and Δrgb;
- Δrgb is bilinear-upsampled, added and clamped back to uint8, and pasted into the frame.
One CUDA graph per region size; the two output buffers alternate.

## Measurement conditions — read before using the numbers
- **The GPU was shared.** At the start of the final run nvidia-smi showed 97% utilisation and 6.9 GB used by other
  engineers' `vfx.app3d` / `bench3d` jobs plus short scratch scripts. At the end it showed 1%. Between runs the same config moved
  ±20–30%: for example 32x4 at 25% measured 2.14, 2.46 and 2.51 ms.
- Tables show the **10th-percentile** per-frame ms. It is more robust than the median to bursty foreign work; `*_med` medians are in the JSON.
  Re-run on an idle GPU for decisions near the line.
- One frame = CUDA events around the whole `infer()` (eager mode therefore includes launch gaps). FP16 throughout.
  "MiB" = `torch.cuda.max_memory_allocated` peak above the pre-run baseline. It excludes the CUDA context (~0.3–0.5 GB, not measured).
- Active brick sets are synthetic: the n bricks nearest a tall plume axis. Occupancy = the fraction of the 512 bricks that the
  net processes (after dilation). Route B input is a random 2560x1440 frame. "bbox" means a 1536x832 region, 35% of the screen.
- Reference: the working 2D DetailNet 64x8 has **688,114 params and 44,820 MACs per HR output pixel**.

## Floor costs (unavoidable or baseline work, CUDA graph)
| op | ms |
|---|---|
| write 256³×2 fp16 (fill) / copy | 0.055 / 0.131 |
| `classical_upscale_3d` trilinear ×4 fp16 (fp32) | **0.155** (0.254) |
| `classical_upscale_3d` nearest ×4 fp16 | 0.145 |
| stock `F.interpolate` trilinear fp16 (not used) | 0.535 |
| trilinear ×2 → 128³ | 0.082 |
| **dense** `warp_prev_3d` 256³ (128³) | **2.64** (0.22) |
| sparse warp + base, 5 / 10 / 25 / 50 / 100% bricks (PyTorch grid_sample) | 0.09 / 0.16 / 0.45 / 0.99 / 2.33 |
| VolumeUpscaler pre-stage alone (cast, canvas, mask; 0 bricks) | **0.27** |
| route B: 1440p RGB8 copy / u8→fp16→u8 round trip | 0.016 / 0.12 |

A dense 256³ warp alone costs more than the whole detail budget. The warp must be sparse; in the fused path it is
part of Warp kernel 1. Route A pays ~0.27 ms before any network runs. Most of that is the full trilinear canvas, which only
exists because `Volume` is a dense grid (see "what would change the answer").

## Route A — VolumeUpscaler 64³→256³, ms/frame, CUDA graph (eager in brackets for 10%)
Fused path, brick 8, halo 2. "dense" = whole volume in one pass. ✔ marks ≤ 2.5 ms.

| width×depth | params | MACs/HR voxel | dense | 100% | 50% | 25% | 10% (eager) | 5% | MiB @25% |
|---|---|---|---|---|---|---|---|---|---|
| 16×2 | 37k | 648 | 3.23 | 9.91 | 1.95✔ | 1.10✔ | 0.67✔ (0.77) | 0.53✔ | 406 |
| 16×4 | 65k | 1,080 | 3.52 | 7.99 | 2.61 | 1.33✔ | 0.79✔ (0.92) | 0.61✔ | 401 |
| 16×8 | 120k | 1,944 | 5.34 | 11.6 | 3.87 | 1.77✔ | 1.01✔ (1.20) | 0.77✔ | 401 |
| 32×2 | 129k | 2,081 | 3.73 | 11.1 | 3.56 | 1.60✔ | 0.85✔ (0.95) | 0.66✔ | 401 |
| **32×4 (default)** | **239k** | 3,809 | 6.73 | 18.6 | 7.86 | 2.51 (2.14 prev. run) | **1.13✔** (1.26) | 0.84✔ | 401 |
| 32×8 | 461k | 7,265 | 8.05 | 30.5 | 11.5 | 3.66 | 1.68✔ (1.89) | 1.19✔ | 402 |
| 48×2 | 276k | 4,378 | 7.45 | 23.8 | 8.65 | 2.97 | 1.37✔ (1.45) | 0.94✔ | 567 |
| 48×4 | 525k | 8,266 | 11.7 | 40.0 | 17.2 | 6.68 | 2.38✔ (2.49) | 1.41✔ | 402 |
| 48×8 | 1.02M | 16,042 | 21.0 | 76.1 | 32.0 | 13.7 | 5.68 (4.84) | 2.60 | 403 |
| 64×2 | 478k | 7,538 | 10.5 | 29.8 | 12.5 | 4.45 | 1.62✔ (1.72) | 1.12✔ | 515 |
| 64×4 | 921k | 14,450 | 16.2 | 54.3 | 23.6 | 10.5 | 2.80 (3.28) | 1.69✔ | 432 |
| 64×8 | 1.81M | 28,274 | 26.2 | 98.6 | 43.8 | 19.9 | 6.97 (7.72) | 3.16 | 433 |

Eager is 5–15% slower at realistic occupancy. Peak VRAM is ≤ 1.5 GB in every configuration (the worst is 48×2 at 100%).
Defaults: `VolumeUpscaler()` = 32×4 measures 0.84 / 1.12 / 1.77 / 2.47 ms at 5 / 10 / 20 / 25% occupancy, with 401 MiB allocator peak.

### Route A variants (CUDA graph)
| variant | params | 25% | 10% |
|---|---|---|---|
| 32×2, unfused (pure PyTorch HR stages) vs fused | 129k | 3.44 vs 1.59 | 1.34 vs 0.85 |
| 32×4, unfused vs fused | 239k | 4.54 vs 2.46 | 1.62 vs 1.12 |
| 32×4 sep (2+1)D | 117k | 1.95 | 1.04 |
| 32×4 mp (SPEC multi-pass idea: 2D convs over slices, axis cycling) | 92k | **1.60** | **0.87** |
| 32×4 dw (depthwise+pointwise) | 33k | 11.8 | 2.93 |
| 64×4 sep / 64×4 mp | 430k / 331k | 6.49 / 3.50 | 1.89 / **1.50** |
| 64×8 dw | 131k | 76.8 | 31.8 |
| 32×2 without channels_last_3d | 129k | 2.28 | 1.07 |
| 32×2 / 32×4 / 64×4 at ×2 upscale (128³ output) | 118k / 228k / 899k | 1.10 / 1.69 / 7.98 | 0.58 / 0.85 / 2.54 |
| 32×2 HR head 3×3×3 (unfused) | 131k | 4.22 | 1.58 |
| 32×2 HR feats 8 / hidden 16 | 137k | 2.44 | 1.09 |
| 32×2 brick 8 halo 1 / halo 4 | 129k | **1.24** / 2.91 | 0.73 / 1.24 |
| 32×2 brick 16 halo 4 / brick 4 halo 2 | 129k | 1.65 / 2.87 | 1.07 / 1.15 |

Notes:
- Depthwise 3D convs are a trap on cuDNN.
- The multi-pass 2D-slice block is the best capacity per ms.
- Halo overhead matters: an S=12 brick does 3.4× the core work. Halo 1 saves ~20%.
- ×2 output (128³) costs ~30% less than ×4, not 4×. The trunk dominates and is shared.
- Small batched 12³ bricks run cuDNN less efficiently than one 64³ volume. At 100% occupancy sparse is 2–4× slower than dense, so the 0.5 dense switch-over matters.

## Route B — ScreenDetail 2560×1440, ms/frame, CUDA graph
Columns give the region processed and the process scale (full = 1.0, half = 0.5, quarter = 0.25).
"bbox" = the effect's screen bbox, 1536×832 = 35% of the screen. MACs are per processed pixel. ✔ marks ≤ 2.5 ms.

| width×depth | params | MACs/px | full frame | ½ | ¼ | bbox | bbox ½ | bbox ¼ | MiB full / bbox |
|---|---|---|---|---|---|---|---|---|---|
| 16×2 | 42k | 2,696 | 3.65 | 1.52✔ | 0.90✔ | 1.01✔ | 0.58✔ | 0.37✔ | 616 / 224 |
| 16×4 | 51k | 3,272 | 3.87 | 1.58✔ | 0.93✔ | 1.07✔ | 0.60✔ | 0.40✔ | 465 / 172 |
| 16×8 | 70k | 4,424 | 5.46 | 1.68✔ | 0.99✔ | 1.21✔ | 0.66✔ | 0.44✔ | 465 / 172 |
| 32×2 | 102k | 6,440 | 4.06 | 1.63✔ | 0.92✔ | 1.17✔ | 0.60✔ | 0.38✔ | 1271 / 446 |
| 32×4 | 139k | 8,744 | 4.22 | 1.72✔ | 0.96✔ | 1.30✔ | 0.64✔ | 0.39✔ | 465 / 172 |
| 32×8 | 213k | 13,352 | 5.96 | 1.92✔ | 1.06✔ | 1.55✔ | 0.75✔ | 0.46✔ | 465 / 172 |
| 48×2 | 180k | 11,336 | 4.37 | 1.75✔ | 0.95✔ | 1.32✔ | 0.66✔ | 0.39✔ | 1278 / 448 |
| 48×4 | 263k | 16,520 | 5.13 | 1.96✔ | 1.02✔ | 1.59✔ | 0.77✔ | 0.41✔ | 473 / 175 |
| 48×8 | 430k | 26,888 | 6.90 | 2.39✔ | 1.16✔ | 2.14✔ | 0.95✔ | 0.50✔ | 473 / 175 |
| 64×2 | 277k | 17,384 | 4.74 | 1.84✔ | 0.97✔ | 1.42✔ | 0.66✔ | 0.39✔ | 1285 / 451 |
| **64×4 (default, ½)** | **425k** | 26,600 | 6.10 | 2.11✔ | 1.04✔ | 1.76✔ | **0.77✔** | 0.42✔ | 480 / 177 |
| 64×8 (≈ the 2D net) | 720k | 45,032 | 8.45 | 2.70 | 1.20✔ | 2.53 (2.37 prev. run) | 0.96✔ | 0.49✔ | 480 / 178 |

### Route B variants (columns: full / ½ / ¼ / bbox / bbox ½ / bbox ¼)
| variant | params | full | ½ | ¼ | bbox | bbox ½ | bbox ¼ |
|---|---|---|---|---|---|---|---|
| 64×4 eager (no graph) | 425k | 6.26 | 2.33 | 1.29 | 2.01 | 1.02 | 0.91 |
| 32×4 trunk stride 2 | 90k | 8.10 | 2.13 | 1.06 | 1.80 | 0.77 | 0.43 |
| 32×4 trunk stride 8 | 332k | 5.67 | 1.69 | 0.94 | 1.16 | 0.64 | 0.41 |
| 64×8 trunk stride 8 | 1.11M | 7.50 | 1.95 | 1.07 | 1.48 | 0.74 | 0.47 |
| 32×4 no aux, no prev | 106k | 2.70 | 1.12 | 0.74 | 0.79 | 0.44 | 0.32 |
| 32×4 HR head 3×3 | 140k | 7.79 | 1.82 | 0.99 | 1.41 | 0.68 | 0.41 |

Route B is dominated by full-resolution glue, not by the net:
- At bbox ½, width/depth move the cost only 0.58 → 0.96 ms.
- Dropping aux+prev saves ~0.5 ms at bbox. That is fp32 aux conversion, warp and pooling, which are fusable like route A's HR stages.
- Full-frame full-res fails the budget for every network. Half-res or bbox is required.

## Recommendation
**Largest architecture that fits 2.5 ms at realistic occupancy**
- **Route A** (localized plume, 10–20% of bricks):
  - 48×4 (525k params) at 10%, or 32×8 (461k) at 1.7 ms;
  - 32×4 (239k) at 20% (1.8 ms) and at the edge at 25%;
  - at 5%, 64×4 (921k) fits in 1.7 ms;
  - at 50%, only 16×2 fits; at 100%, nothing (dense 16×2 = 3.2 ms).

  Capacity vs the 2D net (688k params, 44.8k MACs per output pixel): the 32×4 default has 35% of the params but only
  **8.5% of the compute per output voxel**. The largest at 10% (48×4) has 76% of the params and 18% of the compute.
  Route A can fit at the 2D net's parameter count, but at roughly 1/10–1/5 of its compute per output element.
- **Route B**:
  - the **2D-equivalent 64×8 (720k params, 45k MACs/px = 1.0× the 2D net)** fits on a 1/3-screen bbox at full
    resolution (2.4–2.5 ms, borderline) and comfortably at half resolution (0.96 ms);
  - even 64×8 with a stride-8 trunk (1.1M params) runs in 0.74 ms at bbox ½;
  - whole-screen: everything up to 48×8 fits at ½, and everything at ¼.
- **Defaults (the configs to train next)**:
  - `VolumeUpscaler()` = 32×4 full, brick 8 / halo 2, fused: 0.84–1.77 ms at 5–20%, ~0.4 GB.
    The `mp` block (1.60 ms at 25%) is the first alternative to try if 32×4 is too slow.
  - `ScreenDetail()` = 64×4, stride 4, process_scale 0.5: 0.77 ms on a 1/3-screen bbox, 2.39 ms whole-frame, ≤ 0.42 GB.

  VRAM is a non-issue for both routes: ≤ 1.5 GB worst case, ~0.4 GB at the defaults, against the 4 GB budget.

**Which route.** For a game-engine budget, **route B is the better bet**:
- it is the only route that affords the capacity proven in 2D;
- its cost scales with screen coverage and is independent of how much of the volume is active;
- its floor is ~0.1 ms instead of 0.27 ms plus a 256³ ray-march.

Route A's cost grows linearly with occupancy. A large fireball (50% of bricks) exceeds the budget with anything bigger than 16×2.
The renderer must also march a 256³ volume instead of 64³, which is not measured here and is the renderer owner's 1.5 ms.

Route A's real advantages are not small:
- detail is view-independent and 3D-consistent (shadows, multiple cameras, lighting all see the same structure);
- it has no screen-space problems: disocclusion, and the poorly defined "velocity" of a semi-transparent volume in screen space.

So my judgment is a **hybrid with B as the capacity carrier**: train B (64×4 at ½ res, ~0.8 ms on the effect bbox) first.
Keep a small A (32×2 or 32×4-mp at ≤ 25%, 0.85–1.6 ms) as the view-consistent mid-frequency layer if B alone turns out
temporally or view-inconsistent. Both together fit 2.5 ms at realistic occupancy. A alone is viable only for localized
effects with roughly a tenth of the 2D net's compute per output element; whether that is enough detail is a training
question this phase cannot answer.

**What would change the answer**:
- a sparse brick-pool `Volume` (renderer marches LR + HR bricks) removes A's 0.27 ms canvas floor and the 256³ render cost;
- TensorRT/FP8 for the cuDNN trunk;
- fusing B's aux/prev glue (~0.5 ms);
- an idle-GPU re-run (`bench3d`).

## Not measured
- Quality: everything is untrained.
- Render cost of 256³ vs 64³ volumes (renderer owner).
- TensorRT/FP8.
- Real occupancy distributions from Solver3D runs, and mask lag. The bench forces brick sets.
- Training memory/time for 3D.
- CUDA-context VRAM.
- Uncontended timings: the GPU was shared for most of the runs.
