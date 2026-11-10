# bench3d — 3D feasibility report

Created 2026-11-09 16:27:11 on NVIDIA GeForce RTX 5080, torch 2.11.0+cu128; 100 timed frames (256^3: up to 30) after 10 warm-up frames, scenes pre-simulated 60 steps. Output 2560x1440. GPU times from torch.cuda.Event.

GPU idle at start (no other compute process, utilization <= 10%).

nvidia-smi after run: utilization [100.0]%, other compute processes 0.

## Budget (SPEC)

| stage | budget |
|---|---|
| sim (64^3 step) | 2.0 ms |
| detail (neural / upscale) | 2.5 ms |
| render (1440p) | 1.5 ms |
| frame total | 6.0 ms |
| peak VRAM | 4096 MB |

## Pipelines x stages (mean ms; ✔ within budget, ✘ over)

| scene | pipeline | march | sim | detail | render | frame (p50 / p95) | frames > 50 ms (max) | host wall/frame | VRAM MB | all | x vs brute256 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| campfire | raw64 | 0.5 | 0.82 ✔ | 0.00 ✔ | 0.84 ✔ | 1.66 (1.68 / 2.01) ✔ | 0 (2) | 1.66 | 224 ✔ | PASS | 26.7x |
| campfire | raw64 | 1 | 0.84 ✔ | 0.00 ✔ | 3.18 ✘ | 4.02 (3.97 / 4.55) ✔ | 0 (5) | 4.02 | 344 ✔ | FAIL | 11.3x |
| campfire | tri256 | 0.5 | 0.83 ✔ | 0.18 ✔ | 1.52 ✘ | 2.52 (2.50 / 2.93) ✔ | 0 (3) | 2.52 | 480 ✔ | FAIL | 17.6x |
| campfire | tri256 | 1 | 0.85 ✔ | 0.17 ✔ | 4.04 ✘ | 5.06 (5.04 / 5.56) ✔ | 0 (6) | 5.06 | 606 ✔ | FAIL | 9.0x |
| campfire | routeA | 0.5 | 0.83 ✔ | 4.27 ✘ | 1.50 ✘ | 6.61 (6.65 / 7.31) ✘ | 0 (7) | 6.61 | 824 ✔ | FAIL | 6.7x |
| campfire | routeA | 1 | 0.85 ✔ | 4.33 ✘ | 4.04 ✘ | 9.22 (9.23 / 9.89) ✘ | 0 (10) | 9.22 | 950 ✔ | FAIL | 4.9x |
| campfire | routeB | 0.5 | 0.84 ✔ | 2.08 ✔ | 0.85 ✔ | 3.76 (3.73 / 4.27) ✔ | 0 (4) | 3.76 | 566 ✔ | PASS | 11.8x |
| campfire | routeB | 1 | 0.84 ✔ | 2.10 ✔ | 3.18 ✘ | 6.12 (6.13 / 6.72) ✘ | 0 (7) | 6.12 | 684 ✔ | FAIL | 7.4x |
| campfire | brute128 | 0.5 | 4.13 ✘ | 0.00 ✔ | 0.82 ✔ | 4.95 (4.89 / 5.44) ✔ | 0 (6) | 4.95 | 546 ✔ | FAIL | 9.0x |
| campfire | brute128 | 1 | 4.14 ✘ | 0.00 ✔ | 3.01 ✘ | 7.15 (7.17 / 7.82) ✘ | 0 (8) | 7.15 | 666 ✔ | FAIL | 6.4x |
| campfire | brute256 | 0.5 | 43.03 ✘ | 0.00 ✔ | 1.34 ✔ | 44.37 (44.40 / 44.94) ✘ | 0 (46) | 44.38 | 3260 ✔ | FAIL | 1.0x |
| campfire | brute256 | 1 | 43.18 ✘ | 0.00 ✔ | 2.33 ✘ | 45.51 (45.49 / 46.25) ✘ | 0 (46) | 45.52 | 3386 ✔ | FAIL | 1.0x |
| plume_sphere | raw64 | 0.5 | 0.83 ✔ | 0.00 ✔ | 0.85 ✔ | 1.68 (1.65 / 1.98) ✔ | 0 (2) | 1.68 | 226 ✔ | PASS | 26.6x |
| plume_sphere | raw64 | 1 | 0.84 ✔ | 0.00 ✔ | 2.74 ✘ | 3.57 (3.55 / 4.11) ✔ | 0 (4) | 3.58 | 346 ✔ | FAIL | 12.7x |
| plume_sphere | tri256 | 0.5 | 0.83 ✔ | 0.16 ✔ | 1.59 ✘ | 2.58 (2.56 / 2.99) ✔ | 0 (3) | 2.58 | 598 ✔ | FAIL | 17.3x |
| plume_sphere | tri256 | 1 | 0.85 ✔ | 0.17 ✔ | 3.57 ✘ | 4.59 (4.58 / 5.13) ✔ | 0 (5) | 4.59 | 724 ✔ | FAIL | 9.9x |
| plume_sphere | routeA | 0.5 | 0.83 ✔ | 4.29 ✘ | 1.58 ✘ | 6.70 (6.73 / 7.21) ✘ | 0 (7) | 6.70 | 904 ✔ | FAIL | 6.7x |
| plume_sphere | routeA | 1 | 0.86 ✔ | 4.38 ✘ | 3.57 ✘ | 8.81 (8.81 / 9.59) ✘ | 0 (10) | 8.81 | 1030 ✔ | FAIL | 5.2x |
| plume_sphere | routeB | 0.5 | 0.83 ✔ | 2.07 ✔ | 0.84 ✔ | 3.75 (3.71 / 4.11) ✔ | 0 (4) | 3.75 | 566 ✔ | PASS | 11.9x |
| plume_sphere | routeB | 1 | 0.85 ✔ | 2.09 ✔ | 2.73 ✘ | 5.67 (5.65 / 6.25) ✔ | 0 (6) | 5.67 | 686 ✔ | FAIL | 8.0x |
| plume_sphere | brute128 | 0.5 | 4.14 ✘ | 0.00 ✔ | 0.96 ✔ | 5.09 (5.07 / 5.72) ✔ | 0 (6) | 5.09 | 550 ✔ | FAIL | 8.8x |
| plume_sphere | brute128 | 1 | 4.17 ✘ | 0.00 ✔ | 2.84 ✘ | 7.01 (6.97 / 7.59) ✘ | 0 (8) | 7.02 | 670 ✔ | FAIL | 6.5x |
| plume_sphere | brute256 | 0.5 | 43.02 ✘ | 0.00 ✔ | 1.58 ✘ | 44.60 (44.63 / 45.04) ✘ | 0 (45) | 44.61 | 3358 ✔ | FAIL | 1.0x |
| plume_sphere | brute256 | 1 | 42.96 ✘ | 0.00 ✔ | 2.59 ✘ | 45.54 (45.55 / 45.93) ✘ | 0 (47) | 45.55 | 3484 ✔ | FAIL | 1.0x |

Pass/fail is judged on the mean (the spec budget is per frame, so hitches count). `frames > 50 ms` counts hitches (e.g. graph capture / autotuning when the active-brick count changes).

`x vs brute256` = brute256 frame time (256^3 step + render, same solver config and march_scale) / this pipeline's frame time. `host wall/frame` > frame means the pipeline is launch/CPU-bound.

## Step-only and scene info

| scene | line | config | step mean / p50 / p95 ms | divergence after warm-up |
|---|---|---|---|---|
| campfire | raw64 (ms1) | runtime_config() | 0.84 / 0.80 / 1.10 | 0.000039 |
| campfire | tri256 (ms1) | runtime_config() | 0.85 / 0.80 / 1.06 | 0.000039 |
| campfire | routeA (ms1) | runtime_config() | 0.85 / 0.80 / 1.09 | 0.000039 |
| campfire | routeB (ms1) | runtime_config() | 0.84 / 0.80 / 1.07 | 0.000039 |
| campfire | brute128 (ms1) | runtime_config() | 4.14 / 4.16 / 4.56 | 0.000187 |
| campfire | brute256 (ms1) | runtime_config() | 43.18 / 43.14 / 43.84 | 0.000567 |
| campfire | sim64_default (step_only) | SolverConfig3D() default | 0.88 / 0.88 / 1.14 | 0.000006 |
| campfire | sim256_default (step_only) | SolverConfig3D() default | 61.39 / 61.70 / 63.20 | 0.000019 |
| plume_sphere | raw64 (ms1) | runtime_config() | 0.84 / 0.80 / 1.05 | 0.000134 |
| plume_sphere | tri256 (ms1) | runtime_config() | 0.85 / 0.80 / 1.10 | 0.000134 |
| plume_sphere | routeA (ms1) | runtime_config() | 0.86 / 0.80 / 1.13 | 0.000134 |
| plume_sphere | routeB (ms1) | runtime_config() | 0.85 / 0.80 / 1.07 | 0.000134 |
| plume_sphere | brute128 (ms1) | runtime_config() | 4.17 / 4.18 / 4.57 | 0.002391 |
| plume_sphere | brute256 (ms1) | runtime_config() | 42.96 / 42.96 / 43.36 | 0.003411 |
| plume_sphere | sim64_default (step_only) | SolverConfig3D() default | 1.05 / 1.01 / 1.28 | 0.000013 |
| plume_sphere | sim256_default (step_only) | SolverConfig3D() default | 73.91 / 73.88 / 74.47 | 0.000007 |

Occupancy of the 64^3 state (fraction of active bricks: density > 1e-3 or T > ambient + 1 K):

- **campfire**: 16^3-HR bricks 6.2% after warm-up -> 90.2% at end; 32^3-HR bricks 9.4% -> 99.0%; active voxels 3.3%
- **plume_sphere**: 16^3-HR bricks 6.7% after warm-up -> 72.4% at end; 32^3-HR bricks 10.2% -> 85.9%; active voxels 3.0%
- campfire routeA ms0.5: VolumeUpscaler.timings {"pre": 0.249, "net": 3.841, "post": 0.008, "total": 4.098, "processed_fraction": 1.0}
- campfire routeA ms1: VolumeUpscaler.timings {"pre": 0.25, "net": 4.109, "post": 0.008, "total": 4.367, "processed_fraction": 1.0}
- plume_sphere routeA ms0.5: VolumeUpscaler.timings {"pre": 0.247, "net": 4.368, "post": 0.008, "total": 4.623, "processed_fraction": 1.0}
- plume_sphere routeA ms1: VolumeUpscaler.timings {"pre": 0.248, "net": 3.838, "post": 0.006, "total": 4.093, "processed_fraction": 1.0}

## Caveats

- Untrained network weights in this phase: route A/B costs are for the default architectures, not a quality claim.
- VRAM = max(torch peak allocated, device used-memory growth since the worker's CUDA init); the second includes Warp's pool and is noisy if another process allocates. Per-pipeline subprocesses keep pipelines apart.
- Occupancy grows during the timed frames (continuous emitters), so later frames process more bricks.
