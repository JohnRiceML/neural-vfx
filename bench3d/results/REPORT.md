# bench3d — 3D feasibility report

Created 2026-11-09 15:49:59 on NVIDIA GeForce RTX 5080, torch 2.11.0+cu128; 100 timed frames (256^3: up to 30) after 10 warm-up frames, scenes pre-simulated 60 steps. Output 2560x1440. GPU times from torch.cuda.Event.

> **WARNING: GPU WAS SHARED during this run — numbers are not representative.**
>
> - 2 other compute process(es): 14816 ...s\Python\Python312\python.exe, 25880 ...s\Python\Python312\python.exe
> The lead re-runs `python -m bench3d.run` on an idle GPU before any decision.

nvidia-smi after run: utilization [97.0]%, other compute processes 0.

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
| campfire | raw64 | 0.5 | 1.06 ✔ | 0.00 ✔ | 0.74 ✔ | 1.80 (1.71 / 2.40) ✔ | 0 (4) | 1.80 | 194 ✔ | PASS | 28.1x |
| campfire | raw64 | 1 | 1.06 ✔ | 0.00 ✔ | 2.95 ✘ | 4.02 (3.86 / 5.20) ✔ | 0 (7) | 4.02 | 326 ✔ | FAIL | 11.9x |
| campfire | tri256 | 0.5 | 1.12 ✔ | 0.41 ✔ | 1.37 ✔ | 2.90 (2.84 / 3.42) ✔ | 0 (5) | 2.90 | 472 ✔ | PASS | 17.5x |
| campfire | tri256 | 1 | 1.22 ✔ | 0.59 ✔ | 3.97 ✘ | 5.77 (5.65 / 7.23) ✔ | 0 (9) | 5.79 | 590 ✔ | FAIL | 8.2x |
| campfire | routeA | 0.5 | 0.98 ✔ | 3.77 ✘ | 1.39 ✔ | 6.13 (5.81 / 8.78) ✘ | 0 (9) | 6.13 | 892 ✔ | FAIL | 8.2x |
| campfire | routeA | 1 | 1.05 ✔ | 3.97 ✘ | 3.86 ✘ | 8.88 (8.15 / 13.01) ✘ | 0 (19) | 8.89 | 900 ✔ | FAIL | 5.4x |
| campfire | routeB | 0.5 | 0.99 ✔ | 4.19 ✘ | 0.67 ✔ | 5.86 (5.68 / 7.01) ✔ | 0 (7) | 5.86 | 1292 ✔ | FAIL | 8.6x |
| campfire | routeB | 1 | 1.01 ✔ | 4.51 ✘ | 2.84 ✘ | 8.36 (7.68 / 14.17) ✘ | 0 (15) | 8.37 | 892 ✔ | FAIL | 5.7x |
| campfire | brute128 | 0.5 | 4.30 ✘ | 0.00 ✔ | 0.68 ✔ | 4.98 (4.81 / 6.16) ✔ | 0 (7) | 4.98 | 530 ✔ | FAIL | 10.2x |
| campfire | brute128 | 1 | 4.59 ✘ | 0.00 ✔ | 2.90 ✘ | 7.50 (6.85 / 11.13) ✘ | 0 (18) | 7.50 | 662 ✔ | FAIL | 6.3x |
| campfire | brute256 | 0.5 | 49.15 ✘ | 0.00 ✔ | 1.44 ✔ | 50.59 (43.47 / 83.77) ✘ | 9 (85) | 50.60 | 3244 ✔ | FAIL | 1.0x |
| campfire | brute256 | 1 | 45.36 ✘ | 0.00 ✔ | 2.25 ✘ | 47.61 (44.03 / 67.01) ✘ | 6 (70) | 47.62 | 3382 ✔ | FAIL | 1.0x |
| plume_sphere | raw64 | 0.5 | 0.98 ✔ | 0.00 ✔ | 0.70 ✔ | 1.68 (1.62 / 2.09) ✔ | 0 (2) | 1.68 | 196 ✔ | PASS | 53.0x |
| plume_sphere | raw64 | 1 | 0.96 ✔ | 0.00 ✔ | 2.29 ✘ | 3.24 (3.21 / 3.78) ✔ | 0 (4) | 3.24 | 328 ✔ | FAIL | 20.3x |
| plume_sphere | tri256 | 0.5 | 0.95 ✔ | 0.50 ✔ | 1.55 ✘ | 3.00 (2.89 / 3.85) ✔ | 0 (4) | 3.00 | 582 ✔ | FAIL | 29.7x |
| plume_sphere | tri256 | 1 | 0.93 ✔ | 0.58 ✔ | 3.27 ✘ | 4.77 (4.69 / 6.12) ✔ | 0 (7) | 4.78 | 720 ✔ | FAIL | 13.8x |
| plume_sphere | routeA | 0.5 | 0.99 ✔ | 48.33 ✘ | 1.50 ✘ | 50.83 (6.05 / 11.27) ✘ | 3 (1506) | 50.83 | 1016 ✔ | FAIL | 1.8x |
| plume_sphere | routeA | 1 | 0.95 ✔ | 3.66 ✘ | 3.07 ✘ | 7.68 (7.45 / 8.91) ✘ | 0 (14) | 7.68 | 1290 ✔ | FAIL | 8.6x |
| plume_sphere | routeB | 0.5 | 1.02 ✔ | 4.56 ✘ | 0.67 ✔ | 6.26 (5.88 / 8.33) ✘ | 0 (9) | 6.26 | 1307 ✔ | FAIL | 14.2x |
| plume_sphere | routeB | 1 | 1.22 ✔ | 5.28 ✘ | 3.14 ✘ | 9.65 (8.61 / 18.10) ✘ | 0 (19) | 9.65 | 884 ✔ | FAIL | 6.8x |
| plume_sphere | brute128 | 0.5 | 11.98 ✘ | 0.00 ✔ | 2.36 ✘ | 14.34 (13.28 / 22.76) ✘ | 0 (29) | 14.35 | 536 ✔ | FAIL | 6.2x |
| plume_sphere | brute128 | 1 | 10.47 ✘ | 0.00 ✔ | 6.02 ✘ | 16.49 (16.68 / 22.33) ✘ | 0 (28) | 16.51 | 668 ✔ | FAIL | 4.0x |
| plume_sphere | brute256 | 0.5 | 86.49 ✘ | 0.00 ✔ | 2.51 ✘ | 89.00 (88.58 / 112.34) ✘ | 30 (115) | 89.13 | 3344 ✔ | FAIL | 1.0x |
| plume_sphere | brute256 | 1 | 62.79 ✘ | 0.00 ✔ | 2.89 ✘ | 65.68 (66.61 / 98.43) ✘ | 30 (98) | 65.74 | 3482 ✔ | FAIL | 1.0x |

Pass/fail is judged on the mean (the spec budget is per frame, so hitches count). `frames > 50 ms` counts hitches (e.g. graph capture / autotuning when the active-brick count changes).

`x vs brute256` = brute256 frame time (256^3 step + render, same solver config and march_scale) / this pipeline's frame time. `host wall/frame` > frame means the pipeline is launch/CPU-bound.

## Step-only and scene info

| scene | line | config | step mean / p50 / p95 ms | divergence after warm-up |
|---|---|---|---|---|
| campfire | raw64 (ms1) | runtime_config() | 1.06 / 0.92 / 1.51 | 0.000034 |
| campfire | tri256 (ms1) | runtime_config() | 1.22 / 1.13 / 2.09 | 0.000034 |
| campfire | routeA (ms1) | runtime_config() | 1.05 / 0.92 / 1.56 | 0.000034 |
| campfire | routeB (ms1) | runtime_config() | 1.01 / 0.92 / 1.64 | 0.000034 |
| campfire | brute128 (ms1) | runtime_config() | 4.59 / 4.29 / 6.74 | 0.000160 |
| campfire | brute256 (ms1) | runtime_config() | 45.36 / 41.97 / 64.71 | 0.000672 |
| campfire | sim64_default (step_only) | SolverConfig3D() default | 1.05 / 1.02 / 1.31 | 0.000007 |
| campfire | sim256_default (step_only) | SolverConfig3D() default | 65.82 / 61.22 / 94.00 | 0.000012 |
| plume_sphere | raw64 (ms1) | runtime_config() | 0.96 / 0.92 / 1.19 | 0.000724 |
| plume_sphere | tri256 (ms1) | runtime_config() | 0.93 / 0.90 / 1.14 | 0.000724 |
| plume_sphere | routeA (ms1) | runtime_config() | 0.95 / 0.92 / 1.17 | 0.000724 |
| plume_sphere | routeB (ms1) | runtime_config() | 1.22 / 0.98 / 2.46 | 0.000668 |
| plume_sphere | brute128 (ms1) | runtime_config() | 10.47 / 10.17 / 15.01 | 0.002074 |
| plume_sphere | brute256 (ms1) | runtime_config() | 62.79 / 63.49 / 95.91 | 0.009003 |
| plume_sphere | sim64_default (step_only) | SolverConfig3D() default | 1.60 / 1.39 / 3.13 | 0.000008 |
| plume_sphere | sim256_default (step_only) | SolverConfig3D() default | 116.42 / 88.17 / 187.61 | 0.000007 |

Occupancy of the 64^3 state (fraction of active bricks: density > 1e-3 or T > ambient + 1 K):

- **campfire**: 16^3-HR bricks 6.2% after warm-up -> 91.9% at end; 32^3-HR bricks 9.4% -> 99.4%; active voxels 3.3%
- **plume_sphere**: 16^3-HR bricks 6.9% after warm-up -> 68.7% at end; 32^3-HR bricks 12.9% -> 83.8%; active voxels 3.0%
- campfire routeA ms0.5: VolumeUpscaler.timings {"pre": 0.244, "net": 3.139, "post": 0.01, "total": 3.392, "processed_fraction": 1.0}
- campfire routeA ms1: VolumeUpscaler.timings {"pre": 2.877, "net": 5.357, "post": 0.008, "total": 8.241, "processed_fraction": 1.0}
- plume_sphere routeA ms0.5: VolumeUpscaler.timings {"pre": 0.244, "net": 3.118, "post": 0.008, "total": 3.369, "processed_fraction": 1.0}
- plume_sphere routeA ms1: VolumeUpscaler.timings {"pre": 0.244, "net": 3.426, "post": 0.008, "total": 3.678, "processed_fraction": 1.0}

## Caveats

- Untrained network weights in this phase: route A/B costs are for the default architectures, not a quality claim.
- VRAM = max(torch peak allocated, device used-memory growth since the worker's CUDA init); the second includes Warp's pool and is noisy if another process allocates. Per-pipeline subprocesses keep pipelines apart.
- Occupancy grows during the timed frames (continuous emitters), so later frames process more bricks.
