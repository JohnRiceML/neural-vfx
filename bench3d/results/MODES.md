# Approach comparison - `python -m bench3d.compare_modes`

Generated 2026-11-09 20:53:03. Presets [2], 6 recorded frames after 20 warm-up steps, 1280x720, march_scale 0.5, orbiting camera (start -30.0 deg, 1.0 deg/step, distance 1.9).

> **GPU SHARED during this run** (9 other compute process(es): 5552 ...s\Python\Python312\python.exe, 9044 ...s\Python\Python312\python.exe, 9948 ...s\Python\Python312\python.exe, 22744 ...s\Python\Python312\python.exe, 25792 ...s\Python\Python312\python.exe, 26436 ...s\Python\Python312\python.exe, 26468 ...s\Python\Python312\python.exe, 27620 ...s\Python\Python312\python.exe, 28616 ...s\Python\Python312\python.exe; GPU utilization 99% before the benchmark). All ms numbers are NOT representative; the lead re-runs on an idle GPU.

Lower is better everywhere except the signed columns (2-4px / 8-16px band log10 power ratio to the reference, tLP log ratio: 0 = like the reference, < 0 missing detail / too static, > 0 too much / boiling). `pop` 1 = as bursty as the reference. Ratios (flicker r, jerk r, swim r) are to the reference; they are legacy upper-bound gates that REWARD blur (r < 1 is not better). Quality = mean over presets; cost = mean of per-preset p50 GPU ms, worst = max single frame. Budget sim 2.0 / detail 2.5 / render 1.5 / frame 6.0 ms.

Pixel-aligned (legacy) | distribution-level (fair to unaligned detail) | temporal vs reference | cost

| approach | L1 | MS-DSSIM | spec err | band dist | 2-4px | 8-16px | grad W1 | contr W1 | patch SWD | tLP | tLP log | pop | tOF px | flicker r | jerk r | swim r | sim ms | detail ms | render ms | frame ms | worst ms | VRAM MB | budget |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **steered 256^3 (reference)** | 0.0000 | 0.000 | 0.000 | 0.000 | -0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 | - | 70.38 | 0.00 | 54.94 | 1441.23 | 4298.2 | 555 | offline |
| **raw 64^3** | 0.0244 | 0.073 | 0.145 | 0.256 | -0.06 | -0.44 | 0.178 | 0.153 | 0.170 | 0.270 | -0.31 | 1.00 | 0.17 | 1.00 | 0.65 | - | 1.03 | 0.00 | 0.42 | 1.47 | 12.0 | 51 | yes |
| trilinear 256^3 | 0.0251 | 0.076 | 0.153 | 0.270 | -0.06 | -0.48 | 0.182 | 0.157 | 0.176 | 0.277 | -0.32 | 1.00 | 0.17 | 0.98 | 0.64 | 0.25 | 0.80 | 0.16 | 3.37 | 4.33 | 36.5 | 224 | over: render |
| unsharp mask (screen) | 0.0237 | 0.070 | 0.156 | 0.201 | 0.14 | -0.28 | 0.178 | 0.156 | 0.123 | 0.223 | -0.25 | 1.00 | 0.18 | 1.18 | 0.72 | 0.36 | 0.80 | 0.50 | 2.92 | 7.33 | 16.5 | 69 | over: render,frame |
| route B latest.pt | 0.0220 | 0.063 | 0.111 | 0.177 | -0.02 | -0.30 | 0.142 | 0.124 | 0.144 | 0.235 | -0.27 | 0.98 | 0.22 | 1.09 | 0.69 | 0.72 | 0.80 | 0.62 | 0.83 | 2.24 | 3.6 | 124 | yes |
| route B v1_reg_backup.pt | 0.0220 | 0.063 | 0.111 | 0.177 | -0.02 | -0.30 | 0.142 | 0.124 | 0.144 | 0.235 | -0.27 | 0.98 | 0.22 | 1.09 | 0.69 | 0.72 | 0.82 | 0.62 | 0.76 | 2.75 | 3.5 | 124 | yes |
| route B gan_p6_strong_20261109_201344.pt | 0.0411 | 0.133 | 0.165 | 0.198 | 0.08 | 0.41 | 0.147 | 0.132 | 0.095 | 0.036 | 0.03 | 0.99 | 0.35 | 1.82 | 1.09 | 1.65 | 1.12 | 0.86 | 0.68 | 2.58 | 4.3 | 135 | yes |
| AdvectedDetail | 0.0335 | 0.144 | 0.370 | 0.390 | 0.41 | 0.24 | 0.207 | 0.205 | 0.187 | 0.148 | -0.16 | 0.96 | 0.29 | 1.72 | 0.97 | 1.81 | 1.62 | 0.07 | 0.91 | 3.87 | 11.6 | 63 | yes |
| Solver3D(scalar_res=128) | 0.0058 | 0.006 | 0.031 | 0.050 | 0.02 | 0.05 | 0.010 | 0.009 | 0.014 | 0.026 | 0.03 | 1.00 | 0.09 | 1.02 | 1.03 | - | 1.45 | 0.00 | 0.46 | 2.17 | 2.6 | 90 | yes |
| pressure_mode="fused" | 0.0244 | 0.073 | 0.145 | 0.257 | -0.06 | -0.44 | 0.178 | 0.153 | 0.170 | 0.270 | -0.31 | 1.00 | 0.17 | 1.00 | 0.65 | - | 0.74 | 0.00 | 0.43 | 1.35 | 2.6 | 51 | yes |
| fused + scalar_res=128 + AdvectedDetail | 0.0194 | 0.098 | 0.543 | 0.432 | 0.68 | 0.50 | 0.190 | 0.140 | 0.234 | 0.147 | 0.14 | 0.96 | 0.35 | 2.17 | 1.43 | - | 1.30 | 0.08 | 1.05 | 7.04 | 7.9 | 102 | over: frame |
| direct 128^3 sim | 0.0310 | 0.107 | 0.031 | 0.058 | 0.00 | 0.10 | 0.014 | 0.013 | 0.030 | 0.068 | -0.07 | 0.98 | 0.30 | 0.95 | 0.93 | - | 11.41 | 0.00 | 0.46 | 12.01 | 15.7 | 107 | over: sim,frame |

## Not available (re-run when they land)

- sandbox modes: ['raw', 'trilinear', 'neural_vol', 'neural_screen', 'ref128', 'detail', 'detail_screen', 'hires', 'hires_detail']; new since the screen-space baseline: ['detail', 'detail_screen', 'hires', 'hires_detail'] (harness rows are built from the module interfaces, not the sandbox modes)

## What produced each row

- **steered 256^3 (reference)** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "sim": "Solver3D(256, SolverConfig3D(**preset)) UNGUIDED (cost proxy; guide cost not included)", "config": {"pressure_tol": 2e-05, "pressure_max_iters": 40, "pressure_mode": "pcg"}, "renderer": {"size": [1280, 720], "march_scale": 0.5}}`
- **raw 64^3** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "sim": "Solver3D(64, runtime_config(**{'density_dissipation': 1.2, 'temperature_cooling': 5.0}))", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg"}, "renderer": {"size": [1280, 720], "march_scale": 0.5}}`
- **trilinear 256^3** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "sim": "Solver3D(64, runtime_config(**{'density_dissipation': 1.2, 'temperature_cooling': 5.0}))", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg"}, "upscale": "runtime3d.classical_upscale_3d(state, 4, 'trilinear')", "renderer": {"size": [1280, 720], "march_scale": 0.5}}`
- **unsharp mask (screen)** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "sim": "Solver3D(64, runtime_config(**{'density_dissipation': 1.2, 'temperature_cooling': 5.0}))", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg"}, "unsharp": {"sigma_fullres": 4.0, "amount": 0.25, "source": "C:\\Users\\ricej\\AppData\\Local\\neural-vfx\\runs\\eval3d\\unsharp_s3d_v1_train.json"}, "renderer": {"size": [1280, 720], "march_scale": 0.5}}`
- **route B latest.pt** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "sim": "Solver3D(64, runtime_config(**{'density_dissipation': 1.2, 'temperature_cooling': 5.0}))", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg"}, "checkpoint": {"path": "C:\\Users\\ricej\\AppData\\Local\\neural-vfx\\checkpoints\\screen3d\\latest.pt", "mtime": "2026-11-09 19:00", "note": "current checkpoint", "process_scale": 0.5}, "renderer": {"size": [1280, 720], "march_scale": 0.5}}`
- **route B v1_reg_backup.pt** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "sim": "Solver3D(64, runtime_config(**{'density_dissipation': 1.2, 'temperature_cooling': 5.0}))", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg"}, "checkpoint": {"path": "C:\\Users\\ricej\\AppData\\Local\\neural-vfx\\checkpoints\\screen3d\\v1_reg_backup.pt", "mtime": "2026-11-09 19:30", "note": "regression-trained backup", "process_scale": 0.5}, "renderer": {"size": [1280, 720], "march_scale": 0.5}}`
- **route B gan_p6_strong_20261109_201344.pt** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "sim": "Solver3D(64, runtime_config(**{'density_dissipation': 1.2, 'temperature_cooling': 5.0}))", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg"}, "checkpoint": {"path": "C:\\Users\\ricej\\AppData\\Local\\neural-vfx\\checkpoints\\screen3d\\gan_p6_strong_20261109_201344.pt", "mtime": "2026-11-09 20:18", "note": "newest gan_*.pt (adversarial run; not designated by its owner)", "process_scale": 0.5}, "renderer": {"size": [1280, 720], "march_scale": 0.5}}`
- **AdvectedDetail** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "sim": "Solver3D(64, runtime_config(**{'density_dissipation': 1.2, 'temperature_cooling': 5.0}))", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg"}, "advected_detail": {"params": null}, "renderer": {"size": [1280, 720], "march_scale": 0.5}}`
- **Solver3D(scalar_res=128)** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "sim": "Solver3D(64, runtime_config(**{'density_dissipation': 1.2, 'temperature_cooling': 5.0}), scalar_res=128)", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg"}, "renderer": {"size": [1280, 720], "march_scale": 0.5}}`
- **pressure_mode="fused"** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "sim": "Solver3D(64, runtime_config(**{'density_dissipation': 1.2, 'temperature_cooling': 5.0, 'pressure_mode': 'fused'}))", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "fused"}, "renderer": {"size": [1280, 720], "march_scale": 0.5}}`
- **fused + scalar_res=128 + AdvectedDetail** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "sim": "Solver3D(64, runtime_config(**{'density_dissipation': 1.2, 'temperature_cooling': 5.0, 'pressure_mode': 'fused'}), scalar_res=128)", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "fused"}, "advected_detail": {"params": null}, "renderer": {"size": [1280, 720], "march_scale": 0.5}}`
- **direct 128^3 sim** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "sim": "Solver3D(128, SolverConfig3D(**preset))  (full-quality config, as sandbox ref128)", "config": {"pressure_tol": 2e-05, "pressure_max_iters": 40, "pressure_mode": "pcg"}, "renderer": {"size": [1280, 720], "march_scale": 0.5}}`

## Per preset

### preset 2 - contact sheet `C:\Users\ricej\AppData\Local\Temp\claude\a2724ba67304cd37\8eb588dd-d349-48eb-9c47-aae66145cffc\scratchpad\cm_smoke\contact_p2.png` (frame 2, 1:1 crops 360 px at [[441, 289], [441, 360]])

| approach | L1 | MS-DSSIM | band dist | patch SWD | tLP | tLP log | pop | tOF | frame ms |
|---|---|---|---|---|---|---|---|---|---|
| steered 256^3 (reference) | 0.0000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 1.00 | 0.00 | - |
| raw 64^3 | 0.0244 | 0.073 | 0.256 | 0.170 | 0.270 | -0.31 | 1.00 | 0.17 | - |
| trilinear 256^3 | 0.0251 | 0.076 | 0.270 | 0.176 | 0.277 | -0.32 | 1.00 | 0.17 | - |
| unsharp mask (screen) | 0.0237 | 0.070 | 0.201 | 0.123 | 0.223 | -0.25 | 1.00 | 0.18 | - |
| route B latest.pt | 0.0220 | 0.063 | 0.177 | 0.144 | 0.235 | -0.27 | 0.98 | 0.22 | - |
| route B v1_reg_backup.pt | 0.0220 | 0.063 | 0.177 | 0.144 | 0.235 | -0.27 | 0.98 | 0.22 | - |
| route B gan_p6_strong_20261109_201344.pt | 0.0411 | 0.133 | 0.198 | 0.095 | 0.036 | 0.03 | 0.99 | 0.35 | - |
| AdvectedDetail | 0.0335 | 0.144 | 0.390 | 0.187 | 0.148 | -0.16 | 0.96 | 0.29 | - |
| Solver3D(scalar_res=128) | 0.0058 | 0.006 | 0.050 | 0.014 | 0.026 | 0.03 | 1.00 | 0.09 | - |
| pressure_mode="fused" | 0.0244 | 0.073 | 0.257 | 0.170 | 0.270 | -0.31 | 1.00 | 0.17 | - |
| fused + scalar_res=128 + AdvectedDetail | 0.0194 | 0.098 | 0.432 | 0.234 | 0.147 | 0.14 | 0.96 | 0.35 | - |
| direct 128^3 sim | 0.0310 | 0.107 | 0.058 | 0.030 | 0.068 | -0.07 | 0.98 | 0.30 | - |

## Notes

- Spatial metrics: K evenly spaced full-resolution frames; temporal metrics: all frames 2x2-pooled (the route B processed scale). Effect region = dilated union of raw-64^3 and reference alpha.
- Rows on their own sim (scalar_res, fused, combo, direct 128^3) are NOT steered like the reference, so their pixel-aligned columns (L1, MS-DSSIM) include large-scale drift; compare them on the distribution and temporal columns.
- swim r is only defined for approaches that add detail to the raw 64^3 render (detail layer = out - raw).
- reference cost row = UNGUIDED 256^3 step + render (offline; guide cost excluded).
- route B cost = raw render with aux (render) + ScreenDetail.infer (detail).
