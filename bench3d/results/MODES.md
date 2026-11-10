# Approach comparison - `python -m bench3d.compare_modes`

Generated 2026-11-09 23:14:46. Presets [1, 2, 3, 4], 48 recorded frames after 90 warm-up steps, 2560x1440, march_scale 0.5, orbiting camera (start -30.0 deg, 1.0 deg/step, distance 1.9).

> **GPU SHARED during this run** (1 other compute process(es): 27112 ...s\Python\Python312\python.exe; GPU utilization 39% before the benchmark). All ms numbers are NOT representative; the lead re-runs on an idle GPU.

Lower is better everywhere except the signed columns (2-4px / 8-16px band log10 power ratio to the reference, tLP log ratio: 0 = like the reference, < 0 missing detail / too static, > 0 too much / boiling). `pop` 1 = as bursty as the reference. Ratios (flicker r, jerk r, swim r) are to the reference; they are legacy upper-bound gates that REWARD blur (r < 1 is not better). Quality = mean over presets; cost = mean of per-preset p50 GPU ms, worst = max single frame. Budget sim 2.0 / detail 2.5 / render 1.5 / frame 6.0 ms.

Pixel-aligned (legacy) | distribution-level (fair to unaligned detail) | temporal vs reference | cost

| approach | L1 | MS-DSSIM | spec err | band dist | 2-4px | 8-16px | grad W1 | contr W1 | patch SWD | tLP | tLP log | pop | tOF px | flicker r | jerk r | swim r | sim ms | detail ms | render ms | frame ms | worst ms | VRAM MB | budget |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **steered 256^3 (reference)** | 0.0000 | 0.000 | 0.000 | 0.000 | -0.00 | -0.00 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 | - | 62.45 | 0.00 | 1.31 | 63.76 | 71.7 | 574 | offline |
| **raw 64^3** | 0.0283 | 0.112 | 0.222 | 0.359 | -0.06 | -0.73 | 0.218 | 0.201 | 0.220 | 0.440 | -0.58 | 1.00 | 1.27 | 0.50 | 0.50 | - | 0.79 | 0.00 | 0.72 | 1.52 | 1.9 | 70 | yes |
| trilinear 256^3 | 0.0284 | 0.110 | 0.240 | 0.395 | -0.06 | -0.81 | 0.230 | 0.212 | 0.233 | 0.454 | -0.61 | 1.00 | 1.28 | 0.48 | 0.49 | 0.14 | 0.80 | 0.16 | 1.32 | 2.29 | 2.7 | 243 | yes |
| unsharp mask (screen) | 0.0286 | 0.115 | 0.198 | 0.293 | 0.06 | -0.57 | 0.184 | 0.167 | 0.189 | 0.403 | -0.52 | 1.01 | 1.28 | 0.55 | 0.55 | 0.06 | 0.79 | 2.30 | 0.74 | 3.84 | 4.5 | 215 | yes |
| route B final1_v2_reg_20261109_210028.pt | 0.0284 | 0.118 | 0.088 | 0.088 | -0.01 | -0.10 | 0.163 | 0.156 | 0.125 | 0.305 | -0.37 | 1.04 | 1.35 | 0.68 | 0.69 | 0.45 | 0.79 | 2.10 | 0.81 | 3.71 | 4.2 | 410 | yes |
| route B v1_reg_backup.pt | 0.0272 | 0.110 | 0.134 | 0.185 | -0.02 | -0.31 | 0.187 | 0.173 | 0.169 | 0.354 | -0.44 | 1.00 | 1.35 | 0.60 | 0.62 | 0.42 | 0.79 | 1.99 | 0.81 | 3.60 | 4.1 | 410 | yes |
| route B final2_v2_gan_20261109_212544.pt | 0.0304 | 0.135 | 0.056 | 0.051 | 0.00 | -0.07 | 0.094 | 0.090 | 0.068 | 0.194 | -0.22 | 1.05 | 1.34 | 0.83 | 0.86 | 0.61 | 0.80 | 2.10 | 0.81 | 3.71 | 4.1 | 410 | yes |
| route B teacherA_v2_20261109_214240.pt [report-only] | 0.0280 | 0.110 | 0.139 | 0.191 | -0.01 | -0.26 | 0.207 | 0.196 | 0.174 | 0.388 | -0.50 | 1.03 | 1.33 | 0.56 | 0.59 | 0.35 | 0.80 | 10.82 | 0.86 | 12.49 | 13.3 | 514 | report-only (over: detail,frame) |
| route B teacherA_gan_20261109_222144.pt [report-only] | 0.0297 | 0.128 | 0.080 | 0.096 | -0.00 | -0.14 | 0.116 | 0.108 | 0.097 | 0.254 | -0.30 | 1.02 | 1.32 | 0.78 | 0.79 | 0.58 | 0.80 | 10.84 | 0.84 | 12.49 | 13.1 | 514 | report-only (over: detail,frame) |
| detail (AdvectedDetail medium) | 0.0373 | 0.179 | 0.334 | 0.270 | 0.42 | 0.25 | 0.112 | 0.080 | 0.107 | 0.093 | -0.06 | 1.10 | 1.34 | 0.98 | 0.96 | 0.91 | 0.79 | 0.07 | 1.29 | 2.16 | 2.7 | 82 | yes |
| detail (AdvectedDetail low) | 0.0366 | 0.170 | 0.220 | 0.183 | 0.26 | 0.16 | 0.081 | 0.068 | 0.078 | 0.133 | -0.14 | 1.09 | 1.36 | 0.87 | 0.85 | 0.84 | 0.79 | 0.07 | 1.22 | 2.09 | 2.5 | 82 | yes |
| detail (AdvectedDetail high) | 0.0372 | 0.180 | 0.367 | 0.293 | 0.47 | 0.26 | 0.128 | 0.087 | 0.122 | 0.087 | -0.05 | 1.09 | 1.34 | 1.02 | 0.99 | 0.93 | 0.79 | 0.07 | 1.61 | 2.48 | 3.1 | 82 | over: render |
| hires (scalar_res=128) (unmatched: own 64^3 flow) | 0.0596 | 0.206 | 0.147 | 0.199 | -0.04 | -0.44 | 0.147 | 0.131 | 0.143 | 0.282 | -0.34 | 0.99 | 1.65 | 0.59 | 0.61 | - | 1.18 | 0.00 | 0.76 | 1.95 | 2.4 | 109 | yes |
| hires_detail (scalar_res=128 + detail medium) (unmatched: own 64^3 flow) | 0.0596 | 0.250 | 0.512 | 0.417 | 0.64 | 0.47 | 0.293 | 0.220 | 0.213 | 0.148 | 0.13 | 1.13 | 1.69 | 1.20 | 1.21 | - | 1.18 | 0.07 | 1.35 | 2.61 | 3.1 | 121 | yes |
| hires (scalar_res=128) + fused_config() (unmatched: own 64^3 flow) | 0.0601 | 0.207 | 0.144 | 0.191 | -0.04 | -0.43 | 0.152 | 0.136 | 0.140 | 0.297 | -0.36 | 1.02 | 1.65 | 0.58 | 0.59 | - | 0.93 | 0.00 | 0.76 | 1.72 | 2.4 | 109 | yes |
| hires_detail (scalar_res=128 + detail medium) + fused_config() (unmatched: own 64^3 flow) | 0.0602 | 0.251 | 0.523 | 0.426 | 0.66 | 0.47 | 0.294 | 0.220 | 0.212 | 0.136 | 0.11 | 1.08 | 1.69 | 1.18 | 1.18 | - | 0.93 | 0.07 | 1.36 | 2.38 | 3.0 | 121 | yes |
| hires (scalar_res=128) + confinement_ref_res=128 (unmatched: own 64^3 flow) | 0.0839 | 0.246 | 0.165 | 0.226 | -0.05 | -0.50 | 0.172 | 0.172 | 0.164 | 0.384 | -0.49 | 1.05 | 1.81 | 0.48 | 0.53 | - | 1.18 | 0.00 | 0.75 | 1.94 | 2.5 | 109 | yes |
| hires_detail (scalar_res=128 + detail medium) + confinement_ref_res=128 (unmatched: own 64^3 flow) | 0.0805 | 0.288 | 0.579 | 0.477 | 0.71 | 0.53 | 0.402 | 0.317 | 0.273 | 0.122 | 0.07 | 1.16 | 1.87 | 1.07 | 1.14 | - | 1.18 | 0.07 | 1.32 | 2.58 | 3.1 | 121 | yes |
| raw 64^3 + fused_config() (unmatched: own 64^3 flow) | 0.0494 | 0.168 | 0.219 | 0.355 | -0.06 | -0.72 | 0.220 | 0.204 | 0.223 | 0.462 | -0.62 | 1.01 | 1.56 | 0.48 | 0.48 | - | 0.54 | 0.00 | 0.72 | 1.28 | 1.8 | 70 | yes |
| particles DetailParticles(1M, 128) | 0.0290 | 0.140 | 0.110 | 0.116 | -0.03 | -0.26 | 0.070 | 0.058 | 0.075 | 0.119 | -0.13 | 1.01 | 1.26 | 0.92 | 0.91 | 0.84 | 0.79 | 1.41 | 0.76 | 2.98 | 3.4 | 125 | yes |
| particles 1M@128 + detail medium | 0.0368 | 0.205 | 0.546 | 0.438 | 0.71 | 0.45 | 0.327 | 0.224 | 0.239 | 0.228 | 0.20 | 1.07 | 1.34 | 1.39 | 1.38 | 1.27 | 0.79 | 1.48 | 1.35 | 3.64 | 4.1 | 137 | yes |
| direct 128^3 sim (unmatched: own 64^3 flow) | 0.0805 | 0.236 | 0.163 | 0.246 | -0.05 | -0.46 | 0.168 | 0.156 | 0.160 | 0.218 | -0.25 | 0.99 | 1.92 | 0.61 | 0.75 | - | 5.40 | 0.00 | 0.78 | 6.17 | 7.3 | 126 | over: sim,frame |

## Not available (re-run when they land)

- sandbox modes: ['raw', 'trilinear', 'neural_vol', 'neural_screen', 'ref128', 'detail', 'detail_screen', 'hires', 'hires_detail', 'particles', 'particles_detail', 'raw_default']; new since the screen-space baseline: ['detail', 'detail_screen', 'hires', 'hires_detail', 'particles', 'particles_detail', 'raw_default'] (harness rows are built from the module interfaces, not the sandbox modes)

## What produced each row

- **steered 256^3 (reference)** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "solver_tag": null, "matched_reference": true, "sim": "Solver3D(256, full_config(**preset, **{})) UNGUIDED (cost proxy of the reference; guide excluded)", "config": {"pressure_tol": 2e-05, "pressure_max_iters": 40, "pressure_mode": "pcg", "confinement_ref_res": 0, "vorticity_eps": 6.0}, "renderer": {"size": [2560, 1440], "march_scale": 0.5}}`
- **raw 64^3** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "solver_tag": null, "matched_reference": true, "sim": "Solver3D(64, runtime_config(**preset, **{}))", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg", "confinement_ref_res": 0, "vorticity_eps": 6.0}, "renderer": {"size": [2560, 1440], "march_scale": 0.5}}`
- **trilinear 256^3** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "solver_tag": null, "matched_reference": true, "sim": "Solver3D(64, runtime_config(**preset, **{}))", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg", "confinement_ref_res": 0, "vorticity_eps": 6.0}, "upscale": "runtime3d.classical_upscale_3d(state, 4, 'trilinear')", "renderer": {"size": [2560, 1440], "march_scale": 0.5}}`
- **unsharp mask (screen)** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "solver_tag": null, "matched_reference": true, "sim": "Solver3D(64, runtime_config(**preset, **{}))", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg", "confinement_ref_res": 0, "vorticity_eps": 6.0}, "unsharp": {"sigma_fullres": 4.0, "amount": 0.25, "source": "C:\\Users\\ricej\\AppData\\Local\\neural-vfx\\runs\\eval3d\\unsharp_s3d_v1_train.json"}, "renderer": {"size": [2560, 1440], "march_scale": 0.5}}`
- **route B final1_v2_reg_20261109_210028.pt** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "solver_tag": null, "matched_reference": true, "sim": "Solver3D(64, runtime_config(**preset, **{}))", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg", "confinement_ref_res": 0, "vorticity_eps": 6.0}, "checkpoint": {"path": "C:\\Users\\ricej\\AppData\\Local\\neural-vfx\\checkpoints\\screen3d\\final1_v2_reg_20261109_210028.pt", "mtime": "2026-11-09 21:21", "note": "route B regression (designated; s3d_v2)", "process_scale": 0.5}, "renderer": {"size": [2560, 1440], "march_scale": 0.5}}`
- **route B v1_reg_backup.pt** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "solver_tag": null, "matched_reference": true, "sim": "Solver3D(64, runtime_config(**preset, **{}))", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg", "confinement_ref_res": 0, "vorticity_eps": 6.0}, "checkpoint": {"path": "C:\\Users\\ricej\\AppData\\Local\\neural-vfx\\checkpoints\\screen3d\\v1_reg_backup.pt", "mtime": "2026-11-09 19:30", "note": "route B regression v1 (old, kept for history)", "process_scale": 0.5}, "renderer": {"size": [2560, 1440], "march_scale": 0.5}}`
- **route B final2_v2_gan_20261109_212544.pt** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "solver_tag": null, "matched_reference": true, "sim": "Solver3D(64, runtime_config(**preset, **{}))", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg", "confinement_ref_res": 0, "vorticity_eps": 6.0}, "checkpoint": {"path": "C:\\Users\\ricej\\AppData\\Local\\neural-vfx\\checkpoints\\screen3d\\final2_v2_gan_20261109_212544.pt", "mtime": "2026-11-09 21:41", "note": "route B adversarial (designated)", "process_scale": 0.5}, "renderer": {"size": [2560, 1440], "march_scale": 0.5}}`
- **route B teacherA_v2_20261109_214240.pt [report-only]** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "solver_tag": null, "matched_reference": true, "sim": "Solver3D(64, runtime_config(**preset, **{}))", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg", "confinement_ref_res": 0, "vorticity_eps": 6.0}, "checkpoint": {"path": "C:\\Users\\ricej\\AppData\\Local\\neural-vfx\\checkpoints\\screen3d\\teacherA_v2_20261109_214240.pt", "mtime": "2026-11-09 22:12", "note": "teacher (REPORT-ONLY, not a budget candidate)", "process_scale": 0.5}, "renderer": {"size": [2560, 1440], "march_scale": 0.5}}`
- **route B teacherA_gan_20261109_222144.pt [report-only]** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "solver_tag": null, "matched_reference": true, "sim": "Solver3D(64, runtime_config(**preset, **{}))", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg", "confinement_ref_res": 0, "vorticity_eps": 6.0}, "checkpoint": {"path": "C:\\Users\\ricej\\AppData\\Local\\neural-vfx\\checkpoints\\screen3d\\teacherA_gan_20261109_222144.pt", "mtime": "2026-11-09 22:37", "note": "teacher (REPORT-ONLY, not a budget candidate)", "process_scale": 0.5}, "renderer": {"size": [2560, 1440], "march_scale": 0.5}}`
- **detail (AdvectedDetail medium)** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "solver_tag": null, "matched_reference": true, "sim": "Solver3D(64, runtime_config(**preset, **{}))", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg", "confinement_ref_res": 0, "vorticity_eps": 6.0}, "advected_detail": {"quality": "medium", "params": null}, "renderer": {"size": [2560, 1440], "march_scale": 0.5}}`
- **detail (AdvectedDetail low)** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "solver_tag": null, "matched_reference": true, "sim": "Solver3D(64, runtime_config(**preset, **{}))", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg", "confinement_ref_res": 0, "vorticity_eps": 6.0}, "advected_detail": {"quality": "low", "params": null}, "renderer": {"size": [2560, 1440], "march_scale": 0.5}}`
- **detail (AdvectedDetail high)** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "solver_tag": null, "matched_reference": true, "sim": "Solver3D(64, runtime_config(**preset, **{}))", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg", "confinement_ref_res": 0, "vorticity_eps": 6.0}, "advected_detail": {"quality": "high", "params": null}, "renderer": {"size": [2560, 1440], "march_scale": 0.5}}`
- **hires (scalar_res=128)** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "solver_tag": null, "matched_reference": true, "sim": "Solver3D(64, runtime_config(**preset, **{}), scalar_res=128)", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg", "confinement_ref_res": 0, "vorticity_eps": 6.0}, "renderer": {"size": [2560, 1440], "march_scale": 0.5}}`
- **hires_detail (scalar_res=128 + detail medium)** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "solver_tag": null, "matched_reference": true, "sim": "Solver3D(64, runtime_config(**preset, **{}), scalar_res=128)", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg", "confinement_ref_res": 0, "vorticity_eps": 6.0}, "advected_detail": {"quality": "medium", "params": null}, "renderer": {"size": [2560, 1440], "march_scale": 0.5}}`
- **hires (scalar_res=128) + fused_config()** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "solver_tag": null, "matched_reference": false, "sim": "Solver3D(64, fused_config(**preset, **{}), scalar_res=128)", "config": {"pressure_tol": 5e-05, "pressure_max_iters": 24, "pressure_mode": "fused", "confinement_ref_res": 0, "vorticity_eps": 6.0}, "renderer": {"size": [2560, 1440], "march_scale": 0.5}}`
- **hires_detail (scalar_res=128 + detail medium) + fused_config()** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "solver_tag": null, "matched_reference": false, "sim": "Solver3D(64, fused_config(**preset, **{}), scalar_res=128)", "config": {"pressure_tol": 5e-05, "pressure_max_iters": 24, "pressure_mode": "fused", "confinement_ref_res": 0, "vorticity_eps": 6.0}, "advected_detail": {"quality": "medium", "params": null}, "renderer": {"size": [2560, 1440], "march_scale": 0.5}}`
- **hires (scalar_res=128) + confinement_ref_res=128** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "solver_tag": null, "matched_reference": false, "sim": "Solver3D(64, runtime_config(**preset, **{'confinement_ref_res': 128}), scalar_res=128)", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg", "confinement_ref_res": 128, "vorticity_eps": 6.0}, "renderer": {"size": [2560, 1440], "march_scale": 0.5}}`
- **hires_detail (scalar_res=128 + detail medium) + confinement_ref_res=128** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "solver_tag": null, "matched_reference": false, "sim": "Solver3D(64, runtime_config(**preset, **{'confinement_ref_res': 128}), scalar_res=128)", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg", "confinement_ref_res": 128, "vorticity_eps": 6.0}, "advected_detail": {"quality": "medium", "params": null}, "renderer": {"size": [2560, 1440], "march_scale": 0.5}}`
- **raw 64^3 + fused_config()** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "solver_tag": null, "matched_reference": false, "sim": "Solver3D(64, fused_config(**preset, **{}))", "config": {"pressure_tol": 5e-05, "pressure_max_iters": 24, "pressure_mode": "fused", "confinement_ref_res": 0, "vorticity_eps": 6.0}, "renderer": {"size": [2560, 1440], "march_scale": 0.5}}`
- **particles DetailParticles(1M, 128)** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "solver_tag": null, "matched_reference": true, "sim": "Solver3D(64, runtime_config(**preset, **{}))", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg", "confinement_ref_res": 0, "vorticity_eps": 6.0}, "particles": "DetailParticles(1_000_000, 128, config=runtime_config(**preset)); defaults", "renderer": {"size": [2560, 1440], "march_scale": 0.5}}`
- **particles 1M@128 + detail medium** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "solver_tag": null, "matched_reference": true, "sim": "Solver3D(64, runtime_config(**preset, **{}))", "config": {"pressure_tol": 0.0, "pressure_max_iters": 4, "pressure_mode": "pcg", "confinement_ref_res": 0, "vorticity_eps": 6.0}, "advected_detail": {"quality": "medium", "params": null}, "particles": "DetailParticles(1_000_000, 128, config=runtime_config(**preset)); defaults", "renderer": {"size": [2560, 1440], "march_scale": 0.5}}`
- **direct 128^3 sim** - `{"preset_overrides": {"density_dissipation": 1.2, "temperature_cooling": 5.0}, "solver_version": "sim3d-warp-0.1.0", "solver_tag": null, "matched_reference": false, "sim": "Solver3D(128, full_config(**preset, **{}))", "config": {"pressure_tol": 2e-05, "pressure_max_iters": 40, "pressure_mode": "pcg", "confinement_ref_res": 0, "vorticity_eps": 6.0}, "renderer": {"size": [2560, 1440], "march_scale": 0.5}}`

## Per preset

### preset 1 - contact sheet `C:\Users\ricej\AppData\Local\neural-vfx\runs\compare_modes\latest\contact_p1.png` (frame 27, 1:1 crops 360 px at [[1098, 470], [1098, 39]])

| approach | L1 | MS-DSSIM | band dist | patch SWD | tLP | tLP log | pop | tOF | frame ms |
|---|---|---|---|---|---|---|---|---|---|
| steered 256^3 (reference) | 0.0000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 1.00 | 0.00 | 59.33 |
| raw 64^3 | 0.0308 | 0.126 | 0.499 | 0.252 | 0.500 | -0.70 | 0.99 | 1.58 | 1.52 |
| trilinear 256^3 | 0.0311 | 0.124 | 0.561 | 0.266 | 0.513 | -0.73 | 1.00 | 1.59 | 2.29 |
| unsharp mask (screen) | 0.0310 | 0.128 | 0.404 | 0.221 | 0.473 | -0.65 | 0.98 | 1.57 | 3.86 |
| route B final1_v2_reg_20261109_210028.pt | 0.0308 | 0.134 | 0.132 | 0.157 | 0.341 | -0.42 | 1.13 | 1.70 | 3.71 |
| route B v1_reg_backup.pt | 0.0294 | 0.126 | 0.257 | 0.199 | 0.386 | -0.49 | 1.00 | 1.69 | 3.60 |
| route B final2_v2_gan_20261109_212544.pt | 0.0322 | 0.150 | 0.089 | 0.085 | 0.221 | -0.26 | 1.10 | 1.62 | 3.71 |
| route B teacherA_v2_20261109_214240.pt [report-only] | 0.0304 | 0.125 | 0.270 | 0.206 | 0.436 | -0.58 | 1.02 | 1.65 | 12.42 |
| route B teacherA_gan_20261109_222144.pt [report-only] | 0.0315 | 0.144 | 0.127 | 0.121 | 0.285 | -0.34 | 1.03 | 1.58 | 12.50 |
| detail (AdvectedDetail medium) | 0.0391 | 0.199 | 0.434 | 0.138 | 0.109 | -0.04 | 1.06 | 1.62 | 2.11 |
| detail (AdvectedDetail low) | 0.0383 | 0.188 | 0.329 | 0.096 | 0.137 | -0.13 | 1.06 | 1.64 | 2.05 |
| detail (AdvectedDetail high) | 0.0390 | 0.200 | 0.450 | 0.153 | 0.110 | -0.02 | 1.05 | 1.61 | 2.36 |
| hires (scalar_res=128) | 0.0617 | 0.215 | 0.300 | 0.173 | 0.318 | -0.40 | 0.97 | 1.96 | 1.92 |
| hires_detail (scalar_res=128 + detail medium) | 0.0612 | 0.264 | 0.575 | 0.259 | 0.184 | 0.17 | 1.06 | 1.98 | 2.55 |
| hires (scalar_res=128) + fused_config() | 0.0653 | 0.224 | 0.300 | 0.169 | 0.343 | -0.43 | 1.01 | 1.93 | 1.65 |
| hires_detail (scalar_res=128 + detail medium) + fused_config() | 0.0647 | 0.270 | 0.578 | 0.262 | 0.190 | 0.15 | 1.03 | 1.99 | 2.31 |
| hires (scalar_res=128) + confinement_ref_res=128 | 0.0838 | 0.250 | 0.346 | 0.158 | 0.454 | -0.61 | 1.03 | 1.99 | 1.89 |
| hires_detail (scalar_res=128 + detail medium) + confinement_ref_res=128 | 0.0812 | 0.307 | 0.702 | 0.414 | 0.176 | 0.16 | 1.13 | 2.06 | 2.51 |
| raw 64^3 + fused_config() | 0.0548 | 0.183 | 0.494 | 0.253 | 0.534 | -0.77 | 1.04 | 1.90 | 1.24 |
| particles DetailParticles(1M, 128) | 0.0312 | 0.160 | 0.194 | 0.090 | 0.143 | -0.16 | 1.02 | 1.43 | 2.96 |
| particles 1M@128 + detail medium | 0.0381 | 0.228 | 0.579 | 0.284 | 0.269 | 0.24 | 1.02 | 1.54 | 3.59 |
| direct 128^3 sim | 0.0852 | 0.243 | 0.301 | 0.139 | 0.202 | -0.22 | 0.98 | 2.07 | 5.68 |

### preset 2 - contact sheet `C:\Users\ricej\AppData\Local\neural-vfx\runs\compare_modes\latest\contact_p2.png` (frame 27, 1:1 crops 360 px at [[1062, 672], [1003, 385]])

| approach | L1 | MS-DSSIM | band dist | patch SWD | tLP | tLP log | pop | tOF | frame ms |
|---|---|---|---|---|---|---|---|---|---|
| steered 256^3 (reference) | 0.0000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 1.00 | 0.00 | 70.93 |
| raw 64^3 | 0.0203 | 0.096 | 0.176 | 0.200 | 0.429 | -0.56 | 1.00 | 0.97 | 1.55 |
| trilinear 256^3 | 0.0201 | 0.094 | 0.189 | 0.211 | 0.443 | -0.58 | 0.99 | 0.98 | 2.31 |
| unsharp mask (screen) | 0.0206 | 0.098 | 0.156 | 0.168 | 0.388 | -0.49 | 1.00 | 0.99 | 3.86 |
| route B final1_v2_reg_20261109_210028.pt | 0.0212 | 0.106 | 0.032 | 0.122 | 0.328 | -0.40 | 1.04 | 1.01 | 3.74 |
| route B v1_reg_backup.pt | 0.0200 | 0.097 | 0.104 | 0.163 | 0.381 | -0.48 | 1.01 | 1.01 | 3.62 |
| route B final2_v2_gan_20261109_212544.pt | 0.0242 | 0.124 | 0.028 | 0.064 | 0.229 | -0.26 | 1.09 | 1.03 | 3.74 |
| route B teacherA_v2_20261109_214240.pt [report-only] | 0.0208 | 0.100 | 0.062 | 0.152 | 0.393 | -0.50 | 1.04 | 1.00 | 12.54 |
| route B teacherA_gan_20261109_222144.pt [report-only] | 0.0231 | 0.114 | 0.042 | 0.090 | 0.285 | -0.33 | 1.02 | 1.02 | 12.50 |
| detail (AdvectedDetail medium) | 0.0317 | 0.163 | 0.155 | 0.090 | 0.074 | -0.05 | 1.17 | 1.05 | 2.20 |
| detail (AdvectedDetail low) | 0.0308 | 0.153 | 0.082 | 0.063 | 0.116 | -0.12 | 1.17 | 1.07 | 2.13 |
| detail (AdvectedDetail high) | 0.0316 | 0.164 | 0.176 | 0.107 | 0.063 | -0.02 | 1.15 | 1.04 | 2.51 |
| hires (scalar_res=128) | 0.0509 | 0.188 | 0.099 | 0.124 | 0.260 | -0.30 | 1.01 | 1.21 | 1.98 |
| hires_detail (scalar_res=128 + detail medium) | 0.0525 | 0.231 | 0.293 | 0.189 | 0.235 | 0.21 | 1.22 | 1.27 | 2.68 |
| hires (scalar_res=128) + fused_config() | 0.0480 | 0.178 | 0.093 | 0.125 | 0.285 | -0.33 | 1.03 | 1.16 | 1.75 |
| hires_detail (scalar_res=128 + detail medium) + fused_config() | 0.0500 | 0.226 | 0.284 | 0.181 | 0.191 | 0.17 | 1.18 | 1.22 | 2.43 |
| hires (scalar_res=128) + confinement_ref_res=128 | 0.0908 | 0.252 | 0.180 | 0.173 | 0.342 | -0.42 | 1.06 | 1.47 | 1.97 |
| hires_detail (scalar_res=128 + detail medium) + confinement_ref_res=128 | 0.0855 | 0.285 | 0.311 | 0.183 | 0.135 | 0.12 | 1.18 | 1.53 | 2.64 |
| raw 64^3 + fused_config() | 0.0333 | 0.131 | 0.166 | 0.196 | 0.436 | -0.57 | 1.02 | 1.07 | 1.29 |
| particles DetailParticles(1M, 128) | 0.0237 | 0.123 | 0.037 | 0.070 | 0.112 | -0.12 | 1.01 | 0.98 | 3.03 |
| particles 1M@128 + detail medium | 0.0338 | 0.186 | 0.315 | 0.212 | 0.253 | 0.22 | 1.14 | 1.05 | 3.70 |
| direct 128^3 sim | 0.0815 | 0.237 | 0.176 | 0.174 | 0.135 | -0.15 | 0.98 | 1.77 | 6.91 |

### preset 3 - contact sheet `C:\Users\ricej\AppData\Local\neural-vfx\runs\compare_modes\latest\contact_p3.png` (frame 27, 1:1 crops 360 px at [[1073, 561], [1100, 367]])

| approach | L1 | MS-DSSIM | band dist | patch SWD | tLP | tLP log | pop | tOF | frame ms |
|---|---|---|---|---|---|---|---|---|---|
| steered 256^3 (reference) | 0.0000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 1.00 | 0.00 | 53.63 |
| raw 64^3 | 0.0331 | 0.118 | 0.481 | 0.263 | 0.441 | -0.58 | 1.02 | 1.39 | 1.49 |
| trilinear 256^3 | 0.0333 | 0.117 | 0.524 | 0.279 | 0.457 | -0.61 | 1.02 | 1.40 | 2.24 |
| unsharp mask (screen) | 0.0332 | 0.121 | 0.386 | 0.228 | 0.409 | -0.53 | 1.03 | 1.40 | 3.81 |
| route B final1_v2_reg_20261109_210028.pt | 0.0328 | 0.123 | 0.111 | 0.123 | 0.252 | -0.29 | 1.02 | 1.49 | 3.69 |
| route B v1_reg_backup.pt | 0.0314 | 0.112 | 0.259 | 0.192 | 0.334 | -0.41 | 1.00 | 1.49 | 3.58 |
| route B final2_v2_gan_20261109_212544.pt | 0.0340 | 0.138 | 0.052 | 0.066 | 0.142 | -0.15 | 1.02 | 1.48 | 3.70 |
| route B teacherA_v2_20261109_214240.pt [report-only] | 0.0324 | 0.112 | 0.287 | 0.212 | 0.375 | -0.47 | 1.03 | 1.49 | 12.49 |
| route B teacherA_gan_20261109_222144.pt [report-only] | 0.0336 | 0.131 | 0.137 | 0.108 | 0.211 | -0.24 | 1.03 | 1.47 | 12.48 |
| detail (AdvectedDetail medium) | 0.0397 | 0.172 | 0.245 | 0.098 | 0.137 | -0.15 | 1.02 | 1.49 | 2.12 |
| detail (AdvectedDetail low) | 0.0392 | 0.163 | 0.161 | 0.087 | 0.192 | -0.21 | 1.02 | 1.50 | 2.05 |
| detail (AdvectedDetail high) | 0.0396 | 0.172 | 0.276 | 0.113 | 0.130 | -0.14 | 1.01 | 1.48 | 2.44 |
| hires (scalar_res=128) | 0.0726 | 0.238 | 0.259 | 0.162 | 0.254 | -0.29 | 1.00 | 2.03 | 1.94 |
| hires_detail (scalar_res=128 + detail medium) | 0.0688 | 0.270 | 0.452 | 0.236 | 0.075 | 0.07 | 1.09 | 2.05 | 2.57 |
| hires (scalar_res=128) + fused_config() | 0.0757 | 0.246 | 0.249 | 0.166 | 0.261 | -0.30 | 1.06 | 2.08 | 1.66 |
| hires_detail (scalar_res=128 + detail medium) + fused_config() | 0.0723 | 0.277 | 0.481 | 0.231 | 0.072 | 0.07 | 1.04 | 2.09 | 2.28 |
| hires (scalar_res=128) + confinement_ref_res=128 | 0.0904 | 0.264 | 0.261 | 0.201 | 0.344 | -0.42 | 1.05 | 2.12 | 1.96 |
| hires_detail (scalar_res=128 + detail medium) + confinement_ref_res=128 | 0.0857 | 0.302 | 0.549 | 0.325 | 0.085 | 0.05 | 1.10 | 2.18 | 2.55 |
| raw 64^3 + fused_config() | 0.0635 | 0.201 | 0.469 | 0.265 | 0.472 | -0.64 | 0.99 | 1.87 | 1.22 |
| particles DetailParticles(1M, 128) | 0.0320 | 0.149 | 0.156 | 0.083 | 0.118 | -0.13 | 1.01 | 1.46 | 2.96 |
| particles 1M@128 + detail medium | 0.0378 | 0.200 | 0.431 | 0.233 | 0.129 | 0.12 | 1.00 | 1.51 | 3.60 |
| direct 128^3 sim | 0.0863 | 0.253 | 0.309 | 0.199 | 0.256 | -0.31 | 0.97 | 2.27 | 5.26 |

### preset 4 - contact sheet `C:\Users\ricej\AppData\Local\neural-vfx\runs\compare_modes\latest\contact_p4.png` (frame 27, 1:1 crops 360 px at [[1063, 662], [1086, 572]])

| approach | L1 | MS-DSSIM | band dist | patch SWD | tLP | tLP log | pop | tOF | frame ms |
|---|---|---|---|---|---|---|---|---|---|
| steered 256^3 (reference) | 0.0000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 1.00 | 0.00 | 71.14 |
| raw 64^3 | 0.0291 | 0.109 | 0.279 | 0.166 | 0.389 | -0.49 | 1.02 | 1.13 | 1.53 |
| trilinear 256^3 | 0.0291 | 0.107 | 0.304 | 0.176 | 0.403 | -0.52 | 1.01 | 1.14 | 2.30 |
| unsharp mask (screen) | 0.0295 | 0.112 | 0.226 | 0.137 | 0.340 | -0.42 | 1.02 | 1.14 | 3.85 |
| route B final1_v2_reg_20261109_210028.pt | 0.0287 | 0.110 | 0.078 | 0.101 | 0.298 | -0.35 | 0.99 | 1.20 | 3.71 |
| route B v1_reg_backup.pt | 0.0279 | 0.106 | 0.121 | 0.121 | 0.315 | -0.38 | 1.00 | 1.21 | 3.60 |
| route B final2_v2_gan_20261109_212544.pt | 0.0310 | 0.128 | 0.036 | 0.056 | 0.186 | -0.21 | 0.98 | 1.24 | 3.71 |
| route B teacherA_v2_20261109_214240.pt [report-only] | 0.0284 | 0.104 | 0.145 | 0.127 | 0.349 | -0.43 | 1.02 | 1.18 | 12.50 |
| route B teacherA_gan_20261109_222144.pt [report-only] | 0.0307 | 0.123 | 0.077 | 0.068 | 0.234 | -0.27 | 1.00 | 1.22 | 12.48 |
| detail (AdvectedDetail medium) | 0.0387 | 0.184 | 0.245 | 0.101 | 0.053 | -0.01 | 1.15 | 1.22 | 2.20 |
| detail (AdvectedDetail low) | 0.0380 | 0.174 | 0.161 | 0.063 | 0.085 | -0.08 | 1.13 | 1.24 | 2.14 |
| detail (AdvectedDetail high) | 0.0386 | 0.185 | 0.270 | 0.115 | 0.046 | 0.01 | 1.14 | 1.22 | 2.62 |
| hires (scalar_res=128) | 0.0533 | 0.183 | 0.140 | 0.111 | 0.295 | -0.35 | 0.99 | 1.40 | 1.95 |
| hires_detail (scalar_res=128 + detail medium) | 0.0559 | 0.237 | 0.349 | 0.170 | 0.097 | 0.09 | 1.14 | 1.45 | 2.64 |
| hires (scalar_res=128) + fused_config() | 0.0514 | 0.180 | 0.123 | 0.102 | 0.300 | -0.36 | 0.99 | 1.42 | 1.82 |
| hires_detail (scalar_res=128 + detail medium) + fused_config() | 0.0537 | 0.233 | 0.361 | 0.175 | 0.093 | 0.07 | 1.09 | 1.47 | 2.48 |
| hires (scalar_res=128) + confinement_ref_res=128 | 0.0704 | 0.218 | 0.116 | 0.124 | 0.396 | -0.50 | 1.06 | 1.64 | 1.96 |
| hires_detail (scalar_res=128 + detail medium) + confinement_ref_res=128 | 0.0696 | 0.258 | 0.347 | 0.170 | 0.091 | -0.06 | 1.22 | 1.69 | 2.63 |
| raw 64^3 + fused_config() | 0.0461 | 0.157 | 0.290 | 0.177 | 0.405 | -0.52 | 0.99 | 1.39 | 1.37 |
| particles DetailParticles(1M, 128) | 0.0292 | 0.130 | 0.076 | 0.057 | 0.103 | -0.11 | 1.02 | 1.17 | 2.95 |
| particles 1M@128 + detail medium | 0.0377 | 0.205 | 0.429 | 0.226 | 0.260 | 0.23 | 1.13 | 1.25 | 3.66 |
| direct 128^3 sim | 0.0692 | 0.212 | 0.197 | 0.128 | 0.278 | -0.33 | 1.01 | 1.56 | 6.84 |

## Notes

- Spatial metrics: K evenly spaced full-resolution frames; temporal metrics: all frames 2x2-pooled (the route B processed scale). Effect region = dilated union of raw-64^3 and reference alpha.
- Rows on their own sim (scalar_res, fused, combo, direct 128^3) are NOT steered like the reference, so their pixel-aligned columns (L1, MS-DSSIM) include large-scale drift; compare them on the distribution and temporal columns.
- swim r is only defined for approaches that add detail to the raw 64^3 render (detail layer = out - raw).
- reference cost row = UNGUIDED 256^3 step + render (offline; guide cost excluded).
- route B cost = raw render with aux (render) + ScreenDetail.infer (detail).
- AdvectedDetail cost: detail = AdvectedDetail.step(); its per-sample cost in the march is inside render.
- Identical rows mean identical outputs (e.g. two checkpoint files holding the same weights).
