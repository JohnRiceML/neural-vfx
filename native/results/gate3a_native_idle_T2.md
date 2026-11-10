Gate 3a results (native ray-marcher vs vfx.render3d.VolumeRenderer), 2026-11-09 23:16:59, nvfx 0.1 sim3d-d3d12-0.1.0

DLL `D:\worktrees\neural-vfx-native-runtime\native\build\bin\nvfx.dll`; NVFX_RENDER_DITHER_MODE in the gate process: `unset`; images `D:\worktrees\neural-vfx-native-runtime\native\build\test_results\gate3a_images`.

| check | expected | result | measured |
|---|---|---|---|
| HC3_py_r4_negate_campfire | FAIL | ok | HC3 Python motion negate vs Python campfire: EPE 18.339 px -> FAIL (ok) |
| HC3_py_r4_negate_plume | FAIL | ok | HC3 Python motion negate vs Python plume: EPE 10.498 px -> FAIL (ok) |
| HC3_py_r4_swapxy_campfire | FAIL | ok | HC3 Python motion swapxy vs Python campfire: EPE 11.644 px -> FAIL (ok) |
| HC3_py_r4_swapxy_plume | FAIL | ok | HC3 Python motion swapxy vs Python plume: EPE 7.540 px -> FAIL (ok) |
| HC3_py_self_campfire | PASS | ok | HC3 Python vs Python campfire: R1 mean 0.000, p99 0; bit-identical {'color': True, 'alpha': True, 'depth': True, 'motion': True} |
| HC3_py_self_plume | PASS | ok | HC3 Python vs Python plume: R1 mean 0.000, p99 0; bit-identical {'color': True, 'alpha': True, 'depth': True, 'motion': True} |
| HC3_py_yaw1_campfire | FAIL | ok | HC3 Python yaw 1 deg vs Python campfire: mean 3.960, p99 46 -> FAIL (control ok); proxy struct_energy 0.952 (x60.60) |
| HC3_py_yaw1_plume | FAIL | ok | HC3 Python yaw 1 deg vs Python plume: mean 2.521, p99 40 -> FAIL (control ok); proxy struct_energy 0.926 (x59.90) |
| NC_R1_yaw1_campfire_closeup | FAIL | ok | NC R1 yaw 1 deg campfire_closeup: mean abs 5.091, p99 41 -> R1 FAIL (control ok); proxy struct_energy 0.966 (x61.96 shuffled) |
| NC_R1_yaw1_campfire_default | FAIL | ok | NC R1 yaw 1 deg campfire_default: mean abs 4.000, p99 46 -> R1 FAIL (control ok); proxy struct_energy 0.952 (x60.57 shuffled) |
| NC_R1_yaw1_campfire_orbit60 | FAIL | ok | NC R1 yaw 1 deg campfire_orbit60: mean abs 4.157, p99 48 -> R1 FAIL (control ok); proxy struct_energy 0.951 (x60.98 shuffled) |
| NC_R1_yaw1_plume_closeup | FAIL | ok | NC R1 yaw 1 deg plume_closeup: mean abs 5.205, p99 32 -> R1 FAIL (control ok); proxy struct_energy 0.942 (x60.24 shuffled) |
| NC_R1_yaw1_plume_default | FAIL | ok | NC R1 yaw 1 deg plume_default: mean abs 2.571, p99 40 -> R1 FAIL (control ok); proxy struct_energy 0.925 (x59.84 shuffled) |
| NC_R1_yaw1_plume_orbit60 | FAIL | ok | NC R1 yaw 1 deg plume_orbit60: mean abs 2.505, p99 36 -> R1 FAIL (control ok); proxy struct_energy 0.905 (x57.80 shuffled) |
| NC_R4_negate_campfire_closeup | FAIL | ok | NC R4 negate campfire_closeup: native motion negate vs Python: EPE mean 38.503 px -> R4 FAIL (control ok) |
| NC_R4_negate_campfire_default | FAIL | ok | NC R4 negate campfire_default: native motion negate vs Python: EPE mean 18.334 px -> R4 FAIL (control ok) |
| NC_R4_negate_campfire_orbit60 | FAIL | ok | NC R4 negate campfire_orbit60: native motion negate vs Python: EPE mean 17.159 px -> R4 FAIL (control ok) |
| NC_R4_negate_plume_closeup | FAIL | ok | NC R4 negate plume_closeup: native motion negate vs Python: EPE mean 30.216 px -> R4 FAIL (control ok) |
| NC_R4_negate_plume_default | FAIL | ok | NC R4 negate plume_default: native motion negate vs Python: EPE mean 10.495 px -> R4 FAIL (control ok) |
| NC_R4_negate_plume_orbit60 | FAIL | ok | NC R4 negate plume_orbit60: native motion negate vs Python: EPE mean 13.336 px -> R4 FAIL (control ok) |
| NC_R4_swapxy_campfire_closeup | FAIL | ok | NC R4 swapxy campfire_closeup: native motion swapxy vs Python: EPE mean 26.421 px -> R4 FAIL (control ok) |
| NC_R4_swapxy_campfire_default | FAIL | ok | NC R4 swapxy campfire_default: native motion swapxy vs Python: EPE mean 11.641 px -> R4 FAIL (control ok) |
| NC_R4_swapxy_campfire_orbit60 | FAIL | ok | NC R4 swapxy campfire_orbit60: native motion swapxy vs Python: EPE mean 12.118 px -> R4 FAIL (control ok) |
| NC_R4_swapxy_plume_closeup | FAIL | ok | NC R4 swapxy plume_closeup: native motion swapxy vs Python: EPE mean 21.312 px -> R4 FAIL (control ok) |
| NC_R4_swapxy_plume_default | FAIL | ok | NC R4 swapxy plume_default: native motion swapxy vs Python: EPE mean 7.538 px -> R4 FAIL (control ok) |
| NC_R4_swapxy_plume_orbit60 | FAIL | ok | NC R4 swapxy plume_orbit60: native motion swapxy vs Python: EPE mean 9.789 px -> R4 FAIL (control ok) |
| N_api_defaults | PASS | ok | nvfx_render_params_default == RenderParams3D() and nvfx_camera_default == Camera() (float32) |
| N_march_size | PASS | ok | march buffer native (1280, 720) vs Python (1280, 720) (w, h) |
| PRE_closeup_coverage_campfire | PASS | ok | PRE close-up campfire: camera_at(0.8) Python alpha > 0.05 covers 0.367 of the frame (>= 1/3); alpha > 0.02: 0.421; tried [1.3, 1.15, 1.0, 0.9, 0.8] |
| PRE_closeup_coverage_plume | PASS | ok | PRE close-up plume: camera_at(0.7) Python alpha > 0.05 covers 0.337 of the frame (>= 1/3); alpha > 0.02: 0.534; tried [1.3, 1.15, 1.0, 0.9, 0.8, 0.7] |
| PRE_upload_campfire | PASS | ok | PRE upload campfire: download after nvfx_solver_upload_state: data identical, solid identical (0 solid cells) |
| PRE_upload_plume | PASS | ok | PRE upload plume: download after nvfx_solver_upload_state: data identical, solid identical (1880 solid cells) |
| R1_campfire_closeup | PASS | ok | R1 campfire_closeup: mean abs 0.087 (<= 1.5), p99 1 (<= 8), max 2; values differing 0.087; proxy struct_energy 0.022 vs shuffled 0.017 (x1.29), blocks>1 0.0000; visual review PENDING |
| R1_campfire_default | PASS | ok | R1 campfire_default: mean abs 0.082 (<= 1.5), p99 1 (<= 8), max 2; values differing 0.082; proxy struct_energy 0.018 vs shuffled 0.016 (x1.14), blocks>1 0.0000; visual review PENDING |
| R1_campfire_orbit60 | PASS | ok | R1 campfire_orbit60: mean abs 0.082 (<= 1.5), p99 1 (<= 8), max 3; values differing 0.082; proxy struct_energy 0.018 vs shuffled 0.016 (x1.14), blocks>1 0.0000; visual review PENDING |
| R1_dither_modes | REPORT | ok | R1 per dither mode (worst mean / p99 over 6 cases): unset: 0.0875 / 1; 0: 0.0875 / 1; 1: 0.0129 / 1; 2: 0.0864 / 1. In-process default == fresh-process default: True; default is byte-identical to mode(s) 0. REPORT ONLY: the gate is R1 in the default mode. |
| R1_plume_closeup | PASS | ok | R1 plume_closeup: mean abs 0.087 (<= 1.5), p99 1 (<= 8), max 30; values differing 0.087; proxy struct_energy 0.019 vs shuffled 0.017 (x1.11), blocks>1 0.0000; visual review PENDING |
| R1_plume_default | PASS | ok | R1 plume_default: mean abs 0.079 (<= 1.5), p99 1 (<= 8), max 2; values differing 0.079; proxy struct_energy 0.016 vs shuffled 0.016 (x1.03), blocks>1 0.0000; visual review PENDING |
| R1_plume_orbit60 | PASS | ok | R1 plume_orbit60: mean abs 0.079 (<= 1.5), p99 1 (<= 8), max 2; values differing 0.079; proxy struct_energy 0.016 vs shuffled 0.016 (x1.05), blocks>1 0.0000; visual review PENDING |
| R2_campfire_closeup | PASS | ok | R2 campfire_closeup: alpha mean abs 0.00008 (<= 0.01), p99 0.00049 (<= 0.05), max 0.0010 |
| R2_campfire_default | PASS | ok | R2 campfire_default: alpha mean abs 0.00003 (<= 0.01), p99 0.00049 (<= 0.05), max 0.0023 |
| R2_campfire_orbit60 | PASS | ok | R2 campfire_orbit60: alpha mean abs 0.00003 (<= 0.01), p99 0.00049 (<= 0.05), max 0.0031 |
| R2_plume_closeup | PASS | ok | R2 plume_closeup: alpha mean abs 0.00004 (<= 0.01), p99 0.00024 (<= 0.05), max 0.0367 |
| R2_plume_default | PASS | ok | R2 plume_default: alpha mean abs 0.00001 (<= 0.01), p99 0.00012 (<= 0.05), max 0.0006 |
| R2_plume_orbit60 | PASS | ok | R2 plume_orbit60: alpha mean abs 0.00001 (<= 0.01), p99 0.00012 (<= 0.05), max 0.0009 |
| R3_campfire_closeup | PASS | ok | R3 campfire_closeup: depth mean abs 0.00024 m (<= 0.01) over 1353387 px with Python alpha > 0.05 |
| R3_campfire_default | PASS | ok | R3 campfire_default: depth mean abs 0.00051 m (<= 0.01) over 657475 px with Python alpha > 0.05 |
| R3_campfire_orbit60 | PASS | ok | R3 campfire_orbit60: depth mean abs 0.00053 m (<= 0.01) over 629458 px with Python alpha > 0.05 |
| R3_plume_closeup | PASS | ok | R3 plume_closeup: depth mean abs 0.00025 m (<= 0.01) over 1242973 px with Python alpha > 0.05 |
| R3_plume_default | PASS | ok | R3 plume_default: depth mean abs 0.00049 m (<= 0.01) over 199315 px with Python alpha > 0.05 |
| R3_plume_orbit60 | PASS | ok | R3 plume_orbit60: depth mean abs 0.00049 m (<= 0.01) over 227708 px with Python alpha > 0.05 |
| R4_campfire_closeup_movingcam | PASS | ok | R4 campfire_closeup moving camera (3 deg/step) + volume: EPE mean 0.0132 px (<= 0.5), mean /dx/,/dy/ 0.0032, 0.0117; Python /motion/ median 13.6 px over 1353387 px |
| R4_campfire_closeup_static | PASS | ok | R4 campfire_closeup static camera, moving volume: EPE mean 0.0126 px (<= 0.5); Python /motion/ median 10.71 px, mean (x, y) 0.23, -14.87 |
| R4_campfire_default_movingcam | PASS | ok | R4 campfire_default moving camera (3 deg/step) + volume: EPE mean 0.0054 px (<= 0.5), mean /dx/,/dy/ 0.0027, 0.0034; Python /motion/ median 6.7 px over 657475 px |
| R4_campfire_default_static | PASS | ok | R4 campfire_default static camera, moving volume: EPE mean 0.0044 px (<= 0.5); Python /motion/ median 4.57 px, mean (x, y) -0.13, -2.99 |
| R4_campfire_orbit60_movingcam | PASS | ok | R4 campfire_orbit60 moving camera (3 deg/step) + volume: EPE mean 0.0051 px (<= 0.5), mean /dx/,/dy/ 0.0025, 0.0034; Python /motion/ median 6.7 px over 629458 px |
| R4_campfire_orbit60_static | PASS | ok | R4 campfire_orbit60 static camera, moving volume: EPE mean 0.0044 px (<= 0.5); Python /motion/ median 4.72 px, mean (x, y) -0.15, -2.92 |
| R4_plume_closeup_movingcam | PASS | ok | R4 plume_closeup moving camera (3 deg/step) + volume: EPE mean 0.0099 px (<= 0.5), mean /dx/,/dy/ 0.0053, 0.0063; Python /motion/ median 11.9 px over 1242973 px |
| R4_plume_closeup_static | PASS | ok | R4 plume_closeup static camera, moving volume: EPE mean 0.0077 px (<= 0.5); Python /motion/ median 6.42 px, mean (x, y) 0.75, -5.51 |
| R4_plume_default_movingcam | PASS | ok | R4 plume_default moving camera (3 deg/step) + volume: EPE mean 0.0036 px (<= 0.5), mean /dx/,/dy/ 0.0017, 0.0024; Python /motion/ median 3.6 px over 199315 px |
| R4_plume_default_static | PASS | ok | R4 plume_default static camera, moving volume: EPE mean 0.0030 px (<= 0.5); Python /motion/ median 2.45 px, mean (x, y) 0.33, -2.16 |
| R4_plume_orbit60_movingcam | PASS | ok | R4 plume_orbit60 moving camera (3 deg/step) + volume: EPE mean 0.0044 px (<= 0.5), mean /dx/,/dy/ 0.0024, 0.0027; Python /motion/ median 4.9 px over 227708 px |
| R4_plume_orbit60_static | PASS | ok | R4 plume_orbit60 static camera, moving volume: EPE mean 0.0034 px (<= 0.5); Python /motion/ median 2.52 px, mean (x, y) 0.38, -2.46 |
| R5_campfire_closeup | PASS | ok | R5 campfire_closeup: colour without AUX vs with AUX: bit-identical |
| R5_campfire_default | PASS | ok | R5 campfire_default: colour without AUX vs with AUX: bit-identical |
| R5_campfire_orbit60 | PASS | ok | R5 campfire_orbit60: colour without AUX vs with AUX: bit-identical |
| R5_plume_closeup | PASS | ok | R5 plume_closeup: colour without AUX vs with AUX: bit-identical |
| R5_plume_default | PASS | ok | R5 plume_default: colour without AUX vs with AUX: bit-identical |
| R5_plume_orbit60 | PASS | ok | R5 plume_orbit60: colour without AUX vs with AUX: bit-identical |
| R6_campfire_closeup | PASS | ok | R6 campfire_closeup: repeat draw: color identical, alpha identical, depth identical, motion identical |
| R6_campfire_default | PASS | ok | R6 campfire_default: repeat draw: color identical, alpha identical, depth identical, motion identical |
| R6_campfire_orbit60 | PASS | ok | R6 campfire_orbit60: repeat draw: color identical, alpha identical, depth identical, motion identical |
| R6_plume_closeup | PASS | ok | R6 plume_closeup: repeat draw: color identical, alpha identical, depth identical, motion identical |
| R6_plume_default | PASS | ok | R6 plume_default: repeat draw: color identical, alpha identical, depth identical, motion identical |
| R6_plume_orbit60 | PASS | ok | R6 plume_orbit60: repeat draw: color identical, alpha identical, depth identical, motion identical |
| T2_full | PASS | ok | T2 full, new volume every frame (native solver stepped between draws), aux on, campfire, 2560x1440 x0.5: native total 0.952 (p10 0.841, p95 1.022) ms, n 301 (<= 1.5 ms); prep 0.111, march 0.447, post 0.393; flags ok True [1]; Python total 1.560 (p10 1.354, p95 1.818); GPU util 21/96 %, mem 1189/1203 MiB; GPU SHARED (not the figure of record) |
| T2_view | PASS | ok | T2 view, view-only (REUSE_VOLUME), aux on, campfire, 2560x1440 x0.5: native total 0.664 (p10 0.639, p95 0.679) ms, n 301 (<= 1.5 ms); prep 0.001, march 0.268, post 0.394; flags ok True [3]; Python total 1.703 (p10 1.664, p95 1.964); GPU util 96/90 %, mem 1203/1203 MiB; GPU SHARED (not the figure of record) |

For expected-FAIL controls, `ok` means the check failed as it should (the comparator can fail).
R1 gates only the two numeric thresholds. "No structured error visible" is NOT automated: the x8 difference images are flagged for the lead's visual review; `struct_energy` (share of the error energy that is smooth at the 8x8 block scale; ~0.016 for uncorrelated dither noise, ~1 for a smooth error; also given vs the same error with its pixels shuffled) is a reported proxy, not a gate.

R1 per NVFX_RENDER_DITHER_MODE (fresh process each): mean abs / p99, worst case over the six
(scene, camera) cases; `= in-process default` = native RGB bytes identical to the gate run in every case.

| mode | worst mean abs | worst p99 | R1 pass (all cases) | = in-process default |
|---|---|---|---|---|
| unset | 0.0875 | 1 | True | True |
| 0 | 0.0875 | 1 | True | True |
| 1 | 0.0129 | 1 | True | False |
| 2 | 0.0864 | 1 | True | False |

T2, ms median / p10 / p95 (native: renderer GPU timestamps; Python: VolumeRenderer.timings, CUDA events, same session)

| case | native prep | march | post | total | n | Python prep | view | total | GPU before / after |
|---|---|---|---|---|---|---|---|---|---|
| T2_full | 0.111 / 0.111 / 0.113 | 0.447 / 0.339 / 0.517 | 0.393 / 0.392 / 0.395 | 0.952 / 0.841 / 1.022 | 301 | 0.184 / 0.182 / 0.188 | 1.373 / 1.172 / 1.482 | 1.560 / 1.354 / 1.818 | 21 % 1189 MiB / 96 % 1203 MiB |
| T2_view | 0.001 / 0.001 / 0.002 | 0.268 / 0.244 / 0.281 | 0.394 / 0.392 / 0.399 | 0.664 / 0.639 / 0.679 | 301 | 0.097 / 0.090 / 0.168 | 1.601 / 1.559 / 1.840 | 1.703 / 1.664 / 1.964 | 96 % 1203 MiB / 90 % 1203 MiB |
