Gate 1 results, backend `native`, 2026-11-09 23:16:10, nvfx 0.1 sim3d-d3d12-0.1.0

| check | result | measured |
|---|---|---|
| P1_campfire | PASS | P1 campfire steps 0..119: worst rel_l2 3.315e-05 (temperature, step 50, 0.331x threshold); solid mismatch 0 cells |
| P1_plume | PASS | P1 plume steps 0..119: worst rel_l2 3.096e-05 (temperature, step 28, 0.31x threshold); solid mismatch 0 cells |
| P2_project | PASS | P2 project: worst velocity rel_l2 1.665e-07 (random+sphere), threshold 1e-04 |
| P2_zero_sources_campfire | PASS | P2 zero-sources step campfire@60: velocity rel_l2 2.350e-07, 1.516e-07, 2.384e-07 (x,y,z), threshold 1e-04 |
| P2_zero_sources_plume | PASS | P2 zero-sources step plume@60: velocity rel_l2 3.290e-07, 3.683e-07, 4.277e-07 (x,y,z), threshold 1e-04 |
| P3_campfire | PASS | P3 campfire @120: mass 0.1528 % (<= 1 %), heat 0.4206 % (<= 1 %), density centroid 4.61e-03 m (<= 0.01); final rel_l2 per channel 6.29e-01, 1.00e+00, 7.86e-01, 1.07e+00, 3.31e-01, 3.09e-01 |
| P3_plume | **FAIL** | P3 plume @120: mass 0.0001 % (<= 1 %), heat 2.3806 % (<= 1 %), density centroid 5.85e-03 m (<= 0.01); final rel_l2 per channel 6.75e-01, 1.01e+00, 7.09e-01, 1.05e+00, 2.40e-01, 0.00e+00 |
| S1_campfire | PASS | S1 campfire: max normalized div over steps 1..120 = 2.208e-05 (step 1), final 4.756e-06, threshold < 1e-04 |
| S1_plume | PASS | S1 plume: max normalized div over steps 1..120 = 1.988e-05 (step 31), final 1.219e-05, threshold < 1e-04 |
| S2_mass_box | PASS | S2 mass, closed box + sphere, 1000 steps: +0.00096 % (/./ < 0.1 %); Python selfcheck.mass_drift same session +0.00016 % |
| S3_determinism | PASS | S3 two fresh solvers, 200 steps campfire + sphere + impulses: bit-identical True (0 differing values, max abs 0.000e+00) |
| S3b_reset_reuse | PASS | S3b same solver, reset between two 200-step runs: bit-identical True (0 differing values) |
| S4_stress_full | PASS | S4 stress full 1000 steps: finite True, max/u/ 3.866 m/s, Tmax 2637 K, div 1.42e-05, solid clean 0.0e+00 / Python same session: max/u/ 3.742, Tmax 2664 K, div 1.01e-05 |
| T1_campfire_default | PASS | T1 campfire default: backend total median 1.000 ms (p95 1.014, n 201) <= 2.0 ms; Python step_profiled total median 0.882 ms, graph step 0.879 ms; GPU util 6/1 %, mem 391/774 MiB (before/after) |
| T1_campfire_runtime | PASS | T1 campfire runtime: backend total median 0.560 ms (p95 0.568, n 201) <= 2.0 ms; Python step_profiled total median 0.759 ms, graph step 0.755 ms; GPU util 30/99 %, mem 774/774 MiB (before/after) |
| T1_plume_default | PASS | T1 plume default: backend total median 1.077 ms (p95 1.089, n 201) <= 2.0 ms; Python step_profiled total median 1.008 ms, graph step 1.005 ms; GPU util 99/91 %, mem 774/778 MiB (before/after) |
| T1_plume_runtime | PASS | T1 plume runtime: backend total median 0.554 ms (p95 0.561, n 201) <= 2.0 ms; Python step_profiled total median 0.760 ms, graph step 0.756 ms; GPU util 91/3 %, mem 778/780 MiB (before/after) |

T1, ms median (p95). Backend: GPU timestamps; Python: step_profiled (3 graphs + CUDA events) and the single-graph step, same session.

| scene, config | backend adv | forces | proj | total | n | PCG it | Python adv | forces | proj | total | graph step | GPU util / mem before, after |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| campfire_default | 0.038 (0.040) | 0.057 (0.058) | 0.906 (0.917) | 1.000 (1.014) | 201 | 5 | 0.049 (0.049) | 0.070 (0.070) | 0.764 (0.766) | 0.882 (0.885) | 0.879 (0.890) | 6 % 391 MiB, 1 % 774 MiB |
| campfire_runtime | 0.035 (0.038) | 0.056 (0.056) | 0.469 (0.474) | 0.560 (0.568) | 201 | 4 | 0.049 (0.050) | 0.070 (0.071) | 0.641 (0.643) | 0.759 (0.762) | 0.755 (0.764) | 30 % 774 MiB, 99 % 774 MiB |
| plume_default | 0.036 (0.038) | 0.055 (0.056) | 0.986 (0.997) | 1.077 (1.089) | 201 | 6 | 0.049 (0.049) | 0.068 (0.070) | 0.893 (1.017) | 1.008 (1.133) | 1.005 (1.127) | 99 % 774 MiB, 91 % 778 MiB |
| plume_runtime | 0.036 (0.038) | 0.053 (0.054) | 0.465 (0.471) | 0.554 (0.561) | 201 | 4 | 0.049 (0.049) | 0.068 (0.070) | 0.643 (0.841) | 0.760 (0.955) | 0.756 (0.766) | 91 % 778 MiB, 3 % 780 MiB |
