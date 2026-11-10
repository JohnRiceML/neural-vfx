| control | expected | check result | control ok | measured |
|---|---|---|---|---|
| HC_P1_scale_density_1e-3 | FAIL | FAIL | True | P1 campfire steps 30..35: worst rel_l2 1.000e-03 (density, step 32, 10x threshold); solid mismatch 0 cells; FIRST FAIL step 30 density rel_l2 1.000e-03 > 1e-04 |
| HC_P1_scale_density_1e-6 | PASS | PASS | True | P1 campfire steps 30..35: worst rel_l2 1.002e-06 (density, step 32, 0.01x threshold); solid mismatch 0 cells |
| HC_P1_shift_x | FAIL | FAIL | True | P1 campfire steps 30..35: worst rel_l2 7.189e-01 (temperature, step 35, 7.19e+03x threshold); solid mismatch 0 cells; FIRST FAIL step 30 density rel_l2 6.147e-01 > 1e-04 |
| HC_P1_swap_vx_vz | FAIL | FAIL | True | P1 campfire steps 30..35: worst rel_l2 1.329e+00 (vel_x, step 30, 1.33e+04x threshold); solid mismatch 0 cells; FIRST FAIL step 30 vel_x rel_l2 1.329e+00 > 1e-04 |
| HC_P2_skip_project | FAIL | FAIL | True | P2 project: worst velocity rel_l2 7.428e-01 (random+sphere), threshold 1e-04 |
| HC_P3_scale_density_2e-2 | FAIL | FAIL | True | P3 plume @120: mass 300.0717 % (<= 1 %), heat 0.4257 % (<= 1 %), density centroid 4.06e-02 m (<= 0.01); final rel_l2 per channel 2.49e+00, 1.14e+00, 7.82e-01, 1.15e+00, 2.51e-01, 0.00e+00 |
| HC_S1_weak_projection_plume | FAIL | FAIL | True | S1 plume: max normalized div over steps 1..120 = 1.602e-02 (step 20), final 3.749e-03, threshold < 1e-04 |
| HC_S3_nondeterministic | FAIL | FAIL | True | S3 two fresh solvers, 20 steps campfire + sphere + impulses: bit-identical False (723328 differing values, max abs 5.310e-03) |
| HC_fake_P1_plume_30-35 | PASS | PASS | True | P1 plume steps 30..35: worst rel_l2 0.000e+00 (density, step 30, 0x threshold); solid mismatch 0 cells |
