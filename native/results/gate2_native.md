Gate 2 results (native detail inference), 2026-11-09 21:46:47, nvfx 0.1 sim3d-d3d12-0.1.0, flags 0

Model `C:\Users\ricej\AppData\Local\neural-vfx\native_ref\onnx\screennet_fp16.onnx`, checkpoint `C:\Users\ricej\AppData\Local\neural-vfx\checkpoints\screen3d\latest.pt` (sha256 2e86bf290ae7).

| check | expected | result | measured |
|---|---|---|---|
| D1_NC_alpha0_campfire | FAIL | ok | negative control alpha0 (campfire): D1 FAILS as expected; pixel part fails, region part passes; vs Python: FAIL max 42, >1 500992 px (32.6167 %), region = 1600x960@(554,154) crop 1600x960, outside diff 0 px, A kept, net signal max 42 |
| D1_NC_shift64_campfire | FAIL | ok | negative control shift64 (campfire): D1 FAILS as expected; pixel part fails, region part fails; vs Python: FAIL max 6, >1 8541 px (0.5561 %), region != 1600x960@(554,154) crop 1600x960, outside diff 4983 px, A kept, net signal max 42 |
| D1_campfire | PASS | ok | D1 campfire: 12/12 bboxes pass; 9 buckets (1600x960 yes, full 2560x1472 yes); worst: effect_alpha_bbox: ok max 2, >1 3 px (0.0001 %), region = 1920x1440@(394,0) crop 1920x1472, outside diff 0 px, A kept, net signal max 41 |
| D1_plume | PASS | ok | D1 plume: 12/12 bboxes pass; 9 buckets (1600x960 yes, full 2560x1472 yes); worst: effect_alpha_bbox: ok max 1, >1 0 px (0.0000 %), region = 960x1440@(801,0) crop 960x1472, outside diff 0 px, A kept, net signal max 29 |
| D1_region_sweep | PASS | ok | D1 region sweep: 3203/3203 bboxes give Python's region and crop (nvfx_detail_region vs ScreenDetail._region) |
| D2_campfire | PASS | ok | D2 campfire: one_third_1536x832: run2 0 px differ, run3 (after another bbox) 0; full_frame_none: run2 0 px differ, run3 (after another bbox) 0 |
| D2_plume | PASS | ok | D2 plume: one_third_1536x832: run2 0 px differ, run3 (after another bbox) 0; full_frame_none: run2 0 px differ, run3 (after another bbox) 0 |
| D3 | PASS | **NOT OK** | D3: max CPU wall 12.468 ms (call #164), median 0.489 ms, 6 calls > 10.0 ms; sample D1 10/10 pass (worst max abs 2, worst >1 frac 0.0001 %); create 1.11 s, vram 181 MiB |
| D4 | PASS | ok | D4: nvfx_detail_run returned in max 0.616 ms (median 0.175) over 10 calls; 10/10 trials with >= 20 ms queued ahead (min 33.6, median 96.0 ms; target 45 ms; 10-14 solver steps, calibrated ~10.08 ms/step -> 10; 0 re-aimed call(s)) |
| HC2_py_alpha0_campfire | FAIL | ok | HC2 python-vs-python alpha0 (campfire): D1 FAIL (expected FAIL); : FAIL max 42, >1 500992 px (32.6167 %), region = 1600x960@(554,154) crop 1600x960, outside diff 0 px, A kept, net signal max 42 |
| HC2_py_self_campfire | PASS | ok | HC2 python-vs-python self (campfire): D1 PASS (expected PASS); : ok max 0, >1 0 px (0.0000 %), region = 1600x960@(554,154) crop 1600x960, outside diff 0 px, A kept, net signal max 42 |
| HC2_py_shift64_campfire | FAIL | ok | HC2 python-vs-python shift64 (campfire): D1 FAIL (expected FAIL); : FAIL max 6, >1 8563 px (0.5575 %), region != 1600x960@(554,154) crop 1600x960, outside diff 4985 px, A kept, net signal max 42 |

For expected-FAIL controls, `ok` means the check failed as it should (the comparator can fail).
