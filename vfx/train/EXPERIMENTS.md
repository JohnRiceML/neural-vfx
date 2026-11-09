# Experiment log (vfx/train)

One entry per training run: hypothesis, result, decision. `train.py` appends the hypothesis
(from the config) and measured results automatically; the decision line is filled in by hand.
Entries 0a/0b are the overfit sanity checks (run with `--overfit N`, which does not append).

## 0a overfit-600 / 0b overfit-4000 (synthetic, 2026-11-09)
- hypothesis: pipeline can memorize one fixed batch (16 crops of 256^2).
- result: L1 (normalized) 1.87e-3 -> 7.9e-4 after 600 steps; -> 2.09e-4 after 4000 steps (9x), spectral term 0.162 -> 4.1e-4, still decreasing. ~14 ms/step.
- decision: pipeline OK; proceed to real training runs.

## synthetic_20261109_105758  (2026-11-09 11:03)
- config: `vfx/train/configs/synthetic.json`  dataset: `synthetic`  commit: `ad8d7f0fc949`
- hypothesis: Pipeline validation on procedural synthetic data: short run should beat bicubic on L1 and spectrum error on held-out synthetic scenes.
- result: {"steps": 13385, "minutes": 5.0, "params": 688114, "final_loss": NaN, "val_l1": NaN, "val_bic_l1": 0.00222, "val_spec": NaN, "val_bic_spec": 0.18642, "rollout_eval": {"net_l1_d": NaN, "bic_l1_d": 0.0031, "net_l1_T": NaN, "bic_l1_T": 0.2058, "net_spec": NaN, "bic_spec": 0.49476, "net_flicker": NaN, "bic_flicker": 0.00634, "ref_flicker": 0.00535, "frac": 0.99023, "spec_improvement": NaN, "l1_improvement": NaN, "flicker_ratio_vs_ref": NaN, "n_scenarios": 4, "metrics_source": "vfx.eval.metrics", "lr_key": "lr"}}
- decision: FAILED. Loss went NaN at step ~2300 (val beat bicubic before that: L1 9.7e-4 vs 2.3e-3, spec 0.029 vs 0.19). Weights stayed finite but the un-normalized residual trunk grew activations ~4x per block (73 -> 73k), overflowing fp16. Fix: EDSR residual scaling 0.1, clip normalized inputs (+-20) and outputs ([0,50]), skip non-finite steps, never promote a checkpoint whose eval is non-finite. Rerun.

## synthetic_20261109_110357  (2026-11-09 11:09)
- config: `vfx/train/configs/synthetic.json`  dataset: `synthetic`  commit: `ad8d7f0fc949`
- hypothesis: Pipeline validation on procedural synthetic data: short run should beat bicubic on L1 and spectrum error on held-out synthetic scenes.
- result: {"steps": 15517, "minutes": 5.0, "params": 688114, "final_loss": 0.0016723234439268708, "val_l1": 0.00053, "val_bic_l1": 0.00229, "val_spec": 0.01026, "val_bic_spec": 0.2029, "rollout_eval": {"net_l1_d": 0.00075, "bic_l1_d": 0.0031, "net_l1_T": 0.0932, "bic_l1_T": 0.2058, "net_spec": 0.0649, "bic_spec": 0.49476, "net_flicker": 0.0055, "bic_flicker": 0.00634, "ref_flicker": 0.00535, "frac": 0.99023, "spec_improvement": 0.86882, "l1_improvement": 0.758, "flicker_ratio_vs_ref": 1.02838, "n_scenarios": 4, "metrics_source": "vfx.eval.metrics", "lr_key": "lr"}, "promoted_to_default": true}
- decision: PASS (synthetic only). Stable after the fp16 fix; autoregressive full-frame rollout with the runtime Upscaler beats bicubic: density L1 -76%, spectrum error -87%, flicker 1.03x reference. Synthetic data is procedural, so this validates the pipeline, not quality. Next: same recipe on the real dataset (demo.json, 15 min) and also evaluate with --lr-key lr_sim.
