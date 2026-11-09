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

## demo_20261109_112002  (2026-11-09 11:35)
- config: `vfx/train/configs/demo.json`  dataset: `v1`  commit: `88c7eac79dd6`
- hypothesis: M3 baseline on v1: DetailNet 64x8 (residual on bicubic, warped prev via t-1 velocity), L1+grad+HF+0.002*log-spectrum+temporal, crops biased to smoke, spectral term only on non-flat crops; beats bicubic on L1 and spectrum error (test lr and lr_sim).
- result: {"steps": 52051, "minutes": 15.0, "params": 688114, "final_loss": 0.0025550893042236567, "val_l1": 0.00058, "val_bic_l1": 0.00147, "val_spec": 0.05748, "val_bic_spec": 1.22408, "rollout_eval": {"net_l1_d": 0.00056, "bic_l1_d": 0.00099, "net_l1_T": 0.35522, "bic_l1_T": 0.60082, "net_spec": 0.09554, "bic_spec": 1.44261, "net_flicker": 0.00049, "bic_flicker": 0.0006, "ref_flicker": 0.00056, "frac": 0.78809, "spec_improvement": 0.93377, "l1_improvement": 0.43541, "flicker_ratio_vs_ref": 0.88957, "n_scenarios": 8, "metrics_source": "vfx.eval.metrics", "lr_key": "lr"}, "promoted_to_default": true}
- decision: KEEP as fallback. Full eval (32 test / 32 ood, rollout via Upscaler; vfx.train.evaluate) vs bicubic: test/lr L1 -43%, spec -95%; ood/lr L1 -35%, spec -90%; test/lr_sim L1 +1.4% (worse), spec -86%; ood/lr_sim L1 +1.1%, spec -82%. Layer 4: 6 passed, 4 xpassed, 1 xfailed (v1 flicker on lr_sim: 2.72 vs bicubic 2.20). Headless screenshots: more detail, no brick seams, but isolated speckle pixels (~4.6x the HR reference's count; 1.5x worse with self-feedback over 400 steps). Next: LR degradation (run 2).

## demo_degrade_20261109_113741  (2026-11-09 11:52)
- config: `vfx/train/configs/demo_degrade.json`  dataset: `v1`  commit: `88c7eac79dd6`
- hypothesis: Run 1 is slightly worse than bicubic on lr_sim (L1 -1%, flicker ratio 2.72 vs 2.20). Degrading the LR input in training (p=0.5: Gaussian blur sigma<=1.5 LR cells, 5% relative noise, vel noise) makes the model less literal about LR structure and improves lr_sim L1/flicker without losing much on pooled lr.
- result: {"steps": 52998, "minutes": 15.0, "params": 688114, "final_loss": 0.0033345981501042843, "val_l1": 0.00058, "val_bic_l1": 0.00139, "val_spec": 0.0604, "val_bic_spec": 1.17667, "rollout_eval": {"net_l1_d": 0.00064, "bic_l1_d": 0.00099, "net_l1_T": 0.40366, "bic_l1_T": 0.60082, "net_spec": 0.11008, "bic_spec": 1.44261, "net_flicker": 0.00045, "bic_flicker": 0.0006, "ref_flicker": 0.00056, "frac": 0.78809, "spec_improvement": 0.92369, "l1_improvement": 0.35653, "flicker_ratio_vs_ref": 0.80279, "n_scenarios": 8, "metrics_source": "vfx.eval.metrics", "lr_key": "lr"}, "promoted_to_default": false}
- decision: PARTIAL. vs run 1: lr_sim spec 0.227->0.182 (-89% vs bicubic), own-metric lr_sim flicker 0.324->0.261 (bicubic 0.277), but pooled L1 gain 43%->34%, ood/lr 35%->17%; lr_sim L1 still ~2% worse than bicubic (independent-sim divergence dominates L1). Speckles unchanged (108 vs ref 24). Keep degradation (the app feeds an independent LR sim); attack speckle via band-limited spectral term (run 3).

## demo_20261109_115337  (2026-11-09 12:08)
- config: `vfx/train/configs/demo.json`  dataset: `v1`  commit: `88c7eac79dd6`
- hypothesis: Run 3: run-2 recipe (LR degradation p=0.5) + spectral term restricted to the lower half of radial bins. Neural output has ~4.5x the reference's isolated speckle pixels; matching pixel-scale power likely rewards speckle. Expect fewer speckles with similar spectrum/L1 gains.
- result: {"steps": 52480, "minutes": 15.0, "params": 688114, "final_loss": 0.0029585398733615875, "val_l1": 0.00059, "val_bic_l1": 0.00147, "val_spec": 0.2364, "val_bic_spec": 1.22408, "rollout_eval": {"net_l1_d": 0.00059, "bic_l1_d": 0.00099, "net_l1_T": 0.36601, "bic_l1_T": 0.60082, "net_spec": 0.35914, "bic_spec": 1.44261, "net_flicker": 0.00044, "bic_flicker": 0.0006, "ref_flicker": 0.00056, "frac": 0.78809, "spec_improvement": 0.75105, "l1_improvement": 0.40386, "flicker_ratio_vs_ref": 0.78294, "n_scenarios": 8, "metrics_source": "vfx.eval.metrics", "lr_key": "lr"}, "promoted_to_default": true}
- decision: ADOPTED (default checkpoint; demo.json = this recipe). Speckles gone (0-1 px vs 108; ref 24). Layer 4 report vs bicubic: spectrum test/lr 0.342 vs 1.404, test/lr_sim 0.360 vs 1.657, ood/lr 0.338 vs 1.090, ood/lr_sim 0.413 vs 1.359; L1 density test/lr 0.00066 vs 0.00109, ood/lr 0.00201 vs 0.00267, lr_sim within 2% (worse); flicker ratio lr 0.80 (bic 1.20), lr_sim 2.78 (bic 2.20, v1 gate xfail). Layer 4: 6 passed, 4 xpassed, 1 xfailed. Spectrum gain is lower than run 1 (-76% vs -95%) but visually clean. Open: lr_sim flicker; longer self-feedback horizon needs sequences > 8 frames.
