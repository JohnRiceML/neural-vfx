# vfx/train3d — route B (ScreenDetail) training (owner: 3D ML/runtime engineer)

```
.venv/Scripts/python -m vfx.train3d.train --config vfx/train3d/configs/<name>.json [--minutes M]
.venv/Scripts/python -m vfx.train3d.promote <run name | run dir>     # -> CHECKPOINTS/screen3d/latest.pt
```

| File | Content |
|---|---|
| `data.py` | `GPUWindows`: whole split of `vfx.datagen3d.load_sequences` resident on the GPU (train s3d_v1 ≈ 2.9 GiB), random K-frame windows, crops biased to the effect, h-flip (motion_x negated), effect mask = dilated union of lr/hr alpha |
| `losses.py` | weighted L1, gradient L1, log radial spectrum of luminance on the eval metric's 32 linear bins (restricted to the lowest `spec_max_frac`), unwarped temporal change matching |
| `train.py` | config-driven loop (AdamW, cosine LR, fp16 autocast + GradScaler, grad clip, EMA weights), the rollout that mirrors `ScreenDetail` at process_scale 1.0, final evaluation with `vfx.eval3d.metrics.evaluate_screen_detail` on test + ood through the real runtime, EXPERIMENTS.md entry |
| `adversarial.py` | conditional PatchGAN discriminators (spatial: [y, lr, y-lr, alpha]; temporal: 3 unwarped frames of y and lr + their differences), spectral norm, hinge, masked per patch, feature matching, R1, translation aug. Training only |
| `holdout.py` | model selection on held-out TRAIN scenarios, whole sequences, vfx.eval3d metrics + own diagnostics (mid-band 0.04-0.2 cyc/px power vs hr, luminance and colour speckle counts, unwarped jerk ratio); the pre-registered rule (`select_ok`, `select_key`) |
| `promote.py` | copies a run to `latest.pt` only if it beats lr AND the tuned unsharp on test l1 / spectrum / ms_dssim |
| `configs/` | one JSON per run (merged over `train.DEFAULTS`) |
| `EXPERIMENTS.md` | hypothesis / result / decision per run |

Config sections beyond the regression recipe: `adv` (weight / temporal / fm, D width, d_lr, r1, aug_shift, d_warmup;
all 0 = no discriminator), `select` (`rule: "constrained_spec"` = the pre-registered rule in EXPERIMENTS.md on
`holdout_sequences`), `init` + `init_override` (load a checkpoint into a config with extra zero-initialised paths or
output constraints, e.g. `{"chroma_proj": true, "mid_width": 32}`; `vfx.model3d.screennet.load_screen_net_extended`).

Outputs: `DATA_ROOT/runs/screen3d/<name>_<time>/` (config.json, log.jsonl, curve.json = fixed-batch train/test
curve, summary.json, model.pt), a copy of every run's checkpoint in `DATA_ROOT/checkpoints/screen3d/<run>.pt`.

Training matches the runtime exactly: the dataset frames are already the processed resolution (1440p window,
2×2-pooled), so the net sees `rgb` 0..1, aux = alpha, depth (m), motion (stored px / step) and adds Δrgb; the
runtime (`vfx/runtime3d/screen_detail.py`) pools the renderer's full-res aux the same way (alpha mean,
alpha-weighted depth/motion, motion × process_scale) before calling the net. Aux normalization lives in the model.
