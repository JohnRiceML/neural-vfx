# vfx/train — training (owner: ML/runtime engineer)

```
.venv/Scripts/python -m vfx.train.train --config vfx/train/configs/demo.json              # real data (v1), ~15 min
.venv/Scripts/python -m vfx.train.train --config vfx/train/configs/synthetic.json --overfit 600   # sanity
.venv/Scripts/python -m vfx.train.evaluate --version v1 --split test [--lr-key lr_sim] [--ckpt PATH]
.venv/Scripts/python -m vfx.train.synthetic --version synthetic       # procedural stand-in dataset
```
Overrides: `--minutes`, `--dataset`.

- Data: `data.CropSampler` takes split membership from `vfx.datagen.load_dataset`, then reads the
  `.npy` arrays (RAM-preloaded or mmap) to cut random 256^2 HR crops + matching 64^2 LR crops of
  frames t-2..t (h-flip augmentation negates vel_x). Background threads prefetch.
- Exposure bias: per sample the previous-frame input is teacher (GT + relative noise), rollout
  (model's own prediction of t-1, no grad) or cold start (bicubic, as after `Upscaler.reset()`).
- Loss (normalized units): L1 + gradient + high-frequency band + radial log power spectrum +
  temporal-change consistency (on rollout samples).
- FP16 autocast + GradScaler, AdamW, time-based cosine LR. Logs: JSONL (+TensorBoard if installed)
  in `paths.RUNS/<name>_<timestamp>/`; checkpoint there and copied to `paths.default_checkpoint()`.
- End of run: autoregressive full-frame eval with the runtime `Upscaler` vs bicubic, and an entry
  appended to `EXPERIMENTS.md`.

## v2 sequence training (`train_seq.py`)
```
.venv/Scripts/python -m vfx.train.train_seq --config vfx/train/configs/v2_seq.json
.venv/Scripts/python -m vfx.train.evaluate --scorecard [--ckpt PATH]   # v2 test/ood lr_sim + live flicker
```
- Input = `lr_sim` (the 128² sim, the runtime distribution), target = guided v2 `hr`. Windows of K+1
  consecutive frames (256² HR crops, h-flip with vel_x negated), unrolled like the runtime Upscaler:
  cold start (bicubic prev) or noisy-teacher start, then the model's own output warped with
  `warp_prev` and the lr_sim velocity of the previous frame; gradient through the whole unroll.
- Loss per frame: run-3 terms (L1, grad, HF, low-band log spectrum) on the crop minus a 16 px margin
  (the warp clamps at the crop border), + temporal change matching, + flicker hinge
  `relu(F(pred) - alpha*F(bicubic))` with F = the gate's warped-frame flicker of density.
- `train.p_pooled`: fraction of samples fed the pooled HR (`lr.npy`) instead of lr_sim; without it the
  model over-sharpens pooled input and fails the Layer 4 pooled-L1 gate (run 2 -> run 3).
- Current default = `configs/v2_seq_mix.json` (run 3), fine-tuned from run 2 (`v2_seq_k8.json`), which
  was fine-tuned from the v1 run-3 checkpoint. Backups in `checkpoints/demo/` (`v1_run3_backup.pt`,
  `v2_seq_run1/2/3.pt`).
- `evaluate.py` now calls `vfx.eval.metrics.evaluate_upscaler` (the gate's single flicker definition;
  the old local flicker warped the HR reference with the lr_sim velocity). Runs never auto-promote;
  copy to `paths.default_checkpoint()` by hand after comparing scorecards.
