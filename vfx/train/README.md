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
