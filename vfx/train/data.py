"""Random-crop sampler of 3-frame windows for training.

Scenario membership/completeness come from vfx.datagen.load_dataset (the data engineer's reader);
the raw .npy arrays are then read directly (preloaded into RAM or memory-mapped) so we can cut
random HR crops with matching LR crops without materializing full 512^2 float32 frames.

A batch holds, for each sample, frames t-2, t-1, t:
  lr  (B, 3, 5, h, w)  frames t-2, t-1, t (physical units, float32 on GPU)
  hr  (B, 3, 2, H, W)  frames t-2, t-1, t
"""
from __future__ import annotations

import queue
import threading

import numpy as np
import torch

from vfx.core.fields import CH, UPSCALE


class CropSampler:
    def __init__(self, version: str, split: str = "train", crop_hr: int = 256, batch: int = 16,
                 preload: bool = True, hflip: bool = True, seed: int = 0, device="cuda",
                 active_tries: int = 4, active_thresh: float = 0.01):
        from vfx.datagen import load_dataset
        ds = load_dataset(version, split, include_lr_sim=False)
        if not ds.scenarios:
            raise RuntimeError(f"dataset {version!r} split {split!r} is empty")
        self.scn = []
        for s in ds.scenarios:
            if "nonfinite" in s.get("flags", []):
                continue
            mm = None if preload else "r"
            hr = np.load(s["dir"] / "hr.npy", mmap_mode=mm)
            lr = np.load(s["dir"] / "lr.npy", mmap_mode=mm)
            self.scn.append((hr, lr))
        self.frames = np.array([a[0].shape[0] for a in self.scn])
        self.weights = (self.frames - 2).clip(min=0) / max(1, (self.frames - 2).clip(min=0).sum())
        self.H = self.scn[0][0].shape[-1]
        self.crop_hr = min(crop_hr, self.H)
        self.crop_lr = self.crop_hr // UPSCALE
        self.batch, self.hflip, self.device = batch, hflip, device
        self.active_tries, self.active_thresh = active_tries, active_thresh   # bias crops toward smoke
        self.rng = np.random.default_rng(seed)
        self._q: queue.Queue | None = None

    def _sample_np(self):
        B, cl, ch = self.batch, self.crop_lr, self.crop_hr
        lr_res = self.H // UPSCALE
        lrs = np.empty((B, 3, 5, cl, cl), np.float16)
        hrs = np.empty((B, 3, 2, ch, ch), np.float16)
        flips = np.zeros(B, bool)
        ks = self.rng.choice(len(self.scn), size=B, p=self.weights)
        for i, k in enumerate(ks):
            hr, lr = self.scn[k]
            t = int(self.rng.integers(2, self.frames[k]))
            for _ in range(max(1, self.active_tries)):
                y = int(self.rng.integers(0, lr_res - cl + 1))
                x = int(self.rng.integers(0, lr_res - cl + 1))
                if lr[t, 0, y:y + cl, x:x + cl].max() > self.active_thresh:
                    break
            lrs[i] = lr[t - 2:t + 1, :, y:y + cl, x:x + cl]
            Y, X = y * UPSCALE, x * UPSCALE
            hrs[i] = hr[t - 2:t + 1, :, Y:Y + ch, X:X + ch]
            flips[i] = self.hflip and self.rng.random() < 0.5
        return lrs, hrs, flips

    def _to_gpu(self, lrs, hrs, flips):
        lr = torch.from_numpy(lrs).pin_memory().to(self.device, non_blocking=True).float()
        hr = torch.from_numpy(hrs).pin_memory().to(self.device, non_blocking=True).float()
        f = torch.from_numpy(flips).to(self.device)
        if flips.any():                                   # mirror x; negate vel_x
            fv = f.view(-1, 1, 1, 1, 1)
            lr = torch.where(fv, lr.flip(-1), lr)
            hr = torch.where(fv, hr.flip(-1), hr)
            sign = torch.ones(lr.shape[0], 1, 5, 1, 1, device=lr.device)
            sign[:, :, CH["vel_x"]] = -1.0
            lr = torch.where(fv, lr * sign, lr)
        return lr, hr

    def _worker(self):
        while True:
            self._q.put(self._sample_np())

    def next(self):
        if self._q is None:
            self._q = queue.Queue(maxsize=6)
            for _ in range(2):
                threading.Thread(target=self._worker, daemon=True).start()
        return self._to_gpu(*self._q.get())

    def fixed_batch(self):
        """One deterministic batch (for the overfit sanity check)."""
        return self._to_gpu(*self._sample_np())

    def velocity_std(self, n: int = 64) -> float:
        vs = []
        for _ in range(max(1, n // self.batch)):
            lrs, _, _ = self._sample_np()
            vs.append(lrs[:, :, CH["vel_x"]:CH["vel_y"] + 1].astype(np.float32).ravel())
        return float(np.concatenate(vs).std())
