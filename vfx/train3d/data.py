"""GPU-resident window sampler over vfx.datagen3d sequences (s3d_v1: 3.8 GB on disk, ~2.9 GB for train on GPU).

Each sample: K consecutive frames of one camera sequence, random crop (biased to the effect), optional
h-flip (image mirrored, motion_x negated). Returned as float tensors on the GPU:
  lr, hr (B,K,3,c,c) in 0..1, aux (B,K,4,c,c) raw units (alpha, depth m, motion px/step), mask (B,K,c,c)
  = dilated union of lr and hr alpha (the region the eval metrics score).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GPUWindows:
    def __init__(self, version: str, split: str, device="cuda", max_sequences: int | None = None,
                 seq_range: tuple | None = None):
        """seq_range (a, b): only sequences a..b-1 (2 cameras per scenario, scenario-major: a hold-out of whole
        scenarios is a contiguous even range)."""
        from vfx.datagen3d import load_sequences
        ds = load_sequences(version, split, 16, stride=16)
        a, b = seq_range if seq_range else (0, len(ds))
        b = min(b if b >= 0 else len(ds) + b, len(ds))
        a = a if a >= 0 else len(ds) + a
        if max_sequences is not None:
            b = min(b, a + max_sequences)
        lr, hr, aux, ha, moving = [], [], [], [], []
        for i in range(a, b):
            it = ds[i]
            lr.append(it["lr"].to(device))
            hr.append(it["hr"].to(device))
            aux.append(it["aux"].to(device))
            ha.append(it["hr_alpha"].to(device))
            moving.append(bool(it["moving"]))
        self.lr, self.hr = torch.stack(lr), torch.stack(hr)          # (N,T,3,H,W) uint8
        self.aux, self.hr_alpha = torch.stack(aux), torch.stack(ha)  # (N,T,4,H,W) fp16, (N,T,H,W) uint8
        self.moving = torch.tensor(moving)
        self.N, self.T = self.lr.shape[:2]
        self.H, self.W = self.lr.shape[-2:]
        self.device = device

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.lr, self.hr, self.aux, self.hr_alpha))

    def sample(self, B: int, K: int, crop: int, gen: np.random.Generator, hflip: bool = True,
               min_alpha: float = 0.02, tries: int = 4):
        c = min(crop, self.H, self.W)
        idx = []
        for _ in range(B):
            n = int(gen.integers(0, self.N))
            t0 = int(gen.integers(0, self.T - K + 1))
            for _ in range(tries):     # prefer crops that contain some effect
                y = int(gen.integers(0, self.H - c + 1))
                x = int(gen.integers(0, self.W - c + 1))
                if float(self.aux[n, t0 + K - 1, 0, y:y + c, x:x + c].float().mean()) > min_alpha:
                    break
            idx.append((n, t0, y, x, hflip and gen.random() < 0.5))
        return self._gather(idx, K, c)

    def _gather(self, idx, K, c):
        lr, hr, aux, ha = [], [], [], []
        for n, t0, y, x, fl in idx:
            s = (n, slice(t0, t0 + K), slice(None), slice(y, y + c), slice(x, x + c))
            a = self.aux[s].float()
            l, h = self.lr[s], self.hr[s]
            al = self.hr_alpha[n, t0:t0 + K, y:y + c, x:x + c]
            if fl:
                l, h, al = l.flip(-1), h.flip(-1), al.flip(-1)
                a = a.flip(-1)
                a[:, 2] = -a[:, 2]
            lr.append(l), hr.append(h), aux.append(a), ha.append(al)
        lr = torch.stack(lr).float() * (1 / 255.0)
        hr = torch.stack(hr).float() * (1 / 255.0)
        aux = torch.stack(aux)
        ha = torch.stack(ha).float() * (1 / 255.0)
        m = torch.maximum(aux[:, :, 0].clamp(0, 1), ha) > 0.02
        B = m.shape[0]
        m = F.max_pool2d(m.float().reshape(B * K, 1, c, c), 5, 1, 2).reshape(B, K, c, c)
        return lr, hr, aux, m

    def fixed_batch(self, B: int, K: int, crop: int, seed: int = 123):
        """Deterministic validation batch (no flips)."""
        return self.sample(B, K, crop, np.random.default_rng(seed), hflip=False)
