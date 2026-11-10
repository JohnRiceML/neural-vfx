"""load_sequences(version, split, length): windows of consecutive frames of one camera sequence.

Item (all torch tensors):
  "lr"  (L,3,h,w) uint8    render of the raw 64^3 sim, sandbox settings, cropped + 2x2 pooled
  "hr"  (L,3,h,w) uint8    render of the guided 256^3 sim, same camera / light / renderer settings
  "aux" (L,4,h,w) float16  alpha, depth (m), motion_x, motion_y (STORED px per SIM_DT; x right, y down)
                           from the LR render
  "hr_alpha" (L,h,w) uint8 medium alpha of the HR render x 255 (masks / eval)
  "seed" int, "cam" int, "t0" int (frame index of the window start within the 16-frame sequence),
  "moving" bool (camera moves in this sequence)
Arrays are np.load(mmap_mode="r"), opened lazily per worker (DataLoader num_workers > 0 on Windows).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from vfx.core import paths

from .runner import is_complete, scenario_dir


class SequenceDataset3D(Dataset):
    def __init__(self, root, split: str = "train", length: int = 8, stride: int = 1):
        self.root = Path(root)
        self.split, self.length = split, int(length)
        splits = json.loads((self.root / "splits.json").read_text())
        if split not in splits:
            raise KeyError(f"split {split!r} not in {list(splits)}")
        self.scenarios: list[dict] = []
        self.index: list[tuple[int, int, int]] = []        # (scenario idx, cam, t0)
        for seed in splits[split]:
            d = scenario_dir(self.root, split, seed)
            if not is_complete(d):
                continue
            meta = json.loads((d / "meta.json").read_text())
            T, K = int(meta["frames"]), int(meta["n_cams"])
            if self.length > T:
                raise ValueError(f"length {length} > frames per sequence {T}")
            k = len(self.scenarios)
            self.scenarios.append({"dir": d, "seed": int(seed), "tags": meta["tags"],
                                   "moving": [bool(c["moving"]) for c in meta["cams"]]})
            self.index += [(k, c, t0) for c in range(K) for t0 in range(0, T - self.length + 1, stride)]
        self._mm: dict = {}

    def __len__(self):
        return len(self.index)

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_mm"] = {}
        return s

    def _arrays(self, k):
        a = self._mm.get(k)
        if a is None:
            d = self.scenarios[k]["dir"]
            a = self._mm[k] = {n: np.load(d / f"{n}.npy", mmap_mode="r") for n in ("lr", "hr", "aux", "hr_alpha")}
        return a

    def __getitem__(self, i):
        k, c, t0 = self.index[i]
        a = self._arrays(k)
        sl = slice(t0, t0 + self.length)
        return {"lr": torch.from_numpy(np.ascontiguousarray(a["lr"][c, sl])),
                "hr": torch.from_numpy(np.ascontiguousarray(a["hr"][c, sl])),
                "aux": torch.from_numpy(np.ascontiguousarray(a["aux"][c, sl])),
                "hr_alpha": torch.from_numpy(np.ascontiguousarray(a["hr_alpha"][c, sl])),
                "seed": self.scenarios[k]["seed"], "cam": c, "t0": t0,
                "moving": self.scenarios[k]["moving"][c]}

    def meta(self, i) -> dict:
        """meta.json of item i's scenario (cameras per frame, crop, tracking, ...)."""
        k = self.index[i][0]
        return json.loads((self.scenarios[k]["dir"] / "meta.json").read_text())


def load_sequences(version: str, split: str = "train", length: int = 8, stride: int = 1) -> SequenceDataset3D:
    """Windows of `length` consecutive sim steps of one camera (all start offsets with `stride`)."""
    return SequenceDataset3D(paths.dataset_dir(version), split, length, stride)
