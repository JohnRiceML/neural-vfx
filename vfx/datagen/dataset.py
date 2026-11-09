"""PairedDataset: memory-mapped reader for DATA_ROOT/datasets/<version>/<split>/scn_*/.

Item = one frame pair (t >= 1) of one scenario:
  lr (5,128,128), hr_prev (2,512,512), hr (2,512,512), solid_lr (128,128), solid_hr (512,512)
  as float32 tensors, plus seed:int and t:int. Items carry lr_sim (5,128,128) whenever the scenario
  has one: test/ood in v1, every split in v2 (guided HR; see guide.py).
SequenceDataset = windows of L consecutive frames (multi-step self-feedback rollouts).
Arrays are opened with np.load(mmap_mode='r') lazily per process, so the dataset is cheap to
pickle into DataLoader workers (Windows spawn).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from vfx.core import paths

from .runner import is_complete, scenario_dir


class PairedDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train", include_lr_sim: bool = True):
        self.root = Path(root)
        self.split = split
        splits = json.loads((self.root / "splits.json").read_text())
        if split not in splits:
            raise KeyError(f"split {split!r} not in {list(splits)}")
        self.scenarios: list[dict] = []          # {dir, seed, frames, has_lr_sim}
        self.index: list[tuple[int, int]] = []   # (scenario idx, t)
        for seed in splits[split]:
            d = scenario_dir(self.root, split, seed)
            if not (d / "meta.json").exists():
                continue                          # incomplete build: skip silently
            meta = json.loads((d / "meta.json").read_text())
            if not is_complete(d, meta["has_lr_sim"]):
                continue
            k = len(self.scenarios)
            self.scenarios.append({"dir": d, "seed": int(seed), "frames": int(meta["frames"]),
                                   "has_lr_sim": bool(meta["has_lr_sim"]) and include_lr_sim,
                                   "tags": meta["tags"], "flags": meta.get("flags", [])})
            self.index += [(k, t) for t in range(1, int(meta["frames"]))]
        self._mm: dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.index)

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_mm"] = {}
        return s

    def _arrays(self, k: int) -> dict:
        a = self._mm.get(k)
        if a is None:
            d = self.scenarios[k]["dir"]
            a = {n: np.load(d / f"{n}.npy", mmap_mode="r") for n in ("hr", "lr")}
            a["solid_hr"] = torch.from_numpy(np.load(d / "solid_hr.npy")).float()
            a["solid_lr"] = torch.from_numpy(np.load(d / "solid_lr.npy")).float()
            if self.scenarios[k]["has_lr_sim"]:
                a["lr_sim"] = np.load(d / "lr_sim.npy", mmap_mode="r")
            self._mm[k] = a
        return a

    @staticmethod
    def _f32(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.array(x, dtype=np.float32))

    def __getitem__(self, i: int) -> dict:
        k, t = self.index[i]
        a = self._arrays(k)
        item = {
            "lr": self._f32(a["lr"][t]),
            "hr_prev": self._f32(a["hr"][t - 1]),
            "hr": self._f32(a["hr"][t]),
            "solid_lr": a["solid_lr"],
            "solid_hr": a["solid_hr"],
            "seed": self.scenarios[k]["seed"],
            "t": t,
        }
        if "lr_sim" in a:
            item["lr_sim"] = self._f32(a["lr_sim"][t])
        return item


def load_dataset(version: str, split: str = "train", include_lr_sim: bool = True) -> PairedDataset:
    """Dataset for DATA_ROOT/datasets/<version>/<split>. See vfx/datagen/README.md."""
    return PairedDataset(paths.dataset_dir(version), split, include_lr_sim)


class SequenceDataset(PairedDataset):
    """Item = L consecutive frames t0..t0+L-1 of one scenario (all start offsets):
    lr (L,5,128,128), hr (L,2,512,512), lr_sim (L,5,128,128) if present, solid_lr, solid_hr,
    seed, t0. Frame k -> k+1 is one SIM_DT."""

    def __init__(self, root, split="train", length: int = 8, include_lr_sim: bool = True):
        super().__init__(root, split, include_lr_sim)
        if length < 2:
            raise ValueError("length must be >= 2")
        self.length = int(length)
        self.index = [(k, t0) for k, s in enumerate(self.scenarios)
                      for t0 in range(0, s["frames"] - self.length + 1)]

    def __getitem__(self, i: int) -> dict:
        k, t0 = self.index[i]
        a = self._arrays(k)
        sl = slice(t0, t0 + self.length)
        item = {"lr": self._f32(a["lr"][sl]), "hr": self._f32(a["hr"][sl]),
                "solid_lr": a["solid_lr"], "solid_hr": a["solid_hr"],
                "seed": self.scenarios[k]["seed"], "t0": t0}
        if "lr_sim" in a:
            item["lr_sim"] = self._f32(a["lr_sim"][sl])
        return item


def load_sequences(version: str, split: str = "train", length: int = 8,
                   include_lr_sim: bool = True) -> SequenceDataset:
    """Windows of `length` consecutive frames (v2: up to 16). See vfx/datagen/README.md."""
    return SequenceDataset(paths.dataset_dir(version), split, length, include_lr_sim)
