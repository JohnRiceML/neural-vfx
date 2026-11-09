"""Shared fixtures/helpers for the gate tests (owner: test/eval engineer).

Modules owned by other engineers may not exist yet: tests that need them skip with a
clear reason instead of erroring, so `pytest tests -q` is always runnable.
"""
from __future__ import annotations

import dataclasses
import os

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


def import_or_skip(modname: str, attrs: tuple[str, ...] = ()):
    """Import a module under test, skipping (not erroring) if it or a needed attr is missing."""
    try:
        mod = __import__(modname, fromlist=["_"])
    except ImportError as e:  # module not written yet / broken import
        pytest.skip(f"{modname} not importable yet: {e}")
    missing = [a for a in attrs if not hasattr(mod, a)]
    if missing:
        pytest.skip(f"{modname} does not export {missing} yet (CLAUDE.md interface)")
    return mod


def make_config(sim, **overrides):
    """SolverConfig() with overrides applied (field names are fixed by CLAUDE.md)."""
    cfg = sim.SolverConfig()
    for k in overrides:
        if not hasattr(cfg, k):
            pytest.fail(f"SolverConfig has no field '{k}' (interface in CLAUDE.md requires it)")
    if dataclasses.is_dataclass(cfg):
        return dataclasses.replace(cfg, **overrides)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def dataset_version() -> str:
    return os.environ.get("VFX_DATASET", "v1")
