"""Data factory: randomized scenarios -> paired LR/HR solver runs -> versioned datasets.

Public API (see CLAUDE.md and vfx/datagen/README.md):
    sample_scenario(seed, split) -> Scenario
    run(scenario) -> dict                       # writes one scenario dir, returns meta
    load_dataset(version, split="train") -> PairedDataset
"""
from .dataset import PairedDataset, load_dataset
from .runner import run
from .scenarios import DEFAULT_FRAMES, SPLITS, sample_scenario

__all__ = ["sample_scenario", "run", "load_dataset", "PairedDataset", "SPLITS", "DEFAULT_FRAMES"]
