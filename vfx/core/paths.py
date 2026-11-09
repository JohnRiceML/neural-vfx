"""Where big artifacts live (outside OneDrive)."""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("VFX_DATA", Path(os.environ.get("LOCALAPPDATA", Path.home())) / "neural-vfx"))
DATASETS = DATA_ROOT / "datasets"
CHECKPOINTS = DATA_ROOT / "checkpoints"
RUNS = DATA_ROOT / "runs"
REPO_ROOT = Path(__file__).resolve().parents[2]


def dataset_dir(version: str) -> Path:
    return DATASETS / version


def default_checkpoint() -> Path:
    """The checkpoint the app loads by default (written by train/)."""
    return CHECKPOINTS / "demo" / "latest.pt"
