"""Promote a route B run to the default checkpoint (CHECKPOINTS/screen3d/latest.pt), only if its final eval
(summary.json, vfx.eval3d.metrics through the runtime) beats BOTH baselines (lr, tuned unsharp) on the test split
on l1, spectrum_error and ms_dssim. The previous latest.pt is kept as latest_prev.pt.

    .venv/Scripts/python -m vfx.train3d.promote <run_dir or run name> [--force-reason TEXT]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from vfx.core import paths
from vfx.train3d.train import CKPT_DIR, beats_baselines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    a = ap.parse_args(argv)
    rd = Path(a.run)
    if not rd.exists():
        rd = paths.RUNS / "screen3d" / a.run
    s = json.loads((rd / "summary.json").read_text())
    ev = s.get("eval")
    if not isinstance(ev, dict) or not beats_baselines(ev):
        raise SystemExit(f"NOT promoted: {rd.name} does not beat both baselines on test ({json.dumps(ev)[:400]})")
    dst = CKPT_DIR / "latest.pt"
    if dst.exists():
        shutil.copy2(dst, CKPT_DIR / "latest_prev.pt")
    shutil.copy2(rd / "model.pt", dst)
    (CKPT_DIR / "latest.json").write_text(json.dumps({"run": rd.name, "eval": ev}, indent=1))
    print(f"promoted {rd / 'model.pt'} -> {dst}")


if __name__ == "__main__":
    main()
