"""Gate 3a R1 for one NVFX_RENDER_DITHER_MODE, in a fresh process (OWNER: native test engineer).

Spawned by native/tests/test_gate3a.py::test_R1_dither_modes with the env var set (0, 1, 2) or unset ("default").
Needs no torch / CUDA: it reads the Python reference images and the 64^3 states that the main test wrote to
native/build/gate3a_cache/dither_inputs.npz, uploads each state to a native solver (nvfx_solver_upload_state), draws
each (scene, camera) with its previous camera and NVFX_RENDER_AUX, and computes the R1 numbers + the SHA-256 of the
native RGB bytes. The main test compares those hashes with its own in-process default-mode images.

    <venv python> native/tools/gate3a_dither_probe.py --inputs <npz> --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "native" / "tests", ROOT / "native" / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import gate3a_checks as G
    import nvfx_render_ctypes as NR
    from nvfx_ctypes import NativeSolver

    out = dict(mode=os.environ.get("NVFX_RENDER_DITHER_MODE", "unset"), started=time.strftime("%Y-%m-%d %H:%M:%S"),
               cases={})
    ok, msg = NR.render_api_status()
    if not ok:
        out["error"] = msg
        Path(a.out).write_text(json.dumps(out, indent=1))
        return 2
    out["dll"] = msg
    z = np.load(a.inputs, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    ns = NativeSolver(64)
    nr = NR.NativeRenderer(ns, G.FRAME_W, G.FRAME_H, G.MARCH_SCALE)
    try:
        for scene, cams in meta.items():
            ns.upload(z[f"{scene}__data"], z[f"{scene}__solid"])
            for cam_name, cv in cams.items():
                cam = G.Cam(**{k: tuple(v) if isinstance(v, list) else v for k, v in cv["camera"].items()})
                prev = G.Cam(**{k: tuple(v) if isinstance(v, list) else v for k, v in cv["prev_camera"].items()})
                nr.draw(cam, prev, NR.NVFX_RENDER_AUX)
                rgb = np.ascontiguousarray(nr.color()[..., :3])
                r1 = G.r1_metrics(rgb, z[f"{scene}__{cam_name}__py_rgb"])
                out["cases"][f"{scene}_{cam_name}"] = dict(r1=r1, native_sha256=G.sha256(rgb))
    finally:
        nr.close()
        ns.close()
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
