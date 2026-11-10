"""Gate 2 inputs: real rendered 2560x1440 frames + aux from the Python reference (campfire, plume + sphere).

OWNER: native test engineer. Used by native/tests/gate2_checks.py. Written from vfx/ (reference side) only.

    solver  vfx.sim3d.Solver3D(64, SolverConfig3D()) + the scene's forces (native/tools/nvfx_scenes.py), STEPS steps
    render  vfx.render3d.VolumeRenderer((1440, 2560), march_scale=0.5).draw(volume, camera_at(1.3), solid,
            return_aux=True, velocity=state.velocity, prev_camera=camera_at(1.32))   (camera_at: vfx/render3d/bench.py)

Result (CPU numpy, nothing kept on the GPU): rgb (H,W,3) uint8, alpha / depth (H,W) float16, motion (2,H,W)
float16 (output px per SIM_DT, x right, y down), effect_bbox = bbox of alpha > ALPHA_EFFECT (exclusive end).
The aux arrays are float16 because that is what the renderer returns and what the native R16F / RG16F textures
hold: both sides of the comparison see exactly the same values.

Cache: native/build/gate2_cache/<scene>_<key>.npz (~45 MB per scene, D: drive), key = hash of the vfx sources the
frame depends on + the parameters here. Env NVFX_G2_NO_CACHE=1 disables it.

    <venv python> native/tools/detail_frames.py --scene campfire      # build (or load) and print a summary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "native" / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FRAME_H, FRAME_W = 1440, 2560
MARCH_SCALE = 0.5
STEPS = 150                  # as Gate 3a ("after 150 steps")
CAMERA_DIST = 1.3            # close-up: the effect covers a large part of the frame
PREV_CAMERA_DIST = 1.32      # previous camera -> non-zero camera motion in `motion`
ALPHA_EFFECT = 0.05
SCENES = ("campfire", "plume")
CACHE_DIR = ROOT / "native" / "build" / "gate2_cache"
_SOURCES = ("vfx/render3d/renderer.py", "vfx/render3d/kernels.py", "vfx/render3d/bench.py", "vfx/sim3d/solver.py",
            "vfx/sim3d/kernels.py", "vfx/sim3d/pressure.py", "vfx/sim3d/selfcheck.py", "vfx/core/fields3d.py",
            "vfx/render/blackbody.py", "native/tools/nvfx_scenes.py")


@dataclass
class DetailInputs:
    scene: str
    rgb: np.ndarray            # (H, W, 3) uint8, row 0 = top
    alpha: np.ndarray          # (H, W) float16
    depth: np.ndarray          # (H, W) float16
    motion: np.ndarray         # (2, H, W) float16
    effect_bbox: tuple | None  # (x0, y0, x1, y1), exclusive end; None = no pixel above ALPHA_EFFECT
    meta: dict = field(default_factory=dict)

    @property
    def hw(self) -> tuple[int, int]:
        return self.rgb.shape[0], self.rgb.shape[1]

    def rgba(self) -> np.ndarray:
        """(H, W, 4) uint8 for the native RGBA8 colour texture. A is a fixed non-constant pattern, so the header's
        "A is passed through" is observable (Python ignores A)."""
        H, W = self.hw
        out = np.empty((H, W, 4), np.uint8)
        out[..., :3] = self.rgb
        out[..., 3] = alpha_pattern(H, W)
        return out

    def motion_rg(self) -> np.ndarray:
        """(H, W, 2) float16, interleaved (x, y), for the RG16F motion texture."""
        return np.ascontiguousarray(np.moveaxis(self.motion, 0, -1))


def alpha_pattern(H: int, W: int) -> np.ndarray:
    y, x = np.mgrid[0:H, 0:W]
    return ((x * 7 + y * 13) & 255).astype(np.uint8)


def effect_bbox(alpha: np.ndarray, thr: float = ALPHA_EFFECT):
    m = alpha.astype(np.float32) > thr
    if not m.any():
        return None
    ys, xs = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
    return (int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1)


def _cache_key(scene: str) -> str:
    h = hashlib.sha1()
    for rel in _SOURCES:
        p = ROOT / rel
        h.update(rel.encode())
        h.update(p.read_bytes() if p.is_file() else b"<missing>")
    h.update(json.dumps(dict(scene=scene, H=FRAME_H, W=FRAME_W, ms=MARCH_SCALE, steps=STEPS, cam=CAMERA_DIST,
                             prev=PREV_CAMERA_DIST)).encode())
    return h.hexdigest()[:16]


def _render(scene: str) -> DetailInputs:
    import torch
    import warp as wp
    wp.config.quiet = True
    from vfx.core.fields3d import CH3, Volume
    from vfx.render3d import VolumeRenderer
    from vfx.render3d.bench import camera_at
    from vfx.sim3d import Solver3D, SolverConfig3D
    from nvfx_scenes import DT, get_scene

    sc = get_scene(scene)
    s = Solver3D(64, SolverConfig3D(**sc.config_overrides))
    st = s.reset(sc.obstacles)
    for i in range(STEPS):
        st = s.step(st, sc.forces(i), DT) or st
    vol = Volume(st.data[CH3["density"]].clone(), st.data[CH3["temperature"]].clone())
    vel = st.velocity.clone()
    solid = st.solid.clone() if sc.obstacles else None
    r = VolumeRenderer((FRAME_H, FRAME_W), march_scale=MARCH_SCALE)
    frame, aux = r.draw(vol, camera_at(CAMERA_DIST), solid, return_aux=True, velocity=vel,
                        prev_camera=camera_at(PREV_CAMERA_DIST))
    torch.cuda.synchronize()
    rgb = frame.cpu().numpy().copy()
    alpha = aux["alpha"].to(torch.float16).cpu().numpy().copy()
    depth = aux["depth"].to(torch.float16).cpu().numpy().copy()
    motion = aux["motion"].to(torch.float16).cpu().numpy().copy()
    del r, s, st, vol, vel, solid, frame, aux
    torch.cuda.empty_cache()
    eb = effect_bbox(alpha)
    a32 = alpha.astype(np.float32)
    meta = dict(scene=scene, steps=STEPS, camera=f"camera_at({CAMERA_DIST})", prev_camera=f"camera_at({PREV_CAMERA_DIST})",
                march_scale=MARCH_SCALE, size=[FRAME_W, FRAME_H], effect_bbox=list(eb) if eb else None,
                coverage_alpha_gt_005=float((a32 > ALPHA_EFFECT).mean()),
                motion_absmax=float(np.abs(motion.astype(np.float32)).max()),
                depth_max=float(depth.astype(np.float32).max()))
    return DetailInputs(scene, rgb, alpha, depth, motion, eb, meta)


def make_inputs(scene: str, use_cache: bool | None = None) -> DetailInputs:
    if use_cache is None:
        use_cache = os.environ.get("NVFX_G2_NO_CACHE") != "1"
    key = _cache_key(scene)
    path = CACHE_DIR / f"{scene}_{key}.npz"
    if use_cache and path.is_file():
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        eb = tuple(meta["effect_bbox"]) if meta.get("effect_bbox") else None
        meta["cache"] = str(path)
        return DetailInputs(scene, z["rgb"], z["alpha"], z["depth"], z["motion"], eb, meta)
    inp = _render(scene)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(path, rgb=inp.rgb, alpha=inp.alpha, depth=inp.depth, motion=inp.motion,
                 meta=np.array(json.dumps(inp.meta)))
        inp.meta["cache"] = str(path) + " (written)"
    return inp


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=SCENES, default="campfire")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args(argv)
    inp = make_inputs(a.scene, use_cache=not a.no_cache)
    print(json.dumps(inp.meta, indent=1))


if __name__ == "__main__":
    main()
