"""Dump a Python Solver3D reference run of a named scene, in a format any consumer can replay.

OWNER: native test engineer.

    <venv python> native/tools/dump_reference.py --scene plume --steps 120 [--config default|runtime|p1]
                  [--every 1] [--with-onestep] [--out DIR] [--max-gb 1.5]

Scenes: campfire, plume, determinism, stress, mass_box (see nvfx_scenes.py). Default output directory:
%LOCALAPPDATA%/neural-vfx/native_ref/<scene>_<config>_<steps>  (never inside the repo).

Format "nvfx-ref-1" (one directory):
  meta.json         res, dt, steps, every, channels, layout, ambient, config (SolverConfig3D field names =
                    NvfxSolverConfig field names), obstacles [{kind, pos, size}], solver_version, files.
  solid.npy         (N,N,N) float32, 1 inside obstacles (what reset(obstacles) rasterizes).
  forces.json       list of `steps` entries; entry k = the inputs of step k (state_k -> state_k+1):
                    {"emitters":[{pos,radius,density_rate,temperature,fuel_rate,velocity}], "impulses":
                    [{pos,radius,velocity}], "wind":[x,y,z]}  (world units, = NvfxForces).
  state_KKKKK.npy   (6,N,N,N) float32 C-order, index [c,z,y,x], channels density, vel_x, vel_y, vel_z,
                    temperature, fuel: the state BEFORE step K (state_00000 = the initial state; for
                    mass_box that is after the initial projection). Written for K % every == 0 and K = steps.
  onestep_KKKKK.npy (--with-onestep) Solver3D(P1 config: pressure_tol 0, 8 iters, warm start off).step of
                    state_K with forces[K]: the P1 teacher-forced reference, for K % every == 0, K < steps.
.npy is NumPy format v1: a 128-byte-aligned ASCII header ("\x93NUMPY", dict with descr '<f4',
fortran_order False, shape) followed by raw little-endian float32 data.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vfx.core.fields import AMBIENT_TEMP  # noqa: E402
from vfx.core.fields3d import CHANNELS_3D  # noqa: E402
from vfx.sim3d import Solver3D, SolverConfig3D, runtime_config  # noqa: E402

from nvfx_scenes import DT, SCENE_NAMES, forces_to_json, get_scene  # noqa: E402

FORMAT = "nvfx-ref-1"
P1_CFG = dict(pressure_tol=0.0, pressure_max_iters=8, pressure_warm_start=False)


def make_config(name: str, overrides: dict) -> SolverConfig3D:
    base = {"default": SolverConfig3D(), "runtime": runtime_config(),
            "p1": SolverConfig3D(**P1_CFG)}[name]
    return dataclasses.replace(base, **overrides)


def default_out(scene: str, config: str, steps: int) -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "neural-vfx" / "native_ref"
    return base / f"{scene}_{config}_{steps}"


def dump(scene: str, steps: int, config: str = "default", every: int = 1, with_onestep: bool = False,
         out: Path | None = None, res: int = 64, max_gb: float = 1.5) -> Path:
    sc = get_scene(scene)
    cfg = make_config(config, sc.config_overrides)
    out = Path(out) if out is not None else default_out(scene, config, steps)
    kept = [k for k in range(steps + 1) if k % every == 0 or k == steps]
    n_files = len(kept) + (len([k for k in kept if k < steps]) if with_onestep else 0)
    est_gb = n_files * 6 * res ** 3 * 4 / 1e9
    if est_gb > max_gb:
        raise SystemExit(f"would write {est_gb:.2f} GB > --max-gb {max_gb}; raise --every or lower --steps")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    s = Solver3D(res, cfg)
    st = s.reset(sc.obstacles)
    if sc.initial is not None:
        st.data.copy_(torch.from_numpy(sc.initial(res, st.solid.cpu().numpy())))
    if sc.project_first:
        s.project(st)
    one = st1 = None
    if with_onestep:
        one = Solver3D(res, make_config("p1", sc.config_overrides))
        st1 = one.reset(sc.obstacles)
    np.save(out / "solid.npy", st.solid.cpu().numpy())
    forces = []
    for k in range(steps + 1):
        if k in kept:
            np.save(out / f"state_{k:05d}.npy", st.data.cpu().numpy())
        if k == steps:
            break
        f = sc.forces(k)
        forces.append(forces_to_json(f))
        if with_onestep and k in kept:
            st1.data.copy_(st.data)
            one.step(st1, f, DT)
            np.save(out / f"onestep_{k:05d}.npy", st1.data.cpu().numpy())
        s.step(st, f, DT)
    (out / "forces.json").write_text(json.dumps(forces))
    meta = dict(format=FORMAT, scene=scene, res=res, dt=DT, steps=steps, every=every, kept_steps=kept,
                channels=list(CHANNELS_3D), layout="(C,D,H,W) float32 C-order, index [c,z,y,x]",
                ambient_temp=AMBIENT_TEMP, config_name=config, config=dataclasses.asdict(cfg),
                onestep_config=dataclasses.asdict(make_config("p1", sc.config_overrides)) if with_onestep else None,
                obstacles=[dataclasses.asdict(o) for o in sc.obstacles], project_first=sc.project_first,
                solver_version=st.meta.solver_version,
                files=dict(solid="solid.npy", forces="forces.json", state="state_{K:05d}.npy",
                           onestep="onestep_{K:05d}.npy" if with_onestep else None))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, choices=SCENE_NAMES)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--config", default="default", choices=("default", "runtime", "p1"))
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--with-onestep", action="store_true")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-gb", type=float, default=1.5)
    a = ap.parse_args()
    import warp as wp
    wp.config.quiet = True
    out = dump(a.scene, a.steps, a.config, a.every, a.with_onestep, a.out, a.res, a.max_gb)
    size = sum(p.stat().st_size for p in out.iterdir()) / 1e6
    print(f"wrote {out} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
