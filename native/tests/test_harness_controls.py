"""Harness self-validation (independent of --backend; needs CUDA only, never the DLL).

- ctypes struct layouts match nvfx.h (x64), forces/obstacle conversion is faithful
- the Python one-step solver used as the P1 reference is memoryless
- the DLL-absent path skips with a clear message
- NEGATIVE CONTROLS: deliberately broken fakes must make the parity / spec checks FAIL, and a perturbation
  far below the thresholds must PASS (so the harness is neither blind nor over-sensitive)
- dump_reference.py output can be replayed bit-exactly
"""
from __future__ import annotations

import ctypes as C
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import gate1_checks as G
import nvfx_ctypes as N
from backends import make_backend

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


def fake(mode=None):
    name = "pyfake" if mode is None else f"pyfake[{mode}]"
    return lambda res, cfg: make_backend(name, res, cfg)


def _rec(record, key, r, expect_pass):
    r = dict(r)
    r["expected"] = "PASS" if expect_pass else "FAIL"
    r["pass_control"] = bool(r["pass"]) == expect_pass
    record(key, r)
    assert r["pass"] == expect_pass, f"control expected the check to {'pass' if expect_pass else 'FAIL'}: {r['summary']}"


# ------------------------------------------------------------------ binding
def test_struct_layouts():
    for st, size in N.EXPECTED_SIZES.items():
        assert C.sizeof(st) == size, (st.__name__, C.sizeof(st), size)
    assert N.NvfxForces.impulses.offset == 16 and N.NvfxForces.wind.offset == 28
    assert N.NvfxSolverTimings.pcg_iters.offset == 16 and N.NvfxSolverTimings.step_index.offset == 24
    assert N.NvfxSolverConfig.pressure_max_iters.offset == 40 and N.NvfxSolverConfig.confinement_floor.offset == 60
    assert N.NvfxObstacle.size.offset == 16


def test_conversions():
    from vfx.core.fields3d import Obstacle3D
    from vfx.sim3d import SolverConfig3D, runtime_config
    from nvfx_scenes import get_scene
    for cfg in (SolverConfig3D(), runtime_config(), G.P1_CFG):
        d = N.config_from_c(N.config_to_c(cfg))
        for k, v in d.items():
            assert np.float32(getattr(cfg, k)) == np.float32(v), k
    f = get_scene("stress").forces(0)
    fc = N.ForcesC(f)
    assert fc.s.num_emitters == len(f.emitters) == 6 and fc.s.num_impulses == 1
    e = fc.s.emitters[5]
    assert tuple(e.pos) == pytest.approx(f.emitters[5].pos) and e.fuel_rate == 20.0
    assert tuple(fc.s.impulses[0].velocity) == pytest.approx(f.impulses[0].velocity, rel=1e-6)
    assert tuple(fc.s.wind) == pytest.approx(f.wind)
    arr, n = N.obstacles_to_c([Obstacle3D("box", (0.25, 0.7, 0.3), (0.1, 0.02, 0.15))])
    assert n == 1 and arr[0].kind == N.NVFX_OBSTACLE_BOX and arr[0].size[1] == pytest.approx(0.02)


def test_dll_absent_skips(monkeypatch, tmp_path):
    monkeypatch.setenv("NVFX_DLL", str(tmp_path / "nope.dll"))
    ok, msg = N.dll_available()
    assert not ok and "not found" in msg


# ------------------------------------------------------------------ P1 reference is memoryless
def test_python_one_step_is_memoryless():
    """P1's Python side: Solver3D(P1_CFG) (warm start off -> the rhs kernel zeroes phi; the PCG scalars and
    iteration counter are re-initialised by the init=1 pass; the MG operators depend only on `solid`).
    Stepping x_k must give the same bits whatever the solver stepped before."""
    from vfx.sim3d import Solver3D, SolverConfig3D
    from nvfx_scenes import DT, get_scene
    sc = get_scene("plume")
    traj = Solver3D(64)
    st = traj.reset(sc.obstacles)
    xs = []
    for k in range(40):
        if k in (10, 39):
            xs.append((k, st.data.clone()))
        traj.step(st, sc.forces(k), DT)
    one = Solver3D(64, G.P1_CFG)
    ost = one.reset(sc.obstacles)
    (k1, x1), (k2, x2) = xs
    ost.data.copy_(x1); one.step(ost, sc.forces(k1), DT); a = ost.data.clone()
    ost.data.copy_(x2); one.step(ost, sc.forces(k2), DT)
    ost.data.copy_(x1); one.step(ost, sc.forces(k1), DT); b = ost.data.clone()
    fresh = Solver3D(64, G.P1_CFG)
    fst = fresh.reset(sc.obstacles)
    fst.data.copy_(x1); fresh.step(fst, sc.forces(k1), DT)
    assert torch.equal(a, b) and torch.equal(a, fst.data)
    # and the warm-started default config is NOT memoryless (why P1 turns it off)
    w = Solver3D(64, SolverConfig3D(pressure_tol=0.0, pressure_max_iters=8))
    wst = w.reset(sc.obstacles)
    wst.data.copy_(x1); w.step(wst, sc.forces(k1), DT); c = wst.data.clone()
    wst.data.copy_(x2); w.step(wst, sc.forces(k2), DT)
    wst.data.copy_(x1); w.step(wst, sc.forces(k1), DT)
    assert not torch.equal(c, wst.data)


# ------------------------------------------------------------------ positive control: exact fake
def test_fake_p1_is_exactly_zero(record):
    r = G.check_p1(fake(), "plume", steps=range(30, 36))
    record("HC_fake_P1_plume_30-35", r)
    assert r["pass"] and r["worst"]["value"] == 0.0


# ------------------------------------------------------------------ negative controls
@pytest.mark.parametrize("mode,expect_pass", [("scale_density_1e-3", False), ("swap_vx_vz", False),
                                              ("shift_x", False), ("scale_density_1e-6", True)])
def test_control_p1(record, mode, expect_pass):
    r = G.check_p1(fake(mode), "campfire", steps=range(30, 36))
    _rec(record, f"HC_P1_{mode}", r, expect_pass)


def test_control_p2_skip_project(record):
    _rec(record, "HC_P2_skip_project", G.check_p2_project(fake("skip_project")), False)


def test_control_p3_density_drift(record):
    _rec(record, "HC_P3_scale_density_2e-2", G.check_p3(fake("scale_density_2e-2"), "plume"), False)


def test_control_s1_weak_projection(record):
    _rec(record, "HC_S1_weak_projection_plume", G.check_s1(fake("weak_projection"), "plume"), False)


def test_control_s3_nondeterministic(record):
    _rec(record, "HC_S3_nondeterministic", G.check_s3(fake("nondeterministic"), steps=20), False)


# ------------------------------------------------------------------ reference dumps
def test_dump_reference_replays(tmp_path):
    root = Path(__file__).resolve().parents[2]
    out = tmp_path / "plume_dump"
    subprocess.run([sys.executable, str(root / "native" / "tools" / "dump_reference.py"), "--scene", "plume",
                    "--steps", "5", "--config", "p1", "--out", str(out)], check=True, cwd=root)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["layout"] == "(C,D,H,W) float32 C-order, index [c,z,y,x]" and meta["steps"] == 5
    sys.path.insert(0, str(root / "native" / "tools"))
    from nvfx_scenes import forces_from_json
    from vfx.sim3d import SolverConfig3D
    forces = json.loads((out / "forces.json").read_text())
    cfg = SolverConfig3D(**meta["config"])
    b = make_backend("pyfake", meta["res"], cfg)
    b.reset([])
    b.upload(np.load(out / "state_00000.npy"), np.load(out / "solid.npy"))
    for k in range(5):
        b.step(forces_from_json(forces[k]), meta["dt"])
        assert np.array_equal(b.download(), np.load(out / f"state_{k + 1:05d}.npy")), k
