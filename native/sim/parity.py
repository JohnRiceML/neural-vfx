"""Owner's parity spot-check: native nvfx.dll (ctypes) vs the Python/Warp Solver3D, same inputs.

Run from the worktree root with the main checkout's venv:
    <venv>/python native/sim/parity.py [--steps 120] [--debug] [--stages]

  P2  projection only (fixed 8 PCG iterations, warm start off) on a smoothed random field
  P1  teacher-forced single steps: upload the Python state of step k, step both once (tol 0, 8 iters, cold)
  P3  free-running 120 steps from reset at the default config: mass, heat, centroid
Not the gate tests (native/tests/ is the test engineer's).
"""
from __future__ import annotations

import argparse
import ctypes as C
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from vfx.core.fields import AMBIENT_TEMP
from vfx.core.fields3d import SIM_DT, Forces3D, Obstacle3D
from vfx.sim3d import Solver3D, SolverConfig3D
from vfx.sim3d.selfcheck import SCENES

DLL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "bin", "nvfx.dll")
CH = ["density", "vel_x", "vel_y", "vel_z", "temperature", "fuel"]
AMB = np.array([0, 0, 0, 0, AMBIENT_TEMP, 0], dtype=np.float64)


class Cfg(C.Structure):
    _fields_ = [(n, C.c_float) for n in ("buoyancy_alpha", "buoyancy_beta", "vorticity_eps", "density_dissipation",
                                         "temperature_cooling", "burn_rate", "ignition_temp", "burn_heat",
                                         "smoke_from_fuel", "pressure_tol")] + \
               [("pressure_max_iters", C.c_int32), ("pressure_warm_start", C.c_int32)] + \
               [(n, C.c_float) for n in ("max_temperature", "wind_marker_temp", "heat_diffusion", "confinement_floor")]


class Emitter(C.Structure):
    _fields_ = [("pos", C.c_float * 3), ("radius", C.c_float), ("density_rate", C.c_float), ("temperature", C.c_float),
                ("fuel_rate", C.c_float), ("velocity", C.c_float * 3)]


class Impulse(C.Structure):
    _fields_ = [("pos", C.c_float * 3), ("radius", C.c_float), ("velocity", C.c_float * 3)]


class Forces(C.Structure):
    _fields_ = [("emitters", C.POINTER(Emitter)), ("num_emitters", C.c_int32), ("impulses", C.POINTER(Impulse)),
                ("num_impulses", C.c_int32), ("wind", C.c_float * 3)]


class Obstacle(C.Structure):
    _fields_ = [("kind", C.c_int32), ("pos", C.c_float * 3), ("size", C.c_float * 3)]


class Desc(C.Structure):
    _fields_ = [("adapter_index", C.c_int32), ("enable_debug_layer", C.c_int32)]


class Timings(C.Structure):
    _fields_ = [("advection_ms", C.c_float), ("forces_ms", C.c_float), ("projection_ms", C.c_float),
                ("total_ms", C.c_float), ("pcg_iters", C.c_int32), ("step_index", C.c_uint64)]


class Native:
    def __init__(self, debug=False):
        self.d = C.CDLL(os.path.abspath(DLL))
        d = self.d
        d.nvfx_last_error.restype = C.c_char_p
        d.nvfx_version_string.restype = C.c_char_p
        self.ctx = C.c_void_p()
        self._ck(d.nvfx_context_create(C.byref(Desc(-1, 1 if debug else 0)), C.byref(self.ctx)))
        self.s = None

    def _ck(self, r):
        if r != 0:
            raise RuntimeError(f"nvfx error {r}: {self.d.nvfx_last_error().decode()}")

    def cfg(self, c: SolverConfig3D) -> Cfg:
        x = Cfg()
        for n, _ in Cfg._fields_:
            setattr(x, n, int(getattr(c, n)) if n in ("pressure_max_iters", "pressure_warm_start") else float(getattr(c, n)))
        return x

    def create(self, res, c: SolverConfig3D):
        if self.s:
            self.d.nvfx_solver_destroy(self.s)
        self.s = C.c_void_p()
        self._ck(self.d.nvfx_solver_create(self.ctx, res, C.byref(self.cfg(c)), C.byref(self.s)))
        self.res = res

    def set_config(self, c):
        self._ck(self.d.nvfx_solver_set_config(self.s, C.byref(self.cfg(c))))

    def reset(self, obstacles):
        arr = (Obstacle * max(1, len(obstacles)))()
        for i, o in enumerate(obstacles):
            arr[i].kind = 0 if o.kind == "sphere" else 1
            arr[i].pos[:] = list(o.pos)
            arr[i].size[:] = list(o.size)
        self._ck(self.d.nvfx_solver_reset(self.s, arr, len(obstacles)))

    def step(self, f: Forces3D, dt):
        em = (Emitter * max(1, len(f.emitters)))()
        for i, e in enumerate(f.emitters):
            em[i].pos[:] = list(e.pos)
            em[i].radius, em[i].density_rate, em[i].temperature, em[i].fuel_rate = e.radius, e.density_rate, e.temperature, e.fuel_rate
            em[i].velocity[:] = list(e.velocity)
        im = (Impulse * max(1, len(f.impulses)))()
        for i, m in enumerate(f.impulses):
            im[i].pos[:] = list(m.pos)
            im[i].radius = m.radius
            im[i].velocity[:] = list(m.velocity)
        fo = Forces(em, len(f.emitters), im, len(f.impulses), (C.c_float * 3)(*f.wind))
        self._ck(self.d.nvfx_solver_step(self.s, C.byref(fo), C.c_float(dt)))

    def project(self):
        self._ck(self.d.nvfx_solver_project(self.s))

    def download(self):
        n = self.res
        data = np.empty((6, n, n, n), np.float32)
        sol = np.empty((n, n, n), np.float32)
        self._ck(self.d.nvfx_solver_download_state(self.s, data.ctypes.data_as(C.POINTER(C.c_float)),
                                                   sol.ctypes.data_as(C.POINTER(C.c_float))))
        return data, sol

    def upload(self, data, sol=None):
        data = np.ascontiguousarray(data, np.float32)
        sp = None if sol is None else np.ascontiguousarray(sol, np.float32)
        self._ck(self.d.nvfx_solver_upload_state(self.s, data.ctypes.data_as(C.POINTER(C.c_float)),
                                                 None if sp is None else sp.ctypes.data_as(C.POINTER(C.c_float))))

    def divergence(self):
        n = self.res
        out = np.empty((n, n, n), np.float32)
        self._ck(self.d.nvfx_solver_download_divergence(self.s, out.ctypes.data_as(C.POINTER(C.c_float))))
        return out

    def timings(self):
        t = Timings()
        self._ck(self.d.nvfx_solver_get_timings(self.s, C.byref(t)))
        return t


def errs(n, p):
    """per channel rel_l2, 99.9th pct abs/peak, max abs/peak"""
    out = []
    for c in range(n.shape[0]):
        pc = p[c].astype(np.float64)
        d = np.abs(n[c].astype(np.float64) - pc)
        dev = pc - AMB[c] if n.shape[0] == 6 else pc
        rel = np.sqrt((d ** 2).sum()) / (np.sqrt((dev ** 2).sum()) + 1e-12)
        peak = np.abs(dev).max() + 1e-12
        out.append((rel, np.percentile(d, 99.9) / peak, d.max() / peak))
    return out


def p1_cfg():
    return SolverConfig3D(pressure_tol=0.0, pressure_max_iters=8, pressure_warm_start=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--scenes", default="campfire,plume")
    args = ap.parse_args()
    nat = Native(args.debug)
    print(nat.d.nvfx_version_string().decode())
    N = 64

    # ---- P2: projection only
    torch.manual_seed(1)
    for sph in (False, True):
        obs = [SCENES["plume"][1][0]] if sph else []
        py = Solver3D(N, p1_cfg())
        st = py.reset(obs)
        v = F.avg_pool3d(torch.randn(1, 3, N, N, N, device="cuda"), 3, 1, 1)[0]
        st.data[1:4] = v * (st.solid < 0.5)
        nat.create(N, p1_cfg())
        nat.reset(obs)
        d0, s0 = nat.download()
        assert np.array_equal(s0, st.solid.cpu().numpy()), "solid rasterization differs"
        nat.upload(st.data.cpu().numpy())
        py.project(st)
        nat.project()
        dn, _ = nat.download()
        e = errs(dn[1:4], st.data[1:4].cpu().numpy())
        print(f"P2 projection-only sphere={sph}: vel rel_l2 {[f'{x[0]:.2e}' for x in e]} max abs/peak {[f'{x[2]:.1e}' for x in e]}"
              f" | div py {float(py.divergence(st).max()):.2e} native {nat.divergence().max():.2e}")

    # ---- P1 teacher-forced + P3 free-running
    for name in args.scenes.split(","):
        fn, obs = SCENES[name]
        ref = Solver3D(N)                 # default config: drives the trajectory (also P3 reference)
        one = Solver3D(N, p1_cfg())       # single-step reference
        st = ref.reset(obs)
        nat.create(N, p1_cfg())
        nat.reset(obs)
        worst = np.zeros((6, 3))
        worst_k = np.zeros((6,), int)
        for k in range(args.steps):
            f = fn(k * SIM_DT)
            x = st.data.clone()
            nat.upload(x.cpu().numpy())
            st1 = type(st)(data=x, solid=st.solid, meta=st.meta)
            one.step(st1, f, SIM_DT)
            nat.step(f, SIM_DT)
            dn, _ = nat.download()
            e = np.array(errs(dn, st1.data.cpu().numpy()))
            for c in range(6):
                if e[c, 0] > worst[c, 0]:
                    worst_k[c] = k
            worst = np.maximum(worst, e)
            ref.step(st, f, SIM_DT)
        print(f"P1 {name} teacher-forced steps 0..{args.steps - 1} (worst over steps):")
        for c in range(6):
            print(f"    {CH[c]:12s} rel_l2 {worst[c, 0]:.2e} (step {worst_k[c]})  p99.9 abs/peak {worst[c, 1]:.1e}  max abs/peak {worst[c, 2]:.1e}")

        # P3 free-running, default config
        nat.create(N, SolverConfig3D())
        nat.reset(obs)
        stp = ref.reset(obs)
        ref2 = Solver3D(N)
        curve = []
        for k in range(args.steps):
            f = fn(k * SIM_DT)
            ref2.step(stp, f, SIM_DT)
            nat.step(f, SIM_DT)
            if (k + 1) % 20 == 0:
                dn, _ = nat.download()
                curve.append((k + 1, errs(dn, stp.data.cpu().numpy())))
        dn, _ = nat.download()
        p = stp.data.cpu().numpy().astype(np.float64)
        n = dn.astype(np.float64)
        y = (np.arange(N) + 0.5) / N

        def budget(a):
            m = a[0].sum() / N ** 3
            h = (a[4] - AMBIENT_TEMP).sum() / N ** 3
            yc = (a[0].sum((0, 2)) * y).sum() / (a[0].sum() + 1e-12)
            xs = (a[0].sum((0, 1)) * y).sum() / (a[0].sum() + 1e-12)
            zc = (a[0].sum((1, 2)) * y).sum() / (a[0].sum() + 1e-12)
            return m, h, np.array([xs, yc, zc])
        mp, hp, cp = budget(p)
        mn, hn, cn = budget(n)
        print(f"P3 {name} free-running {args.steps} steps: mass py {mp:.5f} native {mn:.5f} ({100 * (mn / mp - 1):+.3f} %) | "
              f"heat py {hp:.3f} native {hn:.3f} ({100 * (hn / hp - 1):+.3f} %) | centroid diff {np.abs(cn - cp).max():.4f} m")
        print("    rel_l2 curve (density, vel_y, temperature):",
              " ".join(f"{k}:{e[0][0]:.1e}/{e[2][0]:.1e}/{e[4][0]:.1e}" for k, e in curve))
        t = nat.timings()
        print(f"    native last timings: total {t.total_ms:.3f} ms iters {t.pcg_iters} (step {t.step_index}); "
              f"python iters {int(ref2.pressure.it.numpy()[0])}")


if __name__ == "__main__":
    main()
