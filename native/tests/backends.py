"""Backends with the NativeSolver interface (see nvfx_ctypes.py).

  native   nvfx_ctypes.NativeSolver (the DLL)
  pyfake   PyFakeSolver: the same interface implemented on vfx.sim3d.Solver3D. Run through the gate checks
           it must give exactly 0 parity error and the reference numbers: that validates the harness.
  Perturbed(fake, mode): a deliberately wrong fake for the negative controls (the checks must FAIL).
"""
from __future__ import annotations

import os

import numpy as np
import torch

from vfx.core.fields3d import FieldState3D, Forces3D
from vfx.sim3d import Solver3D, SolverConfig3D


class PyFakeSolver:
    """Reference-backed fake. Semantics mirror nvfx.h: upload clears the pressure warm start, reset clears it
    too, step counter lives in the state. Timings come from Solver3D.step_profiled when enabled."""
    name = "pyfake"

    def __init__(self, res: int = 64, config: SolverConfig3D | None = None):
        self.res = int(res)
        self.s = Solver3D(self.res, config if config is not None else SolverConfig3D())
        self.st: FieldState3D = self.s.reset([])
        self._timing = False
        self._last = None
        self._n = 0
        self.version = "pyfake over vfx.sim3d " + self.st.meta.solver_version

    def set_config(self, cfg):
        self.s.config = cfg

    def _clear_warm(self):
        self.s.pressure.phi.zero_()

    def reset(self, obstacles=()):
        self.st = self.s.reset(list(obstacles))
        self._clear_warm()
        self._n = 0

    def step(self, forces: Forces3D, dt: float):
        if self._timing:
            t = self.s.step_profiled(self.st, forces, dt)
            t["pcg_iters"] = int(self.s.pressure.it.numpy()[0])
            t["step_index"] = self._n
            self._last = t
        else:
            self.s.step(self.st, forces, dt)
        self._n += 1

    def project(self):
        self.s.project(self.st)

    def download(self) -> np.ndarray:
        return self.st.data.cpu().numpy().copy()

    def download_solid(self) -> np.ndarray:
        return self.st.solid.cpu().numpy().copy()

    def upload(self, data: np.ndarray, solid: np.ndarray | None = None):
        # copy in place: keeps the Solver3D graph cache keyed on the same buffers
        self.st.data.copy_(torch.from_numpy(np.ascontiguousarray(data, np.float32)))
        if solid is not None:
            self.st.solid.copy_(torch.from_numpy(np.ascontiguousarray(solid, np.float32)))
        self._clear_warm()

    def divergence(self) -> np.ndarray:
        return self.s.divergence(self.st).cpu().numpy()

    def get_timings(self):
        return dict(self._last) if self._last else None

    def set_timing(self, on: bool):
        self._timing = bool(on)

    def wait_idle(self):
        torch.cuda.synchronize()

    def close(self):
        pass


PERTURB_MODES = ("scale_density_1e-3", "scale_density_1e-6", "scale_density_2e-2", "swap_vx_vz", "shift_x",
                 "nondeterministic", "skip_project", "weak_projection")


class Perturbed(PyFakeSolver):
    """A fake with a deliberate defect.
      scale_density_1e-3   download returns density * (1 + 1e-3)          -> P1 must fail
      scale_density_1e-6   download returns density * (1 + 1e-6)          -> P1 must PASS (not over-sensitive)
      scale_density_2e-2   density * 1.02 after each step (persistent)    -> P3 must fail
      swap_vx_vz           download swaps vel_x / vel_z (layout bug)      -> P1 must fail
      shift_x              download rolled by one cell along x            -> P1 must fail
      nondeterministic     one random cell nudged by 1 ulp after each step-> S3 must fail
      skip_project         project() does nothing                         -> P2 must fail
      weak_projection      runs every step with 2 PCG iterations, tol 0   -> S1 must fail (plume)
    """

    def __init__(self, res=64, config=None, mode="scale_density_1e-3"):
        assert mode in PERTURB_MODES, mode
        self.mode = mode
        super().__init__(res, config)
        self.name = f"pyfake[{mode}]"
        if mode == "weak_projection":
            self.set_config(self.s.config)

    def set_config(self, cfg):
        if self.mode == "weak_projection":
            import dataclasses
            cfg = dataclasses.replace(cfg, pressure_tol=0.0, pressure_max_iters=2)
        super().set_config(cfg)

    def step(self, forces, dt):
        super().step(forces, dt)
        if self.mode == "scale_density_2e-2":
            self.st.data[0] *= 1.02
        elif self.mode == "nondeterministic":
            rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
            z, y, x = (int(v) for v in rng.integers(0, self.res, 3))
            v = self.st.data[4, z, y, x]
            self.st.data[4, z, y, x] = torch.nextafter(v, v + 1.0)

    def project(self):
        if self.mode != "skip_project":
            super().project()

    def download(self):
        d = super().download()
        if self.mode == "scale_density_1e-3":
            d[0] *= np.float32(1 + 1e-3)
        elif self.mode == "scale_density_1e-6":
            d[0] = (d[0].astype(np.float64) * (1 + 1e-6)).astype(np.float32)
        elif self.mode == "swap_vx_vz":
            d[[1, 3]] = d[[3, 1]]
        elif self.mode == "shift_x":
            d = np.ascontiguousarray(np.roll(d, 1, axis=3))
        return d


def make_backend(name: str, res: int = 64, config=None):
    if name == "native":
        from nvfx_ctypes import NativeSolver
        return NativeSolver(res, config)
    if name == "pyfake":
        return PyFakeSolver(res, config)
    if name.startswith("pyfake[") and name.endswith("]"):
        return Perturbed(res, config, mode=name[7:-1])
    raise ValueError(f"unknown backend {name!r}")
