"""2D smoke + fire Eulerian solver (Warp), collocated grid, world units.

Step order (output velocity is divergence-free):
  1. advect density, velocity, temperature, fuel by u^n   (MacCormack, RK2 traces, clamped)
  2. density mass correction (global, deterministic tile reduction; see README)
  3. emitters, impulses, combustion, cooling, dissipation
  4. buoyancy, vorticity confinement, wind
  5. pressure projection (MG-PCG, exact for the face-averaged central divergence)

The whole step is captured once per (state.data, state.solid) buffer pair as a CUDA graph;
per-step inputs (dt, forces, config) are uploaded to a small device buffer first.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import fields as dc_fields

import numpy as np
import torch
import warp as wp

from vfx.core.fields import (AMBIENT_TEMP, CH, DOMAIN_SIZE, NUM_CHANNELS, FieldMeta, FieldState,
                             Forces, Obstacle, rasterize_obstacles)

from . import kernels as K
from .config import SOLVER_VERSION, SolverConfig
from .pressure import PressureSolver, _make_reductions

_CFG_SLOTS = {
    "buoyancy_alpha": K.P_ALPHA, "buoyancy_beta": K.P_BETA, "vorticity_eps": K.P_EPS,
    "density_dissipation": K.P_DDISS, "temperature_cooling": K.P_COOL, "burn_rate": K.P_BURN,
    "ignition_temp": K.P_IGN, "burn_heat": K.P_BHEAT, "smoke_from_fuel": K.P_SMOKE,
    "pressure_tol": K.P_TOL, "pressure_max_iters": K.P_MAXIT, "max_temperature": K.P_TMAX,
    "wind_marker_temp": K.P_WMARK,
}


def _make_mass_kernels(H: int, W: int):
    ROWS = max(1, 1024 // W)
    NB = (H + ROWS - 1) // ROWS

    @wp.func
    def fluid_w(s: float) -> float:
        return wp.where(s > 0.5, 0.0, 1.0)

    @wp.kernel(module="unique")
    def k_part_mass(pre: wp.array3d(dtype=float), post: wp.array3d(dtype=float),
                    solid: wp.array2d(dtype=float), part: wp.array(dtype=float)):
        i = wp.tid()
        ts = wp.tile_map(fluid_w, wp.tile_load(solid, shape=(ROWS, W), offset=(i * ROWS, 0)))
        ta = wp.tile_load(pre[0], shape=(ROWS, W), offset=(i * ROWS, 0))
        tb = wp.tile_load(post[0], shape=(ROWS, W), offset=(i * ROWS, 0))
        sa = wp.tile_sum(wp.tile_map(wp.mul, ta, ts))
        sb = wp.tile_sum(wp.tile_map(wp.mul, tb, ts))
        part[i] = sa[0]
        part[NB + i] = sb[0]

    @wp.kernel(module="unique")
    def k_fin_mass(part: wp.array(dtype=float), scal: wp.array(dtype=float)):
        ta = wp.tile_load(part, shape=NB, offset=0)
        tb = wp.tile_load(part, shape=NB, offset=NB)
        a = wp.tile_sum(ta)[0]
        b = wp.tile_sum(tb)[0]
        ratio = float(1.0)
        if a > 0.0 and b > 1.0e-30:
            ratio = a / b
        scal[K.S_MASS_RATIO] = ratio

    return NB, k_part_mass, k_fin_mass


class Solver:
    GRAPH_CACHE = 4
    HOST_RING = 4

    def __init__(self, res: int, config: SolverConfig | None = None, device="cuda"):
        wp.init()
        self.res = int(res)
        self.config = config if config is not None else SolverConfig()
        self.device = str(device)
        if self.device == "cuda":
            self.device = f"cuda:{torch.cuda.current_device()}"
        self.wdev = wp.get_device(self.device)
        self.dx = DOMAIN_SIZE / self.res
        N = self.res
        z = lambda shape: wp.zeros(shape, dtype=float, device=self.wdev)
        self.hat = z((NUM_CHANNELS, N, N))
        self.adv = z((NUM_CHANNELS, N, N))
        self.omega = z((N, N))
        self.prm = z(K.P_SIZE)
        self.scal = z(K.S_SIZE)
        self.pressure = PressureSolver(N, N, self.wdev)
        nb, self.k_part_mass, self.k_fin_mass = _make_mass_kernels(N, N)
        self.mass_part = z(2 * nb)
        self._mass_nb = nb
        self._host = [wp.zeros(K.P_SIZE, dtype=float, device="cpu", pinned=True) for _ in range(self.HOST_RING)]
        self._host_ev = [None] * self.HOST_RING
        self._host_i = 0
        self._graphs: OrderedDict = OrderedDict()
        self._wrap_cache: dict = {}
        self._cap_stream = wp.Stream(self.wdev)
        self._solid_last = None

    # ------------------------------------------------------------------ public API
    def reset(self, obstacles=()) -> FieldState:
        N = self.res
        solid = rasterize_obstacles(list(obstacles), N, device=self.device)
        data = torch.zeros(NUM_CHANNELS, N, N, device=self.device, dtype=torch.float32)
        data[CH["temperature"]] = AMBIENT_TEMP
        meta = FieldMeta.for_res(N)
        meta.solver_version = SOLVER_VERSION
        return FieldState(data=data, solid=solid, meta=meta)

    def step(self, state: FieldState, forces: Forces, dt: float) -> FieldState:
        self._check(state)
        stream = self._torch_stream()
        with wp.ScopedStream(stream, sync_enter=False):
            self._upload(forces, dt, stream)
            g = self._graph("step", state)
            wp.capture_launch(g, stream=stream)
        state.meta.step += 1
        state.meta.dt = float(dt)
        state.meta.solver_version = SOLVER_VERSION
        return state

    def project(self, state: FieldState) -> FieldState:
        self._check(state)
        stream = self._torch_stream()
        with wp.ScopedStream(stream, sync_enter=False):
            self._upload(Forces(), 0.0, stream)
            g = self._graph("project", state)
            wp.capture_launch(g, stream=stream)
        return state

    def divergence(self, state: FieldState) -> torch.Tensor:
        """(H,W) |div u| * dx / (max|u| + 1e-8), fluid cells only (0 in solids).

        div u is the face-averaged central divergence the projection makes zero:
        face velocity = mean of the two adjacent cells, 0 on walls and solid faces."""
        self._check(state)
        out = torch.zeros(self.res, self.res, device=self.device, dtype=torch.float32)
        stream = self._torch_stream()
        with wp.ScopedStream(stream, sync_enter=False):
            self.pressure.record_div(self._wrap(state.data), CH["vel_x"], self._wrap(state.solid),
                                     wp.from_torch(out))
        fluid = state.solid < 0.5
        speed = torch.sqrt(state.data[CH["vel_x"]] ** 2 + state.data[CH["vel_y"]] ** 2)
        umax = torch.where(fluid, speed, torch.zeros_like(speed)).max()
        return out / (umax + 1e-8)

    def advect(self, field: torch.Tensor, velocity: torch.Tensor, dt: float) -> torch.Tensor:
        """MacCormack advection (the scheme step() uses, without the density mass correction)."""
        squeeze = field.dim() == 2
        f = field.unsqueeze(0) if squeeze else field
        assert f.shape[-2:] == (self.res, self.res) and velocity.shape == (2, self.res, self.res)
        f = f.contiguous().float()
        v = velocity.contiguous().float()
        hat = torch.empty_like(f)
        out = torch.empty_like(f)
        prm = torch.zeros(K.P_SIZE, device=self.device, dtype=torch.float32)
        prm[K.P_DT] = float(dt)
        stream = self._torch_stream()
        with wp.ScopedStream(stream, sync_enter=False):
            wf, wv, wh, wo, wpm = (wp.from_torch(t) for t in (f, v, hat, out, prm))
            dims = (self.res, self.res)
            wp.launch(K.advect_sl, dim=dims, inputs=[wf, wv, 0, wpm, 1.0 / self.dx, wh], device=self.wdev)
            wp.launch(K.advect_mc, dim=dims, inputs=[wf, wv, 0, wpm, 1.0 / self.dx, wh, wo], device=self.wdev)
        return out[0] if squeeze else out

    # ------------------------------------------------------------------ internals
    def _check(self, state: FieldState):
        N = self.res
        assert state.data.shape == (NUM_CHANNELS, N, N) and state.data.dtype == torch.float32
        assert state.solid.shape == (N, N) and state.solid.dtype == torch.float32
        assert state.data.is_cuda and state.data.is_contiguous() and state.solid.is_contiguous()

    def _torch_stream(self):
        return wp.stream_from_torch(torch.cuda.current_stream(torch.device(self.device)))

    def _wrap(self, t: torch.Tensor):
        key = (t.data_ptr(), tuple(t.shape))
        a = self._wrap_cache.get(key)
        if a is None:
            if len(self._wrap_cache) > 32:
                self._wrap_cache.clear()
            a = wp.from_torch(t)
            self._wrap_cache[key] = (a, t)  # keep tensor alive while cached
            return a
        return a[0]

    def _upload(self, forces: Forces, dt: float, stream):
        i = self._host_i
        self._host_i = (i + 1) % self.HOST_RING
        if self._host_ev[i] is not None:
            wp.synchronize_event(self._host_ev[i])  # only blocks if the GPU is HOST_RING steps behind
        h = self._host[i].numpy()
        h[:] = 0.0
        h[K.P_DT] = dt
        h[K.P_WIND_X], h[K.P_WIND_Y] = forces.wind
        h[K.P_DX] = self.dx
        h[K.P_AMB] = AMBIENT_TEMP
        cfg = self.config
        for name, slot in _CFG_SLOTS.items():
            h[slot] = float(getattr(cfg, name))
        if len(forces.emitters) > K.MAX_EMITTERS or len(forces.impulses) > K.MAX_IMPULSES:
            raise ValueError(f"at most {K.MAX_EMITTERS} emitters / {K.MAX_IMPULSES} impulses")
        h[K.P_N_EM] = len(forces.emitters)
        for j, e in enumerate(forces.emitters):
            b = K.P_EM0 + j * K.EM_STRIDE
            h[b:b + K.EM_STRIDE] = (e.pos[0], e.pos[1], max(e.radius, 1e-6), e.density_rate,
                                    e.temperature, e.fuel_rate, e.velocity[0], e.velocity[1])
        h[K.P_N_IMP] = len(forces.impulses)
        for j, m in enumerate(forces.impulses):
            b = K.P_IMP0 + j * K.IMP_STRIDE
            h[b:b + K.IMP_STRIDE] = (m.pos[0], m.pos[1], max(m.radius, 1e-6), m.velocity[0], m.velocity[1])
        wp.copy(self.prm, self._host[i], stream=stream)
        self._host_ev[i] = stream.record_event()

    def _graph(self, kind: str, state: FieldState):
        key = (kind, state.data.data_ptr(), state.solid.data_ptr())
        g = self._graphs.get(key)
        if g is not None:
            self._graphs.move_to_end(key)
            return g
        data = self._wrap(state.data)
        solid = self._wrap(state.solid)
        # capture on a side stream (legacy default stream cannot be captured); launch anywhere.
        self._cap_stream.wait_stream(wp.get_stream(self.wdev))
        with wp.ScopedStream(self._cap_stream, sync_enter=False):
            wp.capture_begin(stream=self._cap_stream, force_module_load=True)
            try:
                if kind == "step":
                    self._record_step(data, solid)
                else:
                    self.pressure.record(data, K.C_VX, solid, self.prm, self.scal)
            finally:
                g = wp.capture_end(stream=self._cap_stream)
        self._graphs[key] = g
        while len(self._graphs) > self.GRAPH_CACHE:
            self._graphs.popitem(last=False)
        return g

    def _record_step(self, data, solid):
        N, d = self.res, self.wdev
        dims = (N, N)
        inv_dx = 1.0 / self.dx
        wp.launch(K.advect_sl, dim=dims, inputs=[data, data, K.C_VX, self.prm, inv_dx, self.hat], device=d)
        wp.launch(K.advect_mc, dim=dims, inputs=[data, data, K.C_VX, self.prm, inv_dx, self.hat, self.adv], device=d)
        wp.launch_tiled(self.k_part_mass, dim=[self._mass_nb], inputs=[data, self.adv, solid, self.mass_part],
                        block_dim=256, device=d)
        wp.launch_tiled(self.k_fin_mass, dim=[1], inputs=[self.mass_part, self.scal], block_dim=256, device=d)
        wp.launch(K.sources, dim=dims, inputs=[self.adv, solid, self.prm, self.scal, data], device=d)
        wp.launch(K.curl, dim=dims, inputs=[data, solid, 0.5 * inv_dx, self.omega], device=d)
        wp.launch(K.body_forces, dim=dims, inputs=[data, solid, self.omega, self.prm], device=d)
        self.pressure.record(data, K.C_VX, solid, self.prm, self.scal)
