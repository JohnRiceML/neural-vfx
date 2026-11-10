"""3D smoke + fire Eulerian solver (Warp), collocated grid, world units. Port of vfx/sim (0.2.0).

Step order (output velocity is divergence-free):
  1. advect density, velocity, temperature, fuel by u^n    (MacCormack, RK2 traces, clamped)
  2. density mass correction (global, deterministic tile reduction)
  3. emitters, impulses, combustion, cooling, dissipation; heat + fuel diffusion
  4. buoyancy, vorticity confinement (smoothed vorticity), wind; force applied [1,2,1]^3-smoothed
  5. pressure projection (MG-PCG, exact for the face-averaged central divergence)

The whole step is one CUDA graph per (state.data, state.solid) buffer pair; per-step inputs
(dt, forces, config) are uploaded to a small device buffer first.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
import warp as wp

from vfx.core.fields import AMBIENT_TEMP, DOMAIN_SIZE
from vfx.core.fields3d import (CH3, NUM_CHANNELS_3D, FieldMeta3D, FieldState3D, Forces3D, Volume,
                               rasterize_obstacles_3d)

from . import hires as HR
from . import kernels as K
from .config import SolverConfig3D, solver_version_tag
from .pressure import PressureSolver3D
from .pressure_fused import FusedPressure3D

_CFG_SLOTS = {
    "buoyancy_alpha": K.P_ALPHA, "buoyancy_beta": K.P_BETA, "vorticity_eps": K.P_EPS,
    "density_dissipation": K.P_DDISS, "temperature_cooling": K.P_COOL, "burn_rate": K.P_BURN,
    "ignition_temp": K.P_IGN, "burn_heat": K.P_BHEAT, "smoke_from_fuel": K.P_SMOKE,
    "pressure_tol": K.P_TOL, "pressure_max_iters": K.P_MAXIT, "max_temperature": K.P_TMAX,
    "wind_marker_temp": K.P_WMARK, "heat_diffusion": K.P_KAPPA, "confinement_floor": K.P_CFLOOR,
    "pressure_warm_start": K.P_WARM,
}


def _make_mass_kernels(N: int, block: int):
    total = N * N * N
    cpt = max(1, -(-total // (512 * block)))
    per_block = cpt * block
    NB = -(-total // per_block)

    @wp.kernel(module="unique")
    def k_part_mass(pre: wp.array4d(dtype=float), post: wp.array4d(dtype=float),
                    solid: wp.array3d(dtype=float), part: wp.array(dtype=float)):
        b, l = wp.tid()
        sa = float(0.0)
        sb = float(0.0)
        for k in range(cpt):
            i = b * per_block + k * block + l
            if i < total:
                z = i // (N * N)
                y = (i // N) % N
                x = i % N
                if solid[z, y, x] <= 0.5:
                    sa += pre[0, z, y, x]
                    sb += post[0, z, y, x]
        ta = wp.tile_sum(wp.tile(sa))
        tb = wp.tile_sum(wp.tile(sb))
        if l == 0:
            part[b] = ta[0]
            part[NB + b] = tb[0]

    @wp.kernel(module="unique")
    def k_fin_mass(part: wp.array(dtype=float), scal: wp.array(dtype=float)):
        a = wp.tile_sum(wp.tile_load(part, shape=NB, offset=0))[0]
        b = wp.tile_sum(wp.tile_load(part, shape=NB, offset=NB))[0]
        ratio = float(1.0)
        if a > 0.0 and b > 1.0e-30:
            ratio = a / b
        scal[K.S_MASS_RATIO] = ratio

    return NB, k_part_mass, k_fin_mass


class Solver3D:
    GRAPH_CACHE = 4
    HOST_RING = 4
    BLOCK = 256
    SEPARABLE_SMOOTH = True # [1,2,1]^3 smoothing as 3 separable passes (identical result)
    FAST_VCYCLE = True      # False = plain V-cycle (ablation / reference)

    def __init__(self, res: int, config: SolverConfig3D | None = None, device="cuda", scalar_res: int | None = None):
        """scalar_res (opt-in, e.g. 128): density / temperature / fuel are also carried on a finer
        scalar_res^3 grid by the res^3 velocity and restricted back each step (see hires.py)."""
        wp.init()
        self.res = int(res)
        self.config = config if config is not None else SolverConfig3D()
        self.device = str(device)
        if self.device == "cuda":
            self.device = f"cuda:{torch.cuda.current_device()}"
        self.wdev = wp.get_device(self.device)
        self.dx = DOMAIN_SIZE / self.res
        N = self.res
        z = lambda shape: wp.zeros(shape, dtype=float, device=self.wdev)
        self.hat = z((NUM_CHANNELS_3D, N, N, N))
        self.adv = z((NUM_CHANNELS_3D, N, N, N))
        self.omega = z((3, N, N, N))
        self.omega_s = z((3, N, N, N))
        self.force = z((3, N, N, N))
        # scratch for the separable smoothing: views into adv / hat, both free at that point
        self.tmp_a = self.adv[0:4]
        self.tmp_b = self.hat[0:4]
        # explicit diffusion substeps: lambda = kappa dt / dx^2 per substep is capped at 0.16;
        # one substep covers kappa <= 0.16 * 256^2 * dt at 256^3 (7e-5 m^2/s at 30 Hz).
        self.n_diff = max(1, -(-N // 256))
        self.prm = z(K.P_SIZE)
        self.scal = z(K.S_SIZE)
        self.pressure = PressureSolver3D(N, self.wdev, fast=self.FAST_VCYCLE)
        nb, self.k_part_mass, self.k_fin_mass = _make_mass_kernels(N, self.BLOCK)
        self.mass_part = z(2 * nb)
        self._mass_nb = nb
        self._host = [wp.zeros(K.P_SIZE, dtype=float, device="cpu", pinned=True) for _ in range(self.HOST_RING)]
        self._host_ev = [None] * self.HOST_RING
        self._host_i = 0
        self._graphs: OrderedDict = OrderedDict()
        self._wrap_cache: dict = {}
        self._cap_stream = wp.Stream(self.wdev)
        self.pressure_fused: FusedPressure3D | None = None    # built on first use of pressure_mode "fused"
        self.scalar_res = None if scalar_res is None else int(scalar_res)
        if self.scalar_res is not None:
            self._init_hires()

    @property
    def version(self) -> str:
        """SOLVER3D_VERSION for the default path, plus a suffix per opt-in option in use."""
        return solver_version_tag(self.config, self.scalar_res)

    # ------------------------------------------------------------------ public API
    def reset(self, obstacles=()) -> FieldState3D:
        N = self.res
        solid = rasterize_obstacles_3d(list(obstacles), N, device=self.device)
        data = torch.zeros(NUM_CHANNELS_3D, N, N, N, device=self.device, dtype=torch.float32)
        data[CH3["temperature"]] = AMBIENT_TEMP
        meta = FieldMeta3D.for_res(N)
        meta.solver_version = self.version
        if self.scalar_res is not None:
            self.set_hires_obstacles(obstacles)
            self.hi.zero_()
            self.hi[HR.H_T] = AMBIENT_TEMP
        return FieldState3D(data=data, solid=solid, meta=meta)

    def step(self, state: FieldState3D, forces: Forces3D, dt: float) -> FieldState3D:
        self._check(state)
        stream = self._torch_stream()
        with wp.ScopedStream(stream, sync_enter=False):
            self._upload(forces, dt, stream)
            wp.capture_launch(self._graph("step", state), stream=stream)
        state.meta.step += 1
        state.meta.dt = float(dt)
        state.meta.solver_version = self.version
        return state

    def step_profiled(self, state: FieldState3D, forces: Forces3D, dt: float) -> dict:
        """Same step as step(), launched as three graphs with CUDA events in between (four with
        scalar_res: the fine-scalar stage is timed separately as `scalars`).
        Returns ms per stage {advection, [scalars,] forces, projection, total} (synchronizes)."""
        self._check(state)
        stream = self._torch_stream()
        ts = torch.cuda.current_stream(torch.device(self.device))
        kinds = ("s_adv", "s_forces", "project") if self.scalar_res is None else ("s_adv", "s_hires", "s_forces", "project")
        names = [{"s_adv": "advection", "s_hires": "scalars", "s_forces": "forces", "project": "projection"}[k]
                 for k in kinds]
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(len(kinds) + 1)]
        with wp.ScopedStream(stream, sync_enter=False):
            self._upload(forces, dt, stream)
            graphs = [self._graph(k, state) for k in kinds]
            # keep the GPU busy while the host enqueues, so no stage is charged for launch latency
            torch.cuda._sleep(3_000_000)
            ev[0].record(ts)
            for g, e in zip(graphs, ev[1:]):
                wp.capture_launch(g, stream=stream)
                e.record(ts)
        ev[-1].synchronize()
        state.meta.step += 1
        state.meta.dt = float(dt)
        state.meta.solver_version = self.version
        t = [ev[i].elapsed_time(ev[i + 1]) for i in range(len(kinds))]
        out = dict(zip(names, t))
        out["total"] = sum(t)
        return out

    # ------------------------------------------------------------------ finer scalar grid (opt-in)
    def hires_volume(self) -> Volume | None:
        """The fine density + temperature (views of the solver's buffer, no copy; valid until the next
        step), or None without scalar_res."""
        if self.scalar_res is None:
            return None
        return Volume(density=self.hi[HR.H_D], temperature=self.hi[HR.H_T])

    def hires_fields(self) -> torch.Tensor | None:
        """(3, S, S, S) density, temperature, fuel (the live buffer), or None."""
        return None if self.scalar_res is None else self.hi

    def set_hires_obstacles(self, obstacles=()):
        """Rasterise obstacles at scalar_res (reset() does this). Call it whenever state.solid is changed
        by hand, with the same obstacles: the fine grid does not read state.solid."""
        if self.scalar_res is not None:
            self.hi_solid.copy_(rasterize_obstacles_3d(list(obstacles), self.scalar_res, device=self.device))

    def init_hires_from_state(self, state: FieldState3D):
        """Fill the fine scalars from the coarse ones (trilinear, cell centred; 0 / ambient in fine solids).
        For states built by hand; reset() already starts both grids empty."""
        if self.scalar_res is None:
            return
        import torch.nn.functional as F
        src = state.data[[CH3["density"], CH3["temperature"], CH3["fuel"]]][None]
        up = F.interpolate(src, size=(self.scalar_res,) * 3, mode="trilinear", align_corners=False)[0]
        fl = self.hi_solid < 0.5
        self.hi[HR.H_D] = up[0] * fl
        self.hi[HR.H_T] = torch.where(fl, up[1], torch.full_like(up[1], AMBIENT_TEMP))
        self.hi[HR.H_F] = up[2] * fl

    def _init_hires(self):
        S, dev = self.scalar_res, self.device
        if S < self.res:
            raise ValueError("scalar_res must be >= res")
        self.dxf = DOMAIN_SIZE / S
        self.hi = torch.zeros(3, S, S, S, device=dev, dtype=torch.float32)
        self.hi[HR.H_T] = AMBIENT_TEMP
        self.hi_solid = torch.zeros(S, S, S, device=dev, dtype=torch.float32)
        z = lambda shape: wp.zeros(shape, dtype=float, device=self.wdev)
        self.hi_hat = z((3, S, S, S))
        self.hi_adv = z((3, S, S, S))
        self.hi_scal = z(K.S_SIZE)
        self.hi_wp = wp.from_torch(self.hi)
        self.hi_solid_wp = wp.from_torch(self.hi_solid)
        nb, self.k_part_mass_f, self.k_fin_mass_f = _make_mass_kernels(S, self.BLOCK)
        self.hi_mass_part = z(2 * nb)
        self._hi_mass_nb = nb
        self.n_diff_f = max(1, -(-S // 256))
        self.hi_ratio = S / self.res                       # fine cells per coarse cell
        self.hi_span = int(-(-S // self.res)) + 1          # fine cells a coarse cell can overlap per axis
        # fine index -> coarse index: pc = a pf + b
        self.hi_a = self.res / S
        self.hi_b = 0.5 * self.res / S - 0.5

    def _record_hires(self, data, solid):
        S, d = self.scalar_res, self.wdev
        dims = (S, S, S)
        inv_dx = 1.0 / self.dx
        hi, fs = self.hi_wp, self.hi_solid_wp
        wp.launch(HR.advect_sl_fine, dim=dims, inputs=[hi, data, K.C_VX, self.prm, inv_dx, self.hi_a, self.hi_b,
                                                         self.hi_hat], device=d)
        wp.launch(HR.advect_mc_fine, dim=dims, inputs=[hi, data, K.C_VX, self.prm, inv_dx, self.hi_a, self.hi_b,
                                                         self.hi_hat, self.hi_adv], device=d)
        wp.launch_tiled(self.k_part_mass_f, dim=[self._hi_mass_nb], inputs=[hi, self.hi_adv, fs, self.hi_mass_part],
                        block_dim=self.BLOCK, device=d)
        wp.launch_tiled(self.k_fin_mass_f, dim=[1], inputs=[self.hi_mass_part, self.hi_scal], block_dim=self.BLOCK,
                        device=d)
        nd = self.n_diff_f
        bufs = [hi, self.hi_hat] if nd % 2 == 0 else [self.hi_hat, hi]
        wp.launch(HR.sources_fine, dim=dims, inputs=[self.hi_adv, fs, self.prm, self.hi_scal, self.dxf, bufs[0]], device=d)
        for k in range(nd):
            wp.launch(HR.diffuse_fine, dim=dims, inputs=[bufs[k % 2], fs, self.prm, float(nd), self.dxf,
                                                          bufs[(k + 1) % 2]], device=d)

    def _record_restrict(self, data, solid):
        N = self.res
        wp.launch(HR.restrict_scalars, dim=(N, N, N), inputs=[self.hi_wp, self.hi_solid_wp, solid, self.prm,
                                                               self.hi_ratio, self.hi_span, data], device=self.wdev)

    def project(self, state: FieldState3D) -> FieldState3D:
        self._check(state)
        stream = self._torch_stream()
        with wp.ScopedStream(stream, sync_enter=False):
            self._upload(Forces3D(), 0.0, stream)
            wp.capture_launch(self._graph("project", state), stream=stream)
        return state

    def divergence(self, state: FieldState3D) -> torch.Tensor:
        """(D,H,W) |div u| * dx / (max|u| + 1e-8), fluid cells only (0 in solids). div u is the
        face-averaged central divergence (face velocity = mean of the two adjacent cells, 0 on
        walls / solid faces), the quantity the projection makes zero."""
        self._check(state)
        N = self.res
        out = torch.zeros(N, N, N, device=self.device, dtype=torch.float32)
        stream = self._torch_stream()
        with wp.ScopedStream(stream, sync_enter=False):
            self.pressure.record_div(self._wrap(state.data), K.C_VX, self._wrap(state.solid), wp.from_torch(out))
        fluid = state.solid < 0.5
        speed = state.velocity.norm(dim=0)
        umax = torch.where(fluid, speed, torch.zeros_like(speed)).max()
        return out / (umax + 1e-8)

    def advect(self, field: torch.Tensor, velocity: torch.Tensor, dt: float) -> torch.Tensor:
        """MacCormack advection (the scheme step() uses, without the density mass correction).
        field (D,H,W) or (C,D,H,W) float32 cuda, velocity (3,D,H,W) m/s. Does not see obstacles."""
        squeeze = field.dim() == 3
        f = field.unsqueeze(0) if squeeze else field
        N = self.res
        assert f.shape[-3:] == (N, N, N) and velocity.shape == (3, N, N, N)
        f = f.contiguous().float()
        v = velocity.contiguous().float()
        hat = torch.empty_like(f)
        out = torch.empty_like(f)
        prm = torch.zeros(K.P_SIZE, device=self.device, dtype=torch.float32)
        prm[K.P_DT] = float(dt)
        stream = self._torch_stream()
        with wp.ScopedStream(stream, sync_enter=False):
            wf, wv, wh, wo, wpm = (wp.from_torch(t) for t in (f, v, hat, out, prm))
            dims = (N, N, N)
            wp.launch(K.advect_sl, dim=dims, inputs=[wf, wv, 0, wpm, 1.0 / self.dx, wh], device=self.wdev)
            wp.launch(K.advect_mc, dim=dims, inputs=[wf, wv, 0, wpm, 1.0 / self.dx, wh, wo], device=self.wdev)
        return out[0] if squeeze else out

    # ------------------------------------------------------------------ internals
    def _check(self, state: FieldState3D):
        N = self.res
        assert state.data.shape == (NUM_CHANNELS_3D, N, N, N) and state.data.dtype == torch.float32
        assert state.solid.shape == (N, N, N) and state.solid.dtype == torch.float32
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
            self._wrap_cache[key] = (a, t)
            return a
        return a[0]

    def _upload(self, forces: Forces3D, dt: float, stream):
        i = self._host_i
        self._host_i = (i + 1) % self.HOST_RING
        if self._host_ev[i] is not None:
            wp.synchronize_event(self._host_ev[i])  # only blocks if the GPU is HOST_RING steps behind
        h = self._host[i].numpy()
        h[:] = 0.0
        h[K.P_DT] = dt
        h[K.P_WIND_X], h[K.P_WIND_Y], h[K.P_WIND_Z] = forces.wind
        h[K.P_DX] = self.dx
        h[K.P_AMB] = AMBIENT_TEMP
        cfg = self.config
        for name, slot in _CFG_SLOTS.items():
            h[slot] = float(getattr(cfg, name))
        if cfg.confinement_ref_res:      # opt-in: same eps * dx as a confinement_ref_res^3 run
            h[K.P_EPS] = float(cfg.vorticity_eps) * self.res / float(cfg.confinement_ref_res)
        if len(forces.emitters) > K.MAX_EMITTERS or len(forces.impulses) > K.MAX_IMPULSES:
            raise ValueError(f"at most {K.MAX_EMITTERS} emitters / {K.MAX_IMPULSES} impulses")
        h[K.P_N_EM] = len(forces.emitters)
        for j, e in enumerate(forces.emitters):
            b = K.P_EM0 + j * K.EM_STRIDE
            h[b:b + K.EM_STRIDE] = (*e.pos, max(e.radius, 1e-6), e.density_rate, e.temperature, e.fuel_rate,
                                    *e.velocity)
        h[K.P_N_IMP] = len(forces.impulses)
        for j, m in enumerate(forces.impulses):
            b = K.P_IMP0 + j * K.IMP_STRIDE
            h[b:b + K.IMP_STRIDE] = (*m.pos, max(m.radius, 1e-6), *m.velocity)
        wp.copy(self.prm, self._host[i], stream=stream)
        self._host_ev[i] = stream.record_event()

    def _structure(self) -> tuple:
        """Options that change the recorded launch sequence (part of the graph key). Empty for the
        default path."""
        c = self.config
        if c.pressure_mode == "pcg":
            return ()
        if c.pressure_mode != "fused":
            raise ValueError(f"pressure_mode must be 'pcg' or 'fused', got {c.pressure_mode!r}")
        return ("fused", int(c.pressure_cycles), int(c.pressure_smooth), str(c.pressure_smoother),
                c.pressure_tol > 0, int(c.pressure_max_iters))

    def _record_pressure(self, data, solid):
        c = self.config
        if c.pressure_mode == "pcg":
            self.pressure.record(data, K.C_VX, solid, self.prm, self.scal)
            return
        if self.pressure_fused is None:
            self.pressure_fused = FusedPressure3D(self.pressure)
        self.pressure_fused.record(data, K.C_VX, solid, self.prm, int(c.pressure_cycles), int(c.pressure_smooth),
                                   str(c.pressure_smoother), adaptive=c.pressure_tol > 0,
                                   max_cycles=int(c.pressure_max_iters))

    def pressure_iters(self) -> int:
        """PCG iterations (default) or V-cycles (fused) of the last projection (synchronizes)."""
        p = self.pressure_fused if self.config.pressure_mode == "fused" and self.pressure_fused else self.pressure
        return int(p.it.numpy()[0])

    def _graph(self, kind: str, state: FieldState3D):
        key = (kind, state.data.data_ptr(), state.solid.data_ptr()) + self._structure()
        g = self._graphs.get(key)
        if g is not None:
            self._graphs.move_to_end(key)
            return g
        data = self._wrap(state.data)
        solid = self._wrap(state.solid)
        hires = self.scalar_res is not None
        self._cap_stream.wait_stream(wp.get_stream(self.wdev))
        with wp.ScopedStream(self._cap_stream, sync_enter=False):
            wp.capture_begin(stream=self._cap_stream, force_module_load=True)
            try:
                if kind == "step":
                    self._record_advect(data, solid)
                    if hires:
                        self._record_hires(data, solid)
                    self._record_forces(data, solid)
                    self._record_pressure(data, solid)
                elif kind == "s_adv":
                    self._record_advect(data, solid)
                elif kind == "s_hires":
                    self._record_hires(data, solid)
                elif kind == "s_forces":
                    self._record_forces(data, solid)
                else:
                    self._record_pressure(data, solid)
            finally:
                g = wp.capture_end(stream=self._cap_stream)
        self._graphs[key] = g
        while len(self._graphs) > self.GRAPH_CACHE:
            self._graphs.popitem(last=False)
        return g

    def _record_advect(self, data, solid):
        N, d = self.res, self.wdev
        dims = (N, N, N)
        inv_dx = 1.0 / self.dx
        wp.launch(K.advect_sl, dim=dims, inputs=[data, data, K.C_VX, self.prm, inv_dx, self.hat], device=d)
        wp.launch(K.advect_mc, dim=dims, inputs=[data, data, K.C_VX, self.prm, inv_dx, self.hat, self.adv], device=d)
        wp.launch_tiled(self.k_part_mass, dim=[self._mass_nb], inputs=[data, self.adv, solid, self.mass_part],
                        block_dim=self.BLOCK, device=d)
        wp.launch_tiled(self.k_fin_mass, dim=[1], inputs=[self.mass_part, self.scal], block_dim=self.BLOCK, device=d)

    def _record_forces(self, data, solid):
        N, d = self.res, self.wdev
        dims = (N, N, N)
        inv_dx = 1.0 / self.dx
        # sources write where the diffusion ping-pong ends back in `data`
        nd = self.n_diff
        bufs = [data, self.hat] if nd % 2 == 0 else [self.hat, data]
        wp.launch(K.sources, dim=dims, inputs=[self.adv, solid, self.prm, self.scal, bufs[0]], device=d)
        for k in range(nd):
            wp.launch(K.diffuse_heat_fuel, dim=dims, inputs=[bufs[k % 2], solid, self.prm, float(nd), bufs[(k + 1) % 2]],
                      device=d)
        if self.scalar_res is not None:
            # the coarse scalars the forces read = the fine scalars, box-restricted
            self._record_restrict(data, solid)
        wp.launch(K.curl, dim=dims, inputs=[data, solid, 0.5 * inv_dx, self.omega], device=d)
        if self.SEPARABLE_SMOOTH:
            ta, tb = self.tmp_a, self.tmp_b
            wp.launch(K.sep_x, dim=dims, inputs=[self.omega, 0, solid, ta], device=d)
            wp.launch(K.sep_y, dim=dims, inputs=[ta, tb], device=d)
            wp.launch(K.sep_z_store, dim=dims, inputs=[tb, self.omega_s], device=d)
            wp.launch(K.body_forces, dim=dims, inputs=[data, solid, self.omega_s, self.prm, self.force], device=d)
            wp.launch(K.sep_x, dim=dims, inputs=[self.force, 0, solid, ta], device=d)
            wp.launch(K.sep_y, dim=dims, inputs=[ta, tb], device=d)
            wp.launch(K.sep_z_apply, dim=dims, inputs=[tb, data, solid, self.prm], device=d)
        else:
            wp.launch(K.smooth_curl, dim=dims, inputs=[self.omega, solid, self.omega_s], device=d)
            wp.launch(K.body_forces, dim=dims, inputs=[data, solid, self.omega_s, self.prm, self.force], device=d)
            wp.launch(K.apply_forces, dim=dims, inputs=[data, solid, self.force, self.prm], device=d)
