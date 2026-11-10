"""DetailParticles: sub-grid smoke detail carried on Lagrangian particles advected by the 64^3 velocity and
deposited into an out_res^3 `Volume` every sim tick (see README.md)."""
from __future__ import annotations

import math

import numpy as np
import torch
import warp as wp

from vfx.core.fields3d import AMBIENT_TEMP, DOMAIN_SIZE, FieldState3D, Forces3D, Volume

from . import kernels as K

DEFAULTS = dict(
    sg_amp=0.6,          # sub-grid speed = sg_amp * |curl u|_64 * dx_64 (Smagorinsky-style velocity scale)
    sg_freq=None,        # octave-0 noise frequency (1/m); None -> 1 / (2 dx_64) (finest unresolved scale)
    sg_oct=3,
    sg_lac=2.0,
    sg_gain=2.0 ** (-1.0 / 3.0),   # Kolmogorov: velocity at scale l ~ l^(1/3)
    sg_tfreq=1.5,        # noise evolution in time (1/s)
    sg_max=0.6,          # m/s cap on the sub-grid speed
    rk3=False,           # RK2 midpoint (default) or Ralston RK3
    life_eps=0.02,       # ring lifetime: until smoke mass decays to this fraction (dissipation)
    max_life=6.0,        # s, cap on the ring lifetime (weak dissipation)
    n_full=4.0,          # particles per fine cell (restricted to 64^3) for full particle weight
    strength=1.0,        # weight of the particle modulation (0 = plain trilinear upsample of the solver fields)
    mod_max=8.0,         # cap on the per-cell modulation P / up(R(P))
    heat_strength=0.5,   # weight of the particle modulation of temperature (emission is very nonlinear in T)
    heat_mod_max=2.0,    # cap on the temperature modulation H / up(R(H))
    blur=0,              # [1,2,1]^3 passes on the fine deposit before composing
    fused=True,          # deposit inside the advect kernel (fast); False = separate kernels (stage timing)
    seed=1,
    vel_res=128,         # grid the per-tick particle velocity (coarse + sub-grid noise) is evaluated on
)


def _emitter_integral(pos, radius, res) -> float:
    """Discrete sum of the solver's emitter falloff (1-r^2/R^2)^2 over cell centres * dx^3 (what the grid injects)."""
    dx = DOMAIN_SIZE / res
    lo = [max(0, int((p - radius) / dx) - 1) for p in pos]
    hi = [min(res, int((p + radius) / dx) + 2) for p in pos]
    ax = [(np.arange(l, h) + 0.5) * dx - p for l, h, p in zip(lo, hi, pos)]
    if any(len(a) == 0 for a in ax):
        return 0.0
    x, y, z = np.meshgrid(ax[0], ax[1], ax[2], indexing="ij")
    r2 = (x * x + y * y + z * z) / (radius * radius)
    w = np.where(r2 < 1.0, (1.0 - r2) ** 2, 0.0)
    return float(w.sum() * dx ** 3)


class DetailParticles:
    def __init__(self, count: int, out_res: int = 128, device="cuda", config=None, **params):
        from vfx.sim3d import SolverConfig3D
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"unknown params {sorted(unknown)}")
        self.p = dict(DEFAULTS, **params)
        self.cfg = config if config is not None else SolverConfig3D()
        self.n = int(count)
        self.out_res = int(out_res)
        self.device = device
        self.timings: dict[str, float] = {}
        self._events: list = []
        self._wp_dev = wp.device_from_torch(torch.empty(0, device=device).device)
        n, nf = self.n, self.out_res
        self.pos = wp.zeros(n, dtype=wp.vec3, device=self._wp_dev)
        self.q = wp.zeros(n, dtype=wp.vec4, device=self._wp_dev)
        self.dep = torch.zeros(3, nf, nf, nf, device=device)            # mass, heat, count
        self.dep_tmp = torch.zeros(nf, nf, nf, device=device)
        self.dens = torch.zeros(nf, nf, nf, device=device)
        self.temp = torch.full((nf, nf, nf), AMBIENT_TEMP, device=device)
        self.em_host = [torch.zeros(K.MAX_EMITTERS * K.EM_STRIDE).pin_memory() for _ in range(4)]
        self.em_dev = torch.zeros(K.MAX_EMITTERS * K.EM_STRIDE, device=device)
        self._coarse = None
        self._wmag = None
        self._res = None
        self._em_cache = {}
        self._rms = {}
        self.reset()

    # ------------------------------------------------------------------ public
    def reset(self) -> None:
        self.q.zero_()
        self.pos.zero_()
        self.head = 0
        self.tick = 0
        self.time = 0.0
        self.dens.zero_()
        self.temp.fill_(AMBIENT_TEMP)

    def volume(self) -> Volume:
        return Volume(self.dens, self.temp)

    def k_new(self, dt: float) -> int:
        life = min(self.p["max_life"], math.log(1.0 / self.p["life_eps"]) / max(self.cfg.density_dissipation, 1e-6))
        return max(1, min(self.n, int(math.ceil(self.n / max(1.0, life / dt)))))

    def step(self, state: FieldState3D, forces: Forces3D, dt: float) -> None:
        # every launch goes on torch's current stream, so torch ops (zeroing, the renderer) are ordered with ours
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.dens.device))):
            self._step(state, forces, dt)

    def _step(self, state: FieldState3D, forces: Forces3D, dt: float) -> None:
        res = state.data.shape[-1]
        if self._res != res:
            self._alloc_coarse(res)
        data = wp.from_torch(state.data)
        solid = wp.from_torch(state.solid)
        nf = self.out_res
        f = nf // res
        if f * res != nf:
            raise ValueError(f"out_res {nf} must be a multiple of the sim res {res}")
        sp = self._params(res, dt)
        n_em, k_new = self._upload(forces, res, dt)
        em = wp.from_torch(self.em_dev)
        dep_m, dep_h, dep_n = (wp.from_torch(self.dep[i]) for i in range(3))
        inv_fdx = nf / DOMAIN_SIZE
        fused = 1 if self.p["fused"] else 0
        ev = self._ev
        e0 = ev()
        self.dep.zero_()
        wp.launch(K.vort_mag, dim=(res, res, res), inputs=[data, res, 0.5 / sp.dx, self._wmag], device=self._wp_dev)
        wp.launch(K.vel_field, dim=(sp.vres,) * 3, inputs=[data, self._wmag, sp, self._vel], device=self._wp_dev)
        wp.launch(K.update, dim=self.n, inputs=[self.pos, self.q, self.n, self.head, k_new, self._vel, solid,
                                               em, n_em, sp, self._m_eps, inv_fdx, nf, dep_m, dep_h, dep_n, fused],
                  device=self._wp_dev)
        if k_new > 0:
            wp.launch(K.seed, dim=k_new, inputs=[self.pos, self.q, self.n, self.head, k_new, data, self._vel, solid,
                                                em, n_em, sp, self.p["seed"] * 7919 + self.tick, inv_fdx, nf,
                                                dep_m, dep_h, dep_n, fused], device=self._wp_dev)
        e1 = ev()
        if not fused:
            wp.launch(K.deposit, dim=self.n, inputs=[self.pos, self.q, sp.amb, inv_fdx, nf, dep_m, dep_h, dep_n],
                      device=self._wp_dev)
        e2 = ev()
        tmp = wp.from_torch(self.dep_tmp)
        for _ in range(int(self.p["blur"])):
            for c in range(3):
                src = wp.from_torch(self.dep[c])
                for ax in range(3):
                    wp.launch(K.blur_axis, dim=(nf, nf, nf), inputs=[src, tmp, ax, nf], device=self._wp_dev)
                    wp.copy(src, tmp)
        wp.launch(K.restrict, dim=(res, res, res), inputs=[dep_m, dep_h, dep_n, f, self._coarse], device=self._wp_dev)
        wp.launch(K.compose, dim=(nf, nf, nf),
                  inputs=[dep_m, dep_h, self._coarse, data, res, 1.0 / sp.dx, DOMAIN_SIZE / nf, sp.amb,
                          float(self.p["n_full"]), float(self.p["strength"]), float(self.p["mod_max"]),
                          float(self.p["heat_strength"]) * float(self.p["strength"]), float(self.p["heat_mod_max"]),
                          wp.from_torch(self.dens), wp.from_torch(self.temp)], device=self._wp_dev)
        e3 = ev()
        self._events = [("advect" if not fused else "advect+deposit", e0, e1), ("deposit", e1, e2),
                        ("compose", e2, e3), ("total", e0, e3)]
        self.head = (self.head + k_new) % self.n
        self.tick += 1
        self.time += dt

    @property
    def timings(self) -> dict:
        if self._events:
            self._events[-1][2].synchronize()
            self._timings = {k: a.elapsed_time(b) for k, a, b in self._events}
            self._events = []
        return getattr(self, "_timings", {})

    @timings.setter
    def timings(self, v):
        self._timings = v

    def diagnostics(self, state: FieldState3D) -> dict:
        """Host-synchronizing stats (NOT for the frame loop)."""
        q = wp.to_torch(self.q)
        alive = q[:, 3] > 0
        dx3 = (DOMAIN_SIZE / state.data.shape[-1]) ** 3
        f = self.out_res // state.data.shape[-1]
        dens_c = torch.nn.functional.avg_pool3d(self.dens[None, None], f)[0, 0]
        g = state.data[0]
        return dict(alive=int(alive.sum()), particle_mass=float(q[alive, 0].sum()),
                    particle_fuel=float(q[alive, 1].sum()), grid_mass=float(g.sum() * dx3),
                    volume_mass=float(self.dens.sum() * (DOMAIN_SIZE / self.out_res) ** 3),
                    restrict_rel_err=float((dens_c - g).abs().sum() / g.abs().sum().clamp(min=1e-12)),
                    # the same measure for the plain trilinear upsample of the solver density (strength = 0)
                    tri_restrict_rel_err=float((torch.nn.functional.avg_pool3d(torch.nn.functional.interpolate(
                        g[None, None], size=(self.out_res,) * 3, mode="trilinear", align_corners=False), f)[0, 0] - g
                    ).abs().sum() / g.abs().sum().clamp(min=1e-12)),
                    coverage_mean=float((wp.to_torch(self._coarse)[2][g > 0.02] / self.p["n_full"]).clamp(max=1).mean()) if (g > 0.02).any() else 0.0)

    # ------------------------------------------------------------------ internals
    def _ev(self):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def _alloc_coarse(self, res):
        self._res = res
        self._coarse = wp.zeros((3, res, res, res), dtype=float, device=self._wp_dev)
        self._wmag = wp.zeros((res, res, res), dtype=float, device=self._wp_dev)
        vr = int(self.p["vel_res"])
        self._vel = wp.zeros((3, vr, vr, vr), dtype=float, device=self._wp_dev)

    def _params(self, res, dt) -> K.SimParams:
        c, p = self.cfg, self.p
        sp = K.SimParams()
        sp.dt, sp.dx, sp.res, sp.amb = float(dt), DOMAIN_SIZE / res, int(res), float(AMBIENT_TEMP)
        sp.cool, sp.diss, sp.burn, sp.ign = c.temperature_cooling, c.density_dissipation, c.burn_rate, c.ignition_temp
        sp.bheat, sp.smoke, sp.tmax = c.burn_heat, c.smoke_from_fuel, c.max_temperature
        sp.sg_amp = float(p["sg_amp"])
        sp.sg_freq = float(p["sg_freq"] or res / (2.0 * DOMAIN_SIZE))
        sp.sg_oct, sp.sg_lac, sp.sg_gain = int(p["sg_oct"]), float(p["sg_lac"]), float(p["sg_gain"])
        sp.sg_tfreq, sp.sg_max = float(p["sg_tfreq"]), float(p["sg_max"])
        sp.time, sp.seed, sp.rk3 = float(self.time), int(p["seed"]), 1 if p["rk3"] else 0
        sp.vres = int(p["vel_res"])
        key = (sp.sg_oct, sp.sg_lac, sp.sg_gain, sp.seed)
        if key not in self._rms:
            sp.sg_rms = 1.0
            o = wp.zeros(65536, dtype=float, device=self._wp_dev)
            wp.launch(K.noise_rms, dim=65536, inputs=[sp, o], device=self._wp_dev)
            self._rms[key] = float(math.sqrt(o.numpy().mean()))     # one-time host read (init only)
        # Warp's curlnoise differentiates w.r.t. its own input q = p * freq, so the rms is freq independent
        sp.sg_rms = self._rms[key]
        return sp

    def _upload(self, forces: Forces3D, res: int, dt: float):
        ems = list(forces.emitters)[:K.MAX_EMITTERS]
        k_tot = self.k_new(dt)
        c = self.cfg
        integ = []
        for e in ems:
            key = (tuple(e.pos), e.radius, res)
            if key not in self._em_cache:
                self._em_cache[key] = _emitter_integral(e.pos, max(e.radius, 1e-6), res)
            integ.append(self._em_cache[key])
        rate = [(e.density_rate + c.smoke_from_fuel * e.fuel_rate) * w for e, w in zip(ems, integ)]
        tot = sum(rate)
        h = self.em_host[self.tick % 4]
        h.zero_()
        a = h.numpy()
        k0 = 0
        for i, (e, w, r) in enumerate(zip(ems, integ, rate)):
            ke = int(round(k_tot * r / tot)) if tot > 0 else 0
            if i == len(ems) - 1 and tot > 0:
                ke = k_tot - k0
            b = i * K.EM_STRIDE
            a[b:b + 12] = (*e.pos, max(e.radius, 1e-6), e.density_rate, e.temperature, e.fuel_rate, 0.0,
                           k0, k0 + ke, e.density_rate * dt * w / max(ke, 1), e.fuel_rate * dt * w / max(ke, 1))
            k0 += ke
        self.em_dev.copy_(h, non_blocking=True)
        # cull threshold relative to the per-particle seed mass (an absolute one kills small particles of large rings)
        seeds = [a[i * K.EM_STRIDE + 10] + a[i * K.EM_STRIDE + 11] for i in range(len(ems))
                 if a[i * K.EM_STRIDE + 10] + a[i * K.EM_STRIDE + 11] > 0]
        self._m_eps = 0.5 * self.p["life_eps"] * min(seeds) if seeds else 0.0
        return len(ems), (k0 if tot > 0 else 0)
