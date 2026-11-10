"""DetailParticles: sub-grid smoke detail carried on Lagrangian particles advected by the 64^3 velocity and
deposited into an out_res^3 `Volume` every sim tick (see README.md).

Per tick the host does one small pinned H2D copy (emitters + the per-tick values the kernels read from `dyn`)
and, with graph=True (default), one CUDA-graph launch of the whole tick. The graph is captured once per
(state buffers, settings) and replayed; there is no host sync.
"""
from __future__ import annotations

import math
from collections import OrderedDict

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
    sg_every=4,          # noise keyframe interval (ticks); lerp between keyframes, next one built 1/sg_every per
                         # tick. 1 = evaluate the noise at every tick's time (the original, costlier path)
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
    graph=True,          # capture the tick in a CUDA graph (False = direct launches with per-stage CUDA events)
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
        from vfx.render3d.pinned import PinnedRing
        from vfx.sim3d import SolverConfig3D
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"unknown params {sorted(unknown)}")
        self.p = dict(DEFAULTS, **params)
        self.cfg = config if config is not None else SolverConfig3D()
        self.n = int(count)
        self.out_res = int(out_res)
        self.device = device
        self._events: list = []
        self._timings: dict = {}
        self._wp_dev = wp.device_from_torch(torch.empty(0, device=device).device)
        n, nf = self.n, self.out_res
        self.pos = wp.zeros(n, dtype=wp.vec3, device=self._wp_dev)
        self.q = wp.zeros(n, dtype=wp.vec4, device=self._wp_dev)
        self.dep = torch.zeros(3, nf, nf, nf, device=device)            # mass, heat, count
        self.dep_tmp = torch.zeros(nf, nf, nf, device=device)
        self.dens = torch.zeros(nf, nf, nf, device=device)
        self.temp = torch.full((nf, nf, nf), AMBIENT_TEMP, device=device)
        self._dep_wp = wp.from_torch(self.dep)
        self._dep3 = [wp.from_torch(self.dep[i]) for i in range(3)]
        self._tmp_wp = wp.from_torch(self.dep_tmp)
        self._dens_wp, self._temp_wp = wp.from_torch(self.dens), wp.from_torch(self.temp)
        # emitters + per-tick values: one pinned staging buffer per queued tick (PinnedRing: a slot is reused only
        # after its copy has executed, so any number of ticks can be queued without a host sync)
        self._ring = PinnedRing(K.EM_SIZE + K.DYN_SIZE, slots=4)
        self.up_dev = torch.zeros(K.EM_SIZE + K.DYN_SIZE, device=device)
        self._em_wp = wp.from_torch(self.up_dev[:K.EM_SIZE])
        self._dyn_wp = wp.from_torch(self.up_dev[K.EM_SIZE:])
        self._cap_stream = wp.Stream(self._wp_dev)
        self._graphs: OrderedDict = OrderedDict()
        self._coarse = None
        self._wmag = None
        self._res = None
        self._em_cache = {}
        self._rms = {}
        self._m_eps = 0.0
        self.reset()

    # ------------------------------------------------------------------ public
    def reset(self) -> None:
        self.q.zero_()
        self.pos.zero_()
        self.head = 0
        self.tick = 0
        self.time = 0.0
        self._kf_ok = False            # noise keyframes must be (re)built
        self._kf_static = None
        self.dens.zero_()
        self.temp.fill_(AMBIENT_TEMP)

    def volume(self) -> Volume:
        return Volume(self.dens, self.temp)

    def k_new(self, dt: float) -> int:
        life = min(self.p["max_life"], math.log(1.0 / self.p["life_eps"]) / max(self.cfg.density_dissipation, 1e-6))
        return max(1, min(self.n, int(math.ceil(self.n / max(1.0, life / dt)))))

    def step(self, state: FieldState3D, forces: Forces3D, dt: float) -> None:
        # every launch goes on torch's current stream, so torch ops (the renderer, ...) are ordered with ours
        ts = torch.cuda.current_stream(self.dens.device)
        with wp.ScopedStream(wp.stream_from_torch(ts), sync_enter=False):
            self._step(state, forces, dt, ts)

    @property
    def timings(self) -> dict:
        """ms of the last finished step (CUDA events, resolved lazily: syncs on the last event only when read).
        graph=True: {'total'}; graph=False: {'advect+deposit' or 'advect', 'deposit', 'compose', 'total'}."""
        if self._events:
            self._events[-1][2].synchronize()
            self._timings = {k: a.elapsed_time(b) for k, a, b in self._events}
            self._events = []
        return self._timings

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
                    coverage_mean=float((wp.to_torch(self._coarse)[2][g > 0.02] / self.p["n_full"]).clamp(max=1).mean())
                    if (g > 0.02).any() else 0.0, staging_waits=self._ring.waits)

    # ------------------------------------------------------------------ tick
    def _step(self, state: FieldState3D, forces: Forces3D, dt: float, ts) -> None:
        res = state.data.shape[-1]
        nf = self.out_res
        if nf % res:
            raise ValueError(f"out_res {nf} must be a multiple of the sim res {res}")
        if self._res != res:
            self._alloc_coarse(res)
        sp, sp_key = self._params(res, dt)
        k = self.k_new(dt)
        kf = self._upload(forces, res, dt, k, sp, ts)
        if sp.sg_amp <= 0.0:
            self._kf_ok = False          # keyframes are not maintained while the sub-grid motion is off
        elif not self._kf_ok or self._kf_static != kf["static"]:
            self._init_keyframes(sp, kf)
        data, solid = wp.from_torch(state.data), wp.from_torch(state.solid)
        if self.p["graph"]:
            key = (state.data.data_ptr(), state.solid.data_ptr(), res, k, sp_key, bool(self.p["fused"]),
                   int(self.p["blur"]), self.p["n_full"], self.p["strength"], self.p["mod_max"],
                   self.p["heat_strength"], self.p["heat_mod_max"])
            g = self._graphs.get(key)
            if g is None:
                self._cap_stream.wait_stream(wp.stream_from_torch(ts))
                with wp.ScopedStream(self._cap_stream, sync_enter=False):
                    wp.capture_begin(stream=self._cap_stream, force_module_load=True)
                    try:
                        self._record(data, solid, sp, k, res, None)
                    finally:
                        g = wp.capture_end(stream=self._cap_stream)
                self._graphs[key] = g
                while len(self._graphs) > 4:
                    self._graphs.popitem(last=False)
            else:
                self._graphs.move_to_end(key)
            e0 = self._ev()
            wp.capture_launch(g, stream=wp.stream_from_torch(ts))
            self._events = [("total", e0, self._ev())]
        else:
            ev = []
            self._record(data, solid, sp, k, res, ev)
            self._events = ev
        self.head = (self.head + k) % self.n
        self.tick += 1
        self.time += dt

    def _record(self, data, solid, sp, k, res, ev):
        """Launch (or record into a graph) one tick. ev: list to receive per-stage CUDA events (eager only)."""
        nf, f, d = self.out_res, self.out_res // res, self._wp_dev
        inv_fdx = nf / DOMAIN_SIZE
        fused = 1 if self.p["fused"] else 0
        dep_m, dep_h, dep_n = self._dep3
        marks = []                                   # eager only: (name, event) after each launch -> per-kernel ms

        def mark(name):
            if ev is not None:
                marks.append((name, self._ev()))

        mark("start")
        self._dep_wp.zero_()
        mark("k:zero")
        wp.launch(K.vort_mag, dim=(res, res, res), inputs=[data, res, 0.5 / sp.dx, self._wmag], device=d)
        mark("k:vort_mag")
        if sp.sg_amp > 0.0:
            vr, ke = sp.vres, int(self.p["sg_every"])
            wp.launch(K.noise_build, dim=(-(-vr // ke), vr, vr), inputs=[self._noise, self._dyn_wp, sp, 1, 0, 0, 0.0],
                      device=d)
            mark("k:noise_build")
        wp.launch(K.vel_field, dim=(sp.vres,) * 3, inputs=[data, self._wmag, self._noise, self._dyn_wp, sp, self._vel],
                  device=d)
        mark("k:vel_field")
        wp.launch(K.update, dim=self.n, inputs=[self.pos, self.q, self.n, k, self._vel, solid, self._em_wp,
                                               self._dyn_wp, sp, inv_fdx, nf, dep_m, dep_h, dep_n, fused], device=d)
        mark("k:update")
        wp.launch(K.seed, dim=k, inputs=[self.pos, self.q, self.n, data, self._vel, solid, self._em_wp, self._dyn_wp,
                                         sp, inv_fdx, nf, dep_m, dep_h, dep_n, fused], device=d)
        mark("k:seed")
        if not fused:
            wp.launch(K.deposit, dim=self.n, inputs=[self.pos, self.q, sp.amb, inv_fdx, nf, dep_m, dep_h, dep_n],
                      device=d)
        mark("k:deposit")
        for _ in range(int(self.p["blur"])):
            for src in self._dep3:
                for ax in range(3):
                    wp.launch(K.blur_axis, dim=(nf, nf, nf), inputs=[src, self._tmp_wp, ax, nf], device=d)
                    wp.copy(src, self._tmp_wp)
        mark("k:blur")
        wp.launch(K.restrict, dim=(res, res, res), inputs=[dep_m, dep_h, dep_n, f, self._coarse], device=d)
        mark("k:restrict")
        st = float(self.p["strength"])
        wp.launch(K.compose, dim=(nf, nf, nf),
                  inputs=[dep_m, dep_h, self._coarse, data, res, 1.0 / sp.dx, DOMAIN_SIZE / nf, sp.amb,
                          float(self.p["n_full"]), st, float(self.p["mod_max"]),
                          float(self.p["heat_strength"]) * st, float(self.p["heat_mod_max"]),
                          self._dens_wp, self._temp_wp], device=d)
        mark("k:compose")
        if ev is not None:
            m = dict(marks)
            ev += [("advect" if not fused else "advect+deposit", m["start"], m["k:seed"]),
                   ("deposit", m["k:seed"], m["k:deposit"]), ("compose", m["k:deposit"], m["k:compose"])]
            # per kernel: each mark minus the previous one (launch gaps are included: eager launches from Python)
            ev += [(marks[i][0] if marks[i][0].startswith("k:") else "k:" + marks[i][0], marks[i - 1][1], marks[i][1])
                   for i in range(1, len(marks))]
            ev.append(("total", m["start"], m["k:compose"]))

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
        self._noise = wp.zeros((9, vr, vr, vr), dtype=float, device=self._wp_dev)   # 3 keyframes x 3 components
        self._graphs.clear()
        self._kf_ok = False

    def _params(self, res, dt):
        c, p = self.cfg, self.p
        if int(p["vel_res"]) != self._vel.shape[1]:
            self._alloc_coarse(res)
        sp = K.SimParams()
        sp.dt, sp.dx, sp.res, sp.amb = float(dt), DOMAIN_SIZE / res, int(res), float(AMBIENT_TEMP)
        sp.cool, sp.diss, sp.burn, sp.ign = c.temperature_cooling, c.density_dissipation, c.burn_rate, c.ignition_temp
        sp.bheat, sp.smoke, sp.tmax = c.burn_heat, c.smoke_from_fuel, c.max_temperature
        sp.sg_amp = float(p["sg_amp"])
        sp.sg_freq = float(p["sg_freq"] or res / (2.0 * DOMAIN_SIZE))
        sp.sg_oct, sp.sg_lac, sp.sg_gain = int(p["sg_oct"]), float(p["sg_lac"]), float(p["sg_gain"])
        sp.sg_tfreq, sp.sg_max = float(p["sg_tfreq"]), float(p["sg_max"])
        sp.seed, sp.rk3 = int(p["seed"]), 1 if p["rk3"] else 0
        sp.vres = int(p["vel_res"])
        key = (sp.sg_oct, sp.sg_lac, sp.sg_gain, sp.seed)
        if key not in self._rms:
            sp.sg_rms = 1.0
            o = wp.zeros(65536, dtype=float, device=self._wp_dev)
            wp.launch(K.noise_rms, dim=65536, inputs=[sp, o], device=self._wp_dev)
            self._rms[key] = float(math.sqrt(o.numpy().mean()))     # one-time host read (first tick only)
        # Warp's curlnoise differentiates w.r.t. its own input q = p * freq, so the rms is freq independent
        sp.sg_rms = self._rms[key]
        sp_key = (sp.dt, sp.dx, sp.res, sp.cool, sp.diss, sp.burn, sp.ign, sp.bheat, sp.smoke, sp.tmax, sp.sg_amp,
                  sp.sg_freq, sp.sg_oct, sp.sg_lac, sp.sg_gain, sp.sg_tfreq, sp.sg_max, sp.seed, sp.rk3, sp.vres,
                  int(p["sg_every"]))
        return sp, sp_key

    def _keyframes(self, dt):
        """Noise schedule: keyframe m has time m * sg_every * dt and lives in slot m % 3. At tick n (cycle
        c = n // K, phase j = n % K) the velocity lerps keyframes c -> c+1 by j / K while slab j of keyframe c+2
        is built."""
        ke = int(self.p["sg_every"])
        c, j = divmod(self.tick, ke)
        slab = -(-int(self.p["vel_res"]) // ke)
        T = ke * dt
        return dict(c=c, j=j, phase=j / ke, ka=3 * (c % 3), kb=3 * ((c + 1) % 3), kc=3 * ((c + 2) % 3), z0=j * slab,
                    tc=(c + 2) * T, ta=c * T, tb=(c + 1) * T,
                    static=(ke, T, self.p["sg_freq"], self.p["sg_oct"], self.p["sg_lac"], self.p["sg_gain"],
                            self.p["sg_tfreq"], self.p["seed"], int(self.p["vel_res"])))

    def _init_keyframes(self, sp, kf):
        """Build keyframes A (cycle c) and B (c+1) in full; keyframe C is then built by the ticks of cycle c.
        Runs only after reset or a change of the noise settings (eager launches, outside the graph)."""
        vr = sp.vres
        for base, t in ((kf["ka"], kf["ta"]), (kf["kb"], kf["tb"])):
            wp.launch(K.noise_build, dim=(vr, vr, vr), inputs=[self._noise, self._dyn_wp, sp, 0, 0, base, float(t)],
                      device=self._wp_dev)
        if kf["j"] > 0:          # joined mid-cycle: the slabs of C before this one were never built
            wp.launch(K.noise_build, dim=(kf["z0"], vr, vr),
                      inputs=[self._noise, self._dyn_wp, sp, 0, 0, kf["kc"], float(kf["tc"])], device=self._wp_dev)
        self._kf_ok = True
        self._kf_static = kf["static"]

    def _upload(self, forces: Forces3D, res: int, dt: float, k_tot: int, sp, ts) -> dict:
        ems = list(forces.emitters)[:K.MAX_EMITTERS]
        c = self.cfg
        integ = []
        for e in ems:
            key = (tuple(e.pos), e.radius, res)
            if key not in self._em_cache:
                self._em_cache[key] = _emitter_integral(e.pos, max(e.radius, 1e-6), res)
            integ.append(self._em_cache[key])
        rate = [(e.density_rate + c.smoke_from_fuel * e.fuel_rate) * w for e, w in zip(ems, integ)]
        tot = sum(rate)
        h = self._ring.acquire()
        a = h.numpy()
        a[:] = 0.0
        k0 = 0
        for i, (e, w, r) in enumerate(zip(ems, integ, rate)):
            ke = int(round(k_tot * r / tot)) if tot > 0 else 0
            if i == len(ems) - 1 and tot > 0:
                ke = k_tot - k0
            b = i * K.EM_STRIDE
            a[b:b + 12] = (*e.pos, max(e.radius, 1e-6), e.density_rate, e.temperature, e.fuel_rate, 0.0,
                           k0, k0 + ke, e.density_rate * dt * w / max(ke, 1), e.fuel_rate * dt * w / max(ke, 1))
            k0 += ke
        # cull threshold relative to the per-particle seed mass (an absolute one kills small particles of large rings)
        seeds = [a[i * K.EM_STRIDE + 10] + a[i * K.EM_STRIDE + 11] for i in range(len(ems))
                 if a[i * K.EM_STRIDE + 10] + a[i * K.EM_STRIDE + 11] > 0]
        self._m_eps = 0.5 * self.p["life_eps"] * min(seeds) if seeds else 0.0
        kf = self._keyframes(dt)
        dyn = a[K.EM_SIZE:]
        dyn[K.D_HEAD] = self.head
        dyn[K.D_RSEED] = (int(self.p["seed"]) * 7919 + self.tick) % (1 << 24)
        dyn[K.D_MEPS] = self._m_eps
        dyn[K.D_NEM] = len(ems) if tot > 0 else 0
        dyn[K.D_PHASE], dyn[K.D_KA], dyn[K.D_KB], dyn[K.D_KC] = kf["phase"], kf["ka"], kf["kb"], kf["kc"]
        dyn[K.D_Z0], dyn[K.D_TC] = kf["z0"], kf["tc"]
        self.up_dev.copy_(h, non_blocking=True)
        self._ring.record(ts)
        return kf
