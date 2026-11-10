"""Warp kernels of the detail-particle stage.

Grids cover the unit domain cube like the sim: cell-centred, index [z, y, x], position of cell (i,j,k) =
((x+0.5) dx, (y+0.5) dx, (z+0.5) dx). Particles are a fixed-capacity ring (structure of arrays):
    pos  vec3   world position (m)
    q    vec4   (smoke mass, fuel mass, fuel concentration, temperature K)
                smoke / fuel mass are EXTENSIVE (density units * m^3); fuel concentration and temperature are
                INTENSIVE and follow the solver's per-cell source ODE (combustion, cooling) along the path.
A dead particle has q = 0 (mass 0, temperature 0) and is skipped.
"""
from __future__ import annotations

import warp as wp

wp.config.quiet = True
wp.set_module_options({"enable_backward": False})

C_D = 0
C_VX = 1
C_VY = 2
C_VZ = 3
C_T = 4
C_F = 5

MAX_EMITTERS = 32
EM_STRIDE = 12      # pos xyz, radius, density_rate, temperature, fuel_rate, k_begin, k_end, smoke m, fuel m, 0
EM_SIZE = MAX_EMITTERS * EM_STRIDE

# per-tick values the kernels read from a device array (`dyn`), so one captured CUDA graph serves every tick
D_HEAD = 0          # ring slot of the first particle seeded this tick
D_RSEED = 1         # seed RNG stream for this tick
D_MEPS = 2          # cull threshold (smoke + fuel mass)
D_NEM = 3           # number of emitters
D_PHASE = 4         # sub-grid noise: lerp weight between keyframes A and B
D_KA = 5            # channel base (0, 3 or 6) of keyframe A / B / the one being built
D_KB = 6
D_KC = 7
D_Z0 = 8            # first z slab of keyframe C built this tick
D_TC = 9            # noise time of keyframe C (s)
DYN_SIZE = 16


@wp.struct
class SimParams:
    dt: float
    dx: float            # coarse cell size
    res: int             # coarse res
    amb: float
    cool: float          # temperature_cooling 1/s
    diss: float          # density_dissipation 1/s
    burn: float
    ign: float
    bheat: float
    smoke: float
    tmax: float
    # sub-grid turbulence
    sg_amp: float        # u_sg = sg_amp * |w|_coarse * dx * curlnoise / rms
    sg_freq: float       # noise frequency of octave 0 (1/m)
    sg_oct: int
    sg_lac: float
    sg_gain: float
    sg_rms: float        # measured rms of the raw curl-noise sum (normalisation)
    sg_tfreq: float      # noise time evolution (1/s)
    sg_max: float        # cap on |u_sg| (m/s)
    seed: int
    rk3: int
    vres: int            # resolution of the per-tick velocity grid


@wp.func
def sample_ch(f: wp.array4d(dtype=float), c: int, p: wp.vec3, inv_dx: float, n: int) -> float:
    """Trilinear sample of channel c of a cell-centred (C, n, n, n) grid at world point p (clamped)."""
    gx = wp.clamp(p[0] * inv_dx - 0.5, 0.0, float(n - 1) - 1.0e-4)
    gy = wp.clamp(p[1] * inv_dx - 0.5, 0.0, float(n - 1) - 1.0e-4)
    gz = wp.clamp(p[2] * inv_dx - 0.5, 0.0, float(n - 1) - 1.0e-4)
    x0 = int(gx)
    y0 = int(gy)
    z0 = int(gz)
    fx = gx - float(x0)
    fy = gy - float(y0)
    fz = gz - float(z0)
    c00 = wp.lerp(f[c, z0, y0, x0], f[c, z0, y0, x0 + 1], fx)
    c01 = wp.lerp(f[c, z0, y0 + 1, x0], f[c, z0, y0 + 1, x0 + 1], fx)
    c10 = wp.lerp(f[c, z0 + 1, y0, x0], f[c, z0 + 1, y0, x0 + 1], fx)
    c11 = wp.lerp(f[c, z0 + 1, y0 + 1, x0], f[c, z0 + 1, y0 + 1, x0 + 1], fx)
    return wp.lerp(wp.lerp(c00, c01, fy), wp.lerp(c10, c11, fy), fz)


@wp.func
def sample3(f: wp.array3d(dtype=float), p: wp.vec3, inv_dx: float, n: int) -> float:
    gx = wp.clamp(p[0] * inv_dx - 0.5, 0.0, float(n - 1) - 1.0e-4)
    gy = wp.clamp(p[1] * inv_dx - 0.5, 0.0, float(n - 1) - 1.0e-4)
    gz = wp.clamp(p[2] * inv_dx - 0.5, 0.0, float(n - 1) - 1.0e-4)
    x0 = int(gx)
    y0 = int(gy)
    z0 = int(gz)
    fx = gx - float(x0)
    fy = gy - float(y0)
    fz = gz - float(z0)
    c00 = wp.lerp(f[z0, y0, x0], f[z0, y0, x0 + 1], fx)
    c01 = wp.lerp(f[z0, y0 + 1, x0], f[z0, y0 + 1, x0 + 1], fx)
    c10 = wp.lerp(f[z0 + 1, y0, x0], f[z0 + 1, y0, x0 + 1], fx)
    c11 = wp.lerp(f[z0 + 1, y0 + 1, x0], f[z0 + 1, y0 + 1, x0 + 1], fx)
    return wp.lerp(wp.lerp(c00, c01, fy), wp.lerp(c10, c11, fy), fz)


@wp.func
def solid_at(solid: wp.array3d(dtype=float), p: wp.vec3, inv_dx: float, n: int) -> bool:
    x = int(p[0] * inv_dx)
    y = int(p[1] * inv_dx)
    z = int(p[2] * inv_dx)
    if x < 0 or y < 0 or z < 0 or x >= n or y >= n or z >= n:
        return True
    return solid[z, y, x] > 0.5


@wp.func
def velocity(vel: wp.array4d(dtype=float), p: wp.vec3, sp: SimParams) -> wp.vec3:
    """Trilinear sample of the per-tick velocity field (coarse velocity + sub-grid curl noise, see vel_field)."""
    inv = float(sp.vres) / 1.0
    return wp.vec3(sample_ch(vel, 0, p, inv, sp.vres), sample_ch(vel, 1, p, inv, sp.vres),
                   sample_ch(vel, 2, p, inv, sp.vres))


@wp.func
def trace(vel: wp.array4d(dtype=float), p: wp.vec3, dt: float, sp: SimParams) -> wp.vec3:
    k1 = velocity(vel, p, sp)
    if sp.rk3 == 0:                                      # RK2 midpoint
        k2 = velocity(vel, p + 0.5 * dt * k1, sp)
        return p + dt * k2
    k2 = velocity(vel, p + 0.5 * dt * k1, sp)            # Ralston RK3
    k3 = velocity(vel, p + 0.75 * dt * k2, sp)
    return p + dt * ((2.0 / 9.0) * k1 + (1.0 / 3.0) * k2 + (4.0 / 9.0) * k3)


@wp.kernel
def noise_build(noise: wp.array4d(dtype=float), dyn: wp.array(dtype=float), sp: SimParams, use_dyn: int,
                z0_arg: int, base_arg: int, t_arg: float):
    """Unit curl-noise keyframe N(x, t) = curlnoise(4D: x * freq, t * tfreq) / rms on the vres^3 grid, channels
    base..base+2 of `noise`, z slab [z0, z0 + launch depth). use_dyn = 1: z0, base and t come from dyn (the
    captured per-tick path builds 1/sg_every of the next-but-one keyframe per tick, so the cost is spread evenly)."""
    k, y, x = wp.tid()
    z0 = z0_arg
    base = base_arg
    t = t_arg
    if use_dyn != 0:
        z0 = int(dyn[D_Z0])
        base = int(dyn[D_KC])
        t = dyn[D_TC]
    z = z0 + k
    if z >= sp.vres:
        return
    h = 1.0 / float(sp.vres)
    p = wp.vec3((float(x) + 0.5) * h, (float(y) + 0.5) * h, (float(z) + 0.5) * h)
    q = wp.vec4(p[0] * sp.sg_freq, p[1] * sp.sg_freq, p[2] * sp.sg_freq, t * sp.sg_tfreq)
    v = wp.curlnoise(wp.rand_init(sp.seed), q, wp.uint32(sp.sg_oct), sp.sg_lac, sp.sg_gain) / sp.sg_rms
    noise[base + 0, z, y, x] = v[0]
    noise[base + 1, z, y, x] = v[1]
    noise[base + 2, z, y, x] = v[2]


@wp.kernel
def vel_field(data: wp.array4d(dtype=float), wmag: wp.array3d(dtype=float), noise: wp.array4d(dtype=float),
              dyn: wp.array(dtype=float), sp: SimParams, out: wp.array4d(dtype=float)):
    """Velocity the particles follow this tick, on a vres^3 grid: trilinear coarse velocity + sub-grid velocity
    u_sg = sg_amp * |curl u|_coarse * dx * N(x, t)   (capped at sg_max), N = the unit curl noise, linearly
    interpolated in time between two keyframes (A, B) built ahead of time by noise_build. With sg_every = 1 the
    phase is 0 and A is the noise at this tick's time: identical to evaluating the noise here. The grid makes the
    noise cost independent of the particle count; the keyframes make it independent of the tick."""
    z, y, x = wp.tid()
    h = 1.0 / float(sp.vres)
    p = wp.vec3((float(x) + 0.5) * h, (float(y) + 0.5) * h, (float(z) + 0.5) * h)
    inv = 1.0 / sp.dx
    u = wp.vec3(sample_ch(data, C_VX, p, inv, sp.res), sample_ch(data, C_VY, p, inv, sp.res),
                sample_ch(data, C_VZ, p, inv, sp.res))
    if sp.sg_amp > 0.0:
        w = sample3(wmag, p, inv, sp.res)
        a = sp.sg_amp * w * sp.dx
        if a > 1.0e-5 * sp.sg_rms:
            ka = int(dyn[D_KA])
            kb = int(dyn[D_KB])
            ph = dyn[D_PHASE]
            v = wp.vec3(wp.lerp(noise[ka + 0, z, y, x], noise[kb + 0, z, y, x], ph),
                        wp.lerp(noise[ka + 1, z, y, x], noise[kb + 1, z, y, x], ph),
                        wp.lerp(noise[ka + 2, z, y, x], noise[kb + 2, z, y, x], ph)) * a
            m = wp.length(v)
            if m > sp.sg_max:
                v = v * (sp.sg_max / m)
            u = u + v
    out[0, z, y, x] = u[0]
    out[1, z, y, x] = u[1]
    out[2, z, y, x] = u[2]


@wp.func
def sources(p: wp.vec3, q: wp.vec4, em: wp.array(dtype=float), n_em: int, sp: SimParams) -> wp.vec4:
    """The solver's per-cell source ODE along the particle path (sim3d kernels.sources): emitter temperature
    stamp, first-order combustion with the 100 K ignition ramp and saturating heat release, exponential
    cooling and dissipation. Emitter MASS is injected only by seeding (new particles), not here."""
    m = q[0]
    f = q[1]
    cf = q[2]
    T = q[3]
    for i in range(n_em):
        b = i * EM_STRIDE
        te = em[b + 5]
        if te > 0.0:
            rad = em[b + 3]
            dd = p - wp.vec3(em[b + 0], em[b + 1], em[b + 2])
            r2 = wp.dot(dd, dd) / (rad * rad)
            if r2 < 1.0:
                w = (1.0 - r2) * (1.0 - r2)
                T = wp.max(T, sp.amb + (te - sp.amb) * w)
    ign = wp.clamp((T - sp.ign) / 100.0, 0.0, 1.0)
    bf = (1.0 - wp.exp(-sp.burn * sp.dt)) * ign
    burned = f * bf
    f = f - burned
    m = m + sp.smoke * burned
    bc = cf * bf
    cf = cf - bc
    room = wp.clamp((sp.tmax - T) / (sp.tmax - sp.amb), 0.0, 1.0)
    T = T + sp.bheat * bc * room
    T = sp.amb + (T - sp.amb) * wp.exp(-sp.cool * sp.dt)
    m = m * wp.exp(-sp.diss * sp.dt)
    return wp.vec4(m, f, cf, wp.min(T, sp.tmax))


@wp.func
def splat(p: wp.vec3, m: float, h: float, inv_fdx: float, nf: int,
          dep_m: wp.array3d(dtype=float), dep_h: wp.array3d(dtype=float), dep_n: wp.array3d(dtype=float)):
    """Trilinear (tent, 2 fine cells wide) scatter of mass, heat and count with atomics."""
    gx = wp.clamp(p[0] * inv_fdx - 0.5, 0.0, float(nf - 1) - 1.0e-4)
    gy = wp.clamp(p[1] * inv_fdx - 0.5, 0.0, float(nf - 1) - 1.0e-4)
    gz = wp.clamp(p[2] * inv_fdx - 0.5, 0.0, float(nf - 1) - 1.0e-4)
    x0 = int(gx)
    y0 = int(gy)
    z0 = int(gz)
    fx = gx - float(x0)
    fy = gy - float(y0)
    fz = gz - float(z0)
    for c in range(8):
        dx = c & 1
        dy = (c >> 1) & 1
        dz = (c >> 2) & 1
        wx = wp.where(dx == 1, fx, 1.0 - fx)
        wy = wp.where(dy == 1, fy, 1.0 - fy)
        wz = wp.where(dz == 1, fz, 1.0 - fz)
        w = wx * wy * wz
        wp.atomic_add(dep_m, z0 + dz, y0 + dy, x0 + dx, w * m)
        wp.atomic_add(dep_h, z0 + dz, y0 + dy, x0 + dx, w * h)
        wp.atomic_add(dep_n, z0 + dz, y0 + dy, x0 + dx, w)


# ---------------------------------------------------------------------------------------------- kernels
@wp.kernel
def vort_mag(data: wp.array4d(dtype=float), res: int, inv2dx: float, out: wp.array3d(dtype=float)):
    """|curl u| of the coarse velocity (central differences, one-sided at walls)."""
    z, y, x = wp.tid()
    xm = wp.max(x - 1, 0)
    xp = wp.min(x + 1, res - 1)
    ym = wp.max(y - 1, 0)
    yp = wp.min(y + 1, res - 1)
    zm = wp.max(z - 1, 0)
    zp = wp.min(z + 1, res - 1)
    sx = inv2dx * 2.0 / float(xp - xm)
    sy = inv2dx * 2.0 / float(yp - ym)
    sz = inv2dx * 2.0 / float(zp - zm)
    dvz_dy = (data[C_VZ, z, yp, x] - data[C_VZ, z, ym, x]) * sy
    dvy_dz = (data[C_VY, zp, y, x] - data[C_VY, zm, y, x]) * sz
    dvx_dz = (data[C_VX, zp, y, x] - data[C_VX, zm, y, x]) * sz
    dvz_dx = (data[C_VZ, z, y, xp] - data[C_VZ, z, y, xm]) * sx
    dvy_dx = (data[C_VY, z, y, xp] - data[C_VY, z, y, xm]) * sx
    dvx_dy = (data[C_VX, z, yp, x] - data[C_VX, z, ym, x]) * sy
    w = wp.vec3(dvz_dy - dvy_dz, dvx_dz - dvz_dx, dvy_dx - dvx_dy)
    out[z, y, x] = wp.length(w)


@wp.kernel
def update(pos: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.vec4), n: int, k_new: int,
           vel: wp.array4d(dtype=float), solid: wp.array3d(dtype=float),
           em: wp.array(dtype=float), dyn: wp.array(dtype=float), sp: SimParams,
           inv_fdx: float, nf: int,
           dep_m: wp.array3d(dtype=float), dep_h: wp.array3d(dtype=float), dep_n: wp.array3d(dtype=float),
           splat_on: int):
    """Advance every live particle (except the k_new ring slots being re-seeded this tick): advect (RK2/RK3 of
    coarse + sub-grid velocity), keep out of solids, cull out-of-box / faded, run the source ODE, deposit.
    Per-tick values (ring head, cull threshold, emitter count) come from dyn."""
    i = wp.tid()
    head = int(dyn[D_HEAD])
    n_em = int(dyn[D_NEM])
    m_eps = dyn[D_MEPS]
    if (i - head + n) % n < k_new:
        return
    qi = q[i]
    if qi[3] <= 0.0:
        return
    p0 = pos[i]
    p = trace(vel, p0, sp.dt, sp)
    inv = 1.0 / sp.dx
    if solid_at(solid, p, inv, sp.res):
        p = p0                                            # blocked: stay (velocity is 0 at the wall)
        if solid_at(solid, p, inv, sp.res):
            q[i] = wp.vec4(0.0, 0.0, 0.0, 0.0)
            return
    qi = sources(p, qi, em, n_em, sp)
    if qi[0] + qi[1] < m_eps and qi[3] - sp.amb < 1.0:    # faded smoke and cold: dead
        q[i] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        return
    pos[i] = p
    q[i] = qi
    if splat_on != 0:
        splat(p, qi[0], qi[3] - sp.amb, inv_fdx, nf, dep_m, dep_h, dep_n)


@wp.kernel
def seed(pos: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.vec4), n: int,
         data: wp.array4d(dtype=float), vel: wp.array4d(dtype=float), solid: wp.array3d(dtype=float),
         em: wp.array(dtype=float), dyn: wp.array(dtype=float), sp: SimParams,
         inv_fdx: float, nf: int,
         dep_m: wp.array3d(dtype=float), dep_h: wp.array3d(dtype=float), dep_n: wp.array3d(dtype=float),
         splat_on: int):
    """Recycle the k_new (= launch size) oldest ring slots head..head+k_new: thread j seeds a particle for the
    emitter whose range [k_begin, k_end) contains j (host-computed in proportion to the smoke it will inject,
    density + smoke_from_fuel * fuel), or clears the slot. Positions are drawn with pdf ~ the emitter falloff
    (1-r^2/R^2)^2 (rejection), so equal-mass particles reproduce the injected density profile. Each particle is
    then advanced by a random fraction of dt (no seeding bands)."""
    j = wp.tid()
    head = int(dyn[D_HEAD])
    n_em = int(dyn[D_NEM])
    rseed = int(dyn[D_RSEED])
    e = int(-1)
    for i in range(n_em):
        b = i * EM_STRIDE
        if j >= int(em[b + 8]) and j < int(em[b + 9]):
            e = i
    if e < 0:
        q[(head + j) % n] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        return
    b = e * EM_STRIDE
    c = wp.vec3(em[b + 0], em[b + 1], em[b + 2])
    rad = em[b + 3]
    st = wp.rand_init(rseed, j)
    d = wp.vec3(0.0)
    w = float(0.0)
    for t in range(32):
        d = wp.vec3(wp.randf(st, -1.0, 1.0), wp.randf(st, -1.0, 1.0), wp.randf(st, -1.0, 1.0))
        r2 = wp.dot(d, d)
        if r2 < 1.0:
            w = (1.0 - r2) * (1.0 - r2)
            if wp.randf(st) < w:
                break
    p = c + d * rad
    slot = (head + j) % n
    inv = 1.0 / sp.dx
    if solid_at(solid, p, inv, sp.res):
        q[slot] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        return
    te = em[b + 5]
    T = sample_ch(data, C_T, p, inv, sp.res)
    if te > 0.0:
        T = wp.max(T, sp.amb + (te - sp.amb) * w)
    T = wp.max(T, sp.amb + 1.0e-3)                       # T > 0 marks the particle alive
    cf = sample_ch(data, C_F, p, inv, sp.res)
    # jitter in time: advance by a random fraction of the tick
    p = trace(vel, p, sp.dt * wp.randf(st), sp)
    if solid_at(solid, p, inv, sp.res):
        p = c + d * rad
    pos[slot] = p
    qi = wp.vec4(em[b + 10], em[b + 11], cf, T)
    q[slot] = qi
    if splat_on != 0:
        splat(p, qi[0], qi[3] - sp.amb, inv_fdx, nf, dep_m, dep_h, dep_n)


@wp.kernel
def deposit(pos: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.vec4), amb: float, inv_fdx: float, nf: int,
            dep_m: wp.array3d(dtype=float), dep_h: wp.array3d(dtype=float), dep_n: wp.array3d(dtype=float)):
    """Unfused deposit (used to time advection and deposit separately)."""
    i = wp.tid()
    qi = q[i]
    if qi[3] <= 0.0:
        return
    splat(pos[i], qi[0], qi[3] - amb, inv_fdx, nf, dep_m, dep_h, dep_n)


@wp.kernel
def restrict(dep_m: wp.array3d(dtype=float), dep_h: wp.array3d(dtype=float), dep_n: wp.array3d(dtype=float),
             f: int, out: wp.array4d(dtype=float)):
    """Box restriction of the fine deposits to the sim grid (mean of f^3 fine cells): mass, heat, count."""
    z, y, x = wp.tid()
    sm = float(0.0)
    sh = float(0.0)
    sn = float(0.0)
    for a in range(f):
        for bb in range(f):
            for cc in range(f):
                sm += dep_m[z * f + a, y * f + bb, x * f + cc]
                sh += dep_h[z * f + a, y * f + bb, x * f + cc]
                sn += dep_n[z * f + a, y * f + bb, x * f + cc]
    inv = 1.0 / float(f * f * f)
    out[0, z, y, x] = sm * inv
    out[1, z, y, x] = sh * inv
    out[2, z, y, x] = sn * inv


@wp.kernel
def compose(dep_m: wp.array3d(dtype=float), dep_h: wp.array3d(dtype=float), coarse: wp.array4d(dtype=float),
            data: wp.array4d(dtype=float), res: int, inv_dx: float, fdx: float, amb: float,
            n_full: float, strength: float, mod_max: float, h_strength: float, h_mod_max: float,
            dens: wp.array3d(dtype=float), temp: wp.array3d(dtype=float)):
    """Fine cell: rho = up(G) * (1 - b + b * clamp(P / up(R(P)), 0, mod_max)); same for heat with H.
    up = trilinear from the sim grid, R = box restriction; P / up(R(P)) is the particle detail modulation (mean 1
    over a sim cell, so the restriction of rho stays close to G); b = strength * clamp(count / n_full) fades it out
    where there are too few particles to be smooth."""
    z, y, x = wp.tid()
    p = wp.vec3((float(x) + 0.5) * fdx, (float(y) + 0.5) * fdx, (float(z) + 0.5) * fdx)
    g = sample_ch(data, C_D, p, inv_dx, res)
    th = wp.max(sample_ch(data, C_T, p, inv_dx, res) - amb, 0.0)
    rm = sample_ch(coarse, 0, p, inv_dx, res)
    rh = sample_ch(coarse, 1, p, inv_dx, res)
    b = strength * wp.clamp(sample_ch(coarse, 2, p, inv_dx, res) / n_full, 0.0, 1.0)
    md = 1.0
    mh = 1.0
    if rm > 1.0e-12:
        md = wp.min(dep_m[z, y, x] / rm, mod_max)
    if rh > 1.0e-6:
        mh = wp.min(dep_h[z, y, x] / rh, h_mod_max)
    bh = b * h_strength / wp.max(strength, 1.0e-6)
    dens[z, y, x] = g * (1.0 - b + b * md)
    temp[z, y, x] = amb + th * (1.0 - bh + bh * mh)


@wp.kernel
def blur_axis(src: wp.array3d(dtype=float), dst: wp.array3d(dtype=float), axis: int, n: int):
    """[1,2,1]/4 along one axis (clamped)."""
    z, y, x = wp.tid()
    if axis == 0:
        a = src[z, y, wp.max(x - 1, 0)]
        c = src[z, y, wp.min(x + 1, n - 1)]
    elif axis == 1:
        a = src[z, wp.max(y - 1, 0), x]
        c = src[z, wp.min(y + 1, n - 1), x]
    else:
        a = src[wp.max(z - 1, 0), y, x]
        c = src[wp.min(z + 1, n - 1), y, x]
    dst[z, y, x] = 0.25 * a + 0.5 * src[z, y, x] + 0.25 * c


@wp.kernel
def noise_rms(sp: SimParams, out: wp.array(dtype=float)):
    i = wp.tid()
    st = wp.rand_init(12345, i)
    q = wp.vec4(wp.randf(st) * 37.0, wp.randf(st) * 37.0, wp.randf(st) * 37.0, wp.randf(st) * 37.0)
    v = wp.curlnoise(wp.rand_init(sp.seed), q, wp.uint32(sp.sg_oct), sp.sg_lac, sp.sg_gain)
    out[i] = wp.dot(v, v)
