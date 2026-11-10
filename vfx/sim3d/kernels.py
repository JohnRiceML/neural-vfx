"""Warp kernels for the 3D smoke/fire step (everything except the pressure solve).

Layout: fields are (C, D, H, W) float32, index [c, z, y, x]; y is up. Cell (z, y, x) has its
center at ((x + .5) dx, (y + .5) dx, (z + .5) dx). Per-step parameters live in a small device
buffer `prm` (P_* slots) so the whole step is one CUDA graph replayed with new forces / dt / config.
"""
from __future__ import annotations

import warp as wp

# ---------------------------------------------------------------- parameter buffer layout
P_DT = 0
P_WIND_X = 1
P_WIND_Y = 2
P_WIND_Z = 3
P_N_EM = 4
P_N_IMP = 5
P_DX = 6
P_AMB = 7
P_KAPPA = 8
P_CFLOOR = 9
P_WARM = 10
P_ALPHA = 16
P_BETA = 17
P_EPS = 18
P_DDISS = 19
P_COOL = 20
P_BURN = 21
P_IGN = 22
P_BHEAT = 23
P_SMOKE = 24
P_TOL = 25
P_MAXIT = 26
P_TMAX = 27
P_WMARK = 28
P_EM0 = 32
EM_STRIDE = 10         # px, py, pz, radius, density_rate, temperature, fuel_rate, vx, vy, vz
MAX_EMITTERS = 32
P_IMP0 = P_EM0 + MAX_EMITTERS * EM_STRIDE
IMP_STRIDE = 7         # px, py, pz, radius, vx, vy, vz
MAX_IMPULSES = 32
P_SIZE = P_IMP0 + MAX_IMPULSES * IMP_STRIDE

# scalar slots (device float array written by kernels)
S_MASS_RATIO = 0
S_MAXU = 1
S_RZ = 2
S_ALPHA = 3
S_BETA = 4
S_MAXR = 5
S_THR = 6
S_BEST = 7
S_SIZE = 8

# channels (vfx.core.fields3d.CHANNELS_3D)
C_D = 0
C_VX = 1
C_VY = 2
C_VZ = 3
C_T = 4
C_F = 5

LAM_CAP = 0.16          # explicit 7-point diffusion is monotone for lambda <= 1/6


# ---------------------------------------------------------------- trilinear helpers
@wp.func
def base_idx(g: float, n: int) -> int:
    return wp.min(int(wp.floor(wp.clamp(g, 0.0, float(n - 1)))), n - 2)


@wp.func
def frac(g: float, n: int, i0: int) -> float:
    return wp.clamp(g, 0.0, float(n - 1)) - float(i0)


@wp.func
def tri(f: wp.array4d(dtype=float), c: int, z0: int, y0: int, x0: int, fz: float, fy: float, fx: float) -> float:
    a00 = f[c, z0, y0, x0] * (1.0 - fx) + f[c, z0, y0, x0 + 1] * fx
    a01 = f[c, z0, y0 + 1, x0] * (1.0 - fx) + f[c, z0, y0 + 1, x0 + 1] * fx
    a10 = f[c, z0 + 1, y0, x0] * (1.0 - fx) + f[c, z0 + 1, y0, x0 + 1] * fx
    a11 = f[c, z0 + 1, y0 + 1, x0] * (1.0 - fx) + f[c, z0 + 1, y0 + 1, x0 + 1] * fx
    a0 = a00 * (1.0 - fy) + a01 * fy
    a1 = a10 * (1.0 - fy) + a11 * fy
    return a0 * (1.0 - fz) + a1 * fz


@wp.func
def sample(f: wp.array4d(dtype=float), c: int, p: wp.vec3) -> float:
    """Trilinear sample at grid coords p = (gx, gy, gz) (cell-center index space), clamped."""
    D = f.shape[1]
    H = f.shape[2]
    W = f.shape[3]
    x0 = base_idx(p[0], W)
    y0 = base_idx(p[1], H)
    z0 = base_idx(p[2], D)
    return tri(f, c, z0, y0, x0, frac(p[2], D, z0), frac(p[1], H, y0), frac(p[0], W, x0))


@wp.func
def sample_vel(v: wp.array4d(dtype=float), vc: int, p: wp.vec3) -> wp.vec3:
    D = v.shape[1]
    H = v.shape[2]
    W = v.shape[3]
    x0 = base_idx(p[0], W)
    y0 = base_idx(p[1], H)
    z0 = base_idx(p[2], D)
    fx = frac(p[0], W, x0)
    fy = frac(p[1], H, y0)
    fz = frac(p[2], D, z0)
    return wp.vec3(tri(v, vc, z0, y0, x0, fz, fy, fx), tri(v, vc + 1, z0, y0, x0, fz, fy, fx),
                   tri(v, vc + 2, z0, y0, x0, fz, fy, fx))


@wp.func
def trace(v: wp.array4d(dtype=float), vc: int, p: wp.vec3, s: float) -> wp.vec3:
    """RK2 (midpoint) trace over signed time s (cells per (m/s)): s < 0 back, s > 0 forward."""
    v1 = sample_vel(v, vc, p)
    v2 = sample_vel(v, vc, p + (0.5 * s) * v1)
    return p + s * v2


# ---------------------------------------------------------------- advection (MacCormack)
@wp.kernel
def advect_sl(src: wp.array4d(dtype=float), vel: wp.array4d(dtype=float), vc: int,
              prm: wp.array(dtype=float), inv_dx: float, hat: wp.array4d(dtype=float)):
    """Pass 1: semi-Lagrangian backward step; all channels share one RK2 backtrace."""
    z, y, x = wp.tid()
    q = trace(vel, vc, wp.vec3(float(x), float(y), float(z)), -prm[P_DT] * inv_dx)
    D = src.shape[1]
    H = src.shape[2]
    W = src.shape[3]
    x0 = base_idx(q[0], W)
    y0 = base_idx(q[1], H)
    z0 = base_idx(q[2], D)
    fx = frac(q[0], W, x0)
    fy = frac(q[1], H, y0)
    fz = frac(q[2], D, z0)
    for c in range(src.shape[0]):
        hat[c, z, y, x] = tri(src, c, z0, y0, x0, fz, fy, fx)


@wp.kernel
def advect_mc(src: wp.array4d(dtype=float), vel: wp.array4d(dtype=float), vc: int,
              prm: wp.array(dtype=float), inv_dx: float, hat: wp.array4d(dtype=float),
              out: wp.array4d(dtype=float)):
    """Pass 2: MacCormack correction (Selle et al. 2008), clamped to the 8 source values around
    the backtraced point (falls back to the semi-Lagrangian value where it would overshoot)."""
    z, y, x = wp.tid()
    s = prm[P_DT] * inv_dx
    p = wp.vec3(float(x), float(y), float(z))
    q = trace(vel, vc, p, -s)
    r = trace(vel, vc, p, s)
    D = src.shape[1]
    H = src.shape[2]
    W = src.shape[3]
    qx = base_idx(q[0], W)
    qy = base_idx(q[1], H)
    qz = base_idx(q[2], D)
    rx = base_idx(r[0], W)
    ry = base_idx(r[1], H)
    rz = base_idx(r[2], D)
    fx = frac(r[0], W, rx)
    fy = frac(r[1], H, ry)
    fz = frac(r[2], D, rz)
    for c in range(src.shape[0]):
        h = hat[c, z, y, x]
        bar = tri(hat, c, rz, ry, rx, fz, fy, fx)
        val = h + 0.5 * (src[c, z, y, x] - bar)
        lo = src[c, qz, qy, qx]
        hi = lo
        for k in range(1, 8):
            v = src[c, qz + (k >> 2), qy + ((k >> 1) & 1), qx + (k & 1)]
            lo = wp.min(lo, v)
            hi = wp.max(hi, v)
        if val < lo or val > hi:
            val = h
        out[c, z, y, x] = val


# ---------------------------------------------------------------- sources + reaction
@wp.kernel
def sources(adv: wp.array4d(dtype=float), solid: wp.array3d(dtype=float),
            prm: wp.array(dtype=float), scal: wp.array(dtype=float),
            out: wp.array4d(dtype=float)):
    """Mass correction, emitters, impulses, combustion, cooling, dissipation. adv -> out."""
    z, y, x = wp.tid()
    amb = prm[P_AMB]
    if solid[z, y, x] > 0.5:
        out[C_D, z, y, x] = 0.0
        out[C_VX, z, y, x] = 0.0
        out[C_VY, z, y, x] = 0.0
        out[C_VZ, z, y, x] = 0.0
        out[C_T, z, y, x] = amb
        out[C_F, z, y, x] = 0.0
        return
    dt = prm[P_DT]
    dx = prm[P_DX]
    pos = wp.vec3((float(x) + 0.5) * dx, (float(y) + 0.5) * dx, (float(z) + 0.5) * dx)

    d = adv[C_D, z, y, x] * scal[S_MASS_RATIO]
    vel = wp.vec3(adv[C_VX, z, y, x], adv[C_VY, z, y, x], adv[C_VZ, z, y, x])
    T = adv[C_T, z, y, x]
    F = adv[C_F, z, y, x]

    n_em = int(prm[P_N_EM])
    for i in range(n_em):
        b = P_EM0 + i * EM_STRIDE
        rad = prm[b + 3]
        dd = pos - wp.vec3(prm[b + 0], prm[b + 1], prm[b + 2])
        r2 = wp.dot(dd, dd) / (rad * rad)
        if r2 < 1.0:
            w = (1.0 - r2) * (1.0 - r2)
            d += prm[b + 4] * w * dt
            F += prm[b + 6] * w * dt
            te = prm[b + 5]
            if te > 0.0:
                T = wp.max(T, amb + (te - amb) * w)
            ev = wp.vec3(prm[b + 7], prm[b + 8], prm[b + 9])
            if wp.dot(ev, ev) > 0.0:
                vel = vel + (ev - vel) * w

    n_imp = int(prm[P_N_IMP])
    for i in range(n_imp):
        b = P_IMP0 + i * IMP_STRIDE
        rad = prm[b + 3]
        dd = pos - wp.vec3(prm[b + 0], prm[b + 1], prm[b + 2])
        r2 = wp.dot(dd, dd) / (rad * rad)
        if r2 < 9.0:
            vel = vel + wp.exp(-r2) * wp.vec3(prm[b + 4], prm[b + 5], prm[b + 6])

    # combustion: first-order burn above ignition (100 K ramp); heat release saturates
    # toward max_temperature so a stagnant fed cell cannot heat without bound.
    ign = wp.clamp((T - prm[P_IGN]) / 100.0, 0.0, 1.0)
    burned = F * (1.0 - wp.exp(-prm[P_BURN] * dt)) * ign
    F -= burned
    tmax = prm[P_TMAX]
    room = wp.clamp((tmax - T) / (tmax - amb), 0.0, 1.0)
    T += prm[P_BHEAT] * burned * room
    d += prm[P_SMOKE] * burned

    T = amb + (T - amb) * wp.exp(-prm[P_COOL] * dt)
    d = d * wp.exp(-prm[P_DDISS] * dt)

    out[C_D, z, y, x] = wp.max(d, 0.0)
    out[C_VX, z, y, x] = vel[0]
    out[C_VY, z, y, x] = vel[1]
    out[C_VZ, z, y, x] = vel[2]
    out[C_T, z, y, x] = wp.min(T, tmax)
    out[C_F, z, y, x] = wp.max(F, 0.0)


@wp.func
def solid_at(solid: wp.array3d(dtype=float), z: int, y: int, x: int) -> bool:
    if x < 0 or y < 0 or z < 0 or x >= solid.shape[2] or y >= solid.shape[1] or z >= solid.shape[0]:
        return True
    return solid[z, y, x] > 0.5


@wp.func
def nb_or(f: wp.array4d(dtype=float), solid: wp.array3d(dtype=float), c: int, z: int, y: int, x: int,
          fb: float) -> float:
    if solid_at(solid, z, y, x):
        return fb
    return f[c, z, y, x]


@wp.kernel
def diffuse_heat_fuel(src: wp.array4d(dtype=float), solid: wp.array3d(dtype=float),
                      prm: wp.array(dtype=float), nsub: float, dst: wp.array4d(dtype=float)):
    """One explicit substep of temperature + fuel diffusion (zero flux through walls / solids,
    conservative, monotone: lambda capped at LAM_CAP). Other channels are copied src -> dst."""
    z, y, x = wp.tid()
    dx = prm[P_DX]
    lam = wp.min(prm[P_KAPPA] * prm[P_DT] / (nsub * dx * dx), LAM_CAP)
    for c in range(C_D, C_VZ + 1):
        dst[c, z, y, x] = src[c, z, y, x]
    for c in range(C_T, C_F + 1):
        v = src[c, z, y, x]
        if solid[z, y, x] > 0.5 or lam <= 0.0:
            dst[c, z, y, x] = v
        else:
            s = (nb_or(src, solid, c, z, y, x - 1, v) + nb_or(src, solid, c, z, y, x + 1, v)
                 + nb_or(src, solid, c, z, y - 1, x, v) + nb_or(src, solid, c, z, y + 1, x, v)
                 + nb_or(src, solid, c, z - 1, y, x, v) + nb_or(src, solid, c, z + 1, y, x, v))
            dst[c, z, y, x] = v + lam * (s - 6.0 * v)


# ---------------------------------------------------------------- body forces
@wp.func
def vel_or0(f: wp.array4d(dtype=float), solid: wp.array3d(dtype=float), c: int, z: int, y: int, x: int) -> float:
    if solid_at(solid, z, y, x):
        return 0.0
    return f[c, z, y, x]


@wp.kernel
def curl(f: wp.array4d(dtype=float), solid: wp.array3d(dtype=float), inv2dx: float,
         w_out: wp.array4d(dtype=float)):
    """omega = curl u, central differences; walls / solids count as zero velocity."""
    z, y, x = wp.tid()
    dwdy = vel_or0(f, solid, C_VZ, z, y + 1, x) - vel_or0(f, solid, C_VZ, z, y - 1, x)
    dvdz = vel_or0(f, solid, C_VY, z + 1, y, x) - vel_or0(f, solid, C_VY, z - 1, y, x)
    dudz = vel_or0(f, solid, C_VX, z + 1, y, x) - vel_or0(f, solid, C_VX, z - 1, y, x)
    dwdx = vel_or0(f, solid, C_VZ, z, y, x + 1) - vel_or0(f, solid, C_VZ, z, y, x - 1)
    dvdx = vel_or0(f, solid, C_VY, z, y, x + 1) - vel_or0(f, solid, C_VY, z, y, x - 1)
    dudy = vel_or0(f, solid, C_VX, z, y + 1, x) - vel_or0(f, solid, C_VX, z, y - 1, x)
    w_out[0, z, y, x] = (dwdy - dvdz) * inv2dx
    w_out[1, z, y, x] = (dudz - dwdx) * inv2dx
    w_out[2, z, y, x] = (dvdx - dudy) * inv2dx


@wp.func
def binom3(a: wp.array4d(dtype=float), c0: int, solid: wp.array3d(dtype=float), z: int, y: int, x: int) -> wp.vec3:
    """[1,2,1]^3 / 64 average of channels c0..c0+2 over fluid cells only (renormalised near
    walls / solids)."""
    s = wp.vec3(0.0)
    ws = float(0.0)
    for k in range(-1, 2):
        for j in range(-1, 2):
            for i in range(-1, 2):
                if not solid_at(solid, z + k, y + j, x + i):
                    q = float((2 - wp.abs(i)) * (2 - wp.abs(j)) * (2 - wp.abs(k)))
                    s += q * wp.vec3(a[c0, z + k, y + j, x + i], a[c0 + 1, z + k, y + j, x + i],
                                     a[c0 + 2, z + k, y + j, x + i])
                    ws += q
    if ws > 0.0:
        return s / ws
    return wp.vec3(0.0)


@wp.kernel
def smooth_curl(w: wp.array4d(dtype=float), solid: wp.array3d(dtype=float), out: wp.array4d(dtype=float)):
    z, y, x = wp.tid()
    v = binom3(w, 0, solid, z, y, x)
    out[0, z, y, x] = v[0]
    out[1, z, y, x] = v[1]
    out[2, z, y, x] = v[2]


@wp.func
def wmag(o: wp.array4d(dtype=float), z: int, y: int, x: int) -> float:
    return wp.length(wp.vec3(o[0, z, y, x], o[1, z, y, x], o[2, z, y, x]))


@wp.kernel
def body_forces(f: wp.array4d(dtype=float), solid: wp.array3d(dtype=float),
                omega: wp.array4d(dtype=float), prm: wp.array(dtype=float), force: wp.array4d(dtype=float)):
    """Buoyancy + vorticity confinement + wind-marker force -> force (3,D,H,W)."""
    z, y, x = wp.tid()
    if solid[z, y, x] > 0.5:
        force[0, z, y, x] = 0.0
        force[1, z, y, x] = 0.0
        force[2, z, y, x] = 0.0
        return
    D = f.shape[1]
    H = f.shape[2]
    W = f.shape[3]
    amb = prm[P_AMB]
    d = f[C_D, z, y, x]
    T = f[C_T, z, y, x]
    fv = wp.vec3(0.0, -prm[P_ALPHA] * d + prm[P_BETA] * (T - amb), 0.0)

    eps = prm[P_EPS]
    if eps != 0.0:
        xr = wp.min(x + 1, W - 1)
        xl = wp.max(x - 1, 0)
        yt = wp.min(y + 1, H - 1)
        yb = wp.max(y - 1, 0)
        zf = wp.min(z + 1, D - 1)
        zb = wp.max(z - 1, 0)
        g = wp.vec3((wmag(omega, z, y, xr) - wmag(omega, z, y, xl)) / float(wp.max(xr - xl, 1)),
                    (wmag(omega, z, yt, x) - wmag(omega, z, yb, x)) / float(wp.max(yt - yb, 1)),
                    (wmag(omega, zf, y, x) - wmag(omega, zb, y, x)) / float(wp.max(zf - zb, 1)))
        w = wp.vec3(omega[0, z, y, x], omega[1, z, y, x], omega[2, z, y, x])
        fl = prm[P_CFLOOR] * wp.length(w)
        mag = wp.sqrt(wp.dot(g, g) + fl * fl) + 1.0e-20
        fv += (eps * prm[P_DX] / mag) * wp.cross(g, w)

    wind = wp.vec3(prm[P_WIND_X], prm[P_WIND_Y], prm[P_WIND_Z])
    if wp.dot(wind, wind) > 0.0:
        m = wp.clamp(d + f[C_F, z, y, x] + (T - amb) / prm[P_WMARK], 0.0, 1.0)
        fv += wind * m

    force[0, z, y, x] = fv[0]
    force[1, z, y, x] = fv[1]
    force[2, z, y, x] = fv[2]


# ---- separable version of the fluid-renormalised [1,2,1]^3 smoothing (identical result:
#      binom3 = B(w f) / B(w) with w the fluid mask, zero outside the box, and B separable)
@wp.kernel
def sep_x(a: wp.array4d(dtype=float), c0: int, solid: wp.array3d(dtype=float), out: wp.array4d(dtype=float)):
    """out[0:3] = [1,2,1]_x (w * a[c0:c0+3]), out[3] = [1,2,1]_x w."""
    z, y, x = wp.tid()
    s = wp.vec3(0.0)
    ws = float(0.0)
    for i in range(-1, 2):
        if not solid_at(solid, z, y, x + i):
            q = float(2 - wp.abs(i))
            s += q * wp.vec3(a[c0, z, y, x + i], a[c0 + 1, z, y, x + i], a[c0 + 2, z, y, x + i])
            ws += q
    out[0, z, y, x] = s[0]
    out[1, z, y, x] = s[1]
    out[2, z, y, x] = s[2]
    out[3, z, y, x] = ws


@wp.kernel
def sep_y(a: wp.array4d(dtype=float), out: wp.array4d(dtype=float)):
    z, y, x = wp.tid()
    H = a.shape[2]
    for c in range(4):
        v = 2.0 * a[c, z, y, x]
        if y > 0:
            v += a[c, z, y - 1, x]
        if y + 1 < H:
            v += a[c, z, y + 1, x]
        out[c, z, y, x] = v


@wp.func
def sep_z_ratio(a: wp.array4d(dtype=float), z: int, y: int, x: int) -> wp.vec3:
    D = a.shape[1]
    s = wp.vec3(0.0)
    ws = float(0.0)
    for k in range(-1, 2):
        zz = z + k
        if zz >= 0 and zz < D:
            q = float(2 - wp.abs(k))
            s += q * wp.vec3(a[0, zz, y, x], a[1, zz, y, x], a[2, zz, y, x])
            ws += q * a[3, zz, y, x]
    if ws > 0.0:
        return s / ws
    return wp.vec3(0.0)


@wp.kernel
def sep_z_store(a: wp.array4d(dtype=float), out: wp.array4d(dtype=float)):
    z, y, x = wp.tid()
    v = sep_z_ratio(a, z, y, x)
    out[0, z, y, x] = v[0]
    out[1, z, y, x] = v[1]
    out[2, z, y, x] = v[2]


@wp.kernel
def sep_z_apply(a: wp.array4d(dtype=float), f: wp.array4d(dtype=float), solid: wp.array3d(dtype=float),
                prm: wp.array(dtype=float)):
    """u += dt * smoothed force (fluid cells)."""
    z, y, x = wp.tid()
    if solid[z, y, x] > 0.5:
        return
    dv = prm[P_DT] * sep_z_ratio(a, z, y, x)
    f[C_VX, z, y, x] = f[C_VX, z, y, x] + dv[0]
    f[C_VY, z, y, x] = f[C_VY, z, y, x] + dv[1]
    f[C_VZ, z, y, x] = f[C_VZ, z, y, x] + dv[2]


@wp.kernel
def apply_forces(f: wp.array4d(dtype=float), solid: wp.array3d(dtype=float),
                 force: wp.array4d(dtype=float), prm: wp.array(dtype=float)):
    """u += dt * ([1,2,1]^3-smoothed force): forces resolved over >= 2 cells (see README)."""
    z, y, x = wp.tid()
    if solid[z, y, x] > 0.5:
        return
    dv = prm[P_DT] * binom3(force, 0, solid, z, y, x)
    f[C_VX, z, y, x] = f[C_VX, z, y, x] + dv[0]
    f[C_VY, z, y, x] = f[C_VY, z, y, x] + dv[1]
    f[C_VZ, z, y, x] = f[C_VZ, z, y, x] + dv[2]
