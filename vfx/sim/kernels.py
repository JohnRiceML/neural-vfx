"""Warp kernels for the 2D smoke/fire step (everything except the pressure solve).

Layout: fields are (C, H, W) float32, index [c, y, x], y = 0 is the bottom row.
Cell (y, x) has its center at ((x + 0.5) dx, (y + 0.5) dx).

Per-step parameters live in a small device buffer `prm` (see P_* indices) so the whole
step can be captured once as a CUDA graph and replayed with new forces / dt / config.
"""
from __future__ import annotations

import warp as wp

# ---------------------------------------------------------------- parameter buffer layout
P_DT = 0
P_WIND_X = 1
P_WIND_Y = 2
P_N_EM = 3
P_N_IMP = 4
P_DX = 5
P_AMB = 6
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
EM_STRIDE = 8          # px, py, radius, density_rate, temperature, fuel_rate, vx, vy
MAX_EMITTERS = 32
P_IMP0 = P_EM0 + MAX_EMITTERS * EM_STRIDE
IMP_STRIDE = 5         # px, py, radius, vx, vy
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

# channels
C_D = 0
C_VX = 1
C_VY = 2
C_T = 3
C_F = 4


# ---------------------------------------------------------------- sampling helpers
@wp.func
def sample(f: wp.array3d(dtype=float), c: int, gx: float, gy: float) -> float:
    """Bilinear sample of channel c at grid coords (gx, gy) = cell-center index space,
    clamped to the domain (constant extension past the outermost cell centers)."""
    H = f.shape[1]
    W = f.shape[2]
    gx = wp.clamp(gx, 0.0, float(W - 1))
    gy = wp.clamp(gy, 0.0, float(H - 1))
    x0 = wp.min(int(wp.floor(gx)), W - 2)
    y0 = wp.min(int(wp.floor(gy)), H - 2)
    fx = gx - float(x0)
    fy = gy - float(y0)
    a = f[c, y0, x0] * (1.0 - fx) + f[c, y0, x0 + 1] * fx
    b = f[c, y0 + 1, x0] * (1.0 - fx) + f[c, y0 + 1, x0 + 1] * fx
    return a * (1.0 - fy) + b * fy


@wp.func
def corner_minmax(f: wp.array3d(dtype=float), c: int, gx: float, gy: float) -> wp.vec2:
    H = f.shape[1]
    W = f.shape[2]
    gx = wp.clamp(gx, 0.0, float(W - 1))
    gy = wp.clamp(gy, 0.0, float(H - 1))
    x0 = wp.min(int(wp.floor(gx)), W - 2)
    y0 = wp.min(int(wp.floor(gy)), H - 2)
    a = f[c, y0, x0]
    b = f[c, y0, x0 + 1]
    d = f[c, y0 + 1, x0]
    e = f[c, y0 + 1, x0 + 1]
    return wp.vec2(wp.min(wp.min(a, b), wp.min(d, e)), wp.max(wp.max(a, b), wp.max(d, e)))


@wp.func
def sample_vel(v: wp.array3d(dtype=float), vc: int, gx: float, gy: float) -> wp.vec2:
    return wp.vec2(sample(v, vc, gx, gy), sample(v, vc + 1, gx, gy))


@wp.func
def trace(v: wp.array3d(dtype=float), vc: int, gx: float, gy: float, s: float) -> wp.vec2:
    """RK2 (midpoint) trace over signed time s (in cells per (m/s)): s<0 back, s>0 forward."""
    v1 = sample_vel(v, vc, gx, gy)
    mx = gx + 0.5 * s * v1[0]
    my = gy + 0.5 * s * v1[1]
    v2 = sample_vel(v, vc, mx, my)
    return wp.vec2(gx + s * v2[0], gy + s * v2[1])


# ---------------------------------------------------------------- advection (MacCormack)
@wp.kernel
def advect_sl(src: wp.array3d(dtype=float), vel: wp.array3d(dtype=float), vc: int,
              prm: wp.array(dtype=float), inv_dx: float, hat: wp.array3d(dtype=float)):
    """Pass 1: semi-Lagrangian backward step, all channels share one RK2 backtrace."""
    y, x = wp.tid()
    s = -prm[P_DT] * inv_dx
    q = trace(vel, vc, float(x), float(y), s)
    for c in range(src.shape[0]):
        hat[c, y, x] = sample(src, c, q[0], q[1])


@wp.kernel
def advect_mc(src: wp.array3d(dtype=float), vel: wp.array3d(dtype=float), vc: int,
              prm: wp.array(dtype=float), inv_dx: float, hat: wp.array3d(dtype=float),
              out: wp.array3d(dtype=float)):
    """Pass 2: MacCormack correction (Selle et al. 2008) with min/max clamping to the
    four source values around the backtraced point (keeps it bounded / monotone)."""
    y, x = wp.tid()
    s = prm[P_DT] * inv_dx
    q = trace(vel, vc, float(x), float(y), -s)
    r = trace(vel, vc, float(x), float(y), s)
    for c in range(src.shape[0]):
        h = hat[c, y, x]
        bar = sample(hat, c, r[0], r[1])
        val = h + 0.5 * (src[c, y, x] - bar)
        mm = corner_minmax(src, c, q[0], q[1])
        if val < mm[0] or val > mm[1]:
            val = h
        out[c, y, x] = val


# ---------------------------------------------------------------- sources + reaction
@wp.func
def smooth_fall(r2: float) -> float:
    w = 1.0 - r2
    return w * w


@wp.kernel
def sources(adv: wp.array3d(dtype=float), solid: wp.array2d(dtype=float),
            prm: wp.array(dtype=float), scal: wp.array(dtype=float),
            out: wp.array3d(dtype=float)):
    """Mass correction, emitters, impulses, combustion, cooling, dissipation. adv -> out."""
    y, x = wp.tid()
    amb = prm[P_AMB]
    if solid[y, x] > 0.5:
        out[C_D, y, x] = 0.0
        out[C_VX, y, x] = 0.0
        out[C_VY, y, x] = 0.0
        out[C_T, y, x] = amb
        out[C_F, y, x] = 0.0
        return
    dt = prm[P_DT]
    dx = prm[P_DX]
    px = (float(x) + 0.5) * dx
    py = (float(y) + 0.5) * dx

    d = adv[C_D, y, x] * scal[S_MASS_RATIO]
    vx = adv[C_VX, y, x]
    vy = adv[C_VY, y, x]
    T = adv[C_T, y, x]
    F = adv[C_F, y, x]

    n_em = int(prm[P_N_EM])
    for i in range(n_em):
        b = P_EM0 + i * EM_STRIDE
        rad = prm[b + 2]
        ddx = px - prm[b + 0]
        ddy = py - prm[b + 1]
        r2 = (ddx * ddx + ddy * ddy) / (rad * rad)
        if r2 < 1.0:
            w = smooth_fall(r2)
            d += prm[b + 3] * w * dt
            F += prm[b + 5] * w * dt
            te = prm[b + 4]
            if te > 0.0:
                T = wp.max(T, amb + (te - amb) * w)
            evx = prm[b + 6]
            evy = prm[b + 7]
            if evx * evx + evy * evy > 0.0:
                vx = vx + (evx - vx) * w
                vy = vy + (evy - vy) * w

    n_imp = int(prm[P_N_IMP])
    for i in range(n_imp):
        b = P_IMP0 + i * IMP_STRIDE
        rad = prm[b + 2]
        ddx = px - prm[b + 0]
        ddy = py - prm[b + 1]
        r2 = (ddx * ddx + ddy * ddy) / (rad * rad)
        if r2 < 9.0:
            g = wp.exp(-r2)
            vx += prm[b + 3] * g
            vy += prm[b + 4] * g

    # combustion: first-order burn above ignition (100 K ramp to avoid on/off flicker).
    # Heat release saturates toward max_temperature (an adiabatic-flame-like cap), so a
    # stagnant cell fed by an emitter cannot heat without bound.
    ign = wp.clamp((T - prm[P_IGN]) / 100.0, 0.0, 1.0)
    burned = F * (1.0 - wp.exp(-prm[P_BURN] * dt)) * ign
    F -= burned
    tmax = prm[P_TMAX]
    room = wp.clamp((tmax - T) / (tmax - amb), 0.0, 1.0)
    T += prm[P_BHEAT] * burned * room
    d += prm[P_SMOKE] * burned

    # cooling toward ambient, density dissipation (exponential => dt-consistent)
    T = amb + (T - amb) * wp.exp(-prm[P_COOL] * dt)
    d = d * wp.exp(-prm[P_DDISS] * dt)

    out[C_D, y, x] = wp.max(d, 0.0)
    out[C_VX, y, x] = vx
    out[C_VY, y, x] = vy
    out[C_T, y, x] = wp.min(T, prm[P_TMAX])
    out[C_F, y, x] = wp.max(F, 0.0)


# ---------------------------------------------------------------- body forces
@wp.func
def solid_at(solid: wp.array2d(dtype=float), y: int, x: int) -> bool:
    H = solid.shape[0]
    W = solid.shape[1]
    if x < 0 or y < 0 or x >= W or y >= H:
        return True
    return solid[y, x] > 0.5


@wp.kernel
def curl(f: wp.array3d(dtype=float), solid: wp.array2d(dtype=float), inv2dx: float,
         w_out: wp.array2d(dtype=float)):
    """omega = dvy/dx - dvx/dy, central differences; walls / solids count as zero velocity."""
    y, x = wp.tid()
    H = f.shape[1]
    W = f.shape[2]
    vyr = 0.0
    vyl = 0.0
    vxt = 0.0
    vxb = 0.0
    if not solid_at(solid, y, x + 1):
        vyr = f[C_VY, y, x + 1]
    if not solid_at(solid, y, x - 1):
        vyl = f[C_VY, y, x - 1]
    if not solid_at(solid, y + 1, x):
        vxt = f[C_VX, y + 1, x]
    if not solid_at(solid, y - 1, x):
        vxb = f[C_VX, y - 1, x]
    w_out[y, x] = ((vyr - vyl) - (vxt - vxb)) * inv2dx


@wp.kernel
def body_forces(f: wp.array3d(dtype=float), solid: wp.array2d(dtype=float),
                omega: wp.array2d(dtype=float), prm: wp.array(dtype=float)):
    """Buoyancy + vorticity confinement + wind marker force, explicit Euler, in place."""
    y, x = wp.tid()
    if solid[y, x] > 0.5:
        return
    H = f.shape[1]
    W = f.shape[2]
    dt = prm[P_DT]
    dx = prm[P_DX]
    amb = prm[P_AMB]
    d = f[C_D, y, x]
    T = f[C_T, y, x]
    fx = 0.0
    fy = -prm[P_ALPHA] * d + prm[P_BETA] * (T - amb)

    # vorticity confinement: N = grad|w| / |grad|w||, f = eps dx (N x w)
    eps = prm[P_EPS]
    if eps != 0.0:
        xr = wp.min(x + 1, W - 1)
        xl = wp.max(x - 1, 0)
        yt = wp.min(y + 1, H - 1)
        yb = wp.max(y - 1, 0)
        gx = (wp.abs(omega[y, xr]) - wp.abs(omega[y, xl])) / float(wp.max(xr - xl, 1))
        gy = (wp.abs(omega[yt, x]) - wp.abs(omega[yb, x])) / float(wp.max(yt - yb, 1))
        mag = wp.sqrt(gx * gx + gy * gy) + 1.0e-20
        nx = gx / mag
        ny = gy / mag
        w = omega[y, x]
        fx += eps * dx * (ny * w)
        fy += eps * dx * (-nx * w)

    # wind: a uniform acceleration is a pure gradient in a closed box (the projection would
    # cancel it exactly), so it acts on the gas that carries smoke / fuel / heat instead.
    wxv = prm[P_WIND_X]
    wyv = prm[P_WIND_Y]
    if wxv * wxv + wyv * wyv > 0.0:
        m = wp.clamp(d + f[C_F, y, x] + (T - amb) / prm[P_WMARK], 0.0, 1.0)
        fx += wxv * m
        fy += wyv * m

    f[C_VX, y, x] = f[C_VX, y, x] + fx * dt
    f[C_VY, y, x] = f[C_VY, y, x] + fy * dt
