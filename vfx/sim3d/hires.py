"""Opt-in finer scalar grid (`Solver3D(res, ..., scalar_res=S)`): density, temperature and fuel are
advected, burned, cooled, dissipated and diffused on an S^3 grid by the res^3 velocity (trilinear),
then box-restricted to res^3 each step so buoyancy and the other couplings read a consistent coarse
field. Velocity, forces and the pressure solve stay at res^3.

Fine field layout: (3, S, S, S) float32 = density, temperature (K), fuel; index [c, z, y, x], same
world box as the coarse grid (fine dx = DOMAIN_SIZE / S). Coarse cell-center index space is used
for the velocity traces, so the coarse and fine traces are the same curve in world space.
"""
from __future__ import annotations

import warp as wp

from .kernels import (C_D, C_F, C_T, LAM_CAP, P_AMB, P_BHEAT, P_BURN, P_COOL, P_DDISS, P_DT, P_EM0, P_IGN,
                      P_KAPPA, P_N_EM, P_SMOKE, P_TMAX, EM_STRIDE, S_MASS_RATIO, base_idx, frac, trace, tri)

wp.set_module_options({"enable_backward": False})

H_D = 0
H_T = 1
H_F = 2


@wp.func
def f2c(p: wp.vec3, a: float, b: float) -> wp.vec3:
    """fine index coords -> coarse index coords: (p + .5) * (Nc / Nf) - .5 = a p + b."""
    return wp.vec3(a * p[0] + b, a * p[1] + b, a * p[2] + b)


@wp.kernel
def advect_sl_fine(src: wp.array4d(dtype=float), vel: wp.array4d(dtype=float), vc: int,
                   prm: wp.array(dtype=float), inv_dx_c: float, a: float, b: float, hat: wp.array4d(dtype=float)):
    """Semi-Lagrangian pass on the fine grid; RK2 backtrace through the coarse velocity."""
    z, y, x = wp.tid()
    pc = f2c(wp.vec3(float(x), float(y), float(z)), a, b)
    qc = trace(vel, vc, pc, -prm[P_DT] * inv_dx_c)
    q = wp.vec3((qc[0] - b) / a, (qc[1] - b) / a, (qc[2] - b) / a)
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
def advect_mc_fine(src: wp.array4d(dtype=float), vel: wp.array4d(dtype=float), vc: int,
                   prm: wp.array(dtype=float), inv_dx_c: float, a: float, b: float,
                   hat: wp.array4d(dtype=float), out: wp.array4d(dtype=float)):
    """MacCormack correction on the fine grid, clamped to the 8 source values (as kernels.advect_mc)."""
    z, y, x = wp.tid()
    s = prm[P_DT] * inv_dx_c
    pc = f2c(wp.vec3(float(x), float(y), float(z)), a, b)
    qc = trace(vel, vc, pc, -s)
    rc = trace(vel, vc, pc, s)
    q = wp.vec3((qc[0] - b) / a, (qc[1] - b) / a, (qc[2] - b) / a)
    r = wp.vec3((rc[0] - b) / a, (rc[1] - b) / a, (rc[2] - b) / a)
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


@wp.kernel
def sources_fine(adv: wp.array4d(dtype=float), solid: wp.array3d(dtype=float), prm: wp.array(dtype=float),
                 scal: wp.array(dtype=float), dxf: float, out: wp.array4d(dtype=float)):
    """kernels.sources for the scalar channels only, at fine-cell centers (velocity terms stay coarse)."""
    z, y, x = wp.tid()
    amb = prm[P_AMB]
    if solid[z, y, x] > 0.5:
        out[H_D, z, y, x] = 0.0
        out[H_T, z, y, x] = amb
        out[H_F, z, y, x] = 0.0
        return
    dt = prm[P_DT]
    pos = wp.vec3((float(x) + 0.5) * dxf, (float(y) + 0.5) * dxf, (float(z) + 0.5) * dxf)
    d = adv[H_D, z, y, x] * scal[S_MASS_RATIO]
    T = adv[H_T, z, y, x]
    F = adv[H_F, z, y, x]
    n_em = int(prm[P_N_EM])
    for i in range(n_em):
        bb = P_EM0 + i * EM_STRIDE
        rad = prm[bb + 3]
        dd = pos - wp.vec3(prm[bb + 0], prm[bb + 1], prm[bb + 2])
        r2 = wp.dot(dd, dd) / (rad * rad)
        if r2 < 1.0:
            w = (1.0 - r2) * (1.0 - r2)
            d += prm[bb + 4] * w * dt
            F += prm[bb + 6] * w * dt
            te = prm[bb + 5]
            if te > 0.0:
                T = wp.max(T, amb + (te - amb) * w)
    ign = wp.clamp((T - prm[P_IGN]) / 100.0, 0.0, 1.0)
    burned = F * (1.0 - wp.exp(-prm[P_BURN] * dt)) * ign
    F -= burned
    tmax = prm[P_TMAX]
    room = wp.clamp((tmax - T) / (tmax - amb), 0.0, 1.0)
    T += prm[P_BHEAT] * burned * room
    d += prm[P_SMOKE] * burned
    T = amb + (T - amb) * wp.exp(-prm[P_COOL] * dt)
    d = d * wp.exp(-prm[P_DDISS] * dt)
    out[H_D, z, y, x] = wp.max(d, 0.0)
    out[H_T, z, y, x] = wp.min(T, tmax)
    out[H_F, z, y, x] = wp.max(F, 0.0)


@wp.func
def nbv(f: wp.array4d(dtype=float), solid: wp.array3d(dtype=float), c: int, z: int, y: int, x: int, fb: float) -> float:
    if x < 0 or y < 0 or z < 0 or x >= solid.shape[2] or y >= solid.shape[1] or z >= solid.shape[0]:
        return fb
    if solid[z, y, x] > 0.5:
        return fb
    return f[c, z, y, x]


@wp.kernel
def diffuse_fine(src: wp.array4d(dtype=float), solid: wp.array3d(dtype=float), prm: wp.array(dtype=float),
                 nsub: float, dxf: float, dst: wp.array4d(dtype=float)):
    """kernels.diffuse_heat_fuel on the fine layout (temperature + fuel diffuse, density copied)."""
    z, y, x = wp.tid()
    lam = wp.min(prm[P_KAPPA] * prm[P_DT] / (nsub * dxf * dxf), LAM_CAP)
    dst[H_D, z, y, x] = src[H_D, z, y, x]
    for c in range(H_T, H_F + 1):
        v = src[c, z, y, x]
        if solid[z, y, x] > 0.5 or lam <= 0.0:
            dst[c, z, y, x] = v
        else:
            s = (nbv(src, solid, c, z, y, x - 1, v) + nbv(src, solid, c, z, y, x + 1, v)
                 + nbv(src, solid, c, z, y - 1, x, v) + nbv(src, solid, c, z, y + 1, x, v)
                 + nbv(src, solid, c, z - 1, y, x, v) + nbv(src, solid, c, z + 1, y, x, v))
            dst[c, z, y, x] = v + lam * (s - 6.0 * v)


@wp.func
def ovl(i: int, j: int, ratio: float) -> float:
    """Length of [j, j+1) ∩ [i ratio, (i+1) ratio) in fine cells."""
    return wp.max(0.0, wp.min(float(j + 1), float(i + 1) * ratio) - wp.max(float(j), float(i) * ratio))


@wp.kernel
def restrict_scalars(fine: wp.array4d(dtype=float), fsolid: wp.array3d(dtype=float), csolid: wp.array3d(dtype=float),
                     prm: wp.array(dtype=float), ratio: float, span: int, data: wp.array4d(dtype=float)):
    """Coarse density / temperature / fuel = box average of the overlapping fine cells (exact
    fractional overlaps, so non-integer ratios such as 96/64 are handled; fine solid cells count with
    their 0 / ambient / 0). Coarse solid cells keep 0 / ambient / 0, so the fine smoke that lies in
    fine-fluid parts of coarse-solid cells (a thin sliver at obstacle surfaces) is not in the coarse
    total. The mass rescale acts on the fine field; the coarse field follows by restriction."""
    z, y, x = wp.tid()
    amb = prm[P_AMB]
    if csolid[z, y, x] > 0.5:
        data[C_D, z, y, x] = 0.0
        data[C_T, z, y, x] = amb
        data[C_F, z, y, x] = 0.0
        return
    n = fine.shape[1]
    z0 = int(wp.floor(float(z) * ratio))
    y0 = int(wp.floor(float(y) * ratio))
    x0 = int(wp.floor(float(x) * ratio))
    sw = float(0.0)
    sd = float(0.0)
    st = float(0.0)
    sf = float(0.0)
    for k in range(span):
        zz = z0 + k
        wz = float(0.0)
        if zz < n:
            wz = ovl(z, zz, ratio)
        for j in range(span):
            yy = y0 + j
            wy = float(0.0)
            if yy < n:
                wy = ovl(y, yy, ratio)
            for i in range(span):
                xx = x0 + i
                wx = float(0.0)
                if xx < n:
                    wx = ovl(x, xx, ratio)
                w = wz * wy * wx
                if w > 0.0:
                    # plain box average (fine solids hold 0 / ambient / 0): the exact conservative
                    # restriction, so coarse mass == fine mass over coarse fluid cells
                    sw += w
                    sd += w * fine[H_D, zz, yy, xx]
                    st += w * fine[H_T, zz, yy, xx]
                    sf += w * fine[H_F, zz, yy, xx]
    if sw > 0.0:
        data[C_D, z, y, x] = sd / sw
        data[C_T, z, y, x] = st / sw
        data[C_F, z, y, x] = sf / sw
    else:
        data[C_D, z, y, x] = 0.0
        data[C_T, z, y, x] = amb
        data[C_F, z, y, x] = 0.0
