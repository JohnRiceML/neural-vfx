"""Exact collocated pressure projection in 3D: MG-preconditioned conjugate gradient (Warp).

Same discretisation as the 2D solver (vfx/sim/pressure.py), one more axis:
  face velocity  = mean of the two adjacent cell velocities, 0 on walls / solid faces
  d(u)_i         = sum over open faces of (+/-)(u_i + u_nb)          (= 2 dx div u)
  g(phi)_i       = per axis, sum over open faces of one-sided differences  (g = -d^T)
  solve K phi = -d(u*), K = d d^T (the 2dx "wide" Laplacian), then u = u* - g(phi)
  => the face-averaged central divergence that divergence() reports is zero to solver tolerance.

Preconditioner: symmetric V-cycle in a 2x2x2-block Hadamard basis (8 components; in a box this
block-diagonalises K exactly: the sum component is a Neumann 7-point Laplacian, components with a
parity difference along an axis see Dirichlet walls along that axis). Level-1 stencils are probed
exactly from K (component-diagonal part), coarser levels are Galerkin with constant prolongation
x1/2 (scale 1/16 in 3D), red-black Gauss-Seidel V(1,1), plus one damped-Jacobi sweep on the exact
K before/after (handles the inter-component coupling at obstacles).

Hot full-grid kernels are "flat tiled": each thread handles `CPT` cells, and dot products / maxima
are fused into the kernel that produces the operands (block tile reduction, fixed order, no atomics
=> bit-identical). The loop is a CUDA-graph conditional (wp.capture_while): no host sync.
The MG operators are rebuilt only when `solid` changes (GPU-side compare + wp.capture_if).
"""
from __future__ import annotations

import math

import warp as wp

from .kernels import P_MAXIT, P_TOL, P_WARM, S_ALPHA, S_BEST, S_BETA, S_MAXR, S_MAXU, S_RZ, S_THR

HS = 0.35355339059327373      # 1/sqrt(8): orthonormal 2x2x2 Hadamard

# open-face mask bits: 0 +x, 1 -x, 2 +y, 3 -y, 4 +z, 5 -z ; FLUID bit 6
FLUID = 64


# ---------------------------------------------------------------- mask helpers
@wp.func
def mask_at(fm: wp.array3d(dtype=int), z: int, y: int, x: int) -> int:
    if x < 0 or y < 0 or z < 0 or z >= fm.shape[0] or y >= fm.shape[1] or x >= fm.shape[2]:
        return 0
    return fm[z, y, x]


@wp.func
def ax_off(ax: int) -> wp.vec3i:
    """(dz, dy, dx) unit offset along axis ax (0 = x, 1 = y, 2 = z)."""
    if ax == 0:
        return wp.vec3i(0, 0, 1)
    if ax == 1:
        return wp.vec3i(0, 1, 0)
    return wp.vec3i(1, 0, 0)


@wp.func
def g_ax(p: wp.array3d(dtype=float), fm: wp.array3d(dtype=int), z: int, y: int, x: int, ax: int) -> float:
    """Component ax of g(p) at a fluid cell (sum of one-sided differences over open faces)."""
    m = fm[z, y, x]
    e = ax_off(ax)
    g = float(0.0)
    c = p[z, y, x]
    if (m >> (2 * ax)) & 1 == 1:
        g += p[z + e[0], y + e[1], x + e[2]] - c
    if (m >> (2 * ax + 1)) & 1 == 1:
        g += c - p[z - e[0], y - e[1], x - e[2]]
    return g


@wp.func
def kmul(p: wp.array3d(dtype=float), fm: wp.array3d(dtype=int), z: int, y: int, x: int) -> float:
    """(K p)_i = -d(g(p))_i at a fluid cell."""
    m = fm[z, y, x]
    d = float(0.0)
    for ax in range(3):
        e = ax_off(ax)
        gi = g_ax(p, fm, z, y, x, ax)
        if (m >> (2 * ax)) & 1 == 1:
            d += gi + g_ax(p, fm, z + e[0], y + e[1], x + e[2], ax)
        if (m >> (2 * ax + 1)) & 1 == 1:
            d -= gi + g_ax(p, fm, z - e[0], y - e[1], x - e[2], ax)
    return -d


@wp.func
def kdiag(m: int) -> float:
    """diag(K) = sum over axes of (oL - oR)^2 + oL + oR."""
    s = float(0.0)
    for ax in range(3):
        oR = float((m >> (2 * ax)) & 1)
        oL = float((m >> (2 * ax + 1)) & 1)
        s += (oL - oR) * (oL - oR) + oL + oR
    return s


@wp.func
def div2(v: wp.array4d(dtype=float), vc: int, fm: wp.array3d(dtype=int), z: int, y: int, x: int) -> float:
    """d(u)_i = 2 dx div u (face-averaged central divergence, closed walls / solid faces)."""
    m = fm[z, y, x]
    d = float(0.0)
    for ax in range(3):
        e = ax_off(ax)
        u = v[vc + ax, z, y, x]
        if (m >> (2 * ax)) & 1 == 1:
            d += u + v[vc + ax, z + e[0], y + e[1], x + e[2]]
        if (m >> (2 * ax + 1)) & 1 == 1:
            d -= u + v[vc + ax, z - e[0], y - e[1], x - e[2]]
    return d


@wp.func
def unflat(i: int, N: int) -> wp.vec3i:
    return wp.vec3i(i // (N * N), (i // N) % N, i % N)


# ---------------------------------------------------------------- masks
@wp.kernel
def k_mask(solid: wp.array3d(dtype=float), fm: wp.array3d(dtype=int)):
    z, y, x = wp.tid()
    D = solid.shape[0]
    H = solid.shape[1]
    W = solid.shape[2]
    if solid[z, y, x] > 0.5:
        fm[z, y, x] = 0
        return
    m = FLUID
    if x + 1 < W:
        if solid[z, y, x + 1] <= 0.5:
            m = m | 1
    if x > 0:
        if solid[z, y, x - 1] <= 0.5:
            m = m | 2
    if y + 1 < H:
        if solid[z, y + 1, x] <= 0.5:
            m = m | 4
    if y > 0:
        if solid[z, y - 1, x] <= 0.5:
            m = m | 8
    if z + 1 < D:
        if solid[z + 1, y, x] <= 0.5:
            m = m | 16
    if z > 0:
        if solid[z - 1, y, x] <= 0.5:
            m = m | 32
    fm[z, y, x] = m


@wp.kernel
def k_solid_changed(solid: wp.array3d(dtype=float), cache: wp.array3d(dtype=float), flag: wp.array(dtype=int)):
    """flag[0] = 1 if solid differs from the cached copy (benign same-value race), cache <- solid."""
    z, y, x = wp.tid()
    s = solid[z, y, x]
    if s != cache[z, y, x]:
        flag[0] = 1
        cache[z, y, x] = s


@wp.kernel
def k_div_abs(v: wp.array4d(dtype=float), vc: int, fm: wp.array3d(dtype=int), out: wp.array3d(dtype=float)):
    """|div u| * dx (= |d(u)| / 2), 0 in solids."""
    z, y, x = wp.tid()
    if fm[z, y, x] == 0:
        out[z, y, x] = 0.0
        return
    out[z, y, x] = 0.5 * wp.abs(div2(v, vc, fm, z, y, x))


@wp.kernel
def k_apply(v: wp.array4d(dtype=float), vc: int, fm: wp.array3d(dtype=int), phi: wp.array3d(dtype=float)):
    z, y, x = wp.tid()
    if fm[z, y, x] == 0:
        v[vc, z, y, x] = 0.0
        v[vc + 1, z, y, x] = 0.0
        v[vc + 2, z, y, x] = 0.0
        return
    for ax in range(3):
        v[vc + ax, z, y, x] = v[vc + ax, z, y, x] - g_ax(phi, fm, z, y, x, ax)


# ---------------------------------------------------------------- multigrid (Hadamard basis)
@wp.func
def hsign(c: int, dz: int, dy: int, dx: int) -> float:
    e = (c & 1) * dx + ((c >> 1) & 1) * dy + ((c >> 2) & 1) * dz
    if (e & 1) == 1:
        return -HS
    return HS


@wp.func
def tv(c: int, B: wp.vec3i, z: int, y: int, x: int) -> float:
    """Basis vector t_{c,B} (B = block (K, J, I)) evaluated at fine cell (z, y, x)."""
    if (z >> 1) == B[0] and (y >> 1) == B[1] and (x >> 1) == B[2]:
        return hsign(c, z & 1, y & 1, x & 1)
    return 0.0


@wp.func
def gt(fm: wp.array3d(dtype=int), c: int, B: wp.vec3i, q: wp.vec3i, ax: int) -> float:
    """Component ax of g(t_{c,B}) at fine cell q = (z, y, x)."""
    m = mask_at(fm, q[0], q[1], q[2])
    e = ax_off(ax)
    t0 = tv(c, B, q[0], q[1], q[2])
    g = float(0.0)
    if (m >> (2 * ax)) & 1 == 1:
        g += tv(c, B, q[0] + e[0], q[1] + e[1], q[2] + e[2]) - t0
    if (m >> (2 * ax + 1)) & 1 == 1:
        g += t0 - tv(c, B, q[0] - e[0], q[1] - e[1], q[2] - e[2])
    return g


@wp.func
def axcell(ax: int, o: wp.vec3i, a: int, b: int, c: int) -> wp.vec3i:
    """Fine cell (z, y, x) with coordinate `a` along axis ax and b, c on the two other axes,
    all relative to origin o = (z, y, x)."""
    if ax == 0:
        return wp.vec3i(o[0] + c, o[1] + b, o[2] + a)
    if ax == 1:
        return wp.vec3i(o[0] + c, o[1] + a, o[2] + b)
    return wp.vec3i(o[0] + a, o[1] + c, o[2] + b)


@wp.kernel
def k_probe1(fm: wp.array3d(dtype=int), diag: wp.array4d(dtype=float), off: wp.array4d(dtype=float)):
    """Exact component-diagonal block of T K T^T on the 2x2x2-block grid (K = g^T g).
    off[ax*8 + c] = coupling to the +ax neighbour block."""
    c, K, J, I = wp.tid()
    m = diag.shape[1]
    B = wp.vec3i(K, J, I)
    o = wp.vec3i(2 * K, 2 * J, 2 * I)
    dsum = float(0.0)
    for ax in range(3):
        for b in range(2):
            for cc in range(2):
                for a in range(4):
                    v = gt(fm, c, B, axcell(ax, o, a - 1, b, cc), ax)
                    dsum += v * v
    diag[c, K, J, I] = dsum
    for ax in range(3):
        e = ax_off(ax)
        Bn = B + e
        s = float(0.0)
        if Bn[0] < m and Bn[1] < m and Bn[2] < m:
            for b in range(2):
                for cc in range(2):
                    for a in range(2):
                        q = axcell(ax, o, 1 + a, b, cc)
                        s += gt(fm, c, B, q, ax) * gt(fm, c, Bn, q, ax)
        off[ax * 8 + c, K, J, I] = s


@wp.kernel
def k_galerkin(diag: wp.array4d(dtype=float), off: wp.array4d(dtype=float), scale: float,
               dc: wp.array4d(dtype=float), oc: wp.array4d(dtype=float)):
    c, K, J, I = wp.tid()
    o = wp.vec3i(2 * K, 2 * J, 2 * I)
    d = float(0.0)
    for k in range(8):
        d += diag[c, o[0] + (k >> 2), o[1] + ((k >> 1) & 1), o[2] + (k & 1)]
    for ax in range(3):
        oi = float(0.0)   # links inside the 2x2x2 block along ax
        oo = float(0.0)   # links to the +ax neighbour block
        for b in range(2):
            for cc in range(2):
                q = axcell(ax, o, 0, b, cc)
                oi += off[ax * 8 + c, q[0], q[1], q[2]]
                q2 = axcell(ax, o, 1, b, cc)
                oo += off[ax * 8 + c, q2[0], q2[1], q2[2]]
        d += 2.0 * oi
        oc[ax * 8 + c, K, J, I] = oo * scale
    dc[c, K, J, I] = d * scale


@wp.func
def offsum(xv: wp.array4d(dtype=float), off: wp.array4d(dtype=float), c: int, K: int, J: int, I: int) -> float:
    m = xv.shape[1]
    a = float(0.0)
    if I + 1 < m:
        a += off[c, K, J, I] * xv[c, K, J, I + 1]
    if I > 0:
        a += off[c, K, J, I - 1] * xv[c, K, J, I - 1]
    if J + 1 < m:
        a += off[8 + c, K, J, I] * xv[c, K, J + 1, I]
    if J > 0:
        a += off[8 + c, K, J - 1, I] * xv[c, K, J - 1, I]
    if K + 1 < m:
        a += off[16 + c, K, J, I] * xv[c, K + 1, J, I]
    if K > 0:
        a += off[16 + c, K - 1, J, I] * xv[c, K - 1, J, I]
    return a


@wp.kernel
def k_smooth(xv: wp.array4d(dtype=float), r: wp.array4d(dtype=float), diag: wp.array4d(dtype=float),
             off: wp.array4d(dtype=float), color: int, zero_init: int, dmin: float):
    c, K, J, I = wp.tid()
    if ((I + J + K) & 1) != color:
        return
    d = diag[c, K, J, I]
    if d <= dmin:
        xv[c, K, J, I] = 0.0
        return
    acc = r[c, K, J, I]
    if zero_init == 0:
        acc -= offsum(xv, off, c, K, J, I)
    xv[c, K, J, I] = acc / d


@wp.kernel
def k_resrestrict(xv: wp.array4d(dtype=float), r: wp.array4d(dtype=float), diag: wp.array4d(dtype=float),
                  off: wp.array4d(dtype=float), rc: wp.array4d(dtype=float)):
    c, K, J, I = wp.tid()
    acc = float(0.0)
    for k in range(8):
        z = 2 * K + (k >> 2)
        y = 2 * J + ((k >> 1) & 1)
        x = 2 * I + (k & 1)
        acc += r[c, z, y, x] - diag[c, z, y, x] * xv[c, z, y, x] - offsum(xv, off, c, z, y, x)
    rc[c, K, J, I] = 0.125 * acc


@wp.kernel
def k_prolong(xv: wp.array4d(dtype=float), diag: wp.array4d(dtype=float), xc: wp.array4d(dtype=float),
              dmin: float):
    c, K, J, I = wp.tid()
    if diag[c, K, J, I] <= dmin:
        return
    xv[c, K, J, I] = xv[c, K, J, I] + xc[c, K >> 1, J >> 1, I >> 1]


@wp.kernel
def k_coarsest(xv: wp.array4d(dtype=float), r: wp.array4d(dtype=float), diag: wp.array4d(dtype=float),
               dmin: float):
    c, K, J, I = wp.tid()
    d = diag[c, K, J, I]
    if d <= dmin:
        xv[c, K, J, I] = 0.0
    else:
        xv[c, K, J, I] = r[c, K, J, I] / d


@wp.kernel
def k_fwd_transform(r: wp.array3d(dtype=float), r1: wp.array4d(dtype=float)):
    K, J, I = wp.tid()
    v = wp.vector(dtype=float, length=8)
    for k in range(8):
        v[k] = r[2 * K + (k >> 2), 2 * J + ((k >> 1) & 1), 2 * I + (k & 1)]
    for c in range(8):
        acc = float(0.0)
        for k in range(8):
            acc += hsign(c, k >> 2, (k >> 1) & 1, k & 1) * v[k]
        r1[c, K, J, I] = acc


@wp.kernel
def k_inv_transform(x1: wp.array4d(dtype=float), fm: wp.array3d(dtype=int), base: wp.array3d(dtype=float),
                    z_out: wp.array3d(dtype=float)):
    z, y, x = wp.tid()
    if fm[z, y, x] == 0:
        z_out[z, y, x] = 0.0
        return
    K = z >> 1
    J = y >> 1
    I = x >> 1
    acc = base[z, y, x]
    for c in range(8):
        acc += hsign(c, z & 1, y & 1, x & 1) * x1[c, K, J, I]
    z_out[z, y, x] = acc


# ---------------------------------------------------------------- launch-count-reduced MG kernels
@wp.func
def red_init(r: float, d: float, dmin: float) -> float:
    """First (red) GS sweep from a zero guess: neighbours are all zero."""
    if d <= dmin:
        return 0.0
    return r / d


@wp.kernel
def k_smooth_h(xv: wp.array4d(dtype=float), r: wp.array4d(dtype=float), diag: wp.array4d(dtype=float),
               off: wp.array4d(dtype=float), color: int, dmin: float):
    """One colour of red-black GS, launched on (8, m, m, m/2): only cells of that colour."""
    c, K, J, Ih = wp.tid()
    I = 2 * Ih + ((color + J + K) & 1)
    d = diag[c, K, J, I]
    if d <= dmin:
        xv[c, K, J, I] = 0.0
        return
    xv[c, K, J, I] = (r[c, K, J, I] - offsum(xv, off, c, K, J, I)) / d


@wp.func
def xp(xv: wp.array4d(dtype=float), xc: wp.array4d(dtype=float), c: int, K: int, J: int, I: int) -> float:
    """Fine value after constant prolongation of the coarse correction."""
    return xv[c, K, J, I] + xc[c, K >> 1, J >> 1, I >> 1]


@wp.kernel
def k_prolong_black(xv: wp.array4d(dtype=float), r: wp.array4d(dtype=float), diag: wp.array4d(dtype=float),
                    off: wp.array4d(dtype=float), xc: wp.array4d(dtype=float), dmin: float):
    """Prolongation + the black post-sweep in one launch. A GS update does not depend on the
    cell's own old value, so black cells only need their (red) neighbours prolongated, and the
    red cells are overwritten by the following red sweep, which reads only black cells. Same
    result as prolong-all-then-sweep."""
    c, K, J, Ih = wp.tid()
    I = 2 * Ih + ((1 + J + K) & 1)
    d = diag[c, K, J, I]
    if d <= dmin:
        xv[c, K, J, I] = 0.0
        return
    m = xv.shape[1]
    a = r[c, K, J, I]
    if I + 1 < m:
        a -= off[c, K, J, I] * xp(xv, xc, c, K, J, I + 1)
    if I > 0:
        a -= off[c, K, J, I - 1] * xp(xv, xc, c, K, J, I - 1)
    if J + 1 < m:
        a -= off[8 + c, K, J, I] * xp(xv, xc, c, K, J + 1, I)
    if J > 0:
        a -= off[8 + c, K, J - 1, I] * xp(xv, xc, c, K, J - 1, I)
    if K + 1 < m:
        a -= off[16 + c, K, J, I] * xp(xv, xc, c, K + 1, J, I)
    if K > 0:
        a -= off[16 + c, K - 1, J, I] * xp(xv, xc, c, K - 1, J, I)
    xv[c, K, J, I] = a / d


@wp.kernel
def k_fwd_red(r: wp.array3d(dtype=float), r1: wp.array4d(dtype=float), x1: wp.array4d(dtype=float),
              diag: wp.array4d(dtype=float), dmin: float):
    """Hadamard forward transform to level 1 + the zero-guess red sweep there."""
    K, J, I = wp.tid()
    v = wp.vector(dtype=float, length=8)
    for k in range(8):
        v[k] = r[2 * K + (k >> 2), 2 * J + ((k >> 1) & 1), 2 * I + (k & 1)]
    red = ((I + J + K) & 1) == 0
    for c in range(8):
        acc = float(0.0)
        for k in range(8):
            acc += hsign(c, k >> 2, (k >> 1) & 1, k & 1) * v[k]
        r1[c, K, J, I] = acc
        if red:
            x1[c, K, J, I] = red_init(acc, diag[c, K, J, I], dmin)


@wp.kernel
def k_restrict_red(xv: wp.array4d(dtype=float), r: wp.array4d(dtype=float), diag: wp.array4d(dtype=float),
                   off: wp.array4d(dtype=float), rc: wp.array4d(dtype=float), xc: wp.array4d(dtype=float),
                   dc: wp.array4d(dtype=float), dminc: float, do_red: int):
    """Residual restriction (average of 8 children) + the zero-guess red sweep on the coarse level."""
    c, K, J, I = wp.tid()
    acc = float(0.0)
    for k in range(8):
        z = 2 * K + (k >> 2)
        y = 2 * J + ((k >> 1) & 1)
        x = 2 * I + (k & 1)
        acc += r[c, z, y, x] - diag[c, z, y, x] * xv[c, z, y, x] - offsum(xv, off, c, z, y, x)
    acc *= 0.125
    rc[c, K, J, I] = acc
    if do_red == 1 and ((I + J + K) & 1) == 0:
        xc[c, K, J, I] = red_init(acc, dc[c, K, J, I], dminc)


CB = 512        # threads of the single-block coarse V-cycle kernel


@wp.func
def block_sync():
    """Block-wide barrier (Warp exposes no __syncthreads; a tile reduction contains them, and
    __syncthreads also orders global-memory accesses within the block)."""
    wp.tile_sum(wp.tile(float(0.0)))


@wp.func
def cb_smooth(xv: wp.array4d(dtype=float), r: wp.array4d(dtype=float), diag: wp.array4d(dtype=float),
              off: wp.array4d(dtype=float), color: int, zero_init: int, dmin: float, t: int):
    m = xv.shape[1]
    n = 8 * m * m * m
    for i in range(t, n, CB):
        c = i // (m * m * m)
        K = (i // (m * m)) % m
        J = (i // m) % m
        I = i % m
        if ((I + J + K) & 1) == color:
            d = diag[c, K, J, I]
            v = float(0.0)
            if d > dmin:
                a = r[c, K, J, I]
                if zero_init == 0:
                    a -= offsum(xv, off, c, K, J, I)
                v = a / d
            xv[c, K, J, I] = v
    block_sync()


@wp.func
def cb_restrict(xv: wp.array4d(dtype=float), r: wp.array4d(dtype=float), diag: wp.array4d(dtype=float),
                off: wp.array4d(dtype=float), rc: wp.array4d(dtype=float), t: int):
    m = rc.shape[1]
    n = 8 * m * m * m
    for i in range(t, n, CB):
        c = i // (m * m * m)
        K = (i // (m * m)) % m
        J = (i // m) % m
        I = i % m
        acc = float(0.0)
        for k in range(8):
            z = 2 * K + (k >> 2)
            y = 2 * J + ((k >> 1) & 1)
            x = 2 * I + (k & 1)
            acc += r[c, z, y, x] - diag[c, z, y, x] * xv[c, z, y, x] - offsum(xv, off, c, z, y, x)
        rc[c, K, J, I] = 0.125 * acc
    block_sync()


@wp.func
def cb_prolong(xv: wp.array4d(dtype=float), diag: wp.array4d(dtype=float), xc: wp.array4d(dtype=float),
               dmin: float, t: int):
    m = xv.shape[1]
    n = 8 * m * m * m
    for i in range(t, n, CB):
        c = i // (m * m * m)
        K = (i // (m * m)) % m
        J = (i // m) % m
        I = i % m
        if diag[c, K, J, I] > dmin:
            xv[c, K, J, I] = xv[c, K, J, I] + xc[c, K >> 1, J >> 1, I >> 1]
    block_sync()


@wp.func
def cb_coarsest(xv: wp.array4d(dtype=float), r: wp.array4d(dtype=float), diag: wp.array4d(dtype=float),
                off: wp.array4d(dtype=float), dmin: float, t: int):
    m = xv.shape[1]
    if m == 1:
        if t < 8:
            d = diag[t, 0, 0, 0]
            v = float(0.0)
            if d > dmin:
                v = r[t, 0, 0, 0] / d
            xv[t, 0, 0, 0] = v
        block_sync()
    else:   # odd coarsest size: plain red-black smoothing, as in the reference V-cycle
        for k in range(2 * m):
            zi = int(0)
            if k == 0:
                zi = 1
            cb_smooth(xv, r, diag, off, 0, zi, dmin, t)
            cb_smooth(xv, r, diag, off, 1, 0, dmin, t)


@wp.kernel
def k_coarse_block(x0: wp.array4d(dtype=float), r0: wp.array4d(dtype=float), d0: wp.array4d(dtype=float),
                   o0: wp.array4d(dtype=float), m0: float,
                   x1: wp.array4d(dtype=float), r1: wp.array4d(dtype=float), d1: wp.array4d(dtype=float),
                   o1: wp.array4d(dtype=float), m1: float,
                   x2: wp.array4d(dtype=float), r2: wp.array4d(dtype=float), d2: wp.array4d(dtype=float),
                   o2: wp.array4d(dtype=float), m2: float,
                   x3: wp.array4d(dtype=float), r3: wp.array4d(dtype=float), d3: wp.array4d(dtype=float),
                   o3: wp.array4d(dtype=float), m3: float, nlev: int):
    """The bottom of the V-cycle (levels with m <= BLOCK_M, up to 4 of them; r of the first already
    restricted) in ONE launch of one thread block, block-wide barriers between sweeps. Same
    operations in the same order as the multi-launch reference V-cycle. Only worth it for tiny
    levels: one block has little latency hiding (m <= 2: 9 us vs ~20 us of launches; m <= 8: 67 us)."""
    _b, t = wp.tid()
    if nlev == 1:
        cb_coarsest(x0, r0, d0, o0, m0, t)
        return
    cb_smooth(x0, r0, d0, o0, 0, 1, m0, t)
    cb_smooth(x0, r0, d0, o0, 1, 0, m0, t)
    cb_restrict(x0, r0, d0, o0, r1, t)
    if nlev == 2:
        cb_coarsest(x1, r1, d1, o1, m1, t)
    else:
        cb_smooth(x1, r1, d1, o1, 0, 1, m1, t)
        cb_smooth(x1, r1, d1, o1, 1, 0, m1, t)
        cb_restrict(x1, r1, d1, o1, r2, t)
        if nlev == 3:
            cb_coarsest(x2, r2, d2, o2, m2, t)
        else:
            cb_smooth(x2, r2, d2, o2, 0, 1, m2, t)
            cb_smooth(x2, r2, d2, o2, 1, 0, m2, t)
            cb_restrict(x2, r2, d2, o2, r3, t)
            cb_coarsest(x3, r3, d3, o3, m3, t)
            cb_prolong(x2, d2, x3, m2, t)
            cb_smooth(x2, r2, d2, o2, 1, 0, m2, t)
            cb_smooth(x2, r2, d2, o2, 0, 0, m2, t)
        cb_prolong(x1, d1, x2, m1, t)
        cb_smooth(x1, r1, d1, o1, 1, 0, m1, t)
        cb_smooth(x1, r1, d1, o1, 0, 0, m1, t)
    cb_prolong(x0, d0, x1, m0, t)
    cb_smooth(x0, r0, d0, o0, 1, 0, m0, t)
    cb_smooth(x0, r0, d0, o0, 0, 0, m0, t)


# ---------------------------------------------------------------- fused flat-tiled kernels + finalizers
def _make_flat(N: int, block: int):
    total = N * N * N
    nb_target = 512
    cpt = max(1, -(-total // (nb_target * block)))      # cells per thread
    per_block = cpt * block
    NB = -(-total // per_block)

    @wp.kernel(module="unique")
    def k_rhs(v: wp.array4d(dtype=float), vc: int, fm: wp.array3d(dtype=int), r: wp.array3d(dtype=float),
              phi: wp.array3d(dtype=float), prm: wp.array(dtype=float), part: wp.array(dtype=float)):
        """r = -d(u*) - K phi0 (phi0 = previous solution if warm start, else phi = 0);
        partial max |u*| over fluid cells."""
        b, l = wp.tid()
        mx = float(0.0)
        warm = prm[P_WARM] > 0.5
        for k in range(cpt):
            i = b * per_block + k * block + l
            if i < total:
                q = unflat(i, N)
                if not warm:
                    phi[q[0], q[1], q[2]] = 0.0
                if fm[q[0], q[1], q[2]] == 0:
                    r[q[0], q[1], q[2]] = 0.0
                else:
                    rv = -div2(v, vc, fm, q[0], q[1], q[2])
                    if warm:
                        rv -= kmul(phi, fm, q[0], q[1], q[2])
                    r[q[0], q[1], q[2]] = rv
                    u = wp.vec3(v[vc, q[0], q[1], q[2]], v[vc + 1, q[0], q[1], q[2]], v[vc + 2, q[0], q[1], q[2]])
                    mx = wp.max(mx, wp.length(u))
        t = wp.tile_max(wp.tile(mx))
        if l == 0:
            part[b] = t[0]

    @wp.kernel(module="unique")
    def k_matvec_dot(p: wp.array3d(dtype=float), fm: wp.array3d(dtype=int), q_out: wp.array3d(dtype=float),
                     part: wp.array(dtype=float)):
        """q = K p, partial p.q"""
        b, l = wp.tid()
        s = float(0.0)
        for k in range(cpt):
            i = b * per_block + k * block + l
            if i < total:
                q = unflat(i, N)
                val = float(0.0)
                if fm[q[0], q[1], q[2]] != 0:
                    val = kmul(p, fm, q[0], q[1], q[2])
                q_out[q[0], q[1], q[2]] = val
                s += val * p[q[0], q[1], q[2]]
        t = wp.tile_sum(wp.tile(s))
        if l == 0:
            part[b] = t[0]

    @wp.kernel(module="unique")
    def k_pre(r: wp.array3d(dtype=float), q: wp.array3d(dtype=float), fm: wp.array3d(dtype=int),
              scal: wp.array(dtype=float), part_pq: wp.array(dtype=float), init: int, omega: float,
              z0: wp.array3d(dtype=float)):
        """r -= alpha q (skipped at init; alpha = rz / sum(part_pq), recomputed per block so no
        separate finalize launch is needed); z0 = omega D^-1 r (damped Jacobi from zero)."""
        b, l = wp.tid()
        a = float(0.0)
        if init == 0:
            pq = wp.tile_sum(wp.tile_load(part_pq, shape=NB, offset=0))[0]
            if pq > 0.0:
                a = scal[S_RZ] / pq
        for k in range(cpt):
            i = b * per_block + k * block + l
            if i < total:
                c = unflat(i, N)
                m = fm[c[0], c[1], c[2]]
                rv = r[c[0], c[1], c[2]]
                if init == 0:
                    rv = rv - a * q[c[0], c[1], c[2]]
                    r[c[0], c[1], c[2]] = rv
                zz = float(0.0)
                if m != 0:
                    zz = omega * rv / kdiag(m)
                z0[c[0], c[1], c[2]] = zz

    @wp.kernel(module="unique")
    def k_resid(r: wp.array3d(dtype=float), z0: wp.array3d(dtype=float), fm: wp.array3d(dtype=int),
                rr: wp.array3d(dtype=float)):
        b, l = wp.tid()
        for k in range(cpt):
            i = b * per_block + k * block + l
            if i < total:
                c = unflat(i, N)
                v = float(0.0)
                if fm[c[0], c[1], c[2]] != 0:
                    v = r[c[0], c[1], c[2]] - kmul(z0, fm, c[0], c[1], c[2])
                rr[c[0], c[1], c[2]] = v

    @wp.kernel(module="unique")
    def k_post_rz(zt: wp.array3d(dtype=float), r: wp.array3d(dtype=float), fm: wp.array3d(dtype=int),
                  omega: float, z_out: wp.array3d(dtype=float), part: wp.array(dtype=float)):
        """z = zt + omega D^-1 (r - K zt); partial r.z and max |r|."""
        b, l = wp.tid()
        s = float(0.0)
        mx = float(0.0)
        for k in range(cpt):
            i = b * per_block + k * block + l
            if i < total:
                c = unflat(i, N)
                m = fm[c[0], c[1], c[2]]
                zv = float(0.0)
                rv = r[c[0], c[1], c[2]]
                if m != 0:
                    zv = zt[c[0], c[1], c[2]] + omega * (rv - kmul(zt, fm, c[0], c[1], c[2])) / kdiag(m)
                z_out[c[0], c[1], c[2]] = zv
                s += rv * zv
                mx = wp.max(mx, wp.abs(rv))
        ts = wp.tile_sum(wp.tile(s))
        tm = wp.tile_max(wp.tile(mx))
        if l == 0:
            part[b] = ts[0]
            part[NB + b] = tm[0]

    @wp.kernel(module="unique")
    def k_upd_p(phi: wp.array3d(dtype=float), p: wp.array3d(dtype=float), z: wp.array3d(dtype=float),
                scal: wp.array(dtype=float), init: int):
        """phi += alpha p_old (skipped at init); p = z + beta p_old."""
        b, l = wp.tid()
        a = scal[S_ALPHA]
        be = scal[S_BETA]
        for k in range(cpt):
            i = b * per_block + k * block + l
            if i < total:
                c = unflat(i, N)
                po = p[c[0], c[1], c[2]]
                if init == 0:
                    phi[c[0], c[1], c[2]] = phi[c[0], c[1], c[2]] + a * po
                    p[c[0], c[1], c[2]] = z[c[0], c[1], c[2]] + be * po
                else:
                    p[c[0], c[1], c[2]] = z[c[0], c[1], c[2]]

    @wp.kernel(module="unique")
    def k_fin_maxu(part: wp.array(dtype=float), prm: wp.array(dtype=float), scal: wp.array(dtype=float)):
        m = wp.tile_max(wp.tile_load(part, shape=NB, offset=0))
        scal[S_MAXU] = m[0]
        # float32 floor: past ~1e-7 normalized CG loses orthogonality
        scal[S_THR] = 2.0 * wp.max(prm[P_TOL], 1.0e-6) * m[0]

    @wp.kernel(module="unique")
    def k_fin_rz(part: wp.array(dtype=float), part_pq: wp.array(dtype=float), init: int,
                 prm: wp.array(dtype=float), scal: wp.array(dtype=float), it: wp.array(dtype=int),
                 cond: wp.array(dtype=int)):
        rz_new = wp.tile_sum(wp.tile_load(part, shape=NB, offset=0))[0]
        mr = wp.tile_max(wp.tile_load(part, shape=NB, offset=NB))[0]
        n = int(0)
        if init == 1:
            scal[S_BETA] = 0.0
        else:
            rz = scal[S_RZ]
            pq = wp.tile_sum(wp.tile_load(part_pq, shape=NB, offset=0))[0]
            a = float(0.0)
            if pq > 0.0:
                a = rz / pq
            scal[S_ALPHA] = a          # same value k_pre used; consumed by k_upd_p
            bb = float(0.0)
            if rz > 0.0:
                bb = rz_new / rz
            scal[S_BETA] = bb
            n = it[0] + 1
        scal[S_RZ] = rz_new
        scal[S_MAXR] = mr
        it[0] = n
        best = mr
        if init == 0:
            best = wp.min(scal[S_BEST], mr)
        scal[S_BEST] = best
        # stop at tolerance (tol = 0: only the 1e-6 float32 floor, i.e. a fixed iteration count in
        # practice), max iters, or stagnation (residual 4x above the best seen: float32 CG past
        # convergence loses orthogonality and would otherwise blow the divergence back up).
        go = int(0)
        if n < int(prm[P_MAXIT]) and rz_new > 0.0 and mr > scal[S_THR] and mr <= 4.0 * best:
            go = 1
        cond[0] = go

    return dict(NB=NB, rhs=k_rhs, matvec_dot=k_matvec_dot, pre=k_pre, resid=k_resid, post_rz=k_post_rz,

                upd_p=k_upd_p, fin_maxu=k_fin_maxu, fin_rz=k_fin_rz)


# ---------------------------------------------------------------- driver
class PressureSolver3D:
    """Buffers + launch sequence for one resolution. `record*` only issue launches (graph-capturable)."""

    BLOCK = 256
    GALERKIN_SCALE = 1.0 / 16.0    # 1/2 of average-restriction Galerkin, 3D
    DMIN_REL = 1.0e-3
    OMEGA = 0.8                    # outer damped Jacobi on exact K (swept 0..0.9 at 64^3: 0.8 fewest iters)
    BLOCK_M = 2                    # levels with m <= BLOCK_M run in the single-block kernel


    def __init__(self, N: int, device, smooth_sweeps: int = 1, fast: bool = True):
        assert N % 2 == 0, "even grid size"
        self.N, self.device, self.nu, self.fast = N, device, smooth_sweeps, fast
        z = lambda shape, dt=float: wp.zeros(shape, dtype=dt, device=device)
        g = (N, N, N)
        self.fm = z(g, int)
        self.solid_cache = wp.full(g, -1.0, dtype=float, device=device)
        self.changed = z(1, int)
        self.r, self.z, self.p, self.q = z(g), z(g), z(g), z(g)
        self.phi, self.z0, self.rr, self.zt = z(g), z(g), z(g), z(g)
        self.lv = []
        m = N // 2
        level = 1
        while True:
            dmin = self.DMIN_REL * 6.0 * 0.25 ** (level - 1)
            self.lv.append(dict(m=m, x=z((8, m, m, m)), r=z((8, m, m, m)), diag=z((8, m, m, m)),
                                off=z((24, m, m, m)), dmin=dmin))
            if m % 2 == 1 or m == 1:
                break
            m //= 2
            level += 1
        # fast V-cycle: levels from the first with m <= 8 down run in one single-block kernel
        self.block_level = next((i for i, a in enumerate(self.lv) if a["m"] <= self.BLOCK_M), len(self.lv) - 1)
        if len(self.lv) - self.block_level > 4 or self.lv[self.block_level]["m"] > self.BLOCK_M or self.block_level == 0:
            self.fast = False      # unusual size: plain V-cycle
        self.fk = _make_flat(N, self.BLOCK)
        self.part = z(2 * self.fk["NB"])
        self.part_pq = z(self.fk["NB"])
        self.it = wp.zeros(1, dtype=int, device=device)
        self.cond = wp.zeros(1, dtype=int, device=device)

    # -- launch helpers
    def _flat(self, name, inputs):
        wp.launch_tiled(self.fk[name], dim=[self.fk["NB"]], inputs=inputs, block_dim=self.BLOCK, device=self.device)

    def _fin(self, name, inputs):
        wp.launch_tiled(self.fk[name], dim=[1], inputs=inputs, block_dim=self.BLOCK, device=self.device)

    def _grid(self, kernel, inputs):
        N = self.N
        wp.launch(kernel, dim=(N, N, N), inputs=inputs, device=self.device)

    # -- setup
    def _build_ops(self):
        L, d = self.lv, self.device
        wp.launch(k_probe1, dim=(8, L[0]["m"], L[0]["m"], L[0]["m"]), inputs=[self.fm, L[0]["diag"], L[0]["off"]],
                  device=d)
        for a, b in zip(L[:-1], L[1:]):
            wp.launch(k_galerkin, dim=(8, b["m"], b["m"], b["m"]),
                      inputs=[a["diag"], a["off"], self.GALERKIN_SCALE, b["diag"], b["off"]], device=d)

    def record_setup(self, solid, cached: bool = True):
        """Rebuild masks + MG operators if `solid` changed since the last call (GPU-side test)."""
        if not cached:
            self._grid(k_mask, [solid, self.fm])
            self._build_ops()
            return
        self.changed.zero_()
        self._grid(k_solid_changed, [solid, self.solid_cache, self.changed])

        def rebuild():
            self._grid(k_mask, [solid, self.fm])
            self._build_ops()
        wp.capture_if(self.changed, on_true=rebuild)

    # -- preconditioner
    def _smooth(self, a, color, zero_init):
        wp.launch(k_smooth, dim=(8, a["m"], a["m"], a["m"]),
                  inputs=[a["x"], a["r"], a["diag"], a["off"], color, zero_init, a["dmin"]], device=self.device)

    def _smooth_h(self, a, color):
        m = a["m"]
        wp.launch(k_smooth_h, dim=(8, m, m, m // 2), inputs=[a["x"], a["r"], a["diag"], a["off"], color, a["dmin"]],
                  device=self.device)

    def _vcycle(self, scal, init):
        if self.fast:
            self._vcycle_fast(scal, init)
        else:
            self._vcycle_base(scal, init)

    def _vcycle_fast(self, scal, init):
        """Same V(1,1) as _vcycle_base with fewer launches: half-size red/black launches, the
        zero-guess red sweep fused into the transform / restriction, prolongation fused into the
        black post-sweep, and every level with m <= BLOCK_M run inside one single-block kernel."""
        d, L, om = self.device, self.lv, self.OMEGA
        S = self.block_level
        self._flat("pre", [self.r, self.q, self.fm, scal, self.part_pq, init, om, self.z0])
        self._flat("resid", [self.r, self.z0, self.fm, self.rr])
        m0 = L[0]["m"]
        wp.launch(k_fwd_red, dim=(m0, m0, m0), inputs=[self.rr, L[0]["r"], L[0]["x"], L[0]["diag"], L[0]["dmin"]],
                  device=d)
        for l in range(S):
            a, b = L[l], L[l + 1]
            self._smooth_h(a, 1)
            wp.launch(k_restrict_red, dim=(8, b["m"], b["m"], b["m"]),
                      inputs=[a["x"], a["r"], a["diag"], a["off"], b["r"], b["x"], b["diag"], b["dmin"],
                              0 if l + 1 == S else 1], device=d)
        lv = L[S:]
        args = []
        for k in range(4):
            c = lv[min(k, len(lv) - 1)]
            args += [c["x"], c["r"], c["diag"], c["off"], c["dmin"]]
        wp.launch_tiled(k_coarse_block, dim=[1], inputs=args + [len(lv)], block_dim=CB, device=d)
        for l in range(S - 1, -1, -1):
            a, b = L[l], L[l + 1]
            m = a["m"]
            wp.launch(k_prolong_black, dim=(8, m, m, m // 2),
                      inputs=[a["x"], a["r"], a["diag"], a["off"], b["x"], a["dmin"]], device=d)
            self._smooth_h(a, 0)
        self._grid(k_inv_transform, [L[0]["x"], self.fm, self.z0, self.zt])
        self._flat("post_rz", [self.zt, self.r, self.fm, om, self.z, self.part])

    def _vcycle_base(self, scal, init):
        """z = M^-1 r (with the pending r -= alpha q folded into the first kernel)."""
        d, nu, L, om = self.device, self.nu, self.lv, self.OMEGA
        self._flat("pre", [self.r, self.q, self.fm, scal, self.part_pq, init, om, self.z0])
        self._flat("resid", [self.r, self.z0, self.fm, self.rr])
        m0 = L[0]["m"]
        wp.launch(k_fwd_transform, dim=(m0, m0, m0), inputs=[self.rr, L[0]["r"]], device=d)
        for l in range(len(L) - 1):
            a, b = L[l], L[l + 1]
            for k in range(nu):
                self._smooth(a, 0, 1 if k == 0 else 0)
                self._smooth(a, 1, 0)
            wp.launch(k_resrestrict, dim=(8, b["m"], b["m"], b["m"]),
                      inputs=[a["x"], a["r"], a["diag"], a["off"], b["r"]], device=d)
        c = L[-1]
        if c["m"] == 1:
            wp.launch(k_coarsest, dim=(8, 1, 1, 1), inputs=[c["x"], c["r"], c["diag"], c["dmin"]], device=d)
        else:
            for k in range(2 * c["m"]):
                self._smooth(c, 0, 1 if k == 0 else 0)
                self._smooth(c, 1, 0)
        for l in range(len(L) - 2, -1, -1):
            a, b = L[l], L[l + 1]
            wp.launch(k_prolong, dim=(8, a["m"], a["m"], a["m"]), inputs=[a["x"], a["diag"], b["x"], a["dmin"]],
                      device=d)
            for k in range(nu):
                self._smooth(a, 1, 0)
                self._smooth(a, 0, 0)
        self._grid(k_inv_transform, [L[0]["x"], self.fm, self.z0, self.zt])
        self._flat("post_rz", [self.zt, self.r, self.fm, om, self.z, self.part])

    # -- CG
    def _body(self, prm, scal):
        self._flat("matvec_dot", [self.p, self.fm, self.q, self.part_pq])
        self._vcycle(scal, 0)
        self._fin("fin_rz", [self.part, self.part_pq, 0, prm, scal, self.it, self.cond])
        self._flat("upd_p", [self.phi, self.p, self.z, scal, 0])

    def record(self, vel, vc: int, solid, prm, scal, cached: bool = True):
        """Project vel[vc:vc+3] in place."""
        self.record_setup(solid, cached)
        self._flat("rhs", [vel, vc, self.fm, self.r, self.phi, prm, self.part])
        self._fin("fin_maxu", [self.part, prm, scal])
        self._vcycle(scal, 1)
        self._fin("fin_rz", [self.part, self.part_pq, 1, prm, scal, self.it, self.cond])
        self._flat("upd_p", [self.phi, self.p, self.z, scal, 1])
        wp.capture_while(self.cond, self._body, prm=prm, scal=scal)
        # the last body leaves phi += alpha p pending only inside upd_p, which ran: phi is final
        self._grid(k_apply, [vel, vc, self.fm, self.phi])

    def record_div(self, vel, vc: int, solid, out):
        if not hasattr(self, "fm_div"):
            self.fm_div = wp.zeros((self.N,) * 3, dtype=int, device=self.device)
        self._grid(k_mask, [solid, self.fm_div])
        self._grid(k_div_abs, [vel, vc, self.fm_div, out])
