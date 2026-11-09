"""Exact collocated pressure projection: MG-preconditioned conjugate gradient in Warp.

Discretization (see README):
  face velocity  = average of the two adjacent cell velocities, 0 on walls / solid faces
  d(u)_i         = sum over open faces of (+/-)(u_i + u_nb)          (= 2 dx * div u)
  g(phi)_i       = sum over open faces of (phi_nb - phi_i) one-sided  (= -d^T, the adjoint)
  solve  K phi = -d(u*),  K = -d g = d d^T (SPD on fluid cells, null space = constants)
  update u = u* - g(phi)          => d(u) = 0 to solver tolerance (exact central divergence).

K is the "wide" (2 dx) Laplacian: in the interior it decouples into four parity subgrids,
coupled only at walls / obstacles. The preconditioner is a symmetric multigrid V-cycle in a
2x2-block Hadamard basis, which block-diagonalises K exactly in a box (see the MG section).

Iteration runs inside a CUDA-graph conditional while loop: stops when max|d(u)|/(2 max|u*|)
< pressure_tol or after pressure_max_iters. No host synchronisation.
"""
from __future__ import annotations

import warp as wp

from .kernels import P_MAXIT, P_TOL, S_ALPHA, S_BEST, S_BETA, S_MAXR, S_MAXU, S_RZ, S_THR


# ---------------------------------------------------------------- helpers
@wp.func
def isf(fl: wp.array2d(dtype=int), y: int, x: int) -> int:
    if x < 0 or y < 0 or y >= fl.shape[0] or x >= fl.shape[1]:
        return 0
    return fl[y, x]


@wp.func
def gx_at(p: wp.array2d(dtype=float), fl: wp.array2d(dtype=int), y: int, x: int) -> float:
    g = float(0.0)
    if isf(fl, y, x + 1) == 1:
        g += p[y, x + 1] - p[y, x]
    if isf(fl, y, x - 1) == 1:
        g += p[y, x] - p[y, x - 1]
    return g


@wp.func
def gy_at(p: wp.array2d(dtype=float), fl: wp.array2d(dtype=int), y: int, x: int) -> float:
    g = float(0.0)
    if isf(fl, y + 1, x) == 1:
        g += p[y + 1, x] - p[y, x]
    if isf(fl, y - 1, x) == 1:
        g += p[y, x] - p[y - 1, x]
    return g


@wp.func
def div2(v: wp.array3d(dtype=float), vc: int, fl: wp.array2d(dtype=int), y: int, x: int) -> float:
    """d(u)_i = 2 dx div u (face-averaged central divergence, closed walls/solid faces)."""
    u = v[vc, y, x]
    w = v[vc + 1, y, x]
    d = float(0.0)
    if isf(fl, y, x + 1) == 1:
        d += u + v[vc, y, x + 1]
    if isf(fl, y, x - 1) == 1:
        d -= u + v[vc, y, x - 1]
    if isf(fl, y + 1, x) == 1:
        d += w + v[vc + 1, y + 1, x]
    if isf(fl, y - 1, x) == 1:
        d -= w + v[vc + 1, y - 1, x]
    return d


# ---------------------------------------------------------------- masks
@wp.kernel
def k_mask0(solid: wp.array2d(dtype=float), fl: wp.array2d(dtype=int)):
    y, x = wp.tid()
    if solid[y, x] > 0.5:
        fl[y, x] = 0
    else:
        fl[y, x] = 1


# ---------------------------------------------------------------- operator kernels
@wp.kernel
def k_rhs(v: wp.array3d(dtype=float), vc: int, fl: wp.array2d(dtype=int),
          r: wp.array2d(dtype=float), phi: wp.array2d(dtype=float), sp: wp.array2d(dtype=float)):
    y, x = wp.tid()
    phi[y, x] = 0.0
    if fl[y, x] == 0:
        r[y, x] = 0.0
        sp[y, x] = 0.0
        return
    r[y, x] = -div2(v, vc, fl, y, x)
    a = v[vc, y, x]
    b = v[vc + 1, y, x]
    sp[y, x] = wp.sqrt(a * a + b * b)


@wp.func
def kmul(p: wp.array2d(dtype=float), fl: wp.array2d(dtype=int), y: int, x: int) -> float:
    """(K p)_i = -d(g(p))_i for a fluid cell."""
    gxi = gx_at(p, fl, y, x)
    gyi = gy_at(p, fl, y, x)
    d = float(0.0)
    if isf(fl, y, x + 1) == 1:
        d += gxi + gx_at(p, fl, y, x + 1)
    if isf(fl, y, x - 1) == 1:
        d -= gxi + gx_at(p, fl, y, x - 1)
    if isf(fl, y + 1, x) == 1:
        d += gyi + gy_at(p, fl, y + 1, x)
    if isf(fl, y - 1, x) == 1:
        d -= gyi + gy_at(p, fl, y - 1, x)
    return -d


@wp.func
def kdiag(fl: wp.array2d(dtype=int), y: int, x: int) -> float:
    """diag(K) = sum over axes of (oL - oR)^2 + oL + oR."""
    oR = float(isf(fl, y, x + 1))
    oL = float(isf(fl, y, x - 1))
    oT = float(isf(fl, y + 1, x))
    oB = float(isf(fl, y - 1, x))
    return (oL - oR) * (oL - oR) + oL + oR + (oB - oT) * (oB - oT) + oB + oT


@wp.kernel
def k_matvec(p: wp.array2d(dtype=float), fl: wp.array2d(dtype=int), q: wp.array2d(dtype=float)):
    y, x = wp.tid()
    if fl[y, x] == 0:
        q[y, x] = 0.0
        return
    q[y, x] = kmul(p, fl, y, x)


@wp.kernel
def k_jac_pre(r: wp.array2d(dtype=float), fl: wp.array2d(dtype=int), omega: float, z0: wp.array2d(dtype=float)):
    """z0 = omega D^-1 r (one damped-Jacobi sweep on the exact K from a zero guess)."""
    y, x = wp.tid()
    dg = kdiag(fl, y, x)
    if fl[y, x] == 0 or dg == 0.0:
        z0[y, x] = 0.0
        return
    z0[y, x] = omega * r[y, x] / dg


@wp.kernel
def k_resid(r: wp.array2d(dtype=float), z0: wp.array2d(dtype=float), fl: wp.array2d(dtype=int),
            rr: wp.array2d(dtype=float)):
    y, x = wp.tid()
    if fl[y, x] == 0:
        rr[y, x] = 0.0
        return
    rr[y, x] = r[y, x] - kmul(z0, fl, y, x)


@wp.kernel
def k_jac_post(zin: wp.array2d(dtype=float), r: wp.array2d(dtype=float), fl: wp.array2d(dtype=int),
               omega: float, zout: wp.array2d(dtype=float)):
    y, x = wp.tid()
    dg = kdiag(fl, y, x)
    if fl[y, x] == 0 or dg == 0.0:
        zout[y, x] = 0.0
        return
    zout[y, x] = zin[y, x] + omega * (r[y, x] - kmul(zin, fl, y, x)) / dg


@wp.kernel
def k_apply(v: wp.array3d(dtype=float), vc: int, fl: wp.array2d(dtype=int),
            phi: wp.array2d(dtype=float)):
    y, x = wp.tid()
    if fl[y, x] == 0:
        v[vc, y, x] = 0.0
        v[vc + 1, y, x] = 0.0
        return
    v[vc, y, x] = v[vc, y, x] - gx_at(phi, fl, y, x)
    v[vc + 1, y, x] = v[vc + 1, y, x] - gy_at(phi, fl, y, x)


@wp.kernel
def k_div_abs(v: wp.array3d(dtype=float), vc: int, fl: wp.array2d(dtype=int),
              out: wp.array2d(dtype=float)):
    """|div u| * dx  (= |d(u)| / 2), 0 in solids."""
    y, x = wp.tid()
    if fl[y, x] == 0:
        out[y, x] = 0.0
        return
    out[y, x] = 0.5 * wp.abs(div2(v, vc, fl, y, x))


@wp.kernel
def k_upd_xr(phi: wp.array2d(dtype=float), r: wp.array2d(dtype=float), p: wp.array2d(dtype=float),
             q: wp.array2d(dtype=float), scal: wp.array(dtype=float)):
    y, x = wp.tid()
    a = scal[S_ALPHA]
    phi[y, x] = phi[y, x] + a * p[y, x]
    r[y, x] = r[y, x] - a * q[y, x]


@wp.kernel
def k_upd_p(p: wp.array2d(dtype=float), z: wp.array2d(dtype=float), scal: wp.array(dtype=float)):
    y, x = wp.tid()
    b = scal[S_BETA]
    if b == 0.0:
        p[y, x] = z[y, x]
    else:
        p[y, x] = z[y, x] + b * p[y, x]


# ---------------------------------------------------------------- multigrid preconditioner
# Level 1 lives on 2x2 blocks of the full grid in a Hadamard basis: component c = (cy, cx),
# value = 1/2 sum_{dy,dx} (-1)^(cx dx + cy dy) p[2J+dy, 2I+dx]. In a box this transform makes
# K block-diagonal EXACTLY: the (+,+) component is a Neumann 5-point Laplacian, components
# with a parity difference along an axis see Dirichlet walls along that axis. With obstacles
# we keep each component's exact diagonal block (probed from K) and drop the weak
# inter-component coupling. Coarser levels: Galerkin with constant prolongation, scaled 1/2
# (equivalent to rediscretisation for Laplacians). Arrays are stacked (4, m, m).
@wp.func
def hsign(c: int, dy: int, dx: int) -> float:
    e = (c & 1) * dx + (c >> 1) * dy
    if (e & 1) == 1:
        return -0.5
    return 0.5


@wp.func
def tv(c: int, J: int, I: int, y: int, x: int) -> float:
    if (y >> 1) == J and (x >> 1) == I:
        return hsign(c, y & 1, x & 1)
    return 0.0


@wp.func
def gxt(fl: wp.array2d(dtype=int), c: int, J: int, I: int, y: int, x: int) -> float:
    if isf(fl, y, x) == 0:
        return 0.0
    g = float(0.0)
    if isf(fl, y, x + 1) == 1:
        g += tv(c, J, I, y, x + 1) - tv(c, J, I, y, x)
    if isf(fl, y, x - 1) == 1:
        g += tv(c, J, I, y, x) - tv(c, J, I, y, x - 1)
    return g


@wp.func
def gyt(fl: wp.array2d(dtype=int), c: int, J: int, I: int, y: int, x: int) -> float:
    if isf(fl, y, x) == 0:
        return 0.0
    g = float(0.0)
    if isf(fl, y + 1, x) == 1:
        g += tv(c, J, I, y + 1, x) - tv(c, J, I, y, x)
    if isf(fl, y - 1, x) == 1:
        g += tv(c, J, I, y, x) - tv(c, J, I, y - 1, x)
    return g


@wp.kernel
def k_probe1(fl: wp.array2d(dtype=int), diag: wp.array3d(dtype=float), offE: wp.array3d(dtype=float),
             offN: wp.array3d(dtype=float)):
    """Exact component-diagonal block of T K T^T on the 2x2-block grid (K = g^T g)."""
    c, J, I = wp.tid()
    m = diag.shape[1]
    dsum = float(0.0)
    for dy in range(2):
        for k in range(4):
            a = gxt(fl, c, J, I, 2 * J + dy, 2 * I - 1 + k)
            dsum += a * a
    for dx in range(2):
        for k in range(4):
            a = gyt(fl, c, J, I, 2 * J - 1 + k, 2 * I + dx)
            dsum += a * a
    e = float(0.0)
    if I + 1 < m:
        for dy in range(2):
            for k in range(2):
                y = 2 * J + dy
                x = 2 * I + 1 + k
                e += gxt(fl, c, J, I, y, x) * gxt(fl, c, J, I + 1, y, x)
    n = float(0.0)
    if J + 1 < m:
        for dx in range(2):
            for k in range(2):
                y = 2 * J + 1 + k
                x = 2 * I + dx
                n += gyt(fl, c, J, I, y, x) * gyt(fl, c, J + 1, I, y, x)
    diag[c, J, I] = dsum
    offE[c, J, I] = e
    offN[c, J, I] = n


@wp.kernel
def k_galerkin(diag: wp.array3d(dtype=float), offE: wp.array3d(dtype=float), offN: wp.array3d(dtype=float),
               scale: float, dc: wp.array3d(dtype=float), ec: wp.array3d(dtype=float), nc: wp.array3d(dtype=float)):
    c, J, I = wp.tid()
    y = 2 * J
    x = 2 * I
    d = diag[c, y, x] + diag[c, y, x + 1] + diag[c, y + 1, x] + diag[c, y + 1, x + 1]
    d += 2.0 * (offE[c, y, x] + offE[c, y + 1, x] + offN[c, y, x] + offN[c, y, x + 1])
    dc[c, J, I] = d * scale
    ec[c, J, I] = (offE[c, y, x + 1] + offE[c, y + 1, x + 1]) * scale
    nc[c, J, I] = (offN[c, y + 1, x] + offN[c, y + 1, x + 1]) * scale


@wp.kernel
def k_fwd_transform(r: wp.array2d(dtype=float), r1: wp.array3d(dtype=float)):
    c, J, I = wp.tid()
    acc = float(0.0)
    for dy in range(2):
        for dx in range(2):
            acc += hsign(c, dy, dx) * r[2 * J + dy, 2 * I + dx]
    r1[c, J, I] = acc


@wp.kernel
def k_inv_transform(x1: wp.array3d(dtype=float), fl: wp.array2d(dtype=int), base: wp.array2d(dtype=float),
                    z: wp.array2d(dtype=float)):
    y, x = wp.tid()
    if fl[y, x] == 0:
        z[y, x] = 0.0
        return
    J = y >> 1
    I = x >> 1
    acc = base[y, x]
    for c in range(4):
        acc += hsign(c, y & 1, x & 1) * x1[c, J, I]
    z[y, x] = acc


@wp.func
def offsum(xv: wp.array3d(dtype=float), offE: wp.array3d(dtype=float), offN: wp.array3d(dtype=float),
           c: int, J: int, I: int) -> float:
    m = xv.shape[1]
    a = float(0.0)
    if I + 1 < m:
        a += offE[c, J, I] * xv[c, J, I + 1]
    if I > 0:
        a += offE[c, J, I - 1] * xv[c, J, I - 1]
    if J + 1 < m:
        a += offN[c, J, I] * xv[c, J + 1, I]
    if J > 0:
        a += offN[c, J - 1, I] * xv[c, J - 1, I]
    return a


@wp.kernel
def k_smooth(xv: wp.array3d(dtype=float), r: wp.array3d(dtype=float), diag: wp.array3d(dtype=float),
             offE: wp.array3d(dtype=float), offN: wp.array3d(dtype=float), color: int, zero_init: int,
             dmin: float):
    c, J, I = wp.tid()
    if ((I + J) & 1) != color:
        return
    d = diag[c, J, I]
    if d <= dmin:
        xv[c, J, I] = 0.0
        return
    acc = r[c, J, I]
    if zero_init == 0:
        acc -= offsum(xv, offE, offN, c, J, I)
    xv[c, J, I] = acc / d


@wp.kernel
def k_resrestrict(xv: wp.array3d(dtype=float), r: wp.array3d(dtype=float), diag: wp.array3d(dtype=float),
                  offE: wp.array3d(dtype=float), offN: wp.array3d(dtype=float), rc: wp.array3d(dtype=float)):
    c, J, I = wp.tid()
    acc = float(0.0)
    for dj in range(2):
        for di in range(2):
            y = 2 * J + dj
            x = 2 * I + di
            acc += r[c, y, x] - diag[c, y, x] * xv[c, y, x] - offsum(xv, offE, offN, c, y, x)
    rc[c, J, I] = 0.25 * acc


@wp.kernel
def k_prolong(xv: wp.array3d(dtype=float), diag: wp.array3d(dtype=float), xc: wp.array3d(dtype=float),
              dmin: float):
    c, J, I = wp.tid()
    if diag[c, J, I] <= dmin:
        return
    xv[c, J, I] = xv[c, J, I] + xc[c, J >> 1, I >> 1]


@wp.kernel
def k_coarsest(xv: wp.array3d(dtype=float), r: wp.array3d(dtype=float), diag: wp.array3d(dtype=float),
               dmin: float):
    c, J, I = wp.tid()
    d = diag[c, J, I]
    if d <= dmin:
        xv[c, J, I] = 0.0
    else:
        xv[c, J, I] = r[c, J, I] / d


# ---------------------------------------------------------------- deterministic tile reductions
def _make_reductions(H: int, W: int):
    ROWS = max(1, 1024 // W)
    NB = (H + ROWS - 1) // ROWS

    @wp.kernel(module="unique")
    def k_part_max(a: wp.array2d(dtype=float), part: wp.array(dtype=float)):
        i = wp.tid()
        t = wp.tile_load(a, shape=(ROWS, W), offset=(i * ROWS, 0))
        m = wp.tile_max(t)
        part[i] = m[0]

    @wp.kernel(module="unique")
    def k_fin_maxu(part: wp.array(dtype=float), prm: wp.array(dtype=float), scal: wp.array(dtype=float)):
        t = wp.tile_load(part, shape=NB, offset=0)
        m = wp.tile_max(t)
        scal[S_MAXU] = m[0]
        # float32 floor: past ~1e-7 normalized, CG loses orthogonality and diverges
        scal[S_THR] = 2.0 * wp.max(prm[P_TOL], 1.0e-6) * m[0]

    @wp.kernel(module="unique")
    def k_part_dot(a: wp.array2d(dtype=float), b: wp.array2d(dtype=float), part: wp.array(dtype=float)):
        i = wp.tid()
        ta = wp.tile_load(a, shape=(ROWS, W), offset=(i * ROWS, 0))
        tb = wp.tile_load(b, shape=(ROWS, W), offset=(i * ROWS, 0))
        s = wp.tile_sum(wp.tile_map(wp.mul, ta, tb))
        part[i] = s[0]

    @wp.kernel(module="unique")
    def k_fin_alpha(part: wp.array(dtype=float), scal: wp.array(dtype=float)):
        t = wp.tile_load(part, shape=NB, offset=0)
        s = wp.tile_sum(t)
        pq = s[0]
        a = float(0.0)
        if pq > 0.0:
            a = scal[S_RZ] / pq
        scal[S_ALPHA] = a

    @wp.kernel(module="unique")
    def k_part_rz(r: wp.array2d(dtype=float), z: wp.array2d(dtype=float), part: wp.array(dtype=float)):
        i = wp.tid()
        tr = wp.tile_load(r, shape=(ROWS, W), offset=(i * ROWS, 0))
        tz = wp.tile_load(z, shape=(ROWS, W), offset=(i * ROWS, 0))
        s = wp.tile_sum(wp.tile_map(wp.mul, tr, tz))
        m = wp.tile_max(wp.tile_map(wp.abs, tr))
        part[i] = s[0]
        part[NB + i] = m[0]

    @wp.kernel(module="unique")
    def k_fin_rz(part: wp.array(dtype=float), init: int, prm: wp.array(dtype=float),
                 scal: wp.array(dtype=float), it: wp.array(dtype=int), cond: wp.array(dtype=int)):
        ts = wp.tile_load(part, shape=NB, offset=0)
        tm = wp.tile_load(part, shape=NB, offset=NB)
        s = wp.tile_sum(ts)
        m = wp.tile_max(tm)
        rz_new = s[0]
        n = int(0)
        if init == 1:
            scal[S_BETA] = 0.0
        else:
            rz = scal[S_RZ]
            b = float(0.0)
            if rz > 0.0:
                b = rz_new / rz
            scal[S_BETA] = b
            n = it[0] + 1
        scal[S_RZ] = rz_new
        scal[S_MAXR] = m[0]
        it[0] = n
        # stagnation guard: in float32 the recursive residual eventually stops falling and CG
        # loses orthogonality; stop as soon as it climbs 4x above the best value seen.
        best = m[0]
        if init == 0:
            best = wp.min(scal[S_BEST], m[0])
        scal[S_BEST] = best
        go = int(0)
        if n < int(prm[P_MAXIT]) and m[0] > scal[S_THR] and rz_new > 0.0 and m[0] <= 4.0 * best:
            go = 1
        cond[0] = go

    return dict(ROWS=ROWS, NB=NB, part_max=k_part_max, fin_maxu=k_fin_maxu, part_dot=k_part_dot,
                fin_alpha=k_fin_alpha, part_rz=k_part_rz, fin_rz=k_fin_rz)


# ---------------------------------------------------------------- driver
# ---------------------------------------------------------------- driver
class PressureSolver:
    """Buffers + launch sequence for one resolution. `record` issues launches (graph-capturable)."""

    BLOCK = 256
    GALERKIN_SCALE = 1.0 / 8.0     # 1/2 of (average-restriction) Galerkin
    DMIN_REL = 1.0e-3              # coarse diagonals below this (x interior diag) are null-space
    OMEGA = 0.6                    # full-grid damped-Jacobi weight (0 disables the outer sweeps)

    def __init__(self, H: int, W: int, device, smooth_sweeps: int = 1):
        assert H == W and H % 2 == 0, "square grid with even size"
        self.H, self.W, self.device = H, W, device
        self.nu = smooth_sweeps
        z = lambda shape, dt=float: wp.zeros(shape, dtype=dt, device=device)
        self.fl = z((H, W), int)
        self.r, self.z, self.p, self.q = z((H, W)), z((H, W)), z((H, W)), z((H, W))
        self.phi, self.sp = z((H, W)), z((H, W))
        self.z0, self.rr, self.zt, self.zero = z((H, W)), z((H, W)), z((H, W)), z((H, W))
        self.lv = []
        m = H // 2
        level = 1
        while True:
            # interior diag of level 1 is 4 (unit links); each coarsening divides it by 4
            dmin = self.DMIN_REL * 4.0 * 0.25 ** (level - 1)
            self.lv.append(dict(m=m, x=z((4, m, m)), r=z((4, m, m)), diag=z((4, m, m)),
                                offE=z((4, m, m)), offN=z((4, m, m)), dmin=dmin))
            if m % 2 == 1 or m == 1:
                break
            m //= 2
            level += 1
        self.red = _make_reductions(H, W)
        self.part = z(2 * self.red["NB"])
        self.it = wp.zeros(1, dtype=int, device=device)
        self.cond = wp.zeros(1, dtype=int, device=device)

    def _red(self, kernel, inputs):
        wp.launch_tiled(kernel, dim=[self.red["NB"]], inputs=inputs, block_dim=self.BLOCK, device=self.device)

    def _fin(self, kernel, inputs):
        wp.launch_tiled(kernel, dim=[1], inputs=inputs, block_dim=self.BLOCK, device=self.device)

    def record_masks(self, solid):
        wp.launch(k_mask0, dim=(self.H, self.W), inputs=[solid, self.fl], device=self.device)

    def record_setup(self, solid):
        d = self.device
        self.record_masks(solid)
        L = self.lv
        wp.launch(k_probe1, dim=(4, L[0]["m"], L[0]["m"]),
                  inputs=[self.fl, L[0]["diag"], L[0]["offE"], L[0]["offN"]], device=d)
        for a, b in zip(L[:-1], L[1:]):
            wp.launch(k_galerkin, dim=(4, b["m"], b["m"]),
                      inputs=[a["diag"], a["offE"], a["offN"], self.GALERKIN_SCALE, b["diag"], b["offE"], b["offN"]],
                      device=d)

    def _smooth(self, a, color, zero_init):
        wp.launch(k_smooth, dim=(4, a["m"], a["m"]),
                  inputs=[a["x"], a["r"], a["diag"], a["offE"], a["offN"], color, zero_init, a["dmin"]],
                  device=self.device)

    def _vcycle(self, r, x):
        """x = M^-1 r: damped Jacobi on the exact K (fixes the local error the component-diagonal
        approximation makes at solid boundaries), symmetric V-cycle in the block-Hadamard basis,
        then the adjoint Jacobi sweep. Symmetric, so valid as a CG preconditioner."""
        d, nu, L = self.device, self.nu, self.lv
        H, W, om = self.H, self.W, self.OMEGA
        if om > 0.0:
            wp.launch(k_jac_pre, dim=(H, W), inputs=[r, self.fl, om, self.z0], device=d)
            wp.launch(k_resid, dim=(H, W), inputs=[r, self.z0, self.fl, self.rr], device=d)
            rin, base, xout = self.rr, self.z0, self.zt
        else:
            rin, base, xout = r, self.zero, x
        wp.launch(k_fwd_transform, dim=(4, L[0]["m"], L[0]["m"]), inputs=[rin, L[0]["r"]], device=d)
        for l in range(len(L) - 1):
            a, b = L[l], L[l + 1]
            for k in range(nu):
                self._smooth(a, 0, 1 if k == 0 else 0)
                self._smooth(a, 1, 0)
            wp.launch(k_resrestrict, dim=(4, b["m"], b["m"]),
                      inputs=[a["x"], a["r"], a["diag"], a["offE"], a["offN"], b["r"]], device=d)
        c = L[-1]
        if c["m"] == 1:
            wp.launch(k_coarsest, dim=(4, 1, 1), inputs=[c["x"], c["r"], c["diag"], c["dmin"]], device=d)
        else:  # odd coarsest size: plain smoothing
            for k in range(2 * c["m"]):
                self._smooth(c, 0, 1 if k == 0 else 0)
                self._smooth(c, 1, 0)
        for l in range(len(L) - 2, -1, -1):
            a, b = L[l], L[l + 1]
            wp.launch(k_prolong, dim=(4, a["m"], a["m"]), inputs=[a["x"], a["diag"], b["x"], a["dmin"]], device=d)
            for k in range(nu):
                self._smooth(a, 1, 0)
                self._smooth(a, 0, 0)
        wp.launch(k_inv_transform, dim=(H, W), inputs=[L[0]["x"], self.fl, base, xout], device=d)
        if om > 0.0:
            wp.launch(k_jac_post, dim=(H, W), inputs=[self.zt, r, self.fl, om, x], device=d)

    def _body(self, prm, scal):
        H, W, d, R = self.H, self.W, self.device, self.red
        wp.launch(k_matvec, dim=(H, W), inputs=[self.p, self.fl, self.q], device=d)
        self._red(R["part_dot"], [self.p, self.q, self.part])
        self._fin(R["fin_alpha"], [self.part, scal])
        wp.launch(k_upd_xr, dim=(H, W), inputs=[self.phi, self.r, self.p, self.q, scal], device=d)
        self._vcycle(self.r, self.z)
        self._red(R["part_rz"], [self.r, self.z, self.part])
        self._fin(R["fin_rz"], [self.part, 0, prm, scal, self.it, self.cond])
        wp.launch(k_upd_p, dim=(H, W), inputs=[self.p, self.z, scal], device=d)

    def record_init(self, vel, vc: int, solid, prm, scal):
        H, W, d, R = self.H, self.W, self.device, self.red
        self.record_setup(solid)
        wp.launch(k_rhs, dim=(H, W), inputs=[vel, vc, self.fl, self.r, self.phi, self.sp], device=d)
        self._red(R["part_max"], [self.sp, self.part])
        self._fin(R["fin_maxu"], [self.part, prm, scal])
        self._vcycle(self.r, self.z)
        self._red(R["part_rz"], [self.r, self.z, self.part])
        self._fin(R["fin_rz"], [self.part, 1, prm, scal, self.it, self.cond])
        wp.launch(k_upd_p, dim=(H, W), inputs=[self.p, self.z, scal], device=d)

    def record(self, vel, vc: int, solid, prm, scal):
        """Project vel[vc:vc+2] in place (masks / MG operators rebuilt from `solid` each call)."""
        self.record_init(vel, vc, solid, prm, scal)
        wp.capture_while(self.cond, self._body, prm=prm, scal=scal)
        wp.launch(k_apply, dim=(self.H, self.W), inputs=[vel, vc, self.fl, self.phi], device=self.device)

    def record_div(self, vel, vc: int, solid, out):
        self.record_masks(solid)
        wp.launch(k_div_abs, dim=(self.H, self.W), inputs=[vel, vc, self.fl, out], device=self.device)
