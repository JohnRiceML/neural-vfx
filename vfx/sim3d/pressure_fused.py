"""Opt-in fused pressure solve (`SolverConfig3D.pressure_mode = "fused"`).

Same system and the same multigrid as pressure.py: K phi = -d(u*) with K = d d^T (the 2dx "wide"
Laplacian), so `divergence()` measures exactly what this solve reduces. Only the iteration and
the kernel boundaries differ:

  * no CG: a FIXED number of multigrid cycles used as a stationary iteration on phi, warm-started
    from the previous step's phi. No dot products, no reductions, no graph conditional, so the cost
    is constant and there are no host or device-side stopping tests.
  * between coarse corrections, `smooth` sweeps of Jacobi on the exact K (Chebyshev weights by
    default, which damp a wider band than a fixed omega). The plume-past-sphere scene needs >= 3
    sweeps: the Hadamard-basis MG drops the component coupling at obstacle faces, and only the
    exact-K sweeps remove that error (README: smoother sweep).
  * the Hadamard-basis V(1,1) red-black GS is the same arithmetic as pressure.py, but fused:
      residual + Hadamard transform + level-1 red sweep         -> 1 launch   (was 3)
      black sweep + residual + restriction + coarse red sweep   -> 1 launch per level (was 2)
      prolongation + black sweep + red sweep                    -> 1 launch per level (was 2)
      level-1 up-sweep + inverse transform + phi update         -> 1 launch   (was 3)
    by recomputing the neighbours' black values instead of reading them back (2-cell halo; the
    small levels are latency-bound, so the extra arithmetic is free). Levels with m <= 2 run in the
    single-block kernel of pressure.py.
  Launches per cycle at 64^3: 10 MG + `smooth` sweeps (pressure.py: about 25 per PCG iteration).

Masks, MG operators, phi (the warm start) and the solid-change test are shared with the default
PressureSolver3D instance; nothing in pressure.py is modified.
"""
from __future__ import annotations

import math

import warp as wp

from . import pressure as PM
from .kernels import P_MAXIT, P_TOL, P_WARM
from .pressure import (CB, PressureSolver3D, div2, g_ax, hsign, k_coarse_block, kdiag, kmul, red_init)

wp.set_module_options({"enable_backward": False})   # forward only: the halo-recompute kernels are large


# ---------------------------------------------------------------- outer sweeps on the exact K
@wp.kernel
def k_sweep(src: wp.array3d(dtype=float), dst: wp.array3d(dtype=float), v: wp.array4d(dtype=float), vc: int,
            fm: wp.array3d(dtype=int), w: float, prm: wp.array(dtype=float), first: int):
    """dst = src + w D^-1 (b - K src), b = -d(u*). On the first sweep of a step without warm start,
    src is taken as 0."""
    z, y, x = wp.tid()
    m = fm[z, y, x]
    if m == 0:
        dst[z, y, x] = 0.0
        return
    b = -div2(v, vc, fm, z, y, x)
    if first == 1 and prm[P_WARM] <= 0.5:
        dst[z, y, x] = w * b / kdiag(m)
        return
    dst[z, y, x] = src[z, y, x] + w * (b - kmul(src, fm, z, y, x)) / kdiag(m)


@wp.kernel
def k_apply_copy(v: wp.array4d(dtype=float), vc: int, fm: wp.array3d(dtype=int), phi: wp.array3d(dtype=float),
                 home: wp.array3d(dtype=float), copy: int):
    """u -= g(phi) (0 in solids); optionally home <- phi (keeps the warm start in one buffer)."""
    z, y, x = wp.tid()
    if copy == 1:
        home[z, y, x] = phi[z, y, x]
    if fm[z, y, x] == 0:
        v[vc, z, y, x] = 0.0
        v[vc + 1, z, y, x] = 0.0
        v[vc + 2, z, y, x] = 0.0
        return
    for ax in range(3):
        v[vc + ax, z, y, x] = v[vc + ax, z, y, x] - g_ax(phi, fm, z, y, x, ax)


# ---------------------------------------------------------------- fused V-cycle pieces
@wp.kernel
def k_resid_phi(phi: wp.array3d(dtype=float), v: wp.array4d(dtype=float), vc: int, fm: wp.array3d(dtype=int),
                rr: wp.array3d(dtype=float)):
    """rr = -d(u*) - K phi (0 in solids)."""
    z, y, x = wp.tid()
    val = float(0.0)
    if fm[z, y, x] != 0:
        val = -div2(v, vc, fm, z, y, x) - kmul(phi, fm, z, y, x)
    rr[z, y, x] = val


@wp.kernel
def k_resid_fwd_red(phi: wp.array3d(dtype=float), v: wp.array4d(dtype=float), vc: int, fm: wp.array3d(dtype=int),
                    r1: wp.array4d(dtype=float), x1: wp.array4d(dtype=float), diag: wp.array4d(dtype=float),
                    dmin: float):
    """rr = -d(u*) - K phi on the 8 children, Hadamard transform to level 1, zero-guess red sweep."""
    K, J, I = wp.tid()
    rv = wp.vector(dtype=float, length=8)
    for k in range(8):
        z = 2 * K + (k >> 2)
        y = 2 * J + ((k >> 1) & 1)
        x = 2 * I + (k & 1)
        val = float(0.0)
        if fm[z, y, x] != 0:
            val = -div2(v, vc, fm, z, y, x) - kmul(phi, fm, z, y, x)
        rv[k] = val
    red = ((I + J + K) & 1) == 0
    for c in range(8):
        acc = float(0.0)
        for k in range(8):
            acc += hsign(c, k >> 2, (k >> 1) & 1, k & 1) * rv[k]
        r1[c, K, J, I] = acc
        if red:
            x1[c, K, J, I] = red_init(acc, diag[c, K, J, I], dmin)


@wp.func
def xblack(xv: wp.array4d(dtype=float), r: wp.array4d(dtype=float), diag: wp.array4d(dtype=float),
           off: wp.array4d(dtype=float), dmin: float, c: int, K: int, J: int, I: int) -> float:
    """Black GS value at a black cell from the (red) neighbours in xv (same order as offsum)."""
    d = diag[c, K, J, I]
    if d <= dmin:
        return 0.0
    m = xv.shape[1]
    a = r[c, K, J, I]
    if I + 1 < m:
        a -= off[c, K, J, I] * xv[c, K, J, I + 1]
    if I > 0:
        a -= off[c, K, J, I - 1] * xv[c, K, J, I - 1]
    if J + 1 < m:
        a -= off[8 + c, K, J, I] * xv[c, K, J + 1, I]
    if J > 0:
        a -= off[8 + c, K, J - 1, I] * xv[c, K, J - 1, I]
    if K + 1 < m:
        a -= off[16 + c, K, J, I] * xv[c, K + 1, J, I]
    if K > 0:
        a -= off[16 + c, K - 1, J, I] * xv[c, K - 1, J, I]
    return a / d


@wp.func
def xval(xv: wp.array4d(dtype=float), r: wp.array4d(dtype=float), diag: wp.array4d(dtype=float),
         off: wp.array4d(dtype=float), dmin: float, c: int, K: int, J: int, I: int) -> float:
    """x after the down-sweep (red: stored zero-guess value, black: recomputed GS value)."""
    if ((I + J + K) & 1) == 0:
        return xv[c, K, J, I]
    return xblack(xv, r, diag, off, dmin, c, K, J, I)


@wp.func
def resid_at(xv: wp.array4d(dtype=float), r: wp.array4d(dtype=float), diag: wp.array4d(dtype=float),
             off: wp.array4d(dtype=float), dmin: float, c: int, K: int, J: int, I: int) -> float:
    m = xv.shape[1]
    a = r[c, K, J, I] - diag[c, K, J, I] * xval(xv, r, diag, off, dmin, c, K, J, I)
    if I + 1 < m:
        a -= off[c, K, J, I] * xval(xv, r, diag, off, dmin, c, K, J, I + 1)
    if I > 0:
        a -= off[c, K, J, I - 1] * xval(xv, r, diag, off, dmin, c, K, J, I - 1)
    if J + 1 < m:
        a -= off[8 + c, K, J, I] * xval(xv, r, diag, off, dmin, c, K, J + 1, I)
    if J > 0:
        a -= off[8 + c, K, J - 1, I] * xval(xv, r, diag, off, dmin, c, K, J - 1, I)
    if K + 1 < m:
        a -= off[16 + c, K, J, I] * xval(xv, r, diag, off, dmin, c, K + 1, J, I)
    if K > 0:
        a -= off[16 + c, K - 1, J, I] * xval(xv, r, diag, off, dmin, c, K - 1, J, I)
    return a


@wp.kernel
def k_down(xv: wp.array4d(dtype=float), r: wp.array4d(dtype=float), diag: wp.array4d(dtype=float),
           off: wp.array4d(dtype=float), dmin: float, rc: wp.array4d(dtype=float), xc: wp.array4d(dtype=float),
           dc: wp.array4d(dtype=float), dminc: float, do_red: int):
    """Black sweep (recomputed) + residual + restriction (mean of 8 children) + coarse red sweep."""
    c, K, J, I = wp.tid()
    acc = float(0.0)
    for k in range(8):
        acc += resid_at(xv, r, diag, off, dmin, c, 2 * K + (k >> 2), 2 * J + ((k >> 1) & 1), 2 * I + (k & 1))
    acc *= 0.125
    rc[c, K, J, I] = acc
    if do_red == 1 and ((I + J + K) & 1) == 0:
        xc[c, K, J, I] = red_init(acc, dc[c, K, J, I], dminc)


@wp.func
def xpr(xv: wp.array4d(dtype=float), xc: wp.array4d(dtype=float), c: int, K: int, J: int, I: int) -> float:
    """Red value after constant prolongation of the coarse correction."""
    return xv[c, K, J, I] + xc[c, K >> 1, J >> 1, I >> 1]


@wp.func
def xbnew(xv: wp.array4d(dtype=float), r: wp.array4d(dtype=float), diag: wp.array4d(dtype=float),
          off: wp.array4d(dtype=float), xc: wp.array4d(dtype=float), dmin: float,
          c: int, K: int, J: int, I: int) -> float:
    """Black post-sweep value (neighbours: prolongated reds), as k_prolong_black."""
    d = diag[c, K, J, I]
    if d <= dmin:
        return 0.0
    m = xv.shape[1]
    a = r[c, K, J, I]
    if I + 1 < m:
        a -= off[c, K, J, I] * xpr(xv, xc, c, K, J, I + 1)
    if I > 0:
        a -= off[c, K, J, I - 1] * xpr(xv, xc, c, K, J, I - 1)
    if J + 1 < m:
        a -= off[8 + c, K, J, I] * xpr(xv, xc, c, K, J + 1, I)
    if J > 0:
        a -= off[8 + c, K, J - 1, I] * xpr(xv, xc, c, K, J - 1, I)
    if K + 1 < m:
        a -= off[16 + c, K, J, I] * xpr(xv, xc, c, K + 1, J, I)
    if K > 0:
        a -= off[16 + c, K - 1, J, I] * xpr(xv, xc, c, K - 1, J, I)
    return a / d


@wp.func
def xup(xv: wp.array4d(dtype=float), r: wp.array4d(dtype=float), diag: wp.array4d(dtype=float),
        off: wp.array4d(dtype=float), xc: wp.array4d(dtype=float), dmin: float,
        c: int, K: int, J: int, I: int) -> float:
    """Final x after prolongation + black sweep + red sweep."""
    if ((I + J + K) & 1) == 1:
        return xbnew(xv, r, diag, off, xc, dmin, c, K, J, I)
    d = diag[c, K, J, I]
    if d <= dmin:
        return 0.0
    m = xv.shape[1]
    a = r[c, K, J, I]
    if I + 1 < m:
        a -= off[c, K, J, I] * xbnew(xv, r, diag, off, xc, dmin, c, K, J, I + 1)
    if I > 0:
        a -= off[c, K, J, I - 1] * xbnew(xv, r, diag, off, xc, dmin, c, K, J, I - 1)
    if J + 1 < m:
        a -= off[8 + c, K, J, I] * xbnew(xv, r, diag, off, xc, dmin, c, K, J + 1, I)
    if J > 0:
        a -= off[8 + c, K, J - 1, I] * xbnew(xv, r, diag, off, xc, dmin, c, K, J - 1, I)
    if K + 1 < m:
        a -= off[16 + c, K, J, I] * xbnew(xv, r, diag, off, xc, dmin, c, K + 1, J, I)
    if K > 0:
        a -= off[16 + c, K - 1, J, I] * xbnew(xv, r, diag, off, xc, dmin, c, K - 1, J, I)
    return a / d


@wp.kernel
def k_up(xv: wp.array4d(dtype=float), r: wp.array4d(dtype=float), diag: wp.array4d(dtype=float),
         off: wp.array4d(dtype=float), xc: wp.array4d(dtype=float), dmin: float, xo: wp.array4d(dtype=float)):
    c, K, J, I = wp.tid()
    xo[c, K, J, I] = xup(xv, r, diag, off, xc, dmin, c, K, J, I)


@wp.kernel
def k_up_inv(xv: wp.array4d(dtype=float), r: wp.array4d(dtype=float), diag: wp.array4d(dtype=float),
             off: wp.array4d(dtype=float), xc: wp.array4d(dtype=float), dmin: float,
             fm: wp.array3d(dtype=int), phi: wp.array3d(dtype=float)):
    """Level-1 up-sweep for all 8 components of one block + inverse Hadamard transform, added to phi
    in place (each thread owns its 2x2x2 block of phi)."""
    K, J, I = wp.tid()
    xs = wp.vector(dtype=float, length=8)
    for c in range(8):
        xs[c] = xup(xv, r, diag, off, xc, dmin, c, K, J, I)
    for k in range(8):
        dz = k >> 2
        dy = (k >> 1) & 1
        dx = k & 1
        z = 2 * K + dz
        y = 2 * J + dy
        x = 2 * I + dx
        if fm[z, y, x] == 0:
            phi[z, y, x] = 0.0
        else:
            acc = phi[z, y, x]
            for c in range(8):
                acc += hsign(c, dz, dy, dx) * xs[c]
            phi[z, y, x] = acc


@wp.kernel
def k_inv_add(x1: wp.array4d(dtype=float), fm: wp.array3d(dtype=int), phi: wp.array3d(dtype=float)):
    """phi += T^T x1 in place (0 in solids); local, so in place is safe."""
    z, y, x = wp.tid()
    if fm[z, y, x] == 0:
        phi[z, y, x] = 0.0
        return
    acc = phi[z, y, x]
    for c in range(8):
        acc += hsign(c, z & 1, y & 1, x & 1) * x1[c, z >> 1, y >> 1, x >> 1]
    phi[z, y, x] = acc


@wp.kernel
def k_copy3(src: wp.array3d(dtype=float), dst: wp.array3d(dtype=float)):
    z, y, x = wp.tid()
    dst[z, y, x] = src[z, y, x]


def _make_test(N: int, block: int):
    """Residual + GPU-side stopping test for the adaptive tail (block tile reductions, no atomics)."""
    total = N * N * N
    cpt = max(1, -(-total // (512 * block)))
    per_block = cpt * block
    NB = -(-total // per_block)

    @wp.kernel(module="unique")
    def k_maxu(v: wp.array4d(dtype=float), vc: int, fm: wp.array3d(dtype=int), part: wp.array(dtype=float)):
        b, l = wp.tid()
        mx = float(0.0)
        for k in range(cpt):
            i = b * per_block + k * block + l
            if i < total:
                z = i // (N * N)
                y = (i // N) % N
                x = i % N
                if fm[z, y, x] != 0:
                    mx = wp.max(mx, wp.length(wp.vec3(v[vc, z, y, x], v[vc + 1, z, y, x], v[vc + 2, z, y, x])))
        t = wp.tile_max(wp.tile(mx))
        if l == 0:
            part[b] = t[0]

    @wp.kernel(module="unique")
    def k_resid_max(phi: wp.array3d(dtype=float), v: wp.array4d(dtype=float), vc: int, fm: wp.array3d(dtype=int),
                    rr: wp.array3d(dtype=float), part: wp.array(dtype=float)):
        """rr = -d(u*) - K phi; partial max |rr|."""
        b, l = wp.tid()
        mx = float(0.0)
        for k in range(cpt):
            i = b * per_block + k * block + l
            if i < total:
                z = i // (N * N)
                y = (i // N) % N
                x = i % N
                val = float(0.0)
                if fm[z, y, x] != 0:
                    val = -div2(v, vc, fm, z, y, x) - kmul(phi, fm, z, y, x)
                rr[z, y, x] = val
                mx = wp.max(mx, wp.abs(val))
        t = wp.tile_max(wp.tile(mx))
        if l == 0:
            part[NB + b] = t[0]

    @wp.kernel(module="unique")
    def k_test(part: wp.array(dtype=float), prm: wp.array(dtype=float), n_fixed: int, it: wp.array(dtype=int),
               cond: wp.array(dtype=int), init: int):
        """Continue while 0.5 max|rr| >= tol * max|u*| (the normalized divergence after apply, up to
        the change of max|u|) and fewer than pressure_max_iters cycles have run."""
        umax = wp.tile_max(wp.tile_load(part, shape=NB, offset=0))[0]
        rmax = wp.tile_max(wp.tile_load(part, shape=NB, offset=NB))[0]
        n = n_fixed
        if init == 0:
            n = it[0] + 1
        it[0] = n
        go = int(0)
        if n < int(prm[P_MAXIT]) and 0.5 * rmax >= prm[P_TOL] * umax and umax > 0.0:
            go = 1
        cond[0] = go

    return dict(NB=NB, maxu=k_maxu, resid_max=k_resid_max, test=k_test)


# ---------------------------------------------------------------- driver
def chebyshev_weights(n: int, lam_max: float = 2.0, ratio: float = 4.0) -> list[float]:
    """Jacobi weights whose product polynomial is the degree-n Chebyshev minimax on
    [lam_max / ratio, lam_max] (lam_max(D^-1 K) measured 1.976 at 64^3, bounded by 2)."""
    lo, hi = lam_max / ratio, lam_max
    return [1.0 / ((hi + lo) / 2 + (hi - lo) / 2 * math.cos(math.pi * (2 * i + 1) / (2 * n))) for i in range(n)]


class FusedPressure3D:
    """Fixed-cost fused multigrid projection. Shares fm / MG operators / phi with `base`."""

    def __init__(self, base: PressureSolver3D):
        self.base = base
        N, dev = base.N, base.device
        self.N, self.device = N, dev
        L = base.lv
        # first level small enough for the single-block kernel (else the coarsest level)
        S = next((i for i, a in enumerate(L) if a["m"] <= base.BLOCK_M), len(L) - 1)
        if S == 0 or len(L) - S > 4:
            raise ValueError(f"fused pressure: unsupported grid size {N}")
        self.S = S
        self.phi_alt = wp.zeros((N, N, N), dtype=float, device=dev)
        # up-sweep outputs for levels 1..S-1 (level 0 goes straight into phi)
        self.xo = [None] + [wp.zeros((8, a["m"], a["m"], a["m"]), dtype=float, device=dev) for a in L[1:S]]
        self.tk = _make_test(N, self.BLOCK)
        self.part = wp.zeros(2 * self.tk["NB"], dtype=float, device=dev)
        self.it = wp.zeros(1, dtype=int, device=dev)       # V-cycles run in the last projection
        self.cond = wp.zeros(1, dtype=int, device=dev)
        self.n_launches = 0

    def weights(self, smooth: int, smoother: str) -> list[float]:
        smooth = max(1, int(smooth))
        if smoother == "jacobi":
            return [0.8] * smooth
        if smoother == "chebyshev":
            return chebyshev_weights(smooth)
        raise ValueError(f"pressure_smoother must be 'chebyshev' or 'jacobi', got {smoother!r}")

    def record(self, vel, vc: int, solid, prm, cycles: int, smooth: int, smoother: str, cached: bool = True,
               adaptive: bool = False, max_cycles: int = 0):
        """Project vel[vc:vc+3] in place: `smooth` sweeps, then `cycles` x (V-cycle + `smooth` sweeps);
        with `adaptive`, further cycles while the normalized residual >= prm[P_TOL], up to
        prm[P_MAXIT] cycles in total (`max_cycles` is the host-side copy, only to skip the tail)."""
        b, d, L, S, N = self.base, self.device, self.base.lv, self.S, self.N
        b.record_setup(solid, cached)
        ws = self.weights(smooth, smoother)
        home, alt = b.phi, self.phi_alt
        other = lambda p: alt if p is home else home
        cur = home
        first = 1
        self.n_launches = 0

        def L_(kernel, dim, inputs, tiled=False):
            self.n_launches += 1
            if tiled:
                wp.launch_tiled(kernel, dim=dim, inputs=inputs, block_dim=CB, device=d)
            else:
                wp.launch(kernel, dim=dim, inputs=inputs, device=d)

        def sweeps(cur, first):
            for w in ws:
                dst = other(cur)
                L_(k_sweep, (N, N, N), [cur, dst, vel, vc, b.fm, w, prm, first])
                cur, first = dst, 0
            return cur, first

        m0 = L[0]["m"]

        def vcycle(cur, rr_ready=False, inplace=False):
            """phi += T^T MG(b - K phi). rr_ready: b.rr already holds the residual of cur (adaptive
            tail). inplace: add the correction in place (keeps the phi buffer fixed inside a loop)."""
            # ---- down: residual of phi, transform, red sweep; then black + restrict per level
            if rr_ready:
                L_(PM.k_fwd_red, (m0, m0, m0), [b.rr, L[0]["r"], L[0]["x"], L[0]["diag"], L[0]["dmin"]])
            elif self.FUSE_TOP:
                L_(k_resid_fwd_red, (m0, m0, m0), [cur, vel, vc, b.fm, L[0]["r"], L[0]["x"], L[0]["diag"], L[0]["dmin"]])
            else:
                L_(k_resid_phi, (N, N, N), [cur, vel, vc, b.fm, b.rr])
                L_(PM.k_fwd_red, (m0, m0, m0), [b.rr, L[0]["r"], L[0]["x"], L[0]["diag"], L[0]["dmin"]])
            fused = [l < S and self._fuse_level(l) for l in range(S)]
            for l in range(S):
                a, c = L[l], L[l + 1]
                red = 0 if l + 1 == S else 1
                if fused[l]:
                    L_(k_down, (8, c["m"], c["m"], c["m"]),
                       [a["x"], a["r"], a["diag"], a["off"], a["dmin"], c["r"], c["x"], c["diag"], c["dmin"], red])
                else:
                    m = a["m"]
                    L_(PM.k_smooth_h, (8, m, m, m // 2), [a["x"], a["r"], a["diag"], a["off"], 1, a["dmin"]])
                    L_(PM.k_restrict_red, (8, c["m"], c["m"], c["m"]),
                       [a["x"], a["r"], a["diag"], a["off"], c["r"], c["x"], c["diag"], c["dmin"], red])
            # ---- bottom: single block
            lv = L[S:]
            args = []
            for k in range(4):
                c = lv[min(k, len(lv) - 1)]
                args += [c["x"], c["r"], c["diag"], c["off"], c["dmin"]]
            L_(k_coarse_block, [1], args + [len(lv)], tiled=True)
            # ---- up: prolongation + black + red per level (fused: out of place into xo)
            xc = L[S]["x"]
            for l in range(S - 1, -1, -1):
                a = L[l]
                m = a["m"]
                if l == 0 and self.FUSE_TOP and fused[0]:
                    L_(k_up_inv, (m0, m0, m0), [a["x"], a["r"], a["diag"], a["off"], xc, a["dmin"], b.fm, cur])
                    break
                if fused[l] and l > 0:
                    L_(k_up, (8, m, m, m), [a["x"], a["r"], a["diag"], a["off"], xc, a["dmin"], self.xo[l]])
                    xc = self.xo[l]
                else:
                    L_(PM.k_prolong_black, (8, m, m, m // 2), [a["x"], a["r"], a["diag"], a["off"], xc, a["dmin"]])
                    L_(PM.k_smooth_h, (8, m, m, m // 2), [a["x"], a["r"], a["diag"], a["off"], 0, a["dmin"]])
                    xc = a["x"]
                if l == 0:
                    if inplace:
                        L_(k_inv_add, (N, N, N), [xc, b.fm, cur])
                    else:
                        dst = other(cur)
                        L_(PM.k_inv_transform, (N, N, N), [xc, b.fm, cur, dst])
                        cur = dst
            return cur

        cur, first = sweeps(cur, first)
        for _ in range(max(0, int(cycles))):
            cur = vcycle(cur)
            cur, first = sweeps(cur, first)
        self.it.fill_(max(0, int(cycles)))
        if adaptive and max_cycles > cycles:
            # adaptive tail: more cycles while the residual is above pressure_tol (GPU-side test,
            # graph conditional). The loop body keeps phi in the same buffer.
            t, NB = self.tk, self.tk["NB"]
            wp.launch_tiled(t["maxu"], dim=[NB], inputs=[vel, vc, b.fm, self.part], block_dim=self.BLOCK, device=d)
            wp.launch_tiled(t["resid_max"], dim=[NB], inputs=[cur, vel, vc, b.fm, b.rr, self.part],
                            block_dim=self.BLOCK, device=d)
            wp.launch_tiled(t["test"], dim=[1], inputs=[self.part, prm, int(cycles), self.it, self.cond, 1],
                            block_dim=self.BLOCK, device=d)
            loop_buf = cur

            def body():
                c = vcycle(loop_buf, rr_ready=True, inplace=True)
                c, _ = sweeps(c, 0)
                if c is not loop_buf:
                    wp.launch(k_copy3, dim=(N, N, N), inputs=[c, loop_buf], device=d)
                wp.launch_tiled(t["resid_max"], dim=[NB], inputs=[loop_buf, vel, vc, b.fm, b.rr, self.part],
                                block_dim=self.BLOCK, device=d)
                wp.launch_tiled(t["test"], dim=[1], inputs=[self.part, prm, int(cycles), self.it, self.cond, 0],
                                block_dim=self.BLOCK, device=d)
            wp.capture_while(self.cond, body)
        L_(k_apply_copy, (N, N, N), [vel, vc, b.fm, cur, home, 1 if cur is not home else 0])

    BLOCK = 256
    # Halo-recompute fusion switches. Measured SLOWER than the plain kernels (README: what fusion
    # bought), so both are off by default; kept for comparison.
    FUSE_TOP = False       # residual+transform+red and level-1 up+inverse transform as single kernels
    FUSE_MIN_M = 1 << 30   # fuse black+restrict / prolong+black+red at levels with m >= FUSE_MIN_M

    def _fuse_level(self, l: int) -> bool:
        return self.base.lv[l]["m"] >= self.FUSE_MIN_M
