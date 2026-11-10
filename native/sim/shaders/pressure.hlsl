// Pressure projection: MG-preconditioned CG in the 2x2x2 Hadamard basis (pressure.py), same
// operations in the same order. Flat kernels: 256 threads, 2 cells per thread (pressure.py
// _make_flat: i = b*512 + k*256 + l), one partial per group, fixed-order tree reductions.
#include "common.hlsli"

#define FB 256
#define CPT 2

groupshared float g_s[FB];
groupshared float g_m[FB];

// ---------------------------------------------------------------- helpers
int stride_of(int ax, int n) { return ax == 0 ? 1 : (ax == 1 ? n : n * n); }

float g_ax(RWStructuredBuffer<float> p, RWStructuredBuffer<int> fm, int i, int ax, int n) {
    int m = fm[i];
    int st = stride_of(ax, n);
    float g = 0.0f;
    float c = p[i];
    if (((m >> (2 * ax)) & 1) == 1) g += p[i + st] - c;
    if (((m >> (2 * ax + 1)) & 1) == 1) g += c - p[i - st];
    return g;
}

// (K p)_i = -d(g(p))_i at a fluid cell. pressure.py kmul(), same operation order; all 19 loads
// (fm and p on the 13-point stencil) are issued up front with clamped indices, so the kernel waits for
// one round of memory latency instead of a chain of three (mask -> neighbour mask -> value).
float kmul(RWStructuredBuffer<float> p, RWStructuredBuffer<int> fm, int i, int n) {
    int last = n * n * n - 1;
    int m = fm[i];
    float c = p[i];
    int mp[3], mm[3];
    float p1[3], m1[3], p2[3], m2[3];
    [unroll] for (int a = 0; a < 3; ++a) {
        int st = stride_of(a, n);
        int ip1 = min(i + st, last), im1 = max(i - st, 0), ip2 = min(i + 2 * st, last), im2 = max(i - 2 * st, 0);
        mp[a] = fm[ip1]; mm[a] = fm[im1];
        p1[a] = p[ip1]; m1[a] = p[im1]; p2[a] = p[ip2]; m2[a] = p[im2];
    }
    float d = 0.0f;
    [unroll] for (int ax = 0; ax < 3; ++ax) {
        bool op = ((m >> (2 * ax)) & 1) == 1;
        bool om = ((m >> (2 * ax + 1)) & 1) == 1;
        float gi = 0.0f;                                   // g_ax(p, i)
        if (op) gi += p1[ax] - c;
        if (om) gi += c - m1[ax];
        if (op) {                                          // g_ax(p, i + st)
            float gn = 0.0f;
            if (((mp[ax] >> (2 * ax)) & 1) == 1) gn += p2[ax] - p1[ax];
            if (((mp[ax] >> (2 * ax + 1)) & 1) == 1) gn += p1[ax] - c;
            d += gi + gn;
        }
        if (om) {                                          // g_ax(p, i - st)
            float gn = 0.0f;
            if (((mm[ax] >> (2 * ax)) & 1) == 1) gn += c - m1[ax];
            if (((mm[ax] >> (2 * ax + 1)) & 1) == 1) gn += m1[ax] - m2[ax];
            d -= gi + gn;
        }
    }
    return -d;
}

float kdiag(int m) {
    float s = 0.0f;
    [unroll] for (int ax = 0; ax < 3; ++ax) {
        float oR = float((m >> (2 * ax)) & 1);
        float oL = float((m >> (2 * ax + 1)) & 1);
        s += (oL - oR) * (oL - oR) + oL + oR;
    }
    return s;
}

// d(u)_i (pressure.py div2), same operation order; the 6 neighbour loads are issued unconditionally
// (clamped coordinates) so they do not wait for the mask.
float div2(RWTexture3D<float4> v, int m, int3 c) {
    int n = int(RC.n);
    float3 u = v[c].xyz;
    float xp = v[int3(min(c.x + 1, n - 1), c.y, c.z)].x, xm = v[int3(max(c.x - 1, 0), c.y, c.z)].x;
    float yp = v[int3(c.x, min(c.y + 1, n - 1), c.z)].y, ym = v[int3(c.x, max(c.y - 1, 0), c.z)].y;
    float zp = v[int3(c.x, c.y, min(c.z + 1, n - 1))].z, zm = v[int3(c.x, c.y, max(c.z - 1, 0))].z;
    float d = 0.0f;
    if ((m & 1) != 0) d += u.x + xp;
    if ((m & 2) != 0) d -= u.x + xm;
    if ((m & 4) != 0) d += u.y + yp;
    if ((m & 8) != 0) d -= u.y + ym;
    if ((m & 16) != 0) d += u.z + zp;
    if ((m & 32) != 0) d -= u.z + zm;
    return d;
}

int3 unflat(uint i, uint n) { return int3(i % n, (i / n) % n, i / (n * n)); }   // (x, y, z)

// group-wide sum / max; every thread gets the result
float group_sum(float v, uint l) {
    GroupMemoryBarrierWithGroupSync();
    g_s[l] = v;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint st = FB / 2; st > 0; st >>= 1) {
        if (l < st) g_s[l] += g_s[l + st];
        GroupMemoryBarrierWithGroupSync();
    }
    return g_s[0];
}
float group_max(float v, uint l) {
    GroupMemoryBarrierWithGroupSync();
    g_m[l] = v;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint st = FB / 2; st > 0; st >>= 1) {
        if (l < st) g_m[l] = max(g_m[l], g_m[l + st]);
        GroupMemoryBarrierWithGroupSync();
    }
    return g_m[0];
}
// sum / max of nb partials (same code everywhere => identical value in every group)
float parts_sum(RWStructuredBuffer<float> b, uint off, uint nb, uint l) {
    float a = 0.0f;
    for (uint i = l; i < nb; i += FB) a += b[off + i];
    return group_sum(a, l);
}
float parts_max(RWStructuredBuffer<float> b, uint off, uint nb, uint l) {
    float a = 0.0f;
    for (uint i = l; i < nb; i += FB) a = max(a, b[off + i]);
    return group_max(a, l);
}

float sload(RWByteAddressBuffer s, uint slot) { return asfloat(s.Load(4 * slot)); }
void sstore(RWByteAddressBuffer s, uint slot, float v) { s.Store(4 * slot, asuint(v)); }

uint nb_flat(uint n) { return (n * n * n + FB * CPT - 1) / (FB * CPT); }

// ---------------------------------------------------------------- flat CG kernels
// r = -d(u*) - K phi0 (phi0 = phi if warm start, else phi := 0); partial max |u*| over fluid.
// k0 = VD UAV slot.
[numthreads(FB, 1, 1)]
void p_rhs(uint3 g : SV_GroupID, uint l : SV_GroupIndex) {
    uint n = RC.n, total = n * n * n;
    RWTexture3D<float4> v = ResourceDescriptorHeap[RC.k0];
    RWStructuredBuffer<int> fm = ResourceDescriptorHeap[SL_FM];
    RWStructuredBuffer<float> r = ResourceDescriptorHeap[SL_PR];
    RWStructuredBuffer<float> phi = ResourceDescriptorHeap[SL_PHI];
    RWStructuredBuffer<float> part = ResourceDescriptorHeap[SL_PART];
    bool warm = P.warm > 0.5f;
    float mx = 0.0f;
    [unroll] for (uint k = 0; k < CPT; ++k) {
        uint i = g.x * (FB * CPT) + k * FB + l;
        if (i < total) {
            int3 c = unflat(i, n);
            if (!warm) phi[i] = 0.0f;
            int m = fm[i];
            if (m == 0) {
                r[i] = 0.0f;
            } else {
                float rv = -div2(v, m, c);
                if (warm) rv -= kmul(phi, fm, int(i), int(n));
                r[i] = rv;
                float3 u = v[c].xyz;
                mx = max(mx, sqrt(u.x * u.x + u.y * u.y + u.z * u.z));
            }
        }
    }
    float t = group_max(mx, l);
    if (l == 0) part[g.x] = t;
}

// scal[MAXU], scal[THR]; zero the predicate buffers 1..k0 (k0 = number of recorded iterations)
[numthreads(FB, 1, 1)]
void p_fin_maxu(uint l : SV_GroupIndex) {
    uint nb = nb_flat(RC.n);
    RWStructuredBuffer<float> part = ResourceDescriptorHeap[SL_PART];
    RWByteAddressBuffer scal = ResourceDescriptorHeap[SL_SCAL];
    float m = parts_max(part, 0, nb, l);
    if (l == 0) {
        sstore(scal, S_MAXU, m);
        sstore(scal, S_THR, 2.0f * max(P.tol, 1.0e-6f) * m);
    }
    for (uint k = l + 1; k <= RC.k0; k += FB) {
        RWByteAddressBuffer pr = ResourceDescriptorHeap[NonUniformResourceIndex(SL_PRED0 + k)];
        pr.Store2(0, uint2(0, 0));
    }
}

// q = K p, partial p.q
[numthreads(FB, 1, 1)]
void p_matvec_dot(uint3 g : SV_GroupID, uint l : SV_GroupIndex) {
    uint n = RC.n, total = n * n * n;
    RWStructuredBuffer<int> fm = ResourceDescriptorHeap[SL_FM];
    RWStructuredBuffer<float> p = ResourceDescriptorHeap[SL_PP];
    RWStructuredBuffer<float> q = ResourceDescriptorHeap[SL_PQ];
    RWStructuredBuffer<float> part_pq = ResourceDescriptorHeap[SL_PART_PQ];
    float s = 0.0f;
    float o[CPT];
    [unroll] for (uint k = 0; k < CPT; ++k) {      // all loads of both cells before any store
        uint i = g.x * (FB * CPT) + k * FB + l;
        int ii = i < total ? int(i) : 0;
        float v = kmul(p, fm, ii, int(n));
        float val = (fm[ii] != 0) ? v : 0.0f;
        o[k] = val;
        if (i < total) s += val * p[ii];
    }
    [unroll] for (uint k2 = 0; k2 < CPT; ++k2) {
        uint i = g.x * (FB * CPT) + k2 * FB + l;
        if (i < total) q[i] = o[k2];
    }
    float t = group_sum(s, l);
    if (l == 0) part_pq[g.x] = t;
}

// r -= alpha q (skipped at init; alpha = rz / sum(part_pq) recomputed per group); z0 = omega D^-1 r.
// k0 = init, k1 = bank
[numthreads(FB, 1, 1)]
void p_pre(uint3 g : SV_GroupID, uint l : SV_GroupIndex) {
    uint n = RC.n, total = n * n * n;
    bool init = RC.k0 != 0;
    RWStructuredBuffer<int> fm = ResourceDescriptorHeap[SL_FM];
    RWStructuredBuffer<float> r = ResourceDescriptorHeap[SL_PR];
    RWStructuredBuffer<float> q = ResourceDescriptorHeap[SL_PQ];
    RWStructuredBuffer<float> z0 = ResourceDescriptorHeap[SL_Z0];
    RWStructuredBuffer<float> part_pq = ResourceDescriptorHeap[SL_PART_PQ];
    RWByteAddressBuffer scal = ResourceDescriptorHeap[SL_SCAL];
    float a = 0.0f;
    if (!init) {   // k1 = scalar bank of the fused path (0xFFFFFFFF = reference layout)
        bool ref = RC.k1 == 0xFFFFFFFFu;
        float pq = parts_sum(part_pq, ref ? 0 : RC.k1 * nb_flat(n), nb_flat(n), l);
        if (pq > 0.0f) a = sload(scal, ref ? S_RZ : S_BANK(RC.k1, S_RZ)) / pq;
    }
    [unroll] for (uint k = 0; k < CPT; ++k) {
        uint i = g.x * (FB * CPT) + k * FB + l;
        if (i < total) {
            int m = fm[i];
            float rv = r[i];
            if (!init) {
                rv = rv - a * q[i];
                r[i] = rv;
            }
            float zz = 0.0f;
            if (m != 0) zz = NVFX_MG_OMEGA * rv / kdiag(m);
            z0[i] = zz;
        }
    }
}

// rr = r - K z0
[numthreads(FB, 1, 1)]
void p_resid(uint3 g : SV_GroupID, uint l : SV_GroupIndex) {
    uint n = RC.n, total = n * n * n;
    RWStructuredBuffer<int> fm = ResourceDescriptorHeap[SL_FM];
    RWStructuredBuffer<float> r = ResourceDescriptorHeap[SL_PR];
    RWStructuredBuffer<float> z0 = ResourceDescriptorHeap[SL_Z0];
    RWStructuredBuffer<float> rr = ResourceDescriptorHeap[SL_RR];
    float o[CPT];
    [unroll] for (uint k = 0; k < CPT; ++k) {
        uint i = g.x * (FB * CPT) + k * FB + l;
        int ii = i < total ? int(i) : 0;
        float v = r[ii] - kmul(z0, fm, ii, int(n));
        o[k] = (fm[ii] != 0) ? v : 0.0f;
    }
    [unroll] for (uint k2 = 0; k2 < CPT; ++k2) {
        uint i = g.x * (FB * CPT) + k2 * FB + l;
        if (i < total) rr[i] = o[k2];
    }
}

// z = zt + omega D^-1 (r - K zt); partial r.z and max |r|
[numthreads(FB, 1, 1)]
void p_post_rz(uint3 g : SV_GroupID, uint l : SV_GroupIndex) {
    uint n = RC.n, total = n * n * n;
    RWStructuredBuffer<int> fm = ResourceDescriptorHeap[SL_FM];
    RWStructuredBuffer<float> r = ResourceDescriptorHeap[SL_PR];
    RWStructuredBuffer<float> zt = ResourceDescriptorHeap[SL_ZT];
    RWStructuredBuffer<float> z = ResourceDescriptorHeap[SL_PZ];
    RWStructuredBuffer<float> part = ResourceDescriptorHeap[SL_PART];
    float s = 0.0f, mx = 0.0f;
    float o[CPT];
    [unroll] for (uint k = 0; k < CPT; ++k) {
        uint i = g.x * (FB * CPT) + k * FB + l;
        int ii = i < total ? int(i) : 0;
        int m = fm[ii];
        float rv = r[ii];
        float zf = zt[ii] + NVFX_MG_OMEGA * (rv - kmul(zt, fm, ii, int(n))) / kdiag(m);
        float zv = (m != 0) ? zf : 0.0f;
        o[k] = zv;
        if (i < total) {
            s += rv * zv;
            mx = max(mx, abs(rv));
        }
    }
    [unroll] for (uint k2 = 0; k2 < CPT; ++k2) {
        uint i = g.x * (FB * CPT) + k2 * FB + l;
        if (i < total) z[i] = o[k2];
    }
    float ts = group_sum(s, l);
    float tm = group_max(mx, l);
    if (l == 0) {
        part[g.x] = ts;
        part[nb_flat(n) + g.x] = tm;
    }
}

// CG scalars + the GPU-side stopping test. k0 = init, k1 = predicate slot to write (0 = none).
[numthreads(FB, 1, 1)]
void p_fin_rz(uint l : SV_GroupIndex) {
    uint nb = nb_flat(RC.n);
    bool init = RC.k0 != 0;
    RWStructuredBuffer<float> part = ResourceDescriptorHeap[SL_PART];
    RWStructuredBuffer<float> part_pq = ResourceDescriptorHeap[SL_PART_PQ];
    RWByteAddressBuffer scal = ResourceDescriptorHeap[SL_SCAL];
    float rz_new = parts_sum(part, 0, nb, l);
    float mr = parts_max(part, nb, nb, l);
    float pq = 0.0f;
    if (!init) pq = parts_sum(part_pq, 0, nb, l);
    if (l != 0) return;
    int it = 0;
    if (init) {
        sstore(scal, S_BETA, 0.0f);
    } else {
        float rz = sload(scal, S_RZ);
        float a = 0.0f;
        if (pq > 0.0f) a = rz / pq;
        sstore(scal, S_ALPHA, a);
        float bb = 0.0f;
        if (rz > 0.0f) bb = rz_new / rz;
        sstore(scal, S_BETA, bb);
        it = int(scal.Load(4 * S_IT)) + 1;
    }
    sstore(scal, S_RZ, rz_new);
    sstore(scal, S_MAXR, mr);
    scal.Store(4 * S_IT, uint(it));
    float best = mr;
    if (!init) best = min(sload(scal, S_BEST), mr);
    sstore(scal, S_BEST, best);
    uint go = 0;
    if (it < int(P.maxit) && rz_new > 0.0f && mr > sload(scal, S_THR) && mr <= 4.0f * best) go = 1;
    if (RC.k1 != 0) {
        RWByteAddressBuffer pr = ResourceDescriptorHeap[RC.k1];
        pr.Store2(0, uint2(go, 0));
    }
}

// phi += alpha p_old (skipped at init); p = z + beta p_old. k0 = init
[numthreads(FB, 1, 1)]
void p_upd_p(uint3 g : SV_GroupID, uint l : SV_GroupIndex) {
    uint n = RC.n, total = n * n * n;
    bool init = RC.k0 != 0;
    RWStructuredBuffer<float> phi = ResourceDescriptorHeap[SL_PHI];
    RWStructuredBuffer<float> p = ResourceDescriptorHeap[SL_PP];
    RWStructuredBuffer<float> z = ResourceDescriptorHeap[SL_PZ];
    RWByteAddressBuffer scal = ResourceDescriptorHeap[SL_SCAL];
    float a = sload(scal, S_ALPHA);
    float be = sload(scal, S_BETA);
    [unroll] for (uint k = 0; k < CPT; ++k) {
        uint i = g.x * (FB * CPT) + k * FB + l;
        if (i < total) {
            float po = p[i];
            if (!init) {
                phi[i] = phi[i] + a * po;
                p[i] = z[i] + be * po;
            } else {
                p[i] = z[i];
            }
        }
    }
}

// u -= g(phi) in fluid cells, u = 0 in solids (density untouched). k0 = VD UAV slot.
[numthreads(FB, 1, 1)]
void p_apply(uint3 id : SV_DispatchThreadID) {
    uint n = RC.n;
    uint i = id.x;
    if (i >= n * n * n) return;
    RWTexture3D<float4> v = ResourceDescriptorHeap[RC.k0];
    RWStructuredBuffer<int> fm = ResourceDescriptorHeap[SL_FM];
    RWStructuredBuffer<float> phi = ResourceDescriptorHeap[SL_PHI];
    int3 c = unflat(i, n);
    float4 u = v[c];
    if (fm[i] == 0) {
        v[c] = float4(0.0f, 0.0f, 0.0f, u.w);
        return;
    }
    u.x = u.x - g_ax(phi, fm, int(i), 0, int(n));
    u.y = u.y - g_ax(phi, fm, int(i), 1, int(n));
    u.z = u.z - g_ax(phi, fm, int(i), 2, int(n));
    v[c] = u;
}

// ---------------------------------------------------------------- multigrid (Hadamard basis)
float hsign(int c, int dz, int dy, int dx) {
    int e = (c & 1) * dx + ((c >> 1) & 1) * dy + ((c >> 2) & 1) * dz;
    return ((e & 1) == 1) ? -HS : HS;
}

uint lidx(Level L, int c, int K, int J, int I) { return uint(((c * int(L.m) + K) * int(L.m) + J) * int(L.m) + I); }
uint oidx(Level L, int ax, int c, int K, int J, int I) {
    return L.oo + uint((((ax * 8 + c) * int(L.m) + K) * int(L.m) + J) * int(L.m) + I);
}

float offsum(RWStructuredBuffer<float> X, RWStructuredBuffer<float> OF, Level L, int c, int K, int J, int I) {
    int m = int(L.m);
    float a = 0.0f;
    if (I + 1 < m) a += OF[oidx(L, 0, c, K, J, I)] * X[L.ox + lidx(L, c, K, J, I + 1)];
    if (I > 0) a += OF[oidx(L, 0, c, K, J, I - 1)] * X[L.ox + lidx(L, c, K, J, I - 1)];
    if (J + 1 < m) a += OF[oidx(L, 1, c, K, J, I)] * X[L.ox + lidx(L, c, K, J + 1, I)];
    if (J > 0) a += OF[oidx(L, 1, c, K, J - 1, I)] * X[L.ox + lidx(L, c, K, J - 1, I)];
    if (K + 1 < m) a += OF[oidx(L, 2, c, K, J, I)] * X[L.ox + lidx(L, c, K + 1, J, I)];
    if (K > 0) a += OF[oidx(L, 2, c, K - 1, J, I)] * X[L.ox + lidx(L, c, K - 1, J, I)];
    return a;
}

float red_init(float r, float d, float dmin) { return (d <= dmin) ? 0.0f : r / d; }

#define MG_RES                                                                           \
    StructuredBuffer<Level> LV = ResourceDescriptorHeap[SL_LEVELS];                      \
    RWStructuredBuffer<float> X = ResourceDescriptorHeap[SL_LX];                         \
    RWStructuredBuffer<float> R = ResourceDescriptorHeap[SL_LR];                         \
    RWStructuredBuffer<float> D = ResourceDescriptorHeap[SL_LD];                         \
    RWStructuredBuffer<float> OF = ResourceDescriptorHeap[SL_LO];

// Hadamard forward transform to level 0 + the zero-guess red sweep. items m0^3.
[numthreads(64, 1, 1)]
void p_fwd_red(uint3 id : SV_DispatchThreadID) {
    MG_RES
    Level L = LV[0];
    int m = int(L.m), n = int(RC.n);
    if (id.x >= uint(m * m * m)) return;
    int t = int(id.x);
    int I = t % m, J = (t / m) % m, K = t / (m * m);
    RWStructuredBuffer<float> rr = ResourceDescriptorHeap[SL_RR];
    float v[8];
    [unroll] for (int k = 0; k < 8; ++k)
        v[k] = rr[((2 * K + (k >> 2)) * n + 2 * J + ((k >> 1) & 1)) * n + 2 * I + (k & 1)];
    bool red = ((I + J + K) & 1) == 0;
    bool fwd_red = RC.k0 != 0;
    [unroll] for (int c = 0; c < 8; ++c) {
        float acc = 0.0f;
        [unroll] for (int k2 = 0; k2 < 8; ++k2) acc += hsign(c, k2 >> 2, (k2 >> 1) & 1, k2 & 1) * v[k2];
        uint li = lidx(L, c, K, J, I);
        R[L.orr + li] = acc;
        if (fwd_red && red) X[L.ox + li] = red_init(acc, D[L.od + li], L.dmin);
    }
}

// resid + fwd_red in one pass (fused path): rr = r - K z0 for the 8 fine cells of a level-0 block, then
// the Hadamard transform + zero-guess red sweep, as p_resid followed by p_fwd_red. items m0^3.
[numthreads(64, 1, 1)]
void p_resid_fwd(uint3 id : SV_DispatchThreadID) {
    MG_RES
    Level L = LV[0];
    int m = int(L.m), n = int(RC.n);
    if (id.x >= uint(m * m * m)) return;
    int t = int(id.x);
    int I = t % m, J = (t / m) % m, K = t / (m * m);
    RWStructuredBuffer<int> fm = ResourceDescriptorHeap[SL_FM];
    RWStructuredBuffer<float> r = ResourceDescriptorHeap[SL_PR];
    RWStructuredBuffer<float> z0 = ResourceDescriptorHeap[SL_Z0];
    float v[8];
    [unroll] for (int k = 0; k < 8; ++k) {
        int i = ((2 * K + (k >> 2)) * n + 2 * J + ((k >> 1) & 1)) * n + 2 * I + (k & 1);
        float w = r[i] - kmul(z0, fm, i, n);
        v[k] = (fm[i] != 0) ? w : 0.0f;
    }
    bool red = ((I + J + K) & 1) == 0;
    [unroll] for (int c = 0; c < 8; ++c) {
        float acc = 0.0f;
        [unroll] for (int k2 = 0; k2 < 8; ++k2) acc += hsign(c, k2 >> 2, (k2 >> 1) & 1, k2 & 1) * v[k2];
        uint li = lidx(L, c, K, J, I);
        R[L.orr + li] = acc;
        if (red) X[L.ox + li] = red_init(acc, D[L.od + li], L.dmin);
    }
}
// One colour of red-black GS on level k0 (k1 = colour), only cells of that colour. items 8*m*m*m/2.
[numthreads(64, 1, 1)]
void p_smooth_h(uint3 id : SV_DispatchThreadID) {
    MG_RES
    Level L = LV[RC.k0];
    int m = int(L.m), h = m / 2;
    if (id.x >= uint(8 * m * m * h)) return;
    int t = int(id.x);
    int Ih = t % h, J = (t / h) % m, K = (t / (h * m)) % m, c = t / (h * m * m);
    int I = 2 * Ih + ((int(RC.k1) + J + K) & 1);
    uint li = lidx(L, c, K, J, I);
    float d = D[L.od + li];
    float v = (R[L.orr + li] - offsum(X, OF, L, c, K, J, I)) / d;   // loads issued before the dmin test
    X[L.ox + li] = (d <= L.dmin) ? 0.0f : v;
}

// Residual restriction (level k0 -> k0+1) + zero-guess red sweep on the coarse level if k1 != 0.
[numthreads(64, 1, 1)]
void p_restrict_red(uint3 id : SV_DispatchThreadID) {
    MG_RES
    Level A = LV[RC.k0];
    Level B = LV[RC.k0 + 1];
    int mb = int(B.m);
    if (id.x >= uint(8 * mb * mb * mb)) return;
    int t = int(id.x);
    int I = t % mb, J = (t / mb) % mb, K = (t / (mb * mb)) % mb, c = t / (mb * mb * mb);
    float acc = 0.0f;
    [unroll] for (int k = 0; k < 8; ++k) {
        int z = 2 * K + (k >> 2), y = 2 * J + ((k >> 1) & 1), x = 2 * I + (k & 1);
        uint li = lidx(A, c, z, y, x);
        acc += R[A.orr + li] - D[A.od + li] * X[A.ox + li] - offsum(X, OF, A, c, z, y, x);
    }
    acc *= 0.125f;
    uint lb = lidx(B, c, K, J, I);
    R[B.orr + lb] = acc;
    if (RC.k1 != 0 && ((I + J + K) & 1) == 0) X[B.ox + lb] = red_init(acc, D[B.od + lb], B.dmin);
}

float xp(RWStructuredBuffer<float> X, Level A, Level B, int c, int K, int J, int I) {
    return X[A.ox + lidx(A, c, K, J, I)] + X[B.ox + lidx(B, c, K >> 1, J >> 1, I >> 1)];
}

// Prolongation from level k0+1 + the black post-sweep on level k0. items 8*m*m*m/2.
[numthreads(64, 1, 1)]
void p_prolong_black(uint3 id : SV_DispatchThreadID) {
    MG_RES
    Level A = LV[RC.k0];
    Level B = LV[RC.k0 + 1];
    int m = int(A.m), h = m / 2;
    if (id.x >= uint(8 * m * m * h)) return;
    int t = int(id.x);
    int Ih = t % h, J = (t / h) % m, K = (t / (h * m)) % m, c = t / (h * m * m);
    int I = 2 * Ih + ((1 + J + K) & 1);
    uint li = lidx(A, c, K, J, I);
    float d = D[A.od + li];
    float a = R[A.orr + li];
    if (I + 1 < m) a -= OF[oidx(A, 0, c, K, J, I)] * xp(X, A, B, c, K, J, I + 1);
    if (I > 0) a -= OF[oidx(A, 0, c, K, J, I - 1)] * xp(X, A, B, c, K, J, I - 1);
    if (J + 1 < m) a -= OF[oidx(A, 1, c, K, J, I)] * xp(X, A, B, c, K, J + 1, I);
    if (J > 0) a -= OF[oidx(A, 1, c, K, J - 1, I)] * xp(X, A, B, c, K, J - 1, I);
    if (K + 1 < m) a -= OF[oidx(A, 2, c, K, J, I)] * xp(X, A, B, c, K + 1, J, I);
    if (K > 0) a -= OF[oidx(A, 2, c, K - 1, J, I)] * xp(X, A, B, c, K - 1, J, I);
    X[A.ox + li] = (d <= A.dmin) ? 0.0f : a / d;
}

// zt = z0 + T^T x1 (fluid cells), 0 in solids. One cell per thread.
[numthreads(FB, 1, 1)]
void p_inv_transform(uint3 id : SV_DispatchThreadID) {
    MG_RES
    uint n = RC.n;
    uint i = id.x;
    if (i >= n * n * n) return;
    RWStructuredBuffer<int> fm = ResourceDescriptorHeap[SL_FM];
    RWStructuredBuffer<float> z0 = ResourceDescriptorHeap[SL_Z0];
    RWStructuredBuffer<float> zt = ResourceDescriptorHeap[SL_ZT];
    if (fm[i] == 0) {
        zt[i] = 0.0f;
        return;
    }
    int3 c3 = unflat(i, n);
    Level L = LV[0];
    int K = c3.z >> 1, J = c3.y >> 1, I = c3.x >> 1;
    float acc = z0[i];
    [unroll] for (int c = 0; c < 8; ++c) acc += hsign(c, c3.z & 1, c3.y & 1, c3.x & 1) * X[L.ox + lidx(L, c, K, J, I)];
    zt[i] = acc;
}

// ---- single-group bottom of the V-cycle (pressure.py k_coarse_block): levels k0 .. k0+k1-1
void cb_sync() { DeviceMemoryBarrierWithGroupSync(); }

void cb_smooth(RWStructuredBuffer<float> X, RWStructuredBuffer<float> R, RWStructuredBuffer<float> D,
               RWStructuredBuffer<float> OF, Level L, int color, bool zero_init, int t) {
    int m = int(L.m);
    int n = 8 * m * m * m;
    for (int i = t; i < n; i += NVFX_CB) {
        int c = i / (m * m * m), K = (i / (m * m)) % m, J = (i / m) % m, I = i % m;
        if (((I + J + K) & 1) == color) {
            uint li = lidx(L, c, K, J, I);
            float d = D[L.od + li];
            float v = 0.0f;
            if (d > L.dmin) {
                float a = R[L.orr + li];
                if (!zero_init) a -= offsum(X, OF, L, c, K, J, I);
                v = a / d;
            }
            X[L.ox + li] = v;
        }
    }
    cb_sync();
}

void cb_restrict(RWStructuredBuffer<float> X, RWStructuredBuffer<float> R, RWStructuredBuffer<float> D,
                 RWStructuredBuffer<float> OF, Level A, Level B, int t) {
    int m = int(B.m);
    int n = 8 * m * m * m;
    for (int i = t; i < n; i += NVFX_CB) {
        int c = i / (m * m * m), K = (i / (m * m)) % m, J = (i / m) % m, I = i % m;
        float acc = 0.0f;
        for (int k = 0; k < 8; ++k) {
            int z = 2 * K + (k >> 2), y = 2 * J + ((k >> 1) & 1), x = 2 * I + (k & 1);
            uint li = lidx(A, c, z, y, x);
            acc += R[A.orr + li] - D[A.od + li] * X[A.ox + li] - offsum(X, OF, A, c, z, y, x);
        }
        R[B.orr + lidx(B, c, K, J, I)] = 0.125f * acc;
    }
    cb_sync();
}

void cb_prolong(RWStructuredBuffer<float> X, RWStructuredBuffer<float> D, Level A, Level B, int t) {
    int m = int(A.m);
    int n = 8 * m * m * m;
    for (int i = t; i < n; i += NVFX_CB) {
        int c = i / (m * m * m), K = (i / (m * m)) % m, J = (i / m) % m, I = i % m;
        uint li = lidx(A, c, K, J, I);
        if (D[A.od + li] > A.dmin) X[A.ox + li] = X[A.ox + li] + X[B.ox + lidx(B, c, K >> 1, J >> 1, I >> 1)];
    }
    cb_sync();
}

void cb_coarsest(RWStructuredBuffer<float> X, RWStructuredBuffer<float> R, RWStructuredBuffer<float> D,
                 RWStructuredBuffer<float> OF, Level L, int t) {
    int m = int(L.m);
    if (m == 1) {
        if (t < 8) {
            float d = D[L.od + t];
            float v = 0.0f;
            if (d > L.dmin) v = R[L.orr + t] / d;
            X[L.ox + t] = v;
        }
        cb_sync();
    } else {
        for (int k = 0; k < 2 * m; ++k) {
            cb_smooth(X, R, D, OF, L, 0, k == 0, t);
            cb_smooth(X, R, D, OF, L, 1, false, t);
        }
    }
}

[numthreads(NVFX_CB, 1, 1)]
void p_coarse_block(uint t : SV_GroupIndex) {
    MG_RES
    int s = int(RC.k0), nlev = int(RC.k1);
    int ti = int(t);
    for (int l = 0; l < nlev - 1; ++l) {
        Level A = LV[s + l];
        cb_smooth(X, R, D, OF, A, 0, true, ti);
        cb_smooth(X, R, D, OF, A, 1, false, ti);
        cb_restrict(X, R, D, OF, A, LV[s + l + 1], ti);
    }
    cb_coarsest(X, R, D, OF, LV[s + nlev - 1], ti);
    for (int l2 = nlev - 2; l2 >= 0; --l2) {
        Level A = LV[s + l2];
        cb_prolong(X, D, A, LV[s + l2 + 1], ti);
        cb_smooth(X, R, D, OF, A, 1, false, ti);
        cb_smooth(X, R, D, OF, A, 0, false, ti);
    }
}

// ---- reference (non-fast) V-cycle kernels, used when the level structure does not allow the fast one
// full-level red-black sweep: k0 = level, k1 = colour, k2 = zero_init
[numthreads(64, 1, 1)]
void p_smooth(uint3 id : SV_DispatchThreadID) {
    MG_RES
    Level L = LV[RC.k0];
    int m = int(L.m);
    if (id.x >= uint(8 * m * m * m)) return;
    int t = int(id.x);
    int I = t % m, J = (t / m) % m, K = (t / (m * m)) % m, c = t / (m * m * m);
    if (((I + J + K) & 1) != int(RC.k1)) return;
    uint li = lidx(L, c, K, J, I);
    float d = D[L.od + li];
    if (d <= L.dmin) {
        X[L.ox + li] = 0.0f;
        return;
    }
    float acc = R[L.orr + li];
    if (RC.k2 == 0) acc -= offsum(X, OF, L, c, K, J, I);
    X[L.ox + li] = acc / d;
}

// prolongation level k0+1 -> k0 (all cells)
[numthreads(64, 1, 1)]
void p_prolong(uint3 id : SV_DispatchThreadID) {
    MG_RES
    Level A = LV[RC.k0];
    Level B = LV[RC.k0 + 1];
    int m = int(A.m);
    if (id.x >= uint(8 * m * m * m)) return;
    int t = int(id.x);
    int I = t % m, J = (t / m) % m, K = (t / (m * m)) % m, c = t / (m * m * m);
    uint li = lidx(A, c, K, J, I);
    if (D[A.od + li] <= A.dmin) return;
    X[A.ox + li] = X[A.ox + li] + X[B.ox + lidx(B, c, K >> 1, J >> 1, I >> 1)];
}

// exact solve on a 1^3 level (k0)
[numthreads(8, 1, 1)]
void p_coarsest(uint3 id : SV_DispatchThreadID) {
    MG_RES
    Level L = LV[RC.k0];
    uint li = id.x;
    float d = D[L.od + li];
    X[L.ox + li] = (d <= L.dmin) ? 0.0f : R[L.orr + li] / d;
}

// =====================================================================================================
// Fused path (default; NVFX_SIM_FUSED=0 records the reference sequence above). Same arithmetic,
// fewer dependent dispatches per PCG iteration (25 -> 14 at 64^3):
//   p_cg_update  = fin_rz + upd_p + the next iteration's matvec_dot. The new p is recomputed at the
//                  stencil neighbours with the identical expression; p ping-pongs between iterations,
//                  the CG scalars and p.q partials are banked (read bank j-1, write bank j), so no group
//                  reads what another group of the same dispatch writes.
//   p_resid_fwd  = p_resid + p_fwd_red (rr of the 8 cells of a level-1 block, from the stored z0).
//   p_coarse_lds = the bottom of the V-cycle from m <= 8 in one 1024-thread group, x / r in groupshared.
// (pre+resid and inv_transform+post_rz with on-the-fly recompute were measured slower and dropped.)
// =====================================================================================================
// (K v)_i with v evaluated on the fly by VALU(j) at clamped indices, every load issued up front; needs
// i, n, m = fm[i], c_ = VALU(i) and fm in scope. Same arithmetic as kmul().
#define KMUL_U(OUT, VALU)                                                                              \
{                                                                                                      \
    int last_ = n * n * n - 1;                                                                         \
    int mp_[3], mm_[3];                                                                                \
    float p1_[3], m1_[3], p2_[3], m2_[3];                                                              \
    [unroll] for (int a_ = 0; a_ < 3; ++a_) {                                                          \
        int st_ = stride_of(a_, n);                                                                    \
        int ip1 = min(i + st_, last_), im1 = max(i - st_, 0);                                          \
        int ip2 = min(i + 2 * st_, last_), im2 = max(i - 2 * st_, 0);                                  \
        mp_[a_] = fm[ip1]; mm_[a_] = fm[im1];                                                          \
        p1_[a_] = VALU(ip1, 0); m1_[a_] = VALU(im1, 0); p2_[a_] = VALU(ip2, 0); m2_[a_] = VALU(im2, 0); \
    }                                                                                                  \
    float d_ = 0.0f;                                                                                   \
    [unroll] for (int ax_ = 0; ax_ < 3; ++ax_) {                                                       \
        bool op_ = ((m >> (2 * ax_)) & 1) == 1;                                                        \
        bool om_ = ((m >> (2 * ax_ + 1)) & 1) == 1;                                                    \
        float gi_ = 0.0f;                                                                              \
        if (op_) gi_ += p1_[ax_] - c_;                                                                 \
        if (om_) gi_ += c_ - m1_[ax_];                                                                 \
        if (op_) {                                                                                     \
            float gn_ = 0.0f;                                                                          \
            if (((mp_[ax_] >> (2 * ax_)) & 1) == 1) gn_ += p2_[ax_] - p1_[ax_];                        \
            if (((mp_[ax_] >> (2 * ax_ + 1)) & 1) == 1) gn_ += p1_[ax_] - c_;                          \
            d_ += gi_ + gn_;                                                                           \
        }                                                                                              \
        if (om_) {                                                                                     \
            float gn_ = 0.0f;                                                                          \
            if (((mm_[ax_] >> (2 * ax_)) & 1) == 1) gn_ += c_ - m1_[ax_];                              \
            if (((mm_[ax_] >> (2 * ax_ + 1)) & 1) == 1) gn_ += m1_[ax_] - m2_[ax_];                    \
            d_ -= gi_ + gn_;                                                                           \
        }                                                                                              \
    }                                                                                                  \
    OUT = -d_;                                                                                         \
}

// fin_rz + upd_p + the next iteration's matvec_dot: CG scalars and the stopping test (every group
// computes them identically; group 0 stores them in bank k5 and writes the predicate k1),
// phi += alpha p_old, p_new = z + beta p_old (into slot k3; p_old in slot k2), q = K p_new,
// partial p_new.q into part_pq bank k5. k0 = init, k4 = bank to read.
float pnew_val(RWStructuredBuffer<float> z, RWStructuredBuffer<float> pin, float bb, bool init, int j) {
    return init ? z[j] : z[j] + bb * pin[j];
}
#define PNV(j, mj) pnew_val(z, pin, bb, init, (j))

[numthreads(FB, 1, 1)]
void p_cg_update(uint3 g : SV_GroupID, uint l : SV_GroupIndex) {
    int n = int(RC.n);
    uint total = uint(n * n * n), nb = nb_flat(RC.n);
    bool init = RC.k0 != 0;
    uint br = RC.k4, bw = RC.k5;
    RWStructuredBuffer<int> fm = ResourceDescriptorHeap[SL_FM];
    RWStructuredBuffer<float> part = ResourceDescriptorHeap[SL_PART];
    RWStructuredBuffer<float> part_pq = ResourceDescriptorHeap[SL_PART_PQ];
    RWStructuredBuffer<float> z = ResourceDescriptorHeap[SL_PZ];
    RWStructuredBuffer<float> pin = ResourceDescriptorHeap[RC.k2];
    RWStructuredBuffer<float> pout = ResourceDescriptorHeap[RC.k3];
    RWStructuredBuffer<float> phi = ResourceDescriptorHeap[SL_PHI];
    RWStructuredBuffer<float> q = ResourceDescriptorHeap[SL_PQ];
    RWByteAddressBuffer scal = ResourceDescriptorHeap[SL_SCAL];
    float rz_new = parts_sum(part, 0, nb, l);
    float mr = parts_max(part, nb, nb, l);
    float pq = 0.0f;
    if (!init) pq = parts_sum(part_pq, br * nb, nb, l);
    float a = 0.0f, bb = 0.0f, best = mr;
    int it = 0;
    if (!init) {
        float rz = sload(scal, S_BANK(br, S_RZ));
        if (pq > 0.0f) a = rz / pq;
        if (rz > 0.0f) bb = rz_new / rz;
        it = int(scal.Load(4 * S_BANK(br, S_IT))) + 1;
        best = min(sload(scal, S_BANK(br, S_BEST)), mr);
    }
    if (g.x == 0 && l == 0) {
        sstore(scal, S_BANK(bw, S_ALPHA), a);
        sstore(scal, S_BANK(bw, S_BETA), bb);
        sstore(scal, S_BANK(bw, S_RZ), rz_new);
        sstore(scal, S_BANK(bw, S_MAXR), mr);
        sstore(scal, S_BANK(bw, S_BEST), best);
        scal.Store(4 * S_BANK(bw, S_IT), uint(it));
        scal.Store(4 * S_IT, uint(it));
        uint go = 0;
        if (it < int(P.maxit) && rz_new > 0.0f && mr > sload(scal, S_THR) && mr <= 4.0f * best) go = 1;
        if (RC.k1 != 0) {
            RWByteAddressBuffer pr = ResourceDescriptorHeap[RC.k1];
            pr.Store2(0, uint2(go, 0));
        }
    }
    float s = 0.0f;
    float o_p[CPT], o_q[CPT], o_phi[CPT];
    [unroll] for (uint k = 0; k < CPT; ++k) {
        int i = int(g.x * (FB * CPT) + k * FB + l);
        o_p[k] = 0.0f; o_q[k] = 0.0f; o_phi[k] = 0.0f;
        if (uint(i) >= total) i = 0;
        {
            int m = fm[i];
            float c_ = PNV(i, m);
            if (!init) o_phi[k] = phi[i] + a * pin[i];
            float val = 0.0f;
            KMUL_U(val, PNV)
            if (m == 0) val = 0.0f;
            o_p[k] = c_;
            o_q[k] = val;
            if (g.x * (FB * CPT) + k * FB + l < total) s += val * c_;
        }
    }
    [unroll] for (uint k2 = 0; k2 < CPT; ++k2) {
        uint i = g.x * (FB * CPT) + k2 * FB + l;
        if (i < total) {
            pout[i] = o_p[k2];
            q[i] = o_q[k2];
            if (!init) phi[i] = o_phi[k2];
        }
    }
    float t = group_sum(s, l);
    if (l == 0) part_pq[bw * nb + g.x] = t;
}

// ---- bottom of the V-cycle in ONE group with groupshared x / r (levels k0 .. k0+k1-1, m_k0 <= 8).
// Same operations in the same order as p_coarse_block; r of the first level stays in global memory
// (already restricted), x of the first level is written back for the following prolongation.
#define CL_T 1024
#define CL_XCAP 4680    // 8*(8^3 + 4^3 + 2^3 + 1)
#define CL_RCAP 584     // 8*(4^3 + 2^3 + 1)
groupshared float s_x[CL_XCAP];
groupshared float s_r[CL_RCAP];

struct LL { int m; uint xo; uint ro; Level G; };

float cl_r(RWStructuredBuffer<float> R, LL a, uint li, bool top) { return top ? R[a.G.orr + li] : s_r[a.ro + li]; }

float cl_offsum(RWStructuredBuffer<float> OF, LL a, int c, int K, int J, int I) {
    int m = a.m;
    Level L = a.G;
    float acc = 0.0f;
    if (I + 1 < m) acc += OF[oidx(L, 0, c, K, J, I)] * s_x[a.xo + lidx(L, c, K, J, I + 1)];
    if (I > 0) acc += OF[oidx(L, 0, c, K, J, I - 1)] * s_x[a.xo + lidx(L, c, K, J, I - 1)];
    if (J + 1 < m) acc += OF[oidx(L, 1, c, K, J, I)] * s_x[a.xo + lidx(L, c, K, J + 1, I)];
    if (J > 0) acc += OF[oidx(L, 1, c, K, J - 1, I)] * s_x[a.xo + lidx(L, c, K, J - 1, I)];
    if (K + 1 < m) acc += OF[oidx(L, 2, c, K, J, I)] * s_x[a.xo + lidx(L, c, K + 1, J, I)];
    if (K > 0) acc += OF[oidx(L, 2, c, K - 1, J, I)] * s_x[a.xo + lidx(L, c, K - 1, J, I)];
    return acc;
}

void cl_smooth(RWStructuredBuffer<float> R, RWStructuredBuffer<float> D, RWStructuredBuffer<float> OF, LL a, bool top,
               int color, bool zero_init, int t) {
    int m = a.m, h = (m + 1) / 2;
    int n = 8 * m * m * h;   // cells of one colour (approximately: odd m handled by the colour test)
    for (int u = t; u < n; u += CL_T) {
        int Ih = u % h, J = (u / h) % m, K = (u / (h * m)) % m, c = u / (h * m * m);
        int I = 2 * Ih + ((color + J + K) & 1);
        if (I < m) {
            uint li = lidx(a.G, c, K, J, I);
            float d = D[a.G.od + li];
            float acc = cl_r(R, a, li, top);
            if (!zero_init) acc -= cl_offsum(OF, a, c, K, J, I);
            s_x[a.xo + li] = (d > a.G.dmin) ? acc / d : 0.0f;
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

void cl_restrict(RWStructuredBuffer<float> R, RWStructuredBuffer<float> D, RWStructuredBuffer<float> OF, LL a, LL b,
                 bool top, int t) {
    int m = b.m;
    int n = 8 * m * m * m;
    for (int u = t; u < n; u += CL_T) {
        int c = u / (m * m * m), K = (u / (m * m)) % m, J = (u / m) % m, I = u % m;
        float acc = 0.0f;
        [unroll] for (int k = 0; k < 8; ++k) {
            int zz = 2 * K + (k >> 2), yy = 2 * J + ((k >> 1) & 1), xx = 2 * I + (k & 1);
            uint li = lidx(a.G, c, zz, yy, xx);
            acc += cl_r(R, a, li, top) - D[a.G.od + li] * s_x[a.xo + li] - cl_offsum(OF, a, c, zz, yy, xx);
        }
        s_r[b.ro + lidx(b.G, c, K, J, I)] = 0.125f * acc;
    }
    GroupMemoryBarrierWithGroupSync();
}

void cl_prolong(RWStructuredBuffer<float> D, LL a, LL b, int t) {
    int m = a.m;
    int n = 8 * m * m * m;
    for (int u = t; u < n; u += CL_T) {
        int c = u / (m * m * m), K = (u / (m * m)) % m, J = (u / m) % m, I = u % m;
        uint li = lidx(a.G, c, K, J, I);
        if (D[a.G.od + li] > a.G.dmin) s_x[a.xo + li] = s_x[a.xo + li] + s_x[b.xo + lidx(b.G, c, K >> 1, J >> 1, I >> 1)];
    }
    GroupMemoryBarrierWithGroupSync();
}

[numthreads(CL_T, 1, 1)]
void p_coarse_lds(uint t : SV_GroupIndex) {
    MG_RES
    int s = int(RC.k0), nlev = int(RC.k1);
    int ti = int(t);
    LL lv[4];
    uint xo = 0, ro = 0;
    [unroll] for (int k = 0; k < 4; ++k) {
        if (k < nlev) {
            lv[k].G = LV[s + k];
            lv[k].m = int(lv[k].G.m);
            lv[k].xo = xo;
            lv[k].ro = ro;
            uint sz = 8u * lv[k].G.m * lv[k].G.m * lv[k].G.m;
            xo += sz;
            if (k > 0) ro += sz;
        } else {
            lv[k] = lv[0];
        }
    }
    // shift r offsets: level 0's r is global, level k's r starts at the sum of sizes of levels 1..k-1
    [unroll] for (int k2 = 1; k2 < 4; ++k2) {
        if (k2 < nlev) {
            uint o = 0;
            for (int j = 1; j < k2; ++j) o += 8u * lv[j].G.m * lv[j].G.m * lv[j].G.m;
            lv[k2].ro = o;
        }
    }
    [unroll] for (int d = 0; d < 3; ++d) {
        if (d < nlev - 1) {
            cl_smooth(R, D, OF, lv[d], d == 0, 0, true, ti);
            cl_smooth(R, D, OF, lv[d], d == 0, 1, false, ti);
            cl_restrict(R, D, OF, lv[d], lv[d + 1], d == 0, ti);
        }
    }
    LL c = lv[nlev - 1];
    bool ctop = nlev == 1;
    if (c.m == 1) {
        if (ti < 8) {
            float dd = D[c.G.od + ti];
            float v = 0.0f;
            if (dd > c.G.dmin) v = cl_r(R, c, uint(ti), ctop) / dd;
            s_x[c.xo + ti] = v;
        }
        GroupMemoryBarrierWithGroupSync();
    } else {
        for (int k = 0; k < 2 * c.m; ++k) {
            cl_smooth(R, D, OF, c, ctop, 0, k == 0, ti);
            cl_smooth(R, D, OF, c, ctop, 1, false, ti);
        }
    }
    [unroll] for (int u = 2; u >= 0; --u) {
        if (u < nlev - 1) {
            cl_prolong(D, lv[u], lv[u + 1], ti);
            cl_smooth(R, D, OF, lv[u], u == 0, 1, false, ti);
            cl_smooth(R, D, OF, lv[u], u == 0, 0, false, ti);
        }
    }
    LL a0 = lv[0];
    int n0 = 8 * a0.m * a0.m * a0.m;
    for (int w = ti; w < n0; w += CL_T) X[a0.G.ox + uint(w)] = s_x[a0.xo + uint(w)];
}
