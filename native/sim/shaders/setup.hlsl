// Setup kernels: state init, open-face mask, multigrid operator probe + Galerkin (pressure.py).
#include "common.hlsli"

// k0 bit0: reset both state sets (density 0, u 0, T ambient, fuel 0); bit1: phi := 0
[numthreads(8, 8, 4)]
void s_init(uint3 id : SV_DispatchThreadID) {
    uint n = RC.n;
    if (any(id >= n)) return;
    if ((RC.k0 & 1) != 0) {
        [unroll] for (uint s = 0; s < 2; ++s) {
            RWTexture3D<float4> vd = ResourceDescriptorHeap[SL_VD_UAV0 + s];
            RWTexture3D<float2> tf = ResourceDescriptorHeap[SL_TF_UAV0 + s];
            vd[id] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            tf[id] = float2(RC.k1 == 0 ? 293.0f : asfloat(RC.k1), 0.0f);
        }
    }
    if ((RC.k0 & 2) != 0) {
        RWStructuredBuffer<float> phi = ResourceDescriptorHeap[SL_PHI];
        phi[(id.z * n + id.y) * n + id.x] = 0.0f;
    }
}

// open-face mask: bits 0 +x, 1 -x, 2 +y, 3 -y, 4 +z, 5 -z, FLUID = 64
[numthreads(8, 8, 4)]
void s_mask(uint3 id : SV_DispatchThreadID) {
    int n = int(RC.n);
    if (any(id >= uint(n))) return;
    Texture3D<float> sol = ResourceDescriptorHeap[SL_SOLID_SRV];
    RWStructuredBuffer<int> fm = ResourceDescriptorHeap[SL_FM];
    int3 c = int3(id);
    uint i = (id.z * uint(n) + id.y) * uint(n) + id.x;
    if (sol[c] > 0.5f) {
        fm[i] = 0;
        return;
    }
    int m = 64;
    if (c.x + 1 < n) { if (sol[c + int3(1, 0, 0)] <= 0.5f) m = m | 1; }
    if (c.x > 0)     { if (sol[c - int3(1, 0, 0)] <= 0.5f) m = m | 2; }
    if (c.y + 1 < n) { if (sol[c + int3(0, 1, 0)] <= 0.5f) m = m | 4; }
    if (c.y > 0)     { if (sol[c - int3(0, 1, 0)] <= 0.5f) m = m | 8; }
    if (c.z + 1 < n) { if (sol[c + int3(0, 0, 1)] <= 0.5f) m = m | 16; }
    if (c.z > 0)     { if (sol[c - int3(0, 0, 1)] <= 0.5f) m = m | 32; }
    fm[i] = m;
}

float hsign(int c, int dz, int dy, int dx) {
    int e = (c & 1) * dx + ((c >> 1) & 1) * dy + ((c >> 2) & 1) * dz;
    return ((e & 1) == 1) ? -HS : HS;
}

// coordinates below are (z, y, x) as in pressure.py
int mask_at(RWStructuredBuffer<int> fm, int3 q, int n) {
    if (any(q < 0) || any(q >= n)) return 0;
    return fm[(q.x * n + q.y) * n + q.z];
}
int3 ax_off(int ax) { return ax == 0 ? int3(0, 0, 1) : (ax == 1 ? int3(0, 1, 0) : int3(1, 0, 0)); }

float tv(int c, int3 B, int3 q) {
    if ((q.x >> 1) == B.x && (q.y >> 1) == B.y && (q.z >> 1) == B.z) return hsign(c, q.x & 1, q.y & 1, q.z & 1);
    return 0.0f;
}

float gt(RWStructuredBuffer<int> fm, int n, int c, int3 B, int3 q, int ax) {
    int m = mask_at(fm, q, n);
    int3 e = ax_off(ax);
    float t0 = tv(c, B, q);
    float g = 0.0f;
    if (((m >> (2 * ax)) & 1) == 1) g += tv(c, B, q + e) - t0;
    if (((m >> (2 * ax + 1)) & 1) == 1) g += t0 - tv(c, B, q - e);
    return g;
}

int3 axcell(int ax, int3 o, int a, int b, int c) {
    if (ax == 0) return int3(o.x + c, o.y + b, o.z + a);
    if (ax == 1) return int3(o.x + c, o.y + a, o.z + b);
    return int3(o.x + a, o.y + c, o.z + b);
}

uint lidx(Level L, int c, int K, int J, int I) { return uint(((c * int(L.m) + K) * int(L.m) + J) * int(L.m) + I); }

// exact component-diagonal block of T K T^T on level 0. items 8*m^3.
[numthreads(64, 1, 1)]
void s_probe1(uint3 id : SV_DispatchThreadID) {
    StructuredBuffer<Level> LV = ResourceDescriptorHeap[SL_LEVELS];
    RWStructuredBuffer<float> D = ResourceDescriptorHeap[SL_LD];
    RWStructuredBuffer<float> OF = ResourceDescriptorHeap[SL_LO];
    RWStructuredBuffer<int> fm = ResourceDescriptorHeap[SL_FM];
    Level L = LV[0];
    int m = int(L.m), n = int(RC.n);
    if (id.x >= uint(8 * m * m * m)) return;
    int t = int(id.x);
    int I = t % m, J = (t / m) % m, K = (t / (m * m)) % m, c = t / (m * m * m);
    int3 B = int3(K, J, I);
    int3 o = int3(2 * K, 2 * J, 2 * I);
    float dsum = 0.0f;
    for (int ax = 0; ax < 3; ++ax)
        for (int b = 0; b < 2; ++b)
            for (int cc = 0; cc < 2; ++cc)
                for (int a = 0; a < 4; ++a) {
                    float v = gt(fm, n, c, B, axcell(ax, o, a - 1, b, cc), ax);
                    dsum += v * v;
                }
    D[L.od + lidx(L, c, K, J, I)] = dsum;
    for (int ax2 = 0; ax2 < 3; ++ax2) {
        int3 Bn = B + ax_off(ax2);
        float s = 0.0f;
        if (Bn.x < m && Bn.y < m && Bn.z < m) {
            for (int b = 0; b < 2; ++b)
                for (int cc = 0; cc < 2; ++cc)
                    for (int a = 0; a < 2; ++a) {
                        int3 q = axcell(ax2, o, 1 + a, b, cc);
                        s += gt(fm, n, c, B, q, ax2) * gt(fm, n, c, Bn, q, ax2);
                    }
        }
        OF[L.oo + lidx(L, ax2 * 8 + c, K, J, I)] = s;
    }
}

// Galerkin coarse operator level k0 -> k0+1, scale asfloat(k1). items 8*mb^3.
[numthreads(64, 1, 1)]
void s_galerkin(uint3 id : SV_DispatchThreadID) {
    StructuredBuffer<Level> LV = ResourceDescriptorHeap[SL_LEVELS];
    RWStructuredBuffer<float> D = ResourceDescriptorHeap[SL_LD];
    RWStructuredBuffer<float> OF = ResourceDescriptorHeap[SL_LO];
    Level A = LV[RC.k0];
    Level B = LV[RC.k0 + 1];
    float scale = asfloat(RC.k1);
    int m = int(B.m);
    if (id.x >= uint(8 * m * m * m)) return;
    int t = int(id.x);
    int I = t % m, J = (t / m) % m, K = (t / (m * m)) % m, c = t / (m * m * m);
    int3 o = int3(2 * K, 2 * J, 2 * I);
    float d = 0.0f;
    for (int k = 0; k < 8; ++k) d += D[A.od + lidx(A, c, o.x + (k >> 2), o.y + ((k >> 1) & 1), o.z + (k & 1))];
    for (int ax = 0; ax < 3; ++ax) {
        float oi = 0.0f, oo = 0.0f;
        for (int b = 0; b < 2; ++b)
            for (int cc = 0; cc < 2; ++cc) {
                int3 q = axcell(ax, o, 0, b, cc);
                oi += OF[A.oo + lidx(A, ax * 8 + c, q.x, q.y, q.z)];
                int3 q2 = axcell(ax, o, 1, b, cc);
                oo += OF[A.oo + lidx(A, ax * 8 + c, q2.x, q2.y, q2.z)];
            }
        d += 2.0f * oi;
        OF[B.oo + lidx(B, ax * 8 + c, K, J, I)] = oo * scale;
    }
    D[B.od + lidx(B, c, K, J, I)] = d * scale;
}
