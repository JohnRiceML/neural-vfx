// Advection (kernels.py advect_sl / advect_mc) + the density mass correction reduction
// (solver.py k_part_mass / k_fin_mass). Manual trilinear interpolation from Loads (full fp32).
#include "common.hlsli"

int base_idx(float g, int n) { return min(int(floor(clamp(g, 0.0f, float(n - 1)))), n - 2); }
float frac_of(float g, int n, int i0) { return clamp(g, 0.0f, float(n - 1)) - float(i0); }

struct TriP { int3 b; float3 f; };   // base cell (x, y, z), fractions (x, y, z)

TriP tri_setup(float3 p, int n) {
    TriP s;
    s.b = int3(base_idx(p.x, n), base_idx(p.y, n), base_idx(p.z, n));
    s.f = float3(frac_of(p.x, n, s.b.x), frac_of(p.y, n, s.b.y), frac_of(p.z, n, s.b.z));
    return s;
}

// kernels.py tri(): x first, then y, then z, same operation order.
#define DEF_TRI(NAME, TEX, V)                                                                      \
V NAME(TEX t, TriP s) {                                                                            \
    int x0 = s.b.x; int y0 = s.b.y; int z0 = s.b.z;                                                \
    float fx = s.f.x; float fy = s.f.y; float fz = s.f.z;                                          \
    V a00 = t[int3(x0, y0, z0)] * (1.0f - fx) + t[int3(x0 + 1, y0, z0)] * fx;                      \
    V a01 = t[int3(x0, y0 + 1, z0)] * (1.0f - fx) + t[int3(x0 + 1, y0 + 1, z0)] * fx;              \
    V a10 = t[int3(x0, y0, z0 + 1)] * (1.0f - fx) + t[int3(x0 + 1, y0, z0 + 1)] * fx;              \
    V a11 = t[int3(x0, y0 + 1, z0 + 1)] * (1.0f - fx) + t[int3(x0 + 1, y0 + 1, z0 + 1)] * fx;      \
    V a0 = a00 * (1.0f - fy) + a01 * fy;                                                           \
    V a1 = a10 * (1.0f - fy) + a11 * fy;                                                           \
    return a0 * (1.0f - fz) + a1 * fz;                                                             \
}
DEF_TRI(tri_vd, Texture3D<float4>, float4)
DEF_TRI(tri_tf, Texture3D<float2>, float2)
DEF_TRI(tri_rvd, RWTexture3D<float4>, float4)
DEF_TRI(tri_rtf, RWTexture3D<float2>, float2)

float3 sample_vel(Texture3D<float4> v, float3 p, int n) { return tri_vd(v, tri_setup(p, n)).xyz; }

// RK2 (midpoint) trace over signed time s (cells per (m/s))
float3 trace(Texture3D<float4> v, float3 p, float s, int n) {
    float3 v1 = sample_vel(v, p, n);
    float3 v2 = sample_vel(v, p + (0.5f * s) * v1, n);
    return p + s * v2;
}

// k0 = current state set (0/1)
[numthreads(8, 8, 4)]
void adv_sl(uint3 id : SV_DispatchThreadID) {
    int n = int(RC.n);
    if (any(id >= uint(n))) return;
    Texture3D<float4> vd = ResourceDescriptorHeap[SL_VD_SRV0 + RC.k0];
    Texture3D<float2> tf = ResourceDescriptorHeap[SL_TF_SRV0 + RC.k0];
    RWTexture3D<float4> hvd = ResourceDescriptorHeap[SL_HAT_VD];
    RWTexture3D<float2> htf = ResourceDescriptorHeap[SL_HAT_TF];
    float3 q = trace(vd, float3(id), -P.dt * P.inv_dx, n);
    TriP s = tri_setup(q, n);
    hvd[id] = tri_vd(vd, s);
    htf[id] = tri_tf(tf, s);
}

groupshared float2 g_red[256];

// MacCormack correction, clamped to the 8 source values around the backtraced point; plus the
// per-group partial fluid mass before (state) and after (adv) advection.
// k0 = current set, k1 = NBM (number of groups)
[numthreads(8, 8, 4)]
void adv_mc(uint3 id : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex) {
    int n = int(RC.n);
    float2 mass = float2(0.0f, 0.0f);
    if (all(id < uint(n))) {
        Texture3D<float4> vd = ResourceDescriptorHeap[SL_VD_SRV0 + RC.k0];
        Texture3D<float2> tf = ResourceDescriptorHeap[SL_TF_SRV0 + RC.k0];
        Texture3D<float> sol = ResourceDescriptorHeap[SL_SOLID_SRV];
        RWTexture3D<float4> hvd = ResourceDescriptorHeap[SL_HAT_VD];
        RWTexture3D<float2> htf = ResourceDescriptorHeap[SL_HAT_TF];
        RWTexture3D<float4> avd = ResourceDescriptorHeap[SL_ADV_VD];
        RWTexture3D<float2> atf = ResourceDescriptorHeap[SL_ADV_TF];
        float s = P.dt * P.inv_dx;
        float3 p = float3(id);
        float3 q = trace(vd, p, -s, n);
        float3 r = trace(vd, p, s, n);
        int3 qb = int3(base_idx(q.x, n), base_idx(q.y, n), base_idx(q.z, n));
        TriP sr = tri_setup(r, n);

        float4 src4 = vd[id];
        float4 h4 = hvd[id];
        float4 val4 = h4 + 0.5f * (src4 - tri_rvd(hvd, sr));
        float4 lo4 = vd[qb];
        float4 hi4 = lo4;
        float2 h2 = htf[id];
        float2 val2 = h2 + 0.5f * (tf[id] - tri_rtf(htf, sr));
        float2 lo2 = tf[qb];
        float2 hi2 = lo2;
        [unroll] for (int k = 1; k < 8; ++k) {
            int3 c = qb + int3(k & 1, (k >> 1) & 1, k >> 2);
            float4 v4 = vd[c];
            lo4 = min(lo4, v4);
            hi4 = max(hi4, v4);
            float2 v2 = tf[c];
            lo2 = min(lo2, v2);
            hi2 = max(hi2, v2);
        }
        val4 = select(or(val4 < lo4, val4 > hi4), h4, val4);
        val2 = select(or(val2 < lo2, val2 > hi2), h2, val2);
        avd[id] = val4;
        atf[id] = val2;
        if (sol[id] <= 0.5f) mass = float2(src4.w, val4.w);
    }
    g_red[gi] = mass;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint st = 128; st > 0; st >>= 1) {
        if (gi < st) g_red[gi] += g_red[gi + st];
        GroupMemoryBarrierWithGroupSync();
    }
    if (gi == 0) {
        RWStructuredBuffer<float> part = ResourceDescriptorHeap[SL_MPART];
        uint gx = (uint(n) + 7) / 8, gy = (uint(n) + 7) / 8;
        uint b = (gid.z * gy + gid.y) * gx + gid.x;
        part[b] = g_red[0].x;
        part[RC.k1 + b] = g_red[0].y;
    }
}

// ratio = pre / post fluid mass (1 if either is ~0). k1 = NBM
[numthreads(256, 1, 1)]
void mass_fin(uint gi : SV_GroupIndex) {
    RWStructuredBuffer<float> part = ResourceDescriptorHeap[SL_MPART];
    uint nb = RC.k1;
    float2 acc = float2(0.0f, 0.0f);
    for (uint i = gi; i < nb; i += 256) acc += float2(part[i], part[nb + i]);
    g_red[gi] = acc;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint st = 128; st > 0; st >>= 1) {
        if (gi < st) g_red[gi] += g_red[gi + st];
        GroupMemoryBarrierWithGroupSync();
    }
    if (gi == 0) {
        float a = g_red[0].x, b = g_red[0].y;
        float ratio = 1.0f;
        if (a > 0.0f && b > 1.0e-30f) ratio = a / b;
        RWByteAddressBuffer scal = ResourceDescriptorHeap[SL_SCAL];
        scal.Store(4 * S_MASS_RATIO, asuint(ratio));
    }
}
