// Volume prep (skipped under NVFX_RENDER_REUSE_VOLUME): port of renderer.py _upload_solid / pack + _prep
// (pack_volume, pack_solid + the torch blur, light_volume, the fire-glow pyramid). OWNER: native render engineer.
//
// Six dependent stages; each is one flat dispatch of 64-thread groups that runs up to three independent jobs
// (thread ranges RC.k0 | RC.k1 | RC.k2), so independent work shares a dispatch and a barrier:
//   prep1: pack (density,T)->RG16F, velocity->RGBA16F, solid blur z | clear occupancy
//   prep2: density occupancy scatter, solid blur y                   | glow adaptive pool L^3 -> 32^3
//   prep3: solid blur x -> R8 texture + solid occupancy              | glow blur z (32^3) | glow avg_pool 4 (8^3)
//   prep4: light volume (L^3)                                        | glow blur y        | coarse blur z
//   prep5:                                                           | glow blur x        | coarse blur y
//   prep6: glow = g32b + glow_coarse * trilinear(coarse blur x)  -> RGBA16F texture
#include "render_common.hlsli"

// ---------------------------------------------------------------- jobs
// pack_volume: half2 (max(d,0), T) and the velocity texture (Python _upload_velocity: (vx,vy,vz,0) fp16)
void job_pack(uint i) {
    uint N = P.N;
    int3 c = int3(unflat3(i, N));
    Texture3D<float4> vdT = ResourceDescriptorHeap[P.in_vd];
    Texture3D<float2> tfT = ResourceDescriptorHeap[P.in_tf];
    float4 vd = vdT.Load(int4(c, 0));
    float2 tf = tfT.Load(int4(c, 0));
    RWTexture3D<float2> vol = ResourceDescriptorHeap[R_VOL_UAV];
    RWTexture3D<float4> vel = ResourceDescriptorHeap[R_VEL_UAV];
    vol[c] = float2(half_round(max(vd.w, 0.0f)), half_round(tf.x));
    vel[c] = float4(half_round(vd.x), half_round(vd.y), half_round(vd.z), 0.0f);
    if (P.has_solid != 0) {
        // torch: conv3d over D (z) first, replicate padding, gaussian sigma 1.0 (7 taps)
        Texture3D<float> sT = ResourceDescriptorHeap[P.in_solid];
        RWStructuredBuffer<float> s1 = ResourceDescriptorHeap[R_S1];
        float acc = 0.0f;
        [unroll] for (int k = 0; k < 7; ++k)
            acc += w_gs(k) * sT.Load(int4(c.x, c.y, clamp(c.z + k - 3, 0, int(N) - 1), 0));
        s1[i] = acc;
    }
}

void job_clear_occ(uint j) {
    RWStructuredBuffer<uint> od = ResourceDescriptorHeap[R_OCCD];
    RWStructuredBuffer<uint> os = ResourceDescriptorHeap[R_OCCS];
    od[j] = 0u;
    os[j] = 0u;
}

// occupancy bricks with a one-voxel apron (kernels.py pack_volume / pack_solid); all writers store 1
void mark_bricks(RWStructuredBuffer<uint> occ, int3 c, int N) {
    const int B = R_BRICKS;
    int bz0 = max(c.z - 1, 0) * B / N, bz1 = min(c.z + 1, N - 1) * B / N;
    int by0 = max(c.y - 1, 0) * B / N, by1 = min(c.y + 1, N - 1) * B / N;
    int bx0 = max(c.x - 1, 0) * B / N, bx1 = min(c.x + 1, N - 1) * B / N;
    for (int bz = bz0; bz <= bz1; ++bz)
        for (int by = by0; by <= by1; ++by)
            for (int bx = bx0; bx <= bx1; ++bx)
                occ[flat3(int3(bx, by, bz), B)] = 1u;
}

void job_occ_density(uint i) {
    uint N = P.N;
    int3 c = int3(unflat3(i, N));
    Texture3D<float4> vdT = ResourceDescriptorHeap[P.in_vd];
    Texture3D<float2> tfT = ResourceDescriptorHeap[P.in_tf];
    float d = vdT.Load(int4(c, 0)).w;
    float T = tfT.Load(int4(c, 0)).x;
    if (d > P.d_eps || T > P.t_lo) {     // pack_volume(..., d_eps, t_hot = t_lo), fp32 inputs
        RWStructuredBuffer<uint> occ = ResourceDescriptorHeap[R_OCCD];
        mark_bricks(occ, c, int(N));
    }
    if (P.has_solid != 0) {
        RWStructuredBuffer<float> s1 = ResourceDescriptorHeap[R_S1];
        RWStructuredBuffer<float> s2 = ResourceDescriptorHeap[R_S2];
        float acc = 0.0f;
        [unroll] for (int k = 0; k < 7; ++k)
            acc += w_gs(k) * s1[flat3(int3(c.x, clamp(c.y + k - 3, 0, int(N) - 1), c.z), N)];
        s2[i] = acc;
    }
}

void job_solid_x(uint i) {
    uint N = P.N;
    int3 c = int3(unflat3(i, N));
    RWStructuredBuffer<float> s2 = ResourceDescriptorHeap[R_S2];
    float s = 0.0f;
    [unroll] for (int k = 0; k < 7; ++k)
        s += w_gs(k) * s2[flat3(int3(clamp(c.x + k - 3, 0, int(N) - 1), c.y, c.z), N)];
    // pack_solid: uint8(clamp(int(s * 255 + 0.5), 0, 255)) into a normalized uint8 texture
    int u = clamp(int(s * 255.0f + 0.5f), 0, 255);
    RWTexture3D<unorm float> sol = ResourceDescriptorHeap[R_SOLID8_UAV];
    sol[c] = float(u) / 255.0f;
    if (s > 0.0f) {
        RWStructuredBuffer<uint> occ = ResourceDescriptorHeap[R_OCCS];
        mark_bricks(occ, c, int(N));
    }
}

// emission of one light-grid voxel, as light_volume writes it into glow_src (fp16)
float3 glow_src_at(Texture3D<float2> vol, uint3 v, float inv) {
    float3 pos = float3((float(v.x) + 0.5f) * inv, (float(v.y) + 0.5f) * inv, (float(v.z) + 0.5f) * inv);
    float2 here = vol.SampleLevel(s_lin, pos, 0.0f);
    float3 e = float3(0.0f, 0.0f, 0.0f);
    if (here.y > P.t_lo) e = emission(here.x, here.y);
    return float3(half_round(e.x), half_round(e.y), half_round(e.z));
}

// F.adaptive_avg_pool3d(glow_src, 32): torch window [floor(i*L/32), ceil((i+1)*L/32)), sum z,y,x, / kT / kH / kW.
// The glow source is recomputed here from the volume (bit-identical to light_volume's), so this does not
// wait for the light march.
void job_glow_pool(uint j) {
    const uint G = R_GLOW;
    uint L = P.L;
    uint3 g = unflat3(j, G);
    uint3 s = (g * L) / G;
    uint3 e = ((g + 1u) * L + G - 1u) / G;
    Texture3D<float2> vol = ResourceDescriptorHeap[R_VOL_SRV];
    float inv = 1.0f / float(L);
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    for (uint z = s.z; z < e.z; ++z)
        for (uint y = s.y; y < e.y; ++y)
            for (uint x = s.x; x < e.x; ++x)
                sum += glow_src_at(vol, uint3(x, y, z), inv);
    float3 k = float3(e - s);
    RWStructuredBuffer<float4> g0 = ResourceDescriptorHeap[R_G0];
    g0[j] = float4(sum / k.z / k.y / k.x, 0.0f);
}

// separable 7-tap blur (sigma 1.2) along one axis of an n^3 float4 grid, replicate padding
float3 blur_axis(RWStructuredBuffer<float4> src, int3 c, int n, int axis) {
    float3 acc = float3(0.0f, 0.0f, 0.0f);
    [unroll] for (int k = 0; k < 7; ++k) {
        int3 q = c;
        q[axis] = clamp(c[axis] + k - 3, 0, n - 1);
        acc += w_g3(k) * src[flat3(q, n)].xyz;
    }
    return acc;
}

void job_gblur(uint j, uint src_slot, uint dst_slot, int n, int axis) {
    RWStructuredBuffer<float4> src = ResourceDescriptorHeap[src_slot];
    RWStructuredBuffer<float4> dst = ResourceDescriptorHeap[dst_slot];
    int3 c = int3(unflat3(j, n));
    dst[j] = float4(blur_axis(src, c, n, axis), 0.0f);
}

// F.avg_pool3d(g32, 4): 32^3 -> 8^3, sum z,y,x, / 64
void job_pool4(uint j) {
    RWStructuredBuffer<float4> g0 = ResourceDescriptorHeap[R_G0];
    RWStructuredBuffer<float4> h0 = ResourceDescriptorHeap[R_H0];
    int3 h = int3(unflat3(j, R_GLOW_C));
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    for (int z = 0; z < 4; ++z)
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x)
                sum += g0[flat3(h * 4 + int3(x, y, z), R_GLOW)].xyz;
    h0[j] = float4(sum / 64.0f, 0.0f);
}

// light_volume: transmittance toward the key light (march to the cube exit), sky occlusion (6 steps up)
void job_light(uint i) {
    uint L = P.L;
    uint3 v = unflat3(i, L);
    float inv = 1.0f / float(L);
    float3 pos = float3((float(v.x) + 0.5f) * inv, (float(v.y) + 0.5f) * inv, (float(v.z) + 0.5f) * inv);
    Texture3D<float2> vol = ResourceDescriptorHeap[R_VOL_SRV];
    Texture3D<float> sol = ResourceDescriptorHeap[R_SOLID8_SRV];
    float3 ld = P.light_dir.xyz;
    float2 tb = box_hit(pos, safe_inv(ld), float3(0.0f, 0.0f, 0.0f), float3(1.0f, 1.0f, 1.0f));
    float t_exit = tb.y;
    float ds = 1.5f * inv;
    float od = 0.0f;
    float t = (0.25f + 0.75f * ign(float(v.x) + 7.0f * float(v.z), float(v.y) + 13.0f * float(v.z))) * ds;
    while (t < t_exit) {
        float3 q = pos + ld * t;
        float2 s = vol.SampleLevel(s_lin, q, 0.0f);
        od += P.k_shadow * s.x * ds;
        if (P.has_solid != 0) od += P.k_solid * sol.SampleLevel(s_lin, q, 0.0f) * ds;
        if (od > 12.0f) break;
        t += ds;
    }
    float oda = 0.0f;
    float da = 3.0f * inv;
    [unroll] for (int k = 0; k < 6; ++k) {
        float qy = pos.y + (float(k) + 0.5f) * da;
        if (qy < 1.0f) {
            float2 s = vol.SampleLevel(s_lin, float3(pos.x, qy, pos.z), 0.0f);
            oda += P.k_amb * s.x * da;
        }
    }
    RWTexture3D<float2> light = ResourceDescriptorHeap[R_LIGHT_UAV];
    light[v] = float2(half_round(exp(-od)), half_round(exp(-oda)));
}

// ---------------------------------------------------------------- stages
[numthreads(64, 1, 1)] void r_prep1(uint3 id : SV_DispatchThreadID) {
    uint i = id.y * RC.k7 + id.x;   // flat index over a 2D grid of 64-thread groups
    if (i < RC.k0) { job_pack(i); return; }
    i -= RC.k0;
    if (i < RC.k1) job_clear_occ(i);
}

[numthreads(64, 1, 1)] void r_prep2(uint3 id : SV_DispatchThreadID) {
    uint i = id.y * RC.k7 + id.x;   // flat index over a 2D grid of 64-thread groups
    if (i < RC.k0) { job_occ_density(i); return; }
    i -= RC.k0;
    if (i < RC.k1) job_glow_pool(i);
}

[numthreads(64, 1, 1)] void r_prep3(uint3 id : SV_DispatchThreadID) {
    uint i = id.y * RC.k7 + id.x;   // flat index over a 2D grid of 64-thread groups
    if (i < RC.k0) { job_solid_x(i); return; }
    i -= RC.k0;
    if (i < RC.k1) { job_gblur(i, R_G0, R_G1, R_GLOW, 2); return; }
    i -= RC.k1;
    if (i < RC.k2) job_pool4(i);
}

[numthreads(64, 1, 1)] void r_prep4(uint3 id : SV_DispatchThreadID) {
    uint i = id.y * RC.k7 + id.x;   // flat index over a 2D grid of 64-thread groups
    if (i < RC.k0) { job_light(i); return; }
    i -= RC.k0;
    if (i < RC.k1) { job_gblur(i, R_G1, R_G2, R_GLOW, 1); return; }
    i -= RC.k1;
    if (i < RC.k2) job_gblur(i, R_H0, R_H1, R_GLOW_C, 2);
}

[numthreads(64, 1, 1)] void r_prep5(uint3 id : SV_DispatchThreadID) {
    uint i = id.y * RC.k7 + id.x;   // flat index over a 2D grid of 64-thread groups
    if (i < RC.k0) { job_gblur(i, R_G2, R_G3, R_GLOW, 0); return; }
    i -= RC.k0;
    if (i < RC.k1) job_gblur(i, R_H1, R_H2, R_GLOW_C, 1);
}

// coarse level x-blur at one 8^3 point (the last separable pass, folded into the upsample)
float3 coarse_at(RWStructuredBuffer<float4> h2, int x, int y, int z) {
    return blur_axis(h2, int3(x, y, z), R_GLOW_C, 0);
}

// torch upsample (align_corners=False) source index along one axis: in 8 -> out 32
void up_src(uint d, out int i0, out int ip, out float l0, out float l1) {
    const float scale = float(R_GLOW_C) / float(R_GLOW);
    float src = max(scale * (float(d) + 0.5f) - 0.5f, 0.0f);
    i0 = int(src);
    ip = i0 < R_GLOW_C - 1 ? 1 : 0;
    l1 = src - float(i0);
    l0 = 1.0f - l1;
}

[numthreads(64, 1, 1)] void r_prep6(uint3 id : SV_DispatchThreadID) {
    uint j = id.y * RC.k7 + id.x;
    if (j >= R_GLOW * R_GLOW * R_GLOW) return;
    uint3 g = unflat3(j, R_GLOW);
    int t1, t1p, h1, h1p, w1, w1p;
    float t0l, t1l, h0l, h1l, w0l, w1l;
    up_src(g.z, t1, t1p, t0l, t1l);
    up_src(g.y, h1, h1p, h0l, h1l);
    up_src(g.x, w1, w1p, w0l, w1l);
    RWStructuredBuffer<float4> h2 = ResourceDescriptorHeap[R_H2];
    // torch upsample_trilinear3d evaluation order
    float3 up =
        t0l * (h0l * (w0l * coarse_at(h2, w1, h1, t1) + w1l * coarse_at(h2, w1 + w1p, h1, t1)) +
               h1l * (w0l * coarse_at(h2, w1, h1 + h1p, t1) + w1l * coarse_at(h2, w1 + w1p, h1 + h1p, t1))) +
        t1l * (h0l * (w0l * coarse_at(h2, w1, h1, t1 + t1p) + w1l * coarse_at(h2, w1 + w1p, h1, t1 + t1p)) +
               h1l * (w0l * coarse_at(h2, w1, h1 + h1p, t1 + t1p) + w1l * coarse_at(h2, w1 + w1p, h1 + h1p, t1 + t1p)));
    RWStructuredBuffer<float4> g3 = ResourceDescriptorHeap[R_G3];
    float3 glow = g3[j].xyz + P.glow_coarse * up;
    RWTexture3D<float4> gt = ResourceDescriptorHeap[R_GLOW_UAV];
    gt[g] = float4(half_round(glow.x), half_round(glow.y), half_round(glow.z), 0.0f);
}
