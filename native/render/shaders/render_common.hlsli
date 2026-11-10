// Shared HLSL of the native volume renderer (port of vfx/render3d/kernels.py). OWNER: native render engineer.
// Root signature (render.cpp):
//   b0 = 8 root constants (per dispatch: job counts), b1 = root CBV with the per-draw parameter block (RParams),
//   s0 = static sampler, trilinear, clamp (= wp.Texture3D LINEAR + CLAMP),
//   + CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED: every resource via ResourceDescriptorHeap[R_*].
#include "rslots.h"

struct RootConsts { uint k0, k1, k2, k3, k4, k5, k6, k7; };
ConstantBuffer<RootConsts> RC : register(b0);

// Mirrors struct RParams in render.cpp (every row 16 bytes).
struct RParams {
    float k_abs, k_shadow, k_solid, k_amb;
    float k_emit, emit_min, emit_scale, soot;
    float fire_light, phase_g, exposure, bloom;
    float bloom_thresh, t_lo, t_hi, d_eps;
    float floor_op, ms_a, ms_b, ms_c;
    float rim, contrast, steps, glow_coarse;
    float4 light_dir, key_col, amb_col, smoke_albedo, bg_top, bg_bot, floor_col, grid_col, obst_col;
    float4 cam[4];          // camera_basis(): pos, forward, right, up, tan(fov/2), aspect, 0, 0
    float4 cam_prev[4];
    float out_w, out_h, sim_dt, solid_res;
    uint  N, L, mw, mh;
    uint  ow, oh, bw, bh;
    uint  bw2, bh2, has_solid, has_vel;
    uint  in_vd, in_tf, in_solid, aux_on;
    uint  dither_mode, pad0, pad1, pad2;
    float4 gs[2];           // solid-mask blur, sigma 1.0, 7 taps
    float4 g3[2];           // glow blur, sigma 1.2, 7 taps
    float4 g2[3];           // bloom blur, sigma 2.0, 11 taps
};
ConstantBuffer<RParams> P : register(b1);
SamplerState s_lin : register(s0);

float cam_f(uint i) { return P.cam[i >> 2][i & 3]; }
float camp_f(uint i) { return P.cam_prev[i >> 2][i & 3]; }
float w_gs(int i) { return P.gs[i >> 2][i & 3]; }
float w_g3(int i) { return P.g3[i >> 2][i & 3]; }
float w_g2(int i) { return P.g2[i >> 2][i & 3]; }

// ---------------------------------------------------------------- helpers (kernels.py)
// wp.frac(x) = x - trunc(x)
float wfrac(float x) { return x - trunc(x); }

// wp.normalize / wp.length: a / sqrt(dot(a, a))
float wlength(float3 a) { return sqrt(dot(a, a)); }
float3 wnormalize(float3 a) {
    float l = wlength(a);
    return l > 0.0f ? a / l : float3(0.0f, 0.0f, 0.0f);
}

float smoothstep_w(float e0, float e1, float x) {
    float t = clamp((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

float2 box_hit(float3 o, float3 invd, float3 lo, float3 hi) {
    float3 t0 = (lo - o) * invd;
    float3 t1 = (hi - o) * invd;
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);
    return float2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

float safe_inv1(float d) {
    const float e = 1.0e-8f;
    float x = abs(d) < e ? (d < 0.0f ? -e : e) : d;
    return 1.0f / x;
}
float3 safe_inv(float3 d) { return float3(safe_inv1(d.x), safe_inv1(d.y), safe_inv1(d.z)); }

float hg(float cos_t, float g) {
    float d = 1.0f + g * g - 2.0f * g * cos_t;
    return (1.0f - g * g) / (d * sqrt(d));
}

float3 lut_lookup(float T) {
    StructuredBuffer<float4> lut = ResourceDescriptorHeap[R_LUT];
    const int n = R_LUT_N;
    float f = clamp((T - P.t_lo) / (P.t_hi - P.t_lo), 0.0f, 1.0f) * float(n - 1);
    int i0 = min(int(f), n - 2);
    float t = f - float(i0);
    return lut[i0].xyz * (1.0f - t) + lut[i0 + 1].xyz * t;
}

float3 emission(float d, float T) {
    float e = P.emit_min + (1.0f - P.emit_min) * (1.0f - exp(-P.k_emit * d));
    return lut_lookup(T) * (e * P.emit_scale);
}

// interleaved gradient noise in [0,1)
float ign(float x, float y) {
    float f = 52.9829189f * wfrac(0.06711056f * x + 0.00583715f * y);
    return wfrac(f);
}

float half_round(float x) { return f16tof32(f32tof16(x)); }

// flat index helpers (x fastest)
uint3 unflat3(uint i, uint n) { return uint3(i % n, (i / n) % n, i / (n * n)); }
uint flat3(int3 c, uint n) { return (uint(c.z) * n + uint(c.y)) * n + uint(c.x); }
