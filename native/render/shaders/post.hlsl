// View post-processing: port of renderer.py _view after the march (bloom with torch pool/blur/interpolate,
// kernels.composite, kernels.aux_resolve). OWNER: native render engineer.
//   r_bloom1  (16x16 tiles of the bloom level): adaptive_avg_pool2d(relu(hdr - thresh)) -> blur x -> blur y = b1
//             (written to R_B1) and avg_pool2d(b1, 4, ceil_mode) of the tile's 4x4 blocks (written to R_P)
//   r_bloom2  x-blur of P -> R_U2
//   r_bloom3  bloom = b1 + 1.5 * bilinear(y-blur of U2)  -> R_BLOOM
//   r_composite (output res): Catmull-Rom upsample, background + vignette, bloom, exposure, contrast, ACES,
//             sRGB, dither -> R8G8B8A8; plus, when P.aux_on, aux_resolve -> alpha / depth / motion textures.
#include "render_common.hlsli"

#ifndef DBL
#define DBL 1
#endif

static const int BR = 5;           // bloom blur radius: _gauss1d(2.0) -> 11 taps
#define TILE 16
#define APR (TILE + 2 * BR)

groupshared float3 sh_pool[APR][APR];
groupshared float3 sh_xb[APR][TILE];
groupshared float3 sh_b1[TILE][TILE];

float3 bloom_pooled(RWStructuredBuffer<float4> hdr, int py, int px) {
    const int mh = int(P.mh), mw = int(P.mw), bh = int(P.bh), bw = int(P.bw);
    int hs = (py * mh) / bh, he = ((py + 1) * mh + bh - 1) / bh;
    int ws = (px * mw) / bw, we = ((px + 1) * mw + bw - 1) / bw;
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    for (int y = hs; y < he; ++y)
        for (int x = ws; x < we; ++x)
            sum += max(hdr[y * mw + x].xyz - P.bloom_thresh, 0.0f);
    return sum / float(he - hs) / float(we - ws);
}

[numthreads(TILE, TILE, 1)]
void r_bloom1(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    const int bh = int(P.bh), bw = int(P.bw);
    const int by0 = int(gid.y) * TILE, bx0 = int(gid.x) * TILE;
    const int tid = int(gtid.y) * TILE + int(gtid.x);
    RWStructuredBuffer<float4> hdr = ResourceDescriptorHeap[R_HDR];
    for (int k = tid; k < APR * APR; k += TILE * TILE) {
        int ly = k / APR, lx = k % APR;
        int py = clamp(by0 - BR + ly, 0, bh - 1), px = clamp(bx0 - BR + lx, 0, bw - 1);   // replicate padding
        sh_pool[ly][lx] = bloom_pooled(hdr, py, px);
    }
    GroupMemoryBarrierWithGroupSync();
    for (int k2 = tid; k2 < APR * TILE; k2 += TILE * TILE) {
        int ly = k2 / TILE, lx = k2 % TILE;
        float3 acc = float3(0.0f, 0.0f, 0.0f);
        [unroll] for (int j = 0; j < 2 * BR + 1; ++j) acc += sh_pool[ly][lx + j] * w_g2(j);
        sh_xb[ly][lx] = acc;
    }
    GroupMemoryBarrierWithGroupSync();
    {
        int ty = int(gtid.y), tx = int(gtid.x);
        float3 acc = float3(0.0f, 0.0f, 0.0f);
        [unroll] for (int j = 0; j < 2 * BR + 1; ++j) acc += sh_xb[ty + j][tx] * w_g2(j);
        sh_b1[ty][tx] = acc;
        int y = by0 + ty, x = bx0 + tx;
        if (y < bh && x < bw) {
            RWStructuredBuffer<float4> b1 = ResourceDescriptorHeap[R_B1];
            b1[y * bw + x] = float4(acc, 0.0f);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    // avg_pool2d(b1, 4, ceil_mode=True): windows clipped at the edge, divisor = clipped window size
    if (tid < (TILE / 4) * (TILE / 4)) {
        int qy = tid / (TILE / 4), qx = tid % (TILE / 4);
        int gy = by0 / 4 + qy, gx = bx0 / 4 + qx;
        if (gy < int(P.bh2) && gx < int(P.bw2)) {
            int hs = gy * 4, he = min(hs + 4, bh), ws = gx * 4, we = min(ws + 4, bw);
            float3 sum = float3(0.0f, 0.0f, 0.0f);
            for (int y = hs; y < he; ++y)
                for (int x = ws; x < we; ++x) sum += sh_b1[y - by0][x - bx0];
            RWStructuredBuffer<float4> pp = ResourceDescriptorHeap[R_P];
            pp[gy * int(P.bw2) + gx] = float4(sum / float((he - hs) * (we - ws)), 0.0f);
        }
    }
}

[numthreads(64, 1, 1)]
void r_bloom2(uint3 id : SV_DispatchThreadID) {
    const int bh2 = int(P.bh2), bw2 = int(P.bw2);
    int i = int(id.y * RC.k7 + id.x);
    if (i >= bh2 * bw2) return;
    int y = i / bw2, x = i % bw2;
    RWStructuredBuffer<float4> pp = ResourceDescriptorHeap[R_P];
    RWStructuredBuffer<float4> u2 = ResourceDescriptorHeap[R_U2];
    float3 acc = float3(0.0f, 0.0f, 0.0f);
    [unroll] for (int j = 0; j < 2 * BR + 1; ++j) acc += pp[y * bw2 + clamp(x + j - BR, 0, bw2 - 1)].xyz * w_g2(j);
    u2[i] = float4(acc, 0.0f);
}

float3 u3_at(RWStructuredBuffer<float4> u2, int y, int x) {
    const int bh2 = int(P.bh2), bw2 = int(P.bw2);
    float3 acc = float3(0.0f, 0.0f, 0.0f);
    [unroll] for (int j = 0; j < 2 * BR + 1; ++j) acc += u2[clamp(y + j - BR, 0, bh2 - 1) * bw2 + x].xyz * w_g2(j);
    return acc;
}

[numthreads(8, 8, 1)]
void r_bloom3(uint3 id : SV_DispatchThreadID) {
    const int bh = int(P.bh), bw = int(P.bw), bh2 = int(P.bh2), bw2 = int(P.bw2);
    int x = int(id.x), y = int(id.y);
    if (x >= bw || y >= bh) return;
    RWStructuredBuffer<float4> u2 = ResourceDescriptorHeap[R_U2];
    RWStructuredBuffer<float4> b1 = ResourceDescriptorHeap[R_B1];
    RWStructuredBuffer<float4> bl = ResourceDescriptorHeap[R_BLOOM];
    // F.interpolate(b2, size=b1, mode="bilinear", align_corners=False) (torch upsample_bilinear2d)
    float rh = float(bh2) / float(bh), rw = float(bw2) / float(bw);
    float h1r = max(rh * (float(y) + 0.5f) - 0.5f, 0.0f);
    int h1 = int(h1r);
    int h1p = h1 < bh2 - 1 ? 1 : 0;
    float h1l = h1r - float(h1), h0l = 1.0f - h1l;
    float w1r = max(rw * (float(x) + 0.5f) - 0.5f, 0.0f);
    int w1 = int(w1r);
    int w1p = w1 < bw2 - 1 ? 1 : 0;
    float w1l = w1r - float(w1), w0l = 1.0f - w1l;
    float3 up = h0l * (w0l * u3_at(u2, h1, w1) + w1l * u3_at(u2, h1, w1 + w1p)) +
                h1l * (w0l * u3_at(u2, h1 + h1p, w1) + w1l * u3_at(u2, h1 + h1p, w1 + w1p));
    int i = y * bw + x;
    bl[i] = float4(b1[i].xyz + 1.5f * up, 0.0f);
}

// ---------------------------------------------------------------- composite
float4 cr_w(float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    return float4(-0.5f * t3 + t2 - 0.5f * t, 1.5f * t3 - 2.5f * t2 + 1.0f, -1.5f * t3 + 2.0f * t2 + 0.5f * t,
                  0.5f * t3 - 0.5f * t2);
}

// Catmull-Rom upsample of the march buffer, clamped to the 2x2 bilinear footprint's range (no ringing)
float4 sample_v4_cr(RWStructuredBuffer<float4> a, int H, int W, float u, float v) {
    float fx = u * float(W) - 0.5f;
    float fy = v * float(H) - 0.5f;
    float x0f = floor(fx);
    float y0f = floor(fy);
    float4 wx = cr_w(fx - x0f);
    float4 wy = cr_w(fy - y0f);
    int xi = int(x0f);
    int yi = int(y0f);
    float4 acc = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 lo = float4(1.0e9f, 1.0e9f, 1.0e9f, 1.0e9f);
    float4 hi = float4(-1.0e9f, -1.0e9f, -1.0e9f, -1.0e9f);
    [unroll] for (int j = 0; j < 4; ++j) {
        int yy = clamp(yi - 1 + j, 0, H - 1);
        float4 row = float4(0.0f, 0.0f, 0.0f, 0.0f);
        [unroll] for (int i = 0; i < 4; ++i) {
            int xx = clamp(xi - 1 + i, 0, W - 1);
            float4 s = a[yy * W + xx];
            row = row + s * wx[i];
            if ((i == 1 || i == 2) && (j == 1 || j == 2)) {
                lo = min(lo, s);
                hi = max(hi, s);
            }
        }
        acc = acc + row * wy[j];
    }
    return clamp(acc, lo, hi);
}

float4 sample_v4(RWStructuredBuffer<float4> a, int H, int W, float u, float v) {
    float fx = u * float(W) - 0.5f;
    float fy = v * float(H) - 0.5f;
    float x0f = floor(fx);
    float y0f = floor(fy);
    float tx = fx - x0f;
    float ty = fy - y0f;
    int x0 = clamp(int(x0f), 0, W - 1);
    int y0 = clamp(int(y0f), 0, H - 1);
    int x1 = clamp(int(x0f) + 1, 0, W - 1);
    int y1 = clamp(int(y0f) + 1, 0, H - 1);
    float4 a0 = a[y0 * W + x0] * (1.0f - tx) + a[y0 * W + x1] * tx;
    float4 a1 = a[y1 * W + x0] * (1.0f - tx) + a[y1 * W + x1] * tx;
    return a0 * (1.0f - ty) + a1 * ty;
}

float aces(float x) {
    return clamp((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f), 0.0f, 1.0f);
}

uint to_srgb8(float x, float dither) {
    float s;
    if (x <= 0.0031308f) s = 12.92f * x;
    else s = 1.055f * pow(x, 1.0f / 2.4f) - 0.055f;
    return uint(clamp(int(s * 255.0f + 0.5f + dither), 0, 255));
}

// Python: sin(float(r) * 12.9898 + float(c) * 78.233) * 43758.5453 with CUDA sinf (accurate, Payne-Hanek range
// reduction for large arguments) and nvcc FMA contraction (Warp fuse_fp=True). The argument reaches ~2e5 rad, where
// 1 ulp of the argument already decorrelates the dither, so the argument's rounding and an accurate sin both matter:
// the argument is formed as one fused multiply-add (emulated exactly in double; which product nvcc fused is not
// known: P.dither_mode 0 = fma(r, 12.9898, c * 78.233), 1 = fma(c, 78.233, r * 12.9898), 2 = no fusion), and sin
// is range-reduced and evaluated in double. DBL=0 (device without double support): plain float, dither differs.
float dither_arg(float fr, float fc) {
#if DBL
    if (P.dither_mode == 0u) {
        float pc = fc * 78.233f;
        return float((double)fr * (double)12.9898f + (double)pc);
    }
    if (P.dither_mode == 1u) {
        float pr = fr * 12.9898f;
        return float((double)fc * (double)78.233f + (double)pr);
    }
#endif
    return fr * 12.9898f + fc * 78.233f;
}

float sin_accurate(float xf) {
#if DBL
    double x = (double)xf;                                      // xf >= 0 here (pixel indices)
    int k = int(x * 0.15915494309189535L + 0.5L);               // round(x / 2pi), x < 2^31
    double r = x - (double)k * 6.283185307179586L;              // |r| <= pi (+ ~1e-11)
    double r2 = r * r;
    // Taylor series to x^25 (truncation < 3e-15 on [-pi, pi])
    // sin r = r * sum_n (-1)^n r^(2n) / (2n+1)!, Horner in r^2 from n = 12
    double s = 6.4469502843844734e-26L;                         // +1/25!
    s = s * r2 - 3.8681701706306841e-23L;                       // -1/23!
    s = s * r2 + 1.9572941063391263e-20L;                       // +1/21!
    s = s * r2 - 8.2206352466243295e-18L;                       // -1/19!
    s = s * r2 + 2.8114572543455206e-15L;                       // +1/17!
    s = s * r2 - 7.6471637318198164e-13L;                       // -1/15!
    s = s * r2 + 1.6059043836821613e-10L;                       // +1/13!
    s = s * r2 - 2.5052108385441720e-08L;                       // -1/11!
    s = s * r2 + 2.7557319223985893e-06L;                       // +1/9!
    s = s * r2 - 1.9841269841269841e-04L;                       // -1/7!
    s = s * r2 + 8.3333333333333333e-03L;                       // +1/5!
    s = s * r2 - 1.6666666666666667e-01L;                       // -1/3!
    s = s * r2 + 1.0L;
    return float(r * s);
#else
    return sin(xf);
#endif
}

[numthreads(8, 8, 1)]
void r_composite(uint3 id : SV_DispatchThreadID) {
    const int Wo = int(P.ow), Ho = int(P.oh);
    const int c = int(id.x), r = int(id.y);
    if (c >= Wo || r >= Ho) return;
    const int mh = int(P.mh), mw = int(P.mw);
    float u = (float(c) + 0.5f) / float(Wo);
    float v = (float(r) + 0.5f) / float(Ho);
    RWStructuredBuffer<float4> hdr = ResourceDescriptorHeap[R_HDR];
    RWStructuredBuffer<float4> bl = ResourceDescriptorHeap[R_BLOOM];
    float4 px;
    if (mh == Ho && mw == Wo) px = hdr[r * mw + c];
    else px = sample_v4_cr(hdr, mh, mw, u, v);
    float vy = 1.0f - v;
    float vig = 1.0f - 0.6f * ((u - 0.5f) * (u - 0.5f) + (vy - 0.6f) * (vy - 0.6f));
    float3 bg = (P.bg_bot.xyz * (1.0f - vy) + P.bg_top.xyz * vy) * vig;
    float3 col = float3(px.x, px.y, px.z) + bg * px.w + sample_v4(bl, int(P.bh), int(P.bw), u, v).xyz * P.bloom;
    col = col * P.exposure;
    if (P.contrast != 1.0f) {
        col = float3(0.18f * pow(max(col.x, 0.0f) / 0.18f, P.contrast), 0.18f * pow(max(col.y, 0.0f) / 0.18f, P.contrast),
                     0.18f * pow(max(col.z, 0.0f) / 0.18f, P.contrast));
    }
    float hh = sin_accurate(dither_arg(float(r), float(c))) * 43758.5453f;
    float dither = (hh - floor(hh)) - 0.5f;
    uint3 o8 = uint3(to_srgb8(aces(col.x), dither), to_srgb8(aces(col.y), dither), to_srgb8(aces(col.z), dither));
    RWTexture2D<unorm float4> outc = ResourceDescriptorHeap[R_COLOR_UAV];
    outc[uint2(c, r)] = float4(float(o8.x) / 255.0f, float(o8.y) / 255.0f, float(o8.z) / 255.0f, 1.0f);

    if (P.aux_on != 0u) {
        // aux_resolve: bilinear upsample of the premultiplied march-res aux, then un-premultiply (a_min 1e-3)
        RWStructuredBuffer<float4> aux = ResourceDescriptorHeap[R_AUXHDR];
        float4 x;
        if (mh == Ho && mw == Wo) x = aux[r * mw + c];
        else x = sample_v4(aux, mh, mw, u, v);
        float al = x.x;
        float dep = 0.0f, mx = 0.0f, my = 0.0f;
        if (al >= 1.0e-3f) {
            dep = x.y / al;
            mx = x.z / al;
            my = x.w / al;
        }
        RWTexture2D<float> oa = ResourceDescriptorHeap[R_ALPHA_UAV];
        RWTexture2D<float> od = ResourceDescriptorHeap[R_DEPTH_UAV];
        RWTexture2D<float2> om = ResourceDescriptorHeap[R_MOTION_UAV];
        oa[uint2(c, r)] = half_round(al);
        od[uint2(c, r)] = half_round(dep);
        om[uint2(c, r)] = float2(half_round(mx), half_round(my));
    }
}
