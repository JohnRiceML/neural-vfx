// Route B pre / post passes (vfx/runtime3d/screen_detail.py ScreenDetail._body + the output constraints at the end of
// vfx/model3d/screennet.py ScreenNet.forward), bindless (SM 6.6, compiled with -enable-16bit-types).
// Reproduces PyTorch's arithmetic step by step, including every fp16 rounding (round-to-nearest-even, done in
// integer arithmetic so it does not depend on the driver's float->half conversion).
//
// Network tensors are NCHW fp16 in raw buffers: element (c, y, x) at index c*ph*pw + y*pw + x.
// pw is even (crop sizes are multiples of 64), so a uint holds pixels x, x+1 of one row.
//
// Passes, DirectML backend:  pre_dml -> [DirectML: trunk + HR head = hr2 output "draw"] -> lowfreq -> fin -> post
//   pre_dml  crop, u8->fp16, pool, aux normalisation; writes X0 = PixelUnshuffle(cat(rgb, norm_aux)) for the trunk,
//            rgb (HR head input) and the pooled alpha (alpha gate)
//   lowfreq  grey_lowfreq: avg_pool(k) of draw
//   fin      d - lo + mean_c(lo) (lo = bilinear x k of the pooled), then the alpha gate -> delta
//   post     bilinear upsample of delta, add, clamp, round, uint8; outside the region: copy of the frame
// Passes, ONNX Runtime backend: pre (rgb + aux, the ONNX model's inputs) -> [ORT: whole ScreenNet.forward] -> post

cbuffer K : register(b0) {
    uint color_srv;   // Texture2D<float4>  R8G8B8A8_UNORM, frame
    uint alpha_srv;   // Texture2D<float>   R16_FLOAT (null descriptor -> 0)
    uint depth_srv;   // Texture2D<float>   R16_FLOAT
    uint motion_srv;  // Texture2D<float2>  R16G16_FLOAT
    uint buf_a;       // ORT: rgb (1,3,ph,pw)      DML: X0 (1, 7 r^2, ph/r, pw/r)
    uint buf_b;       // ORT: aux (1,4,ph,pw)      DML: rgb (1,3,ph,pw)
    uint delta_uav;   // (1,3,ph,pw) fp16 final delta rgb (ORT: the net output; DML: written by fin)
    uint out_uav;     // RWTexture2D<unorm float4> R8G8B8A8_UNORM
    int  y0, x0;      // crop origin in the frame
    int  vh, vw;      // valid crop size (inside the frame); crop rows/cols >= vh/vw are zero padding
    uint ph, pw;      // processed (network) size = crop size * process_scale
    uint fw, fh;      // frame size
    uint alpha_buf;   // DML: pooled alpha (1,1,ph,pw) fp16 (aux channel 0 as the net sees it)
    uint draw_buf;    // DML: hr2 output (1,3,ph,pw) fp16
    uint coarse_buf;  // DML: (3, ph/k, pw/k) float32 holding fp16 values
    uint grey_k;      // grey_lowfreq k (0 = off)
    float gate;       // alpha_gate g (0 = off)
    float clip;       // input_clip
    float depth_scale, motion_scale;
    uint r;           // trunk stride
    uint pad0, pad1, pad2;
};

// fp32 -> fp16 bits, round to nearest even (normal + subnormal; overflow -> inf)
uint f2h(float f) {
    uint x = asuint(f);
    uint s = (x >> 16) & 0x8000u;
    uint a = x & 0x7fffffffu;
    uint h;
    if (a >= 0x47800000u) {
        h = a > 0x7f800000u ? 0x7e00u : 0x7c00u;
    } else if (a >= 0x38800000u) {
        h = (a - 0x38000000u + 0x0fffu + ((a >> 13) & 1u)) >> 13;
    } else {
        h = (uint)round(asfloat(a) * 16777216.0f);   // round() = round_ne in DXIL
    }
    return s | h;
}
float hr(float f) { return f16tof32(f2h(f)); }      // round to the nearest fp16 value
float3 hr3(float3 v) { return float3(hr(v.x), hr(v.y), hr(v.z)); }

// uint8 frame value as Python sees it: u8 -> half -> * (1/255) in fp32 -> half
float3 rgb_half(float4 c) {
    float3 u = round(c.rgb * 255.0f);               // exact integer whatever the UNORM->float conversion error
    return hr3(u * (1.0f / 255.0f));
}

void store2(RWByteAddressBuffer b, uint index, float lo, float hi) {   // index even
    b.Store(index * 2u, f2h(lo) | (f2h(hi) << 16));
}

float load_h(RWByteAddressBuffer b, uint index) {
    uint w = b.Load((index & ~1u) * 2u);
    return f16tof32((index & 1u) ? (w >> 16) : (w & 0xffffu));
}

// ------------------------------------------------------------------ pooled inputs of one processed pixel
// v[0..2] rgb (fp16 values), v[3..6] = aux as ScreenDetail._body builds it BEFORE .half(): alpha mean,
// alpha-weighted depth, alpha-weighted motion x process_scale (K = 2); raw fp16 aux (K = 1).
void pooled(uint px, uint py, uint K, out float v[7]) {
    Texture2D<float4> col = ResourceDescriptorHeap[color_srv];
    Texture2D<float>  alp = ResourceDescriptorHeap[alpha_srv];
    Texture2D<float>  dep = ResourceDescriptorHeap[depth_srv];
    Texture2D<float2> mot = ResourceDescriptorHeap[motion_srv];
    if (K == 1) {
        int cy = int(py), cx = int(px);
        float3 rr = 0;
        float a = 0, d = 0;
        float2 m = 0;
        if (cy < vh && cx < vw) {
            int3 p = int3(x0 + cx, y0 + cy, 0);
            rr = rgb_half(col.Load(p));
            a = alp.Load(p);
            d = dep.Load(p);
            m = mot.Load(p);
        }
        v[0] = rr.r; v[1] = rr.g; v[2] = rr.b; v[3] = a; v[4] = d; v[5] = m.x; v[6] = m.y;
        return;
    }
    float3 s = 0;
    float sa = 0, sd = 0, smx = 0, smy = 0;
    [unroll] for (int dy = 0; dy < 2; ++dy) {          // avg_pool2d sums rows, then columns
        [unroll] for (int dx = 0; dx < 2; ++dx) {
            int cy = int(py) * 2 + dy, cx = int(px) * 2 + dx;
            float3 rr = 0;
            float a = 0, d = 0;
            float2 m = 0;
            if (cy < vh && cx < vw) {
                int3 p = int3(x0 + cx, y0 + cy, 0);
                rr = rgb_half(col.Load(p));
                a = alp.Load(p);
                d = dep.Load(p);
                m = mot.Load(p);
            }
            float al = clamp(a, 0.0f, 1.0f);
            s += rr;
            sa += al;
            sd += d * al;
            smx += m.x * al;
            smy += m.y * al;
        }
    }
    float3 x = hr3(s / 4.0f);
    float w = sa / 4.0f;
    float wc = max(w, 1e-4f);
    float keep = w > 1e-4f ? 1.0f : 0.0f;
    v[0] = x.r; v[1] = x.g; v[2] = x.b;
    v[3] = w;
    v[4] = ((sd / 4.0f) / wc) * keep;
    v[5] = (((smx / 4.0f) / wc) * keep) * 0.5f;      // motion x process_scale
    v[6] = (((smy / 4.0f) / wc) * keep) * 0.5f;
}

// ------------------------------------------------------------------ pre, ONNX Runtime backend: (rgb, aux) NCHW
void pre_ort(uint3 id, uint K) {
    uint px0 = id.x * 2u, py = id.y;
    if (px0 >= pw || py >= ph) return;
    RWByteAddressBuffer rgbb = ResourceDescriptorHeap[buf_a];
    RWByteAddressBuffer auxb = ResourceDescriptorHeap[buf_b];
    float o0[7], o1[7];
    pooled(px0, py, K, o0);
    pooled(px0 + 1u, py, K, o1);
    uint plane = ph * pw, e = py * pw + px0;
    [unroll] for (uint c = 0; c < 3; ++c) store2(rgbb, c * plane + e, o0[c], o1[c]);
    [unroll] for (uint c2 = 3; c2 < 7; ++c2) store2(auxb, (c2 - 3) * plane + e, o0[c2], o1[c2]);
}
[numthreads(8, 8, 1)] void pre_s2(uint3 id : SV_DispatchThreadID) { pre_ort(id, 2); }
[numthreads(8, 8, 1)] void pre_s1(uint3 id : SV_DispatchThreadID) { pre_ort(id, 1); }

// ------------------------------------------------------------------ pre, DirectML backend
void pre_dml(uint3 id, uint K) {
    uint px0 = id.x * 2u, py = id.y;
    if (px0 >= pw || py >= ph) return;
    RWByteAddressBuffer x0b = ResourceDescriptorHeap[buf_a];
    RWByteAddressBuffer rgbb = ResourceDescriptorHeap[buf_b];
    RWByteAddressBuffer alb = ResourceDescriptorHeap[alpha_buf];
    float o[2][7];
    float t[7];
    pooled(px0, py, K, t);
    [unroll] for (uint c0 = 0; c0 < 7; ++c0) o[0][c0] = t[c0];
    pooled(px0 + 1u, py, K, t);
    [unroll] for (uint c1 = 0; c1 < 7; ++c1) o[1][c1] = t[c1];
    uint plane = ph * pw, e = py * pw + px0;
    [unroll] for (uint c = 0; c < 3; ++c) store2(rgbb, c * plane + e, o[0][c], o[1][c]);
    float a0 = hr(o[0][3]), a1 = hr(o[1][3]);         // xa = cat(...).half()
    store2(alb, e, a0, a1);
    // ScreenNet: x = cat(rgb, (aux * aux_scale).clamp(-clip, clip)); X0 = PixelUnshuffle(r)(x)
    uint th = ph / r, tw = pw / r, rr = r * r;
    const float sc[4] = {1.0f, depth_scale, motion_scale, motion_scale};
    [unroll] for (uint j = 0; j < 2; ++j) {
        uint px = px0 + j;
        uint sub = (py % r) * r + (px % r);
        uint pos = (py / r) * tw + (px / r);
        [unroll] for (uint c = 0; c < 7; ++c) {
            float v = o[j][c];
            if (c >= 3) v = clamp(hr(hr(v) * sc[c - 3]), -clip, clip);
            uint idx = (c * rr + sub) * th * tw + pos;
            x0b.Store<uint16_t>(idx * 2u, uint16_t(f2h(v)));
        }
    }
}
[numthreads(8, 8, 1)] void pre_dml_s2(uint3 id : SV_DispatchThreadID) { pre_dml(id, 2); }
[numthreads(8, 8, 1)] void pre_dml_s1(uint3 id : SV_DispatchThreadID) { pre_dml(id, 1); }

// ------------------------------------------------------------------ grey_lowfreq: avg_pool2d(d, k)
[numthreads(8, 8, 1)]
void lowfreq(uint3 id : SV_DispatchThreadID) {
    uint k = grey_k, cph = ph / k, cpw = pw / k;
    if (id.x >= cpw || id.y >= cph) return;
    RWByteAddressBuffer db = ResourceDescriptorHeap[draw_buf];
    RWByteAddressBuffer cb = ResourceDescriptorHeap[coarse_buf];
    uint plane = ph * pw;
    for (uint ch = 0; ch < 3; ++ch) {
        float s = 0;
        for (uint i = 0; i < k; ++i)
            for (uint j = 0; j < k; ++j)
                s += load_h(db, ch * plane + (id.y * k + i) * pw + id.x * k + j);
        cb.Store((ch * cph * cpw + id.y * cpw + id.x) * 4u, asuint(hr(s / float(k * k))));
    }
}

// ------------------------------------------------------------------ d - lo + mean_c(lo), alpha gate
float coarse(RWByteAddressBuffer cb, uint ch, uint y, uint x, uint cph, uint cpw) {
    return asfloat(cb.Load((ch * cph * cpw + y * cpw + x) * 4u));
}

float3 fin_px(uint px, uint py) {
    RWByteAddressBuffer db = ResourceDescriptorHeap[draw_buf];
    uint plane = ph * pw, e = py * pw + px;
    float3 d = float3(load_h(db, e), load_h(db, plane + e), load_h(db, 2 * plane + e));
    if (grey_k) {
        RWByteAddressBuffer cb = ResourceDescriptorHeap[coarse_buf];
        uint k = grey_k, cph = ph / k, cpw = pw / k;
        float sc = 1.0f / float(k);                  // upsample_bilinear2d with scale_factor k: src scale 1/k
        float sy = max(sc * (float(py) + 0.5f) - 0.5f, 0.0f);
        float sx = max(sc * (float(px) + 0.5f) - 0.5f, 0.0f);
        uint iy = uint(sy), ix = uint(sx);
        uint y1 = iy + (iy < cph - 1 ? 1u : 0u), x1 = ix + (ix < cpw - 1 ? 1u : 0u);
        float ly1 = sy - float(iy), lx1 = sx - float(ix);
        float ly0 = 1.0f - ly1, lx0 = 1.0f - lx1;
        float3 lo;
        [unroll] for (uint ch = 0; ch < 3; ++ch)
            lo[ch] = hr(ly0 * (lx0 * coarse(cb, ch, iy, ix, cph, cpw) + lx1 * coarse(cb, ch, iy, x1, cph, cpw)) +
                        ly1 * (lx0 * coarse(cb, ch, y1, ix, cph, cpw) + lx1 * coarse(cb, ch, y1, x1, cph, cpw)));
        float mean = hr((lo.x + lo.y + lo.z) * (1.0f / 3.0f));     // mean = sum * (1/N) in fp32
        d = float3(hr(hr(d.x - lo.x) + mean), hr(hr(d.y - lo.y) + mean), hr(hr(d.z - lo.z) + mean));
    }
    if (gate > 0.0f) {
        RWByteAddressBuffer alb = ResourceDescriptorHeap[alpha_buf];
        float lim = hr(clamp(load_h(alb, e), 0.0f, 1.0f) * gate);
        d = min(max(d, -lim), lim);
    }
    return d;
}

[numthreads(8, 8, 1)]
void fin(uint3 id : SV_DispatchThreadID) {
    uint px0 = id.x * 2u, py = id.y;
    if (px0 >= pw || py >= ph) return;
    RWByteAddressBuffer ob = ResourceDescriptorHeap[delta_uav];
    float3 a = fin_px(px0, py), b = fin_px(px0 + 1u, py);
    uint plane = ph * pw, e = py * pw + px0;
    [unroll] for (uint ch = 0; ch < 3; ++ch) store2(ob, ch * plane + e, a[ch], b[ch]);
}

// ------------------------------------------------------------------ post
// y = ((rgb + delta).clamp(0, 1) * 255 + 0.5).to(uint8), every op in fp16 like the PyTorch graph
float finish(float rgb_h, float delta_h) {
    float s = clamp(hr(rgb_h + delta_h), 0.0f, 1.0f);
    float t = hr(hr(s * 255.0f) + 0.5f);
    return floor(t);                                 // .to(uint8) truncates (t >= 0.5)
}

void post_common(uint2 p, bool up2) {
    Texture2D<float4> col = ResourceDescriptorHeap[color_srv];
    RWTexture2D<unorm float4> outt = ResourceDescriptorHeap[out_uav];
    float4 c = col.Load(int3(p, 0));
    int cx = int(p.x) - x0, cy = int(p.y) - y0;
    if (cx < 0 || cy < 0 || cx >= vw || cy >= vh) {
        outt[p] = c;
        return;
    }
    RWByteAddressBuffer db = ResourceDescriptorHeap[delta_uav];
    float3 rgb = rgb_half(c);
    float3 dl;
    uint plane = ph * pw;
    if (up2) {
        // upsample_bilinear2d, align_corners=False, scale = in/out = 0.5: src = 0.5*(dst+0.5)-0.5, clamped at 0
        float sy = max(0.5f * (float(cy) + 0.5f) - 0.5f, 0.0f);
        float sx = max(0.5f * (float(cx) + 0.5f) - 0.5f, 0.0f);
        uint iy = uint(sy), ix = uint(sx);
        uint py1 = iy < ph - 1 ? 1u : 0u, px1 = ix < pw - 1 ? 1u : 0u;
        float ly1 = sy - float(iy), lx1 = sx - float(ix);
        float ly0 = 1.0f - ly1, lx0 = 1.0f - lx1;
        uint e00 = iy * pw + ix;
        uint e10 = (iy + py1) * pw + ix;
        [unroll] for (uint ch = 0; ch < 3; ++ch) {
            uint b = ch * plane;
            float v = ly0 * (lx0 * load_h(db, b + e00) + lx1 * load_h(db, b + e00 + px1)) +
                      ly1 * (lx0 * load_h(db, b + e10) + lx1 * load_h(db, b + e10 + px1));
            dl[ch] = hr(v);
        }
    } else {
        uint e = uint(cy) * pw + uint(cx);
        [unroll] for (uint ch = 0; ch < 3; ++ch) dl[ch] = load_h(db, ch * plane + e);
    }
    float3 o = float3(finish(rgb.r, dl.r), finish(rgb.g, dl.g), finish(rgb.b, dl.b));
    outt[p] = float4(o * (1.0f / 255.0f), c.a);
}

[numthreads(8, 8, 1)]
void post_s2(uint3 id : SV_DispatchThreadID) {
    if (id.x >= fw || id.y >= fh) return;
    post_common(id.xy, true);
}

[numthreads(8, 8, 1)]
void post_s1(uint3 id : SV_DispatchThreadID) {
    if (id.x >= fw || id.y >= fh) return;
    post_common(id.xy, false);
}
