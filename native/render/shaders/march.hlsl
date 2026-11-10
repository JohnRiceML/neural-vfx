// The ray march: port of kernels.py _make_march(AUX). Compiled twice: AUX=0 (colour only, "march") and AUX=1
// ("march_aux"). As in Python, the aux code is removed at compile time from the colour-only variant, and the
// colour arithmetic is the same source in both. OWNER: native render engineer.
//
// One ray per march pixel: front-to-back compositing of smoke/fire inside [0,1]^3 with empty-brick skipping on
// the jittered step lattice, opaque obstacles (blurred solid iso 0.5, bisection-refined), then the floor y = 0.
// Writes R_HDR = (premultiplied radiance rgb, transmittance) and, AUX=1, R_AUXHDR = premultiplied aux sums.
// RC.k0 = 1 in the AUX variant used for a colour-only draw (NVFX_RENDER_SINGLE_MARCH=1): skip the aux write.
#include "render_common.hlsli"

#ifndef AUX
#define AUX 0
#endif

// light-volume lookup valid for points outside the cube too (uses the entry point toward the light)
float2 light_at(Texture3D<float2> light, float3 ld, float3 q) {
    bool inside = q.x >= 0.0f && q.x <= 1.0f && q.y >= 0.0f && q.y <= 1.0f && q.z >= 0.0f && q.z <= 1.0f;
    if (inside) return light.SampleLevel(s_lin, q, 0.0f);
    float2 tb = box_hit(q, safe_inv(ld), float3(0.0f, 0.0f, 0.0f), float3(1.0f, 1.0f, 1.0f));
    if (tb.x <= tb.y && tb.y > 0.0f) {
        float3 e = q + ld * (max(tb.x, 0.0f) + 1.0e-3f);
        return float2(light.SampleLevel(s_lin, e, 0.0f).x, 1.0f);
    }
    return float2(1.0f, 1.0f);
}

float3 glow_at(Texture3D<float4> glow, float3 q) {
    float3 c = float3(clamp(q.x, 0.0f, 1.0f), clamp(q.y, 0.0f, 1.0f), clamp(q.z, 0.0f, 1.0f));
    float4 g4 = glow.SampleLevel(s_lin, c, 0.0f);
    float dout = wlength(q - c);
    return float3(g4.x, g4.y, g4.z) / (1.0f + 60.0f * dout * dout);
}

// world point -> continuous output-pixel coordinates (x right, y down) through the previous camera basis
float2 project_prev(float3 q) {
    float3 rel = q - float3(camp_f(0), camp_f(1), camp_f(2));
    float z = max(dot(rel, float3(camp_f(3), camp_f(4), camp_f(5))), 1.0e-4f);
    float sx = dot(rel, float3(camp_f(6), camp_f(7), camp_f(8))) / (z * camp_f(12) * camp_f(13));
    float sy = dot(rel, float3(camp_f(9), camp_f(10), camp_f(11))) / (z * camp_f(12));
    return float2((sx + 1.0f) * 0.5f * P.out_w, (1.0f - sy) * 0.5f * P.out_h);
}

// C round(): half away from zero (wp.round)
float round_away(float x) { return x < 0.0f ? -floor(-x + 0.5f) : floor(x + 0.5f); }

[numthreads(8, 8, 1)]
void r_march(uint3 id : SV_DispatchThreadID) {
    const int h = int(P.mh), w = int(P.mw);
    const int c = int(id.x), r = int(id.y);
    if (c >= w || r >= h) return;

    Texture3D<float2> vol = ResourceDescriptorHeap[R_VOL_SRV];
    Texture3D<float2> light = ResourceDescriptorHeap[R_LIGHT_SRV];
    Texture3D<float4> glow = ResourceDescriptorHeap[R_GLOW_SRV];
    Texture3D<float> solid = ResourceDescriptorHeap[R_SOLID8_SRV];
    RWStructuredBuffer<uint> occ = ResourceDescriptorHeap[R_OCCD];
    RWStructuredBuffer<uint> occ_s = ResourceDescriptorHeap[R_OCCS];
    const bool has_solid = P.has_solid != 0;

    float3 o = float3(cam_f(0), cam_f(1), cam_f(2));
    float3 fw = float3(cam_f(3), cam_f(4), cam_f(5));
    float3 rt = float3(cam_f(6), cam_f(7), cam_f(8));
    float3 up = float3(cam_f(9), cam_f(10), cam_f(11));
    float th = cam_f(12);
    float aspect = cam_f(13);
    float sx = (2.0f * (float(c) + 0.5f) / float(w) - 1.0f) * th * aspect;
    float sy = (1.0f - 2.0f * (float(r) + 0.5f) / float(h)) * th;
    float3 d = wnormalize(fw + rt * sx + up * sy);
    float3 invd = safe_inv(d);
    float3 ld = P.light_dir.xyz;

    float3 rad = float3(0.0f, 0.0f, 0.0f);
    float T = 1.0f;
    float2 tb = box_hit(o, invd, float3(0.0f, 0.0f, 0.0f), float3(1.0f, 1.0f, 1.0f));
    float t_near = max(tb.x, 0.0f);
    float t_far = tb.y;
    float ds = 1.0f / P.steps;
    float cos_t = dot(d, ld);
    float ph0 = hg(cos_t, P.phase_g) * P.rim + hg(cos_t, -0.25f) * (1.0f - P.rim);
    float ph1 = hg(cos_t, P.phase_g * P.ms_c) * P.rim + hg(cos_t, -0.25f * P.ms_c) * (1.0f - P.rim);
    float ph2 = hg(cos_t, P.phase_g * P.ms_c * P.ms_c) * P.rim + (1.0f - P.rim);
    float a1 = P.ms_a;
    float a2 = P.ms_a * P.ms_a;
    float b1 = P.ms_b;
    float b2 = P.ms_b * P.ms_b;
#if AUX
    float Tm = 1.0f;                          // transmittance of the medium only (not solid / floor)
    float aw = 0.0f;                          // sum of opacity weights T*a
    float ad = 0.0f;                          // sum w * distance
    float3 av = float3(0.0f, 0.0f, 0.0f);     // sum w * velocity
    Texture3D<float4> vel = ResourceDescriptorHeap[R_VEL_SRV];
#endif
    if (t_near < t_far) {
        float jit = ign(float(c), float(r));
        float t0 = t_near + jit * ds;
        float t = t0;
        float t_prev = t_near;
        [loop] while (t < t_far) {
            float3 q = o + d * t;
            int bx = clamp(int(q.x * float(R_BRICKS)), 0, R_BRICKS - 1);
            int by = clamp(int(q.y * float(R_BRICKS)), 0, R_BRICKS - 1);
            int bz = clamp(int(q.z * float(R_BRICKS)), 0, R_BRICKS - 1);
            uint bi = flat3(int3(bx, by, bz), R_BRICKS);
            uint o_d = occ[bi];
            uint o_s = 0u;
            if (has_solid) o_s = occ_s[bi];
            if (o_d == 0u && o_s == 0u) {
                // skip to the exit of this empty brick, staying on the jittered step lattice
                float inv_b = 1.0f / float(R_BRICKS);
                float3 lo = float3(float(bx), float(by), float(bz)) * inv_b;
                float2 bb = box_hit(o, invd, lo, lo + float3(inv_b, inv_b, inv_b));
                float k = ceil((bb.y - t0) / ds + 1.0e-4f);
                t_prev = t;
                t = max(t0 + k * ds, t + ds);
                continue;
            }
            if (o_s != 0u) {
                if (solid.SampleLevel(s_lin, q, 0.0f) > 0.5f) {
                    // refine the surface crossing between t_prev and t
                    float a = t_prev;
                    float b = t;
                    [unroll] for (int it = 0; it < 5; ++it) {
                        float m = 0.5f * (a + b);
                        if (solid.SampleLevel(s_lin, o + d * m, 0.0f) > 0.5f) b = m;
                        else a = m;
                    }
                    float3 ph = o + d * b;
                    float e = 1.0f / P.solid_res;
                    float gx = solid.SampleLevel(s_lin, ph + float3(e, 0.0f, 0.0f), 0.0f) - solid.SampleLevel(s_lin, ph - float3(e, 0.0f, 0.0f), 0.0f);
                    float gy = solid.SampleLevel(s_lin, ph + float3(0.0f, e, 0.0f), 0.0f) - solid.SampleLevel(s_lin, ph - float3(0.0f, e, 0.0f), 0.0f);
                    float gz = solid.SampleLevel(s_lin, ph + float3(0.0f, 0.0f, e), 0.0f) - solid.SampleLevel(s_lin, ph - float3(0.0f, 0.0f, e), 0.0f);
                    float3 n = -float3(gx, gy, gz);
                    if (wlength(n) < 1.0e-6f) n = -d;
                    n = wnormalize(n);
                    float3 lq = ph + n * 0.03f;
                    float2 lv = light.SampleLevel(s_lin, lq, 0.0f);
                    float3 gl = glow_at(glow, lq);
                    float ndl = max(dot(n, ld), 0.0f);
                    float3 hv = wnormalize(ld - d);
                    float spec = pow(max(dot(n, hv), 0.0f), 40.0f) * 0.25f;
                    float hemi = 0.45f + 0.55f * (0.5f + 0.5f * n.y);
                    float fres = pow(1.0f - max(dot(n, -d), 0.0f), 4.0f) * 0.08f;
                    float3 col = P.obst_col.xyz * (P.key_col.xyz * (ndl * lv.x) + P.amb_col.xyz * (hemi * lv.y) + gl * (2.0f * P.fire_light));
                    col = col + P.key_col.xyz * (spec * lv.x) + P.amb_col.xyz * fres;
                    rad = rad + col * T;
                    T = 0.0f;
                    break;
                }
            }
            if (o_d != 0u) {
                float2 s = vol.SampleLevel(s_lin, q, 0.0f);
                float dn = s.x;
                float Tk = s.y;
                bool hot = Tk > P.t_lo;
                if (dn > P.d_eps || hot) {
                    float sigma = P.k_abs * max(dn, 0.0f);
                    float a = 1.0f - exp(-sigma * ds);
                    float3 col = float3(0.0f, 0.0f, 0.0f);
                    if (a > 1.0e-5f) {
                        float2 lv = light.SampleLevel(s_lin, q, 0.0f);
                        float3 gl = glow_at(glow, q);
                        float sh = smoothstep_w(P.t_lo, P.t_lo + 500.0f, Tk);
                        float3 alb = P.smoke_albedo.xyz * (1.0f - P.soot * sh);
                        float tl = max(lv.x, 1.0e-6f);
                        float key = tl * ph0 + a1 * pow(tl, b1) * ph1 + a2 * pow(tl, b2) * ph2;
                        float3 lit = P.key_col.xyz * key + P.amb_col.xyz * lv.y + gl * P.fire_light;
                        col = (alb * lit) * a;
                    }
                    if (hot) col = col + emission(dn, Tk) * ds;
#if AUX
                    float wgt = T * a;
                    if (wgt > 0.0f) {
                        aw += wgt;
                        ad += wgt * t;
                        if (P.has_vel != 0) {
                            float4 v4 = vel.SampleLevel(s_lin, q, 0.0f);
                            av = av + float3(v4.x, v4.y, v4.z) * wgt;
                        }
                    }
#endif
                    rad = rad + col * T;
                    T = T * (1.0f - a);
                    if (T < 0.004f) {
                        T = 0.0f;
#if AUX
                        Tm = 0.0f;
#endif
                        break;
                    }
#if AUX
                    Tm = T;
#endif
                }
            }
            t_prev = t;
            t = t + ds;
        }
    }

    // floor plane y = 0 (behind everything inside the cube)
    if (T > 0.0f && d.y < -1.0e-4f && o.y > 0.0f) {
        float tf = -o.y / d.y;
        float3 pf = o + d * tf;
        float cx = pf.x - 0.5f;
        float cz = pf.z - 0.5f;
        float rr = sqrt(cx * cx + cz * cz);
        float fade = (1.0f - smoothstep_w(0.35f, 1.9f, rr)) * P.floor_op;
        if (fade > 0.0f) {
            float pix = tf * 2.0f * th / float(h);          // world size of one pixel at the floor
            float gs = 0.125f;
            float gxl = abs(pf.x / gs - round_away(pf.x / gs)) * gs;
            float gzl = abs(pf.z / gs - round_away(pf.z / gs)) * gs;
            float line_ = 1.0f - smoothstep_w(0.0f, 1.2f * pix + 0.0015f, min(gxl, gzl));
            bool inside = pf.x >= 0.0f && pf.x <= 1.0f && pf.z >= 0.0f && pf.z <= 1.0f;
            float3 lq = float3(pf.x, 0.004f, pf.z);
            float2 lv = light_at(light, ld, lq);
            float3 gl = glow_at(glow, lq);
            float3 base = P.floor_col.xyz + P.grid_col.xyz * (line_ * (inside ? 1.0f : 0.45f));
            float3 lit = P.key_col.xyz * (ld.y * lv.x) + P.amb_col.xyz * (0.6f + 0.4f * lv.y) + gl * (2.5f * P.fire_light);
            float3 fc = base * lit;
            rad = rad + fc * (T * fade);
            T = T * (1.0f - fade);
        }
    }

    uint pi = uint(r) * uint(w) + uint(c);
    RWStructuredBuffer<float4> hdr = ResourceDescriptorHeap[R_HDR];
    hdr[pi] = float4(rad.x, rad.y, rad.z, T);
#if AUX
    if (RC.k0 == 0u) {
        RWStructuredBuffer<float4> aux = ResourceDescriptorHeap[R_AUXHDR];
        float al = 1.0f - Tm;
        if (aw > 1.0e-6f && al > 0.0f) {
            // opacity-weighted mean point on the ray and mean velocity; the point was at (qm - vm*dt) one sim
            // step ago, seen through the previous camera. The current projection of any point on this ray
            // is the pixel centre itself.
            float dep = ad / aw;
            float3 qm = o + d * dep;
            float3 qp = qm - av * (P.sim_dt / aw);
            float2 cur = float2((float(c) + 0.5f) * P.out_w / float(w), (float(r) + 0.5f) * P.out_h / float(h));
            float2 mv = cur - project_prev(qp);
            aux[pi] = float4(al, dep * al, mv.x * al, mv.y * al);
        } else {
            aux[pi] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
#endif
}
