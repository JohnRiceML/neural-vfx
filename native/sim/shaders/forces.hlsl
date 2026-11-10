// Sources, heat+fuel diffusion, curl, separable [1,2,1]^3 smoothing, body forces (kernels.py).
#include "common.hlsli"

bool solid_at(Texture3D<float> sol, int3 c, int n) {
    if (any(c < 0) || any(c >= n)) return true;
    return sol[c] > 0.5f;
}

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// adv -> out (k0 = VD UAV slot, k1 = TF UAV slot). Mass correction, emitters, impulses,
// combustion, cooling, dissipation.
[numthreads(8, 8, 4)]
void sources(uint3 id : SV_DispatchThreadID) {
    int n = int(RC.n);
    if (any(id >= uint(n))) return;
    Texture3D<float> sol = ResourceDescriptorHeap[SL_SOLID_SRV];
    RWTexture3D<float4> avd = ResourceDescriptorHeap[SL_ADV_VD];
    RWTexture3D<float2> atf = ResourceDescriptorHeap[SL_ADV_TF];
    RWTexture3D<float4> ovd = ResourceDescriptorHeap[RC.k0];
    RWTexture3D<float2> otf = ResourceDescriptorHeap[RC.k1];
    float amb = P.amb;
    if (sol[id] > 0.5f) {
        ovd[id] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        otf[id] = float2(amb, 0.0f);
        return;
    }
    RWByteAddressBuffer scal = ResourceDescriptorHeap[SL_SCAL];
    float dt = P.dt;
    float dx = P.dx;
    float3 pos = float3((float(id.x) + 0.5f) * dx, (float(id.y) + 0.5f) * dx, (float(id.z) + 0.5f) * dx);
    float4 a4 = avd[id];
    float2 a2 = atf[id];
    float d = a4.w * asfloat(scal.Load(4 * S_MASS_RATIO));
    float3 vel = a4.xyz;
    float T = a2.x;
    float F = a2.y;

    int n_em = int(P.n_em);
    for (int i = 0; i < n_em; ++i) {
        float4 e0 = P.em[3 * i], e1 = P.em[3 * i + 1], e2 = P.em[3 * i + 2];
        float rad = e0.w;
        float3 dd = pos - e0.xyz;
        float r2 = dot3(dd, dd) / (rad * rad);
        if (r2 < 1.0f) {
            float w = (1.0f - r2) * (1.0f - r2);
            d += e1.x * w * dt;
            F += e1.z * w * dt;
            float te = e1.y;
            if (te > 0.0f) T = max(T, amb + (te - amb) * w);
            float3 ev = e2.xyz;
            if (dot3(ev, ev) > 0.0f) vel = vel + (ev - vel) * w;
        }
    }
    int n_imp = int(P.n_imp);
    for (int j = 0; j < n_imp; ++j) {
        float4 i0 = P.imp[2 * j], i1 = P.imp[2 * j + 1];
        float rad = i0.w;
        float3 dd = pos - i0.xyz;
        float r2 = dot3(dd, dd) / (rad * rad);
        if (r2 < 9.0f) vel = vel + exp(-r2) * i1.xyz;
    }

    float ign = clamp((T - P.ign) / 100.0f, 0.0f, 1.0f);
    float burned = F * (1.0f - exp(-P.burn * dt)) * ign;
    F -= burned;
    float tmax = P.tmax;
    float room = clamp((tmax - T) / (tmax - amb), 0.0f, 1.0f);
    T += P.bheat * burned * room;
    d += P.smoke * burned;

    T = amb + (T - amb) * exp(-P.cool * dt);
    d = d * exp(-P.ddiss * dt);

    ovd[id] = float4(vel, max(d, 0.0f));
    otf[id] = float2(min(T, tmax), max(F, 0.0f));
}

// One explicit substep of temperature + fuel diffusion. k0/k1 = src VD/TF slots, k2/k3 = dst VD/TF.
[numthreads(8, 8, 4)]
void diffuse(uint3 id : SV_DispatchThreadID) {
    int n = int(RC.n);
    if (any(id >= uint(n))) return;
    Texture3D<float> sol = ResourceDescriptorHeap[SL_SOLID_SRV];
    RWTexture3D<float4> svd = ResourceDescriptorHeap[RC.k0];
    RWTexture3D<float2> stf = ResourceDescriptorHeap[RC.k1];
    RWTexture3D<float4> dvd = ResourceDescriptorHeap[RC.k2];
    RWTexture3D<float2> dtf = ResourceDescriptorHeap[RC.k3];
    float dx = P.dx;
    float lam = min(P.kappa * P.dt / (P.nsub * dx * dx), 0.16f);
    dvd[id] = svd[id];
    float2 v = stf[id];
    int3 c = int3(id);
    if (sol[c] > 0.5f || lam <= 0.0f) {
        dtf[id] = v;
        return;
    }
    static const int3 offs[6] = { int3(-1, 0, 0), int3(1, 0, 0), int3(0, -1, 0), int3(0, 1, 0), int3(0, 0, -1), int3(0, 0, 1) };
    float2 s = float2(0.0f, 0.0f);
    [unroll] for (int k = 0; k < 6; ++k) {
        int3 q = c + offs[k];
        float2 nb = solid_at(sol, q, n) ? v : stf[q];
        s = (k == 0) ? nb : s + nb;
    }
    dtf[id] = v + lam * (s - 6.0f * v);
}

float3 vel_or0(RWTexture3D<float4> f, Texture3D<float> sol, int3 c, int n) {
    if (solid_at(sol, c, n)) return float3(0.0f, 0.0f, 0.0f);
    return f[c].xyz;
}

// omega = curl u (central differences, walls / solids = zero velocity). k0 = VD UAV slot.
[numthreads(8, 8, 4)]
void curl(uint3 id : SV_DispatchThreadID) {
    int n = int(RC.n);
    if (any(id >= uint(n))) return;
    Texture3D<float> sol = ResourceDescriptorHeap[SL_SOLID_SRV];
    RWTexture3D<float4> f = ResourceDescriptorHeap[RC.k0];
    RWTexture3D<float4> w = ResourceDescriptorHeap[SL_OMEGA];
    int3 c = int3(id);
    float3 yp = vel_or0(f, sol, c + int3(0, 1, 0), n), ym = vel_or0(f, sol, c - int3(0, 1, 0), n);
    float3 zp = vel_or0(f, sol, c + int3(0, 0, 1), n), zm = vel_or0(f, sol, c - int3(0, 0, 1), n);
    float3 xp = vel_or0(f, sol, c + int3(1, 0, 0), n), xm = vel_or0(f, sol, c - int3(1, 0, 0), n);
    float dwdy = yp.z - ym.z;
    float dvdz = zp.y - zm.y;
    float dudz = zp.x - zm.x;
    float dwdx = xp.z - xm.z;
    float dvdx = xp.y - xm.y;
    float dudy = yp.x - ym.x;
    float inv2dx = 0.5f * P.inv_dx;
    w[id] = float4((dwdy - dvdz) * inv2dx, (dudz - dwdx) * inv2dx, (dvdx - dudy) * inv2dx, 0.0f);
}

// out.xyz = [1,2,1]_x (w * a.xyz), out.w = [1,2,1]_x w   (w = fluid mask). k0 = source slot.
[numthreads(8, 8, 4)]
void sep_x(uint3 id : SV_DispatchThreadID) {
    int n = int(RC.n);
    if (any(id >= uint(n))) return;
    Texture3D<float> sol = ResourceDescriptorHeap[SL_SOLID_SRV];
    RWTexture3D<float4> a = ResourceDescriptorHeap[RC.k0];
    RWTexture3D<float4> o = ResourceDescriptorHeap[SL_TA];
    int3 c = int3(id);
    float3 s = float3(0.0f, 0.0f, 0.0f);
    float ws = 0.0f;
    [unroll] for (int i = -1; i <= 1; ++i) {
        int3 q = c + int3(i, 0, 0);
        if (!solid_at(sol, q, n)) {
            float qq = float(2 - abs(i));
            s += qq * a[q].xyz;
            ws += qq;
        }
    }
    o[id] = float4(s, ws);
}

[numthreads(8, 8, 4)]
void sep_y(uint3 id : SV_DispatchThreadID) {
    int n = int(RC.n);
    if (any(id >= uint(n))) return;
    RWTexture3D<float4> a = ResourceDescriptorHeap[SL_TA];
    RWTexture3D<float4> o = ResourceDescriptorHeap[SL_TB];
    int3 c = int3(id);
    float4 v = 2.0f * a[c];
    if (c.y > 0) v += a[c - int3(0, 1, 0)];
    if (c.y + 1 < n) v += a[c + int3(0, 1, 0)];
    o[id] = v;
}

float3 sep_z_ratio(RWTexture3D<float4> a, int3 c, int n) {
    float3 s = float3(0.0f, 0.0f, 0.0f);
    float ws = 0.0f;
    [unroll] for (int k = -1; k <= 1; ++k) {
        int zz = c.z + k;
        if (zz >= 0 && zz < n) {
            float q = float(2 - abs(k));
            float4 v = a[int3(c.x, c.y, zz)];
            s += q * v.xyz;
            ws += q * v.w;
        }
    }
    if (ws > 0.0f) return s / ws;
    return float3(0.0f, 0.0f, 0.0f);
}

[numthreads(8, 8, 4)]
void sep_z_store(uint3 id : SV_DispatchThreadID) {
    int n = int(RC.n);
    if (any(id >= uint(n))) return;
    RWTexture3D<float4> a = ResourceDescriptorHeap[SL_TB];
    RWTexture3D<float4> o = ResourceDescriptorHeap[SL_OMEGA_S];
    o[id] = float4(sep_z_ratio(a, int3(id), n), 0.0f);
}

// u += dt * smoothed force (fluid cells). k0 = VD UAV slot.
[numthreads(8, 8, 4)]
void sep_z_apply(uint3 id : SV_DispatchThreadID) {
    int n = int(RC.n);
    if (any(id >= uint(n))) return;
    Texture3D<float> sol = ResourceDescriptorHeap[SL_SOLID_SRV];
    if (sol[id] > 0.5f) return;
    RWTexture3D<float4> a = ResourceDescriptorHeap[SL_TB];
    RWTexture3D<float4> f = ResourceDescriptorHeap[RC.k0];
    float3 dv = P.dt * sep_z_ratio(a, int3(id), n);
    float4 v = f[id];
    f[id] = float4(v.x + dv.x, v.y + dv.y, v.z + dv.z, v.w);
}

float wmag(RWTexture3D<float4> o, int3 c) {
    float3 v = o[c].xyz;
    return sqrt(dot3(v, v));
}

// Buoyancy + vorticity confinement + wind-marker force -> SL_FORCE. k0/k1 = VD/TF UAV slots.
[numthreads(8, 8, 4)]
void body_forces(uint3 id : SV_DispatchThreadID) {
    int n = int(RC.n);
    if (any(id >= uint(n))) return;
    Texture3D<float> sol = ResourceDescriptorHeap[SL_SOLID_SRV];
    RWTexture3D<float4> force = ResourceDescriptorHeap[SL_FORCE];
    if (sol[id] > 0.5f) {
        force[id] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }
    RWTexture3D<float4> fvd = ResourceDescriptorHeap[RC.k0];
    RWTexture3D<float2> ftf = ResourceDescriptorHeap[RC.k1];
    RWTexture3D<float4> om = ResourceDescriptorHeap[SL_OMEGA_S];
    int3 c = int3(id);
    float amb = P.amb;
    float d = fvd[c].w;
    float2 tf = ftf[c];
    float T = tf.x;
    float3 fv = float3(0.0f, -P.alpha * d + P.beta * (T - amb), 0.0f);

    float eps = P.eps;
    if (eps != 0.0f) {
        int xr = min(c.x + 1, n - 1), xl = max(c.x - 1, 0);
        int yt = min(c.y + 1, n - 1), yb = max(c.y - 1, 0);
        int zf = min(c.z + 1, n - 1), zb = max(c.z - 1, 0);
        float3 g = float3((wmag(om, int3(xr, c.y, c.z)) - wmag(om, int3(xl, c.y, c.z))) / float(max(xr - xl, 1)),
                          (wmag(om, int3(c.x, yt, c.z)) - wmag(om, int3(c.x, yb, c.z))) / float(max(yt - yb, 1)),
                          (wmag(om, int3(c.x, c.y, zf)) - wmag(om, int3(c.x, c.y, zb))) / float(max(zf - zb, 1)));
        float3 w = om[c].xyz;
        float fl = P.cfloor * sqrt(dot3(w, w));
        float mag = sqrt(dot3(g, g) + fl * fl) + 1.0e-20f;
        float3 cr = float3(g.y * w.z - g.z * w.y, g.z * w.x - g.x * w.z, g.x * w.y - g.y * w.x);
        fv += (eps * P.dx / mag) * cr;
    }
    float3 wind = float3(P.wind_x, P.wind_y, P.wind_z);
    if (dot3(wind, wind) > 0.0f) {
        float m = clamp(d + tf.y + (T - amb) / P.wmark, 0.0f, 1.0f);
        fv += wind * m;
    }
    force[id] = float4(fv, 0.0f);
}
