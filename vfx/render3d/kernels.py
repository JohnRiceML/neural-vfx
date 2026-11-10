"""Warp kernels for the 3D volume renderer (see renderer.py for the pipeline).

World: the domain cube spans [0,1]^3 (DOMAIN_SIZE = 1 m), x right, y up, z depth. Volumes are
(D,H,W) = [z,y,x] arrays, so a world point p=(x,y,z) is directly the normalized texture coordinate
(u,v,w) of a wp.Texture3D built from them (hardware trilinear filtering, ~1.8x faster than manual
trilinear on array loads, measured on the RTX 5080).
"""
from __future__ import annotations

import warp as wp

wp.config.quiet = True

BRICKS = 32            # empty-space skipping grid: 32^3 bricks over the unit cube


@wp.struct
class Shade3D:
    k_abs: float          # smoke extinction per meter per unit density
    k_shadow: float       # extinction toward the light (light-volume march)
    k_solid: float        # obstacle extinction toward the light (per meter at solid=1)
    k_amb: float          # extinction for the sky-ambient occlusion march (straight up)
    k_emit: float         # emissivity growth with density
    emit_min: float       # emissivity of hot but clear gas
    emit_scale: float     # emission radiance per meter (multiplies the blackbody LUT)
    soot: float           # how much hot smoke loses its scattering albedo (0..1)
    fire_light: float     # how strongly the fire glow volume lights smoke / obstacles / floor
    phase_g: float        # Henyey-Greenstein anisotropy (forward scatter > 0)
    exposure: float
    bloom: float
    bloom_thresh: float
    t_lo: float
    t_hi: float
    d_eps: float          # density below which a sample is treated as empty
    floor: float          # floor plane opacity (0 disables it)
    light_dir: wp.vec3    # unit, TOWARD the light
    key_col: wp.vec3
    amb_col: wp.vec3
    smoke_albedo: wp.vec3
    bg_top: wp.vec3
    bg_bot: wp.vec3
    floor_col: wp.vec3
    grid_col: wp.vec3
    obst_col: wp.vec3


# ---------------------------------------------------------------------- helpers
@wp.func
def lut_lookup(lut: wp.array(dtype=wp.vec3), T: float, t_lo: float, t_hi: float) -> wp.vec3:
    n = lut.shape[0]
    f = wp.clamp((T - t_lo) / (t_hi - t_lo), 0.0, 1.0) * float(n - 1)
    i0 = wp.min(int(f), n - 2)
    t = f - float(i0)
    return lut[i0] * (1.0 - t) + lut[i0 + 1] * t


@wp.func
def smoothstep(e0: float, e1: float, x: float) -> float:
    t = wp.clamp((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@wp.func
def box_hit(o: wp.vec3, invd: wp.vec3, lo: wp.vec3, hi: wp.vec3) -> wp.vec2:
    """Slab test. Returns (t_near, t_far); miss when t_near > t_far."""
    t0 = wp.cw_mul(lo - o, invd)
    t1 = wp.cw_mul(hi - o, invd)
    tmin = wp.vec3(wp.min(t0[0], t1[0]), wp.min(t0[1], t1[1]), wp.min(t0[2], t1[2]))
    tmax = wp.vec3(wp.max(t0[0], t1[0]), wp.max(t0[1], t1[1]), wp.max(t0[2], t1[2]))
    return wp.vec2(wp.max(wp.max(tmin[0], tmin[1]), tmin[2]), wp.min(wp.min(tmax[0], tmax[1]), tmax[2]))


@wp.func
def safe_inv(d: wp.vec3) -> wp.vec3:
    e = 1.0e-8
    x = wp.where(wp.abs(d[0]) < e, wp.where(d[0] < 0.0, -e, e), d[0])
    y = wp.where(wp.abs(d[1]) < e, wp.where(d[1] < 0.0, -e, e), d[1])
    z = wp.where(wp.abs(d[2]) < e, wp.where(d[2] < 0.0, -e, e), d[2])
    return wp.vec3(1.0 / x, 1.0 / y, 1.0 / z)


@wp.func
def hg(cos_t: float, g: float) -> float:
    d = 1.0 + g * g - 2.0 * g * cos_t
    return (1.0 - g * g) / (d * wp.sqrt(d))          # x 1/(4 pi) omitted: isotropic == 1


@wp.func
def emission(lut: wp.array(dtype=wp.vec3), p: Shade3D, d: float, T: float) -> wp.vec3:
    e = p.emit_min + (1.0 - p.emit_min) * (1.0 - wp.exp(-p.k_emit * d))
    return lut_lookup(lut, T, p.t_lo, p.t_hi) * (e * p.emit_scale)


@wp.func
def aces(x: float) -> float:
    return wp.clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


@wp.func
def to_srgb8(x: float, dither: float) -> wp.uint8:
    s = float(0.0)
    if x <= 0.0031308:
        s = 12.92 * x
    else:
        s = 1.055 * wp.pow(x, 1.0 / 2.4) - 0.055
    return wp.uint8(wp.clamp(int(s * 255.0 + 0.5 + dither), 0, 255))


@wp.func
def ign(x: float, y: float) -> float:
    """Interleaved gradient noise in [0,1): low-discrepancy per-pixel ray jitter."""
    f = 52.9829189 * wp.frac(0.06711056 * x + 0.00583715 * y)
    return wp.frac(f)


# ---------------------------------------------------------------------- prep kernels
@wp.kernel
def pack_volume(dens: wp.array3d(dtype=wp.float32), temp: wp.array3d(dtype=wp.float32),
                packed: wp.array3d(dtype=wp.vec2h), occ: wp.array3d(dtype=wp.uint8),
                d_eps: float, t_hot: float):
    """(density, temperature) -> packed half2 for the texture upload + occupancy bricks (with a
    one-voxel apron so trilinear footprints that straddle a brick boundary are kept)."""
    z, y, x = wp.tid()
    d = dens[z, y, x]
    T = temp[z, y, x]
    packed[z, y, x] = wp.vec2h(wp.float16(wp.max(d, 0.0)), wp.float16(T))
    if d > d_eps or T > t_hot:
        D = dens.shape[0]
        H = dens.shape[1]
        W = dens.shape[2]
        bz0 = wp.max(z - 1, 0) * BRICKS // D
        bz1 = wp.min(z + 1, D - 1) * BRICKS // D
        by0 = wp.max(y - 1, 0) * BRICKS // H
        by1 = wp.min(y + 1, H - 1) * BRICKS // H
        bx0 = wp.max(x - 1, 0) * BRICKS // W
        bx1 = wp.min(x + 1, W - 1) * BRICKS // W
        for bz in range(bz0, bz1 + 1):
            for by in range(by0, by1 + 1):
                for bx in range(bx0, bx1 + 1):
                    occ[bz, by, bx] = wp.uint8(1)


@wp.kernel
def pack_volume_h(dens: wp.array3d(dtype=wp.float16), temp: wp.array3d(dtype=wp.float16),
                  packed: wp.array3d(dtype=wp.vec2h), occ: wp.array3d(dtype=wp.uint8),
                  d_eps: float, t_hot: float):
    z, y, x = wp.tid()
    dh = dens[z, y, x]
    d = float(dh)
    T = float(temp[z, y, x])
    packed[z, y, x] = wp.vec2h(wp.float16(wp.max(d, 0.0)), temp[z, y, x])
    if d > d_eps or T > t_hot:
        D = dens.shape[0]
        H = dens.shape[1]
        W = dens.shape[2]
        bz0 = wp.max(z - 1, 0) * BRICKS // D
        bz1 = wp.min(z + 1, D - 1) * BRICKS // D
        by0 = wp.max(y - 1, 0) * BRICKS // H
        by1 = wp.min(y + 1, H - 1) * BRICKS // H
        bx0 = wp.max(x - 1, 0) * BRICKS // W
        bx1 = wp.min(x + 1, W - 1) * BRICKS // W
        for bz in range(bz0, bz1 + 1):
            for by in range(by0, by1 + 1):
                for bx in range(bx0, bx1 + 1):
                    occ[bz, by, bx] = wp.uint8(1)


@wp.kernel
def pack_solid(solid: wp.array3d(dtype=wp.float32), out: wp.array3d(dtype=wp.uint8),
               occ: wp.array3d(dtype=wp.uint8)):
    z, y, x = wp.tid()
    s = solid[z, y, x]
    out[z, y, x] = wp.uint8(wp.clamp(int(s * 255.0 + 0.5), 0, 255))
    if s > 0.0:
        D = solid.shape[0]
        H = solid.shape[1]
        W = solid.shape[2]
        bz0 = wp.max(z - 1, 0) * BRICKS // D
        bz1 = wp.min(z + 1, D - 1) * BRICKS // D
        by0 = wp.max(y - 1, 0) * BRICKS // H
        by1 = wp.min(y + 1, H - 1) * BRICKS // H
        bx0 = wp.max(x - 1, 0) * BRICKS // W
        bx1 = wp.min(x + 1, W - 1) * BRICKS // W
        for bz in range(bz0, bz1 + 1):
            for by in range(by0, by1 + 1):
                for bx in range(bx0, bx1 + 1):
                    occ[bz, by, bx] = wp.uint8(1)


@wp.kernel
def light_volume(vol: wp.Texture3D, solid: wp.Texture3D, has_solid: int, lut: wp.array(dtype=wp.vec3),
                 p: Shade3D, light: wp.array3d(dtype=wp.vec2h), glow_src: wp.array3d(dtype=wp.vec4h)):
    """Per light-grid voxel: transmittance toward the key light (march to the cube exit),
    sky-ambient occlusion (short march up), and the local blackbody emission (glow source)."""
    z, y, x = wp.tid()
    L = light.shape[0]
    inv = 1.0 / float(L)
    pos = wp.vec3((float(x) + 0.5) * inv, (float(y) + 0.5) * inv, (float(z) + 0.5) * inv)
    here = wp.texture_sample(vol, pos, dtype=wp.vec2f)
    ld = p.light_dir
    tb = box_hit(pos, safe_inv(ld), wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 1.0, 1.0))
    t_exit = tb[1]
    ds = 2.0 * inv                      # 2 light voxels per step: shadows are soft at this res anyway
    od = float(0.0)
    t = 0.5 * ds
    while t < t_exit:
        q = pos + ld * t
        s = wp.texture_sample(vol, q, dtype=wp.vec2f)
        od += p.k_shadow * s[0] * ds
        if has_solid != 0:
            od += p.k_solid * wp.texture_sample(solid, q, dtype=float) * ds
        if od > 12.0:
            break
        t += ds
    # sky ambient: 6 coarse steps straight up
    oda = float(0.0)
    da = 3.0 * inv
    for i in range(6):
        qy = pos[1] + (float(i) + 0.5) * da
        if qy < 1.0:
            s = wp.texture_sample(vol, wp.vec3(pos[0], qy, pos[2]), dtype=wp.vec2f)
            oda += p.k_amb * s[0] * da
    light[z, y, x] = wp.vec2h(wp.float16(wp.exp(-od)), wp.float16(wp.exp(-oda)))
    e = wp.vec3(0.0, 0.0, 0.0)
    if here[1] > p.t_lo:
        e = emission(lut, p, here[0], here[1])
    glow_src[z, y, x] = wp.vec4h(wp.float16(e[0]), wp.float16(e[1]), wp.float16(e[2]), wp.float16(0.0))


# ---------------------------------------------------------------------- main march
@wp.func
def light_at(light: wp.Texture3D, ld: wp.vec3, q: wp.vec3) -> wp.vec2:
    """Light-volume lookup valid for points outside the cube too (uses the entry point toward the light)."""
    inside = q[0] >= 0.0 and q[0] <= 1.0 and q[1] >= 0.0 and q[1] <= 1.0 and q[2] >= 0.0 and q[2] <= 1.0
    if inside:
        return wp.texture_sample(light, q, dtype=wp.vec2f)
    tb = box_hit(q, safe_inv(ld), wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 1.0, 1.0))
    if tb[0] <= tb[1] and tb[1] > 0.0:
        e = q + ld * (wp.max(tb[0], 0.0) + 1.0e-3)
        return wp.vec2(wp.texture_sample(light, e, dtype=wp.vec2f)[0], 1.0)
    return wp.vec2(1.0, 1.0)


@wp.func
def glow_at(glow: wp.Texture3D, q: wp.vec3) -> wp.vec3:
    c = wp.vec3(wp.clamp(q[0], 0.0, 1.0), wp.clamp(q[1], 0.0, 1.0), wp.clamp(q[2], 0.0, 1.0))
    g4 = wp.texture_sample(glow, c, dtype=wp.vec4f)
    dout = wp.length(q - c)
    return wp.vec3(g4[0], g4[1], g4[2]) / (1.0 + 60.0 * dout * dout)


@wp.kernel
def march(vol: wp.Texture3D, light: wp.Texture3D, glow: wp.Texture3D, solid: wp.Texture3D,
          occ: wp.array3d(dtype=wp.uint8), occ_s: wp.array3d(dtype=wp.uint8), has_solid: int,
          solid_res: float, lut: wp.array(dtype=wp.vec3), p: Shade3D, cam: wp.array(dtype=float),
          steps: float, out: wp.array2d(dtype=wp.vec4)):
    """One ray per pixel: front-to-back compositing of smoke/fire inside [0,1]^3, opaque obstacles
    (solid iso 0.5), then the floor plane y=0. Writes (premultiplied radiance rgb, transmittance)."""
    r, c = wp.tid()
    h = out.shape[0]
    w = out.shape[1]
    o = wp.vec3(cam[0], cam[1], cam[2])
    fw = wp.vec3(cam[3], cam[4], cam[5])
    rt = wp.vec3(cam[6], cam[7], cam[8])
    up = wp.vec3(cam[9], cam[10], cam[11])
    th = cam[12]
    aspect = cam[13]
    sx = (2.0 * (float(c) + 0.5) / float(w) - 1.0) * th * aspect
    sy = (1.0 - 2.0 * (float(r) + 0.5) / float(h)) * th
    d = wp.normalize(fw + rt * sx + up * sy)
    invd = safe_inv(d)
    ld = p.light_dir

    rad = wp.vec3(0.0, 0.0, 0.0)
    T = float(1.0)
    tb = box_hit(o, invd, wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 1.0, 1.0))
    t_near = wp.max(tb[0], 0.0)
    t_far = tb[1]
    ds = 1.0 / steps
    cos_t = wp.dot(d, ld)
    phase = hg(cos_t, p.phase_g) * 0.7 + hg(cos_t, -0.25) * 0.3
    if t_near < t_far:
        jit = ign(float(c), float(r))
        t0 = t_near + jit * ds
        t = t0
        t_prev = t_near
        while t < t_far:
            q = o + d * t
            bx = wp.clamp(int(q[0] * float(BRICKS)), 0, BRICKS - 1)
            by = wp.clamp(int(q[1] * float(BRICKS)), 0, BRICKS - 1)
            bz = wp.clamp(int(q[2] * float(BRICKS)), 0, BRICKS - 1)
            o_d = occ[bz, by, bx]
            o_s = wp.uint8(0)
            if has_solid != 0:
                o_s = occ_s[bz, by, bx]
            if o_d == wp.uint8(0) and o_s == wp.uint8(0):
                # skip to the exit of this empty brick, staying on the jittered step lattice
                inv_b = 1.0 / float(BRICKS)
                lo = wp.vec3(float(bx), float(by), float(bz)) * inv_b
                bb = box_hit(o, invd, lo, lo + wp.vec3(inv_b, inv_b, inv_b))
                k = wp.ceil((bb[1] - t0) / ds + 1.0e-4)
                t_prev = t
                t = wp.max(t0 + k * ds, t + ds)
                continue
            if o_s != wp.uint8(0):
                if wp.texture_sample(solid, q, dtype=float) > 0.5:
                    # refine the surface crossing between t_prev and t
                    a = t_prev
                    b = t
                    for _i in range(5):
                        m = 0.5 * (a + b)
                        if wp.texture_sample(solid, o + d * m, dtype=float) > 0.5:
                            b = m
                        else:
                            a = m
                    ph = o + d * b
                    e = 1.0 / solid_res
                    gx = wp.texture_sample(solid, ph + wp.vec3(e, 0.0, 0.0), dtype=float) - wp.texture_sample(solid, ph - wp.vec3(e, 0.0, 0.0), dtype=float)
                    gy = wp.texture_sample(solid, ph + wp.vec3(0.0, e, 0.0), dtype=float) - wp.texture_sample(solid, ph - wp.vec3(0.0, e, 0.0), dtype=float)
                    gz = wp.texture_sample(solid, ph + wp.vec3(0.0, 0.0, e), dtype=float) - wp.texture_sample(solid, ph - wp.vec3(0.0, 0.0, e), dtype=float)
                    n = -wp.vec3(gx, gy, gz)
                    if wp.length(n) < 1.0e-6:
                        n = -d
                    n = wp.normalize(n)
                    lq = ph + n * 0.03
                    lv = wp.texture_sample(light, lq, dtype=wp.vec2f)
                    gl = glow_at(glow, lq)
                    ndl = wp.max(wp.dot(n, ld), 0.0)
                    hv = wp.normalize(ld - d)
                    spec = wp.pow(wp.max(wp.dot(n, hv), 0.0), 40.0) * 0.25
                    hemi = 0.45 + 0.55 * (0.5 + 0.5 * n[1])
                    fres = wp.pow(1.0 - wp.max(wp.dot(n, -d), 0.0), 4.0) * 0.08
                    col = wp.cw_mul(p.obst_col, p.key_col * (ndl * lv[0]) + p.amb_col * (hemi * lv[1]) + gl * (2.0 * p.fire_light))
                    col = col + p.key_col * (spec * lv[0]) + p.amb_col * fres
                    rad = rad + col * T
                    T = 0.0
                    break
            if o_d != wp.uint8(0):
                s = wp.texture_sample(vol, q, dtype=wp.vec2f)
                dn = s[0]
                Tk = s[1]
                hot = Tk > p.t_lo
                if dn > p.d_eps or hot:
                    sigma = p.k_abs * wp.max(dn, 0.0)
                    a = 1.0 - wp.exp(-sigma * ds)
                    col = wp.vec3(0.0, 0.0, 0.0)
                    if a > 1.0e-5:
                        lv = wp.texture_sample(light, q, dtype=wp.vec2f)
                        gl = glow_at(glow, q)
                        sh = smoothstep(p.t_lo, p.t_lo + 500.0, Tk)
                        alb = p.smoke_albedo * (1.0 - p.soot * sh)
                        lit = p.key_col * (lv[0] * phase) + p.amb_col * lv[1] + gl * p.fire_light
                        col = wp.cw_mul(alb, lit) * a
                    if hot:
                        col = col + emission(lut, p, dn, Tk) * ds
                    rad = rad + col * T
                    T = T * (1.0 - a)
                    if T < 0.004:
                        T = 0.0
                        break
            t_prev = t
            t = t + ds

    # floor plane y = 0 (behind everything inside the cube, see README)
    if T > 0.0 and d[1] < -1.0e-4 and o[1] > 0.0:
        tf = -o[1] / d[1]
        pf = o + d * tf
        cx = pf[0] - 0.5
        cz = pf[2] - 0.5
        rr = wp.sqrt(cx * cx + cz * cz)
        fade = (1.0 - smoothstep(0.35, 1.9, rr)) * p.floor
        if fade > 0.0:
            pix = tf * 2.0 * th / float(h)                   # world size of one pixel at the floor
            gs = 0.125
            gxl = wp.abs(pf[0] / gs - wp.round(pf[0] / gs)) * gs
            gzl = wp.abs(pf[2] / gs - wp.round(pf[2] / gs)) * gs
            line = 1.0 - smoothstep(0.0, 1.2 * pix + 0.0015, wp.min(gxl, gzl))
            inside = pf[0] >= 0.0 and pf[0] <= 1.0 and pf[2] >= 0.0 and pf[2] <= 1.0
            lq = wp.vec3(pf[0], 0.004, pf[2])
            lv = light_at(light, ld, lq)
            gl = glow_at(glow, lq)
            base = p.floor_col + p.grid_col * (line * wp.where(inside, 1.0, 0.45))
            lit = p.key_col * (ld[1] * lv[0]) + p.amb_col * (0.6 + 0.4 * lv[1]) + gl * (2.5 * p.fire_light)
            fc = wp.cw_mul(base, lit)
            rad = rad + fc * (T * fade)
            T = T * (1.0 - fade)

    out[r, c] = wp.vec4(rad[0], rad[1], rad[2], T)


# ---------------------------------------------------------------------- composite
@wp.func
def sample_v4(a: wp.array2d(dtype=wp.vec4), u: float, v: float) -> wp.vec4:
    H = a.shape[0]
    W = a.shape[1]
    fx = u * float(W) - 0.5
    fy = v * float(H) - 0.5
    x0f = wp.floor(fx)
    y0f = wp.floor(fy)
    tx = fx - x0f
    ty = fy - y0f
    x0 = wp.clamp(int(x0f), 0, W - 1)
    y0 = wp.clamp(int(y0f), 0, H - 1)
    x1 = wp.clamp(int(x0f) + 1, 0, W - 1)
    y1 = wp.clamp(int(y0f) + 1, 0, H - 1)
    a0 = a[y0, x0] * (1.0 - tx) + a[y0, x1] * tx
    a1 = a[y1, x0] * (1.0 - tx) + a[y1, x1] * tx
    return a0 * (1.0 - ty) + a1 * ty


@wp.func
def sample_v3(a: wp.array2d(dtype=wp.vec3), u: float, v: float) -> wp.vec3:
    H = a.shape[0]
    W = a.shape[1]
    fx = u * float(W) - 0.5
    fy = v * float(H) - 0.5
    x0f = wp.floor(fx)
    y0f = wp.floor(fy)
    tx = fx - x0f
    ty = fy - y0f
    x0 = wp.clamp(int(x0f), 0, W - 1)
    y0 = wp.clamp(int(y0f), 0, H - 1)
    x1 = wp.clamp(int(x0f) + 1, 0, W - 1)
    y1 = wp.clamp(int(y0f) + 1, 0, H - 1)
    a0 = a[y0, x0] * (1.0 - tx) + a[y0, x1] * tx
    a1 = a[y1, x0] * (1.0 - tx) + a[y1, x1] * tx
    return a0 * (1.0 - ty) + a1 * ty


@wp.kernel
def composite(hdr: wp.array2d(dtype=wp.vec4), bloom: wp.array2d(dtype=wp.vec3), p: Shade3D,
              out: wp.array3d(dtype=wp.uint8)):
    """Upsample the march buffer to the output size, add background + bloom, tone-map, sRGB, dither."""
    r, c = wp.tid()
    Ho = out.shape[0]
    Wo = out.shape[1]
    u = (float(c) + 0.5) / float(Wo)
    v = (float(r) + 0.5) / float(Ho)
    px = wp.vec4(0.0, 0.0, 0.0, 1.0)
    if hdr.shape[0] == Ho and hdr.shape[1] == Wo:
        px = hdr[r, c]
    else:
        px = sample_v4(hdr, u, v)
    vy = 1.0 - v
    vig = 1.0 - 0.6 * ((u - 0.5) * (u - 0.5) + (vy - 0.6) * (vy - 0.6))
    bg = (p.bg_bot * (1.0 - vy) + p.bg_top * vy) * vig
    col = wp.vec3(px[0], px[1], px[2]) + bg * px[3] + sample_v3(bloom, u, v) * p.bloom
    col = col * p.exposure
    hh = wp.sin(float(r) * 12.9898 + float(c) * 78.233) * 43758.5453
    dither = (hh - wp.floor(hh)) - 0.5
    out[r, c, 0] = to_srgb8(aces(col[0]), dither)
    out[r, c, 1] = to_srgb8(aces(col[1]), dither)
    out[r, c, 2] = to_srgb8(aces(col[2]), dither)
