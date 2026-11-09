"""Warp kernel for the full-resolution composite: one fused pass per output pixel.

All inputs are y-up grids (row 0 = bottom, vfx.core.fields convention); the kernel writes a
top-down (row 0 = top of screen) uint8 RGB image. Low-res helper textures (shadow optical depth,
bloom) are produced in torch by renderer.py and sampled bilinearly here.
"""
from __future__ import annotations

import warp as wp

wp.config.quiet = True


@wp.struct
class Shade:
    k_abs: float          # smoke extinction: alpha = 1 - exp(-k_abs * density)
    k_shadow: float       # extinction toward the light, applied to marched optical depth
    k_emit: float         # emissivity growth with density
    emit_min: float       # emissivity of hot but clear gas
    emit_scale: float     # emission strength (multiplies the blackbody LUT)
    bloom_scale: float    # additive bloom strength
    fire_light: float     # how strongly fire glow lights nearby smoke / obstacles
    rim: float            # thin-edge forward-scatter strength
    exposure: float
    t_lo: float           # LUT temperature range (K)
    t_hi: float
    light_dir: wp.vec2    # direction TOWARD the light, y up
    key_col: wp.vec3
    amb_col: wp.vec3
    rim_col: wp.vec3
    smoke_albedo: wp.vec3
    bg_top: wp.vec3
    bg_bot: wp.vec3
    obst_col: wp.vec3


@wp.func
def sample2(a: wp.array2d(dtype=float), u: float, v: float, nearest: int) -> float:
    """Sample grid `a` at domain coords (u, v) in [0,1]^2 (y up); pixel-center convention."""
    H = a.shape[0]
    W = a.shape[1]
    fx = u * float(W) - 0.5
    fy = v * float(H) - 0.5
    if nearest != 0:
        ix = wp.clamp(int(wp.floor(fx + 0.5)), 0, W - 1)
        iy = wp.clamp(int(wp.floor(fy + 0.5)), 0, H - 1)
        return a[iy, ix]
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
def sample_vec3(a: wp.array2d(dtype=wp.vec3), u: float, v: float) -> wp.vec3:
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
def lut_lookup(lut: wp.array(dtype=wp.vec3), T: float, t_lo: float, t_hi: float) -> wp.vec3:
    n = lut.shape[0]
    f = wp.clamp((T - t_lo) / (t_hi - t_lo), 0.0, 1.0) * float(n - 1)
    i0 = wp.min(int(f), n - 2)
    t = f - float(i0)
    return lut[i0] * (1.0 - t) + lut[i0 + 1] * t


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
def smoothstep(e0: float, e1: float, x: float) -> float:
    t = wp.clamp((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@wp.kernel
def composite(dens: wp.array2d(dtype=float),
              temp: wp.array2d(dtype=float),
              solid: wp.array2d(dtype=float),
              tau: wp.array2d(dtype=float),
              bloom: wp.array2d(dtype=wp.vec3),
              lut: wp.array(dtype=wp.vec3),
              p: Shade,
              nearest: int,
              has_solid: int,
              out: wp.array3d(dtype=wp.uint8)):
    r, c = wp.tid()
    Ho = out.shape[0]
    Wo = out.shape[1]
    u = (float(c) + 0.5) / float(Wo)
    v = 1.0 - (float(r) + 0.5) / float(Ho)          # flip: row 0 = top of screen

    # --- background: vertical gradient + soft vignette
    vig = 1.0 - 0.55 * ((u - 0.5) * (u - 0.5) + (v - 0.55) * (v - 0.55))
    bg = (p.bg_bot * (1.0 - v) + p.bg_top * v) * vig

    # --- smoke: absorbing / scattering medium with marched self-shadow
    d = wp.max(sample2(dens, u, v, nearest), 0.0)
    T = sample2(temp, u, v, nearest)
    a = 1.0 - wp.exp(-p.k_abs * d)
    tl = wp.exp(-p.k_shadow * sample2(tau, u, v, 0))     # transmittance toward the light
    glow = sample_vec3(bloom, u, v)
    lit = p.amb_col + p.key_col * tl + glow * p.fire_light
    hot = smoothstep(p.t_lo + 100.0, p.t_lo + 700.0, T)    # flame region: soot absorbs, little scatter
    smoke = wp.cw_mul(p.smoke_albedo * (1.0 - 0.85 * hot), lit) + p.rim_col * (p.rim * tl * 4.0 * a * (1.0 - a))
    col = bg * (1.0 - a) + smoke * a

    # --- fire: blackbody emission from temperature, emissivity grows with soot density
    emiss = p.emit_min + (1.0 - p.emit_min) * (1.0 - wp.exp(-p.k_emit * d))
    col = col + lut_lookup(lut, T, p.t_lo, p.t_hi) * (emiss * p.emit_scale) + glow * p.bloom_scale

    # --- obstacles: solid shapes with a lit rim, tinted by nearby fire glow
    if has_solid != 0:
        s = sample2(solid, u, v, nearest)
        m = smoothstep(0.3, 0.7, s)
        if m > 0.0:
            eu = 1.5 / float(solid.shape[1])
            ev = 1.5 / float(solid.shape[0])
            gx = sample2(solid, u + eu, v, nearest) - sample2(solid, u - eu, v, nearest)
            gy = sample2(solid, u, v + ev, nearest) - sample2(solid, u, v - ev, nearest)
            nrm = -wp.vec2(gx, gy)
            edge = wp.max(wp.dot(nrm, p.light_dir), 0.0)
            ob = p.obst_col * (0.75 + 0.5 * v) + p.key_col * (0.35 * edge) + wp.cw_mul(p.obst_col, glow) * (3.0 * p.fire_light)
            col = col * (1.0 - m) + ob * m

    # --- tone map + sRGB + tiny dither (hides gradient banding)
    col = col * p.exposure
    h = wp.sin(float(r) * 12.9898 + float(c) * 78.233) * 43758.5453
    dither = (h - wp.floor(h)) - 0.5
    out[r, c, 0] = to_srgb8(aces(col[0]), dither)
    out[r, c, 1] = to_srgb8(aces(col[1]), dither)
    out[r, c, 2] = to_srgb8(aces(col[2]), dither)
