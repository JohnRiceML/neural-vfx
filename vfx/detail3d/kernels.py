"""Warp kernels of the advected-detail stage + the sampling function the ray-marcher calls.

Grid: the detail grid covers the unit domain cube like the sim (cell-centred, res^3, index [z, y, x]).
Coordinate fields are stored as DISPLACEMENTS D = R - x of the rest (noise) coordinate R, so the
identity part is exact and only the (small, smooth) deformation is interpolated.
"""
from __future__ import annotations

import warp as wp

wp.config.quiet = True


@wp.struct
class DetailShade:
    disp: float          # displacement amplitude at octave 0, meters (x per-cell amplitude 0..1)
    disp_gain: float     # per-octave factor of the displacement (0.5 = proportional to wavelength)
    mult: float          # multiplicative density detail amplitude at octave 0 (0 = off)
    mult_gain: float
    octaves: int         # noise taps per coordinate set (quality)
    freq0: float         # noise tiles per meter at octave 0
    lac: float           # lacunarity (frequency factor between octaves)
    amp_eps: float       # below this per-cell amplitude the sample is left untouched (cost)
    rot: wp.mat33        # rotation applied to the rest coordinate between octaves


# ---------------------------------------------------------------------- sampling (render side)
@wp.func
def detail_warp(q: wp.vec3, dc0: wp.Texture3D, dc1: wp.Texture3D, noise: wp.Texture3D, ds: DetailShade,
                dyn: wp.array(dtype=float)) -> wp.vec4:
    """World sample point q -> (displaced point xyz, density multiplier). dyn = [w0, w1, off0 xyz,
    off1 xyz] written once per sim tick (variance-normalized phase weights + regeneration offsets)."""
    c0 = wp.texture_sample(dc0, q, dtype=wp.vec4f)
    amp = c0[3]
    if amp <= ds.amp_eps:
        return wp.vec4(q[0], q[1], q[2], 1.0)
    c1 = wp.texture_sample(dc1, q, dtype=wp.vec4f)
    w0 = dyn[0]
    w1 = dyn[1]
    p0 = (q + wp.vec3(c0[0], c0[1], c0[2]) + wp.vec3(dyn[2], dyn[3], dyn[4])) * ds.freq0
    p1 = (q + wp.vec3(c1[0], c1[1], c1[2]) + wp.vec3(dyn[5], dyn[6], dyn[7])) * ds.freq0
    disp = wp.vec3(0.0, 0.0, 0.0)
    m = float(0.0)
    ad = ds.disp
    am = ds.mult
    for _o in range(ds.octaves):
        n = wp.vec4(0.0, 0.0, 0.0, 0.0)
        # wrap explicitly at the tap: the texture unit returned 0 for normalized coords of a few hundred
        # tiles (octaves 3-4 silently vanished); coordinates stay small anyway (offsets < 1 m)
        if w0 > 0.0:
            n = n + wp.texture_sample(noise, wp.vec3(wp.frac(p0[0]), wp.frac(p0[1]), wp.frac(p0[2])), dtype=wp.vec4f) * w0
        if w1 > 0.0:
            n = n + wp.texture_sample(noise, wp.vec3(wp.frac(p1[0]), wp.frac(p1[1]), wp.frac(p1[2])), dtype=wp.vec4f) * w1
        disp = disp + wp.vec3(n[0], n[1], n[2]) * ad
        m = m + n[3] * am
        p0 = (ds.rot * p0) * ds.lac          # continuous (no wrap here: rot of a wrapped coord has seams)
        p1 = (ds.rot * p1) * ds.lac
        ad = ad * ds.disp_gain
        am = am * ds.mult_gain
    qd = q + disp * amp
    return wp.vec4(qd[0], qd[1], qd[2], wp.max(1.0 + m * amp, 0.0))


@wp.func
def detail_sample(vol: wp.Texture3D, q: wp.vec3, dc0: wp.Texture3D, dc1: wp.Texture3D, noise: wp.Texture3D,
                  ds: DetailShade, dyn: wp.array(dtype=float)) -> wp.vec2:
    """(density, temperature) of the detailed field at q: both read at the displaced point (flames and
    smoke move together: tongues, no colour shift), density times the multiplicative factor."""
    w = detail_warp(q, dc0, dc1, noise, ds, dyn)
    s = wp.texture_sample(vol, wp.vec3(w[0], w[1], w[2]), dtype=wp.vec2f)
    return wp.vec2(s[0] * w[3], s[1])


@wp.kernel
def dilate_occ(src: wp.array3d(dtype=wp.uint8), dst: wp.array3d(dtype=wp.uint8)):
    """Occupancy bricks dilated by one brick: displaced samples may read smoke up to ~1 brick away."""
    z, y, x = wp.tid()
    n = src.shape[0]
    v = wp.uint8(0)
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                zz = wp.clamp(z + dz, 0, n - 1)
                yy = wp.clamp(y + dy, 0, n - 1)
                xx = wp.clamp(x + dx, 0, n - 1)
                if src[zz, yy, xx] != wp.uint8(0):
                    v = wp.uint8(1)
    dst[z, y, x] = v


# ---------------------------------------------------------------------- per-tick (sim side)
@wp.func
def trilin3(f: wp.array4d(dtype=float), c0: int, p: wp.vec3) -> wp.vec3:
    """Trilinear sample of channels c0..c0+2 of a (C, D, H, W) cell-centred field at world point p."""
    D = f.shape[1]
    H = f.shape[2]
    W = f.shape[3]
    gx = wp.clamp(p[0] * float(W) - 0.5, 0.0, float(W - 1))
    gy = wp.clamp(p[1] * float(H) - 0.5, 0.0, float(H - 1))
    gz = wp.clamp(p[2] * float(D) - 0.5, 0.0, float(D - 1))
    x0 = wp.min(int(gx), W - 2)
    y0 = wp.min(int(gy), H - 2)
    z0 = wp.min(int(gz), D - 2)
    tx = gx - float(x0)
    ty = gy - float(y0)
    tz = gz - float(z0)
    out = wp.vec3(0.0, 0.0, 0.0)
    for c in range(3):
        a00 = f[c0 + c, z0, y0, x0] * (1.0 - tx) + f[c0 + c, z0, y0, x0 + 1] * tx
        a01 = f[c0 + c, z0, y0 + 1, x0] * (1.0 - tx) + f[c0 + c, z0, y0 + 1, x0 + 1] * tx
        a10 = f[c0 + c, z0 + 1, y0, x0] * (1.0 - tx) + f[c0 + c, z0 + 1, y0, x0 + 1] * tx
        a11 = f[c0 + c, z0 + 1, y0 + 1, x0] * (1.0 - tx) + f[c0 + c, z0 + 1, y0 + 1, x0 + 1] * tx
        out[c] = (a00 * (1.0 - ty) + a01 * ty) * (1.0 - tz) + (a10 * (1.0 - ty) + a11 * ty) * tz
    return out


@wp.kernel
def advect_coords(data: wp.array4d(dtype=float), vc: int, old0: wp.Texture3D, old1: wp.Texture3D,
                  amp: wp.array3d(dtype=float), dt: float, reset: wp.array(dtype=int),
                  new0: wp.array3d(dtype=wp.vec4), new1: wp.array3d(dtype=wp.vec4), max_disp: float):
    """Semi-Lagrangian (RK2 backtrace) transport of both displacement sets by the sim velocity
    (channels vc..vc+2 of `data`, m/s, any resolution):  D_new(x) = D_old(x_b) + (x_b - x).
    A set whose reset flag is 1 restarts at D = 0 (its blend weight is 0 at that tick). The per-cell
    amplitude goes into set 0's w channel. |D| is capped (a runaway map is regenerated anyway)."""
    z, y, x = wp.tid()
    n = new0.shape[0]
    inv = 1.0 / float(n)
    p = wp.vec3((float(x) + 0.5) * inv, (float(y) + 0.5) * inv, (float(z) + 0.5) * inv)
    u1 = trilin3(data, vc, p)
    pm = p - u1 * (0.5 * dt)
    u2 = trilin3(data, vc, pm)
    pb = p - u2 * dt
    pb = wp.vec3(wp.clamp(pb[0], 0.0, 1.0), wp.clamp(pb[1], 0.0, 1.0), wp.clamp(pb[2], 0.0, 1.0))
    back = pb - p
    a = amp[z, y, x]
    d0 = wp.vec3(0.0, 0.0, 0.0)
    if reset[0] == 0:
        s = wp.texture_sample(old0, pb, dtype=wp.vec4f)
        d0 = wp.vec3(s[0], s[1], s[2]) + back
        l0 = wp.length(d0)
        if l0 > max_disp:
            d0 = d0 * (max_disp / l0)
    d1 = wp.vec3(0.0, 0.0, 0.0)
    if reset[1] == 0:
        s = wp.texture_sample(old1, pb, dtype=wp.vec4f)
        d1 = wp.vec3(s[0], s[1], s[2]) + back
        l1 = wp.length(d1)
        if l1 > max_disp:
            d1 = d1 * (max_disp / l1)
    new0[z, y, x] = wp.vec4(d0[0], d0[1], d0[2], a)
    new1[z, y, x] = wp.vec4(d1[0], d1[1], d1[2], 0.0)


@wp.kernel
def energy_amp(data: wp.array4d(dtype=float), solid: wp.array3d(dtype=float), ci_d: int, ci_v: int, ci_t: int,
               w_vort: float, w_heat: float, heat_damp: float, t_amb: float, d_lo: float, d_hi: float,
               out: wp.array3d(dtype=float)):
    """Raw per-cell detail amplitude, before smoothing:
       energy  = |curl u| * dx  (m/s: velocity difference across one cell = the resolved small-scale
                 motion; the band the 64^3 grid can just represent, extrapolated to finer bands with a
                 Kolmogorov falloff by the per-octave gains)
       a       = 1 - exp(-w_vort * energy - w_heat * heat)        (saturating, 0..1)
       x  smoothstep(d_lo, d_hi, max density over the 3^3 neighbourhood) or heat (no detail in empty space)
       x  (1 - heat_damp * local heat): the flame core stays coherent, tongues form at its edge
       x  0 in and next to obstacles."""
    z, y, x = wp.tid()
    D = data.shape[1]
    H = data.shape[2]
    W = data.shape[3]
    zm = wp.max(z - 1, 0)
    zp = wp.min(z + 1, D - 1)
    ym = wp.max(y - 1, 0)
    yp = wp.min(y + 1, H - 1)
    xm = wp.max(x - 1, 0)
    xp = wp.min(x + 1, W - 1)
    vx = ci_v
    vy = ci_v + 1
    vz = ci_v + 2
    # central differences in cells (x dx cancels against the /dx of the derivative)
    dwz_dy = data[vz, z, yp, x] - data[vz, z, ym, x]
    dwy_dz = data[vy, zp, y, x] - data[vy, zm, y, x]
    dwx_dz = data[vx, zp, y, x] - data[vx, zm, y, x]
    dwz_dx = data[vz, z, y, xp] - data[vz, z, y, xm]
    dwy_dx = data[vy, z, y, xp] - data[vy, z, y, xm]
    dwx_dy = data[vx, z, yp, x] - data[vx, z, ym, x]
    e = 0.5 * wp.length(wp.vec3(dwz_dy - dwy_dz, dwx_dz - dwz_dx, dwy_dx - dwx_dy))
    dmax = float(0.0)
    smax = float(0.0)
    hmax = float(0.0)
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                zz = wp.clamp(z + dz, 0, D - 1)
                yy = wp.clamp(y + dy, 0, H - 1)
                xx = wp.clamp(x + dx, 0, W - 1)
                dmax = wp.max(dmax, data[ci_d, zz, yy, xx])
                hmax = wp.max(hmax, data[ci_t, zz, yy, xx] - t_amb)
                smax = wp.max(smax, solid[zz, yy, xx])
    heat = wp.clamp(hmax / 1000.0, 0.0, 1.0)
    a = 1.0 - wp.exp(-w_vort * e - w_heat * heat)
    t = wp.clamp((dmax - d_lo) / (d_hi - d_lo), 0.0, 1.0)
    pres = wp.max(t * t * (3.0 - 2.0 * t), wp.clamp(heat * 4.0, 0.0, 1.0))
    core = wp.clamp((data[ci_t, z, y, x] - t_amb) / 1000.0, 0.0, 1.0)
    out[z, y, x] = a * pres * (1.0 - smax) * (1.0 - heat_damp * core)


@wp.kernel
def eval_detail_grid(vol: wp.Texture3D, dc0: wp.Texture3D, dc1: wp.Texture3D, noise: wp.Texture3D,
                     ds: DetailShade, dyn: wp.array(dtype=float), use_detail: int,
                     out_d: wp.array3d(dtype=float), out_t: wp.array3d(dtype=float)):
    """Analysis only (never in the frame loop): the detailed field point-sampled on an N^3 grid."""
    z, y, x = wp.tid()
    n = out_d.shape[0]
    inv = 1.0 / float(n)
    q = wp.vec3((float(x) + 0.5) * inv, (float(y) + 0.5) * inv, (float(z) + 0.5) * inv)
    s = wp.texture_sample(vol, q, dtype=wp.vec2f)
    if use_detail != 0:
        s = detail_sample(vol, q, dc0, dc1, noise, ds, dyn)
    out_d[z, y, x] = s[0]
    out_t[z, y, x] = s[1]
