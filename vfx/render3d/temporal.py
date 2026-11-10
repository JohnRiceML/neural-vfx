"""Temporal amortization of the ray-march (shortlist #2, opt-in: VolumeRenderer(..., temporal=True)).

Per draw: the march runs with a per-frame offset added to the interleaved-gradient ray-start jitter
(cam[14], golden-ratio sequence), so successive frames sample different depths along each ray. The
resolve kernel (march resolution) then blends the new march buffer into a history buffer that is
reprojected by CAMERA MOTION ONLY, at the opacity-weighted mean depth of the medium (the aux depth),
the floor-plane hit for floor pixels, and the domain centre distance for empty pixels. The gas's own
velocity is never used for reprojection (measured worse than no warp; see reports/...).

History rectification that respects transparency:
  * 3x3 neighbourhood of the current frame in YCoCg (radiance) and in transmittance T, variance clip
    (mean +- gamma * sigma, intersected with min/max) — T is clipped like a colour channel, so a
    history pixel that was smoke where there is now none (or the reverse) is pulled back;
  * the current-frame weight rises where the medium alpha changes fast (|T_cur - T_hist| large) and
    where the neighbourhood itself is high-contrast in T (thin, moving edges);
  * full reset on disocclusion (reprojected point off screen / behind the previous camera), on the
    first frame, and after resize / parameter changes.
Buffers are (rgb premultiplied radiance, T) at march res, like the march output, so bloom and the
composite run unchanged on the resolved buffer.
"""
from __future__ import annotations

import warp as wp

from vfx.render3d.kernels import box_hit, safe_inv


@wp.func
def rgb2ycocg(c: wp.vec3) -> wp.vec3:
    return wp.vec3(0.25 * c[0] + 0.5 * c[1] + 0.25 * c[2], 0.5 * c[0] - 0.5 * c[2], -0.25 * c[0] + 0.5 * c[1] - 0.25 * c[2])


@wp.func
def ycocg2rgb(c: wp.vec3) -> wp.vec3:
    return wp.vec3(c[0] + c[1] - c[2], c[0] + c[2], c[0] - c[1] - c[2])


@wp.func
def to_c4(v: wp.vec4) -> wp.vec4:
    """(rgb, T) -> (Y, Co, Cg, T), with a soft tone curve on luma so bright fire does not dominate."""
    y = rgb2ycocg(wp.vec3(v[0], v[1], v[2]))
    s = 1.0 / (1.0 + wp.max(y[0], 0.0))
    return wp.vec4(y[0] * s, y[1] * s, y[2] * s, v[3])


@wp.func
def from_c4(c: wp.vec4) -> wp.vec4:
    s = 1.0 / wp.max(1.0 - c[0], 1.0e-3)
    rgb = ycocg2rgb(wp.vec3(c[0] * s, c[1] * s, c[2] * s))
    return wp.vec4(rgb[0], rgb[1], rgb[2], c[3])


@wp.func
def bilin4(a: wp.array2d(dtype=wp.vec4), x: float, y: float) -> wp.vec4:
    """Bilinear sample at continuous pixel coords (pixel centres at i + 0.5), clamped."""
    H = a.shape[0]
    W = a.shape[1]
    fx = x - 0.5
    fy = y - 0.5
    x0f = wp.floor(fx)
    y0f = wp.floor(fy)
    tx = fx - x0f
    ty = fy - y0f
    x0 = wp.clamp(int(x0f), 0, W - 1)
    y0 = wp.clamp(int(y0f), 0, H - 1)
    x1 = wp.clamp(int(x0f) + 1, 0, W - 1)
    y1 = wp.clamp(int(y0f) + 1, 0, H - 1)
    return (a[y0, x0] * (1.0 - tx) + a[y0, x1] * tx) * (1.0 - ty) + (a[y1, x0] * (1.0 - tx) + a[y1, x1] * tx) * ty


@wp.kernel
def temporal_resolve(cur: wp.array2d(dtype=wp.vec4), aux: wp.array2d(dtype=wp.vec4),
                     hist: wp.array2d(dtype=wp.vec4), cam: wp.array(dtype=float), tcam: wp.array(dtype=float),
                     out: wp.array2d(dtype=wp.vec4), alpha_base: float, gamma: float, k_t: float):
    """cur: this frame's march buffer; aux: (alpha, alpha*depth, ...) premultiplied from the same march;
    hist: last resolved buffer; cam: current camera basis (16); tcam: previous draw's camera basis
    (16; tcam[15] = history valid flag). out: resolved (rgb, T) = the new history."""
    r, c = wp.tid()
    H = cur.shape[0]
    W = cur.shape[1]
    now = cur[r, c]
    if tcam[15] < 0.5:
        out[r, c] = now
        return
    o = wp.vec3(cam[0], cam[1], cam[2])
    fw = wp.vec3(cam[3], cam[4], cam[5])
    rt = wp.vec3(cam[6], cam[7], cam[8])
    up = wp.vec3(cam[9], cam[10], cam[11])
    th = cam[12]
    aspect = cam[13]
    sx = (2.0 * (float(c) + 0.5) / float(W) - 1.0) * th * aspect
    sy = (1.0 - 2.0 * (float(r) + 0.5) / float(H)) * th
    d = wp.normalize(fw + rt * sx + up * sy)
    # representative depth: medium (opacity-weighted) > floor plane > domain centre
    ax = aux[r, c]
    dep = float(0.0)
    if ax[0] > 0.02:
        dep = ax[1] / ax[0]
    elif d[1] < -1.0e-4 and o[1] > 0.0:
        dep = -o[1] / d[1]
    else:
        dep = wp.length(wp.vec3(0.5, 0.5, 0.5) - o)
    q = o + d * dep
    # project into the previous camera (march-res pixels)
    po = wp.vec3(tcam[0], tcam[1], tcam[2])
    rel = q - po
    z = wp.dot(rel, wp.vec3(tcam[3], tcam[4], tcam[5]))
    valid = z > 1.0e-3
    px = float(0.0)
    py = float(0.0)
    if valid:
        psx = wp.dot(rel, wp.vec3(tcam[6], tcam[7], tcam[8])) / (z * tcam[12] * tcam[13])
        psy = wp.dot(rel, wp.vec3(tcam[9], tcam[10], tcam[11])) / (z * tcam[12])
        px = (psx + 1.0) * 0.5 * float(W)
        py = (1.0 - psy) * 0.5 * float(H)
        valid = px >= 0.0 and px <= float(W) and py >= 0.0 and py <= float(H)
    if not valid:
        out[r, c] = now                 # disocclusion / off-screen: reset
        return
    hp = to_c4(bilin4(hist, px, py))
    # neighbourhood statistics of the current frame (variance clip in YCoCg + T)
    m1 = wp.vec4(0.0, 0.0, 0.0, 0.0)
    m2 = wp.vec4(0.0, 0.0, 0.0, 0.0)
    lo = wp.vec4(1.0e9, 1.0e9, 1.0e9, 1.0e9)
    hi = wp.vec4(-1.0e9, -1.0e9, -1.0e9, -1.0e9)
    for j in range(-1, 2):
        for i in range(-1, 2):
            s = to_c4(cur[wp.clamp(r + j, 0, H - 1), wp.clamp(c + i, 0, W - 1)])
            m1 = m1 + s
            m2 = m2 + wp.cw_mul(s, s)
            lo = wp.vec4(wp.min(lo[0], s[0]), wp.min(lo[1], s[1]), wp.min(lo[2], s[2]), wp.min(lo[3], s[3]))
            hi = wp.vec4(wp.max(hi[0], s[0]), wp.max(hi[1], s[1]), wp.max(hi[2], s[2]), wp.max(hi[3], s[3]))
    m1 = m1 / 9.0
    m2 = m2 / 9.0
    cl = wp.vec4(0.0, 0.0, 0.0, 0.0)
    for k in range(4):
        sig = wp.sqrt(wp.max(m2[k] - m1[k] * m1[k], 0.0))
        a = wp.max(lo[k], m1[k] - gamma * sig)
        b = wp.min(hi[k], m1[k] + gamma * sig)
        cl[k] = wp.clamp(hp[k], a, b)
    nc = to_c4(now)
    # transparency-aware weight: trust history less where the medium's T changed or the T
    # neighbourhood is high-contrast (thin moving edges), and where the clip had to move history far
    dT = wp.abs(nc[3] - hp[3])
    w = wp.clamp(alpha_base + k_t * dT + 0.5 * k_t * (hi[3] - lo[3]) * dT, alpha_base, 1.0)
    res = cl * (1.0 - w) + nc * w
    out[r, c] = from_c4(res)
