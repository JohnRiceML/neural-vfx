"""Fused HR-resolution stages of VolumeNet for active bricks, as NVIDIA Warp kernels (one thread per HR voxel).

Profiling the pure-PyTorch brick path showed ~50% of GPU time in strided copies (3D pixel (un)shuffle
permutes, layout conversions, concat) and elementwise normalisation at HR resolution, ~25% in convs.
A game engine would write these stages as compute shaders; these kernels are that, and are captured in
the same CUDA graph as the torch convs (launched on torch's current stream).

  prev_unshuffle_kernel : sparse semi-Lagrangian warp of prev HR (velocity = trilinear LR velocity of
                          frame t-1), normalise + clip, written directly in pixel_unshuffle3d channel order
                          into a channels-last (n*b^3, 2*s^3) buffer -> input of VolumeNet.prev_in.
  hr_head_kernel        : pixel_shuffle3d of the tail output, trilinear base, HR head (two 1x1x1 convs),
                          residual add, clamp, denorm, scatter into the HR canvas.
Math == VolumeNet.forward with hr_kernel=1 (fp32 inside the kernels; checked by selfcheck_fused()).
"""
from __future__ import annotations

import torch
import warp as wp

wp.config.quiet = True
wp.init()


@wp.func
def _clampi(i: int, n: int):
    return wp.max(0, wp.min(i, n - 1))


@wp.func
def _tri(a: wp.array(dtype=wp.float16, ndim=4), c: int, pz: float, py: float, px: float,
         D: int, H: int, W: int):
    """Trilinear sample of channel c at fractional index position, border clamp (== grid_sample border,
    align_corners=False, with p = voxel-centre coordinate in index units)."""
    pz = wp.clamp(pz, 0.0, float(D - 1))
    py = wp.clamp(py, 0.0, float(H - 1))
    px = wp.clamp(px, 0.0, float(W - 1))
    z0 = int(wp.floor(pz))
    y0 = int(wp.floor(py))
    x0 = int(wp.floor(px))
    fz = pz - float(z0)
    fy = py - float(y0)
    fx = px - float(x0)
    z1 = _clampi(z0 + 1, D)
    y1 = _clampi(y0 + 1, H)
    x1 = _clampi(x0 + 1, W)
    c00 = float(a[c, z0, y0, x0]) * (1.0 - fx) + float(a[c, z0, y0, x1]) * fx
    c01 = float(a[c, z0, y1, x0]) * (1.0 - fx) + float(a[c, z0, y1, x1]) * fx
    c10 = float(a[c, z1, y0, x0]) * (1.0 - fx) + float(a[c, z1, y0, x1]) * fx
    c11 = float(a[c, z1, y1, x0]) * (1.0 - fx) + float(a[c, z1, y1, x1]) * fx
    c0 = c00 * (1.0 - fy) + c01 * fy
    c1 = c10 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


@wp.kernel
def prev_unshuffle_kernel(ids: wp.array(dtype=wp.int32), samp: wp.array(dtype=wp.float16, ndim=4),
                          prev: wp.array(dtype=wp.float16, ndim=4), P: wp.array(dtype=wp.float16, ndim=2),
                          nb: int, b: int, s: int, R: int, N: int, vscale: wp.array(dtype=float),
                          t_off: float, inv_d: float, inv_t: float, clip: float):
    t = wp.tid()
    sb = b * s
    sb3 = sb * sb * sb
    i = t // sb3
    r = t - i * sb3
    lz = r // (sb * sb)
    ly = (r // sb) % sb
    lx = r % sb
    bid = ids[i]
    gz = (bid // (nb * nb)) * sb + lz
    gy = ((bid // nb) % nb) * sb + ly
    gx = (bid % nb) * sb + lx
    inv_s = 1.0 / float(s)
    pz = (float(gz) + 0.5) * inv_s - 0.5
    py = (float(gy) + 0.5) * inv_s - 0.5
    px = (float(gx) + 0.5) * inv_s - 0.5
    vx = _tri(samp, 2, pz, py, px, R, R, R)
    vy = _tri(samp, 3, pz, py, px, R, R, R)
    vz = _tri(samp, 4, pz, py, px, R, R, R)
    vs = vscale[0]
    qz = float(gz) - vz * vs
    qy = float(gy) - vy * vs
    qx = float(gx) - vx * vs
    d = _tri(prev, 0, qz, qy, qx, N, N, N)
    T = _tri(prev, 1, qz, qy, qx, N, N, N)
    cell = i * b * b * b + (lz // s) * b * b + (ly // s) * b + (lx // s)
    sub = (lz % s) * s * s + (ly % s) * s + (lx % s)
    s3 = s * s * s
    P[cell, sub] = wp.float16(wp.clamp(d * inv_d, -clip, clip))
    P[cell, s3 + sub] = wp.float16(wp.clamp((T - t_off) * inv_t, -clip, clip))


@wp.kernel
def hr_head_kernel(ids: wp.array(dtype=wp.int32), samp: wp.array(dtype=wp.float16, ndim=4),
                   tail: wp.array(dtype=wp.float16, ndim=2), P: wp.array(dtype=wp.float16, ndim=2),
                   w1: wp.array(dtype=float, ndim=2), b1: wp.array(dtype=float),
                   w2: wp.array(dtype=float, ndim=2), b2: wp.array(dtype=float),
                   out: wp.array(dtype=wp.float16, ndim=4),
                   nb: int, b: int, s: int, R: int, nf: int, nh: int,
                   t_off: float, sc_d: float, sc_t: float, clip: float, out_clip: float):
    t = wp.tid()
    sb = b * s
    sb3 = sb * sb * sb
    i = t // sb3
    r = t - i * sb3
    lz = r // (sb * sb)
    ly = (r // sb) % sb
    lx = r % sb
    bid = ids[i]
    gz = (bid // (nb * nb)) * sb + lz
    gy = ((bid // nb) % nb) * sb + ly
    gx = (bid % nb) * sb + lx
    inv_s = 1.0 / float(s)
    pz = (float(gz) + 0.5) * inv_s - 0.5
    py = (float(gy) + 0.5) * inv_s - 0.5
    px = (float(gx) + 0.5) * inv_s - 0.5
    bd = _tri(samp, 0, pz, py, px, R, R, R) / sc_d
    bt = (_tri(samp, 1, pz, py, px, R, R, R) - t_off) / sc_t
    cell = i * b * b * b + (lz // s) * b * b + (ly // s) * b + (lx // s)
    sub = (lz % s) * s * s + (ly % s) * s + (lx % s)
    s3 = s * s * s
    pd = float(P[cell, sub])
    pt = float(P[cell, s3 + sub])
    bdc = wp.clamp(bd, -clip, clip)
    btc = wp.clamp(bt, -clip, clip)
    y0 = b2[0]
    y1 = b2[1]
    for h in range(nh):
        acc = b1[h]
        for f in range(nf):
            acc += w1[h, f] * wp.max(float(tail[cell, f * s3 + sub]), 0.0)
        acc += w1[h, nf] * pd + w1[h, nf + 1] * pt + w1[h, nf + 2] * bdc + w1[h, nf + 3] * btc
        acc = wp.max(acc, 0.0)
        y0 += w2[0, h] * acc
        y1 += w2[1, h] * acc
    y0 = wp.clamp(bd + y0, 0.0, out_clip)
    y1 = wp.clamp(bt + y1, 0.0, out_clip)
    out[0, gz, gy, gx] = wp.float16(y0 * sc_d)
    out[1, gz, gy, gx] = wp.float16(y1 * sc_t + t_off)


def _wp(t: torch.Tensor):
    return wp.from_torch(t)


def _stream():
    return wp.stream_from_torch(torch.cuda.current_stream())


class FusedHR:
    """Holds the HR-head weights as fp32 Warp arrays and launches the two kernels."""

    def __init__(self, model, device):
        cfg = model.cfg
        if cfg.hr_kernel != 1:
            raise ValueError("fused HR path needs hr_kernel=1")
        self.cfg = cfg
        device = torch.device(device)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.dev = wp.device_from_torch(device)
        f32 = lambda t: t.detach().float().reshape(t.shape[0], -1).contiguous()   # noqa: E731
        self._w = [f32(model.hr1.weight), model.hr1.bias.detach().float().contiguous(),
                   f32(model.hr2.weight), model.hr2.bias.detach().float().contiguous()]
        self.w = [_wp(t) for t in self._w]

    def refresh(self, model) -> None:
        """Re-read the HR-head weights from `model` IN PLACE (captured CUDA graphs keep these buffers)."""
        src = [model.hr1.weight, model.hr1.bias, model.hr2.weight, model.hr2.bias]
        for dst, w in zip(self._w, src):
            dst.copy_(w.detach().float().reshape(dst.shape))

    def prev_unshuffle(self, ids32, samp, prev, P2d, nb, b, R, N, vscale):
        """vscale: (1,) float32 cuda tensor = dt / dx_hr (on the device, so CUDA graphs do not bake dt in)."""
        c = self.cfg
        if not torch.is_tensor(vscale):
            vscale = torch.full((1,), float(vscale), device=ids32.device, dtype=torch.float32)
        wp.launch(prev_unshuffle_kernel, dim=ids32.shape[0] * (b * c.scale) ** 3,
                  inputs=[_wp(ids32), _wp(samp), _wp(prev), _wp(P2d), nb, b, c.scale, R, N, _wp(vscale),
                          c.temp_offset, 1.0 / c.density_scale, 1.0 / c.temp_scale, c.input_clip],
                  device=self.dev, stream=_stream())

    def head(self, ids32, samp, tail2d, P2d, out, nb, b, R):
        c = self.cfg
        wp.launch(hr_head_kernel, dim=ids32.shape[0] * (b * c.scale) ** 3,
                  inputs=[_wp(ids32), _wp(samp), _wp(tail2d), _wp(P2d), *self.w, _wp(out), nb, b, c.scale, R,
                          c.hr_feats, c.hr_hidden, c.temp_offset, c.density_scale, c.temp_scale,
                          c.input_clip, c.output_clip],
                  device=self.dev, stream=_stream())
