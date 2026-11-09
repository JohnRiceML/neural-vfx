"""2D smoke + fire renderer: HRFields (any resolution) -> (out_h, out_w, 3) uint8 on cuda.

Pipeline per draw (all GPU, no host sync):
  1. torch, at a small working resolution (256^2): optical depth toward the light by marching
     N density samples (grid_sample, one batched call), blackbody emission -> 2-level blurred bloom.
  2. Warp, at output resolution: one fused kernel (kernels.composite) that samples the input
     fields, shades smoke/fire/obstacles over a gradient background, tone-maps and writes uint8.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import warp as wp

from vfx.core.fields import HRFields
from vfx.render import kernels as K
from vfx.render.blackbody import blackbody_lut


@dataclass
class RenderParams:
    k_abs: float = 2.2
    k_shadow: float = 9.0
    k_emit: float = 4.0
    emit_min: float = 0.45
    emit_scale: float = 1.6
    bloom_scale: float = 0.55
    fire_light: float = 0.6
    rim: float = 0.35
    exposure: float = 1.0
    t_lo: float = 700.0
    t_hi: float = 3000.0
    t_ref: float = 1400.0                     # LUT luminance 1.0 at this temperature
    light_dir: tuple = (-0.45, 0.89)          # toward the light, y up (upper-left key)
    key_col: tuple = (0.70, 0.64, 0.56)
    amb_col: tuple = (0.10, 0.13, 0.19)
    rim_col: tuple = (0.55, 0.62, 0.75)
    smoke_albedo: tuple = (0.62, 0.62, 0.66)
    bg_top: tuple = (0.030, 0.036, 0.052)
    bg_bot: tuple = (0.006, 0.006, 0.010)
    obst_col: tuple = (0.030, 0.031, 0.036)
    shadow_steps: int = 24
    shadow_step_m: float = 0.011              # world meters per march step
    shadow_solid: float = 3.0                 # obstacles also cast shadow (density-equivalent)
    work_res: int = 256


def _resize(x: torch.Tensor, n: int) -> torch.Tensor:
    """(H,W) -> (1,1,n,n): area-downsample or bilinear-upsample."""
    x = x[None, None]
    if x.shape[-1] == n and x.shape[-2] == n:
        return x
    if x.shape[-1] > n:
        if x.shape[-1] % n == 0 and x.shape[-2] % n == 0:
            return F.avg_pool2d(x, (x.shape[-2] // n, x.shape[-1] // n))
        return F.interpolate(x, size=(n, n), mode="area")
    return F.interpolate(x, size=(n, n), mode="bilinear", align_corners=False)


def _gauss_kernel(sigma: float, device) -> torch.Tensor:
    r = int(math.ceil(2.5 * sigma))
    t = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


class Renderer:
    def __init__(self, out_size: tuple[int, int], device: str = "cuda", params: RenderParams | None = None,
                 cuda_graph: bool = True):
        self.out_h, self.out_w = int(out_size[0]), int(out_size[1])
        self.device = torch.device(device)
        self.p = params or RenderParams()
        wp.init()
        p = self.p
        lut = blackbody_lut(p.t_lo, p.t_hi, 256, t_ref=p.t_ref)
        self.lut_t = torch.from_numpy(lut).to(self.device)                     # (n,3)
        self.lut_wp = wp.from_torch(self.lut_t, dtype=wp.vec3)
        self.out = torch.empty(self.out_h, self.out_w, 3, dtype=torch.uint8, device=self.device)
        self.out_wp = wp.from_torch(self.out)
        self._zero_solid = torch.zeros(1, 1, device=self.device)
        self._shade = self._make_shade()
        n = p.work_res
        # march grid: (S, n, n, 2) normalized sample positions offset toward the light
        c = (torch.arange(n, device=self.device, dtype=torch.float32) * 2 + 1) / n - 1
        yy, xx = torch.meshgrid(c, c, indexing="ij")          # row 0 = bottom (y up) in both
        L = torch.tensor(p.light_dir, device=self.device, dtype=torch.float32)
        L = L / L.norm()
        ks = torch.arange(1, p.shadow_steps + 1, device=self.device, dtype=torch.float32)
        off = ks[:, None] * (2.0 * p.shadow_step_m) * L[None, :]             # domain size 1 m -> 2 normalized
        self._march = torch.stack([xx, yy], -1)[None] + off[:, None, None, :]
        self._g1 = _gauss_kernel(2.0, self.device)
        self._g2 = _gauss_kernel(2.0, self.device)
        self._wp_stream = None
        self._wp_stream_key = None
        self.cuda_graph = cuda_graph
        self._graphs: dict = {}

    def _make_shade(self) -> K.Shade:
        p, s = self.p, K.Shade()
        for name in ("k_abs", "k_shadow", "k_emit", "emit_min", "emit_scale", "bloom_scale",
                     "fire_light", "rim", "exposure", "t_lo", "t_hi"):
            setattr(s, name, float(getattr(p, name)))
        d = p.light_dir
        nrm = math.hypot(*d)
        s.light_dir = wp.vec2(d[0] / nrm, d[1] / nrm)
        for name in ("key_col", "amb_col", "rim_col", "smoke_albedo", "bg_top", "bg_bot", "obst_col"):
            setattr(s, name, wp.vec3(*getattr(p, name)))
        return s

    def set_params(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self.p, k, v)
        self._shade = self._make_shade()
        self._graphs.clear()                     # captured graphs baked the old parameters

    # ------------------------------------------------------------------ helpers
    def _blur(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Separable Gaussian blur of (1,C,H,W)."""
        C, r = x.shape[1], k.numel() // 2
        kx = k.view(1, 1, 1, -1).expand(C, 1, 1, -1)
        ky = k.view(1, 1, -1, 1).expand(C, 1, -1, 1)
        x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kx, groups=C)
        return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), ky, groups=C)

    def _emission(self, d: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """(1,1,n,n) density/temp -> (1,3,n,n) linear emission (same model as the kernel)."""
        p, n = self.p, self.lut_t.shape[0]
        f = ((T - p.t_lo) / (p.t_hi - p.t_lo)).clamp(0, 1) * (n - 1)
        i0 = f.long().clamp(max=n - 2)
        t = (f - i0)[0, 0, ..., None]
        e = self.lut_t[i0[0, 0]] * (1 - t) + self.lut_t[i0[0, 0] + 1] * t       # (n,n,3)
        emiss = p.emit_min + (1 - p.emit_min) * (1 - torch.exp(-p.k_emit * d.clamp(min=0)))
        return e.permute(2, 0, 1)[None] * (emiss * p.emit_scale)

    def _stream(self):
        ts = torch.cuda.current_stream(self.device)
        if self._wp_stream_key != ts.cuda_stream:
            self._wp_stream = wp.stream_from_torch(ts)
            self._wp_stream_key = ts.cuda_stream
        return self._wp_stream

    # ------------------------------------------------------------------ main entry
    @torch.no_grad()
    def draw(self, fields: HRFields, solid: torch.Tensor | None = None, nearest: bool = False) -> torch.Tensor:
        """Render to (out_h, out_w, 3) uint8 cuda, row 0 = top. Input grids are y-up, any size.

        nearest=True shows raw cells (used for the "LR raw" view); otherwise bilinear.
        The returned tensor is an internal buffer reused by the next draw() call.
        With cuda_graph=True the whole draw is one CUDA-graph replay per (input shape, solid shape,
        nearest) after a one-time capture: ~3 input copies + 1 launch instead of ~50 launches.
        """
        if not self.cuda_graph:
            return self._draw_impl(fields.density.float().contiguous(), fields.temperature.float().contiguous(),
                                   None if solid is None else solid.float().contiguous(), nearest)
        key = (tuple(fields.density.shape), None if solid is None else tuple(solid.shape), bool(nearest))
        g = self._graphs.get(key)
        if g is None:
            g = self._capture(key)
        sd, st, ss, graph = g
        sd.copy_(fields.density)
        st.copy_(fields.temperature)
        if ss is not None:
            ss.copy_(solid)
        graph.replay()
        return self.out

    def _capture(self, key):
        dshape, sshape, nearest = key
        sd = torch.zeros(dshape, device=self.device)
        st = torch.full(dshape, 293.0, device=self.device)
        ss = None if sshape is None else torch.zeros(sshape, device=self.device)
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            self._draw_impl(sd, st, ss, nearest)           # warm-up (allocations, kernel load)
        torch.cuda.current_stream(self.device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._draw_impl(sd, st, ss, nearest)
        self._graphs[key] = (sd, st, ss, graph)
        return self._graphs[key]

    def _draw_impl(self, dens, temp, solid, nearest: bool) -> torch.Tensor:
        p = self.p
        n = p.work_res
        dw = _resize(dens, n)
        tw = _resize(temp, n)
        occl = dw
        if solid is not None:
            occl = dw + p.shadow_solid * _resize(solid, n)
        S = p.shadow_steps
        tau = F.grid_sample(occl.expand(S, 1, n, n), self._march, mode="bilinear",
                            padding_mode="zeros", align_corners=False).sum(0, keepdim=True) * p.shadow_step_m
        tau = self._blur(tau, self._g1)[0, 0]          # soften march banding
        # bloom: emission at 128^2 (narrow) + 32^2 (wide)
        e = F.avg_pool2d(self._emission(dw, tw), 2)
        b1 = self._blur(e, self._g1)
        b2 = self._blur(F.avg_pool2d(b1, 4), self._g2)
        bloom = b1 + 1.5 * F.interpolate(b2, size=b1.shape[-2:], mode="bilinear", align_corners=False)
        bloom = bloom[0].permute(1, 2, 0).contiguous()

        with wp.ScopedStream(self._stream(), sync_enter=False, sync_exit=False):
            wp.launch(K.composite, dim=(self.out_h, self.out_w),
                      inputs=[wp.from_torch(dens), wp.from_torch(temp),
                              wp.from_torch(solid if solid is not None else self._zero_solid),
                              wp.from_torch(tau.contiguous()), wp.from_torch(bloom, dtype=wp.vec3),
                              self.lut_wp, self._shade, int(nearest), int(solid is not None),
                              self.out_wp])
        return self.out
