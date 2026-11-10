"""Route B runtime: ScreenDetail — 2D neural detail on the rendered frame, FP16, CUDA graphs, no host sync.

Per frame:
  1. pick the processed region: aux["bbox"] = (x0, y0, x1, y1) screen pixels of the effect (or the whole
     frame), grown per axis to the next of a few fixed sizes (`crop_fracs` of the frame side, rounded up to
     `quantum` px) and shifted to stay inside the frame. So at most len(crop_fracs)^2 CUDA graphs exist per
     frame size (36 by default, ~25 ms each), captured lazily or all at once by `warmup(H, W)`; an orbiting camera
     never triggers a capture after that.
  2. copy the region of frame / aux / previous output into static buffers (plain slicing, no sync).
  3. graph: uint8 -> fp16, warp the previous output by the screen velocity (aux, px/frame), optional
     downscale to `process_scale`, ScreenNet -> Δrgb, bilinear upsample of Δ, add, clamp, -> uint8.
  4. output = input frame with the processed region pasted in; it is the "previous output" next frame
     (two alternating output buffers, so no extra copy).
aux (all optional, (h,w) or (h,w,2) cuda tensors): "transmittance", "depth" (0..1), "velocity" (px/frame).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from vfx.model3d import AUX_CHANNELS, ScreenNet, ScreenNetConfig, load_screen_net, param_count
from vfx.runtime3d.volume_upscaler import capture_graph, cudnn_tuned

# the config we would train next (see README)
DEFAULT_ARCH = dict(width=64, n_blocks=4, trunk_stride=4, hr_feats=4, hr_hidden=8, hr_kernel=1,
                    use_aux=True, use_prev=True)
VEL_NORM = 16.0     # px/frame -> network units


class ScreenDetail:
    def __init__(self, checkpoint_path=None, device="cuda", *, process_scale: float = 0.5, quantum: int = 64,
                 crop_fracs=(0.25, 0.375, 0.5, 0.625, 0.75, 1.0), tune: bool = False, prewarm: bool = True, cuda_graph: bool = True,
                 channels_last: bool = True, **arch):
        self.crop_fracs = tuple(sorted(crop_fracs))
        self.tune = tune          # cudnn autotune per crop shape: ~1 s per capture and no faster, see README
        self.prewarm = prewarm    # capture every crop size on the first frame of a new frame size
        self.device = torch.device(device)
        if checkpoint_path:
            self.model, self.untrained = load_screen_net(checkpoint_path, device), False
        else:
            self.model = ScreenNet(ScreenNetConfig(**{**DEFAULT_ARCH, **arch})).to(self.device)
            self.untrained = True
        self.cl = channels_last
        fmt = torch.channels_last if channels_last else torch.contiguous_format
        self.model = self.model.half().to(memory_format=fmt).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.params = param_count(self.model)
        self.scale = process_scale
        r = self.model.cfg.trunk_stride
        self.quantum = max(quantum, int(round(r / process_scale)))
        while (self.quantum * process_scale) % r:
            self.quantum *= 2
        self.cuda_graph = cuda_graph and self.device.type == "cuda"
        self._pool = torch.cuda.graph_pool_handle() if self.cuda_graph else None
        self._graphs: dict = {}
        self._bufs: dict = {}
        self._store = None
        self._out = None
        self._ev = {k: torch.cuda.Event(enable_timing=True) for k in ("start", "net", "end")}
        self.reset()

    def reset(self) -> None:
        self._frame = 0
        self._cold = True
        self._last_region = None

    # ------------------------------------------------------------------ graph body
    def _body(self, key):
        rgb_u8, aux, prev_u8, out_u8 = self._bufs[key]
        ch, cw = key
        fmt = torch.channels_last if self.cl else torch.contiguous_format
        rgb = rgb_u8.permute(2, 0, 1).unsqueeze(0).half() * (1.0 / 255.0)
        k = int(round(1.0 / self.scale))
        pool = (lambda t: t) if k == 1 else (lambda t: F.avg_pool2d(t, k))     # downsample first
        x, xa = pool(rgb), pool(aux.unsqueeze(0))
        xp = None
        if self.model.cfg.use_prev:          # warp the previous output at the processed resolution
            xp = pool(prev_u8.permute(2, 0, 1).unsqueeze(0).half() * (1.0 / 255.0))
            h2, w2 = xp.shape[-2:]
            ys = (torch.arange(h2, device=self.device, dtype=torch.float32) * 2 + 1) / h2 - 1
            xs = (torch.arange(w2, device=self.device, dtype=torch.float32) * 2 + 1) / w2 - 1
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            vel = xa[0, 2:4].float() * (VEL_NORM / k)                 # px/frame at this res (x right, y down)
            grid = torch.stack([xx - vel[0] * (2.0 / w2), yy - vel[1] * (2.0 / h2)], -1)
            xp = F.grid_sample(xp, grid.unsqueeze(0).half(), mode="bilinear", padding_mode="border",
                               align_corners=False)
        cvt = (lambda t: None if t is None else t.contiguous(memory_format=fmt))
        delta = self.model(cvt(x), cvt(xa) if self.model.cfg.use_aux else None, cvt(xp))
        if self.scale != 1.0:
            delta = F.interpolate(delta, size=(ch, cw), mode="bilinear", align_corners=False)
        y = ((rgb + delta).clamp(0.0, 1.0) * 255.0 + 0.5)
        out_u8.copy_(y[0].permute(1, 2, 0).to(torch.uint8))
        return out_u8

    def _alloc(self, key):
        """Static graph buffers for a crop size = views into ONE shared max-size storage (only one crop
        size runs per frame), so memory does not grow with the number of crop sizes."""
        if key not in self._bufs:
            ch, cw = key
            n = ch * cw
            if self._store is None or self._store[0].numel() < 3 * n:
                if self._bufs:
                    raise RuntimeError("ScreenDetail: crop buffer outgrew its storage (captured graphs hold views)")
                z = dict(device=self.device)
                self._store = (torch.zeros(3 * n, dtype=torch.uint8, **z), torch.zeros(AUX_CHANNELS * n, dtype=torch.half, **z),
                               torch.zeros(3 * n, dtype=torch.uint8, **z), torch.zeros(3 * n, dtype=torch.uint8, **z))
            r, a, p, o = self._store
            self._bufs[key] = (r[:3 * n].view(ch, cw, 3), a[:AUX_CHANNELS * n].view(AUX_CHANNELS, ch, cw),
                               p[:3 * n].view(ch, cw, 3), o[:3 * n].view(ch, cw, 3))

    def _reset_size(self, H, W):
        """New frame size: drop graphs/buffers, size the shared storage for the largest crop."""
        self._graphs.clear()
        self._bufs.clear()
        self._store = None
        self._alloc((self.sizes(H)[-1], self.sizes(W)[-1]))

    def _run(self, key):
        if not self.cuda_graph:
            with torch.inference_mode(), cudnn_tuned(self.tune):
                return self._body(key)
        g = self._graphs.get(key)
        if g is None:
            g = self._graphs[key] = capture_graph(lambda: self._body(key), self.device, self._pool, warm=1,
                                                  tune=self.tune)
        g[0].replay()
        return g[1]

    def sizes(self, L: int) -> list[int]:
        """The fixed crop sizes (px) used along an axis of length L."""
        q = self.quantum
        return sorted({-(-max(int(L * f), 1) // q) * q for f in self.crop_fracs})

    def _region(self, H, W, bbox):
        if bbox is None:
            x0, y0, x1, y1 = 0, 0, W, H
        else:
            x0, y0, x1, y1 = (int(v) for v in bbox)
        def fit(a, b, L):
            a, b = max(a, 0), min(b, L)
            sz = self.sizes(L)
            n = next((v for v in sz if v >= b - a), sz[-1])
            if n >= L:
                return 0, n, L            # whole axis; buffer padded to n
            a = min(max(a - (n - (b - a)) // 2, 0), L - n)
            return a, n, n
        x0, cw, vw = fit(x0, x1, W)
        y0, ch, vh = fit(y0, y1, H)
        return y0, x0, ch, cw, vh, vw

    @torch.inference_mode()
    def infer(self, frame: torch.Tensor, aux: dict | None = None) -> torch.Tensor:
        """frame (h,w,3) uint8 cuda -> (h,w,3) uint8 cuda (frame + neural detail inside the effect bbox)."""
        aux = aux or {}
        ev = self._ev
        ev["start"].record()
        H, W = frame.shape[:2]
        if self._out is None or self._out[0].shape[:2] != (H, W):
            self._out = [torch.empty_like(frame) for _ in range(2)]
            self._cold = True
            self._reset_size(H, W)
            if self.prewarm:
                self._capture_all(H, W)
        y0, x0, ch, cw, vh, vw = self._region(H, W, aux.get("bbox"))
        self._last_region = (y0, x0, vh, vw)
        key = (ch, cw)
        out = self._out[self._frame % 2]
        prev_full = frame if self._cold else self._out[1 - self._frame % 2]
        out.copy_(frame)
        self._alloc(key)
        rgb_b, aux_b, prev_b, _ = self._bufs[key]
        if vh < ch or vw < cw:          # crop wider than the frame: zero the pad (storage is shared)
            for t in (rgb_b, prev_b):
                t[vh:].zero_()
                t[:, vw:].zero_()
            aux_b[:, vh:].zero_()
            aux_b[:, :, vw:].zero_()
        rgb_b[:vh, :vw].copy_(frame[y0:y0 + vh, x0:x0 + vw])
        prev_b[:vh, :vw].copy_(prev_full[y0:y0 + vh, x0:x0 + vw])
        for i, name in enumerate(("transmittance", "depth")):
            if name in aux:
                aux_b[i, :vh, :vw].copy_(aux[name][y0:y0 + vh, x0:x0 + vw])
        if "velocity" in aux:
            v = aux["velocity"][y0:y0 + vh, x0:x0 + vw]
            aux_b[2:4, :vh, :vw].copy_(v.permute(2, 0, 1) * (1.0 / VEL_NORM))
        res = self._run(key)
        ev["net"].record()
        out[y0:y0 + vh, x0:x0 + vw].copy_(res[:vh, :vw])
        self._cold = False
        self._frame += 1
        ev["end"].record()
        return out

    def warmup(self, H: int, W: int) -> None:
        """Capture the graphs of every crop size for an (H, W) frame up front (infer does this itself on
        the first frame of a new size when prewarm=True: ~0.5 s once for 36 sizes at 1440p)."""
        if not self.cuda_graph:
            return
        if self._out is None or self._out[0].shape[:2] != (H, W):
            self._out = [torch.zeros(H, W, 3, device=self.device, dtype=torch.uint8) for _ in range(2)]
            self._cold = True
            self._reset_size(H, W)
        self._capture_all(H, W)
        torch.cuda.synchronize(self.device)

    def _capture_all(self, H, W):
        for ch in self.sizes(H):
            for cw in self.sizes(W):
                if (ch, cw) not in self._graphs:
                    self._alloc((ch, cw))
                    self._run((ch, cw))

    @property
    def timings(self) -> dict:
        """ms of the last infer: net (copies in + graph), paste, total; processed fraction of the frame."""
        ev = self._ev
        ev["end"].synchronize()
        frac = 0.0
        if self._last_region and self._out is not None:
            H, W = self._out[0].shape[:2]
            frac = self._last_region[2] * self._last_region[3] / float(H * W)
        return {"net": ev["start"].elapsed_time(ev["net"]), "post": ev["net"].elapsed_time(ev["end"]),
                "total": ev["start"].elapsed_time(ev["end"]), "processed_fraction": frac}
