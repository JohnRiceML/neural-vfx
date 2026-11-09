"""Neural Upscaler: temporal state, FP16, sparse brick cropping, CUDA graphs. No host sync in infer().

Per frame:
  1. pre-stage (one CUDA graph, fixed shapes): fp16 LR -> bicubic base, warp of the previous output
     along the upsampled LR velocity, and the brick-activity mask of this frame.
  2. network on the active crop (one CUDA graph per quantized crop shape), or the full frame.
  3. compose: bicubic everywhere, network result pasted on the active bounding box.
  4. async copy of this frame's brick mask into a ring of pinned host buffers.

Sparse bricks: BRICK x BRICK LR cells (16 -> 8x8 bricks at 128^2). A brick is active if density >
density_thresh or T > ambient + temp_thresh; the mask is dilated by one brick. A frame uses the newest
mask whose copy has COMPLETED (non-blocking event.query()) and is at most `max_lag` frames old
(normally the previous frame); otherwise it runs the full frame. The dilation (64 HR px ~ 0.125 m per
lagged frame) covers smoke entering new bricks. The crop = active bounding box + `halo` LR cells of
context, sized to a multiple of 32 LR cells (<= 16 shapes -> 16 graphs, captured lazily or via warmup()).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vfx.core import paths
from vfx.core.fields import AMBIENT_TEMP, CH, DOMAIN_SIZE, HR_RES, LR_RES, NUM_CHANNELS, UPSCALE, FieldState, HRFields
from vfx.model import load_model
from vfx.runtime.warp import upscale_lr_tensor, warp_prev

_DT = slice(CH["density"], CH["temperature"] + 1, CH["temperature"] - CH["density"])
_VEL = slice(CH["vel_x"], CH["vel_y"] + 1)


def _capture(fn, device, pool):
    """Warm up fn() on a side stream (cudnn autotune) and capture it. Returns (graph, outputs)."""
    s = torch.cuda.Stream(device)
    s.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(3):
            fn()
    torch.cuda.current_stream(device).wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(g, pool=pool):
        out = fn()
    return g, out


class Upscaler:
    def __init__(self, checkpoint_path: str | None = None, device="cuda", *, sparse: bool = True,
                 cuda_graph: bool = True, density_thresh: float = 1e-3, temp_thresh: float = 1.0,
                 brick: int = 16, halo: int = 8, crop_quantum: int = 32, max_lag: int = 2):
        path = Path(checkpoint_path) if checkpoint_path else paths.default_checkpoint()
        if not path.is_file():
            raise FileNotFoundError(
                f"Upscaler: no DetailNet checkpoint at '{path}'. Train one with "
                f"`.venv/Scripts/python -m vfx.train.train --config vfx/train/configs/demo.json` "
                f"(callers should fall back to vfx.runtime.classical_upscale).")
        torch.backends.cudnn.benchmark = True
        self.device = torch.device(device)
        self.checkpoint_path = str(path)
        self.model = load_model(path, device).half().to(memory_format=torch.channels_last).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.sparse = sparse
        self.cuda_graph = cuda_graph and self.device.type == "cuda"
        self.density_thresh, self.temp_thresh = density_thresh, temp_thresh
        self.brick, self.halo, self.quantum, self.max_lag = brick, halo, crop_quantum, max_lag
        self.dx_hr = DOMAIN_SIZE / HR_RES
        self._pool = torch.cuda.graph_pool_handle() if self.cuda_graph else None
        self._net_graphs: dict = {}
        self._pre_graph = None
        self._pre_dt = None
        # static buffers (graph inputs)
        h = LR_RES
        self._lr = torch.zeros(1, NUM_CHANNELS, h, h, device=self.device, dtype=torch.half)
        self._lr[:, CH["temperature"]] = AMBIENT_TEMP
        self._prev = torch.zeros(1, 2, HR_RES, HR_RES, device=self.device, dtype=torch.half)
        self._prev[:, 1] = AMBIENT_TEMP
        nb = LR_RES // brick
        self._ring = [{"host": torch.zeros((nb, nb), dtype=torch.uint8, pin_memory=True),
                       "event": torch.cuda.Event(), "frame": -10 ** 9} for _ in range(max_lag + 2)]
        self._ev = {k: torch.cuda.Event(enable_timing=True) for k in ("start", "warp", "net", "end")}
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self._cold = True                # next frame has no previous output: use bicubic as "prev"
        self._frame = 0
        self._last_region = None
        for r in self._ring:
            r["frame"] = -10 ** 9

    # ------------------------------------------------------------------ stages
    def _pre_fn(self, dt):
        lr = self._lr
        base = upscale_lr_tensor(lr[:, _DT], "bicubic")
        prevw = warp_prev(self._prev, lr[:, _VEL], self.dx_hr, dt)
        b = self.brick
        d = F.max_pool2d(lr[:, CH["density"]:CH["density"] + 1], b)
        t = F.max_pool2d(lr[:, CH["temperature"]:CH["temperature"] + 1], b)
        act = ((d > self.density_thresh) | (t > AMBIENT_TEMP + self.temp_thresh)).half()
        act = F.max_pool2d(act, 3, stride=1, padding=1)[0, 0].to(torch.uint8)
        return base, prevw, act

    def _pre(self, dt):
        if not self.cuda_graph:
            return self._pre_fn(dt)
        if self._pre_graph is None or self._pre_dt != dt:
            saved = self._prev.clone()
            self._pre_graph, self._pre_out = _capture(lambda: self._pre_fn(dt), self.device, self._pool)
            self._prev.copy_(saved)
            self._pre_dt = dt
        self._pre_graph.replay()
        return self._pre_out

    def _net(self, lr, prev):
        if not self.cuda_graph:
            with torch.inference_mode():
                return self.model(lr.contiguous(memory_format=torch.channels_last),
                                  prev.contiguous(memory_format=torch.channels_last))
        key = (lr.shape[-2], lr.shape[-1])
        g = self._net_graphs.get(key)
        if g is None:
            sl = torch.zeros(lr.shape, device=self.device, dtype=torch.half).contiguous(memory_format=torch.channels_last)
            sp = torch.zeros(prev.shape, device=self.device, dtype=torch.half).contiguous(memory_format=torch.channels_last)
            sl[:, CH["temperature"]] = AMBIENT_TEMP
            sp[:, 1] = AMBIENT_TEMP
            graph, out = _capture(lambda: self.model(sl, sp), self.device, self._pool)
            g = self._net_graphs[key] = (graph, sl, sp, out)
        graph, sl, sp, out = g
        sl.copy_(lr)
        sp.copy_(prev)
        graph.replay()
        return out

    def warmup(self, dt: float = 1.0 / 30.0) -> None:
        """Capture every graph up front (pre-stage + all crop shapes) so the frame loop never hitches."""
        sizes = sorted({min(LR_RES, q) for q in range(self.quantum, LR_RES + self.quantum, self.quantum)})
        self._pre(dt)
        for h in sizes:
            for w in sizes:
                self._net(torch.zeros(1, NUM_CHANNELS, h, w, device=self.device, dtype=torch.half),
                          torch.zeros(1, 2, h * UPSCALE, w * UPSCALE, device=self.device, dtype=torch.half))
        torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------------ bricks
    def _crop_1d(self, a: int, b: int):
        L = b - a
        cs = min(LR_RES, -(-(L + 2 * self.halo) // self.quantum) * self.quantum)
        start = min(max(a - (cs - L) // 2, 0), LR_RES - cs)
        return int(start), int(cs)

    def _region(self):
        """None = full frame, () = nothing active, else (cy, ch, cx, cw, y0, y1, x0, x1) in LR cells."""
        if not self.sparse:
            return None
        for r in sorted(self._ring, key=lambda r: -r["frame"]):
            if self._frame - r["frame"] > self.max_lag:
                return None
            if r["event"].query():
                m = r["host"].numpy()
                break
        else:
            return None
        if not m.any():
            return ()
        ys, xs = np.nonzero(m.any(axis=1))[0], np.nonzero(m.any(axis=0))[0]
        y0, y1 = int(ys[0]) * self.brick, (int(ys[-1]) + 1) * self.brick
        x0, x1 = int(xs[0]) * self.brick, (int(xs[-1]) + 1) * self.brick
        cy, ch = self._crop_1d(y0, y1)
        cx, cw = self._crop_1d(x0, x1)
        if ch == LR_RES and cw == LR_RES:
            return None
        return (cy, ch, cx, cw, y0, y1, x0, x1)

    # ------------------------------------------------------------------ main
    def infer(self, lr_state: FieldState) -> HRFields:
        """LR FieldState -> HR density, temperature (fp16, HR_RES^2). No host syncs."""
        ev = self._ev
        ev["start"].record()
        self._lr.copy_(lr_state.data.unsqueeze(0))            # fp32 -> fp16 into the static buffer
        base, prevw, act = self._pre(float(lr_state.meta.dt))
        if self._cold:
            prevw = base
        ev["warp"].record()

        region = self._region()
        self._last_region = region
        s = UPSCALE
        if region is None:
            out = self._net(self._lr, prevw).clone()
        elif region == ():
            out = base.clone()
        else:
            cy, ch, cx, cw, y0, y1, x0, x1 = region
            pred = self._net(self._lr[:, :, cy:cy + ch, cx:cx + cw],
                             prevw[:, :, cy * s:(cy + ch) * s, cx * s:(cx + cw) * s])
            out = base.clone()
            out[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = \
                pred[:, :, (y0 - cy) * s:(y1 - cy) * s, (x0 - cx) * s:(x1 - cx) * s]
        ev["net"].record()
        slot = self._ring[self._frame % len(self._ring)]
        slot["host"].copy_(act, non_blocking=True)
        slot["event"].record()
        slot["frame"] = self._frame
        self._prev.copy_(out)
        self._cold = False
        self._frame += 1
        ev["end"].record()
        return HRFields(density=out[0, 0], temperature=out[0, 1])

    # ------------------------------------------------------------------ timings
    @property
    def timings(self) -> dict:
        """ms of the last infer (CUDA events): pre (cast+bicubic+warp+mask), net (+compose), post, total.
        Reading this waits for the last frame's GPU work: read it in the HUD, not inside the frame loop."""
        ev = self._ev
        ev["end"].synchronize()
        r = self._last_region
        frac = 1.0 if r is None else 0.0 if r == () else (r[1] * r[3]) / float(LR_RES * LR_RES)
        return {
            "warp": ev["start"].elapsed_time(ev["warp"]),
            "net": ev["warp"].elapsed_time(ev["net"]),
            "post": ev["net"].elapsed_time(ev["end"]),
            "total": ev["start"].elapsed_time(ev["end"]),
            "processed_fraction": frac,
        }
