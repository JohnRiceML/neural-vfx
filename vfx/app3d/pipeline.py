"""GPU side of the 3D sandbox: solvers, per-mode volumes, rendering, CUDA-event timing.

No host syncs in here. `GpuTimer.resolve()` reads CUDA event timestamps and is called by main.py
only after the display blit (which has already synchronized the frame).
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn.functional as F

from vfx.core.fields3d import CH3, LR_RES_3D, SIM_DT, Volume, rasterize_obstacles_3d
from vfx.render3d import VolumeRenderer, camera_basis

MODES = ["raw", "trilinear", "neural_vol", "neural_screen", "ref128"]
LABELS = {
    "raw": "raw 64³",
    "trilinear": "classical trilinear ×4 (256³)",
    "neural_vol": "neural volume (route A)",
    "neural_screen": "neural screen-space (route B)",
    "ref128": "reference: direct 128³ sim",
}
UNTRAINED = {"neural_vol", "neural_screen"}
REF_RES = 128
SOLID_RENDER_RES = 128          # obstacles are rasterized at this res for display in every mode


class GpuTimer:
    """Named CUDA-event intervals; resolved lazily (after the frame's sync point)."""

    def __init__(self, history: int = 1200):
        self.pending: list = []
        self.hist: dict = defaultdict(lambda: deque(maxlen=history))
        self.frame_hist: deque = deque(maxlen=history)
        self._acc = 0.0

    def start(self, name: str):
        s = torch.cuda.Event(enable_timing=True)
        s.record()
        return (name, s)

    def stop(self, tok) -> None:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.pending.append((tok[0], tok[1], e))

    def resolve(self) -> None:
        for name, s, e in self.pending:
            ms = s.elapsed_time(e)
            self.hist[name].append(ms)
            self._acc += ms
        self.pending.clear()
        self.frame_hist.append(self._acc)
        self._acc = 0.0

    def last(self, name: str, n: int = 30) -> float:
        h = self.hist.get(name)
        if not h:
            return 0.0
        v = list(h)[-n:]
        return sum(v) / len(v)


class Pipeline:
    def __init__(self, backends, device: str = "cuda", march_scale: float = 0.5, steps: float | None = None):
        self.b = backends
        self.device = torch.device(device)
        self.solver = backends.Solver(LR_RES_3D, backends.runtime_config(), device=device)
        self.ref_solver = None
        self.timer = GpuTimer()
        self.renderers: dict = {}
        self.march_scale = march_scale
        self.steps = steps
        self.obstacles: list = []
        self.state = None
        self.ref_state = None
        self.vols: dict = {}
        self.sim_steps = 0
        self.unavailable = dict(backends.reasons)
        self.neural_live = False          # VolumeUpscaler temporal state valid (ran last tick)
        self.tick_id = 0                  # renderers re-prepare their volume only when this changes

    def available(self, mode: str) -> bool:
        return mode not in self.unavailable

    # ------------------------------------------------------------------ scene
    def reset(self, obstacles: list) -> None:
        self.obstacles = list(obstacles)
        self.state = self.solver.reset(self.obstacles)
        self.ref_state = None
        self.solid_render = rasterize_obstacles_3d(self.obstacles, SOLID_RENDER_RES, str(self.device)) \
            if self.obstacles else None
        self.vols.clear()
        self.neural_live = False

    def set_obstacles(self, obstacles: list) -> None:
        """Change obstacles keeping the fluid (re-reset the solver, copy fields back)."""
        old = self.state.data.clone()
        self.reset(obstacles)
        self.state.data.copy_(old)
        fluid = 1.0 - self.state.solid
        for c in ("density", "vel_x", "vel_y", "vel_z", "fuel"):
            self.state.data[CH3[c]].mul_(fluid)

    def _ensure_ref(self) -> None:
        """Start the 128^3 reference from the current 64^3 state (trilinear) so switching is seamless."""
        if self.ref_solver is None:
            self.ref_solver = self.b.Solver(REF_RES, self.b.SolverConfig(), device=str(self.device))
        if self.ref_state is None:
            self.ref_state = self.ref_solver.reset(self.obstacles)
            up = F.interpolate(self.state.data[None], size=(REF_RES,) * 3, mode="trilinear", align_corners=False)[0]
            self.ref_state.data.copy_(up)
            fluid = 1.0 - self.ref_state.solid
            for c in ("density", "vel_x", "vel_y", "vel_z", "fuel"):
                self.ref_state.data[CH3[c]].mul_(fluid)

    # ------------------------------------------------------------------ 30 Hz tick
    @torch.no_grad()
    def tick(self, forces, modes: set) -> None:
        t = self.timer.start("sim")
        self.state = self.solver.step(self.state, forces, SIM_DT)
        self.timer.stop(t)
        self.sim_steps += 1
        self.tick_id += 1
        st = self.state
        if "ref128" in modes:
            self._ensure_ref()
            t = self.timer.start("sim_ref")
            self.ref_state = self.ref_solver.step(self.ref_state, forces, SIM_DT)
            self.timer.stop(t)
        else:
            self.ref_state = None
        raw = Volume(st.data[CH3["density"]], st.data[CH3["temperature"]])
        self.vols = {"raw": raw, "neural_screen": raw}
        if "trilinear" in modes:
            t = self.timer.start("upscale")
            self.vols["trilinear"] = self.b.classical_upscale(st, 4, "trilinear")
            self.timer.stop(t)
        if "neural_vol" in modes and self.available("neural_vol"):
            up = self.b.volume_upscaler
            if not self.neural_live and hasattr(up, "reset"):
                up.reset()                               # temporal state is stale after a gap
            t = self.timer.start("neural_vol")
            try:
                self.vols["neural_vol"] = up.infer(st)
                self.neural_live = True
            except Exception as e:                       # runtime failure -> mark unavailable
                self.unavailable["neural_vol"] = f"{type(e).__name__}: {str(e)[:120]}"
            self.timer.stop(t)
        else:
            self.neural_live = False
        if self.ref_state is not None:
            self.vols["ref128"] = Volume(self.ref_state.data[CH3["density"]], self.ref_state.data[CH3["temperature"]])

    # ------------------------------------------------------------------ 60 Hz render
    def renderer(self, size, slot: int = 0) -> VolumeRenderer:
        """One renderer per (view size, view slot) so each keeps its own prepared volume."""
        r = self.renderers.get((size, slot))
        if r is None:
            r = self.renderers[(size, slot)] = VolumeRenderer((size[1], size[0]), device=str(self.device),
                                                              march_scale=self.march_scale)
            r.app_key = None
            if self.steps:
                r.set_params(steps=float(self.steps))
        return r

    def set_march_scale(self, s: float) -> None:
        self.march_scale = s
        for r in self.renderers.values():
            r.set_march_scale(s)

    def _screen_aux(self, camera, size, slot: int = 0) -> dict:
        """Route-B aux: bbox of the projected domain cube (host math, no sync) + transmittance."""
        w, h = size
        b = camera_basis(camera, w / h)
        o, fw, rt, up, th = b[0:3], b[3:6], b[6:9], b[9:12], b[12]
        corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], np.float32) - o
        zc = corners @ fw
        if (zc <= 1e-3).any():
            bbox = (0, 0, w, h)
        else:
            sx = (corners @ rt) / zc / (th * w / h)
            sy = (corners @ up) / zc / th
            px, py = (sx + 1) * 0.5 * w, (1 - sy) * 0.5 * h
            x0, x1 = float(np.clip(px.min(), 0, w)), float(np.clip(px.max(), 0, w))
            y0, y1 = float(np.clip(py.min(), 0, h)), float(np.clip(py.max(), 0, h))
            # bucket the box size to quarters of the view: ScreenDetail captures one CUDA graph per
            # crop size, so a free-floating bbox (orbiting camera) would re-capture every few frames
            bw = min(w, int(np.ceil((x1 - x0) / (w / 4)) * (w / 4)))
            bh = min(h, int(np.ceil((y1 - y0) / (h / 4)) * (h / 4)))
            cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            bx0 = int(np.clip(cx - bw / 2, 0, w - bw))
            by0 = int(np.clip(cy - bh / 2, 0, h - bh))
            bbox = (bx0, by0, bx0 + bw, by0 + bh)
        r = self.renderer(size, slot)
        trans = r.hdr[..., 3]
        if trans.shape != (h, w):
            trans = F.interpolate(trans[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        return {"bbox": bbox, "transmittance": trans}

    @torch.no_grad()
    def draw(self, mode: str, camera, size, slot: int = 0) -> torch.Tensor | None:
        vol = self.vols.get(mode)
        if vol is None or not self.available(mode):
            return None
        r = self.renderer(size, slot)
        key = (mode, self.tick_id, id(self.solid_render), r.march_scale)
        reuse = r.app_key == key           # no sim tick since this view last prepared its volume
        r.app_key = key
        t = self.timer.start("render_view" if reuse else "render")
        img = r.draw(vol, camera, self.solid_render, reuse_volume=reuse)
        self.timer.stop(t)
        if mode == "neural_screen":
            t = self.timer.start("detail")
            try:
                img = self.b.screen_detail.infer(img, self._screen_aux(camera, size, slot))
            except Exception as e:
                self.unavailable["neural_screen"] = f"{type(e).__name__}: {str(e)[:120]}"
            self.timer.stop(t)
        return img
