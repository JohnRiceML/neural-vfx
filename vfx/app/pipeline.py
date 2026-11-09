"""GPU side of the app: solver state, upscalers, 30 Hz -> 60 Hz interpolation, rendering, timing.

No host syncs in here. `GpuTimer.resolve()` reads CUDA event timestamps and is called by main.py
only after the display blit (which has already synchronized the frame).
"""
from __future__ import annotations

from collections import defaultdict, deque

import torch

from vfx.core.fields import CH, HR_RES, LR_RES, SIM_DT, Forces, HRFields, rasterize_obstacles
from vfx.render import Renderer

VIEW_LABELS = {"lr": "LR raw (128)", "bicubic": "classical bicubic", "neural": "neural"}


class GpuTimer:
    """Named CUDA-event intervals; resolved lazily (after the frame's sync point)."""

    def __init__(self, history: int = 1200):
        self.pending: list = []
        self.hist: dict = defaultdict(lambda: deque(maxlen=history))
        self.frame_hist: deque = deque(maxlen=history)   # GPU ms summed per display frame
        self._frame_acc = 0.0

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
            self._frame_acc += ms
        self.pending.clear()
        self.frame_hist.append(self._frame_acc)
        self._frame_acc = 0.0

    def last(self, name: str, n: int = 30) -> float:
        h = self.hist.get(name)
        if not h:
            return float("nan")
        v = list(h)[-n:]
        return sum(v) / len(v)


class Pipeline:
    def __init__(self, backends, device: str = "cuda"):
        self.b = backends
        self.device = torch.device(device)
        self.solver = backends.Solver(LR_RES, backends.SolverConfig(), device=device)
        self.timer = GpuTimer()
        self.renderers: dict = {}
        self.obstacles: list = []
        self.state = None
        self.buf: dict = {}               # view -> [prev (2,h,w), curr (2,h,w)] float32
        self._scratch: dict = {}
        self.neural_live = False          # neural upscaler state is valid (ran last tick)
        self.sim_steps = 0

    @property
    def neural_available(self) -> bool:
        return self.b.upscaler is not None

    # ------------------------------------------------------------------ scene
    def reset(self, obstacles: list) -> None:
        self.obstacles = list(obstacles)
        self.state = self.solver.reset(self.obstacles)
        self._solids()
        self.buf.clear()
        if self.b.upscaler is not None:
            self.b.upscaler.reset()
        self.neural_live = False

    def set_obstacles(self, obstacles: list) -> None:
        """Change obstacles keeping the fluid (re-reset the solver, copy fields back)."""
        old = self.state.data.clone()
        self.obstacles = list(obstacles)
        self.state = self.solver.reset(self.obstacles)
        self.state.data.copy_(old)
        fluid = 1.0 - self.state.solid
        for c in ("density", "vel_x", "vel_y", "fuel"):
            self.state.data[CH[c]].mul_(fluid)
        self._solids()

    def _solids(self) -> None:
        self.solid_lr = self.state.solid.float()
        self.solid_hr = rasterize_obstacles(self.obstacles, HR_RES, str(self.device))

    # ------------------------------------------------------------------ 30 Hz tick
    def _push(self, view: str, fields: HRFields) -> None:
        new = torch.stack([fields.density.float(), fields.temperature.float()])
        b = self.buf.get(view)
        if b is None or b[0].shape != new.shape:
            self.buf[view] = [new, new.clone()]
        else:
            b[0], b[1] = b[1], b[0]
            b[1].copy_(new)

    @torch.no_grad()
    def tick(self, forces: Forces, views: set) -> None:
        t = self.timer.start("sim")
        self.state = self.solver.step(self.state, forces, SIM_DT)
        self.timer.stop(t)
        self.sim_steps += 1
        lr = self.state
        for v in [v for v in self.buf if v not in views]:
            del self.buf[v]                 # stale views must not be interpolated from later
        if "lr" in views:
            self._push("lr", HRFields(lr.data[CH["density"]], lr.data[CH["temperature"]]))
        if "bicubic" in views:
            t = self.timer.start("upscale_bicubic")
            self._push("bicubic", self.b.classical_upscale(lr, "bicubic"))
            self.timer.stop(t)
        if "neural" in views and self.b.upscaler is not None:
            if not self.neural_live:
                self.b.upscaler.reset()
                self.buf.pop("neural", None)
            t = self.timer.start("upscale_neural")
            self._push("neural", self.b.upscaler.infer(lr))
            self.timer.stop(t)
            self.neural_live = True
        else:
            self.neural_live = False

    # ------------------------------------------------------------------ 60 Hz render
    def renderer(self, size: int) -> Renderer:
        r = self.renderers.get(size)
        if r is None:
            r = self.renderers[size] = Renderer((size, size), device=str(self.device))
        return r

    @torch.no_grad()
    def draw(self, view: str, alpha: float, size: int) -> torch.Tensor | None:
        b = self.buf.get(view)
        if b is None:
            return None
        key = (view, tuple(b[0].shape))
        s = self._scratch.get(key)
        if s is None:
            s = self._scratch[key] = torch.empty_like(b[0])
        torch.lerp(b[0], b[1], float(alpha), out=s)
        lr_view = view == "lr"
        t = self.timer.start("render")
        img = self.renderer(size).draw(HRFields(s[0], s[1]), self.solid_lr if lr_view else self.solid_hr,
                                       nearest=lr_view)
        self.timer.stop(t)
        return img
