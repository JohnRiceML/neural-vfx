"""GPU side of the 3D sandbox: solvers, per-mode volumes, rendering, CUDA-event timing.

No host syncs in here. `GpuTimer.resolve()` reads CUDA event timestamps and is called by main.py
only after the display blit (which has already synchronized the frame).
"""
from __future__ import annotations

from collections import defaultdict, deque

import dataclasses

import numpy as np
import torch
import torch.nn.functional as F

from vfx.core.fields3d import CH3, LR_RES_3D, SIM_DT, Volume, rasterize_obstacles_3d
from vfx.render3d import VolumeRenderer, camera_basis

MODES = ["raw", "trilinear", "neural_vol", "neural_screen", "ref128",
         "detail", "detail_screen", "hires", "hires_detail"]          # sim-carried detail phase (additive)
LABELS = {
    "raw": "raw 64³",
    "trilinear": "classical trilinear ×4 (256³)",
    "neural_vol": "neural volume (route A)",
    "neural_screen": "neural screen-space (route B)",
    "ref128": "reference: direct 128³ sim",
    "detail": "64³ + advected detail",
    "detail_screen": "64³ + advected detail + route B",
    "hires": "64³ velocity + 128³ density (hires)",
    "hires_detail": "hires 128³ density + advected detail",
}
DETAIL_MODES = {"detail", "detail_screen", "hires_detail"}
HIRES_MODES = {"hires", "hires_detail"}
SCREEN_MODES = {"neural_screen", "detail_screen"}
HIRES_RES = 128
UNTRAINED = {"neural_vol", "neural_screen", "detail_screen"}  # unless Pipeline.is_trained(mode) (a checkpoint was loaded)
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
        self.sim_overrides: dict = {}
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
        self.cam_hist: dict = {}          # view slot -> {tick_id: [cameras drawn during that tick]}
        # sim-carried detail phase (all lazy: nothing is created or run unless such a mode is visible)
        self.detail_quality = "medium"
        self.temporal = False             # renderer temporal amortization (shortlist #2), toggled with J
        self.details: dict = {}           # "lr" (64^3 sim) / "hires" (hires solver) -> AdvectedDetail
        self.detail_live: dict = {}
        self.hires_solver = None
        self.hires_state = None
        self.screen_detail2 = None        # second route B instance for detail_screen (own temporal state)
        self._check_detail_modes()

    def _check_detail_modes(self) -> None:
        try:
            import vfx.detail3d  # noqa: F401
        except Exception as e:
            for m in DETAIL_MODES:
                self.unavailable[m] = f"vfx.detail3d: {type(e).__name__}: {str(e)[:100]}"
        import inspect
        S = self.b.Solver
        try:
            ok = "scalar_res" in inspect.signature(S.__init__).parameters and hasattr(S, "hires_volume")
        except (TypeError, ValueError):
            ok = False
        if not ok:
            for m in HIRES_MODES:
                self.unavailable.setdefault(m, "sim3d has no Solver3D(scalar_res=...) / hires_volume() yet")
        if "neural_screen" in self.unavailable:
            self.unavailable.setdefault("detail_screen", self.unavailable["neural_screen"])

    def detail(self, which: str = "lr"):
        d = self.details.get(which)
        if d is None:
            from vfx.detail3d import AdvectedDetail
            d = self.details[which] = AdvectedDetail(LR_RES_3D, device=str(self.device), quality=self.detail_quality)
        return d

    def set_detail_quality(self, q: str) -> None:
        self.detail_quality = q
        for d in self.details.values():
            d.set_quality(q)

    def _step_detail(self, which: str, state) -> None:
        d = self.detail(which)
        if not self.detail_live.get(which):
            d.reset()                       # coordinates are stale after a gap: restart (weights fade in)
        t = self.timer.start(f"detail_step.{which}")
        d.step(state, SIM_DT)
        self.timer.stop(t)
        self.detail_live[which] = True

    def _ensure_hires(self) -> None:
        if self.hires_solver is None:
            self.hires_solver = self.b.Solver(LR_RES_3D, self._cfg(self.b.runtime_config, self.sim_overrides),
                                              device=str(self.device), scalar_res=HIRES_RES)
        if self.hires_state is None:
            self.hires_state = self.hires_solver.reset(self.obstacles)
            self.hires_state.data.copy_(self.state.data)
            hf = self.hires_solver.hires_fields() if hasattr(self.hires_solver, "hires_fields") else None
            if hf is not None:            # seed the fine scalars from the 64^3 state: switching is seamless
                src = self.state.data[[CH3["density"], CH3["temperature"], CH3["fuel"]]][None]
                hf.copy_(F.interpolate(src, size=tuple(hf.shape[1:]), mode="trilinear", align_corners=False)[0])

    def available(self, mode: str) -> bool:
        return mode not in self.unavailable

    # ------------------------------------------------------------------ scene
    @staticmethod
    def _cfg(factory, overrides: dict):
        """factory(**overrides), keeping only fields the config type has (the stub has fewer)."""
        base = factory()
        names = {f.name for f in dataclasses.fields(base)} if dataclasses.is_dataclass(base) else set()
        kw = {k: v for k, v in overrides.items() if k in names}
        return factory(**kw) if kw else base

    def _apply_config(self, solver, cfg) -> None:
        # Solver3D reads `config` every step (uploaded with the forces, no graph recapture); the stub uses `cfg`
        for attr in ("config", "cfg"):
            if hasattr(solver, attr):
                setattr(solver, attr, cfg)

    def set_sim_config(self, overrides: dict) -> None:
        """Preset SolverConfig3D overrides (dissipation etc.) for the 64^3 sim AND the 128^3 reference."""
        self.sim_overrides = dict(overrides)
        self._apply_config(self.solver, self._cfg(self.b.runtime_config, self.sim_overrides))
        if self.ref_solver is not None:
            self._apply_config(self.ref_solver, self._cfg(self.b.SolverConfig, self.sim_overrides))
        if self.hires_solver is not None:
            self._apply_config(self.hires_solver, self._cfg(self.b.runtime_config, self.sim_overrides))

    def reset(self, obstacles: list) -> None:
        self.obstacles = list(obstacles)
        self.state = self.solver.reset(self.obstacles)
        self.ref_state = None
        self.hires_state = None
        self.detail_live = {}
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
            self.ref_solver = self.b.Solver(REF_RES, self._cfg(self.b.SolverConfig, self.sim_overrides),
                                            device=str(self.device))
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
        # --- sim-carried detail phase (opt-in modes only)
        if modes & {"detail", "detail_screen"} and self.available("detail"):
            self._step_detail("lr", st)
            self.vols["detail"] = self.vols["detail_screen"] = raw
        else:
            self.detail_live["lr"] = False
        if modes & HIRES_MODES and self.available("hires"):
            try:
                self._ensure_hires()
                t = self.timer.start("sim_hires")
                self.hires_state = self.hires_solver.step(self.hires_state, forces, SIM_DT)
                self.timer.stop(t)
                hv = self.hires_solver.hires_volume()
                if hv is not None:
                    self.vols["hires"] = self.vols["hires_detail"] = hv
                if "hires_detail" in modes and self.available("hires_detail"):
                    self._step_detail("hires", self.hires_state)
            except Exception as e:                      # in-progress solver API -> mark unavailable
                for m in HIRES_MODES:
                    self.unavailable[m] = f"{type(e).__name__}: {str(e)[:120]}"
        else:
            self.hires_state = None
            self.detail_live["hires"] = False

    # ------------------------------------------------------------------ 60 Hz render
    def renderer(self, size, slot: int = 0) -> VolumeRenderer:
        """One renderer per (view size, view slot) so each keeps its own prepared volume."""
        r = self.renderers.get((size, slot))
        if r is None:
            r = self.renderers[(size, slot)] = VolumeRenderer((size[1], size[0]), device=str(self.device),
                                                              march_scale=self.march_scale, temporal=self.temporal)
            r.app_key = None
            if self.steps:
                r.set_params(steps=float(self.steps))
        return r

    def set_temporal(self, on: bool) -> None:
        self.temporal = bool(on)
        for r in self.renderers.values():
            r.set_temporal(on)

    def set_march_scale(self, s: float) -> None:
        self.march_scale = s
        for r in self.renderers.values():
            r.set_march_scale(s)

    def is_trained(self, mode: str) -> str | None:
        """Checkpoint name if `mode` runs trained weights, else None (route B: see backends)."""
        return self.b.trained.get("neural_screen" if mode == "detail_screen" else mode)

    def prev_camera(self, camera, slot: int):
        """The camera one sim step (SIM_DT) ago for this view: the camera of the draw at the same
        position within the previous sim tick (sim 30 Hz, display 60 Hz -> 2 draws per tick), so
        motion vectors are in pixels per SIM_DT on every display frame, as in the training data."""
        h = self.cam_hist.setdefault(slot, {})
        cur = h.setdefault(self.tick_id, [])
        prev = h.get(self.tick_id - 1)
        pc = prev[min(len(cur), len(prev) - 1)] if prev else None
        cur.append(camera)
        for k in [k for k in h if k < self.tick_id - 1]:
            del h[k]
        return pc

    def _screen_aux(self, camera, size, slot: int = 0, aux: dict | None = None) -> dict:
        """Route-B aux: bbox of the projected domain cube (host math, no sync) + the renderer's aux
        (alpha, depth, motion: see render3d README) + legacy 'transmittance' (= renderer T)."""
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
            # ScreenDetail buckets the crop size itself (fixed fractions of the frame -> a few CUDA graphs),
            # so the exact projected box is passed (the app-side bucketing workaround is gone)
            bbox = (int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1)))
        r = self.renderer(size, slot)
        trans = r.hdr[..., 3]
        if trans.shape != (h, w):
            trans = F.interpolate(trans[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        return {"bbox": bbox, "transmittance": trans, **(aux or {})}

    @torch.no_grad()
    def draw(self, mode: str, camera, size, slot: int = 0) -> torch.Tensor | None:
        vol = self.vols.get(mode)
        if vol is None or not self.available(mode):
            return None
        r = self.renderer(size, slot)
        key = (mode, self.tick_id, id(self.solid_render), r.march_scale)
        reuse = r.app_key == key           # no sim tick since this view last prepared its volume
        r.app_key = key
        t = self.timer.start(f"{'render_view' if reuse else 'render'}.{slot}")
        if mode in DETAIL_MODES:            # sim-carried detail sampled in the march (+ light volume)
            det = self.details.get("hires" if mode == "hires_detail" else "lr")
            if mode == "detail_screen":
                img, aux = r.draw(vol, camera, self.solid_render, reuse_volume=reuse, return_aux=True,
                                  velocity=self.state.velocity, prev_camera=self.prev_camera(camera, slot), detail=det)
            else:
                img, aux = r.draw(vol, camera, self.solid_render, reuse_volume=reuse, detail=det), None
        elif mode == "neural_screen":     # route B: render with aux (alpha, depth, motion) from the same march
            img, aux = r.draw(vol, camera, self.solid_render, reuse_volume=reuse, return_aux=True,
                              velocity=self.state.velocity, prev_camera=self.prev_camera(camera, slot))
        else:
            img, aux = r.draw(vol, camera, self.solid_render, reuse_volume=reuse), None
        self.timer.stop(t)
        if mode in SCREEN_MODES:
            t = self.timer.start(f"detail.{slot}")
            try:
                sd = self.b.screen_detail if mode == "neural_screen" else self._screen_detail2()
                img = sd.infer(img, self._screen_aux(camera, size, slot, aux))
            except Exception as e:
                self.unavailable[mode] = f"{type(e).__name__}: {str(e)[:120]}"
            self.timer.stop(t)
        return img

    def _screen_detail2(self):
        """detail_screen gets its own ScreenDetail (same checkpoint) so its temporal history is not
        shared with a neural_screen view shown side by side."""
        if self.screen_detail2 is None:
            from vfx.app3d.backends import make_screen_detail
            ck = getattr(self.b, "screen_ckpt", None)
            self.screen_detail2, _ = make_screen_detail(ck, str(self.device))
        return self.screen_detail2
