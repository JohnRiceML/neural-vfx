"""3D neural-VFX sandbox (pygame-ce). Run: .venv/Scripts/python -m vfx.app3d.main [--help]

Sim (+ volume upscale) at 30 Hz, render at display rate (60 Hz cap) from the latest volume while the
orbit camera moves every frame. The only GPU->CPU copy per frame is the final uint8 canvas blit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from vfx.core.fields3d import SIM_DT, Camera, Forces3D, Impulse3D
from vfx.app3d.backends import load_backends
from vfx.app3d import pipeline as _pipeline
from vfx.app3d.pipeline import LABELS, MODES, UNTRAINED, Pipeline
from vfx.app3d.presets import DEFAULT_OBSTACLE, PRESETS, fire_emitter, puff_emitter
from vfx.render3d import camera_basis

WIN_W, WIN_H = 1600, 900
STRIP_H = 210                # HUD strip under the views (never overlaps them); H hides it
BG = (8, 9, 12)
TAP_S = 0.25                 # F shorter than this = toggle; longer = torch
BUDGET_MS = 6.0
STAGE_BUDGET = {"sim": 2.0, "detail": 2.5, "render": 1.5}     # spec, per 60 Hz frame
MARCH_SCALES = [1.0, 0.5, 0.25]
SHORT = {"raw": "raw 64³", "trilinear": "trilinear 256³", "neural_vol": "neural vol (A)",
         "neural_screen": "neural screen (B)", "ref128": "ref 128³ sim",
         "detail": "64³ + detail", "detail_screen": "detail + B", "hires": "hires 128³", "hires_detail": "hires + detail",
         "particles": "particles 128³", "particles_detail": "particles + detail", "raw_default": "raw 64³ (default sim)"}
DETAILED = ("detail", "detail_screen", "hires_detail", "particles_detail")     # render row includes detail sampling
DETAIL_QUALITIES = ["calibrated", "low", "medium", "high"]
DETAIL_BENCH_JSON = Path(__file__).resolve().parents[1] / "detail3d" / "bench_latest.json"


def detail_march_delta() -> dict:
    """Bench-measured added march ms per quality at 1440p / scale 0.5 (vfx.detail3d.bench), {} if absent."""
    try:
        rows = json.loads(DETAIL_BENCH_JSON.read_text(encoding="utf-8")).get("render", [])
        return {r["detail"]: r for r in rows if r.get("march_scale") == 0.5 and r["detail"] != "none"}
    except Exception:
        return {}
DIM_TXT = (150, 158, 175)
BENCH_JSON = Path(__file__).resolve().parents[2] / "bench3d" / "results" / "latest.json"


def brute256_sim_ms() -> dict:
    """Measured direct 256^3 sim ms/step per bench3d scene (mean over its march scales); {} if absent."""
    try:
        rows = json.loads(BENCH_JSON.read_text(encoding="utf-8")).get("rows", [])
        out: dict = {}
        for r in rows:
            if r.get("pipeline") == "brute256" and isinstance(r.get("sim"), (int, float)):
                out.setdefault(r["scene"], []).append(float(r["sim"]))
        return {k: sum(v) / len(v) for k, v in out.items()}
    except Exception:
        return {}


class Orbit:
    def __init__(self):
        self.target = np.array([0.5, 0.42, 0.5])
        self.yaw, self.pitch, self.dist, self.fov = 0.0, 0.20, 2.15, 40.0

    def camera(self) -> Camera:
        cp = math.cos(self.pitch)
        off = np.array([cp * math.sin(self.yaw), math.sin(self.pitch), -cp * math.cos(self.yaw)]) * self.dist
        return Camera(position=tuple((self.target + off).tolist()), target=tuple(self.target.tolist()),
                      fov_y_deg=self.fov)

    def cursor_point(self, px, py, w, h):
        """World point under the cursor on the plane through the orbit target facing the camera."""
        b = camera_basis(self.camera(), w / h)
        o, fw, rt, up, th = b[0:3], b[3:6], b[6:9], b[9:12], b[12]
        sx = (2.0 * (px + 0.5) / w - 1.0) * th * (w / h)
        sy = (1.0 - 2.0 * (py + 0.5) / h) * th
        d = fw + rt * sx + up * sy
        d /= np.linalg.norm(d)
        t = float(np.dot(self.target - o, fw) / max(np.dot(d, fw), 1e-6))
        return np.clip(o + d * t, 0.03, 0.97)


class Controls:
    """Host-side input state -> Forces3D for the next sim tick."""

    def __init__(self):
        self.fires: list = []
        self.impulses: list = []
        self.puffs: list = []           # [emitter, ticks_left]
        self.cursor = np.array([0.5, 0.3, 0.5])
        self.cursor_in = False
        self.f_down_t = None
        self.torch_on = False

    def clear(self):
        self.fires.clear(); self.impulses.clear(); self.puffs.clear()

    def push(self, p0, p1, frame_dt):
        v = (np.asarray(p1) - np.asarray(p0)) / max(frame_dt, 1 / 240) * 0.6
        m = float(np.linalg.norm(v))
        if m > 2.0:
            v *= 2.0 / m
        if m > 1e-4:
            self.impulses.append(Impulse3D(pos=tuple(map(float, p1)), radius=0.07, velocity=tuple(map(float, v))))

    def puff(self):
        if self.cursor_in:
            self.puffs.append([puff_emitter(tuple(map(float, self.cursor))), 8])

    def f_press(self, now):
        self.f_down_t = now

    def f_release(self, now):
        if self.f_down_t is not None and now - self.f_down_t < TAP_S and self.cursor_in:
            near = [e for e in self.fires if math.dist(e.pos, self.cursor) < 0.08]
            if near:
                self.fires.remove(near[0])
            elif len(self.fires) < 12:
                self.fires.append(fire_emitter(tuple(map(float, self.cursor))))
        self.f_down_t = None
        self.torch_on = False

    def update(self, now):
        self.torch_on = self.f_down_t is not None and now - self.f_down_t >= TAP_S and self.cursor_in

    def forces(self, preset) -> Forces3D:
        em = list(preset.emitters) + list(self.fires) + [p[0] for p in self.puffs]
        if self.torch_on:
            em.append(fire_emitter(tuple(map(float, self.cursor)), 0.04))
        for p in self.puffs:
            p[1] -= 1
        self.puffs = [p for p in self.puffs if p[1] > 0]
        f = Forces3D(emitters=em, impulses=self.impulses[-8:], wind=preset.wind)
        self.impulses = []
        return f


class App:
    def __init__(self, args):
        self.args = args
        self.b = load_backends(force_stub_sim=args.stub_sim, warmup=args.warmup_neural,
                               checkpoint_b=args.screen_ckpt)
        self.b.screen_ckpt = args.screen_ckpt
        self.pipe = Pipeline(self.b, march_scale=args.march_scale, steps=args.steps)
        self.pipe.set_detail_quality(args.detail_quality)
        self.pipe.set_temporal(args.temporal)
        for name in ("fused", "confref"):
            if getattr(args, f"sim_{name}"):
                if not self.pipe.set_sim_option(name, True):
                    print(f"solver option {name}: unavailable ({self.pipe.sim_option_available(name)})")
        self.detail_delta = detail_march_delta()
        self.ctl = Controls()
        self.orbit = Orbit()
        self.auto_orbit = args.orbit
        self.mode = args.mode
        self.mode2 = args.mode2
        self.side = args.side_by_side
        self.paused = False
        self.hud_on = True
        self.brute256 = brute256_sim_ms()
        self.load_preset(args.preset)
        self.canvas = torch.empty(WIN_H, WIN_W, 3, dtype=torch.uint8, device="cuda")
        self._clear_canvas()
        self.frame_ms: deque = deque(maxlen=3000)
        self.blit_ms: deque = deque(maxlen=3000)
        self.frames = 0
        self._hud_cache = None
        self._hud_t = 0.0

    # ------------------------------------------------------------------ scene control
    def load_preset(self, k):
        self.preset_id = k
        self.preset = PRESETS[k]
        self.ctl.clear()
        self.pipe.set_sim_config(self.preset.sim)
        self.pipe.reset(self.preset.obstacles)

    def toggle_obstacle(self):
        self.pipe.set_obstacles([] if self.pipe.obstacles else (self.preset.obstacles or [DEFAULT_OBSTACLE]))

    def cycle(self, which: str):
        cur = getattr(self, which)
        setattr(self, which, MODES[(MODES.index(cur) + 1) % len(MODES)])

    def toggle_sim_option(self, name):
        self.pipe.set_sim_option(name, not self.pipe.sim_opts[name])

    def cycle_detail_quality(self):
        q = self.pipe.detail_quality
        self.pipe.set_detail_quality(DETAIL_QUALITIES[(DETAIL_QUALITIES.index(q) + 1) % len(DETAIL_QUALITIES)])

    def cycle_march_scale(self):
        i = (MARCH_SCALES.index(self.pipe.march_scale) + 1) % len(MARCH_SCALES) \
            if self.pipe.march_scale in MARCH_SCALES else 0
        self.pipe.set_march_scale(MARCH_SCALES[i])

    def view_h(self) -> int:
        return WIN_H - STRIP_H if self.hud_on else WIN_H

    def layout(self):
        vh = self.view_h()
        if self.side:
            hw = WIN_W // 2
            return [(self.mode, (0, 0, hw - 1, vh)), (self.mode2, (hw + 1, 0, WIN_W - hw - 1, vh))]
        return [(self.mode, (0, 0, WIN_W, vh))]

    def modes(self):
        return {m for m, _ in self.layout()}

    def hit(self, px, py):
        for _, (x, y, w, h) in self.layout():
            if x <= px < x + w and y <= py < y + h:
                return self.orbit.cursor_point(px - x, py - y, w, h), True
        return self.ctl.cursor, False

    # ------------------------------------------------------------------ GPU frame
    def _clear_canvas(self):
        self.canvas[:] = torch.tensor(BG, dtype=torch.uint8, device="cuda")

    def sim_tick(self):
        self.pipe.tick(self.ctl.forces(self.preset), self.modes())

    def compose(self) -> torch.Tensor:
        if (self.side, self.mode, self.mode2, self.hud_on) != getattr(self, "_last_layout", None):
            self._clear_canvas()
            self._last_layout = (self.side, self.mode, self.mode2, self.hud_on)
            if any(m not in self.pipe.vols for m in self.modes()):
                self.sim_tick()                      # make sure newly visible modes have a volume
        cam = self.orbit.camera()
        for slot, (mode, (x, y, w, h)) in enumerate(self.layout()):
            img = self.pipe.draw(mode, cam, (w, h), slot)
            if img is None:
                self.canvas[y:y + h, x:x + w] = torch.tensor(BG, dtype=torch.uint8, device="cuda")
            else:
                self.canvas[y:y + h, x:x + w] = img
        return self.canvas

    def frame(self):
        """GPU compose + the single display blit. Returns host uint8 (H,W,3) numpy."""
        canvas = self.compose()
        t0 = time.perf_counter()
        host = canvas.cpu().numpy()          # the ONE CPU copy per frame (display blit)
        self.blit_ms.append((time.perf_counter() - t0) * 1000)
        self.pipe.timer.resolve()            # events are complete after the blit's sync
        self.frames += 1
        return host

    # ------------------------------------------------------------------ HUD / stats
    def stage_ms(self, mode, slot: int = 0) -> dict:
        """GPU ms of one view's stages: per tick (30 Hz work) and per 60 Hz display frame (amortized:
        sim / volume upscale / volume prep run every other frame, screen-space detail every frame)."""
        t = self.pipe.timer
        sim = t.last("sim_ref") if mode == "ref128" else t.last("sim_hires") if mode in ("hires", "hires_detail") \
            else t.last("sim_base") if mode == "raw_default" else t.last("sim")
        if mode == "trilinear":
            det_tick, det_frame, det_lab = t.last("upscale"), 0.5 * t.last("upscale"), "detail  upscale@30"
        elif mode == "neural_vol":
            det_tick, det_frame, det_lab = t.last("neural_vol"), 0.5 * t.last("neural_vol"), "detail  net @30Hz"
        elif mode == "neural_screen":
            d = t.last(f"detail.{slot}")
            det_tick, det_frame, det_lab = d, d, "detail  net @60Hz"
        elif mode in ("particles", "particles_detail"):
            # 1M particles advected + deposited to 128³ every tick (amortized /2 per 60 Hz frame); shown honestly
            # against the detail budget line even when it is over it (quiet-card: 3.2 ms/tick)
            pt = t.last("particles")
            det_tick, det_frame, det_lab = pt, 0.5 * pt, "detail  particles@30"
            if mode == "particles_detail":
                st_ = t.last("detail_step.particles")
                det_tick, det_frame, det_lab = pt + st_, 0.5 * (pt + st_), "detail  part+adv@30"
        elif mode in ("detail", "detail_screen", "hires_detail"):
            # advected-detail step per tick (amortized /2); its march sampling is inside the render row
            st_ = t.last("detail_step.hires" if mode == "hires_detail" else "detail_step.lr")
            det_tick, det_frame = st_, 0.5 * st_
            det_lab = f"detail  adv.{self.pipe.detail_quality[:3]}@30"
            if mode == "detail_screen":
                d = t.last(f"detail.{slot}")
                det_tick, det_frame, det_lab = det_tick + d, det_frame + d, "detail  adv+net"
        else:
            det_tick = det_frame = 0.0
            det_lab = "detail  (none)"
        rn, rv = t.last(f"render.{slot}"), t.last(f"render_view.{slot}")
        res = f"{_pipeline.REF_RES}³" if mode == "ref128" else "64³+128³ρ" if mode in ("hires", "hires_detail") else "64³"
        if mode not in ("ref128", "raw_default") and self.pipe.sim_opts_label() != "default":
            res += "*"                      # solver opt-ins on (named in the status column and on the view)
        return {
            "rows": [("sim", f"sim {res}  @30Hz", f"{sim:5.2f}", 0.5 * sim, STAGE_BUDGET["sim"]),
                     ("detail", det_lab, f"{det_tick:5.2f}", det_frame, STAGE_BUDGET["detail"]),
                     ("render", "render+det new|reuse" if mode in DETAILED
                      else "render  new|reuse", f"{rn:4.2f}|{rv:4.2f}", 0.5 * (rn + rv), STAGE_BUDGET["render"])],
            "tick_ms": {"sim": sim, "detail": det_tick, "render": rn},
            "tick_total": sim + det_tick + rn,
            "frame_ms": 0.5 * sim + det_frame + 0.5 * (rn + rv),
        }

    def budget_info(self) -> dict:
        b = self.stage_ms(self.mode, 0)
        b.update(title="GPU ms" + (" (left)" if self.side else ""), limit=BUDGET_MS, stage_scale=4.0,
                 note="solid = per 60 Hz frame, outline = this tick")
        extra = []
        if self.side:
            r = self.stage_ms(self.mode2, 1)
            extra.append((f"right view ({SHORT[self.mode2]}): {r['frame_ms']:4.2f} ms/frame, "
                          f"{r['tick_total']:4.2f} tick", DIM_TXT))
        scene = "plume_sphere" if self.preset_id == 2 else "campfire"
        ref = self.brute256.get(scene)
        if ref and self.mode != "ref128" and b["tick_total"] > 0:
            extra.append((f"direct 256³ sim {ref:4.1f} ms/step (bench3d {scene}) = "
                          f"{ref / b['tick_total']:4.1f}x this tick", (190, 200, 255)))
        if self.mode in DETAILED:
            q = self.pipe.detail_quality
            dd = self.detail_delta.get(q)
            extra.append((f"detail {q} (Q): in render row; 1440p bench "
                          + (f"+{dd['added_full']:.2f}|+{dd['added_view']:.2f} ms" if dd else "n/a"), (150, 220, 150)))
        b["extra"] = extra
        return b

    def status_lines(self):
        ft = np.array(self.frame_ms)[30:][-600:]          # skip start-up frames (graph capture, kernel load)
        if not len(ft):
            ft = np.array([16.7])
        fps = 1000.0 / ft[-60:].mean()
        low = 1000.0 / np.sort(ft)[-max(1, len(ft) // 100):].mean()
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        free, total = torch.cuda.mem_get_info()
        W, Y, G, R = (225, 228, 235), (255, 210, 120), (140, 220, 140), (255, 120, 110)
        lay = self.layout()[0][1]
        r = self.pipe.renderers.get(((lay[2], lay[3]), 0))
        rr = f"{r.mw}x{r.mh} -> {r.out_w}x{r.out_h}" if r else "-"
        lines = [
            (f"{SHORT[self.mode]}" + (f"  |  {SHORT[self.mode2]}" if self.side else ""), Y),
            (f"preset {self.preset_id}: {self.preset.name}" + ("  [PAUSED]" if self.paused else ""), W),
            (f"fps {fps:5.1f}   1% low {low:5.1f}", G if low >= 50 else R),
            (f"march {rr}" + ("  TEMPORAL" if self.pipe.temporal else ""), DIM_TXT),
            (f"solver opt-ins: {self.pipe.sim_opts_label()}  (P fused, K conf@128)",
             (150, 220, 150) if self.pipe.sim_opts_label() != "default" else DIM_TXT),
            (f"VRAM torch peak {peak:4.0f} MB  dev {(total - free) / 2**20:5.0f}", W),
            (f"sim step {self.pipe.sim_steps}  fires {len(self.ctl.fires)}" + ("  TORCH" if self.ctl.torch_on else ""), W),
            (f"dissipation {self.pipe.sim_overrides.get('density_dissipation', '-')}/s  "
             f"cooling {self.pipe.sim_overrides.get('temperature_cooling', '-')}/s", DIM_TXT),
        ]
        if self.b.stubs:
            lines.append(("stubs: " + ", ".join(self.b.stubs), R))
        for m in ("neural_vol", "neural_screen") + tuple(m_ for m_ in ("hires",) if m_ in self.modes()):
            if not self.pipe.available(m):
                lines.append((f"{m}: unavailable", R))
        return lines

    def draw_hud(self, surf, hud, force=False):
        for mode, (x, y, w, h) in self.layout():
            name = LABELS[mode]
            if not self.pipe.available(mode):
                hud.label(surf, name, (x + 12, y + 12), sub="UNAVAILABLE")
                hud.center_note(surf, (x, y, w, h), ["UNAVAILABLE", self.pipe.unavailable[mode][:70]])
            elif mode in UNTRAINED and self.pipe.is_trained(mode):
                hud.label(surf, name, (x + 12, y + 12), sub=f"trained: {self.pipe.is_trained(mode)}",
                          sub_color=(140, 220, 140))
            elif mode in UNTRAINED:
                hud.label(surf, name, (x + 12, y + 12), sub="UNTRAINED - timing only, not a result")
            elif mode not in ("ref128", "raw_default") and self.pipe.sim_opts_label() != "default":
                hud.label(surf, name, (x + 12, y + 12), sub=f"solver opt-ins: {self.pipe.sim_opts_label()}",
                          sub_color=(140, 220, 140))
            else:
                hud.label(surf, name, (x + 12, y + 12))
        if not self.hud_on:
            return
        now = time.perf_counter()
        if force or self._hud_cache is None or now - self._hud_t > 0.25:
            self._hud_t = now
            self._hud_cache = hud.strip((WIN_W, STRIP_H), self.status_lines(), self.budget_info())
        surf.blit(self._hud_cache, (0, WIN_H - STRIP_H))

    def stats(self, wall_s, warmup: int = 30):
        ft = np.array(self.frame_ms)[warmup:]
        if not len(ft):
            ft = np.array(self.frame_ms)
        worst = np.sort(ft)[-max(1, len(ft) // 100):]
        t = self.pipe.timer

        def agg(name):
            h = np.array(t.hist.get(name, []))[warmup // 2:]
            if not len(h):
                return None
            return {"mean": float(h.mean()), "p50": float(np.median(h)), "p99": float(np.percentile(h, 99)), "n": int(len(h))}
        gf = np.array(t.frame_hist)[warmup:]
        lay = self.layout()[0][1]
        r = self.pipe.renderers.get(((lay[2], lay[3]), 0))
        return {
            "frames": self.frames, "warmup_frames_excluded": warmup, "sim_steps": self.pipe.sim_steps, "wall_s": wall_s,
            "fps_mean": float(1000.0 / ft.mean()), "fps_1pct_low": float(1000.0 / worst.mean()),
            "frame_ms": {"mean": float(ft.mean()), "p99": float(np.percentile(ft, 99))},
            "gpu_ms": {k: agg(k) for k in ("sim", "sim_ref", "upscale", "neural_vol", "detail.0", "render.0",
                                           "render_view.0", "detail.1", "render.1", "render_view.1",
                                           "detail_step.lr", "detail_step.hires", "detail_step.particles", "sim_hires",
                                           "particles", "sim_base")},
            "sim_opts": dict(self.pipe.sim_opts),
            "detail_quality": self.pipe.detail_quality, "temporal": self.pipe.temporal,
            "budget_main_view": {k: self.stage_ms(self.mode, 0)[k] for k in ("tick_total", "frame_ms")},
            "sim_config": self.pipe.sim_overrides,
            "gpu_ms_per_frame": {"mean": float(gf.mean()) if len(gf) else None,
                                 "p99": float(np.percentile(gf, 99)) if len(gf) else None},
            "blit_ms": {"mean": float(np.mean(self.blit_ms)), "p99": float(np.percentile(self.blit_ms, 99))},
            "vram_peak_torch_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
            "mode": self.mode, "mode2": self.mode2 if self.side else None, "side_by_side": self.side,
            "preset": self.preset_id, "march_scale": self.pipe.march_scale,
            "render_size": [r.out_w, r.out_h] if r else None, "march_size": [r.mw, r.mh] if r else None,
            "steps_per_m": r.p.steps if r else None,
            "unavailable": self.pipe.unavailable, "stubs": self.b.stubs, "backend_notes": self.b.notes, "trained": self.b.trained,
            "solver": f"{self.b.Solver.__module__}.{self.b.Solver.__name__}",
        }


# ---------------------------------------------------------------------- headless
def scripted_input(app, i, fdt):
    """--demo-input: fire tap at frame 20, a push drag at frames 90-120, a smoke puff at 150."""
    c = app.ctl
    c.cursor_in = True
    if i == 20:
        c.cursor = np.array([0.32, 0.12, 0.45])
        c.f_press(0.0); c.f_release(0.1)
    if 90 <= i < 120:
        p0 = np.array([0.3 + (i - 91) * 0.012, 0.45, 0.5])
        p1 = np.array([0.3 + (i - 90) * 0.012, 0.45, 0.5])
        c.push(p0, p1, fdt)
    if i == 150:
        c.cursor = np.array([0.7, 0.3, 0.5])
        c.puff()


def run_headless(args):
    import pygame
    from vfx.app3d.hud import Hud
    app = App(args)
    hud = Hud()
    fdt = 1.0 / 60.0
    acc = SIM_DT
    t_start = last = time.perf_counter()
    host = None
    for i in range(args.frames):
        if args.demo_input:
            scripted_input(app, i, fdt)
        if app.auto_orbit:
            app.orbit.yaw += 0.35 * fdt
        acc += fdt
        while acc >= SIM_DT - 1e-9:
            app.sim_tick()
            acc -= SIM_DT
        host = app.frame()
        now = time.perf_counter()
        app.frame_ms.append((now - last) * 1000)
        last = now
    wall = time.perf_counter() - t_start
    st = app.stats(wall)
    print(json.dumps({k: st[k] for k in ("fps_mean", "fps_1pct_low", "gpu_ms", "gpu_ms_per_frame", "stubs",
                                         "unavailable", "solver")}, indent=1))
    if args.screenshot:
        surf = pygame.image.frombuffer(host.tobytes(), (WIN_W, WIN_H), "RGB").copy()
        app.draw_hud(surf, hud, force=True)
        Path(args.screenshot).parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(surf, args.screenshot)
        print("screenshot ->", args.screenshot)
    if args.stats:
        Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats).write_text(json.dumps(st, indent=2))
        print("stats ->", args.stats)


# ---------------------------------------------------------------------- windowed
def run_window(args):
    import pygame
    from vfx.app3d.hud import Hud
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("neural-vfx 3D sandbox")
    app = App(args)
    hud = Hud()
    clock = pygame.time.Clock()
    t0 = last = time.perf_counter()
    acc = SIM_DT
    prev_push = None
    running = True
    while running:
        now = time.perf_counter()
        fdt = now - last
        last = now
        app.frame_ms.append(fdt * 1000)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k == pygame.K_n:
                    app.cycle("mode")
                elif k == pygame.K_b:
                    app.cycle("mode2")
                elif k == pygame.K_v:
                    app.side = not app.side
                elif k == pygame.K_t:
                    app.cycle_march_scale()
                elif k == pygame.K_q:
                    app.cycle_detail_quality()
                elif k == pygame.K_j:
                    app.pipe.set_temporal(not app.pipe.temporal)
                elif k == pygame.K_p:
                    app.toggle_sim_option("fused")
                elif k == pygame.K_k:
                    app.toggle_sim_option("confref")
                elif k == pygame.K_o:
                    app.toggle_obstacle()
                elif k == pygame.K_r:
                    app.load_preset(app.preset_id)
                elif k == pygame.K_SPACE:
                    app.paused = not app.paused
                elif k == pygame.K_h:
                    app.hud_on = not app.hud_on
                    app._hud_cache = None
                elif k == pygame.K_c:
                    app.auto_orbit = not app.auto_orbit
                elif k == pygame.K_f:
                    app.ctl.f_press(now)
                elif k == pygame.K_g:
                    app.ctl.puff()
                elif pygame.K_1 <= k <= pygame.K_4:
                    app.load_preset(k - pygame.K_0)
            elif ev.type == pygame.KEYUP and ev.key == pygame.K_f:
                app.ctl.f_release(now)
            elif ev.type == pygame.MOUSEWHEEL:
                app.orbit.dist = float(np.clip(app.orbit.dist * (0.9 ** ev.y), 0.9, 4.5))
            elif ev.type == pygame.MOUSEMOTION:
                if ev.buttons[2]:
                    app.orbit.yaw -= ev.rel[0] * 0.006
                    app.orbit.pitch = float(np.clip(app.orbit.pitch + ev.rel[1] * 0.004, -0.15, 1.35))
                w, inside = app.hit(*ev.pos)
                app.ctl.cursor, app.ctl.cursor_in = w, inside
                if ev.buttons[0] and inside:
                    if prev_push is not None:
                        app.ctl.push(prev_push, w, max(fdt, 1 / 240))
                    prev_push = w
                else:
                    prev_push = None
        app.ctl.update(now)
        if app.auto_orbit:
            app.orbit.yaw += 0.35 * fdt
        if not app.paused:
            acc = min(acc + fdt, 2 * SIM_DT)
            while acc >= SIM_DT:
                app.sim_tick()
                acc -= SIM_DT
        host = app.frame()
        surf = pygame.image.frombuffer(host, (WIN_W, WIN_H), "RGB")
        screen.blit(surf, (0, 0))
        app.draw_hud(screen, hud)
        pygame.display.flip()
        clock.tick(args.fps)
        if args.quit_after and now - t0 > args.quit_after:
            running = False
    st = app.stats(time.perf_counter() - t0)
    print(json.dumps({k: st[k] for k in ("frames", "fps_mean", "fps_1pct_low", "gpu_ms_per_frame", "stubs", "unavailable")}))
    if args.stats:
        Path(args.stats).write_text(json.dumps(st, indent=2))
    pygame.quit()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="neural-vfx 3D sandbox")
    ap.add_argument("--headless", action="store_true", help="no window; run --frames and write outputs")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--preset", type=int, default=1, choices=sorted(PRESETS))
    ap.add_argument("--mode", default="trilinear", choices=MODES)
    ap.add_argument("--mode2", default="raw", choices=MODES, help="right view in side-by-side")
    ap.add_argument("--side-by-side", action="store_true")
    ap.add_argument("--screenshot", default=None)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--orbit", action="store_true", help="auto-rotate the camera")
    ap.add_argument("--march-scale", type=float, default=0.5)
    ap.add_argument("--steps", type=float, default=None, help="ray-march samples per meter (default: renderer's)")
    ap.add_argument("--stub-sim", action="store_true", help="force the app's toy solver")
    ap.add_argument("--screen-ckpt", default=None,
                    help="route B checkpoint (default: runtime3d's default path if that file exists, else untrained)")
    ap.add_argument("--warmup-neural", action="store_true",
                    help="pre-capture every VolumeUpscaler graph at start-up (slow: ~70 s on a shared GPU); "
                         "otherwise graphs are captured on first use of each brick count (hitches)")
    ap.add_argument("--demo-input", action="store_true", help="headless: scripted fire / push / puff input")
    ap.add_argument("--quit-after", type=float, default=0.0, help="windowed: exit after N seconds")
    ap.add_argument("--fps", type=int, default=60, help="windowed frame cap")
    ap.add_argument("--ref-res", type=int, default=128,
                    help="resolution of the direct-sim reference mode (128 default; 192 or 256 cost far more GPU time and VRAM)")
    ap.add_argument("--window", default="1600x900", help="window size WxH")
    ap.add_argument("--sim-fused", action="store_true",
                    help="solver opt-in: fused multigrid pressure (vfx.sim3d.fused_config) instead of 4-iteration PCG; P toggles")
    ap.add_argument("--sim-confref", action="store_true",
                    help="solver opt-in: confinement_ref_res=128 (resolution-consistent swirl); K toggles")
    ap.add_argument("--best", action="store_true",
                    help="the best measured combination (see BEST in main.py: particles + calibrated advected detail, "
                         "fused pressure), side by side with raw 64³ from the default solver (raw_default)")
    ap.add_argument("--best-particles", action="store_true", help="alias of --best (kept for old scripts)")
    ap.add_argument("--best-light", action="store_true", help="like --best without particles: calibrated advected "
                    "detail only (cheapest)")
    ap.add_argument("--temporal", action="store_true",
                    help="temporally amortized ray-march (jittered starts + camera-reprojected history); J toggles")
    ap.add_argument("--detail-quality", default="calibrated", choices=DETAIL_QUALITIES,
                    help="advected-detail quality (noise octaves 1/2/4) for the detail modes; Q cycles it")
    return ap.parse_args(argv)


# --best / --best-light, chosen from measurements (presets 1-4 vs the steered 256³; bench3d/results/MODES.md and
# `python -m vfx.detail3d.tune_bands`, which reproduces MODES.md's numbers), NOT by assumption:
#                                band dist  patch SWD   tLP    (0 = like the reference; lower is better)
#   raw 64³                        0.359      0.220     0.440
#   detail medium (old default)    0.270      0.107     0.093   visible 2-4 px crosshatch at 1:1
#   detail calibrated              0.119      0.080     0.167   no crosshatch; calmer in motion than the reference
#   particles 1M@128³              0.118      0.075     0.129
#   particles + calibrated detail  0.051      0.041     0.050   cov_damp 0.7 (STACKED); best on all three
#   (route B adversarial net:      0.051      0.068     0.194 — for scale)
#  * fused pressure: same quality as the default PCG (raw + fused: band 0.355 vs 0.359), ~half the solve -> ON.
#  * confinement_ref_res=128: worse vs the reference in every measured row (hires + conf@128: band 0.226 vs 0.199,
#    tLP 0.384 vs 0.282) -> OFF (K / --sim-confref still toggles it).
#  Cost (quiet card, per sim tick): particles 1.41 ms + detail step ~0.1 ms; render with detail ~1.1 ms per frame.
BEST = {"mode": "particles_detail", "detail_quality": "calibrated", "sim_fused": True, "sim_confref": False}
BEST_LIGHT = {"mode": "detail", "detail_quality": "calibrated", "sim_fused": True, "sim_confref": False}


def apply_best(args) -> None:
    """--best: one launch for the current best measured combination next to the honest baseline."""
    if args.best or args.best_particles or args.best_light:
        b = BEST_LIGHT if args.best_light else BEST          # --best-particles: old name, same as --best
        args.mode, args.detail_quality = b["mode"], b["detail_quality"]
        args.sim_fused, args.sim_confref = b["sim_fused"], b["sim_confref"]
        args.mode2 = "raw_default"
        args.side_by_side = True


def main(argv=None):
    args = parse_args(argv)
    apply_best(args)
    global WIN_W, WIN_H
    WIN_W, WIN_H = (int(v) for v in args.window.lower().split("x"))
    _pipeline.REF_RES = args.ref_res
    LABELS["ref128"] = f"reference: direct {args.ref_res}³ sim"
    SHORT["ref128"] = f"ref {args.ref_res}³ sim"
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    if args.headless:
        run_headless(args)
    else:
        run_window(args)


if __name__ == "__main__":
    main()
