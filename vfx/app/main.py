"""Neural VFX playground (pygame-ce). Run: .venv/Scripts/python -m vfx.app.main [--help]

Sim + upscale at 30 Hz, render at 60 Hz by interpolating the last two upscaled frames.
The only GPU->CPU copy per frame is the final uint8 canvas blit.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from vfx.core.fields import SIM_DT, Emitter, Forces, Impulse
from vfx.app.backends import load_backends
from vfx.app.pipeline import VIEW_LABELS, Pipeline
from vfx.app.presets import DEFAULT_OBSTACLE, PRESETS, fire_emitter

WIN_W, WIN_H = 1600, 1000
SINGLE = (960, [(20, 20)])
SIDE = (760, [(20, 20), (820, 20)])
BG = (12, 13, 17)
MODES = ["lr", "bicubic", "neural"]
TAP_S = 0.25            # F shorter than this = toggle; longer = torch


class Controls:
    """Host-side input state -> Forces for the next sim tick."""

    def __init__(self):
        self.fires: list = []
        self.impulses: list = []
        self.cursor = (0.5, 0.5)
        self.cursor_in = False
        self.rmb = False
        self.f_down_t = None
        self.torch_on = False

    def clear(self):
        self.fires.clear()
        self.impulses.clear()

    def drag(self, p0, p1, frame_dt):
        vx = (p1[0] - p0[0]) / max(frame_dt, 1 / 240) * 0.5
        vy = (p1[1] - p0[1]) / max(frame_dt, 1 / 240) * 0.5
        m = math.hypot(vx, vy)
        if m > 4.0:
            vx, vy = vx * 4.0 / m, vy * 4.0 / m
        if m > 1e-4:
            self.impulses.append(Impulse(pos=p1, radius=0.05, velocity=(vx, vy)))

    def f_press(self, now):
        self.f_down_t = now

    def f_release(self, now):
        if self.f_down_t is not None and now - self.f_down_t < TAP_S and self.cursor_in:
            near = [e for e in self.fires if math.dist(e.pos, self.cursor) < 0.07]
            if near:
                self.fires.remove(near[0])
            elif len(self.fires) < 16:         # solver supports 32 emitters total
                self.fires.append(fire_emitter(self.cursor))
        self.f_down_t = None
        self.torch_on = False

    def update(self, now):
        self.torch_on = self.f_down_t is not None and now - self.f_down_t >= TAP_S and self.cursor_in

    def forces(self, preset) -> Forces:
        em = list(preset.emitters) + list(self.fires)
        if self.torch_on:
            em.append(fire_emitter(self.cursor, 0.035))
        if self.rmb and self.cursor_in:
            em.append(Emitter(pos=self.cursor, radius=0.035, density_rate=8.0, temperature=330.0))
        f = Forces(emitters=em, impulses=self.impulses[-8:], wind=preset.wind)
        self.impulses = []
        return f


class App:
    def __init__(self, args):
        self.args = args
        self.b = load_backends(args.checkpoint, force_stub_sim=args.stub_sim)
        self.pipe = Pipeline(self.b)
        self.ctl = Controls()
        self.mode = args.mode
        if self.mode == "neural" and not self.pipe.neural_available:
            print(f"[app] neural unavailable ({self.b.neural_reason}); falling back to bicubic", file=sys.stderr)
            self.mode = "bicubic"
        self.side = args.side_by_side
        self.paused = False
        self.hud_on = True
        self.preset_id = args.preset
        self.load_preset(self.preset_id)
        self.canvas = torch.empty(WIN_H, WIN_W, 3, dtype=torch.uint8, device="cuda")
        self.canvas[:] = torch.tensor(BG, dtype=torch.uint8, device="cuda")
        self.frame_ms: deque = deque(maxlen=2000)
        self.blit_ms: deque = deque(maxlen=2000)
        self.frames = 0
        self._hud_surf = None
        self._hud_t = 0.0
        try:
            import warp as wp
            wp.config.quiet = True
            wp.init()
            self._wp = wp
            self._wp_stream = wp.stream_from_torch(torch.cuda.current_stream())
        except Exception:
            self._wp = None

    # ------------------------------------------------------------------ scene control
    def load_preset(self, k):
        self.preset_id = k
        self.preset = PRESETS[k]
        self.ctl.clear()
        self.pipe.reset(self.preset.obstacles)

    def toggle_obstacle(self):
        if self.pipe.obstacles:
            self.pipe.set_obstacles([])
        else:
            self.pipe.set_obstacles(self.preset.obstacles or [DEFAULT_OBSTACLE])

    def cycle_mode(self):
        avail = [m for m in MODES if m != "neural" or self.pipe.neural_available]
        self.mode = avail[(avail.index(self.mode) + 1) % len(avail)] if self.mode in avail else avail[0]

    def layout(self):
        if self.side:
            right = "neural" if self.pipe.neural_available else "lr"
            return SIDE[0], list(zip(["bicubic", right], SIDE[1]))
        return SINGLE[0], [(self.mode, SINGLE[1][0])]

    def views(self):
        return {v for v, _ in self.layout()[1]}

    def to_world(self, px, py):
        size, items = self.layout()
        for _, (x, y) in items:
            if x <= px < x + size and y <= py < y + size:
                return ((px - x) / size, 1.0 - (py - y) / size), True
        return self.ctl.cursor, False

    # ------------------------------------------------------------------ GPU frame
    def gpu_scope(self):
        if self._wp is None:
            return contextlib.nullcontext()
        return self._wp.ScopedStream(self._wp_stream, sync_enter=False, sync_exit=False)

    def sim_tick(self):
        self.pipe.tick(self.ctl.forces(self.preset), self.views())

    def compose(self, alpha) -> torch.Tensor:
        size, items = self.layout()
        if self.side != getattr(self, "_last_side", None):
            self.canvas[:] = torch.tensor(BG, dtype=torch.uint8, device="cuda")
            self._last_side = self.side
        for view, (x, y) in items:
            img = self.pipe.draw(view, alpha, size)
            if img is not None:
                self.canvas[y:y + size, x:x + size] = img
        return self.canvas

    def ensure_buffers(self):
        """Mode switch: make sure every visible view has data (one extra tick if needed)."""
        if any(v not in self.pipe.buf for v in self.views()):
            self.sim_tick()

    # ------------------------------------------------------------------ HUD
    def stats_lines(self):
        t = self.pipe.timer
        ft = np.array(self.frame_ms)[-600:] if self.frame_ms else np.array([16.7])
        fps = 1000.0 / ft[-60:].mean()
        worst = np.sort(ft)[-max(1, len(ft) // 100):]
        low = 1000.0 / worst.mean()
        sim, rnd = t.last("sim"), t.last("render")
        upn, upb = t.last("upscale_neural"), t.last("upscale_bicubic")
        views = self.views()
        ups = (upn if "neural" in views else 0.0) + (upb if "bicubic" in views else 0.0)
        nrend = len(views)
        per_frame = 0.5 * (sim + ups) + nrend * rnd
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        free, total = torch.cuda.mem_get_info()
        W, Y, G, R = (235, 235, 240), (255, 210, 120), (150, 220, 150), (255, 130, 120)
        mode = "side-by-side: " + " | ".join(VIEW_LABELS[v] for v, _ in self.layout()[1]) if self.side else VIEW_LABELS[self.mode]
        lines = [
            (f"mode   {mode}", Y),
            (f"preset {self.preset_id}: {self.preset.name}" + ("   [PAUSED]" if self.paused else ""), W),
            None,
            (f"fps {fps:6.1f}    1% low {low:6.1f}", G if low >= 50 else R),
            (f"GPU ms  sim {sim:5.2f}/step   render {rnd:5.2f}/view", W),
            (f"        upscale bicubic {upb:5.2f}  neural {upn:5.2f} /step", W),
            (f"        total {per_frame:5.2f} ms/frame (30 Hz stages amortized)", W),
            (f"VRAM peak torch {peak:6.0f} MB   device used {(total - free) / 2**20:6.0f} MB", W),
            (f"sim step {self.pipe.sim_steps}   fires {len(self.ctl.fires)}" + ("   TORCH" if self.ctl.torch_on else ""), W),
            ("neural: " + ("ready" if self.pipe.neural_available else f"unavailable ({self.b.neural_reason[:60]})"),
             G if self.pipe.neural_available else R),
        ]
        if self.b.stubs:
            lines.append(("stubs: " + ", ".join(self.b.stubs), R))
        return lines

    def draw_hud(self, surf, hud, force=False):
        import pygame
        size, items = self.layout()
        for view, (x, y) in items:
            name = VIEW_LABELS[view]
            if view == "neural" and not self.pipe.neural_available:
                name += " (unavailable)"
            hud.label(surf, name, (x + 8, y + 8))
        if not self.hud_on:
            return
        now = time.perf_counter()
        if force or self._hud_surf is None or now - self._hud_t > 0.25:
            self._hud_t = now
            if self.side:
                lines = self.stats_lines()
                a = hud.render_panel(lines[:9], (900, 200), help_on=False)
                b = hud.render_panel(lines[9:], (640, 200), help_on=True)
                s = pygame.Surface((WIN_W, 200), pygame.SRCALPHA)
                s.blit(a, (0, 0)); s.blit(b, (900, 0))
                self._hud_surf = (s, (24, 792))
            else:
                self._hud_surf = (hud.render_panel(self.stats_lines(), (580, 960)), (1000, 24))
        surf.blit(*self._hud_surf)

    # ------------------------------------------------------------------ frame
    def frame(self, alpha):
        """GPU compose + the single display blit. Returns host uint8 (H,W,3) numpy."""
        with self.gpu_scope():
            canvas = self.compose(alpha)
        t0 = time.perf_counter()
        host = canvas.cpu().numpy()          # the ONE CPU copy per frame (display blit)
        self.blit_ms.append((time.perf_counter() - t0) * 1000)
        self.pipe.timer.resolve()            # events are complete after the blit's sync
        self.frames += 1
        return host

    def stats(self, wall_s, warmup: int = 30):
        ft = np.array(self.frame_ms)[warmup:]          # skip graph capture / kernel load frames
        worst = np.sort(ft)[-max(1, len(ft) // 100):]
        def agg(name):
            h = np.array(self.pipe.timer.hist.get(name, []))[warmup // 2:]
            if not len(h):
                return None
            return {"mean": float(h.mean()), "p50": float(np.median(h)), "p99": float(np.percentile(h, 99)), "n": int(len(h))}
        gf = np.array(self.pipe.timer.frame_hist)[warmup:]
        return {
            "frames": self.frames, "warmup_frames_excluded": warmup, "sim_steps": self.pipe.sim_steps, "wall_s": wall_s,
            "fps_mean": float(1000.0 / ft.mean()), "fps_1pct_low": float(1000.0 / worst.mean()),
            "frame_ms": {"mean": float(ft.mean()), "p99": float(np.percentile(ft, 99))},
            "gpu_ms": {k: agg(k) for k in ("sim", "upscale_bicubic", "upscale_neural", "render")},
            "gpu_ms_per_frame": {"mean": float(gf.mean()), "p99": float(np.percentile(gf, 99))},
            "blit_ms": {"mean": float(np.mean(self.blit_ms)), "p99": float(np.percentile(self.blit_ms, 99))},
            "vram_peak_torch_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
            "mode": self.mode, "side_by_side": self.side, "preset": self.preset_id,
            "views": sorted(self.views()), "view_size": self.layout()[0],
            "neural_available": self.pipe.neural_available, "neural_reason": self.b.neural_reason,
            "stubs": self.b.stubs, "backend_notes": self.b.notes,
            "solver": f"{self.b.Solver.__module__}.{self.b.Solver.__name__}",
        }


# ---------------------------------------------------------------------- headless
def run_headless(args):
    import pygame
    from vfx.app.hud import Hud
    app = App(args)
    hud = Hud()
    app.ensure_buffers()
    fdt = 1.0 / 60.0
    acc = 0.0
    t_start = time.perf_counter()
    last = t_start
    for i in range(args.frames):
        if args.demo_input:
            scripted_input(app, i, fdt)
        acc += fdt
        while acc >= SIM_DT - 1e-9:
            with app.gpu_scope():
                app.sim_tick()
            acc -= SIM_DT
        host = app.frame(acc / SIM_DT)
        now = time.perf_counter()
        app.frame_ms.append((now - last) * 1000)
        last = now
    wall = time.perf_counter() - t_start
    st = app.stats(wall)
    print(json.dumps({k: st[k] for k in ("fps_mean", "fps_1pct_low", "gpu_ms", "gpu_ms_per_frame", "stubs",
                                         "neural_available", "solver")}, indent=1))
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


def scripted_input(app, i, fdt):
    """--demo-input: a mouse drag at frames 100-130, a held torch 160-400, smoke paint 250-300."""
    c = app.ctl
    if 100 <= i < 130:
        y = 0.45
        c.drag((0.2 + (i - 101) * 0.02, y), (0.2 + (i - 100) * 0.02, y), fdt)
    c.cursor, c.cursor_in = (0.25, 0.3), True
    c.torch_on = 160 <= i < 400
    c.rmb = 250 <= i < 300
    if c.rmb:
        c.cursor = (0.75, 0.6)


# ---------------------------------------------------------------------- windowed
def run_window(args):
    import pygame
    from vfx.app.hud import Hud
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("neural-vfx playground")
    app = App(args)
    hud = Hud()
    app.ensure_buffers()
    clock = pygame.time.Clock()
    t0 = last = time.perf_counter()
    acc = 0.0
    prev_mouse = None
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
                    app.cycle_mode()
                elif k == pygame.K_v:
                    app.side = not app.side
                elif k == pygame.K_o:
                    app.toggle_obstacle()
                elif k == pygame.K_r:
                    app.load_preset(app.preset_id)
                elif k == pygame.K_SPACE:
                    app.paused = not app.paused
                elif k == pygame.K_h:
                    app.hud_on = not app.hud_on
                elif k == pygame.K_f:
                    app.ctl.f_press(now)
                elif pygame.K_1 <= k <= pygame.K_4:
                    app.load_preset(k - pygame.K_0)
            elif ev.type == pygame.KEYUP and ev.key == pygame.K_f:
                app.ctl.f_release(now)
            elif ev.type == pygame.MOUSEMOTION:
                w, inside = app.to_world(*ev.pos)
                app.ctl.cursor, app.ctl.cursor_in = w, inside
                if ev.buttons[0] and inside and prev_mouse is not None:
                    app.ctl.drag(prev_mouse, w, max(fdt, 1 / 240))
                prev_mouse = w if inside else None
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
                app.ctl.rmb = True
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 3:
                app.ctl.rmb = False
        app.ctl.update(now)
        app.ensure_buffers()
        if not app.paused:
            acc = min(acc + fdt, 2 * SIM_DT)
            while acc >= SIM_DT:
                with app.gpu_scope():
                    app.sim_tick()
                acc -= SIM_DT
        alpha = 1.0 if app.paused else acc / SIM_DT
        host = app.frame(alpha)
        surf = pygame.image.frombuffer(host, (WIN_W, WIN_H), "RGB")
        screen.blit(surf, (0, 0))
        app.draw_hud(screen, hud)
        pygame.display.flip()
        clock.tick(args.fps)
        if args.quit_after and now - t0 > args.quit_after:
            running = False
    st = app.stats(time.perf_counter() - t0)
    print(json.dumps({k: st[k] for k in ("frames", "fps_mean", "fps_1pct_low", "gpu_ms_per_frame", "stubs")}))
    if args.stats:
        Path(args.stats).write_text(json.dumps(st, indent=2))
    pygame.quit()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="neural-vfx playground")
    ap.add_argument("--headless", action="store_true", help="no window; run --frames and write outputs")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--preset", type=int, default=1, choices=sorted(PRESETS))
    ap.add_argument("--mode", default="neural", choices=MODES)
    ap.add_argument("--side-by-side", action="store_true")
    ap.add_argument("--screenshot", default=None)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--checkpoint", default=None, help="DetailNet checkpoint (default: paths.default_checkpoint())")
    ap.add_argument("--stub-sim", action="store_true", help="force the app's toy solver")
    ap.add_argument("--demo-input", action="store_true", help="headless: scripted drag/torch/paint input")
    ap.add_argument("--quit-after", type=float, default=0.0, help="windowed: exit after N seconds")
    ap.add_argument("--fps", type=int, default=60, help="windowed frame cap")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    if args.headless:
        run_headless(args)
    else:
        run_window(args)


if __name__ == "__main__":
    main()
