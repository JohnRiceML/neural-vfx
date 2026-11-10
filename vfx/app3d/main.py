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
from vfx.app3d.pipeline import LABELS, MODES, UNTRAINED, Pipeline
from vfx.app3d.presets import DEFAULT_OBSTACLE, PRESETS, fire_emitter, puff_emitter
from vfx.render3d import camera_basis

WIN_W, WIN_H = 1600, 900
BG = (8, 9, 12)
TAP_S = 0.25                 # F shorter than this = toggle; longer = torch
BUDGET_MS = 6.0
MARCH_SCALES = [1.0, 0.5, 0.25]


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
        self.b = load_backends(force_stub_sim=args.stub_sim, warmup=args.warmup_neural)
        self.pipe = Pipeline(self.b, march_scale=args.march_scale, steps=args.steps)
        self.ctl = Controls()
        self.orbit = Orbit()
        self.auto_orbit = args.orbit
        self.mode = args.mode
        self.mode2 = args.mode2
        self.side = args.side_by_side
        self.paused = False
        self.hud_on = True
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
        self.pipe.reset(self.preset.obstacles)

    def toggle_obstacle(self):
        self.pipe.set_obstacles([] if self.pipe.obstacles else (self.preset.obstacles or [DEFAULT_OBSTACLE]))

    def cycle(self, which: str):
        cur = getattr(self, which)
        setattr(self, which, MODES[(MODES.index(cur) + 1) % len(MODES)])

    def cycle_march_scale(self):
        i = (MARCH_SCALES.index(self.pipe.march_scale) + 1) % len(MARCH_SCALES) \
            if self.pipe.march_scale in MARCH_SCALES else 0
        self.pipe.set_march_scale(MARCH_SCALES[i])

    def layout(self):
        if self.side:
            hw = WIN_W // 2
            return [(self.mode, (0, 0, hw, WIN_H)), (self.mode2, (hw, 0, WIN_W - hw, WIN_H))]
        return [(self.mode, (0, 0, WIN_W, WIN_H))]

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
        if (self.side, self.mode, self.mode2) != getattr(self, "_last_layout", None):
            self._clear_canvas()
            self._last_layout = (self.side, self.mode, self.mode2)
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
    def budget(self, mode):
        t = self.pipe.timer
        sim = t.last("sim_ref") if mode == "ref128" else t.last("sim")
        up = {"trilinear": t.last("upscale"), "neural_vol": t.last("neural_vol")}.get(mode, 0.0)
        det = t.last("detail") if mode == "neural_screen" else 0.0
        return [("sim", sim), ("upscale", up), ("detail", det), ("render", t.last("render")),
                ("view", t.last("render_view"))]

    def stats_lines(self):
        ft = np.array(self.frame_ms)[30:][-600:]          # skip start-up frames (graph capture, kernel load)
        if not len(ft):
            ft = np.array([16.7])
        fps = 1000.0 / ft[-60:].mean()
        low = 1000.0 / np.sort(ft)[-max(1, len(ft) // 100):].mean()
        full = self.budget(self.mode)
        st = full[:4]                                   # tick frame: sim + upscale + detail + full render
        worst = sum(ms for _, ms in st)
        amort = 0.5 * (st[0][1] + st[1][1] + st[3][1] + full[4][1]) + st[2][1]
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        free, total = torch.cuda.mem_get_info()
        W, Y, G, R = (235, 235, 240), (255, 210, 120), (150, 220, 150), (255, 130, 120)
        r = self.pipe.renderers.get(((self.layout()[0][1][2], self.layout()[0][1][3]), 0))
        rr = f"{r.mw}x{r.mh} march -> {r.out_w}x{r.out_h}" if r else "-"
        lines = [
            (f"mode   {LABELS[self.mode]}" + (f"  |  {LABELS[self.mode2]}" if self.side else ""), Y),
            (f"preset {self.preset_id}: {self.preset.name}" + ("   [PAUSED]" if self.paused else ""), W),
            None,
            (f"fps {fps:6.1f}    1% low {low:6.1f}", G if low >= 50 else R),
            (f"GPU ms  sim {st[0][1]:5.2f}/tick   upscale {st[1][1]:5.2f}/tick", W),
            (f"        detail {st[2][1]:5.2f}   render {st[3][1]:5.2f} new vol / {full[4][1]:4.2f} reuse", W),
            (f"        render {rr}, scale {self.pipe.march_scale:g}", (180, 185, 200)),
            (f"total {worst:5.2f} ms tick-frame ({amort:4.2f} amortized) / {BUDGET_MS:.0f}", G if worst <= BUDGET_MS else R),
            (f"VRAM peak torch {peak:6.0f} MB   device used {(total - free) / 2**20:6.0f} MB", W),
            (f"sim step {self.pipe.sim_steps}   fires {len(self.ctl.fires)}" + ("   TORCH" if self.ctl.torch_on else ""), W),
        ]
        if self.b.stubs:
            lines.append(("stubs: " + ", ".join(self.b.stubs), R))
        for m in ("neural_vol", "neural_screen"):
            if not self.pipe.available(m):
                lines.append((f"{m}: unavailable ({self.pipe.unavailable[m][:46]})", R))
        return lines, st

    def draw_hud(self, surf, hud, force=False):
        for mode, (x, y, w, h) in self.layout():
            name = LABELS[mode]
            if not self.pipe.available(mode):
                hud.label(surf, name, (x + 12, y + 12), sub="UNAVAILABLE")
                hud.center_note(surf, (x, y, w, h), ["UNAVAILABLE", self.pipe.unavailable[mode][:70]])
            elif mode in UNTRAINED:
                hud.label(surf, name, (x + 12, y + 12), sub="UNTRAINED - timing only, not a result")
            else:
                hud.label(surf, name, (x + 12, y + 12))
        if not self.hud_on:
            return
        now = time.perf_counter()
        if force or self._hud_cache is None or now - self._hud_t > 0.25:
            self._hud_t = now
            lines, st = self.stats_lines()
            self._hud_cache = hud.panel(lines, {"stages": st, "limit": BUDGET_MS}, width=560)
        pos = (12, 72) if not self.side else (12, WIN_H - self._hud_cache.get_height() - 12)
        surf.blit(self._hud_cache, pos)

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
        r = self.pipe.renderers.get(((self.layout()[0][1][2], self.layout()[0][1][3]), 0))
        return {
            "frames": self.frames, "warmup_frames_excluded": warmup, "sim_steps": self.pipe.sim_steps, "wall_s": wall_s,
            "fps_mean": float(1000.0 / ft.mean()), "fps_1pct_low": float(1000.0 / worst.mean()),
            "frame_ms": {"mean": float(ft.mean()), "p99": float(np.percentile(ft, 99))},
            "gpu_ms": {k: agg(k) for k in ("sim", "sim_ref", "upscale", "neural_vol", "detail", "render", "render_view")},
            "gpu_ms_per_frame": {"mean": float(gf.mean()) if len(gf) else None,
                                 "p99": float(np.percentile(gf, 99)) if len(gf) else None},
            "blit_ms": {"mean": float(np.mean(self.blit_ms)), "p99": float(np.percentile(self.blit_ms, 99))},
            "vram_peak_torch_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
            "mode": self.mode, "mode2": self.mode2 if self.side else None, "side_by_side": self.side,
            "preset": self.preset_id, "march_scale": self.pipe.march_scale,
            "render_size": [r.out_w, r.out_h] if r else None, "march_size": [r.mw, r.mh] if r else None,
            "steps_per_m": r.p.steps if r else None,
            "unavailable": self.pipe.unavailable, "stubs": self.b.stubs, "backend_notes": self.b.notes,
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
                elif k == pygame.K_o:
                    app.toggle_obstacle()
                elif k == pygame.K_r:
                    app.load_preset(app.preset_id)
                elif k == pygame.K_SPACE:
                    app.paused = not app.paused
                elif k == pygame.K_h:
                    app.hud_on = not app.hud_on
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
    ap.add_argument("--warmup-neural", action="store_true",
                    help="pre-capture every VolumeUpscaler graph at start-up (slow: ~70 s on a shared GPU); "
                         "otherwise graphs are captured on first use of each brick count (hitches)")
    ap.add_argument("--demo-input", action="store_true", help="headless: scripted fire / push / puff input")
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
