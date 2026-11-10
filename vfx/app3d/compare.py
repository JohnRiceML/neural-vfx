"""Offline showcase comparison for route B (quality review), side by side, one orbiting camera:

    python -m vfx.app3d.compare --preset 1 --frames 120 --out <dir> [--size 960x540] [--every 1]
           [--warmup 90] [--orbit-deg 1.0] [--march-scale 0.5] [--screen-ckpt PATH] [--no-hr]

Panels (left to right), all rendered by the same VolumeRenderer settings and camera:
  1. raw 64³              render of the 64³ runtime sim (the sandbox's solver + preset dissipation)
  2. trilinear 256³       render of runtime3d.classical_upscale_3d of that state
  3. route B              ScreenDetail.infer(panel 1, aux) — aux = bbox + renderer alpha/depth/motion,
                          exactly what the sandbox feeds it; runs every sim step (temporal state)
  4. guided 256³ (ref)    render of vfx.datagen3d's 256³ sim guided to follow the 64³ run (the route B
                          training target); skipped cleanly if vfx.datagen3d.sim.run_pair is unavailable
With --detail (sim-carried detail phase; off by default, so the default output is unchanged) the
reference moves to the right and these panels are inserted before it:
  4. 64³ + detail          the raw 64³ state rendered with vfx.detail3d.AdvectedDetail (stepped with the
                           64³ velocity every sim step, including warm-up)
  5. detail + route B      ScreenDetail (own instance, own temporal state) on panel 4's render + its aux
  6. hires + detail        Solver3D(64, scalar_res=128).hires_volume() + its own AdvectedDetail; only if
                           vfx.sim3d supports it (else skipped with a note)
With --particles (off by default) two more panels go before the reference:
  particles 128³ 1M        vfx.particles3d.DetailParticles(1M, 128, config = the 64³ run's config), stepped
                           after every 64³ step with the same forces (warm-up included), its volume() rendered
  particles + detail       the same particle volume drawn with its own AdvectedDetail (64³ velocity)
One frame = one sim step (SIM_DT). Writes frame_XXXX.png (labelled strip, per-panel GPU ms),
contact_sheet.png (up to 8 frames x panels) and timings.json. ms are CUDA-event GPU times of the panel's
own work (render [+ upscale or + detail]); the guided 256³ sim cost is reported but is an offline reference.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vfx.core.fields3d import CH3, SIM_DT, Camera, Volume, rasterize_obstacles_3d

TARGET = (0.5, 0.42, 0.5)


def orbit_cam(deg: float, dist: float = 1.9, pitch: float = 0.2) -> Camera:
    a = math.radians(deg)
    pos = (TARGET[0] + dist * math.cos(pitch) * math.sin(a), TARGET[1] + dist * math.sin(pitch),
           TARGET[2] - dist * math.cos(pitch) * math.cos(a))
    return Camera(position=pos, target=TARGET, fov_y_deg=40.0)


class PresetScenario:
    """Adapter: app3d Preset -> the scenario interface vfx.datagen3d.sim.run_pair expects."""

    def __init__(self, pre, steps: int):
        self.sim, self.obstacles, self.num_steps = dict(pre.sim), list(pre.obstacles), steps
        self._pre = pre

    def forces(self, step: int):
        from vfx.core.fields3d import Forces3D
        return Forces3D(emitters=list(self._pre.emitters), wind=tuple(self._pre.wind))


def sim_stream(pre, steps: int, want_hr: bool, backends):
    """Yield (step, lr_state, hr_state or None). Guided pair from vfx.datagen3d if available."""
    note = None
    if want_hr:
        try:
            from vfx.datagen3d.sim import run_pair
            from vfx.datagen3d.guide3d import DEFAULT_GUIDE
            gen = run_pair(PresetScenario(pre, steps), DEFAULT_GUIDE)
            first = next(gen)
            def _g():
                yield first[0], first[1], first[2]
                for s, lr, hr, _ in gen:
                    yield s, lr, hr
            return _g(), "guided 256³: vfx.datagen3d.sim.run_pair (DEFAULT_GUIDE)"
        except Exception as e:                            # skip the panel cleanly
            note = f"guided 256³ panel skipped: {type(e).__name__}: {str(e)[:120]}"
    from vfx.core.fields3d import Forces3D
    s = backends.Solver(64, backends.runtime_config(), device="cuda")
    cfg = backends.runtime_config(**pre.sim) if pre.sim else backends.runtime_config()
    for attr in ("config", "cfg"):
        if hasattr(s, attr):
            setattr(s, attr, cfg)
    st = s.reset(pre.obstacles)

    def _g():
        nonlocal st
        for i in range(steps):
            st = s.step(st, Forces3D(emitters=list(pre.emitters), wind=pre.wind), SIM_DT)
            yield i, st, None
    return _g(), note or "guided 256³ panel off (--no-hr)"


def screen_bbox(cam: Camera, w: int, h: int):
    from vfx.render3d import camera_basis
    b = camera_basis(cam, w / h)
    o, fw, rt, up, th = b[0:3], b[3:6], b[6:9], b[9:12], b[12]
    c = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], np.float32) - o
    zc = c @ fw
    if (zc <= 1e-3).any():
        return (0, 0, w, h)
    px = ((c @ rt) / zc / (th * w / h) + 1) * 0.5 * w
    py = (1 - (c @ up) / zc / th) * 0.5 * h
    return (int(np.clip(px.min(), 0, w)), int(np.clip(py.min(), 0, h)),
            int(np.ceil(np.clip(px.max(), 0, w))), int(np.ceil(np.clip(py.max(), 0, h))))


class Timer:
    def __init__(self):
        self.pending, self.acc = [], {}

    def run(self, name, fn):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        out = fn()
        b.record()
        self.pending.append((name, a, b))
        return out

    def resolve(self) -> dict:
        last = {}
        torch.cuda.synchronize()        # events must have completed before elapsed_time()
        for n, a, b in self.pending:
            ms = a.elapsed_time(b)
            last[n] = last.get(n, 0.0) + ms
        for n, ms in last.items():
            self.acc.setdefault(n, []).append(ms)
        self.pending.clear()
        return last


class ParticlePanels:
    """--particles: the particle study as panels (vfx.particles3d; its README for the method)."""

    def __init__(self, a, pre, backends, W, H):
        from vfx.detail3d import AdvectedDetail
        from vfx.particles3d import DetailParticles
        from vfx.render3d import VolumeRenderer
        cfg = backends.runtime_config(**pre.sim) if pre.sim else backends.runtime_config()   # == run_pair's LR config
        from vfx.detail3d import STACKED
        self.dp = DetailParticles(1_000_000, 128, config=cfg)
        self.det = AdvectedDetail(64, quality=a.detail_quality, **STACKED["particles"])   # base-aware
        self.rend = {k: VolumeRenderer((H, W), march_scale=a.march_scale) for k in ("p", "pd")}
        self.names = ["particles 128³ 1M", f"particles + detail ({a.detail_quality})"]
        self.pre = pre
        self.imgs = {}

    def _forces(self):
        from vfx.core.fields3d import Forces3D
        return Forces3D(emitters=list(self.pre.emitters), wind=tuple(self.pre.wind))

    def _cov(self):
        dep = getattr(self.dp, "dep", None)
        return dep[2] if dep is not None else None          # particles per 128³ cell this tick

    def warm(self, lr):
        self.dp.step(lr, self._forces(), SIM_DT)
        self.det.step(lr, SIM_DT, coverage=self._cov())

    def tick(self, tm, lr, cam, solid):
        tm.run("particles step", lambda: self.dp.step(lr, self._forces(), SIM_DT))
        tm.run("particles detail step", lambda: self.det.step(lr, SIM_DT, coverage=self._cov()))
        v = self.dp.volume()
        self.imgs["p"] = tm.run("particles render", lambda: self.rend["p"].draw(v, cam, solid)).clone()
        self.imgs["pd"] = tm.run("particles+detail render",
                                 lambda: self.rend["pd"].draw(v, cam, solid, detail=self.det)).clone()

    def panels(self, ms):
        ps = ms.get("particles step", 0)
        return ([self.imgs.get("p"), self.imgs.get("pd")],
                [ps + ms.get("particles render", 0),
                 ps + ms.get("particles detail step", 0) + ms.get("particles+detail render", 0)],
                [f"particles {ps:.2f} + render {ms.get('particles render', 0):.2f} ms",
                 f"particles {ps:.2f} + detail {ms.get('particles detail step', 0):.2f} + render "
                 f"{ms.get('particles+detail render', 0):.2f}"])


class DetailPanels:
    """--detail: advected-detail panels (sim-carried detail phase). Stepped every sim step (warm-up
    included) so the detail coordinates are as old as in the sandbox."""

    def __init__(self, a, pre, backends, W, H, rb_label):
        from vfx.detail3d import AdvectedDetail
        from vfx.render3d import VolumeRenderer
        self.notes = []
        self.det = AdvectedDetail(64, quality=a.detail_quality)
        self.names = [f"64³ + detail ({a.detail_quality})"]
        self.rend = {"det": VolumeRenderer((H, W), march_scale=a.march_scale)}
        self.sd = None
        if not a.no_route_b_detail:
            from vfx.app3d.backends import make_screen_detail
            self.sd, _ = make_screen_detail(a.screen_ckpt)
            self.names.append("detail + " + rb_label)
        self.hs = self.hst = self.hdet = None
        try:
            import inspect
            S = backends.Solver
            if "scalar_res" in inspect.signature(S.__init__).parameters and hasattr(S, "hires_volume"):
                cfg = backends.runtime_config(**pre.sim) if pre.sim else backends.runtime_config()
                self.hs = S(64, cfg, device="cuda", scalar_res=128)
                self.hst = self.hs.reset(pre.obstacles)
                from vfx.detail3d import STACKED
                self.hdet = AdvectedDetail(64, quality=a.detail_quality, **STACKED["hires"])   # base-aware
                self.rend["hires"] = VolumeRenderer((H, W), march_scale=a.march_scale)
                self.names.append(f"hires 128³ + detail ({a.detail_quality})")
            else:
                self.notes.append("hires + detail panel skipped: sim3d has no scalar_res / hires_volume()")
        except Exception as e:
            self.hs = None
            self.notes.append(f"hires + detail panel skipped: {type(e).__name__}: {str(e)[:100]}")
        self.pre = pre
        self.W, self.H = W, H
        self.imgs = {}

    def warm(self, lr):
        from vfx.core.fields3d import Forces3D
        self.det.step(lr, SIM_DT)
        if self.hs is not None:
            self.hst = self.hs.step(self.hst, Forces3D(emitters=list(self.pre.emitters), wind=tuple(self.pre.wind)), SIM_DT)
            self.hdet.step(self.hst, SIM_DT)

    def tick(self, tm, lr, cam, prev_cam, solid, step):
        from vfx.core.fields3d import Forces3D
        tm.run("detail step", lambda: self.det.step(lr, SIM_DT))
        vol = Volume(lr.data[CH3["density"]], lr.data[CH3["temperature"]])
        r = self.rend["det"]
        if self.sd is not None:
            img, aux = tm.run("detail render", lambda: r.draw(vol, cam, solid, return_aux=True, velocity=lr.velocity,
                                                              prev_camera=prev_cam, detail=self.det))
            img = img.clone()
            trans = r.hdr[..., 3]
            if trans.shape != (self.H, self.W):
                trans = F.interpolate(trans[None, None], size=(self.H, self.W), mode="bilinear", align_corners=False)[0, 0]
            aux = {"bbox": screen_bbox(cam, self.W, self.H), "transmittance": trans, **aux}
            self.imgs["det"] = img
            self.imgs["detB"] = tm.run("detail route B", lambda: self.sd.infer(img, aux)).clone()
        else:
            self.imgs["det"] = tm.run("detail render", lambda: r.draw(vol, cam, solid, detail=self.det)).clone()
        if self.hs is not None:
            f = Forces3D(emitters=list(self.pre.emitters), wind=tuple(self.pre.wind))
            self.hst = tm.run("hires sim", lambda: self.hs.step(self.hst, f, SIM_DT))
            tm.run("hires detail step", lambda: self.hdet.step(self.hst, SIM_DT))
            hv = self.hs.hires_volume()
            if hv is not None:
                self.imgs["hires"] = tm.run("hires render", lambda: self.rend["hires"].draw(hv, cam, solid, detail=self.hdet)).clone()

    def panels(self, ms):
        p, m, s = [self.imgs.get("det")], [ms.get("detail step", 0) + ms.get("detail render", 0)], \
            [f"step {ms.get('detail step', 0):.2f} + render {ms.get('detail render', 0):.2f} ms"]
        if self.sd is not None:
            p.append(self.imgs.get("detB"))
            m.append(m[0] + ms.get("detail route B", 0))
            s.append(f"detail panel + net {ms.get('detail route B', 0):.2f} ms")
        if self.hs is not None:
            p.append(self.imgs.get("hires"))
            m.append(ms.get("hires detail step", 0) + ms.get("hires render", 0))
            s.append(f"sim {ms.get('hires sim', 0):.2f} | step {ms.get('hires detail step', 0):.2f} + render {ms.get('hires render', 0):.2f}")
        return p, m, s


def to_surface(img: torch.Tensor):
    import pygame
    h, w = img.shape[:2]
    return pygame.image.frombuffer(img.contiguous().cpu().numpy().tobytes(), (w, h), "RGB")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", type=int, default=1)
    ap.add_argument("--frames", type=int, default=120, help="recorded sim steps (after --warmup)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="960x540", help="panel size WxH")
    ap.add_argument("--every", type=int, default=1, help="write a PNG every N recorded steps")
    ap.add_argument("--warmup", type=int, default=90, help="sim steps before recording (plume build-up)")
    ap.add_argument("--orbit-deg", type=float, default=1.0, help="camera orbit per sim step (degrees)")
    ap.add_argument("--start-deg", type=float, default=-30.0)
    ap.add_argument("--dist", type=float, default=1.9)
    ap.add_argument("--march-scale", type=float, default=0.5)
    ap.add_argument("--screen-ckpt", default=None, help="route B checkpoint (default: runtime3d default path if present)")
    ap.add_argument("--no-hr", action="store_true", help="skip the guided 256³ reference panel")
    ap.add_argument("--detail", action="store_true", help="add the advected-detail panels (see module doc)")
    ap.add_argument("--detail-quality", default="calibrated", choices=["calibrated", "low", "medium", "high"])
    ap.add_argument("--no-route-b-detail", action="store_true", help="with --detail: skip the detail + route B panel")
    ap.add_argument("--particles", action="store_true", help="add the particle-study panels (see module doc)")
    a = ap.parse_args(argv)

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    pygame.font.init()
    font = pygame.font.SysFont("consolas", 18, bold=True)
    small = pygame.font.SysFont("consolas", 15)

    from vfx.app3d.backends import load_backends, make_screen_detail, route_b_checkpoint
    from vfx.app3d.presets import PRESETS
    from vfx.render3d import VolumeRenderer
    W, H = (int(v) for v in a.size.lower().split("x"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pre = PRESETS[a.preset]
    b = load_backends(warmup=False)
    if b.stubs:
        print("WARNING: stub backends:", b.stubs, b.notes)
    sd, ck_name = make_screen_detail(a.screen_ckpt)
    rb_label = f"route B ({ck_name})" if ck_name else "route B (UNTRAINED)"
    print(f"route B: {ck_name or 'untrained'} (default checkpoint path: {route_b_checkpoint()})")

    stream, hr_note = sim_stream(pre, a.warmup + a.frames, not a.no_hr, b)
    print(hr_note.replace("³", "^3"))
    solid = rasterize_obstacles_3d(pre.obstacles, 128) if pre.obstacles else None
    rend = {k: VolumeRenderer((H, W), march_scale=a.march_scale) for k in ("raw", "tri", "hr")}
    tm = Timer()
    names = ["raw 64³", "trilinear 256³", rb_label, "guided 256³ (reference)"]
    dx = None
    if a.detail:
        dx = DetailPanels(a, pre, b, W, H, rb_label)
        names = names[:3] + dx.names + names[3:]
    px = None
    if a.particles:
        px = ParticlePanels(a, pre, b, W, H)
        names = names[:-1] + px.names + names[-1:]
    sheet_frames: list = []
    prev_cam = None
    has_hr = False
    pick = set(np.linspace(0, a.frames - 1, min(8, a.frames)).round().astype(int).tolist())
    for step, lr, hr in stream:
        k = step - a.warmup
        if k < 0:
            if dx is not None:
                dx.warm(lr)
            if px is not None:
                px.warm(lr)
            continue
        cam = orbit_cam(a.start_deg + a.orbit_deg * k, a.dist)
        save = k % a.every == 0 or k in pick
        raw = Volume(lr.data[CH3["density"]], lr.data[CH3["temperature"]])
        img_raw, aux = tm.run("raw", lambda: rend["raw"].draw(raw, cam, solid, return_aux=True,
                                                              velocity=lr.velocity, prev_camera=prev_cam))
        img_raw = img_raw.clone()
        trans = rend["raw"].hdr[..., 3]                    # legacy key, as Pipeline._screen_aux passes it
        if trans.shape != (H, W):
            trans = F.interpolate(trans[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        aux = {"bbox": screen_bbox(cam, W, H), "transmittance": trans, **aux}
        img_rb = tm.run("route B detail", lambda: sd.infer(img_raw, aux))
        if dx is not None:
            dx.tick(tm, lr, cam, prev_cam, solid, step)
        if px is not None:
            px.tick(tm, lr, cam, solid)
        prev_cam = cam
        if not save:
            tm.resolve()
            continue
        panels = [img_raw, None, img_rb.clone(), None]
        vt = tm.run("trilinear upscale", lambda: b.classical_upscale(lr, 4, "trilinear"))
        panels[1] = tm.run("trilinear render", lambda: rend["tri"].draw(vt, cam, solid)).clone()
        if hr is not None:
            has_hr = True
            vh = Volume(hr.data[CH3["density"]], hr.data[CH3["temperature"]])
            panels[3] = tm.run("hr render", lambda: rend["hr"].draw(vh, cam, solid)).clone()
        torch.cuda.synchronize()
        ms = tm.resolve()
        pms = [ms.get("raw", 0), ms.get("trilinear upscale", 0) + ms.get("trilinear render", 0),
               ms.get("raw", 0) + ms.get("route B detail", 0), ms.get("hr render", 0)]
        subs = [f"render {ms.get('raw', 0):.2f} ms (incl. aux)",
                f"upscale {ms.get('trilinear upscale', 0):.2f} + render {ms.get('trilinear render', 0):.2f} ms",
                f"raw render {ms.get('raw', 0):.2f} + detail {ms.get('route B detail', 0):.2f} ms",
                f"render {ms.get('hr render', 0):.2f} ms (offline sim)"]
        if dx is not None:
            p_, m_, s_ = dx.panels(ms)
            panels = panels[:3] + p_ + panels[3:]
            pms = pms[:3] + m_ + pms[3:]
            subs = subs[:3] + s_ + subs[3:]
        if px is not None:
            p_, m_, s_ = px.panels(ms)
            panels = panels[:-1] + p_ + panels[-1:]
            pms = pms[:-1] + m_ + pms[-1:]
            subs = subs[:-1] + s_ + subs[-1:]
        n = len(names) if has_hr else len(names) - 1
        strip = pygame.Surface((n * W, H))
        for i in range(n):
            if panels[i] is None:
                continue
            strip.blit(to_surface(panels[i]), (i * W, 0))
            t1 = font.render(names[i], True, (255, 230, 150))
            t2 = small.render(f"{subs[i]}   = {pms[i]:.2f} ms", True, (210, 215, 225))
            bg = pygame.Surface((max(t1.get_width(), t2.get_width()) + 14, 48), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 150))
            strip.blit(bg, (i * W + 8, 8))
            strip.blit(t1, (i * W + 15, 11))
            strip.blit(t2, (i * W + 15, 33))
            if i:
                pygame.draw.line(strip, (90, 90, 100), (i * W, 0), (i * W, H), 2)
        ft = small.render(f"preset {a.preset} {pre.name}   step {step}   cam {a.start_deg + a.orbit_deg * k:.0f} deg",
                          True, (170, 175, 185))
        strip.blit(ft, (10, H - 24))
        pygame.image.save(strip, str(out / f"frame_{k:04d}.png"))
        if k in pick:
            sheet_frames.append(pygame.transform.smoothscale(strip, (n * W // 2, H // 2)))
        print(f"frame {k:4d}: " + "  ".join(f"{nm.split(' (')[0]} {v:.2f}" for nm, v in zip(names[:n], pms[:n])).replace("³", "^3") + " ms")
    if sheet_frames:
        sw, sh = sheet_frames[0].get_size()
        sheet = pygame.Surface((sw, sh * len(sheet_frames)))
        for i, f in enumerate(sheet_frames):
            sheet.blit(f, (0, i * sh))
        pygame.image.save(sheet, str(out / "contact_sheet.png"))
    # drop the first 2 samples per stage (CUDA-graph capture, kernel load)
    summary = {n_: {"mean_ms": float(np.mean(v[2:] or v)), "median_ms": float(np.median(v[2:] or v)), "n": len(v[2:] or v)}
               for n_, v in tm.acc.items()}
    meta = {"preset": a.preset, "frames": a.frames, "warmup": a.warmup, "size": [W, H], "march_scale": a.march_scale,
            "route_b": ck_name or "untrained", "hr": hr_note, "stubs": b.stubs, "ms": summary}
    if dx is not None:
        meta["detail"] = {"quality": a.detail_quality, "panels": dx.names, "notes": dx.notes}
    if px is not None:
        meta["particles"] = {"count": 1_000_000, "out_res": 128, "panels": px.names}
    (out / "timings.json").write_text(json.dumps(meta, indent=2))
    from vfx.render3d.bench import gpu_note
    meta["gpu"] = gpu_note()
    (out / "timings.json").write_text(json.dumps(meta, indent=2))
    print("GPU:", meta["gpu"][:60], "(shared if util is high / other python jobs)")
    print("per-stage GPU ms (median, first 2 frames dropped):")
    for n_, v in summary.items():
        print(f"  {n_:18s} {v['median_ms']:6.2f}  (n={v['n']})")
    print("->", out)


if __name__ == "__main__":
    main()
