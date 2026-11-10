"""Side-by-side quality review of particle-carried detail against the steered 256^3 reference.

    python -m vfx.particles3d.compare --preset 2 --frames 12 --out DIR [--size 640x360] [--warmup 90]
           [--count 1000000] [--res 128,192] [--orbit-deg 1.0] [--every 1] [--param k=v ...]

Panels (same VolumeRenderer settings and camera, as vfx.app3d.compare):
  raw 64^3 | trilinear 256^3 | particles <res>^3 (one panel per --res) | steered 256^3 (reference)
The 64^3 run and its steered 256^3 twin come from vfx.datagen3d.sim.run_pair (via vfx.app3d.compare.sim_stream,
read-only). Each particle panel owns a DetailParticles(count, res) stepped every sim tick from the 64^3 state.
Writes frame_XXXX.png (labelled strip, per-panel GPU ms), contact_sheet.png, spectrum.png (radial luminance power
spectrum of each panel relative to the reference) and metrics.json (spectrum error / log ratio, tLP).
ms = CUDA-event GPU time of the panel's own work: particle step (advect+seed+deposit+compose) + render.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from vfx.core.fields3d import CH3, SIM_DT, Forces3D, Volume, rasterize_obstacles_3d


def _parse_params(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _spectrum_plot(path, spectra: dict, names: list, colors: list):
    import pygame
    from vfx.eval3d.metrics import BIN_EDGES
    W, H, L, B = 760, 440, 70, 50
    s = pygame.Surface((W, H))
    s.fill((20, 22, 28))
    font = pygame.font.SysFont("consolas", 14)
    fc = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
    lx = np.log10(fc)
    ref = spectra[names[-1]]
    ratios = {n: np.log10(spectra[n] + 1e-12) - np.log10(ref + 1e-12) for n in names}
    lo, hi = -2.0, 1.0
    X = lambda v: L + (v - lx[0]) / (lx[-1] - lx[0]) * (W - L - 20)
    Y = lambda v: 20 + (hi - np.clip(v, lo, hi)) / (hi - lo) * (H - 20 - B)
    for g in np.arange(lo, hi + 0.01, 0.5):
        pygame.draw.line(s, (60, 60, 70) if g else (140, 140, 150), (L, Y(g)), (W - 20, Y(g)), 1)
        s.blit(font.render(f"{g:+.1f}", True, (180, 180, 190)), (8, Y(g) - 7))
    for f in (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2):
        x = X(np.log10(f))
        pygame.draw.line(s, (50, 50, 60), (x, 20), (x, H - B), 1)
        s.blit(font.render(f"1/{int(round(1 / f))}", True, (180, 180, 190)), (x - 14, H - B + 6))
    s.blit(font.render("log10 P / P_ref  vs  cycles/pixel (luminance, effect bbox)", True, (220, 220, 230)), (L, H - 22))
    for i, n in enumerate(names):
        r = ratios[n]
        pts = [(X(a), Y(b)) for a, b in zip(lx, r) if np.isfinite(b)]
        if len(pts) > 1:
            pygame.draw.lines(s, colors[i], False, pts, 2)
        s.blit(font.render(n, True, colors[i]), (W - 260, 26 + 16 * i))
    pygame.image.save(s, str(path))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", type=int, default=2)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="640x360")
    ap.add_argument("--warmup", type=int, default=90)
    ap.add_argument("--count", type=int, default=1_000_000)
    ap.add_argument("--res", default="128,192")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--orbit-deg", type=float, default=1.0)
    ap.add_argument("--start-deg", type=float, default=-30.0)
    ap.add_argument("--dist", type=float, default=1.9)
    ap.add_argument("--march-scale", type=float, default=0.5)
    ap.add_argument("--param", nargs="*", help="DetailParticles params, k=v (JSON values)")
    ap.add_argument("--no-hr", action="store_true")
    ap.add_argument("--cache", default=os.environ.get("PARTICLES3D_CACHE", "D:/neural-vfx-data/scratch/particles3d"),
                    help="where rendered reference frames are cached (the 256^3 run happens once per setting)")
    a = ap.parse_args(argv)

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    pygame.font.init()
    font = pygame.font.SysFont("consolas", 16, bold=True)
    small = pygame.font.SysFont("consolas", 13)

    from vfx.app3d.backends import load_backends
    from vfx.app3d.compare import Timer, orbit_cam, sim_stream, to_surface
    from vfx.app3d.presets import PRESETS
    from vfx.eval3d import metrics as M
    from vfx.render3d import VolumeRenderer
    from vfx.sim3d import runtime_config

    from .particles import DetailParticles

    W, H = (int(v) for v in a.size.lower().split("x"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pre = PRESETS[a.preset]
    params = _parse_params(a.param)
    b = load_backends(warmup=False)
    cfg = runtime_config(**pre.sim)
    resl = [int(r) for r in a.res.split(",") if r]
    dps = {r: DetailParticles(a.count, r, config=cfg, **params) for r in resl}
    # the steered 256^3 run is expensive (3.4 GB VRAM): render its frames once per setting and cache them (uint8)
    ck = dict(p=a.preset, w=a.warmup, f=a.frames, s=f"{W}x{H}", o=a.orbit_deg, sd=a.start_deg, d=a.dist, m=a.march_scale)
    cache = Path(a.cache) / ("ref_" + "_".join(f"{k}{v}" for k, v in ck.items()) + ".pt")
    ref_cache = None
    if not a.no_hr and cache.exists():
        ref_cache = torch.load(cache)
        print("reference frames from cache:", cache)
    if ref_cache is not None or a.no_hr:
        from vfx.app3d.compare import PresetScenario
        from vfx.datagen3d.sim import run_pair
        gen = run_pair(PresetScenario(pre, a.warmup + a.frames), None, with_hr=False)
        stream = ((s_, lr_, None) for s_, lr_, _, _ in gen)
        hr_note = "steered 256^3: cached frames" if ref_cache is not None else "no reference (--no-hr)"
    else:
        stream, hr_note = sim_stream(pre, a.warmup + a.frames, True, b)
    new_ref = {"frames": [], "alpha": []}
    print(hr_note.replace("³", "^3"))
    solid = rasterize_obstacles_3d(pre.obstacles, 128) if pre.obstacles else None
    keys = ["raw", "tri"] + [f"p{r}" for r in resl] + ["hr"]
    names = ["raw 64^3", "trilinear 256^3"] + [f"particles {r}^3 ({a.count / 1e6:g}M)" for r in resl] + \
            ["steered 256^3 (reference)"]
    colors = [(150, 150, 160), (90, 160, 255)] + [(255, 170, 60), (120, 230, 120), (230, 110, 230)][:len(resl)] + \
             [(255, 255, 255)]
    rend = {k: VolumeRenderer((H, W), march_scale=a.march_scale) for k in keys}
    tm = Timer()
    seqs = {k: [] for k in keys}
    alphas = []
    sheet = []
    pick = set(np.linspace(0, a.frames - 1, min(8, a.frames)).round().astype(int).tolist())
    diag = {}
    forces = Forces3D(emitters=list(pre.emitters), wind=tuple(pre.wind))
    for step, lr, hr in stream:
        for r, dp in dps.items():
            tm.run(f"p{r} step", lambda: dp.step(lr, forces, SIM_DT))
        k = step - a.warmup
        if k < 0:
            tm.resolve()
            continue
        cam = orbit_cam(a.start_deg + a.orbit_deg * k, a.dist)
        panels = {}
        panels["raw"] = tm.run("raw render", lambda: rend["raw"].draw(
            Volume(lr.data[CH3["density"]], lr.data[CH3["temperature"]]), cam, solid)).clone()
        vt = tm.run("tri upscale", lambda: b.classical_upscale(lr, 4, "trilinear"))
        panels["tri"] = tm.run("tri render", lambda: rend["tri"].draw(vt, cam, solid)).clone()
        for r, dp in dps.items():
            panels[f"p{r}"] = tm.run(f"p{r} render", lambda: rend[f"p{r}"].draw(dp.volume(), cam, solid)).clone()
        if hr is not None:
            img, aux = rend["hr"].draw(Volume(hr.data[CH3["density"]], hr.data[CH3["temperature"]]), cam, solid,
                                       return_aux=True)
            panels["hr"] = img.clone()
            alphas.append(aux["alpha"].float().clone())
            new_ref["frames"].append(img.cpu().clone())
            new_ref["alpha"].append(aux["alpha"].cpu().clone())
        elif ref_cache is not None:
            panels["hr"] = ref_cache["frames"][k].cuda()
            alphas.append(ref_cache["alpha"][k].cuda().float())
        ms = tm.resolve()
        pms = {"raw": (ms["raw render"], f"render {ms['raw render']:.2f}"),
               "tri": (ms["tri upscale"] + ms["tri render"],
                       f"upscale {ms['tri upscale']:.2f} + render {ms['tri render']:.2f}")}
        for r in resl:
            st, rr = ms[f"p{r} step"], ms[f"p{r} render"]
            pms[f"p{r}"] = (st + rr, f"particles {st:.2f} + render {rr:.2f}")
        pms["hr"] = (0.0, "offline 256^3 sim")
        for kk, im in panels.items():
            seqs[kk].append(im.permute(2, 0, 1).contiguous())
        if k % a.every == 0 or k in pick:
            n = len(panels)
            strip = pygame.Surface((n * W, H))
            for i, kk in enumerate([x for x in keys if x in panels]):
                strip.blit(to_surface(panels[kk]), (i * W, 0))
                t1 = font.render(names[keys.index(kk)], True, (255, 230, 150))
                ms_, sub = pms[kk]
                t2 = small.render(f"{sub} = {ms_:.2f} ms" if kk != "hr" else sub, True, (210, 215, 225))
                bg = pygame.Surface((max(t1.get_width(), t2.get_width()) + 12, 42), pygame.SRCALPHA)
                bg.fill((0, 0, 0, 150))
                strip.blit(bg, (i * W + 6, 6))
                strip.blit(t1, (i * W + 12, 9))
                strip.blit(t2, (i * W + 12, 29))
                if i:
                    pygame.draw.line(strip, (90, 90, 100), (i * W, 0), (i * W, H), 2)
            strip.blit(small.render(f"preset {a.preset} {pre.name}  step {step}  cam {a.start_deg + a.orbit_deg * k:.0f} deg"
                                    f"  (GPU shared)", True, (170, 175, 185)), (8, H - 20))
            pygame.image.save(strip, str(out / f"frame_{k:04d}.png"))
            if k in pick:
                sheet.append(pygame.transform.smoothscale(strip, (n * W // 2, H // 2)))
        print(f"frame {k:3d}: " + "  ".join(f"{kk} {pms[kk][0]:.2f}" for kk in keys if kk in panels) + " ms")
        if k == a.frames - 1:
            for r, dp in dps.items():
                diag[f"p{r}"] = dp.diagnostics(lr)
    if new_ref["frames"]:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k_: torch.stack(v) for k_, v in new_ref.items()}, cache)
        print("cached reference frames ->", cache)
    if sheet:
        sw, sh = sheet[0].get_size()
        cs = pygame.Surface((sw, sh * len(sheet)))
        for i, f in enumerate(sheet):
            cs.blit(f, (0, i * sh))
        pygame.image.save(cs, str(out / "contact_sheet.png"))
    metrics = {}
    if seqs["hr"]:
        mask = M.effect_mask(torch.stack(alphas))
        ref = torch.stack(seqs["hr"])
        spectra = {}
        for kk in keys:
            x = torch.stack(seqs[kk])
            p_, _ = M.radial_power_spectrum(x, mask)
            spectra[names[keys.index(kk)]] = p_
            if kk == "hr":
                continue
            d = dict(spectrum_error=M.spectrum_error(x, ref, mask), spectrum_log_ratio=M.spectrum_log_ratio(x, ref, mask),
                     hf_power_ratio=M.band_power(x, mask, 0.0625, 0.5) / max(M.band_power(ref, mask, 0.0625, 0.5), 1e-20))
            if len(x) >= 3:
                t = M.tlp(x, ref, mask)
                d.update({kk2: float(v) for kk2, v in t.items() if np.isscalar(v) or np.ndim(v) == 0})
            metrics[kk] = d
        _spectrum_plot(out / "spectrum.png", spectra, names, colors)
    summary = {n_: float(np.median(v[2:] or v)) for n_, v in tm.acc.items()}
    from vfx.render3d.bench import gpu_note
    meta = dict(preset=a.preset, frames=a.frames, warmup=a.warmup, size=[W, H], count=a.count, res=resl,
                params=params, hr=hr_note, median_ms=summary, metrics=metrics, diagnostics=diag, gpu=gpu_note())
    (out / "metrics.json").write_text(json.dumps(meta, indent=2, default=float))
    print(json.dumps({"median_ms": summary, "metrics": metrics, "diagnostics": diag}, indent=1, default=float))
    print("->", out)


if __name__ == "__main__":
    main()
