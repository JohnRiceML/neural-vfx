"""Side-by-side quality review of particle-carried detail against the steered 256^3 reference.

    python -m vfx.particles3d.compare --preset 2 --frames 12 --out DIR [--size 640x360] [--warmup 90]
           [--count 1000000] [--res 128,192] [--variants "label:count=1e6,res=128,sg_amp=0.6;..."]
           [--orbit-deg 1.0] [--every 1] [--param k=v ...] [--no-tri] [--no-detail] [--cols N]

Panels (same VolumeRenderer settings and camera, as vfx.app3d.compare):
  raw 64^3 | trilinear 256^3 | 64^3 + advected noise (vfx.detail3d, read-only) | particle variants | steered 256^3
The 64^3 run and its steered 256^3 twin come from vfx.datagen3d.sim.run_pair (read-only). The 256^3 run is done
once per (preset, frames, camera, size) setting: its rendered frames + alpha are cached (uint8, --cache, default
D:/neural-vfx-data/scratch/particles3d) and later runs only re-run the 64^3 sim (bit-identical). Each particle panel
owns a DetailParticles(count, res) stepped every sim tick from the 64^3 state.
Writes frame_XXXX.png (labelled panels, per-panel GPU ms), contact_sheet.png, spectrum.png (radial luminance power
spectrum of each panel relative to the reference) and metrics.json (spectrum error / log ratio, tLP, diagnostics).
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
    W, H, L, B = 860, 460, 70, 50
    s = pygame.Surface((W, H))
    s.fill((20, 22, 28))
    font = pygame.font.SysFont("consolas", 14)
    fc = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
    lx = np.log10(fc)
    ref = spectra[names[-1]]
    ratios = {n: np.log10(spectra[n] + 1e-12) - np.log10(ref + 1e-12) for n in names}
    lo, hi = -2.0, 1.0
    X = lambda v: L + (v - lx[0]) / (lx[-1] - lx[0]) * (W - 330)
    Y = lambda v: 20 + (hi - np.clip(v, lo, hi)) / (hi - lo) * (H - 20 - B)
    for g in np.arange(lo, hi + 0.01, 0.5):
        pygame.draw.line(s, (60, 60, 70) if g else (140, 140, 150), (L, Y(g)), (W - 280, Y(g)), 1)
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
        s.blit(font.render(n[:34], True, colors[i]), (W - 270, 26 + 16 * i))
    pygame.image.save(s, str(path))


def _variants(a):
    """--variants "label:count=1e6,res=128,sg_amp=0.6;label2:..." -> [(label, count, res, params)]."""
    base = _parse_params(a.param)
    if not a.variants:
        return [(f"particles {r}^3 ({a.count / 1e6:g}M)", a.count, r, dict(base)) for r in
                [int(x) for x in a.res.split(",") if x]]
    out = []
    for item in a.variants.split(";"):
        if not item.strip():
            continue
        label, _, kv = item.partition(":")
        prm = dict(base)
        prm.update(_parse_params([x for x in kv.split(",") if x]))
        count = int(float(prm.pop("count", a.count)))
        res = int(prm.pop("res", 128))
        out.append((label.strip(), count, res, prm))
    return out


def _palette(n):
    base = [(150, 150, 160), (90, 160, 255), (80, 220, 220), (255, 170, 60), (120, 230, 120), (230, 110, 230),
            (250, 90, 90), (200, 200, 90), (160, 120, 255), (255, 140, 190), (120, 180, 140), (240, 200, 160)]
    return [base[i % len(base)] for i in range(n - 1)] + [(255, 255, 255)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", type=int, default=2)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="640x360")
    ap.add_argument("--warmup", type=int, default=90)
    ap.add_argument("--count", type=int, default=1_000_000)
    ap.add_argument("--res", default="128,192")
    ap.add_argument("--variants", default=None, help='"label:count=1e6,res=128,sg_amp=0.6;label2:..." (overrides --res)')
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--orbit-deg", type=float, default=1.0)
    ap.add_argument("--start-deg", type=float, default=-30.0)
    ap.add_argument("--dist", type=float, default=1.9)
    ap.add_argument("--march-scale", type=float, default=0.5)
    ap.add_argument("--param", nargs="*", help="DetailParticles params for every variant, k=v (JSON values)")
    ap.add_argument("--no-tri", action="store_true", help="drop the trilinear 256^3 panel")
    ap.add_argument("--no-detail", action="store_true", help="drop the vfx.detail3d advected-noise panel")
    ap.add_argument("--detail-quality", default="medium")
    ap.add_argument("--cols", type=int, default=0, help="panels per row (0 = one row)")
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
    from vfx.app3d.compare import PresetScenario, Timer, orbit_cam, sim_stream, to_surface
    from vfx.app3d.presets import PRESETS
    from vfx.eval3d import metrics as M
    from vfx.render3d import VolumeRenderer
    from vfx.sim3d import runtime_config

    from .particles import DetailParticles

    W, H = (int(v) for v in a.size.lower().split("x"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pre = PRESETS[a.preset]
    b = load_backends(warmup=False)
    cfg = runtime_config(**pre.sim)
    var = _variants(a)
    dps = {f"v{i}": DetailParticles(c, r, config=cfg, **prm) for i, (_, c, r, prm) in enumerate(var)}
    det = None
    if not a.no_detail:
        from vfx.detail3d import AdvectedDetail
        det = AdvectedDetail(64, quality=a.detail_quality)
    # the steered 256^3 run is expensive (3.4 GB VRAM): render its frames once per setting and cache them (uint8)
    ck = dict(p=a.preset, w=a.warmup, f=a.frames, s=f"{W}x{H}", o=a.orbit_deg, sd=a.start_deg, d=a.dist,
              m=a.march_scale)
    cache = Path(a.cache) / ("ref_" + "_".join(f"{k}{v}" for k, v in ck.items()) + ".pt")
    ref_cache = None
    if not a.no_hr and cache.exists():
        ref_cache = torch.load(cache)
        print("reference frames from cache:", cache)
    if ref_cache is not None or a.no_hr:
        from vfx.datagen3d.sim import run_pair
        gen = run_pair(PresetScenario(pre, a.warmup + a.frames), None, with_hr=False)
        stream = ((s_, lr_, None) for s_, lr_, _, _ in gen)
        hr_note = "steered 256^3: cached frames" if ref_cache is not None else "no reference (--no-hr)"
    else:
        stream, hr_note = sim_stream(pre, a.warmup + a.frames, True, b)
    new_ref = {"frames": [], "alpha": []}
    print(hr_note.encode("ascii", "replace").decode())
    solid = rasterize_obstacles_3d(pre.obstacles, 128) if pre.obstacles else None
    keys = ["raw"] + ([] if a.no_tri else ["tri"]) + ([] if det is None else ["det"]) + list(dps) + ["hr"]
    label = {"raw": "raw 64^3", "tri": "trilinear 256^3", "det": f"64^3 + advected noise ({a.detail_quality})",
             "hr": "steered 256^3 (reference)"}
    label.update({f"v{i}": v[0] for i, v in enumerate(var)})
    names = [label[k] for k in keys]
    rend = {k: VolumeRenderer((H, W), march_scale=a.march_scale) for k in keys}
    tm = Timer()
    seqs = {k: [] for k in keys}
    alphas = []
    sheet = []
    pick = set(np.linspace(0, a.frames - 1, min(8, a.frames)).round().astype(int).tolist())
    diag = {}
    forces = Forces3D(emitters=list(pre.emitters), wind=tuple(pre.wind))
    for step, lr, hr in stream:
        for kk, dp in dps.items():
            tm.run(f"{kk} step", lambda: dp.step(lr, forces, SIM_DT))
        if det is not None:
            tm.run("det step", lambda: det.step(lr, SIM_DT))
        k = step - a.warmup
        if k < 0:
            tm.resolve()
            continue
        cam = orbit_cam(a.start_deg + a.orbit_deg * k, a.dist)
        raw = Volume(lr.data[CH3["density"]], lr.data[CH3["temperature"]])
        panels = {"raw": tm.run("raw render", lambda: rend["raw"].draw(raw, cam, solid)).clone()}
        if "tri" in keys:
            vt = tm.run("tri upscale", lambda: b.classical_upscale(lr, 4, "trilinear"))
            panels["tri"] = tm.run("tri render", lambda: rend["tri"].draw(vt, cam, solid)).clone()
        if det is not None:
            panels["det"] = tm.run("det render", lambda: rend["det"].draw(raw, cam, solid, detail=det)).clone()
        for kk, dp in dps.items():
            panels[kk] = tm.run(f"{kk} render", lambda: rend[kk].draw(dp.volume(), cam, solid)).clone()
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
        pms = {"raw": (ms["raw render"], f"render {ms['raw render']:.2f}"), "hr": (0.0, "offline 256^3 sim")}
        if "tri" in keys:
            pms["tri"] = (ms["tri upscale"] + ms["tri render"],
                          f"upscale {ms['tri upscale']:.2f} + render {ms['tri render']:.2f}")
        if det is not None:
            pms["det"] = (ms["det step"] + ms["det render"], f"step {ms['det step']:.2f} + render {ms['det render']:.2f}")
        for kk in dps:
            st, rr = ms[f"{kk} step"], ms[f"{kk} render"]
            pms[kk] = (st + rr, f"particles {st:.2f} + render {rr:.2f}")
        for kk, im in panels.items():
            seqs[kk].append(im.permute(2, 0, 1).contiguous())
        if k % a.every == 0 or k in pick:
            shown = [x for x in keys if x in panels]
            n = len(shown)
            cols = a.cols or n
            rows = (n + cols - 1) // cols
            strip = pygame.Surface((cols * W, rows * H))
            for i, kk in enumerate(shown):
                x0, y0 = (i % cols) * W, (i // cols) * H
                strip.blit(to_surface(panels[kk]), (x0, y0))
                t1 = font.render(label[kk], True, (255, 230, 150))
                ms_, sub = pms[kk]
                t2 = small.render(f"{sub} = {ms_:.2f} ms" if kk != "hr" else sub, True, (210, 215, 225))
                bg = pygame.Surface((max(t1.get_width(), t2.get_width()) + 12, 42), pygame.SRCALPHA)
                bg.fill((0, 0, 0, 150))
                strip.blit(bg, (x0 + 6, y0 + 6))
                strip.blit(t1, (x0 + 12, y0 + 9))
                strip.blit(t2, (x0 + 12, y0 + 29))
                pygame.draw.rect(strip, (90, 90, 100), (x0, y0, W, H), 1)
            strip.blit(small.render(f"preset {a.preset} {pre.name}  step {step}  cam {a.start_deg + a.orbit_deg * k:.0f}"
                                    f" deg  (GPU shared: ms inflated)", True, (170, 175, 185)), (8, rows * H - 20))
            pygame.image.save(strip, str(out / f"frame_{k:04d}.png"))
            if k in pick:
                sheet.append(pygame.transform.smoothscale(strip, (cols * W // 2, rows * H // 2)))
        print(f"frame {k:3d}: " + "  ".join(f"{kk} {pms[kk][0]:.2f}" for kk in keys if kk in panels) + " ms",
              flush=True)
        if k == a.frames - 1:
            for kk, dp in dps.items():
                diag[label[kk]] = dp.diagnostics(lr)
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
            spectra[label[kk]] = p_
            if kk == "hr":
                continue
            d = dict(spectrum_error=M.spectrum_error(x, ref, mask), spectrum_log_ratio=M.spectrum_log_ratio(x, ref, mask),
                     hf_power_ratio=M.band_power(x, mask, 0.0625, 0.5) / max(M.band_power(ref, mask, 0.0625, 0.5), 1e-20))
            if len(x) >= 3:
                t = M.tlp(x, ref, mask)
                d.update({kk2: float(v) for kk2, v in t.items() if np.isscalar(v) or np.ndim(v) == 0})
            metrics[label[kk]] = d
        _spectrum_plot(out / "spectrum.png", spectra, names, _palette(len(names)))
    summary = {n_: float(np.median(v[2:] or v)) for n_, v in tm.acc.items()}
    from vfx.render3d.bench import gpu_note
    meta = dict(preset=a.preset, frames=a.frames, warmup=a.warmup, size=[W, H],
                variants=[dict(label=l_, count=c, res=r, params=p_) for l_, c, r, p_ in var], hr=hr_note,
                median_ms=summary, metrics=metrics, diagnostics=diag, gpu=gpu_note())
    (out / "metrics.json").write_text(json.dumps(meta, indent=2, default=float))
    print(json.dumps({"metrics": metrics, "diagnostics": diag}, indent=1, default=float))
    print("->", out)


if __name__ == "__main__":
    main()
