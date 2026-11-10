"""Render-cost feasibility table at 2560x1440: volume res x march_scale x step count.

  .venv/Scripts/python -m vfx.render3d.bench [--out DIR] [--frames 60] [--stub-sim]

Content: a campfire plume simulated at 64^3 (real vfx.sim3d if it runs, else the app3d stub),
trilinearly resampled to 64/128/256^3 (render cost depends on occupancy and step count, not on the
detail inside the plume). The camera distance is chosen so the effect covers ~1/3 of the screen.
Timing: CUDA events around N back-to-back draw() calls (synchronized), after warm-up / graph capture.
Writes table.md, results.json and comparison PNGs (+ contact sheets) to --out.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vfx.core.fields3d import CH3, SIM_DT, Camera, Forces3D, Volume, rasterize_obstacles_3d
from vfx.render3d import VolumeRenderer

OUT_W, OUT_H = 2560, 1440
RESES = (64, 128, 256)
SCALES = (1.0, 0.5, 0.25)
STEPS = (64, 128, 256)
TARGET = (0.5, 0.42, 0.5)


def make_content(frames: int = 150, stub_sim: bool = False):
    """(density, temperature) at 64^3 + obstacles of the campfire preset, + solver name."""
    from vfx.app3d.backends import load_backends
    from vfx.app3d.presets import PRESETS
    pre = PRESETS[1]
    b = load_backends(force_stub_sim=stub_sim, warmup=False)
    s = b.Solver(64, b.runtime_config(), device="cuda")
    st = s.reset(pre.obstacles)
    for _ in range(frames):
        st = s.step(st, Forces3D(emitters=list(pre.emitters), wind=pre.wind), SIM_DT)
    torch.cuda.synchronize()
    name = f"{b.Solver.__module__}.{b.Solver.__name__}"
    return st.data[CH3["density"]].clone(), st.data[CH3["temperature"]].clone(), pre.obstacles, name


def resample(x: torch.Tensor, n: int) -> torch.Tensor:
    if x.shape[0] == n:
        return x.contiguous()
    return F.interpolate(x[None, None], size=(n, n, n), mode="trilinear", align_corners=False)[0, 0].contiguous()


def camera_at(dist: float) -> Camera:
    pitch = 0.20
    pos = (TARGET[0], TARGET[1] + dist * np.sin(pitch), TARGET[2] - dist * np.cos(pitch))
    return Camera(position=pos, target=TARGET, fov_y_deg=40.0)


def coverage(r: VolumeRenderer, vol: Volume, cam: Camera, solid) -> float:
    """Fraction of pixels where the effect (smoke/fire/obstacles) is visible: transmittance < 0.98,
    measured with the floor disabled (floor would count as coverage)."""
    r.set_params(floor=0.0)
    r.draw(vol, cam, solid)
    torch.cuda.synchronize()
    cov = float((r.hdr[..., 3] < 0.98).float().mean())
    r.set_params(floor=1.0)
    return cov


def time_draws(r: VolumeRenderer, vol, cam, solid, n: int, trials: int = 9, reuse: bool = False) -> tuple[float, float]:
    """(min, median) over `trials` batches of n back-to-back draws. On a shared GPU the min batch is
    the best estimate of the uncontended cost; the median shows the contention."""
    for _ in range(5):
        r.draw(vol, cam, solid)
    torch.cuda.synchronize()
    res = []
    for _ in range(trials):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            r.draw(vol, cam, solid, reuse_volume=reuse)
        e.record()
        e.synchronize()
        res.append(s.elapsed_time(e) / n)
    return float(np.min(res)), float(np.median(res))


def profile_stages(vol, cam, solid, scale, steps, n: int = 40) -> dict:
    """Per-stage ms (no graph, events between stages); the lower quartile per stage (shared GPU)."""
    r = VolumeRenderer((OUT_H, OUT_W), march_scale=scale, profile=True)
    r.set_params(steps=float(steps))
    acc: dict = {}
    for i in range(n + 3):
        r.draw(vol, cam, solid)
        t = r.last_timings(sync=True)
        if i >= 3:
            for k, v in t.items():
                acc.setdefault(k, []).append(v)
    return {k: float(np.percentile(v, 25)) for k, v in acc.items()}


def save_png(img: torch.Tensor, path: Path) -> None:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    h, w = img.shape[:2]
    surf = pygame.image.frombuffer(img.cpu().numpy().tobytes(), (w, h), "RGB")
    pygame.image.save(surf, str(path))


def contact_sheet(tiles: dict, rows, cols, row_lab, col_lab, path: Path, tw=640, th=360) -> None:
    import pygame
    pygame.font.init()
    font = pygame.font.SysFont("consolas", 20, bold=True)
    sheet = pygame.Surface((len(cols) * tw, len(rows) * th))
    for i, rk in enumerate(rows):
        for j, ck in enumerate(cols):
            img = tiles.get((rk, ck))
            if img is None:
                continue
            src = img[None].permute(0, 3, 1, 2).float()
            small = F.interpolate(src, size=(th, tw), mode="area")[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)
            surf = pygame.image.frombuffer(small.cpu().numpy().tobytes(), (tw, th), "RGB")
            sheet.blit(surf, (j * tw, i * th))
            t = font.render(f"{row_lab(rk)}  {col_lab(ck)}", True, (255, 230, 150))
            sheet.blit(t, (j * tw + 10, i * th + 8))
    pygame.image.save(sheet, str(path))


def gpu_note() -> str:
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10).stdout.strip()
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=process_name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.split()
        names = sorted({os.path.basename(a) for a in apps if a.lower().endswith(".exe")})
        return f"util,mem: {q}; GPU processes: {', '.join(names) or 'none'}"
    except Exception as e:
        return f"nvidia-smi failed: {e}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(os.environ.get("TEMP", ".")) / "render3d_bench"))
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--res", type=int, nargs="*", default=list(RESES))
    ap.add_argument("--stub-sim", action="store_true", help="content from the app3d toy solver")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    note_before = gpu_note()
    d64, t64, obstacles, solver_name = make_content(stub_sim=args.stub_sim)
    solid = rasterize_obstacles_3d(obstacles, 128)
    vols = {n: Volume(resample(d64, n), resample(t64, n)) for n in RESES}

    # camera distance so the effect covers ~1/3 of the frame (measured at 256^3, scale 0.25)
    probe = VolumeRenderer((OUT_H, OUT_W), march_scale=0.25)
    lo, hi = 0.35, 4.0
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        c = coverage(probe, vols[256], camera_at(mid), solid)
        lo, hi = (mid, hi) if c > 0.33 else (lo, mid)
    dist = 0.5 * (lo + hi)
    cam = camera_at(dist)
    cov = coverage(probe, vols[256], cam, solid)
    del probe
    print(f"camera dist {dist:.2f} m -> coverage {cov:.3f}; solver {solver_name}")

    rows, tiles = [], {}
    for n in args.res:
        for sc in SCALES:
            for steps in STEPS:
                torch.cuda.empty_cache()
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                r = VolumeRenderer((OUT_H, OUT_W), march_scale=sc)
                r.set_params(steps=float(steps))
                ms, ms_med = time_draws(r, vols[n], cam, solid, args.frames)
                r.draw(vols[n], cam, solid)
                ms_view, _ = time_draws(r, vols[n], cam, solid, args.frames, reuse=True)
                peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
                tex = (n ** 3 * 4 + r.p.light_res ** 3 * 4 + 32 ** 3 * 8 + 128 ** 3) / 2 ** 20
                prof = profile_stages(vols[n], cam, solid, sc, steps)
                row = {"res": n, "march_scale": sc, "steps_per_m": steps, "march_px": [r.mw, r.mh],
                       "ms": ms, "ms_median": ms_med, "ms_view_only": ms_view, "fits_1p5ms": ms <= 1.5, "vram_mb": peak + tex, "stages_ms": prof}
                rows.append(row)
                print(f"{n:4d}^3 scale {sc:4.2f} steps {steps:4d}: {ms:6.3f} ms (median {ms_med:6.3f}, view-only {ms_view:6.3f})  vram {peak + tex:6.0f} MB  "
                      + " ".join(f"{k} {v:.3f}" for k, v in prof.items()))
                if steps == 128 or (n == 256 and sc == 1.0):
                    img = r.draw(vols[n], cam, solid).clone()
                    tiles[(n, sc, steps)] = img
                    save_png(img, out / f"render_{n}_s{sc:g}_st{steps}.png")
                del r
    note_after = gpu_note()
    contact_sheet({(k[0], k[1]): v for k, v in tiles.items() if k[2] == 128}, RESES, SCALES,
                  lambda n: f"{n}^3", lambda s: f"scale {s:g}  (128 steps/m)", out / "sheet_res_x_scale.png")
    contact_sheet({(0, k[2]): v for k, v in tiles.items() if k[0] == 256 and k[1] == 1.0}, [0], STEPS,
                  lambda _: "256^3 scale 1", lambda s: f"{s} steps/m", out / "sheet_steps.png")

    lines = [f"Render 2560x1440, plume coverage {cov:.2f} of frame, camera dist {dist:.2f} m, content: {solver_name}",
             f"GPU before: {note_before}", f"GPU after: {note_after}", "",
             "full = new volume every draw (pack + upload + light/glow + march + composite); view = reuse_volume=True",
             "(camera-only redraw, the 60 Hz frames between 30 Hz sim ticks); 60 Hz avg = (full + view) / 2.",
             "ms = min over 9 batches of N draws (median of full in brackets); stages = lower quartile, profiled without CUDA graphs.", "",
             "| volume | march_scale | steps/m | march px | full ms (median) | view ms | 60 Hz avg | pack | upload+light+glow | march | bloom+composite | VRAM MB | full <=1.5 | avg <=1.5 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r_ in rows:
        s = r_["stages_ms"]
        avg = 0.5 * (r_["ms"] + r_["ms_view_only"])
        lines.append(f"| {r_['res']}^3 | {r_['march_scale']:g} | {r_['steps_per_m']} | {r_['march_px'][0]}x{r_['march_px'][1]} "
                     f"| {r_['ms']:.3f} ({r_['ms_median']:.2f}) | {r_['ms_view_only']:.3f} | {avg:.3f} | {s.get('pack', 0):.3f} "
                     f"| {s.get('light', 0):.3f} | {s.get('march', 0):.3f} | {s.get('composite', 0):.3f} | {r_['vram_mb']:.0f} "
                     f"| {'yes' if r_['fits_1p5ms'] else 'no'} | {'yes' if avg <= 1.5 else 'no'} |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    (out / "results.json").write_text(json.dumps({"coverage": cov, "dist": dist, "solver": solver_name,
                                                   "gpu_before": note_before, "gpu_after": note_after,
                                                   "rows": rows}, indent=2))
    print("\n".join(lines))
    print("->", out)


if __name__ == "__main__":
    main()
