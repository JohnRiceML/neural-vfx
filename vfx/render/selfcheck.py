"""Quick owner self-check: render procedural fields, save a PNG, time draw() at 1024^2.

    .venv/Scripts/python -m vfx.render.selfcheck [--out path.png] [--res 512] [--size 1024]
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from vfx.core.fields import AMBIENT_TEMP, HRFields, Obstacle, rasterize_obstacles
from vfx.render import Renderer


def procedural_fields(res: int, device="cuda") -> tuple[HRFields, torch.Tensor]:
    c = (torch.arange(res, device=device, dtype=torch.float32) + 0.5) / res
    y, x = torch.meshgrid(c, c, indexing="ij")             # y up
    g = torch.Generator(device=device).manual_seed(0)
    noise = torch.zeros(res, res, device=device)
    for octave in range(5):                                  # cheap fBm via upsampled noise
        n = 4 * 2 ** octave
        z = torch.rand(1, 1, n, n, device=device, generator=g)
        noise += torch.nn.functional.interpolate(z, size=(res, res), mode="bicubic", align_corners=False)[0, 0] / 2 ** octave
    noise = noise / 2.0
    # smoke plume rising from (0.3, 0.1), widening with height, wiggling
    cx = 0.3 + 0.06 * torch.sin(y * 9.0) + 0.1 * (noise - 0.5)
    width = 0.04 + 0.18 * y
    plume = torch.exp(-((x - cx) / width) ** 2) * ((y > 0.08).float()) * (1.2 - y) * (0.6 + noise)
    # fire at (0.72, 0.12): hot core tapering upward
    fx = 0.72 + 0.03 * torch.sin(y * 25.0 + noise * 6)
    flame = torch.exp(-((x - fx) / (0.05 * (1.0 - 1.5 * (y - 0.1)).clamp(min=0.05))) ** 2) * torch.exp(-((y - 0.12) / 0.22).clamp(min=0) ** 1.5)
    flame = flame * (y > 0.08).float() * (0.6 + 0.8 * noise)
    temp = AMBIENT_TEMP + 1900.0 * flame.clamp(0, 1.2)
    soot = torch.exp(-((x - fx) / (0.04 + 0.2 * (y - 0.3).clamp(min=0))) ** 2) * (y > 0.35).float() * 0.9 * noise
    dens = (0.8 * plume * noise + 0.6 * soot + 0.2 * flame).clamp(min=0)
    solid = rasterize_obstacles([Obstacle("circle", (0.3, 0.55), (0.07, 0.07)),
                                 Obstacle("box", (0.75, 0.04), (0.08, 0.03))], res, device)
    dens = dens * (1 - solid)
    return HRFields(dens, temp), solid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--nearest", action="store_true")
    a = ap.parse_args()
    fields, solid = procedural_fields(a.res)
    r = Renderer((a.size, a.size))
    img = r.draw(fields, solid, nearest=a.nearest)
    for _ in range(20):
        r.draw(fields, solid)
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(200):
        st.record(); r.draw(fields, solid); en.record(); torch.cuda.synchronize()
        times.append(st.elapsed_time(en))
    t = np.array(times)
    print(f"draw {a.res}^2 -> {a.size}^2: mean {t.mean():.3f} ms  p50 {np.median(t):.3f}  p99 {np.percentile(t, 99):.3f}")
    if a.out:
        import pygame
        img = r.draw(fields, solid, nearest=a.nearest).cpu().numpy()
        pygame.image.save(pygame.image.frombuffer(img.tobytes(), (a.size, a.size), "RGB"), a.out)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
