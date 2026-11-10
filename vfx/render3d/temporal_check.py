"""Temporal amortization: cost table + ghosting/lag pictures.

    python -m vfx.render3d.temporal_check --out DIR [--size 2560x1440] [--skip-time] [--skip-pics]

Cost (2560x1440, CUDA events, min of interleaved batches): VolumeRenderer full/view ms for march scale
0.5 and 0.25 at 192 and 96 steps/m, temporal off/on, with and without advected detail.
Pictures (960x540, scale 0.5): three consecutive frames + |frame t - frame t-1| x4 for
  * fast plume: preset 2 smoke with a strong sideways push (impulses) during recording,
  * fire: preset 1 campfire, static camera,
  * fast orbit: preset 1 with the camera orbiting 6 deg per frame,
each temporal off vs on (and a reference: temporal off at 4x steps), plus a noise/banding crop:
temporal on at 96 steps/m vs off at 96 and 384 steps/m.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def _time(fn, n=10, rep=5):
    best = []
    for _ in range(rep):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(n):
            fn()
        b.record()
        b.synchronize()
        best.append(a.elapsed_time(b) / n)
    return float(np.min(best))


def _sim(preset, steps, impulse=None):
    from vfx.app3d.presets import PRESETS
    from vfx.core.fields3d import SIM_DT, Forces3D, Impulse3D
    from vfx.detail3d import AdvectedDetail
    from vfx.sim3d import Solver3D, runtime_config
    pre = PRESETS[preset]
    s = Solver3D(64, runtime_config(**pre.sim))
    st = s.reset(pre.obstacles)
    det = AdvectedDetail(64)

    def step(push=False):
        nonlocal st
        imp = [Impulse3D(pos=(0.45, 0.45, 0.5), radius=0.15, velocity=(2.0, 0.3, 0.0))] if push else []
        st = s.step(st, Forces3D(emitters=list(pre.emitters), impulses=imp, wind=pre.wind), SIM_DT)
        det.step(st, SIM_DT)
        return st
    for _ in range(steps):
        step()
    return pre, step, det


def timing(out):
    from vfx.app3d.compare import orbit_cam
    from vfx.core.fields3d import CH3, Volume, rasterize_obstacles_3d
    from vfx.render3d import VolumeRenderer
    from vfx.render3d.bench import gpu_note
    pre, step, det = _sim(1, 150)
    st = step()
    solid = rasterize_obstacles_3d(pre.obstacles, 128) if pre.obstacles else None
    vol = Volume(st.data[CH3["density"]], st.data[CH3["temperature"]])
    cam = orbit_cam(-30.0)
    rows = []
    note0 = gpu_note()
    for ms in (0.5, 0.25):
        for steps in (192.0, 96.0):
            for d in (None, det):
                rs = {t: VolumeRenderer((1440, 2560), march_scale=ms, temporal=t) for t in (False, True)}
                for r in rs.values():
                    r.set_params(steps=steps)
                    for _ in range(3):
                        r.draw(vol, cam, solid, detail=d)
                        r.draw(vol, cam, solid, reuse_volume=True, detail=d)
                res = {t: {"full": [], "view": []} for t in rs}
                for _ in range(3):
                    for t, r in rs.items():
                        res[t]["full"].append(_time(lambda r=r: r.draw(vol, cam, solid, detail=d)))
                        r.draw(vol, cam, solid, detail=d)
                        res[t]["view"].append(_time(lambda r=r: r.draw(vol, cam, solid, reuse_volume=True, detail=d)))
                for t in rs:
                    f_, v_ = min(res[t]["full"]), min(res[t]["view"])
                    rows.append(dict(march_scale=ms, steps=steps, detail=d is not None, temporal=t, full=f_, view=v_,
                                     avg60=0.5 * (f_ + v_)))
    print("| march | steps/m | detail | temporal | full ms | view ms | 60 Hz avg |")
    for r_ in rows:
        print(f"| {r_['march_scale']} | {r_['steps']:.0f} | {'yes' if r_['detail'] else 'no'} | "
              f"{'on' if r_['temporal'] else 'off'} | {r_['full']:.2f} | {r_['view']:.2f} | {r_['avg60']:.2f} |")
    (out / "temporal_timing.json").write_text(json.dumps({"gpu_before": note0, "gpu_after": gpu_note(), "rows": rows}, indent=2))


def pictures(out):
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    from vfx.app3d.compare import orbit_cam
    from vfx.core.fields3d import CH3, Volume, rasterize_obstacles_3d
    from vfx.render3d import VolumeRenderer
    W, H = 960, 540

    def surf(t):
        a = t.contiguous().cpu().numpy()
        return pygame.image.frombuffer(a.tobytes(), (a.shape[1], a.shape[0]), "RGB")

    cases = [("fast_plume", 2, True, 0.0), ("fire", 1, False, 0.0), ("fast_orbit", 1, False, 6.0)]
    for name, preset, push, orbit in cases:
        pre, step, det = _sim(preset, 120)
        solid = rasterize_obstacles_3d(pre.obstacles, 128) if pre.obstacles else None
        cfgs = {"off": (False, 192.0), "on": (True, 192.0), "on_96": (True, 96.0), "ref_768": (False, 768.0)}
        rs = {}
        for k, (t, steps) in cfgs.items():
            rs[k] = VolumeRenderer((H, W), march_scale=0.5, temporal=t)
            rs[k].set_params(steps=steps)
        frames = {k: [] for k in rs}
        for i in range(24):                       # 12 sim ticks at 2 draws per tick (60 Hz over 30 Hz)
            if i % 2 == 0:
                st = step(push=push and i >= 6)
                vol = Volume(st.data[CH3["density"]], st.data[CH3["temperature"]])
            cam = orbit_cam(-30.0 + orbit * i, 1.9)
            for k, r in rs.items():
                img = r.draw(vol, cam, solid, reuse_volume=(i % 2 == 1), detail=det)
                if i >= 21:
                    frames[k].append(img.clone())
        sheet = pygame.Surface((4 * W // 2, len(rs) * H // 2))
        crops = pygame.Surface((4 * 300, len(rs) * 300))
        for j, k in enumerate(rs):
            fr = frames[k]
            diff = ((fr[2].float() - fr[1].float()).abs() * 4).clamp(0, 255).byte()
            for i, im in enumerate(fr + [diff]):
                sheet.blit(pygame.transform.smoothscale(surf(im), (W // 2, H // 2)), (i * W // 2, j * H // 2))
                crops.blit(surf(im[120:420, 330:630]), (i * 300, j * 300))
        pygame.image.save(sheet, str(out / f"temporal_{name}.png"))
        pygame.image.save(crops, str(out / f"temporal_{name}_crops.png"))
        print(name, "rows:", list(rs), "cols: t-2, t-1, t, |t - (t-1)| x4")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-time", action="store_true")
    ap.add_argument("--skip-pics", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.skip_pics:
        pictures(out)
    if not a.skip_time:
        timing(out)


if __name__ == "__main__":
    main()
