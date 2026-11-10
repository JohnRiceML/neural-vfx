"""One short GPU session for the sandbox integration (particles, solver opt-ins, --best):

    python -m vfx.app3d.integration_check --out DIR [--frames 120]

For each case runs the headless App (no window), saves a screenshot with the HUD, and checks: the mode is
available, no stubs, every view produced an image, stage timers recorded. Cases:
  * particles | raw, particles_detail | detail                 (preset 1, then 2)
  * opt-in toggles mid-run: fused on -> confref on -> both off  (detail | raw)
  * --best and --best-particles                               (preset 3)
  * default raw | trilinear with no opt-ins: frame hash equal to a run with the toggles never touched
Prints a summary table (GPU ms per stage, p50) and PASS/FAIL. GPU shared state is printed first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


def _run(argv, frames, out, name, script=None):
    import pygame
    from vfx.app3d import main as M
    from vfx.app3d.hud import Hud
    from vfx.core.fields3d import SIM_DT
    args = M.parse_args(argv + ["--headless", "--frames", str(frames)])
    M.apply_best(args)
    M.WIN_W, M.WIN_H = 1600, 900
    app = M.App(args)
    fdt, acc, host = 1.0 / 60.0, SIM_DT, None
    for i in range(frames):
        if script:
            script(app, i)
        acc += fdt
        while acc >= SIM_DT - 1e-9:
            app.sim_tick()
            acc -= SIM_DT
        host = app.frame()
        app.frame_ms.append(16.7)
    surf = pygame.image.frombuffer(host.tobytes(), (M.WIN_W, M.WIN_H), "RGB").copy()
    app.draw_hud(surf, Hud(), force=True)
    pygame.image.save(surf, str(out / f"{name}.png"))
    st = app.stats(1.0)
    st["frame_hash"] = hashlib.sha256(host.tobytes()).hexdigest()[:16]
    st["modes"] = [m for m, _ in app.layout()]
    return st


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=120)
    a = ap.parse_args(argv)
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    pygame.font.init()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from vfx.render3d.bench import gpu_note
    print("GPU:", gpu_note())

    def toggles(app, i):
        if i == a.frames // 4:
            app.toggle_sim_option("fused")
        if i == a.frames // 2:
            app.toggle_sim_option("confref")
        if i == 3 * a.frames // 4:
            app.toggle_sim_option("fused")
            app.toggle_sim_option("confref")

    cases = {
        "p1_particles_vs_raw": (["--preset", "1", "--mode", "particles", "--side-by-side", "--mode2", "raw"], None),
        "p2_particles_detail_vs_detail": (["--preset", "2", "--mode", "particles_detail", "--side-by-side", "--mode2", "detail"], None),
        "p1_optin_toggles": (["--preset", "1", "--mode", "detail", "--side-by-side", "--mode2", "raw"], toggles),
        "p3_best": (["--preset", "3", "--best"], None),
        "p3_best_light": (["--preset", "3", "--best-light"], None),
        "p1_default_a": (["--preset", "1", "--mode", "raw", "--side-by-side", "--mode2", "trilinear"], None),
        "p1_default_b": (["--preset", "1", "--mode", "raw", "--side-by-side", "--mode2", "trilinear"],
                         lambda app, i: (app.toggle_sim_option("fused"), app.toggle_sim_option("fused")) if i == 0 else None),
    }
    res, ok = {}, True
    for name, (argv_, script) in cases.items():
        try:
            st = _run(argv_, a.frames, out, name, script)
        except Exception as e:
            import traceback
            traceback.print_exc()
            res[name] = {"error": f"{type(e).__name__}: {e}"}
            ok = False
            continue
        bad = [m for m in st["modes"] if m in st["unavailable"]]
        if bad or st["stubs"]:
            ok = False
        res[name] = st
        g = {k: round(v["p50"], 3) for k, v in st["gpu_ms"].items() if v}
        print(f"{name:32s} modes {st['modes']}  opts {st.get('sim_opts')}  unavailable {bad}  gpu p50 {g}")
    if "p1_default_a" in res and "p1_default_b" in res and "frame_hash" in res["p1_default_a"]:
        same = res["p1_default_a"]["frame_hash"] == res["p1_default_b"]["frame_hash"]
        print("default path unaffected by toggling an opt-in on and off before the first tick:", same)
        ok &= same
    (out / "integration.json").write_text(json.dumps(res, indent=1, default=str))
    print("PASS" if ok else "FAIL", "->", out)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
