"""Live-app-style flicker: no dataset, the situation the user actually sees (owner: test/eval engineer).

    .venv/Scripts/python -m vfx.eval.live_flicker [--steps 300] [--warmup 30] [--ckpt PATH]

For each fixed scene (fire; smoke + circle obstacle) a 128^2 vfx.sim.Solver runs `steps` steps
and every state is fed to vfx.runtime.Upscaler (neural, temporal state, sparse bricks as in the app)
and vfx.runtime.classical_upscale (bicubic, what the app shows without a checkpoint). Flicker of the
HR density uses the velocity of the 128^2 run that drove it (vfx.eval.metrics.flicker definition,
frame t-1 -> t warped by the velocity stored after step t-1), averaged over frames after `warmup`.

Reference (indicative only): a 512^2 Solver run of the same scene and forces, warped with its own
velocity 4x average-pooled to 128^2 (same velocity resolution as the upscalers get, and the same
convention as the dataset reference). The 512^2 flow is a different realization (it diverges from
the 128^2 run), so ratio = flicker(upscaler) / flicker(512^2 run) compares statistically similar
sequences, not pixel-matched ones. `ref_fullvel` (512^2 velocity) is reported as a diagnostic.

The scenes are fixed here (copied from vfx/app/presets.py presets 1 and 2 on 2026-11-09) so the
measurement does not move when the app presets change.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from vfx.core.fields import CH, DOMAIN_SIZE, HR_RES, LR_RES, SIM_DT, UPSCALE, Emitter, Forces, Obstacle
from vfx.eval.metrics import flicker_step

SCENES = {
    "fire": dict(
        emitters=[Emitter(pos=(0.5, 0.08), radius=0.06, density_rate=0.0, temperature=1200.0, fuel_rate=6.0,
                          velocity=(0.0, 0.35))],
        obstacles=[Obstacle("box", (0.5, 0.02), (0.12, 0.02))]),
    "smoke_obstacle": dict(
        emitters=[Emitter(pos=(0.5, 0.1), radius=0.06, density_rate=0.9, temperature=420.0, velocity=(0.0, 0.4))],
        obstacles=[Obstacle("circle", (0.5, 0.48), (0.1, 0.1))]),
}


class _Acc:
    """Streaming flicker: mean over frames of mean |x_t - warp(x_{t-1}, v_{t-1})|."""

    def __init__(self, dx: float):
        self.dx, self.prev, self.total, self.n = dx, None, 0.0, 0

    def add(self, x: torch.Tensor, prev_vel: torch.Tensor | None, count: bool):
        if count and self.prev is not None and prev_vel is not None:
            self.total += flicker_step(self.prev, x, prev_vel, self.dx, SIM_DT)
            self.n += 1
        self.prev = x

    @property
    def value(self) -> float:
        return self.total / max(self.n, 1) if self.n else float("nan")


def _run_lr(scene: dict, steps: int, warmup: int, upscalers: dict) -> dict:
    from vfx.sim import Solver
    solver = Solver(LR_RES)
    st = solver.reset(list(scene["obstacles"]))
    forces = Forces(emitters=list(scene["emitters"]))
    dx_hr = DOMAIN_SIZE / HR_RES
    accs = {k: _Acc(dx_hr) for k in upscalers}
    for k, u in upscalers.items():
        if hasattr(u, "reset"):
            u.reset()
    prev_vel = None
    for i in range(steps):
        st = solver.step(st, forces, SIM_DT)
        with torch.no_grad():
            for k, u in upscalers.items():
                out = getattr(u, "infer", u)(st)
                accs[k].add(out.density.float().clone(), prev_vel, i >= warmup)
        prev_vel = st.velocity.clone()          # step() updates in place
    return {k: a.value for k, a in accs.items()}


def _run_hr(scene: dict, steps: int, warmup: int) -> dict:
    from vfx.sim import Solver
    solver = Solver(HR_RES)
    st = solver.reset(list(scene["obstacles"]))
    forces = Forces(emitters=list(scene["emitters"]))
    dx = DOMAIN_SIZE / HR_RES
    pooled, full = _Acc(dx), _Acc(dx)
    pv = fv = None
    for i in range(steps):
        st = solver.step(st, forces, SIM_DT)
        d = st.ch("density").clone()
        pooled.add(d, pv, i >= warmup)
        full.add(d, fv, i >= warmup)
        fv = st.velocity.clone()
        pv = F.avg_pool2d(fv[None], UPSCALE)[0]
    return {"ref": pooled.value, "ref_fullvel": full.value}


def live_flicker(steps: int = 300, warmup: int = 30, checkpoint: str | None = None,
                 scenes: tuple[str, ...] = tuple(SCENES), device: str = "cuda") -> dict:
    """Returns {scene: {neural, bicubic, ref, ref_fullvel, ratio_neural, ratio_bicubic, neural_over_bicubic}}.
    `neural` is absent if no checkpoint exists."""
    from vfx.core.paths import default_checkpoint
    from vfx.runtime import Upscaler, classical_upscale
    ckpt = checkpoint or str(default_checkpoint())
    ups = {"bicubic": classical_upscale}
    try:
        ups["neural"] = Upscaler(ckpt, device=device)
    except FileNotFoundError:
        pass
    out = {}
    for name in scenes:
        r = _run_lr(SCENES[name], steps, warmup, ups)
        r.update(_run_hr(SCENES[name], steps, warmup))
        r["ratio_bicubic"] = r["bicubic"] / max(r["ref"], 1e-12)
        if "neural" in r:
            r["ratio_neural"] = r["neural"] / max(r["ref"], 1e-12)
            r["neural_over_bicubic"] = r["neural"] / max(r["bicubic"], 1e-12)
        out[name] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()
    from vfx.core.paths import RUNS, default_checkpoint
    res = live_flicker(a.steps, a.warmup, a.ckpt)
    rep = {"checkpoint": a.ckpt or str(default_checkpoint()), "steps": a.steps, "warmup": a.warmup,
           "created": time.strftime("%Y-%m-%d %H:%M:%S"), "results": res}
    out = RUNS / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "live_flicker_latest.json").write_text(json.dumps(rep, indent=1))
    for k, r in res.items():
        print(f"{k:15s} " + "  ".join(f"{m} {v:.5g}" for m, v in r.items()))


if __name__ == "__main__":
    main()
