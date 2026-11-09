"""Self-check timing of Upscaler.infer at 512^2 (owner check; the gate bench lives in bench/).

    .venv/Scripts/python -m vfx.runtime.bench_selfcheck [--ckpt PATH]

Scenes: "full" (smoke everywhere -> dense path), "plume" (one 0.04 m-sigma plume) and "small" (a
0.015 m-sigma puff, i.e. a mostly-empty frame). Two timings per configuration:
  loop : CUDA events around 200 back-to-back infer() calls (free-running, no host sync)
  app  : the same with one torch.cuda.synchronize() per frame (as the app's display blit does);
         wall time per frame after the sync, so it includes un-overlapped CPU launch cost
plus the per-stage GPU breakdown from Upscaler.timings. Without a checkpoint, a randomly initialised model with
the default architecture is timed (speed does not depend on the weights).
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

from vfx.core import paths
from vfx.core.fields import AMBIENT_TEMP, CH, LR_RES, NUM_CHANNELS, FieldMeta, FieldState
from vfx.model import DetailNet, DetailNetConfig, save_model
from vfx.runtime import Upscaler, classical_upscale


def make_state(kind: str) -> FieldState:
    g = torch.Generator(device="cuda").manual_seed(0)
    d = torch.zeros(NUM_CHANNELS, LR_RES, LR_RES, device="cuda")
    c = (torch.arange(LR_RES, device="cuda") + 0.5) / LR_RES
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    if kind == "full":
        dens = torch.rand(LR_RES, LR_RES, device="cuda", generator=g) + 0.1
    else:   # one plume / puff near the bottom-left
        sig = 0.04 if kind == "plume" else 0.015
        dens = torch.exp(-((xx - 0.2) ** 2 + (yy - 0.15) ** 2) / (2 * sig ** 2))
        dens = dens * (dens > 1e-3)
    d[CH["density"]] = dens
    d[CH["temperature"]] = AMBIENT_TEMP + 300.0 * dens
    d[CH["vel_y"]] = 0.3 * dens
    return FieldState(d, torch.zeros(LR_RES, LR_RES, device="cuda"), FieldMeta.for_res(LR_RES))


def time_loop(fn, n=200, warm=30) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def time_app(fn, n=200, warm=30) -> float:
    import time
    for _ in range(warm):
        fn()
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1e3 / n


def breakdown(ups: Upscaler, st: FieldState, n=50) -> dict:
    acc: dict = {}
    for _ in range(n):
        ups.infer(st)
        for k, v in ups.timings.items():
            acc[k] = acc.get(k, 0.0) + v / n
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()
    ckpt = a.ckpt or paths.default_checkpoint()
    if not Path(ckpt).is_file():
        ckpt = Path(tempfile.gettempdir()) / "vfx_bench_random.pt"
        save_model(DetailNet(DetailNetConfig(n_blocks=8)), ckpt)
        print(f"(no checkpoint found; timing a random-init default model: {ckpt})")
    print(f"checkpoint: {ckpt}")
    print(f"budget: 2.5 ms per 512^2 frame incl. warp\n")
    for kind in ("full", "plume", "small"):
        st = make_state(kind)
        print(f"bicubic baseline [{kind}]: {time_loop(lambda: classical_upscale(st)):.3f} ms")
        for graph in (True, False):
            for sparse in (True, False):
                ups = Upscaler(str(ckpt), sparse=sparse, cuda_graph=graph)
                if graph:
                    ups.warmup()
                ms = time_loop(lambda: ups.infer(st))
                app = time_app(lambda: ups.infer(st))
                bd = breakdown(ups, st)
                tag = f"[{kind:5s}] graph={int(graph)} sparse={int(sparse)}"
                print(f"{tag}: loop {ms:.3f} ms  app {app:.3f} ms  processed={bd['processed_fraction']:.3f}  "
                      f"(gpu events: pre {bd['warp']:.3f} | net+compose {bd['net']:.3f} | post {bd['post']:.3f})"
                      f"  {'OK' if max(ms, app) <= 2.5 else 'OVER BUDGET'}")
        print()


if __name__ == "__main__":
    main()
