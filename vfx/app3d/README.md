# vfx/app3d — 3D sandbox

```
.venv/Scripts/python -m vfx.app3d.main                                   # window 1600x900
.venv/Scripts/python -m vfx.app3d.main --headless --frames 300 --preset 1 --mode trilinear \
    --screenshot out.png --stats out.json [--side-by-side --mode2 raw] [--orbit] [--demo-input]
.venv/Scripts/python -m vfx.app3d.main --quit-after 10                    # windowed smoke test
```
Flags: `--preset 1-4`, `--mode / --mode2 raw|trilinear|neural_vol|neural_screen|ref128`,
`--side-by-side`, `--warmup-neural`, `--orbit` (auto-rotate, for showcase captures), `--march-scale` (default 0.5),
`--steps` (ray-march samples per meter), `--stub-sim` (force the toy solver), `--demo-input`
(headless scripted fire tap / push drag / smoke puff), `--quit-after SECONDS`, `--fps` cap (60).

## Modes (N cycles the main view, B cycles the right view in side-by-side)
| mode | what is rendered |
|---|---|
| `raw` — raw 64³ | the 64³ runtime solver state, directly |
| `trilinear` — classical trilinear ×4 (256³) | `runtime3d.classical_upscale_3d` of the 64³ state |
| `neural_vol` — neural volume (route A) | `runtime3d.VolumeUpscaler.infer` → 256³ (**untrained — timing only**) |
| `neural_screen` — neural screen-space (route B) | raw 64³ render → `runtime3d.ScreenDetail.infer` (**untrained — timing only**) |
| `ref128` — reference: direct 128³ sim | a second, full-quality 128³ solver driven by the same forces (brute force cost) |

The neural modes carry a red "UNTRAINED - timing only, not a result" label; their pictures mean
nothing in this phase, only their GPU ms. If a module fails to import/construct the view shows
UNAVAILABLE with the reason. `ref128` starts from a trilinear copy of the current 64³ state when it
becomes visible, then evolves independently.

## Controls
| key | action |
|---|---|
| RMB drag / wheel | orbit / zoom |
| LMB drag | push the smoke: impulse along the view plane at the cursor (plane through the orbit target) |
| F | tap: toggle a fire emitter at the cursor; hold (>0.25 s): torch follows the cursor |
| G | smoke puff at the cursor |
| N / B / V | cycle main mode / cycle right mode / side-by-side |
| T | cycle march scale 1.0 → 0.5 → 0.25 |
| 1–4 | presets: campfire, smoke plume past a sphere, twin torches + wind, box obstacle |
| O / R / Space / H / C / Esc | obstacle on/off (keeps the fluid) / reset / pause / HUD / auto-orbit / quit |

The cursor "depth" is the plane through the orbit target facing the camera (no GPU read-back), so
F/G/push act in the middle of the volume under the cursor.

## Frame loop and timing
Sim (+ volume upscale) at 30 Hz (`SIM_DT`), render every display frame (60 Hz cap) from the latest
volume; the camera moves every frame. Each view has its own renderer; on frames without a sim tick
it redraws with `reuse_volume=True` (skips pack/upload/light volume), shown in the HUD as
"render X new vol / Y reuse". Each frame: render each visible view into one 1600x900 GPU
canvas → ONE `.cpu()` blit → pygame. CUDA events per stage (`sim`, `sim_ref`, `upscale`,
`neural_vol`, `detail`, `render`) are read only after that blit.
HUD: fps / 1% low, GPU ms per stage, **tick-frame total** (sim + upscale + detail + render, i.e. the
worst frame, the one with a sim tick) and the amortized total against the 6 ms budget with a bar,
VRAM, mode, key help.

**The window renders at 1600x900 (800x450 marched at scale 0.5), not 1440p.** A 1440p canvas would
make the single allowed CPU blit (~11 MB/frame) dominate the frame. The 1440p render cost is
measured GPU-side by `python -m vfx.render3d.bench` (see `vfx/render3d/README.md`).

## Backends
`backends.py` imports `vfx.sim3d` (Solver3D, `runtime_config` for the 64³ sim, full-quality
`SolverConfig3D` for the 128³ reference) and `vfx.runtime3d` (classical_upscale_3d, VolumeUpscaler,
ScreenDetail). The real solver is smoke-tested with one step at start-up; if import or that step
fails, `_stubs.py` (torch toy solver, trilinear upscale) is used and listed in the HUD / stats JSON.
`VolumeUpscaler` captures a CUDA graph per active-brick count on first use (hitches the first time
the plume grows); `--warmup-neural` pre-captures all of them at start-up instead (took 70 s on the
shared GPU). The toy stub solver is a slow torch fallback (~45 ms/tick), not a performance number. Route B gets
`aux={"bbox": projected domain-cube bbox, "transmittance": renderer alpha}`; the bbox size is
bucketed to quarters of the view because ScreenDetail captures one CUDA graph per crop size (an
orbiting camera otherwise re-captured every few frames, ~1.9 s hitch each).

Files: `main.py` (window/headless loops, input, HUD layout), `pipeline.py` (GPU state, timers),
`hud.py`, `presets.py`, `backends.py`, `_stubs.py`.

## Measured (shared GPU, 96 % busy with other processes; see render3d README)
Window `--quit-after 10 --orbit`, campfire, trilinear 256�: 60.5 fps (capped), 1% low 55, 2.4 ms
GPU per frame. Headless uncapped (lighter contention): trilinear 405 fps, route B 421 fps; route A
110 fps (neural_vol 2.4 ms/tick); ref128 85 fps (128� full-quality sim 14.7 ms/tick).
