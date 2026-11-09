# vfx/app — interactive playground

```
.venv/Scripts/python -m vfx.app.main                       # window 1600x1000
.venv/Scripts/python -m vfx.app.main --headless --frames 600 --preset 1 --mode neural \
    --screenshot out.png --stats out.json [--side-by-side] [--demo-input]
```
Flags: `--preset 1-4`, `--mode lr|bicubic|neural`, `--side-by-side`, `--checkpoint PATH`
(default `paths.default_checkpoint()`), `--quit-after SECONDS` (windowed), `--fps` cap,
`--stub-sim` (force the toy solver), `--demo-input` (headless scripted drag / torch / paint).

## Controls
| key | action |
|---|---|
| N | cycle mode: LR raw (128) → classical bicubic → neural (skipped if unavailable) |
| V | side-by-side: bicubic \| neural (bicubic \| LR raw if neural is unavailable) |
| LMB drag | velocity push (Impulse) |
| RMB hold | paint smoke |
| F | tap: toggle a fire emitter at the cursor; hold (>0.25 s): torch follows the cursor |
| 1–4 | presets: campfire, smoke stack + circle, twin torches + wind, box obstacles (OOD) |
| O | toggle obstacle (keeps the fluid) |
| R / Space / H / Esc | reset / pause / HUD / quit |

## Frame loop
Sim + upscale at 30 Hz (`SIM_DT`), render at display rate by lerping the last two upscaled
frames. Input is applied at the next sim tick; it becomes visible within 2–3 display frames.
Each frame: GPU compose of all views into one 1600x1000 canvas → ONE `.cpu()` blit →
`pygame.image.frombuffer`. CUDA-event timings (sim, upscale_bicubic, upscale_neural, render) are
read only after that blit. HUD: fps, 1% low, GPU ms per stage, amortized total, VRAM peak.

## Backends
`backends.py` imports `vfx.sim.Solver` and `vfx.runtime` (Upscaler, classical_upscale); if an
import fails the app falls back to `_stubs.py` (torch toy solver, bicubic) and lists the stubs in
the HUD / stats JSON. If `Upscaler(...)` raises (e.g. no checkpoint) neural mode is shown as
unavailable and `--mode neural` falls back to bicubic.

Files: `main.py` (window/headless loops, input), `pipeline.py` (GPU state, interpolation,
timers), `hud.py`, `presets.py`, `backends.py`, `_stubs.py`.
