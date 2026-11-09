# Neural VFX Engine — v0 2D demo

Real-time smoke and fire: a 128² Warp solver plus a learned 4× detail network (to 512²), rendered with blackbody fire.
Design: [SPEC.md](SPEC.md). Conventions and module contracts: [CLAUDE.md](CLAUDE.md).

## Run the demo
```
.venv/Scripts/python -m vfx.app.main
```
| Key | Action |
|---|---|
| N | cycle mode: LR raw 128² → classical bicubic → neural |
| V | side-by-side bicubic \| neural |
| LMB drag | push the smoke |
| RMB | paint smoke |
| F | tap: toggle a fire at the cursor · hold: torch |
| 1–4 | presets: campfire, smoke stack + obstacle, twin torches + wind, box obstacle (out-of-distribution) |
| O / R / Space / H / Esc | obstacle, reset, pause, HUD, quit |

## Measured (RTX 5080, 2026-11-09)
- Windowed, neural side-by-side: 60.4 fps, 1% low 55.5 fps (spec ≥ 60 / ≥ 50)
- GPU per stage: sim 0.78 ms, neural upscale 0.75 ms, render 0.18 ms, frame 1.7 ms (budget 6 ms)
- Detail vs bicubic: spectrum error 70–77% lower (spec v1 gate ≥ 50%)
- Known gaps: on the live independent 128² sim the neural output flickers more than bicubic (2.8× reference vs 2.2×; spec gate 1.2×), and per-pixel L1 is ~2% worse than bicubic, because the 128² sim drifts from its 512² twin. See vfx/train/EXPERIMENTS.md.

## Pipeline
```
.venv/Scripts/python -m vfx.datagen.build --version v1                        # ~2 min, 5.7 GB in %LOCALAPPDATA%/neural-vfx
.venv/Scripts/python -m vfx.train.train --config vfx/train/configs/demo.json  # 15 min
.venv/Scripts/python -m pytest tests -q
.venv/Scripts/python -m bench.run
```
