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

## Measured (RTX 5080, 2026-11-09, checkpoint v2_seq_run3)
- Windowed, neural side-by-side: 60.5 fps, 1% low 55.7 fps (spec >= 60 / >= 50)
- GPU per stage: sim 0.76 ms, neural upscale 0.71 ms, render 0.17 ms, frame ~1.9 ms (budget 6 ms)
- Detail vs bicubic on the live-style 128² sim input: spectrum error 75% lower on test scenes, 67% on unfamiliar scenes (spec gate >= 50%)
- Flicker vs high-res reference (spec gate <= 1.2x): 0.58 test, 0.59 unfamiliar, 0.45 live fire, 1.02 live smoke + obstacle (bicubic: 0.83, 0.80, 0.61, 1.41)
- Trained on dataset v2: the 512² sim is steered to follow the independent 128² sim the app runs. History in vfx/train/EXPERIMENTS.md.
- Known gaps: a few small detached wisps at plume edges; jagged flame column in the 512² solver output.

## Pipeline
```
.venv/Scripts/python -m vfx.datagen.build --version v2                        # ~2 min, 5.9 GB in %LOCALAPPDATA%/neural-vfx
.venv/Scripts/python -m vfx.train.train_seq --config vfx/train/configs/v2_seq_mix.json  # 15 min per run
.venv/Scripts/python -m pytest tests -q
.venv/Scripts/python -m bench.run
```
