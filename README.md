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

## Measured, 2D demo (RTX 5080, idle GPU, 2026-11-09, checkpoint v3_seq_run1, solver 0.2.0)
- Windowed, neural side-by-side: 60.7 fps, 1% low 56.2 fps (spec >= 60 / >= 50)
- GPU per stage: sim 0.76 ms, neural upscale 0.71 ms, render 0.17 ms, frame ~1.8 ms (budget 6 ms)
- Detail vs bicubic on the live-style 128² sim input: spectrum error 79% lower on test scenes, 71% on unfamiliar scenes (spec gate >= 50%)
- Flicker vs high-res reference (spec gate <= 1.2x): 0.65 test, 0.65 unfamiliar, 0.50 live fire, 1.01 live smoke + obstacle (bicubic: 0.88, 0.85, 0.64, 1.37)
- Trained on dataset v3: the 512² sim is steered to follow the independent 128² sim the app runs. History in vfx/train/EXPERIMENTS.md.
- Known gaps: a few small detached wisps at plume edges; faint dark rim at the smoke boundary in preset 3.

## Pipeline
```
.venv/Scripts/python -m vfx.datagen.build --version v3 --guided --n-train 200 --n-test 32 --n-ood 32 --frames 16   # ~2 min, 5.9 GB in %LOCALAPPDATA%/neural-vfx
.venv/Scripts/python -m vfx.train.train_seq --config vfx/train/configs/v3_seq_ft.json  # 15 min per run
.venv/Scripts/python -m pytest tests -q
.venv/Scripts/python -m bench.run
```

## 3D sandbox (feasibility phase — no 3D network is trained yet)
```
.venv/Scripts/python -m vfx.app3d.main
```
Orbit with RMB drag, zoom with the wheel, LMB drag pushes smoke, F fire, G smoke puff, N / B change the left / right
mode, V side-by-side, 1–4 presets, T march scale, C auto-orbit. Modes: raw 64³, classical trilinear ×4 (256³),
direct 128³ sim (reference), and the two neural routes, which run UNTRAINED weights for timing only.
Full controls: vfx/app3d/README.md.

### 3D feasibility, measured (RTX 5080, idle GPU, 2560×1440, half-resolution march; full table in bench3d/results/REPORT.md)
| Pipeline | sim | detail | render | frame | vs direct 256³ sim |
|---|---|---|---|---|---|
| raw 64³ | 0.82 | – | 0.84 | 1.66 ms | 27× |
| 64³ + trilinear 256³ | 0.83 | 0.18 | 1.52 | 2.52 ms | 18× |
| route A: neural volume 64³→256³ (untrained, box full of smoke) | 0.83 | 4.27 | 1.50 | 6.61 ms | 6.7× |
| route B: neural screen-space detail (untrained) | 0.84 | 2.08 | 0.85 | 3.76 ms | 12× |
| direct 128³ sim | 4.13 | – | 0.82 | 4.95 ms | 9× |
| direct 256³ sim | 43.0 | – | 1.34 | 44.4 ms | 1× |

Budget: sim 2.0, detail 2.5, render 1.5, frame 6.0 ms. Full-resolution marching misses the render budget (2.7–4.0 ms).
Route A's cost scales with how much of the box holds smoke (0.8 ms at 5% of bricks, 1.1 at 10%, ~2.3 at 25%); the bench
scenes fill the closed box, so it is measured at its worst case there. Sandbox presets dissipate smoke and hold at 31–39% active bricks.
```
.venv/Scripts/python -m pytest tests3d -q      # 57 tests
.venv/Scripts/python -m bench3d.run            # feasibility report
```
