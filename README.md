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

## 3D sandbox — current best (2026-11-10)
```
.venv/Scripts/python -m vfx.app3d.main --best          # particles + calibrated sim-carried noise + faster pressure solve, next to untouched raw 64³
.venv/Scripts/python -m vfx.app3d.main --best-light    # same without particles (cheaper)
```
Orbit with RMB drag, zoom with the wheel, LMB drag pushes smoke, F fire, G smoke puff, 1–4 presets, V side-by-side,
N / B change the left / right mode, Q detail quality, P faster pressure solve, K resolution-consistent swirl strength,
C auto-orbit. Every earlier mode is still there (raw, trilinear, neural_vol, neural_screen, ref128, detail, hires, ...).
Full controls: vfx/app3d/README.md. State before this phase: git tag `v0.3-screen-space-baseline`.

### How fine detail is produced
Detail is carried by the 64³ sim, not invented on the image:
- `vfx/detail3d` — noise coordinates advected by the 64³ velocity, sampled inside the ray-marcher (no 256³ volume is written).
- `vfx/particles3d` — 1M particles advected by the 64³ velocity, deposited into a 128³ volume each sim step.
- `vfx/sim3d` opt-ins — `fused_config()` pressure solve, `scalar_res=128` smoke field, `confinement_ref_res=128`.
The screen-space network (`vfx/runtime3d.ScreenDetail`) sharpens but does not produce wisps: an unconstrained teacher
network (5× the compute) and a sharper 320³ target both failed to change that (vfx/train3d/EXPERIMENTS.md).
Background: reports/Real time neural smoke techniques.md.

### Quality vs the steered 256³ reference (four held-out presets, lower is better, reference = 0)
| Approach | detail by scale | patch statistics | motion match |
|---|---|---|---|
| raw 64³ | 0.359 | 0.220 | 0.440 |
| screen-space network (regression) | 0.088 | 0.125 | 0.305 |
| screen-space network (adversarial; rejected on pictures) | 0.051 | 0.068 | 0.194 |
| sim-carried noise, calibrated | 0.119 | 0.080 | 0.167 |
| particles 1M @128³ | 0.118 | 0.075 | 0.129 |
| **particles + calibrated noise (`--best`)** | **0.051** | **0.041** | **0.050** |
Crops: reports/images/best_vs_reference_crops.png. All approaches and metrics: bench3d/results/MODES.md.

### Cost on an idle RTX 5080 at 2560×1440, half-resolution march (budget: sim 2.0 / detail 2.5 / render 1.5 / frame 6.0 ms)
| Stage | ms |
|---|---|
| 64³ sim step, default / faster pressure solve | 0.88–1.01 / 0.48–0.62 |
| + 128³ smoke field | +0.38 |
| sim-carried noise, per sim step | 0.08–0.11 |
| particles 1M @128³, per sim step | 1.41 |
| render: plain / with calibrated noise (average of new-volume and reuse frames) | 0.74 / 1.08 |
| `--best` in the sandbox (800×690 views), per 60 Hz frame | ~2.0 (sim 0.32, detail 0.79, render 0.85) |
| direct 256³ sim step, for comparison | 43 |
Windowed `--best`: 60.6 fps, 1% low 55.7 (frame-capped).

### Known gaps
- Fire: particle modes put a speckled grain into flames; the 256³ reference flame is smooth and taller.
- The 64³ sim over-stirs plumes; `confinement_ref_res=128` fixes most of it (error vs a free-running 128³ sim 4.61 → 1.75)
  but changes the flow the datasets and network were built on, so it is opt-in (vfx/correct3d/README.md).
- Native runtime (D3D12 + DirectML, no Python) lives on branch `native-runtime`, not merged.
```
.venv/Scripts/python -m pytest tests3d -q        # 167 pass; 2 known failures: the first screen-space network vs the corrected detail gates
.venv/Scripts/python -m bench3d.compare_modes    # all-approaches table
```
