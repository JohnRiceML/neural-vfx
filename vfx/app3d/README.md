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
### Window layout and HUD
The views sit above a dedicated 210 px HUD strip, so the HUD never covers a view (single: one
1600x690 view; side-by-side: two 799x690 views). `H` hides the strip and the views take the full window.
Strip columns: status (modes, preset, fps / 1% low, march size, VRAM, sim step, dissipation) | GPU
budget | key help. The budget table is for the main (left) view:

| column | meaning |
|---|---|
| tick | GPU ms of the stage when it runs (one 30 Hz sim tick; render = new-volume draw / camera-only redraw) |
| /frame | amortized per 60 Hz display frame, the figure the spec budgets: sim, volume upscale (trilinear or route A) and the render's volume prep run every other frame, so they count half; route B detail runs every frame; render = (new + reuse) / 2 |
| budget | spec per-stage budget: sim 2.0, detail 2.5 (volume upscale or screen detail), render 1.5; frame 6.0 |

Each stage has a bar (solid = /frame, outline = tick, white mark = its budget). The stacked bar is
the /frame total against the 6 ms frame budget (grey ticks at the stage allocations 2.0 / 4.5; the outline
is the tick-frame total, i.e. the worst frame). In side-by-side, a line gives the right view's own
/frame and tick totals. If `bench3d/results/latest.json` exists, the last line is the speed ratio:
measured direct 256³ sim ms/step (bench3d `brute256` rows, mean over march scales; `plume_sphere` for
preset 2, `campfire` otherwise) / this view's tick-frame total. It is omitted for `ref128`.

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
`aux={"bbox": projected domain-cube bbox, "transmittance": renderer alpha}`. The exact bbox is passed:
ScreenDetail now buckets crop sizes itself (fixed fractions of the frame), so the app-side bucketing
workaround was removed. Checked: 900 headless frames of route B, orbiting and zooming (dist 1.0-4.4):
4 graphs, 0 frames > 50 ms, max frame 11 ms.

Files: `main.py` (window/headless loops, input, HUD layout), `pipeline.py` (GPU state, timers),
`hud.py`, `presets.py`, `backends.py`, `_stubs.py`, `compare.py` (offline route B comparison).

## Route B: trained checkpoint, aux (2026-11-09)
* **Aux**: in `neural_screen` the raw 64³ view is drawn with `return_aux=True, velocity=<64³ state
  velocity>, prev_camera=<camera one SIM_DT ago>` and ScreenDetail gets
  `{"bbox", "alpha", "depth", "motion", "transmittance"}` (render3d README "Aux buffers"; `transmittance`
  = renderer T, kept for the current ScreenDetail). prev_camera = the camera of the draw at the same
  position within the previous sim tick (`Pipeline.prev_camera`), so motion is pixels per SIM_DT on
  every 60 Hz frame, as in the dataset. Adds ~0.15 ms to that view's render.
* **Checkpoint**: `backends.route_b_checkpoint()` asks `vfx.runtime3d` for a default path
  (`default_screen_checkpoint()` / `SCREEN_CHECKPOINT` / `DEFAULT_CHECKPOINT`; none exists yet) and
  falls back to `vfx.core.paths.CHECKPOINTS / "screen3d" / "latest.pt"`. If that file exists (or
  `--screen-ckpt PATH` is given) ScreenDetail loads it, the red UNTRAINED label is replaced by a green
  `trained: <dir>/<file> (<mtime>)`, and the stats JSON lists it under `trained`.
* **Comparison**: `python -m vfx.app3d.compare --preset K --frames N --out DIR [--size 960x540]
  [--every 1] [--warmup 90] [--orbit-deg 1] [--screen-ckpt PATH] [--no-hr]` runs the 64³ sim with an
  orbiting camera and writes labelled strips `frame_XXXX.png` (raw 64³ | trilinear 256³ | route B |
  guided 256³ reference), `contact_sheet.png` (8 frames) and `timings.json` (per-stage GPU ms, GPU
  note). The guided 256³ panel uses `vfx.datagen3d.sim.run_pair` with `guide3d.DEFAULT_GUIDE` (that
  run also provides the 64³ state, same runtime config as the sandbox); if it cannot be imported or
  fails, the panel is skipped and the 64³ sim runs on the sandbox backends. Route B runs every sim
  step (its temporal state), with the same aux as the sandbox.

## Presets: game-effect dissipation
`Preset.sim` holds SolverConfig3D overrides applied to the 64³ runtime solver AND the 128³ reference
(`Pipeline.set_sim_config`; Solver3D reads `config` every step, so there is no recapture). With the
solver defaults (density dissipation 0.05 /s) the smoke never leaves the closed 1 m box and after
~30 s it is a grey haze. Every preset now uses `density_dissipation 1.2 /s, temperature_cooling 5 /s`,
with stronger emitters (fire fuel 7 -> 10, smoke 2.5 -> 4.0 / 2.0 -> 3.2) to keep the plume body.
Steady state after 1800 sim steps (60 s), active 8³-cell bricks by the VolumeUpscaler definition
(density > 1e-3 or T > ambient + 1 K, dilated 2 cells), and voxels with density > 0.02:

| preset | active bricks (old config) | active bricks | voxels > 0.02 |
|---|---|---|---|
| 1 campfire | 100 % | 31 % | 0.8 % |
| 2 plume + sphere | 100 % | 38 % | 1.4 % |
| 3 twin torches | 100 % | 39 % | 0.7 % |
| 4 box obstacle | 100 % | 39 % | 1.2 % |

The fraction is flat from step 600 to 1800 (within ±3 points). The 1e-3 threshold plus the 2-cell
dilation counts very thin smoke under the lid, so ~1/3 of the bricks stay active while the visible
smoke is a localized plume. Fewer active bricks would need stronger dissipation (2.0 /s: 22-28 %),
which shortens the visible plume. The 128³ reference plume rises faster than 64³ (see sim3d README),
so it still reaches the lid and spreads there before it fades.

## Measured (shared GPU, 96 % busy with other processes; see render3d README)
Window `--quit-after 10 --orbit`, campfire, trilinear 256³: 60.5 fps (capped), 1% low 55, 2.4 ms
GPU per frame. Headless uncapped (lighter contention): trilinear 405 fps, route B 421 fps; route A
110 fps (neural_vol 2.4 ms/tick); ref128 85 fps (128³ full-quality sim 14.7 ms/tick).

## Sim-carried detail modes (2026-11-09; additive — every earlier mode is unchanged)
| mode | what is rendered |
|---|---|
| `detail` — 64³ + advected detail | raw 64³ volume drawn with `detail=AdvectedDetail` (vfx/detail3d), stepped with the 64³ velocity every tick |
| `detail_screen` — detail + route B | the `detail` render (+ aux) fed to a second ScreenDetail instance (same checkpoint, own temporal history) |
| `hires` — 64³ velocity + 128³ density | `Solver3D(64, scalar_res=128).hires_volume()` (sim3d opt-in; a separate solver instance, seeded from the current 64³ state when the mode becomes visible) |
| `hires_detail` — hires + detail | the 128³ density drawn with its own AdvectedDetail stepped by that solver's velocity |

```
.venv/Scripts/python -m vfx.app3d.main --mode detail --side-by-side --mode2 raw [--detail-quality high]
.venv/Scripts/python -m vfx.app3d.main --headless --frames 240 --mode hires_detail --side-by-side --mode2 detail --screenshot out.png
.venv/Scripts/python -m vfx.app3d.compare --preset 1 --frames 24 --every 8 --detail --out DIR
```
`Q` cycles the detail quality (low / medium / high = 1 / 2 / 4 noise octaves). Nothing is created or
stepped unless one of these modes is visible. HUD for these modes: the **detail** row is the
`AdvectedDetail.step` GPU ms per tick (amortized /2 per 60 Hz frame; for `detail_screen` plus the
network), the **render** row ("render+det") includes the detail sampling in the march and in the
light volume, and an extra green line gives the bench-measured 1440p march delta for the current
quality (from `vfx/detail3d/bench_latest.json`, written by `python -m vfx.detail3d.bench --out ...`).
In `hires*` modes the sim row is the scalar_res solver's step. If sim3d lacked `scalar_res` /
`hires_volume()` the hires modes would show UNAVAILABLE with that reason.

`compare --detail` (off by default, so the default panels and contact sheet are unchanged) inserts
`64³ + detail | detail + route B | hires 128³ + detail` between route B and the guided 256³ reference
(the hires panel only if sim3d supports scalar_res); `--detail-quality`, `--no-route-b-detail`.
