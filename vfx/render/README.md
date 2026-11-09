# vfx/render — 2D smoke + fire renderer

`Renderer(out_size).draw(fields: HRFields, solid=None, nearest=False) -> (out_h, out_w, 3) uint8 cuda`
(row 0 = top of screen). Accepts any input resolution (128 LR, 512 HR, ...); `nearest=True` shows
raw cells (the app's "LR raw" view). The returned tensor is an internal buffer reused by the next call.

## Look
- Smoke: absorbing/scattering medium, `alpha = 1 - exp(-k_abs * density)`, albedo × (ambient + key light × shadow).
- Self-shadow: optical depth toward the light (upper-left) from 24 bilinear density samples
  (one batched `grid_sample` at 256², blurred). Obstacles also cast shadow.
- Rim / forward scatter on thin, lit edges: `rim * T_light * 4a(1-a)`.
- Fire: blackbody emission from temperature via a Planck→linear-sRGB LUT (700–3000 K, CIE 1931
  Wyman fit, luminance compressed `(Y/Y_1400K)^0.33`), emissivity grows with soot density;
  hot regions darken smoke albedo (soot). Additive 2-level bloom (128² + 32²) that also lights
  nearby smoke and obstacles.
- Dark vertical-gradient background with vignette, obstacles as solid shapes with a lit rim.
- ACES-fit tone map, sRGB encode, ±0.5 LSB dither.
- All knobs live in `RenderParams` (`renderer.set_params(...)` re-bakes and drops captured graphs).

## Implementation
- `renderer.py`: torch prep at 256² (shadow march, emission → bloom) + one fused Warp kernel
  (`kernels.composite`) at output resolution. The whole draw is captured once per
  (input shape, solid shape, nearest) as a CUDA graph; each draw = 3 input copies + 1 replay.
- `blackbody.py`: the LUT (numpy, computed once at init).
- `selfcheck.py`: procedural fields → PNG + timing:
  `.venv/Scripts/python -m vfx.render.selfcheck --out x.png --size 1024`

## Measured (RTX 5080, shared with other GPU jobs during measurement)
- 512² → 1024² draw: ~0.25 ms mean, ~0.18 ms p50 (CUDA events, graph replay); GPU-bound
  throughput 0.24 ms/draw. Eager (no graph) is ~0.26 ms of GPU work but ~1.2 ms of launch overhead.
- In the app at 960² per view: ~0.18 ms p50.
