# vfx/detail3d — sim-carried procedural detail (shortlist #1)

```python
from vfx.detail3d import AdvectedDetail
det = AdvectedDetail(64, quality="medium")         # low | medium | high (= 1 / 2 / 4 noise octaves)
det.step(state, SIM_DT)                            # every sim tick, after solver.step (no host sync)
img = renderer.draw(volume, camera, solid, detail=det)   # march + light volume sample the detail
det.timings                                        # {'step': ms} of the last finished step (lazy)
det.set_params(disp=0.03, freq0=10.0)              # any DetailParams field
```
`detail=None` (the default) runs the original kernels: frames are bit-identical to before (checked by
hashing 20 frames — 2 presets × march 0.5/1.0 × plain / reuse / aux / no solid / 256³ volume).

## Method
The detail lives in the volume, not on the image, so it is view-consistent by construction and needs
no motion vectors or training data. Nothing above 64³ is ever written.

**Per sim tick** (`step`, one CUDA graph: 2 Warp kernels + a box blur + 2 texture copies)
1. *Amplitude* (where detail belongs): per cell `a = 1 - exp(-w_vort |curl u| dx)` — the resolved
   small-scale motion of the coarse sim (the velocity difference across one cell), saturated. This is
   Houdini Gas Up Res' "Curl" source / the energy estimate of wavelet turbulence: detail only appears
   where the coarse sim is already turbulent. Gated by a density/heat presence ramp over the 3³
   neighbourhood (no detail in empty space), zeroed in and next to obstacles, damped in the hot flame
   core (`heat_damp`: the core stays coherent, tongues form at its edge), 3³ box-smoothed (no kinks).
2. *Advected noise coordinates*: two displacement fields `D_k = R_k - x` (rest coordinate minus
   position, 64³ float32 textures) are carried by the sim velocity (RK2 semi-Lagrangian):
   `D_new(x) = D_old(x_b) + (x_b - x)`. Stretching is handled the Neyret / flow-map way: each set lives
   one `period` (1 s), the two are half a period apart, a set restarts at `D = 0` with a new random
   noise offset when its phase wraps, and its weight `w_k = sin²(π phase_k)` is exactly 0 at that
   moment — regeneration never pops. Weights are variance-normalized (`w / |w|`) so contrast does not
   dip mid-blend.

**Per ray-march sample** (`kernels.detail_sample`, compiled into the march and the light volume)
`q' = q + amp(q) · Σ_o disp·g^o · Σ_k w_k · curl_noise(rot^o (q + D_k(q) + off_k) · f0 · lac^o)`;
density and temperature are both read at `q'` (optionally density × `(1 + mult·amp·scalar_noise)`).
Displacing the sample point advects the *shape*: the smooth 64³ edges fold into filaments and
billows; the interior (flat density) is unchanged. The curl noise is divergence-free, so the map is
volume-preserving to first order and smoke is neither added nor removed on average. Flames get
tongues from the same displacement, with no colour shift (same blackbody of a displaced field).
Noise: a 64³ periodic tile, band-limited by construction in the FFT (|k| in [5, 10) cycles/tile,
raised-cosine edges), curl computed spectrally (exactly divergence-free) + a scalar channel, fp16,
hardware trilinear + WRAP. One octave = one texture tap per set; successive octaves are rotated by
a fixed irrational rotation so tiles never line up. Octave 0 is centred at `freq0` cycles/m.
**Lighting sees the detail**: the light volume (key-light transmittance + sky occlusion) marches the
detailed density too (`light_volume_detail`), at its own resolution (80³). Occupancy bricks are
dilated by one brick for detail draws (displaced samples can read smoke up to a brick away).

### Tried and rejected / kept off
* Multiplicative density noise alone (`mult`, `disp=0`): adds high-frequency power (spectrum close to
  the reference) but looks like dirt on glass — grain inside the plume, no new silhouette. Combined
  with displacement it adds little visible; default `mult = 0`.
* High noise frequencies (`freq0` 28 cycles/m, 2 octaves to ~112): fine grain/fizz, the fire breaks into
  speckles. The 256³ reference's extra structure is mostly at 2–4 coarse cells (billows, curled sheet
  edges), so octave 0 now sits at 12 cycles/m with a large displacement (2.5 cm, ~1.6 cells).
* Heat-driven amplitude (`w_heat`): shattered the flame core. Replaced by `heat_damp` (less there).

## Parameters (`DetailParams`; defaults are the tuned values)
| name | default | meaning |
|---|---|---|
| octaves | 2 (quality) | noise taps per coordinate set: low 1 / medium 2 / high 4 (centres 12-96 cycles/m, band edge 128 = 256³ Nyquist) |
| freq0, lac | 12 cycles/m, 2 | centre frequency of octave 0 (64³ Nyquist = 32), lacunarity |
| disp, disp_gain | 0.025 m, 0.5 | displacement at octave 0 (× amplitude), per-octave factor (0.5 ∝ wavelength) |
| mult, mult_gain | 0, 0.7 | multiplicative density detail (off) |
| period | 1.0 s | lifetime of a coordinate set (regeneration) |
| w_vort | 12 | amplitude = 1 - exp(-w_vort · |curl u| dx) |
| heat_damp | 0.85 | amplitude × (1 - heat_damp · (T - T_amb)/1000) |
| d_lo, d_hi | 0.004, 0.04 | density presence ramp (3³ max) |
| smooth | 1 | box-blur passes on the amplitude |
| max_disp | 0.3 m | cap on the advected displacement |

## Measured costs
See "Costs" below (filled from `python -m vfx.detail3d.bench --out vfx/detail3d/bench_latest.json`;
the sandbox HUD reads that file for the march delta).

## Sources (research notes)
`reports/Real time neural smoke techniques.md` (ranked shortlist #1, "Procedural upres is cheap,
production-proven"), `research_notes/.../fluid_detail_synthesis.md` §2 (wavelet turbulence, Kim et al.
2008; Houdini Gas Up Res "Wavelets"/"Curl" sources, frequency cutoff, amplitude blur, rest field),
`alternative_representations.md` §2 (Neyret 2003 advected textures; Yu et al. 2011 Lagrangian texture
advection; Niagara Fluids Cine template advecting texture coordinates).

Files: `advected.py` (AdvectedDetail, DetailParams, QUALITY), `kernels.py` (per-tick kernels +
`detail_sample` used by vfx/render3d), `noise.py` (band-limited curl/scalar noise tile), `bench.py`.
