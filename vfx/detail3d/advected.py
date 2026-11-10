"""AdvectedDetail: procedural sub-grid detail carried by the coarse sim's velocity (shortlist #1).

Per sim tick (`step`, one small CUDA graph):
  1. amplitude: per-cell small-scale energy of the coarse sim (|curl u| dx, + heat), saturated to 0..1,
     gated by (dilated) density / heat presence and zeroed in and next to obstacles, 3^3 box-smoothed.
  2. two displacement sets D_k = R_k - x of noise rest coordinates are advected by the sim velocity
     (RK2 semi-Lagrangian). Each set lives one `period`, phase-offset by half a period, and restarts at
     D = 0 with a new random offset when its phase wraps; its blend weight w_k = sin^2(pi phase_k) is
     0 at that moment, so regeneration never pops (Neyret 2003 advected textures / flow-map blending
     as in Niagara's advected noise coordinates). Weights are variance-normalized (w_k / |w|) so the
     noise contrast does not dip mid-blend.
Per ray-march sample (`vfx.detail3d.kernels.detail_sample`, called by VolumeRenderer.draw(detail=...)):
  q -> q + amp(q) * sum_o disp * g^o * curl_noise(rot^o R_k(q) f0 lac^o), blended over k
  density and temperature are both read at the displaced point (wisps by "advecting the shape";
  flames get tongues, no colour shift), optionally x (1 + mult * amp * scalar noise).
The noise is a 64^3 periodic band-limited tile (noise.py), so each octave is ONE hardware-filtered
texture tap per set. No high-res volume is ever written.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace

import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

from vfx.core.fields3d import AMBIENT_TEMP, CH3
from vfx.detail3d import kernels as K
from vfx.detail3d.noise import band_noise_tile, rotation

TILE_N, TILE_KLO, TILE_KHI = 64, 5.0, 10.0
TILE_KC = 7.5                      # centre of the tile's band, cycles per tile


@dataclass
class DetailParams:
    octaves: int = 2               # noise taps per set (quality)
    freq0: float = 12.0            # centre frequency of octave 0, cycles per meter (64^3 Nyquist = 32)
    lac: float = 2.0
    disp: float = 0.025            # displacement at octave 0, meters (a 64^3 cell = 0.0156)
    disp_gain: float = 0.5         # per octave (0.5 = displacement proportional to wavelength)
    mult: float = 0.0              # multiplicative density detail (0 = displacement only)
    mult_gain: float = 0.7
    period: float = 1.0            # seconds each coordinate set lives before it is regenerated
    w_vort: float = 12.0           # amplitude = 1 - exp(-w_vort |curl u| dx - w_heat heat)
    w_heat: float = 0.0
    heat_damp: float = 0.95        # less displacement in the hot flame core (keeps it coherent)
    damp_nb: float = 1.0           # 0: damp by the cell's own heat, 1: by the 3^3 max (thin flames)
    d_lo: float = 0.004            # density presence ramp (3^3 max) -> amplitude
    d_hi: float = 0.04
    amp_floor: float = 0.0         # minimum amplitude where smoke is present (0 = purely energy-keyed)
    smooth: int = 1                # box-blur passes on the amplitude (avoid kinks)
    max_disp: float = 0.3          # cap on |D| (m)
    amp_eps: float = 0.02
    t_disp: float = 0.3            # fraction of the displacement applied to temperature in flames
    t_hot: float = 700.0           # K: above this the temperature is re-read at the t_disp point
    light_octaves: int = 1         # octaves the light/shadow volume samples (finer ones are below its res)
    seed: int = 1
    # --- spectrum calibration and base awareness (2026-11-09, after bench3d MODES.md)
    octave_amps: tuple = ()        # per-octave displacement (m), overrides disp * disp_gain^o when given (<= 4)
    base_res: int = 64             # effective resolution of the base volume the detail is drawn on. 128 (hires
                                   # scalars, particles) shifts the noise up one octave and halves the
                                   # displacement: only the bands the base field lacks are added
    amp_scale: float = 1.0         # multiplies every octave's displacement (stacked modes: base already detailed)
    cov_damp: float = 0.0          # amplitude x (1 - cov_damp * coverage) where step() gets a coverage field
    cov_full: float = 4.0          # coverage count per fine cell that counts as fully covered


QUALITY = {
    "low": dict(octaves=1, freq0=12.0, disp_gain=0.5, octave_amps=()),
    "medium": dict(octaves=2, freq0=12.0, disp_gain=0.5, octave_amps=()),
    "high": dict(octaves=4, freq0=12.0, disp_gain=0.45, octave_amps=()),
    # band-calibrated (tune_bands --search, 2026-11-09): ONE octave centred at 9 cycles/m with 2.5 cm displacement.
    # Mean over presets 1-4 vs the steered 256³: band distance 0.119 (medium 0.270, low 0.183), patch SWD 0.080
    # (0.107 / 0.078), tLP 0.167 (0.093 / 0.133). 8 cycles/m scored 0.096 / 0.090 / 0.189 but failed
    # tests3d test_detail_moves_with_the_flow[8] (corr 0.20 < 0.3: decorrelates faster in a rotating flow);
    # 9 passes. A second, finer octave measured no better at any amplitude (it only adds 2-4 px hatching).
    "calibrated": dict(octaves=1, freq0=9.0, octave_amps=(0.025,)),      # octave centres 12, 24, 48, 96 cycles/m: up to the 256^3 Nyquist (128)
}


# Detail drawn on a base volume that already has sub-grid structure (bench3d MODES.md: stacking medium detail on
# hires / particles overshoots the reference's fine bands). Only the bands the base lacks are added.
# Measured with `python -m vfx.detail3d.tune_bands` (calibrated amps, presets 1-4 vs steered 256³):
#   particles + calibrated detail: cov_damp 0.7 -> band distance 0.051, patch SWD 0.041, tLP 0.050 (particles alone
#     0.118 / 0.075 / 0.129; un-damped stacking 0.255 / 0.108; base_res 128 measured slightly worse -> not used).
#   hires + calibrated detail: amp_scale 0.5 -> 0.087 / 0.077 / tLP 0.142 (hires alone 0.199 / 0.143 / 0.282; full
#     amplitude 0.222; base_res 128 (noise one octave up) 0.216 -> not used: the 128³ field lacks power at 8-16 px,
#     not only above its Nyquist).
STACKED = {
    "hires": dict(amp_scale=0.5),
    "particles": dict(cov_damp=0.7),
}


class AdvectedDetail:
    def __init__(self, res: int = 64, device="cuda", quality: str = "calibrated", cuda_graph: bool = True, **params):
        wp.init()
        self.res = int(res)
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.wp_device = wp.device_from_torch(self.device)
        self.p = DetailParams()
        self.quality = None
        self.set_quality(quality)
        self.set_params(**params)
        self.cuda_graph = cuda_graph
        self.timings: dict[str, float] = {}
        self._events = None
        n = self.res
        tile = band_noise_tile(TILE_N, TILE_KLO, TILE_KHI, seed=11, device=self.device)
        self._tile_h = tile.half().contiguous()
        self.noise_tex = wp.Texture3D(width=TILE_N, height=TILE_N, depth=TILE_N, num_channels=4, dtype=wp.float16,
                                      filter_mode=wp.TextureFilterMode.LINEAR,
                                      address_mode=wp.TextureAddressMode.WRAP, device=self.wp_device)
        self.noise_tex.copy_from(wp.from_torch(self._tile_h, dtype=wp.vec4h))
        self.tex = [self._tex(n) for _ in range(2)]
        self.new_t = [torch.zeros(n, n, n, 4, device=self.device) for _ in range(2)]
        self.new = [wp.from_torch(t, dtype=wp.vec4) for t in self.new_t]
        self.amp_raw = torch.zeros(n, n, n, device=self.device)
        self.amp = torch.zeros(n, n, n, device=self.device)
        self.dyn = torch.zeros(8, device=self.device)               # read by the march (see kernels)
        self.reset_flags = torch.zeros(2, dtype=torch.int32, device=self.device)
        from vfx.render3d.pinned import PinnedRing
        self._pin = PinnedRing(8)                 # per-tick staging, safe however many ticks are queued
        self._pin_r = PinnedRing(2, torch.int32)
        self._graphs: dict = {}
        self._wp_stream = None
        self._wp_stream_key = None
        self.reset()

    # ------------------------------------------------------------------ config
    def _tex(self, n):
        return wp.Texture3D(width=n, height=n, depth=n, num_channels=4, dtype=wp.float32,
                            filter_mode=wp.TextureFilterMode.LINEAR, address_mode=wp.TextureAddressMode.CLAMP,
                            device=self.wp_device)

    def set_quality(self, q: str) -> None:
        if q not in QUALITY:
            raise ValueError(f"quality {q!r} not in {list(QUALITY)}")
        self.quality = q
        self.p = replace(self.p, **QUALITY[q])
        self.version = getattr(self, "version", 0) + 1

    def set_params(self, **kw) -> None:
        names = {f.name for f in fields(DetailParams)}
        bad = set(kw) - names
        if bad:
            raise TypeError(f"unknown detail params {sorted(bad)}")
        if kw:
            self.p = replace(self.p, **kw)
        self.version = getattr(self, "version", 0) + 1
        if hasattr(self, "_graphs"):
            self._graphs.clear()

    def shade(self, light: bool = False) -> K.DetailShade:
        p, s = self.p, K.DetailShade()
        k = float(p.base_res) / 64.0 / float(p.amp_scale)   # base-aware: same detail relative to the base grid
        s.disp, s.disp_gain, s.mult, s.mult_gain = float(p.disp) / k, float(p.disp_gain), float(p.mult), float(p.mult_gain)
        s.octaves = int(min(p.octaves, p.light_octaves)) if light else int(p.octaves)
        s.freq0 = float(p.freq0) * float(p.base_res) / 64.0 / TILE_KC      # tiles per meter
        oa = p.octave_amps if isinstance(p.octave_amps, (tuple, list)) else (p.octave_amps,)   # a number = 1 octave
        amps = [float(a) / k for a in oa][:4]
        s.use_amps = 1 if amps else 0
        s.amps = wp.vec4(*(amps + [0.0] * (4 - len(amps))))
        if amps:
            s.octaves = min(s.octaves, len(amps))
        s.lac = float(p.lac)
        s.amp_eps = float(p.amp_eps)
        s.rot = wp.mat33(*[v for row in rotation() for v in row])
        s.t_disp, s.t_hot = float(p.t_disp), float(p.t_hot)
        return s

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.tick = 0
        self.rng = np.random.default_rng(self.p.seed)
        self.offsets = self.rng.uniform(0, 1, size=(2, 3))
        for t in self.new_t:
            t.zero_()
        for k in range(2):
            self.tex[k].copy_from(self.new[k])
        self.amp.zero_()
        self._write_dyn(np.zeros(2, np.int32))

    def phases(self, tick: int) -> np.ndarray:
        n = max(1, int(round(self.p.period / (1.0 / 30.0))))     # ticks per period (SIM_DT = 1/30)
        return np.array([((tick / n) + 0.5 * k) % 1.0 for k in range(2)])

    def _write_dyn(self, resets) -> None:
        ph = self.phases(self.tick)
        w = np.sin(np.pi * ph) ** 2
        w = w / max(1e-6, math.sqrt(float((w ** 2).sum())))
        pin, pr = self._pin.acquire(), self._pin_r.acquire()
        pin[:2] = torch.from_numpy(w.astype(np.float32))
        pin[2:8] = torch.from_numpy(self.offsets.reshape(-1).astype(np.float32))
        pr[:] = torch.from_numpy(np.asarray(resets, np.int32))
        self.dyn.copy_(pin, non_blocking=True)
        self.reset_flags.copy_(pr, non_blocking=True)
        self._pin.record()
        self._pin_r.record()

    def _stream(self):
        ts = torch.cuda.current_stream(self.device)
        if self._wp_stream_key != ts.cuda_stream:
            self._wp_stream = wp.stream_from_torch(ts)
            self._wp_stream_key = ts.cuda_stream
        return self._wp_stream

    # ------------------------------------------------------------------ per tick
    @torch.no_grad()
    def step(self, state, dt: float, coverage: torch.Tensor | None = None) -> None:
        """Advect the detail coordinates by state's velocity and refresh the amplitude. No host sync.
        coverage (optional, any res, e.g. particle count per 128^3 cell): with cov_damp > 0 the amplitude
        is reduced where the base volume already carries sub-grid structure."""
        self._resolve()
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        ev[0].record()
        self.tick += 1
        ph_prev, ph = self.phases(self.tick - 1), self.phases(self.tick)
        resets = (ph < ph_prev).astype(np.int32)                # phase wrapped -> regenerate that set
        for k in np.nonzero(resets)[0]:
            self.offsets[k] = self.rng.uniform(0, 1, size=3)
        with wp.ScopedStream(self._stream(), sync_enter=False, sync_exit=False):
            self._write_dyn(resets)
            data = state.data
            cov = coverage if (coverage is not None and self.p.cov_damp > 0) else None
            key = (data.data_ptr(), tuple(data.shape), state.solid.data_ptr(), float(dt), self.version,
                   None if cov is None else (cov.data_ptr(), tuple(cov.shape)))
            fn = lambda: self._tick_work(data, state.solid, float(dt), cov)   # noqa: E731
            if self.cuda_graph:
                g = self._graphs.get(key)
                if g is None:
                    self._graphs.clear()
                    g = self._graphs[key] = self._capture(fn)
                g.replay()
            else:
                fn()
        ev[1].record()
        self._events = ev

    def _tick_work(self, data, solid, dt, cov=None) -> None:
        p = self.p
        D = data.shape[1]
        amp_raw = self.amp_raw if D == self.res else torch.zeros(D, D, D, device=self.device)
        wp.launch(K.energy_amp, dim=(D, D, D),
                  inputs=[wp.from_torch(data), wp.from_torch(solid), CH3["density"], CH3["vel_x"], CH3["temperature"],
                          float(p.w_vort), float(p.w_heat), float(p.heat_damp), float(p.damp_nb), float(AMBIENT_TEMP), float(p.d_lo), float(p.d_hi),
                          wp.from_torch(amp_raw)])
        a = amp_raw[None, None]
        if D != self.res:
            a = F.interpolate(a, size=(self.res,) * 3, mode="trilinear", align_corners=False)
        for _ in range(int(p.smooth)):
            a = F.avg_pool3d(F.pad(a, (1,) * 6, mode="replicate"), 3, 1)
        if p.amp_floor > 0:
            pres = (a > 1e-4).float()
            a = torch.maximum(a, p.amp_floor * pres)
        if cov is not None:
            c = F.adaptive_avg_pool3d(cov.float()[None, None], self.res) if cov.shape[-1] != self.res else cov.float()[None, None]
            a = a * (1.0 - p.cov_damp * (c / p.cov_full).clamp(0.0, 1.0))
        self.amp.copy_(a[0, 0])
        n = self.res
        wp.launch(K.advect_coords, dim=(n, n, n),
                  inputs=[wp.from_torch(data), CH3["vel_x"], self.tex[0], self.tex[1], wp.from_torch(self.amp),
                          dt, wp.from_torch(self.reset_flags), self.new[0], self.new[1], float(p.max_disp)])
        self.tex[0].copy_from(self.new[0])
        self.tex[1].copy_from(self.new[1])

    def _capture(self, fn):
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        # warm-up must not advance the coordinates: run it on scratch copies? The work is idempotent
        # only per tick, so warm up on a throwaway pass and restore the inputs afterwards.
        saved = [t.clone() for t in self.new_t]
        with torch.cuda.stream(side):
            with wp.ScopedStream(wp.stream_from_torch(side), sync_enter=False, sync_exit=False):
                fn()
        torch.cuda.current_stream(self.device).wait_stream(side)
        for t, s in zip(self.new_t, saved):
            t.copy_(s)
        for k in range(2):
            self.tex[k].copy_from(self.new[k])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with wp.ScopedStream(self._stream_for_capture(), sync_enter=False, sync_exit=False):
                fn()
        return graph

    def _stream_for_capture(self):
        return wp.stream_from_torch(torch.cuda.current_stream(self.device))

    # ------------------------------------------------------------------ render interface
    def fields(self) -> dict:
        """What the ray-marcher samples: two coordinate textures (xyz displacement; set 0's w =
        amplitude), the noise tile, the static shade struct and the per-tick device params."""
        return {"coords": self.tex, "noise": self.noise_tex, "shade": self.shade(), "dyn": self.dyn,
                "shade_light": self.shade(light=True), "amp": self.amp, "version": self.version}

    # ------------------------------------------------------------------ timing
    def _resolve(self) -> None:
        ev = self._events
        if ev is None or not ev[1].query():
            return
        self.timings = {"step": ev[0].elapsed_time(ev[1])}

    def last_timings(self, sync: bool = False) -> dict:
        if sync and self._events is not None:
            self._events[1].synchronize()
        self._resolve()
        return dict(self.timings)

    # ------------------------------------------------------------------ analysis (not frame loop)
    @torch.no_grad()
    def evaluate(self, volume, n: int = 256, detail: bool = True):
        """Analysis: the detailed (density, temperature) point-sampled on an n^3 grid. Writes a big
        volume -- only for checks (mean conservation, spectra), never in the frame loop."""
        d = volume.density.float().contiguous()
        t = volume.temperature.float().contiguous()
        D, H, W = d.shape
        packed = torch.stack([d.clamp(min=0), t], -1).contiguous()
        tex = wp.Texture3D(width=W, height=H, depth=D, num_channels=2, dtype=wp.float32,
                           filter_mode=wp.TextureFilterMode.LINEAR, address_mode=wp.TextureAddressMode.CLAMP,
                           device=self.wp_device)
        tex.copy_from(wp.from_torch(packed, dtype=wp.vec2))
        od = torch.zeros(n, n, n, device=self.device)
        ot = torch.zeros(n, n, n, device=self.device)
        wp.launch(K.eval_detail_grid, dim=(n, n, n),
                  inputs=[tex, self.tex[0], self.tex[1], self.noise_tex, self.shade(), wp.from_torch(self.dyn),
                          int(detail), wp.from_torch(od), wp.from_torch(ot)])
        return od, ot
