"""3D smoke + fire volume renderer: Volume (any res) + Camera -> (out_h, out_w, 3) uint8 cuda.

Pipeline per draw (all GPU, no host sync; one tiny pinned H2D copy for the camera):
  volume prep (skipped with reuse_volume=True, i.e. 60 Hz frames between 30 Hz sim ticks):
    1. pack (direct launch): density/temperature -> half2 volume + 32^3 occupancy bricks.
    2. CUDA graph "prep": upload the half2 volume into a wp.Texture3D (hardware trilinear),
       light volume (light_res^3: transmittance toward the key light, sky occlusion, emission),
       fire-glow volume (32^3 + 8^3 blurred emission) -> lights smoke, obstacles and floor
  view (every draw), CUDA graph "view":
       d. march: one ray per pixel at march_scale x output res, front-to-back compositing with
          empty-brick skipping and early termination; opaque obstacles (solid iso 0.5); floor plane
       e. bloom from the HDR march buffer (2 levels), composite at output res: upsample, background,
          ACES tone-map, sRGB, dither -> uint8.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

from vfx.core.fields import SIM_DT
from vfx.core.fields3d import Camera, Volume
from vfx.render.blackbody import blackbody_lut
from vfx.render3d import kernels as K
from vfx.render3d.pinned import PinnedRing
from vfx.render3d import temporal as TK

GOLDEN = 0.6180339887498949


@dataclass
class RenderParams3D:
    steps: float = 192.0                 # ray-march samples per meter (cube side = 1 m)
    light_res: int = 80                  # light / shadow volume resolution (prep runs at 30 Hz only)
    k_abs: float = 40.0
    k_shadow: float = 120.0              # strong self-shadowing; the scattering octaves fill it back in
    k_solid: float = 60.0
    k_amb: float = 14.0
    k_emit: float = 3.0
    emit_min: float = 0.35
    emit_scale: float = 28.0
    soot: float = 0.9
    fire_light: float = 0.9
    phase_g: float = 0.6
    rim: float = 0.75                    # forward-lobe weight of the dual-lobe phase function
    ms_a: float = 0.3                    # multiple-scattering octaves (energy, extinction, anisotropy)
    ms_b: float = 0.5
    ms_c: float = 0.5
    exposure: float = 1.0
    contrast: float = 1.15               # power curve around mid-grey before ACES (1 = off)
    bloom: float = 0.7
    bloom_thresh: float = 0.8
    t_lo: float = 700.0
    t_hi: float = 3000.0
    t_ref: float = 1400.0
    d_eps: float = 2e-3
    light_dir: tuple = (-0.80, 0.50, -0.20)      # toward the light: low from the left -> form + long shadows
    key_col: tuple = (1.35, 1.25, 1.08)
    amb_col: tuple = (0.05, 0.065, 0.10)
    smoke_albedo: tuple = (0.68, 0.68, 0.72)
    bg_top: tuple = (0.040, 0.048, 0.070)
    bg_bot: tuple = (0.008, 0.008, 0.012)
    floor_col: tuple = (0.030, 0.031, 0.036)
    grid_col: tuple = (0.030, 0.034, 0.042)
    obst_col: tuple = (0.16, 0.155, 0.15)
    glow_coarse: float = 2.0             # weight of the wide (8^3) glow level
    floor: float = 1.0                   # floor plane opacity (0 = no floor)


def camera_basis(cam: Camera, aspect: float) -> np.ndarray:
    """16 floats: pos, forward, right, up, tan(fov/2), aspect, 0, 0."""
    pos = np.asarray(cam.position, np.float64)
    fw = np.asarray(cam.target, np.float64) - pos
    fw /= np.linalg.norm(fw) + 1e-12
    up0 = np.asarray(cam.up, np.float64)
    rt = np.cross(up0, fw)                  # y up, z forward -> x right (left-handed view basis)
    if np.linalg.norm(rt) < 1e-6:
        rt = np.array([1.0, 0.0, 0.0])
    rt /= np.linalg.norm(rt)
    up = np.cross(fw, rt)
    th = math.tan(math.radians(cam.fov_y_deg) * 0.5)
    return np.concatenate([pos, fw, rt, up, [th, aspect, 0.0, 0.0]]).astype(np.float32)


def _gauss1d(sigma: float, device) -> torch.Tensor:
    r = max(1, int(math.ceil(2.5 * sigma)))
    t = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


class VolumeRenderer:
    def __init__(self, out_size: tuple[int, int], device: str = "cuda", march_scale: float = 1.0,
                 params: RenderParams3D | None = None, cuda_graph: bool = True, profile: bool = False,
                 temporal: bool = False):
        wp.init()
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.wp_device = wp.device_from_torch(self.device)
        self.p = params or RenderParams3D()
        self.cuda_graph = cuda_graph and not profile
        self.profile = profile                      # per-stage events (no graph); for the benchmark
        self.timings: dict[str, float] = {}
        self._events = None
        self._stage_events: list = []
        lut = blackbody_lut(self.p.t_lo, self.p.t_hi, 256, t_ref=self.p.t_ref)
        self.lut_t = torch.from_numpy(lut).to(self.device)
        self.lut_wp = wp.from_torch(self.lut_t, dtype=wp.vec3)
        self._cam_dev = torch.zeros(32, device=self.device)        # [current camera | previous camera]
        self._cam_ring = PinnedRing(32)          # per-draw staging; safe for any number of queued draws
        self._vol: dict = {}                # per volume shape: packed array, texture, occupancy
        self._graphs: dict = {}
        self._last_vol_key = None
        self._solid_key = None
        self._solid_res = 1.0
        self._wp_stream = None
        self._wp_stream_key = None
        self._make_light()
        self._dummy_solid()
        self._vel_tex = self._vel_tex0 = self._tex(2, 2, 2, 4, wp.float16)
        self._vel_buf = None
        self._vel_key = None
        self._dummy_detail()
        self._last_detail = None
        # temporal amortization (opt-in, shortlist #2): see temporal.py
        self.temporal = bool(temporal)
        self.t_alpha, self.t_gamma, self.t_kT = 0.1, 1.0, 4.0       # base current weight, clip width, T-change gain
        self.t_alpha_tick = 0.5          # current weight on draws with a new volume (the gas moved: less history)
        self._tcam_dev = torch.zeros(16, device=self.device)        # previous draw's camera + history-valid flag
        self._tcam_ring = PinnedRing(16)
        self._tprev = None
        self._tframe = 0
        self.set_output(out_size, march_scale)

    # ------------------------------------------------------------------ configuration
    def set_output(self, out_size, march_scale: float | None = None) -> None:
        self.out_h, self.out_w = int(out_size[0]), int(out_size[1])
        if march_scale is not None:
            self.march_scale = float(march_scale)
        self.mh = max(8, int(round(self.out_h * self.march_scale)))
        self.mw = max(8, int(round(self.out_w * self.march_scale)))
        self.out = torch.empty(self.out_h, self.out_w, 3, dtype=torch.uint8, device=self.device)
        self.hdr = torch.zeros(self.mh, self.mw, 4, device=self.device)
        bh, bw = max(2, self.mh // 4), max(2, self.mw // 4)
        self.bloom_buf = torch.zeros(bh, bw, 3, device=self.device)
        self._aux_hdr = None                      # (mh, mw, 4) premultiplied aux at march res (lazy)
        self._aux_out = None                      # (4, out_h, out_w) fp16 at output res (lazy)
        self._hist = self._tres = None            # temporal history / resolve target at march res (lazy)
        self._tprev = None                        # history invalid after a resize
        self._graphs.clear()

    def set_temporal(self, on: bool) -> None:
        """Toggle temporal amortization; turning it on starts from an empty history."""
        self.temporal = bool(on)
        self._tprev = None

    def reset_history(self) -> None:
        self._tprev = None

    def set_march_scale(self, s: float) -> None:
        self.set_output((self.out_h, self.out_w), s)

    def set_params(self, **kw) -> None:
        light_changed = "light_res" in kw and kw["light_res"] != self.p.light_res
        for k, v in kw.items():
            setattr(self.p, k, v)
        if light_changed:
            self._make_light()
        self._graphs.clear()

    def _shade(self) -> K.Shade3D:
        p, s = self.p, K.Shade3D()
        for n in ("k_abs", "k_shadow", "k_solid", "k_amb", "k_emit", "emit_min", "emit_scale", "soot",
                  "fire_light", "phase_g", "exposure", "bloom", "bloom_thresh", "t_lo", "t_hi", "d_eps", "floor",
                  "ms_a", "ms_b", "ms_c", "rim", "contrast"):
            setattr(s, n, float(getattr(p, n)))
        L = np.asarray(p.light_dir, np.float64)
        L /= np.linalg.norm(L)
        s.light_dir = wp.vec3(*L.tolist())
        for n in ("key_col", "amb_col", "smoke_albedo", "bg_top", "bg_bot", "floor_col", "grid_col", "obst_col"):
            setattr(s, n, wp.vec3(*getattr(p, n)))
        return s

    def _tex(self, d, h, w, ch, dtype):
        return wp.Texture3D(width=w, height=h, depth=d, num_channels=ch, dtype=dtype,
                            filter_mode=wp.TextureFilterMode.LINEAR, address_mode=wp.TextureAddressMode.CLAMP,
                            device=self.wp_device)

    def _make_light(self) -> None:
        L = int(self.p.light_res)
        self.light_arr = wp.zeros((L, L, L), dtype=wp.vec2h, device=self.wp_device)
        self.glow_src_t = torch.zeros(L, L, L, 4, dtype=torch.float16, device=self.device)
        self.glow_src = wp.from_torch(self.glow_src_t, dtype=wp.vec4h)
        self.light_tex = self._tex(L, L, L, 2, wp.float16)
        G = 32
        self.glow_t = torch.zeros(G, G, G, 4, dtype=torch.float16, device=self.device)
        self.glow_arr = wp.from_torch(self.glow_t, dtype=wp.vec4h)
        self.glow_tex = self._tex(G, G, G, 4, wp.float16)
        self._g3 = _gauss1d(1.2, self.device)
        self._g2 = _gauss1d(2.0, self.device)
        self._graphs.clear()

    def _dummy_solid(self) -> None:
        self.solid_tex0 = self._tex(2, 2, 2, 1, wp.uint8)
        self.occ_s0 = wp.zeros((K.BRICKS,) * 3, dtype=wp.uint8, device=self.wp_device)
        self.solid_tex, self.occ_s = self.solid_tex0, self.occ_s0

    def _dummy_detail(self) -> None:
        """Placeholders for the detail arguments of the shared kernel signatures (detail=None draws
        compile to the original code via wp.static; these are never sampled)."""
        self._d_tex0 = self._tex(2, 2, 2, 4, wp.float32)
        self._d_noise0 = self._tex(2, 2, 2, 4, wp.float16)
        self._d_shade0 = K.DetailShade()
        self._d_dyn0 = torch.zeros(8, device=self.device)

    def _detail_args(self, f: dict | None, light: bool = False) -> list:
        if f is None:
            return [self._d_tex0, self._d_tex0, self._d_noise0, self._d_shade0, wp.from_torch(self._d_dyn0)]
        return [f["coords"][0], f["coords"][1], f["noise"], f["shade_light" if light else "shade"], wp.from_torch(f["dyn"])]

    def _stream(self):
        ts = torch.cuda.current_stream(self.device)
        if self._wp_stream_key != ts.cuda_stream:
            self._wp_stream = wp.stream_from_torch(ts)
            self._wp_stream_key = ts.cuda_stream
        return self._wp_stream

    # ------------------------------------------------------------------ per-shape resources
    def _vol_res(self, shape):
        r = self._vol.get(shape)
        if r is None:
            D, H, W = shape
            packed_t = torch.zeros(D, H, W, 2, dtype=torch.float16, device=self.device)
            occ_t = torch.zeros(K.BRICKS, K.BRICKS, K.BRICKS, dtype=torch.uint8, device=self.device)
            occd_t = torch.zeros_like(occ_t)
            r = self._vol[shape] = {
                "packed_t": packed_t, "packed": wp.from_torch(packed_t, dtype=wp.vec2h),
                "occ_t": occ_t, "occ": wp.from_torch(occ_t),
                "occd_t": occd_t, "occd": wp.from_torch(occd_t),      # dilated by 1 brick (detail draws)
                "tex": self._tex(D, H, W, 2, wp.float16),
            }
        return r

    def _upload_solid(self, solid: torch.Tensor) -> None:
        key = (solid.data_ptr(), solid._version, tuple(solid.shape))
        if key == self._solid_key:
            return
        # smooth the binary mask so the 0.5 iso-surface (and its gradient normals) is not stair-stepped
        s = self._blur3(solid.float()[None, None], _gauss1d(1.0, self.device))[0, 0].contiguous()
        D, H, W = s.shape
        if self.solid_tex is self.solid_tex0 or (self.solid_tex.depth, self.solid_tex.height, self.solid_tex.width) != (D, H, W):
            self.solid_tex = self._tex(D, H, W, 1, wp.uint8)
            self._solid_u8 = wp.zeros((D, H, W), dtype=wp.uint8, device=self.wp_device)
            self._occ_s_t = torch.zeros(K.BRICKS, K.BRICKS, K.BRICKS, dtype=torch.uint8, device=self.device)
            self.occ_s = wp.from_torch(self._occ_s_t)
            self._graphs.clear()
        self._occ_s_t.zero_()
        wp.launch(K.pack_solid, dim=(D, H, W), inputs=[wp.from_torch(s), self._solid_u8, self.occ_s])
        self.solid_tex.copy_from(self._solid_u8)
        self._solid_res = float(max(D, H, W))
        self._solid_key = key

    def _upload_velocity(self, velocity: torch.Tensor) -> None:
        """(3,D,H,W) m/s (x,y,z) -> 4-channel fp16 texture (hardware trilinear, cell-centred)."""
        _, D, H, W = velocity.shape
        if self._vel_buf is None or tuple(self._vel_buf.shape[:3]) != (D, H, W):
            self._vel_buf = torch.zeros(D, H, W, 4, dtype=torch.float16, device=self.device)
            self._vel_wp = wp.from_torch(self._vel_buf, dtype=wp.vec4h)
            self._vel_tex = self._tex(D, H, W, 4, wp.float16)
            self._graphs.clear()
        self._vel_buf[..., :3].copy_(velocity.permute(1, 2, 3, 0))
        self._vel_tex.copy_from(self._vel_wp)

    def _ensure_hist(self) -> None:
        if self._hist is None:
            self._hist = torch.zeros(self.mh, self.mw, 4, device=self.device)
            self._tres = torch.zeros(self.mh, self.mw, 4, device=self.device)
            self._tprev = None
            self._graphs.clear()

    def _ensure_aux(self) -> None:
        if self._aux_out is None:
            self._aux_hdr = torch.zeros(self.mh, self.mw, 4, device=self.device)
            self._aux_out = torch.zeros(4, self.out_h, self.out_w, dtype=torch.float16, device=self.device)
            self.aux = {"alpha": self._aux_out[0], "depth": self._aux_out[1], "motion": self._aux_out[2:4]}

    # ------------------------------------------------------------------ main entry
    @torch.no_grad()
    def draw(self, volume: Volume, camera: Camera, solid: torch.Tensor | None = None,
             reuse_volume: bool = False, *, return_aux: bool = False, velocity: torch.Tensor | None = None,
             prev_camera: Camera | None = None, detail=None):
        """Render to (out_h, out_w, 3) uint8 cuda, row 0 = top. The result is an internal buffer
        reused by the next draw(). Timings ('prep', 'view', 'total' ms) resolve lazily.

        reuse_volume=True skips the volume-only work (pack, texture upload, light + glow volumes)
        and re-renders the volume this renderer prepared last with the new camera. Callers that
        simulate at 30 Hz and render at 60 Hz pass it on frames without a new volume. Only valid if
        the previous draw() of THIS renderer was given the same volume/solid and nothing changed.

        return_aux=True -> (frame, aux), aux = {"alpha" (h,w), "depth" (h,w), "motion" (2,h,w)} fp16
        cuda at out_size, computed in the same march as the colour (see README "Aux buffers"); also
        internal buffers reused by the next draw. `velocity` (3,D,H,W) m/s at any resolution and
        `prev_camera` (the camera of the previous sim step) feed "motion"; both optional and
        ignored when return_aux=False (which runs the unchanged colour-only march).

        detail: a vfx.detail3d.AdvectedDetail (opt-in). The march and the light volume then sample the
        detailed field (displaced / modulated by advected noise, see vfx/detail3d) instead of the plain
        trilinear volume; no high-res volume is written. detail=None runs the original kernels and is
        bit-identical to before. reuse_volume=True also requires the same detail object/version."""
        has_solid = solid is not None
        dfields = detail.fields() if detail is not None else None
        dkey = (id(detail), dfields["version"]) if detail is not None else None
        if reuse_volume and self._last_vol_key is not None and self._last_vol_key[1] == has_solid \
                and dkey == self._last_detail:
            shape = self._last_vol_key[0]
            res = self._vol[shape]
        else:
            reuse_volume = False
        self._resolve_timings()
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        ev[0].record()
        if self.profile:
            self._stage_events = []
        with wp.ScopedStream(self._stream(), sync_enter=False, sync_exit=False):
            pin = self._cam_ring.acquire()
            asp = self.mw / self.mh
            cb = camera_basis(camera, asp)
            if self.temporal:
                self._tframe += 1
                cb = cb.copy()
                cb[14] = (self._tframe * GOLDEN) % 1.0          # per-frame ray-start jitter offset
                tp = self._tcam_ring.acquire()
                if self._tprev is None:
                    tp.zero_()
                else:
                    tp.copy_(torch.from_numpy(self._tprev))
                    tp[14] = 0.0 if reuse_volume else self.t_alpha_tick
                    tp[15] = 1.0                                   # history valid
                self._tcam_dev.copy_(tp, non_blocking=True)
                self._tcam_ring.record()
                self._tprev = cb.copy()
            pin[:16].copy_(torch.from_numpy(cb))
            pin[16:].copy_(torch.from_numpy(camera_basis(prev_camera, asp) if (return_aux and prev_camera is not None) else cb))
            self._cam_dev.copy_(pin, non_blocking=True)
            self._cam_ring.record()
            if not reuse_volume:
                dens, temp = volume.density, volume.temperature
                if dens.dtype != temp.dtype or dens.dtype not in (torch.float32, torch.float16):
                    dens, temp = dens.float(), temp.float()
                dens, temp = dens.contiguous(), temp.contiguous()
                shape = tuple(dens.shape)
                res = self._vol_res(shape)
                if has_solid:
                    self._upload_solid(solid)
                self._mark(self.profile, "start")
                res["occ_t"].zero_()
                kern = K.pack_volume if dens.dtype == torch.float32 else K.pack_volume_h
                wdt = wp.float32 if dens.dtype == torch.float32 else wp.float16
                wp.launch(kern, dim=shape, inputs=[wp.from_torch(dens, dtype=wdt), wp.from_torch(temp, dtype=wdt),
                                                   res["packed"], res["occ"], float(self.p.d_eps), float(self.p.t_lo)])
                self._mark(self.profile, "pack")
                pkey = ("prep", shape, has_solid, id(self.solid_tex)) + ((("detail",) + dkey,) if dkey else ())
                self._run(pkey, lambda: self._prep(res, has_solid, dfields))
                self._last_vol_key = (shape, has_solid)
                self._last_detail = dkey
            has_vel = 0
            if return_aux:
                self._ensure_aux()
                if velocity is not None:
                    vkey = (velocity.data_ptr(), tuple(velocity.shape))
                    if not reuse_volume or vkey != self._vel_key:     # same contract as reuse_volume
                        self._upload_velocity(velocity)
                        self._vel_key = vkey
                    has_vel = 1
            ev[1].record()
            if self.profile and reuse_volume:
                self._mark(True, "start")
            if self.temporal:
                self._ensure_aux()
                self._ensure_hist()
            key = ("view", shape, has_solid, id(self.solid_tex), bool(return_aux), has_vel,
                   id(self._vel_tex) if return_aux else 0) + ((("detail",) + dkey,) if dkey else ()) \
                + ((("temporal",),) if self.temporal else ())
            self._run(key, lambda: self._view(res, has_solid, return_aux, has_vel, dfields, self.temporal))
        ev[2].record()
        self._events = ev
        if return_aux:
            return self.out, self.aux
        return self.out

    def _run(self, key, fn) -> None:
        if self.profile or not self.cuda_graph:
            fn()
            return
        g = self._graphs.get(key)
        if g is None:
            g = self._graphs[key] = self._capture(fn)
        g.replay()

    def _capture(self, fn):
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            with wp.ScopedStream(self._stream(), sync_enter=False, sync_exit=False):
                fn()                                    # warm-up: module load, allocator
        torch.cuda.current_stream(self.device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with wp.ScopedStream(self._stream(), sync_enter=False, sync_exit=False):
                fn()
        return graph

    def _mark(self, prof, name):
        if prof:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self._stage_events.append((name, e))

    def _blur3(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        C, r = x.shape[1], k.numel() // 2
        for dim in range(3):
            shp = [1, 1, 1]
            shp[dim] = -1
            kk = k.view(1, 1, *shp).expand(C, 1, *shp)
            pad = [0] * 6
            pad[2 * (2 - dim)] = pad[2 * (2 - dim) + 1] = r
            x = F.conv3d(F.pad(x, pad, mode="replicate"), kk, groups=C)
        return x

    def _blur2(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        C, r = x.shape[1], k.numel() // 2
        kx = k.view(1, 1, 1, -1).expand(C, 1, 1, -1)
        ky = k.view(1, 1, -1, 1).expand(C, 1, -1, 1)
        x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kx, groups=C)
        return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), ky, groups=C)

    def _prep(self, res, has_solid: bool, dfields: dict | None = None) -> None:
        """Volume-only work: texture upload, light volume, fire-glow volume."""
        p, prof = self.p, self.profile
        shade = self._shade()
        L = int(p.light_res)
        res["tex"].copy_from(res["packed"])
        if dfields is not None:
            wp.launch(K.dilate_occ, dim=(K.BRICKS,) * 3, inputs=[res["occ"], res["occd"]])
        self._mark(prof, "upload")
        wp.launch(K.light_volume if dfields is None else K.light_volume_detail, dim=(L, L, L),
                  inputs=[res["tex"], self.solid_tex, int(has_solid), self.lut_wp, shade, self.light_arr, self.glow_src]
                  + self._detail_args(dfields, light=True))
        self.light_tex.copy_from(self.light_arr)
        # fire glow: emission pyramid 32^3 (narrow) + 8^3 (wide)
        g = self.glow_src_t[..., :3].float().permute(3, 0, 1, 2)[None]          # (1,3,L,L,L)
        g32 = F.adaptive_avg_pool3d(g, 32)
        g32b = self._blur3(g32, self._g3)
        g8 = self._blur3(F.avg_pool3d(g32, 4), self._g3)
        glow = g32b + p.glow_coarse * F.interpolate(g8, size=(32, 32, 32), mode="trilinear", align_corners=False)
        self.glow_t[..., :3].copy_(glow[0].permute(1, 2, 3, 0))
        self.glow_tex.copy_from(self.glow_arr)
        self._mark(prof, "light")

    def _view(self, res, has_solid: bool, aux: bool = False, has_vel: int = 0, dfields: dict | None = None,
              temporal: bool = False) -> None:
        """Camera-dependent work: march (+ aux), [temporal resolve], bloom, composite."""
        p, prof = self.p, self.profile
        shade = self._shade()
        want_aux = aux
        aux = aux or temporal                               # temporal needs the march's opacity-weighted depth
        aux_hdr = self._aux_hdr if aux else self.hdr        # unused by the colour-only variant
        if dfields is None:
            kern, occ = (K.march_aux if aux else K.march), res["occ"]
        else:
            kern, occ = (K.march_aux_detail if aux else K.march_detail), res["occd"]
        wp.launch(kern, dim=(self.mh, self.mw),
                  inputs=[res["tex"], self.light_tex, self.glow_tex, self.solid_tex, occ, self.occ_s,
                          int(has_solid), float(self._solid_res), self.lut_wp, shade,
                          wp.from_torch(self._cam_dev[:16]), float(p.steps), wp.from_torch(self.hdr, dtype=wp.vec4),
                          self._vel_tex if aux else self._vel_tex0, int(has_vel), wp.from_torch(self._cam_dev[16:]),
                          float(SIM_DT), float(self.out_w), float(self.out_h), wp.from_torch(aux_hdr, dtype=wp.vec4)]
                  + self._detail_args(dfields))
        self._mark(prof, "march")
        if temporal:
            wp.launch(TK.temporal_resolve, dim=(self.mh, self.mw),
                      inputs=[wp.from_torch(self.hdr, dtype=wp.vec4), wp.from_torch(self._aux_hdr, dtype=wp.vec4),
                              wp.from_torch(self._hist, dtype=wp.vec4), wp.from_torch(self._cam_dev[:16]),
                              wp.from_torch(self._tcam_dev), wp.from_torch(self._tres, dtype=wp.vec4),
                              float(self.t_alpha), float(self.t_gamma), float(self.t_kT)])
            self._hist.copy_(self._tres)
            self.hdr.copy_(self._tres)
            self._mark(prof, "temporal")
        if want_aux:
            wp.launch(K.aux_resolve, dim=(self.out_h, self.out_w),
                      inputs=[wp.from_torch(self._aux_hdr, dtype=wp.vec4), wp.from_torch(self._aux_out), 1.0e-3])
            self._mark(prof, "aux")
        x = self.hdr[..., :3].permute(2, 0, 1)[None]
        x = F.relu(x - p.bloom_thresh)
        b1 = self._blur2(F.adaptive_avg_pool2d(x, self.bloom_buf.shape[:2]), self._g2)
        b2 = self._blur2(F.avg_pool2d(b1, 4, ceil_mode=True), self._g2)
        b = b1 + 1.5 * F.interpolate(b2, size=b1.shape[-2:], mode="bilinear", align_corners=False)
        self.bloom_buf.copy_(b[0].permute(1, 2, 0))
        wp.launch(K.composite, dim=(self.out_h, self.out_w),
                  inputs=[wp.from_torch(self.hdr, dtype=wp.vec4), wp.from_torch(self.bloom_buf, dtype=wp.vec3),
                          shade, wp.from_torch(self.out)])
        self._mark(prof, "composite")

    def _resolve_timings(self) -> None:
        ev = self._events
        if ev is None or not ev[2].query():
            return
        self.timings = {"prep": ev[0].elapsed_time(ev[1]), "view": ev[1].elapsed_time(ev[2]),
                        "total": ev[0].elapsed_time(ev[2])}
        if self.profile and self._stage_events:
            se = self._stage_events
            for (_, a), (n, b) in zip(se[:-1], se[1:]):
                self.timings[n] = a.elapsed_time(b)

    def last_timings(self, sync: bool = False) -> dict:
        """Timings of the most recent draw; sync=True waits for it (benchmarks only)."""
        if sync and self._events is not None:
            self._events[2].synchronize()
        self._resolve_timings()
        return dict(self.timings)
