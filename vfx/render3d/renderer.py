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

from vfx.core.fields3d import Camera, Volume
from vfx.render.blackbody import blackbody_lut
from vfx.render3d import kernels as K


@dataclass
class RenderParams3D:
    steps: float = 192.0                 # ray-march samples per meter (cube side = 1 m)
    light_res: int = 64                  # light / shadow volume resolution
    k_abs: float = 26.0
    k_shadow: float = 40.0
    k_solid: float = 60.0
    k_amb: float = 14.0
    k_emit: float = 3.0
    emit_min: float = 0.35
    emit_scale: float = 28.0
    soot: float = 0.9
    fire_light: float = 0.9
    phase_g: float = 0.55
    exposure: float = 1.0
    bloom: float = 0.7
    bloom_thresh: float = 0.8
    t_lo: float = 700.0
    t_hi: float = 3000.0
    t_ref: float = 1400.0
    d_eps: float = 2e-3
    light_dir: tuple = (-0.55, 0.78, -0.30)      # toward the light (upper-left, camera side)
    key_col: tuple = (1.50, 1.40, 1.25)
    amb_col: tuple = (0.10, 0.13, 0.20)
    smoke_albedo: tuple = (0.72, 0.72, 0.76)
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
                 params: RenderParams3D | None = None, cuda_graph: bool = True, profile: bool = False):
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
        self._cam_dev = torch.zeros(16, device=self.device)
        self._cam_pin = [torch.zeros(16).pin_memory() for _ in range(4)]
        self._cam_i = 0
        self._vol: dict = {}                # per volume shape: packed array, texture, occupancy
        self._graphs: dict = {}
        self._last_vol_key = None
        self._solid_key = None
        self._solid_res = 1.0
        self._wp_stream = None
        self._wp_stream_key = None
        self._make_light()
        self._dummy_solid()
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
        self._graphs.clear()

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
                  "fire_light", "phase_g", "exposure", "bloom", "bloom_thresh", "t_lo", "t_hi", "d_eps", "floor"):
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
            r = self._vol[shape] = {
                "packed_t": packed_t, "packed": wp.from_torch(packed_t, dtype=wp.vec2h),
                "occ_t": occ_t, "occ": wp.from_torch(occ_t),
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

    # ------------------------------------------------------------------ main entry
    @torch.no_grad()
    def draw(self, volume: Volume, camera: Camera, solid: torch.Tensor | None = None,
             reuse_volume: bool = False) -> torch.Tensor:
        """Render to (out_h, out_w, 3) uint8 cuda, row 0 = top. The result is an internal buffer
        reused by the next draw(). Timings ('prep', 'view', 'total' ms) resolve lazily.

        reuse_volume=True skips the volume-only work (pack, texture upload, light + glow volumes)
        and re-renders the volume this renderer prepared last with the new camera. Callers that
        simulate at 30 Hz and render at 60 Hz pass it on frames without a new volume. Only valid if
        the previous draw() of THIS renderer was given the same volume/solid and nothing changed."""
        has_solid = solid is not None
        if reuse_volume and self._last_vol_key is not None and self._last_vol_key[1] == has_solid:
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
            pin = self._cam_pin[self._cam_i]
            self._cam_i = (self._cam_i + 1) % len(self._cam_pin)
            pin.copy_(torch.from_numpy(camera_basis(camera, self.mw / self.mh)))
            self._cam_dev.copy_(pin, non_blocking=True)
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
                self._run(("prep", shape, has_solid, id(self.solid_tex)), lambda: self._prep(res, has_solid))
                self._last_vol_key = (shape, has_solid)
            ev[1].record()
            if self.profile and reuse_volume:
                self._mark(True, "start")
            self._run(("view", shape, has_solid, id(self.solid_tex)), lambda: self._view(res, has_solid))
        ev[2].record()
        self._events = ev
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

    def _prep(self, res, has_solid: bool) -> None:
        """Volume-only work: texture upload, light volume, fire-glow volume."""
        p, prof = self.p, self.profile
        shade = self._shade()
        L = int(p.light_res)
        res["tex"].copy_from(res["packed"])
        self._mark(prof, "upload")
        wp.launch(K.light_volume, dim=(L, L, L),
                  inputs=[res["tex"], self.solid_tex, int(has_solid), self.lut_wp, shade, self.light_arr, self.glow_src])
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

    def _view(self, res, has_solid: bool) -> None:
        """Camera-dependent work: march, bloom, composite."""
        p, prof = self.p, self.profile
        shade = self._shade()
        wp.launch(K.march, dim=(self.mh, self.mw),
                  inputs=[res["tex"], self.light_tex, self.glow_tex, self.solid_tex, res["occ"], self.occ_s,
                          int(has_solid), float(self._solid_res), self.lut_wp, shade,
                          wp.from_torch(self._cam_dev), float(p.steps), wp.from_torch(self.hdr, dtype=wp.vec4)])
        self._mark(prof, "march")
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
