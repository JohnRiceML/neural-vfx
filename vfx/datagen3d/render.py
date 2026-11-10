"""Cameras, crops and the paired LR / HR renders for one scenario.

Runtime footprint being matched: the sandbox renders 2560x1440 with march_scale 0.5 (one ray per
2x2 output pixels) and ScreenDetail processes the effect's screen box at process_scale 0.5 (2x2
average pool of the 1440p frame). So we render the FULL 2560x1440 frame with the sandbox's renderer
settings, cut a fixed CROP_OUT x CROP_OUT window around the effect (same window for every frame of a
camera sequence, so motion vectors stay in one pixel frame), and 2x2-average-pool it to CROP x CROP:
one stored pixel = one runtime processed pixel = one march ray, at the exact runtime perspective.

Cameras: the sandbox orbit (vfx/app3d/main.py Orbit: yaw, pitch, dist, fov 40) around a target near
the effect's centroid. Distance log-uniform 0.9-4.0 m (close-up: a 256 px crop spans ~0.23 m;
far: ~1 m, the whole box). A share of sequences orbit / dolly slowly (camera motion vectors).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from vfx.core.fields3d import CH3, Camera, Volume, rasterize_obstacles_3d

OUT_SIZE = (1440, 2560)        # (h, w) sandbox 1440p
MARCH_SCALE = 0.5              # sandbox default
PROCESS_SCALE = 0.5            # ScreenDetail default: stored pixel = 2x2 output pixels
CROP = 256                     # stored crop side (processed px)
CROP_OUT = int(round(CROP / PROCESS_SCALE))   # 512 output px
SOLID_RENDER_RES = 128         # sandbox: obstacles rasterized at 128^3 for display in every mode
FOV = 40.0

CAM_RANGES = {
    "dist": (0.9, 4.0),                 # log-uniform, m (sandbox zoom range 0.9-4.5)
    "pitch": (-0.1, 1.1),               # rad (sandbox -0.15..1.35)
    "target_jitter": 0.05,              # m, gaussian, around the effect centroid
    "crop_jitter": 0.2,                 # crop centre offset, fraction of CROP_OUT
    "moving_prob": 0.4,
    "yaw_rate": (0.15, 0.7),            # rad/s, random sign (sandbox auto-orbit 0.35)
    "pitch_rate": (-0.15, 0.15),        # rad/s
    "dolly_rate": (-0.25, 0.25),        # d(log dist)/s
}


@dataclass
class OrbitCam:
    target: tuple
    yaw: float
    pitch: float
    dist: float
    yaw_rate: float = 0.0
    pitch_rate: float = 0.0
    dolly_rate: float = 0.0
    fov: float = FOV

    @property
    def moving(self) -> bool:
        return bool(self.yaw_rate or self.pitch_rate or self.dolly_rate)

    def at(self, t: float) -> Camera:
        """Camera at time t seconds relative to frame 0 (t may be negative)."""
        yaw = self.yaw + self.yaw_rate * t
        pitch = float(np.clip(self.pitch + self.pitch_rate * t, -0.15, 1.35))
        dist = self.dist * math.exp(self.dolly_rate * t)
        cp = math.cos(pitch)
        off = np.array([cp * math.sin(yaw), math.sin(pitch), -cp * math.cos(yaw)]) * dist
        tg = np.asarray(self.target)
        return Camera(position=tuple((tg + off).tolist()), target=tuple(tg.tolist()), fov_y_deg=self.fov)


def sample_cameras(rng: np.random.Generator, centroid, n: int) -> list[OrbitCam]:
    R = CAM_RANGES
    cams = []
    yaw0 = rng.uniform(0, 2 * math.pi)
    for k in range(n):
        tg = np.asarray(centroid) + rng.normal(0, R["target_jitter"], 3)
        tg = np.clip(tg, [0.1, 0.1, 0.1], [0.9, 0.85, 0.9])
        c = OrbitCam(target=tuple(float(v) for v in tg),
                     yaw=float(yaw0 + 2 * math.pi * k / n + rng.uniform(-0.6, 0.6)),   # spread around the effect
                     pitch=float(rng.uniform(*R["pitch"])),
                     dist=float(math.exp(rng.uniform(math.log(R["dist"][0]), math.log(R["dist"][1])))))
        if rng.random() < R["moving_prob"]:
            c.yaw_rate = float(rng.choice([-1, 1]) * rng.uniform(*R["yaw_rate"]))
            c.pitch_rate = float(rng.uniform(*R["pitch_rate"]))
            c.dolly_rate = float(rng.uniform(*R["dolly_rate"]))
        cams.append(c)
    return cams


def project(cam: Camera, p, out_size=OUT_SIZE) -> tuple[float, float, float]:
    """World point -> (px, py, view_z) in output pixels (row 0 = top), same basis as the renderer."""
    from vfx.render3d import camera_basis
    h, w = out_size
    b = camera_basis(cam, w / h)
    o, fw, rt, up, th = b[0:3], b[3:6], b[6:9], b[9:12], b[12]
    d = np.asarray(p, np.float64) - o
    z = float(d @ fw)
    sx = float(d @ rt) / max(z, 1e-6) / (th * w / h)
    sy = float(d @ up) / max(z, 1e-6) / th
    return (sx + 1) * 0.5 * w, (1 - sy) * 0.5 * h, z


def domain_bbox(cam: Camera, out_size=OUT_SIZE):
    """Projected bbox of the 1 m domain cube (what the sandbox passes ScreenDetail), output px."""
    h, w = out_size
    pts = [project(cam, (x, y, z), out_size) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    if any(p[2] <= 1e-3 for p in pts):
        return [0, 0, w, h]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [int(np.clip(min(xs), 0, w)), int(np.clip(min(ys), 0, h)),
            int(np.ceil(np.clip(max(xs), 0, w))), int(np.ceil(np.clip(max(ys), 0, h)))]


def choose_crop(rng, cam: OrbitCam, frames: int, dt: float) -> list[int]:
    """Fixed CROP_OUT window (x0, y0, output px; even) around the mean projected target over the sequence."""
    h, w = OUT_SIZE
    pts = [project(cam.at(i * dt), cam.target) for i in range(frames)]
    cx = np.mean([p[0] for p in pts]) + rng.uniform(-1, 1) * CAM_RANGES["crop_jitter"] * CROP_OUT
    cy = np.mean([p[1] for p in pts]) + rng.uniform(-1, 1) * CAM_RANGES["crop_jitter"] * CROP_OUT
    x0 = int(np.clip(cx - CROP_OUT / 2, 0, w - CROP_OUT)) // 2 * 2
    y0 = int(np.clip(cy - CROP_OUT / 2, 0, h - CROP_OUT)) // 2 * 2
    return [x0, y0]


def effect_centroid(state) -> list[float]:
    """Visible-content centroid (density + hot gas) of a FieldState3D, world m (one host sync)."""
    d = state.data
    n = d.shape[-1]
    w = d[CH3["density"]] + ((d[CH3["temperature"]] - 700.0) / 700.0).clamp(0, 1)
    c = (torch.arange(n, device=d.device, dtype=torch.float32) + 0.5) / n
    tot = w.sum() + 1e-6
    z = (w.sum((1, 2)) * c).sum() / tot
    y = (w.sum((0, 2)) * c).sum() / tot
    x = (w.sum((0, 1)) * c).sum() / tot
    v = torch.stack([x, y, z, tot]).cpu().tolist()
    if v[3] < 1e-3:
        return [0.5, 0.35, 0.5]
    return v[:3]


def renderer_settings(r) -> dict:
    return {"out_size": list(OUT_SIZE), "march_scale": MARCH_SCALE, "process_scale": PROCESS_SCALE,
            "crop": CROP, "crop_out": CROP_OUT, "solid_render_res": SOLID_RENDER_RES,
            "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(r.p).items()}}


class PairRenderer:
    """Renders LR and HR states for K cameras and accumulates cropped, pooled frames on the GPU."""

    def __init__(self):
        from vfx.render3d import VolumeRenderer
        self.r = VolumeRenderer(OUT_SIZE, march_scale=MARCH_SCALE)

    def begin(self, obstacles, cams: list[OrbitCam], crops: list, frames: int):
        K, s = len(cams), CROP
        dev = "cuda"
        self.solid = rasterize_obstacles_3d(obstacles, SOLID_RENDER_RES) if obstacles else None
        self.cams, self.crops = cams, crops
        self.lr = torch.empty(K, frames, 3, s, s, dtype=torch.uint8, device=dev)
        self.hr = torch.empty_like(self.lr)
        self.aux = torch.empty(K, frames, 4, s, s, dtype=torch.float16, device=dev)
        self.hr_alpha = torch.empty(K, frames, s, s, dtype=torch.uint8, device=dev)

    def _crop(self, x, k):
        x0, y0 = self.crops[k]
        return x[..., y0:y0 + CROP_OUT, x0:x0 + CROP_OUT]

    @staticmethod
    def _pool_rgb(img):
        x = img.permute(2, 0, 1).float()[None]
        return F.avg_pool2d(x, 2)[0].round_().clamp_(0, 255).to(torch.uint8)

    @staticmethod
    def _pool_aux(a):
        """(4,H,W) alpha, depth, mx, my at output res -> pooled; depth/motion alpha-weighted, motion / 2."""
        a = a.float()
        alpha = a[0:1]
        wsum = F.avg_pool2d(alpha[None], 2)[0]
        rest = F.avg_pool2d((a[1:] * alpha)[None], 2)[0] / wsum.clamp_min(1e-4)
        rest = torch.where(wsum > 1e-4, rest, torch.zeros_like(rest))
        rest[1:] *= PROCESS_SCALE
        return torch.cat([wsum, rest]).half()

    @torch.no_grad()
    def frame(self, i: int, lr_state, hr_state, dt: float):
        lr_vol = Volume(lr_state.data[CH3["density"]], lr_state.data[CH3["temperature"]])
        vel = lr_state.velocity
        for k, c in enumerate(self.cams):
            img, aux = self.r.draw(lr_vol, c.at(i * dt), self.solid, reuse_volume=k > 0, return_aux=True,
                                   velocity=vel, prev_camera=c.at((i - 1) * dt))
            y0, x0 = self.crops[k][1], self.crops[k][0]
            self.lr[k, i] = self._pool_rgb(img[y0:y0 + CROP_OUT, x0:x0 + CROP_OUT])
            a =torch.cat([aux["alpha"][None], aux["depth"][None], aux["motion"]])
            self.aux[k, i] = self._pool_aux(self._crop(a, k))
        if hr_state is not None:
            hr_vol = Volume(hr_state.data[CH3["density"]], hr_state.data[CH3["temperature"]])
            for k, c in enumerate(self.cams):
                img, aux = self.r.draw(hr_vol, c.at(i * dt), self.solid, reuse_volume=k > 0, return_aux=True)
                y0, x0 = self.crops[k][1], self.crops[k][0]
                self.hr[k, i] = self._pool_rgb(img[y0:y0 + CROP_OUT, x0:x0 + CROP_OUT])
                al = F.avg_pool2d(self._crop(aux["alpha"], k).float()[None, None], 2)[0, 0]
                self.hr_alpha[k, i] = (al * 255).round().clamp(0, 255).to(torch.uint8)

    def full_frame(self, state, cam: Camera, velocity=None, prev=None):
        """Uncropped (1440,2560,3) uint8 frame (cpu numpy) + aux, for previews."""
        vol = Volume(state.data[CH3["density"]], state.data[CH3["temperature"]])
        img, aux = self.r.draw(vol, cam, self.solid, return_aux=True, velocity=velocity, prev_camera=prev)
        return img.cpu().numpy(), {k: v.float().cpu().numpy() for k, v in aux.items()}
