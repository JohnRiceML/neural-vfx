"""Self-check + cost of VolumeRenderer.draw(return_aux=True). Not a gate test (those live in tests3d/).

    python -m vfx.render3d.aux_check [--h 720 --w 1280] [--time]

1. frame identity: the colour frame with return_aux=True is bit-identical to return_aux=False.
2. motion vectors: a Gaussian blob is (a) translated by a known world velocity, (b) viewed from a
   moved camera, (c) both. Frame t-1 is backward-warped by `motion` (sample t-1 at p - motion) and
   compared to frame t: MSE(no warp) / MSE(warp) is reported (higher = better; 1 = useless), over
   the whole frame and over pixels with alpha_t > 0.5. Floor off (it is not part of the medium).
3. --time: added cost at 2560x1440, march_scale 0.5, campfire 64^3 sim with its velocity.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from vfx.core.fields import SIM_DT
from vfx.core.fields3d import AMBIENT_TEMP, Camera, Volume
from vfx.render3d import VolumeRenderer

N = 64


def blob(center, sigma=0.09, n=N):
    c = (torch.arange(n, device="cuda", dtype=torch.float32) + 0.5) / n
    z, y, x = torch.meshgrid(c, c, c, indexing="ij")
    r2 = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    d = torch.exp(-0.5 * r2 / sigma ** 2) * 1.5
    # a little texture so a warp error is visible inside the blob too
    d = d * (1.0 + 0.4 * torch.sin(40 * x + 9) * torch.sin(37 * y + 3) * torch.sin(43 * z + 5))
    return Volume(d.clamp_min(0).contiguous(), torch.full_like(d, AMBIENT_TEMP))


def orbit(angle_deg, dist=1.9, target=(0.5, 0.45, 0.5), pitch=0.2):
    a = np.radians(angle_deg)
    pos = (target[0] + dist * np.cos(pitch) * np.sin(a), target[1] + dist * np.sin(pitch),
           target[2] - dist * np.cos(pitch) * np.cos(a))
    return Camera(position=tuple(pos), target=target, fov_y_deg=40.0)


def truck(cam, off):
    """Translate camera and target together (screen-space pan + parallax)."""
    return Camera(position=tuple(a + b for a, b in zip(cam.position, off)),
                  target=tuple(a + b for a, b in zip(cam.target, off)), fov_y_deg=cam.fov_y_deg)


def warp_back(prev_u8, motion):
    """Sample prev at (p - motion); motion (2,h,w) pixels, x right, y down."""
    h, w = prev_u8.shape[:2]
    ys, xs = torch.meshgrid(torch.arange(h, device="cuda", dtype=torch.float32) + 0.5,
                            torch.arange(w, device="cuda", dtype=torch.float32) + 0.5, indexing="ij")
    sx = xs - motion[0].float()
    sy = ys - motion[1].float()
    grid = torch.stack([sx / w * 2 - 1, sy / h * 2 - 1], -1)[None]
    src = prev_u8.permute(2, 0, 1)[None].float()
    return F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=False)[0].permute(1, 2, 0)


def case(r, name, vol0, cam0, vol1, cam1, vel):
    f0 = r.draw(vol0, cam0).clone().float()
    f1, aux = r.draw(vol1, cam1, return_aux=True, velocity=vel, prev_camera=cam0)
    f1 = f1.clone().float()
    m = aux["motion"].clone()
    a = aux["alpha"].float()
    wpd = warp_back(f0, m)
    e_none = (f1 - f0).pow(2).mean(-1)
    e_warp = (f1 - wpd).pow(2).mean(-1)
    msk = a > 0.5
    r_all = float(e_none.mean() / e_warp.mean().clamp_min(1e-9))
    r_in = float(e_none[msk].mean() / e_warp[msk].mean().clamp_min(1e-9))
    mag = m.float().norm(dim=0)[msk]
    print(f"  {name:28s} |motion| in blob: median {float(mag.median()):6.2f} px  "
          f"MSE none {float(e_none.mean()):7.3f} warp {float(e_warp.mean()):6.3f}  ratio all {r_all:6.1f}  "
          f"in alpha>0.5: {r_in:6.1f}")
    return r_all, r_in


def check_motion(h, w, scale):
    r = VolumeRenderer((h, w), march_scale=scale)
    r.set_params(floor=0.0)
    c0 = (0.45, 0.45, 0.5)
    v = torch.tensor([1.2, 0.6, -0.9], device="cuda")            # m/s -> ~0.05 m per SIM_DT
    c1 = tuple(float(a + b * SIM_DT) for a, b in zip(c0, v.tolist()))
    vel = v.view(3, 1, 1, 1).expand(3, N, N, N).contiguous()
    cam = orbit(0.0)
    print(f"motion-vector check at {w}x{h}, march_scale {scale} (pixels per SIM_DT = {SIM_DT:.4f} s)")
    out = {}
    out["translate"] = case(r, "blob translated, cam fixed", blob(c0), cam, blob(c1), cam, vel)
    cam_prev = orbit(-4.0)                       # orbit about the domain centre ...
    cam_prev = truck(cam_prev, (-0.05, 0.03, 0.0))   # ... plus a pan, so the blob moves on screen
    out["camera"] = case(r, "blob fixed, camera orbit+pan", blob(c0), cam_prev, blob(c0), cam, None)
    out["both"] = case(r, "blob moved + camera orbit+pan", blob(c0), cam_prev, blob(c1), cam, vel)
    # sign sanity: blob moving +x (right on screen) and +y (up) -> motion_x > 0, motion_y < 0
    _, aux = r.draw(blob(c1), cam, return_aux=True, velocity=vel)
    m = aux["motion"].float()
    msk = aux["alpha"].float() > 0.5
    print(f"  sign check (v=+x,+y,-z): mean motion_x {float(m[0][msk].mean()):+.2f}, motion_y {float(m[1][msk].mean()):+.2f}")
    return out


def check_identity(h, w, scale):
    r = VolumeRenderer((h, w), march_scale=scale)
    vol, cam = blob((0.5, 0.45, 0.5)), orbit(10.0)
    a = r.draw(vol, cam).clone()
    b, aux = r.draw(vol, cam, return_aux=True, velocity=torch.ones(3, 16, 16, 16, device="cuda"), prev_camera=orbit(9.0))
    c = r.draw(vol, cam).clone()
    same = bool(torch.equal(a, b)) and bool(torch.equal(a, c))
    print(f"frame identity aux on/off: {'bit-identical' if same else 'DIFFERENT, max ' + str(int((a.int() - b.int()).abs().max()))}")
    al = aux["alpha"]
    print(f"  aux shapes alpha {tuple(al.shape)} {al.dtype}, depth {tuple(aux['depth'].shape)}, motion {tuple(aux['motion'].shape)}; "
          f"alpha range [{float(al.min()):.3f}, {float(al.max()):.3f}], depth range in blob "
          f"[{float(aux['depth'][al > 0.5].min()):.3f}, {float(aux['depth'][al > 0.5].max()):.3f}] m; "
          f"depth==0 where alpha==0: {bool((aux['depth'][al == 0] == 0).all())}")
    return same


def time_cost(trials=25, n=30):
    from vfx.core.fields3d import CH3
    from vfx.render3d.bench import camera_at, gpu_note
    from vfx.app3d.backends import load_backends
    from vfx.app3d.presets import PRESETS
    from vfx.core.fields3d import Forces3D, rasterize_obstacles_3d
    print("GPU:", gpu_note())
    pre = PRESETS[1]
    b = load_backends(warmup=False)
    s = b.Solver(64, b.runtime_config(), device="cuda")
    st = s.reset(pre.obstacles)
    for _ in range(150):
        st = s.step(st, Forces3D(emitters=list(pre.emitters), wind=pre.wind), SIM_DT)
    vol = Volume(st.data[CH3["density"]].clone(), st.data[CH3["temperature"]].clone())
    vel = st.velocity.clone()
    solid = rasterize_obstacles_3d(pre.obstacles, 128)
    cam, cam0 = camera_at(1.9), camera_at(1.88)
    r = VolumeRenderer((1440, 2560), march_scale=0.5)
    modes = {
        "colour only": lambda reuse: r.draw(vol, cam, solid, reuse_volume=reuse),
        "aux (vel + prev cam)": lambda reuse: r.draw(vol, cam, solid, reuse_volume=reuse, return_aux=True,
                                                     velocity=vel, prev_camera=cam0),
    }
    res = {}
    for reuse in (False, True):
        for fn in modes.values():
            for _ in range(5):
                fn(reuse)
        torch.cuda.synchronize()
        ts = {k: [] for k in modes}
        for _ in range(trials):                  # interleaved, so contention hits both modes alike
            for name, fn in modes.items():
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(n):
                    fn(reuse)
                e1.record()
                e1.synchronize()
                ts[name].append(e0.elapsed_time(e1) / n)
        for name in modes:
            res[(name, reuse)] = (float(np.min(ts[name])), float(np.median(ts[name])))
    for reuse in (False, True):
        a, b_ = res[("colour only", reuse)], res[("aux (vel + prev cam)", reuse)]
        print(f"  2560x1440 scale 0.5 {'view-only' if reuse else 'full     '}: colour {a[0]:.3f} ms (med {a[1]:.3f}), "
              f"+aux {b_[0]:.3f} ms (med {b_[1]:.3f})  -> added {b_[0] - a[0]:+.3f} ms (min) / {b_[1] - a[1]:+.3f} (median)")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--time", action="store_true")
    a = ap.parse_args(argv)
    check_identity(a.h, a.w, a.scale)
    check_motion(a.h, a.w, a.scale)
    if a.time:
        time_cost()


if __name__ == "__main__":
    main()
