"""Native ray-marcher vs vfx.render3d.VolumeRenderer on the same solver state (owner's debugging tool).

OWNER: native render engineer. Not the gate test (native/tests/ is written independently from the public API).

Steps the NATIVE solver (vfx/sim3d/selfcheck.py scenes, default config) for --steps steps, downloads its state, renders
that exact state with the Python VolumeRenderer((h, w), march_scale) (return_aux=True, velocity, prev_camera, solid) and
with the native renderer (reading the solver's textures), and diffs colour / alpha / depth / motion with the Gate 3a
measures (native/README.md R1-R4). Writes the Python frame, the native frame and x8 difference images to --out.

    <venv python> native/render/compare.py --out D:/somewhere/render_cmp [--scene campfire|plume|all] [--steps 150]
        [--cams default,orbit60,closeup] [--size 2560x1440] [--scale 0.5] [--dither-modes] [--dll PATH]

--dither-modes renders the native frame with NVFX_RENDER_DITHER_MODE 0/1/2 (which product nvcc fused into an FMA in
the composite's dither hash, see shaders/post.hlsl) and reports which one matches Python best.
Run from the worktree root with the main checkout's venv. Uses the GPU.
"""
from __future__ import annotations

import argparse
import ctypes as C
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

f3 = C.c_float * 3


class NvfxContextDesc(C.Structure):
    _fields_ = [("adapter_index", C.c_int32), ("enable_debug_layer", C.c_int32)]


class NvfxEmitter(C.Structure):
    _fields_ = [("pos", f3), ("radius", C.c_float), ("density_rate", C.c_float), ("temperature", C.c_float),
                ("fuel_rate", C.c_float), ("velocity", f3)]


class NvfxImpulse(C.Structure):
    _fields_ = [("pos", f3), ("radius", C.c_float), ("velocity", f3)]


class NvfxForces(C.Structure):
    _fields_ = [("emitters", C.POINTER(NvfxEmitter)), ("num_emitters", C.c_int32),
                ("impulses", C.POINTER(NvfxImpulse)), ("num_impulses", C.c_int32), ("wind", f3)]


class NvfxObstacle(C.Structure):
    _fields_ = [("kind", C.c_int32), ("pos", f3), ("size", f3)]


class NvfxCamera(C.Structure):
    _fields_ = [("position", f3), ("target", f3), ("up", f3), ("fov_y_deg", C.c_float)]


_PARAM_SCALARS = ["steps", "light_res", "k_abs", "k_shadow", "k_solid", "k_amb", "k_emit", "emit_min", "emit_scale",
                  "soot", "fire_light", "phase_g", "rim", "ms_a", "ms_b", "ms_c", "exposure", "contrast", "bloom",
                  "bloom_thresh", "t_lo", "t_hi", "t_ref", "d_eps"]
_PARAM_VEC3 = ["light_dir", "key_col", "amb_col", "smoke_albedo", "bg_top", "bg_bot", "floor_col", "grid_col", "obst_col"]


class NvfxRenderParams(C.Structure):
    _fields_ = ([(n, C.c_int32 if n == "light_res" else C.c_float) for n in _PARAM_SCALARS]
                + [(n, f3) for n in _PARAM_VEC3] + [("glow_coarse", C.c_float), ("floor_opacity", C.c_float)])


class NvfxRenderTimings(C.Structure):
    _fields_ = [("prep_ms", C.c_float), ("march_ms", C.c_float), ("post_ms", C.c_float), ("total_ms", C.c_float),
                ("flags", C.c_int32), ("reserved", C.c_int32), ("draw_index", C.c_uint64)]


AUX, REUSE_VOLUME, NO_SOLID, NO_VELOCITY = 1, 2, 4, 8
TEX_COLOR, TEX_ALPHA, TEX_DEPTH, TEX_MOTION = 0, 1, 2, 3


class Native:
    def __init__(self, dll_path: Path):
        os.add_dll_directory(str(dll_path.parent))
        self.d = d = C.CDLL(str(dll_path))
        vp, fp = C.c_void_p, C.POINTER(C.c_float)
        protos = {
            "nvfx_last_error": (C.c_char_p, []),
            "nvfx_context_create": (C.c_int, [C.POINTER(NvfxContextDesc), C.POINTER(vp)]),
            "nvfx_context_destroy": (None, [vp]),
            "nvfx_context_wait_idle": (C.c_int, [vp]),
            "nvfx_solver_create": (C.c_int, [vp, C.c_int32, vp, C.POINTER(vp)]),
            "nvfx_solver_destroy": (None, [vp]),
            "nvfx_solver_reset": (C.c_int, [vp, C.POINTER(NvfxObstacle), C.c_int32]),
            "nvfx_solver_step": (C.c_int, [vp, C.POINTER(NvfxForces), C.c_float]),
            "nvfx_solver_download_state": (C.c_int, [vp, fp, fp]),
            "nvfx_render_params_default": (None, [C.POINTER(NvfxRenderParams)]),
            "nvfx_renderer_create": (C.c_int, [vp, C.c_int32, C.c_int32, C.c_float, C.POINTER(vp)]),
            "nvfx_renderer_destroy": (None, [vp]),
            "nvfx_renderer_set_params": (C.c_int, [vp, C.POINTER(NvfxRenderParams)]),
            "nvfx_renderer_draw": (C.c_int, [vp, vp, C.POINTER(NvfxCamera), C.POINTER(NvfxCamera), C.c_int32]),
            "nvfx_renderer_download": (C.c_int, [vp, C.c_int, vp]),
            "nvfx_renderer_get_timings": (C.c_int, [vp, C.POINTER(NvfxRenderTimings)]),
            "nvfx_render_screen_bbox": (None, [C.POINTER(NvfxCamera), C.c_int32, C.c_int32, C.POINTER(C.c_int32)]),
        }
        for name, (res, args) in protos.items():
            f = getattr(d, name)
            f.restype, f.argtypes = res, args
        self.ctx = vp()
        self.ok(d.nvfx_context_create(C.byref(NvfxContextDesc(-1, 0)), C.byref(self.ctx)))

    def ok(self, r):
        if r != 0:
            raise RuntimeError(f"nvfx error {r}: {self.d.nvfx_last_error().decode()}")


def cam_c(cam) -> NvfxCamera:
    return NvfxCamera(f3(*cam.position), f3(*cam.target), f3(*cam.up), cam.fov_y_deg)


def f32(v):
    return tuple(float(np.float32(x)) for x in v)


def make_cams():
    from vfx.core.fields3d import Camera
    d = Camera()

    def orbit(deg):
        a = math.radians(deg)
        ox, oz = d.position[0] - d.target[0], d.position[2] - d.target[2]
        p = (d.target[0] + math.cos(a) * ox + math.sin(a) * oz, d.position[1],
             d.target[2] - math.sin(a) * ox + math.cos(a) * oz)
        return Camera(position=f32(p), target=f32(d.target), up=f32(d.up), fov_y_deg=float(np.float32(d.fov_y_deg)))

    close = Camera(position=(0.5, 0.5, -0.45), target=(0.5, 0.4, 0.5), up=(0.0, 1.0, 0.0), fov_y_deg=40.0)
    close = Camera(position=f32(close.position), target=f32(close.target), up=f32(close.up), fov_y_deg=40.0)
    # (camera, previous camera one sim step ago): same definitions as selfcheck.cpp
    return {"default": (orbit(0.0), orbit(-1.0)), "orbit60": (orbit(60.0), orbit(59.0)), "closeup": (close, close)}


def scene_forces(name, i):
    from vfx.core.fields import SIM_DT
    from vfx.sim3d import selfcheck as SC
    f = SC.campfire(i * SIM_DT) if name == "campfire" else SC.plume(i * SIM_DT)
    em = (NvfxEmitter * max(1, len(f.emitters)))()
    for k, e in enumerate(f.emitters):
        em[k] = NvfxEmitter(f3(*e.pos), e.radius, e.density_rate, e.temperature, e.fuel_rate, f3(*e.velocity))
    return NvfxForces(em, len(f.emitters), None, 0, f3(*f.wind))   # ctypes keeps `em` alive via _objects


def save_rgb(path: Path, img: np.ndarray):
    try:
        from PIL import Image
        Image.fromarray(img).save(path.with_suffix(".png"))
    except ImportError:
        h, w = img.shape[:2]
        with open(path.with_suffix(".ppm"), "wb") as f:
            f.write(b"P6\n%d %d\n255\n" % (w, h))
            f.write(np.ascontiguousarray(img[..., :3]).tobytes())


def stats(name, a, b, mask=None):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    if mask is not None:
        d = d[mask]
    if d.size == 0:
        return f"{name}: (empty mask)"
    return f"{name}: mean {d.mean():.4f}  p99 {np.percentile(d, 99):.4f}  max {d.max():.4f}  (n={d.size})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", default="all", choices=["campfire", "plume", "all"])
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--cams", default="default,orbit60,closeup")
    ap.add_argument("--size", default="2560x1440")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--dither-modes", action="store_true")
    ap.add_argument("--dll", default=os.environ.get("NVFX_DLL", str(ROOT / "native" / "build-render" / "bin" / "nvfx.dll")))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    W, H = (int(x) for x in args.size.split("x"))

    import torch
    from vfx.core.fields import SIM_DT
    from vfx.core.fields3d import Volume
    from vfx.render3d import VolumeRenderer

    nat = Native(Path(args.dll))
    d = nat.d
    pyr = VolumeRenderer((H, W), march_scale=args.scale)
    modes = [0, 1, 2] if args.dither_modes else [None]
    rends = {}
    for m in modes:
        if m is not None:
            os.environ["NVFX_RENDER_DITHER_MODE"] = str(m)   # read by nvfx_renderer_create
        rp = C.c_void_p()
        nat.ok(d.nvfx_renderer_create(nat.ctx, W, H, args.scale, C.byref(rp)))
        rends[m] = rp
    cams = make_cams()
    scenes = ["campfire", "plume"] if args.scene == "all" else [args.scene]
    for sc in scenes:
        s = C.c_void_p()
        nat.ok(d.nvfx_solver_create(nat.ctx, 64, None, C.byref(s)))
        obs = (NvfxObstacle * 1)(NvfxObstacle(0, f3(0.5, 0.55, 0.5), f3(0.12, 0.12, 0.12)))
        nat.ok(d.nvfx_solver_reset(s, obs if sc == "plume" else None, 1 if sc == "plume" else 0))
        for i in range(args.steps):
            nat.ok(d.nvfx_solver_step(s, C.byref(scene_forces(sc, i)), C.c_float(SIM_DT)))
        N = 64
        data = np.zeros((6, N, N, N), np.float32)
        solid = np.zeros((N, N, N), np.float32)
        fp = C.POINTER(C.c_float)
        nat.ok(d.nvfx_solver_download_state(s, data.ctypes.data_as(fp), solid.ctypes.data_as(fp)))
        t = torch.from_numpy(data).cuda()
        st = torch.from_numpy(solid).cuda()
        vol = Volume(density=t[0], temperature=t[4])
        print(f"\n[{sc}] after {args.steps} native steps: density max {data[0].max():.3f}, T max {data[4].max():.0f} K")
        for cname in args.cams.split(","):
            cam, prev = cams[cname]
            frame, aux = pyr.draw(vol, cam, st, return_aux=True, velocity=t[1:4], prev_camera=prev)
            py_rgb = frame.cpu().numpy()
            py_a = aux["alpha"].float().cpu().numpy()
            py_d = aux["depth"].float().cpu().numpy()
            py_m = aux["motion"].float().cpu().numpy()
            tag = f"{sc}_{cname}"
            save_rgb(out / f"{tag}_python", py_rgb)
            for m, rp in rends.items():
                nat.ok(d.nvfx_renderer_draw(rp, s, C.byref(cam_c(cam)), C.byref(cam_c(prev)), AUX))
                col = np.zeros((H, W, 4), np.uint8)
                na = np.zeros((H, W), np.float32)
                nd = np.zeros((H, W), np.float32)
                nm = np.zeros((2, H, W), np.float32)
                for which, arr in ((TEX_COLOR, col), (TEX_ALPHA, na), (TEX_DEPTH, nd), (TEX_MOTION, nm)):
                    nat.ok(d.nvfx_renderer_download(rp, which, arr.ctypes.data_as(C.c_void_p)))
                rgb = col[..., :3]
                mtag = "" if m is None else f" dither_mode {m}"
                mask = py_a > 0.05
                print(f"  {cname}{mtag}")
                print("    R1 " + stats("colour (8-bit)", rgb, py_rgb))
                print("    R2 " + stats("alpha", na, py_a))
                print("    R3 " + stats("depth  [py alpha > 0.05]", nd, py_d, mask))
                print("    R4 " + stats("motion [py alpha > 0.05]", nm, py_m, np.broadcast_to(mask, nm.shape)))
                bb = (C.c_int32 * 4)()
                d.nvfx_render_screen_bbox(C.byref(cam_c(cam)), W, H, bb)
                print(f"    bbox {list(bb)}, py alpha coverage {mask.mean() * 100:.1f} %")
                if m in (None, 0):
                    save_rgb(out / f"{tag}_native", rgb)
                    diff = np.clip(np.abs(rgb.astype(np.int16) - py_rgb.astype(np.int16)) * 8, 0, 255).astype(np.uint8)
                    save_rgb(out / f"{tag}_diff_x8", diff)
                    np.savez_compressed(out / f"{tag}.npz", py_rgb=py_rgb, py_alpha=py_a, py_depth=py_d, py_motion=py_m,
                                        nv_rgb=rgb, nv_alpha=na, nv_depth=nd, nv_motion=nm)
        torch.cuda.synchronize()
        tm = NvfxRenderTimings()
        nat.ok(d.nvfx_context_wait_idle(nat.ctx))
        nat.ok(d.nvfx_renderer_get_timings(rends[modes[0]], C.byref(tm)))
        print(f"  timings (last draw, not a benchmark): native prep {tm.prep_ms:.3f} march {tm.march_ms:.3f} "
              f"post {tm.post_ms:.3f} total {tm.total_ms:.3f} ms; python {pyr.last_timings(sync=True)}")
        d.nvfx_solver_destroy(s)
    for rp in rends.values():
        d.nvfx_renderer_destroy(rp)
    d.nvfx_context_destroy(nat.ctx)


if __name__ == "__main__":
    main()
