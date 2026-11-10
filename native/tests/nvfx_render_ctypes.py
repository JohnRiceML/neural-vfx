"""ctypes binding of native/render/nvfx_render.h (the proposed native ray-marcher API) + a NativeRenderer wrapper.

OWNER: native test engineer. Written from nvfx_render.h and the "API" part of native/render/README.md only
(render.cpp, the shaders, selfcheck.cpp and compare.py were not read).

DLL: the renderer lives in the render build. Lookup order: env NVFX_DLL, else native/build-render/bin/nvfx.dll if it
exists, else native/build/bin/nvfx.dll. `nvfx_ctypes.load_library()` loads ONE DLL per process, so this module sets
NVFX_DLL for the process before the first load; run test_gate3a.py in its own pytest invocation.

    NativeRenderer(solver: NativeSolver, width, height, march_scale)
    .draw(camera, prev_camera=None, flags=AUX)       Camera-like objects (position, target, up, fov_y_deg)
    .color() -> (H,W,4) uint8   .alpha() / .depth() -> (H,W) f32   .motion() -> (2,H,W) f32   (download: syncs)
    .timings() -> dict | None   .march_size() -> (w, h)   .get_params() -> dict   .close()
"""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import numpy as np

import nvfx_ctypes as NC

ROOT = Path(__file__).resolve().parents[2]
RENDER_DLL = ROOT / "native" / "build-render" / "bin" / "nvfx.dll"

f3 = C.c_float * 3

NVFX_RENDER_AUX = 1
NVFX_RENDER_REUSE_VOLUME = 2
NVFX_RENDER_NO_SOLID = 4
NVFX_RENDER_NO_VELOCITY = 8
TEX_COLOR, TEX_ALPHA, TEX_DEPTH, TEX_MOTION = 0, 1, 2, 3


class NvfxCamera(C.Structure):
    _fields_ = [("position", f3), ("target", f3), ("up", f3), ("fov_y_deg", C.c_float)]


# field for field = vfx.render3d.RenderParams3D (floor -> floor_opacity); order checked against renderer.py by a unit test
PARAM_SCALARS_1 = ["k_abs", "k_shadow", "k_solid", "k_amb", "k_emit", "emit_min", "emit_scale", "soot", "fire_light",
                   "phase_g", "rim", "ms_a", "ms_b", "ms_c", "exposure", "contrast", "bloom", "bloom_thresh", "t_lo",
                   "t_hi", "t_ref", "d_eps"]
PARAM_VEC3 = ["light_dir", "key_col", "amb_col", "smoke_albedo", "bg_top", "bg_bot", "floor_col", "grid_col", "obst_col"]


class NvfxRenderParams(C.Structure):
    _fields_ = ([("steps", C.c_float), ("light_res", C.c_int32)] + [(n, C.c_float) for n in PARAM_SCALARS_1]
                + [(n, f3) for n in PARAM_VEC3] + [("glow_coarse", C.c_float), ("floor_opacity", C.c_float)])


class NvfxRenderTimings(C.Structure):
    _fields_ = [("prep_ms", C.c_float), ("march_ms", C.c_float), ("post_ms", C.c_float), ("total_ms", C.c_float),
                ("flags", C.c_int32), ("reserved", C.c_int32), ("draw_index", C.c_uint64)]


EXPECTED_SIZES = {NvfxCamera: 40, NvfxRenderParams: 212, NvfxRenderTimings: 32}

_p = C.c_void_p
_PROTOS = {
    "nvfx_camera_default": (None, [C.POINTER(NvfxCamera)]),
    "nvfx_render_params_default": (None, [C.POINTER(NvfxRenderParams)]),
    "nvfx_renderer_create": (C.c_int, [_p, C.c_int32, C.c_int32, C.c_float, C.POINTER(_p)]),
    "nvfx_renderer_destroy": (None, [_p]),
    "nvfx_renderer_set_params": (C.c_int, [_p, C.POINTER(NvfxRenderParams)]),
    "nvfx_renderer_get_params": (C.c_int, [_p, C.POINTER(NvfxRenderParams)]),
    "nvfx_renderer_march_size": (None, [_p, C.POINTER(C.c_int32), C.POINTER(C.c_int32)]),
    "nvfx_renderer_draw": (C.c_int, [_p, _p, C.POINTER(NvfxCamera), C.POINTER(NvfxCamera), C.c_int32]),
    "nvfx_renderer_record_draw": (C.c_int, [_p, _p, _p, C.POINTER(NvfxCamera), C.POINTER(NvfxCamera), C.c_int32]),
    "nvfx_renderer_texture": (C.c_void_p, [_p, C.c_int]),
    "nvfx_renderer_get_timings": (C.c_int, [_p, C.POINTER(NvfxRenderTimings)]),
    "nvfx_renderer_download": (C.c_int, [_p, C.c_int, C.c_void_p]),
    "nvfx_render_screen_bbox": (None, [C.POINTER(NvfxCamera), C.c_int32, C.c_int32, C.POINTER(C.c_int32)]),
}


def render_dll_path() -> Path:
    if os.environ.get("NVFX_DLL"):
        return Path(os.environ["NVFX_DLL"])
    return RENDER_DLL if RENDER_DLL.is_file() else NC.DEFAULT_DLL


_RLIB = None


def render_api_status() -> tuple[bool, str]:
    """(ok, message). Loads the DLL (no device is created by loading) and declares the render prototypes."""
    global _RLIB
    p = render_dll_path()
    if not p.is_file():
        return False, (f"native renderer: nvfx.dll not found at {p} (build native/render into native/build-render, "
                       f"or set NVFX_DLL)")
    if NC._LIB is not None and Path(os.environ.get("NVFX_DLL") or NC.DEFAULT_DLL) != p:
        return False, "native renderer: another nvfx.dll is already loaded in this process; run test_gate3a.py alone"
    os.environ["NVFX_DLL"] = str(p)
    lib = NC.load_library()
    missing = []
    for name, (res, args) in _PROTOS.items():
        try:
            fn = getattr(lib, name)
        except AttributeError:
            missing.append(name)
            continue
        fn.restype, fn.argtypes = res, args
    if missing:
        return False, f"native renderer: {p} does not export {missing}"
    _RLIB = lib
    return True, str(p)


def lib():
    if _RLIB is None:
        ok, msg = render_api_status()
        if not ok:
            raise NC.NvfxError(msg)
    return _RLIB


# ------------------------------------------------------------------ conversions
def camera_to_c(cam) -> NvfxCamera:
    return NvfxCamera(f3(*map(float, cam.position)), f3(*map(float, cam.target)), f3(*map(float, cam.up)),
                      float(cam.fov_y_deg))


def camera_from_c(c: NvfxCamera) -> dict:
    return dict(position=tuple(c.position), target=tuple(c.target), up=tuple(c.up), fov_y_deg=c.fov_y_deg)


def params_from_c(p: NvfxRenderParams) -> dict:
    out = {}
    for name, typ in NvfxRenderParams._fields_:
        v = getattr(p, name)
        out["floor" if name == "floor_opacity" else name] = tuple(v) if typ is f3 else v
    return out


def camera_default() -> dict:
    c = NvfxCamera()
    lib().nvfx_camera_default(C.byref(c))
    return camera_from_c(c)


def params_default() -> dict:
    p = NvfxRenderParams()
    lib().nvfx_render_params_default(C.byref(p))
    return params_from_c(p)


def screen_bbox(cam, w: int, h: int) -> tuple:
    out = (C.c_int32 * 4)()
    c = camera_to_c(cam)
    lib().nvfx_render_screen_bbox(C.byref(c), w, h, out)
    return tuple(out)


# ------------------------------------------------------------------ wrapper
class NativeRenderer:
    def __init__(self, solver, width: int, height: int, march_scale: float):
        self.lib = lib()
        self.solver = solver
        self.W, self.H = int(width), int(height)
        self._r = _p()
        NC.check(self.lib.nvfx_renderer_create(solver._ctx, self.W, self.H, float(march_scale), C.byref(self._r)),
                 "nvfx_renderer_create")
        self.n_draws = 0

    def draw(self, camera, prev_camera=None, flags: int = NVFX_RENDER_AUX):
        c = camera_to_c(camera)
        pc = camera_to_c(prev_camera) if prev_camera is not None else None
        NC.check(self.lib.nvfx_renderer_draw(self._r, self.solver._s, C.byref(c), C.byref(pc) if pc is not None else None,
                                             int(flags)), "nvfx_renderer_draw")
        self.n_draws += 1

    def _download(self, which: int, arr: np.ndarray) -> np.ndarray:
        assert arr.flags["C_CONTIGUOUS"]
        NC.check(self.lib.nvfx_renderer_download(self._r, which, arr.ctypes.data_as(C.c_void_p)),
                 f"nvfx_renderer_download({which})")
        return arr

    def color(self) -> np.ndarray:
        return self._download(TEX_COLOR, np.empty((self.H, self.W, 4), np.uint8))

    def alpha(self) -> np.ndarray:
        return self._download(TEX_ALPHA, np.empty((self.H, self.W), np.float32))

    def depth(self) -> np.ndarray:
        return self._download(TEX_DEPTH, np.empty((self.H, self.W), np.float32))

    def motion(self) -> np.ndarray:
        return self._download(TEX_MOTION, np.empty((2, self.H, self.W), np.float32))

    def all_outputs(self) -> dict:
        return dict(color=self.color(), alpha=self.alpha(), depth=self.depth(), motion=self.motion())

    def timings(self) -> dict | None:
        t = NvfxRenderTimings()
        NC.check(self.lib.nvfx_renderer_get_timings(self._r, C.byref(t)), "nvfx_renderer_get_timings")
        if t.draw_index == 0:
            return None
        return dict(prep=t.prep_ms, march=t.march_ms, post=t.post_ms, total=t.total_ms, flags=int(t.flags),
                    draw_index=int(t.draw_index))

    def march_size(self) -> tuple[int, int]:
        w, h = C.c_int32(), C.c_int32()
        self.lib.nvfx_renderer_march_size(self._r, C.byref(w), C.byref(h))
        return w.value, h.value

    def get_params(self) -> dict:
        p = NvfxRenderParams()
        NC.check(self.lib.nvfx_renderer_get_params(self._r, C.byref(p)), "nvfx_renderer_get_params")
        return params_from_c(p)

    def close(self):
        if getattr(self, "_r", None):
            self.lib.nvfx_renderer_destroy(self._r)
            self._r = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
