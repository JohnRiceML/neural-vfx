"""ctypes binding of native/infer/nvfx_detail.h (Gate 2) + the 2D texture helpers the tests need.

OWNER: native test engineer. Written from the public headers only (nvfx_detail.h, nvfx.h); native/infer internals
are a black box.

CONTRACT GAP. nvfx_detail_run takes ID3D12Resource* textures; neither nvfx.h nor nvfx_detail.h lets a caller
create a Texture2D or move CPU data in / out of one, and a Python test cannot create D3D12 resources by itself.
The tests are therefore written against these PROPOSED helpers (all synchronize; for tests and tools), and skip with
a message naming them until nvfx.dll exports them:

    /* Committed Texture2D, 1 mip, default heap, on the context's device. dxgi_format: a DXGI_FORMAT value (28 RGBA8_UNORM,
       54 R16_FLOAT, 34 R16G16_FLOAT at least); resource_flags: D3D12_RESOURCE_FLAGS (4 = ALLOW_UNORDERED_ACCESS);
       state: the D3D12_RESOURCE_STATES it is created in. *out_resource is an ID3D12Resource* the caller owns (1 ref). */
    NVFX_API NvfxResult nvfx_texture2d_create(NvfxContext* ctx, int32_t width, int32_t height, uint32_t dxgi_format,
                                              uint32_t resource_flags, uint32_t state, void** out_resource);
    /* Releases the caller's reference (ID3D12Resource::Release). NULL is a no-op. */
    NVFX_API void       nvfx_texture2d_release(void* resource);
    /* Copies width x height texels from data (row r at data + r * row_pitch bytes, row 0 = top of the image) into the
       texture. The resource is in `state` on entry and is left in `state` on exit. Synchronizes. */
    NVFX_API NvfxResult nvfx_texture2d_upload(NvfxContext* ctx, void* resource, uint32_t state,
                                              const void* data, uint32_t row_pitch);
    /* Waits for all work submitted to the context queue, then copies the texture into data (same layout as upload).
       The resource is in `state` on entry and is left in `state` on exit. Synchronizes. */
    NVFX_API NvfxResult nvfx_texture2d_download(NvfxContext* ctx, void* resource, uint32_t state,
                                                void* data, uint32_t row_pitch);

NativeDetail wraps one context + one NvfxDetail: upload_inputs(name, DetailInputs, zero_alpha) -> InputSet,
make_output(), fill_output(tex, rgba), run(inset, out, bbox) -> CPU wall ms, download(out) -> (H,W,4) uint8,
region(bbox) -> dict, timings() -> dict | None, wait_idle(), vram_bytes(), close().
bbox None = whole frame (passed to the API as (0,0,0,0), which nvfx_detail.h defines as "whole frame").
"""
from __future__ import annotations

import ctypes as C
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import nvfx_ctypes as N

# ---------------------------------------------------------------- constants
NVFX_DETAIL_BACKEND_ORT = 1
NVFX_DETAIL_DML_HALF_COMPUTE = 2
NVFX_DETAIL_DML_NO_METACMD = 4
NVFX_DETAIL_DML_NHWC = 8
NVFX_DETAIL_ORT_NO_FUSION = 16
NVFX_DETAIL_LAZY_BUCKETS = 32

DXGI_FORMAT_R8G8B8A8_UNORM = 28
DXGI_FORMAT_R16G16_FLOAT = 34
DXGI_FORMAT_R16_FLOAT = 54
D3D12_RESOURCE_FLAG_NONE = 0
D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS = 0x4
D3D12_RESOURCE_STATE_UNORDERED_ACCESS = 0x8
D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE = 0x40


# ---------------------------------------------------------------- structs (nvfx_detail.h, exact field order)
class NvfxDetailDesc(C.Structure):
    _fields_ = [("model_path", C.c_char_p), ("width", C.c_int32), ("height", C.c_int32),
                ("process_scale", C.c_float), ("flags", C.c_int32)]


class NvfxDetailFrame(C.Structure):
    _fields_ = [("color", C.c_void_p), ("alpha", C.c_void_p), ("depth", C.c_void_p), ("motion", C.c_void_p),
                ("output", C.c_void_p), ("bbox", C.c_int32 * 4)]


class NvfxDetailTimings(C.Structure):
    _fields_ = [("pre_ms", C.c_float), ("net_ms", C.c_float), ("post_ms", C.c_float), ("total_ms", C.c_float),
                ("cpu_ms", C.c_float), ("region", C.c_int32 * 4), ("crop", C.c_int32 * 2),
                ("frame_index", C.c_uint64)]


# x64 sizes implied by the header (checked by test_gate2.py without the DLL)
EXPECTED_SIZES = {NvfxDetailDesc: 24, NvfxDetailFrame: 56, NvfxDetailTimings: 56}

_vp = C.c_void_p
_i32 = C.c_int32
_u32 = C.c_uint32
DETAIL_PROTOS = {
    "nvfx_detail_create": (C.c_int, [_vp, C.POINTER(NvfxDetailDesc), C.POINTER(_vp)]),
    "nvfx_detail_destroy": (None, [_vp]),
    "nvfx_detail_run": (C.c_int, [_vp, C.POINTER(NvfxDetailFrame)]),
    "nvfx_detail_record": (C.c_int, [_vp, _vp, C.POINTER(NvfxDetailFrame)]),
    "nvfx_detail_get_timings": (C.c_int, [_vp, C.POINTER(NvfxDetailTimings)]),
    "nvfx_detail_region": (C.c_int, [_vp, C.POINTER(_i32), C.POINTER(_i32)]),
    "nvfx_detail_vram_bytes": (C.c_uint64, [_vp]),
}
HELPER_PROTOS = {   # PROPOSED (see module docstring)
    "nvfx_texture2d_create": (C.c_int, [_vp, _i32, _i32, _u32, _u32, _u32, C.POINTER(_vp)]),
    "nvfx_texture2d_release": (None, [_vp]),
    "nvfx_texture2d_upload": (C.c_int, [_vp, _vp, _u32, _vp, _u32]),
    "nvfx_texture2d_download": (C.c_int, [_vp, _vp, _u32, _vp, _u32]),
}
HELPER_SIGNATURES = """\
NVFX_API NvfxResult nvfx_texture2d_create(NvfxContext* ctx, int32_t width, int32_t height, uint32_t dxgi_format, uint32_t resource_flags, uint32_t state, void** out_resource);
NVFX_API void       nvfx_texture2d_release(void* resource);
NVFX_API NvfxResult nvfx_texture2d_upload(NvfxContext* ctx, void* resource, uint32_t state, const void* data, uint32_t row_pitch);
NVFX_API NvfxResult nvfx_texture2d_download(NvfxContext* ctx, void* resource, uint32_t state, void* data, uint32_t row_pitch);"""

DEFAULT_MODEL = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "neural-vfx" / "native_ref" / "onnx" / "screennet_fp16.onnx"


def model_path() -> Path:
    return Path(os.environ.get("NVFX_DETAIL_MODEL") or DEFAULT_MODEL)


def detail_flags() -> int:
    return int(os.environ.get("NVFX_DETAIL_FLAGS", "0"))


_DLIB = None


def detail_api_status() -> tuple[bool, str]:
    """(ok, message). Loads nvfx.dll (no device is created) and checks every nvfx_detail_* export and the proposed
    texture helpers. Call only from a fixture: it is not run at collection time."""
    global _DLIB
    ok, msg = N.dll_available()
    if not ok:
        return False, f"native backend: {msg}"
    try:
        lib = N.load_library()
    except Exception as e:
        return False, f"native backend: {e!r}"
    miss_d = [n for n in DETAIL_PROTOS if not hasattr(lib, n)]
    miss_h = [n for n in HELPER_PROTOS if not hasattr(lib, n)]
    if miss_d:
        return False, f"{N.dll_path()} does not export the Gate 2 API {miss_d} (nvfx_detail.h)"
    if miss_h:
        return False, ("CONTRACT GAP: Gate 2 needs CPU<->Texture2D helpers that nvfx.h / nvfx_detail.h do not provide; "
                       f"{N.dll_path()} lacks {miss_h}. Proposed signatures (native/tests/nvfx_detail_ctypes.py):\n"
                       + HELPER_SIGNATURES)
    for name, (res, args) in {**DETAIL_PROTOS, **HELPER_PROTOS}.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, args
    _DLIB = lib
    return True, msg


def _lib():
    if _DLIB is None:
        ok, msg = detail_api_status()
        if not ok:
            raise N.NvfxError(msg)
    return _DLIB


def bbox_c(bbox) -> tuple:
    return (0, 0, 0, 0) if bbox is None else tuple(int(v) for v in bbox)


@dataclass
class InputSet:
    name: str
    color: int
    alpha: int | None
    depth: int | None
    motion: int | None
    rgba: np.ndarray          # what was uploaded to `color` (for the "outside identical to input" check)


class NativeDetail:
    def __init__(self, model: Path | str, width: int, height: int, process_scale: float = 0.5, flags: int = 0):
        self.lib = _lib()
        self.W, self.H = int(width), int(height)
        self.flags = int(flags)
        desc = N.NvfxContextDesc(-1, 1 if os.environ.get("NVFX_DEBUG_LAYER") == "1" else 0)
        self.ctx = C.c_void_p()
        N.check(self.lib.nvfx_context_create(C.byref(desc), C.byref(self.ctx)), "nvfx_context_create")
        self._model = str(model).encode("utf-8")
        d = NvfxDetailDesc(self._model, self.W, self.H, float(process_scale), self.flags)
        self.d = C.c_void_p()
        t0 = time.perf_counter()
        N.check(self.lib.nvfx_detail_create(self.ctx, C.byref(d), C.byref(self.d)), "nvfx_detail_create")
        self.create_s = time.perf_counter() - t0
        self._textures: list[int] = []
        self.inputs: dict[str, InputSet] = {}
        self.n_runs = 0

    # ------------------------------------------------------------ textures (proposed helpers)
    def _tex(self, fmt: int, flags: int, state: int) -> int:
        r = C.c_void_p()
        N.check(self.lib.nvfx_texture2d_create(self.ctx, self.W, self.H, fmt, flags, state, C.byref(r)),
                "nvfx_texture2d_create")
        self._textures.append(r.value)
        return r.value

    def _upload(self, tex: int, state: int, a: np.ndarray):
        a = np.ascontiguousarray(a)
        assert a.shape[0] == self.H and a.shape[1] == self.W
        pitch = a.strides[0]
        N.check(self.lib.nvfx_texture2d_upload(self.ctx, tex, state, a.ctypes.data, pitch), "nvfx_texture2d_upload")

    def upload_inputs(self, name: str, inp, zero_alpha: bool = False, null_aux: bool = False) -> InputSet:
        """inp: detail_frames.DetailInputs. zero_alpha: upload an all-zero alpha texture (negative control)."""
        S = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
        rgba = inp.rgba()
        color = self._tex(DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, S)
        self._upload(color, S, rgba)
        if null_aux:
            s = InputSet(name, color, None, None, None, rgba)
        else:
            alpha = self._tex(DXGI_FORMAT_R16_FLOAT, D3D12_RESOURCE_FLAG_NONE, S)
            self._upload(alpha, S, np.zeros_like(inp.alpha) if zero_alpha else inp.alpha.astype(np.float16))
            depth = self._tex(DXGI_FORMAT_R16_FLOAT, D3D12_RESOURCE_FLAG_NONE, S)
            self._upload(depth, S, inp.depth.astype(np.float16))
            motion = self._tex(DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_FLAG_NONE, S)
            self._upload(motion, S, inp.motion_rg().astype(np.float16))
            s = InputSet(name, color, alpha, depth, motion, rgba)
        self.inputs[name] = s
        return s

    def make_output(self) -> int:
        return self._tex(DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS)

    def fill_output(self, tex: int, rgba: np.ndarray):
        """Pre-fills an output texture (garbage pattern), so 'fully written' is observable."""
        self._upload(tex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, rgba)

    def download(self, tex: int) -> np.ndarray:
        self.wait_idle()
        out = np.empty((self.H, self.W, 4), np.uint8)
        N.check(self.lib.nvfx_texture2d_download(self.ctx, tex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                 out.ctypes.data, out.strides[0]), "nvfx_texture2d_download")
        return out

    # ------------------------------------------------------------ detail API
    def frame(self, s: InputSet, out: int, bbox) -> NvfxDetailFrame:
        return NvfxDetailFrame(s.color, s.alpha, s.depth, s.motion, out, (C.c_int32 * 4)(*bbox_c(bbox)))

    def run(self, s: InputSet, out: int, bbox) -> float:
        """nvfx_detail_run; returns the CPU wall time of the call in ms."""
        f = self.frame(s, out, bbox)
        t0 = time.perf_counter()
        rc = self.lib.nvfx_detail_run(self.d, C.byref(f))
        t1 = time.perf_counter()
        N.check(rc, "nvfx_detail_run")
        self.n_runs += 1
        return (t1 - t0) * 1e3

    def region(self, bbox) -> dict:
        b = (C.c_int32 * 4)(*bbox_c(bbox))
        o = (C.c_int32 * 6)()
        N.check(self.lib.nvfx_detail_region(self.d, b, o), "nvfx_detail_region")
        return dict(x0=o[0], y0=o[1], w=o[2], h=o[3], crop_w=o[4], crop_h=o[5])

    def timings(self) -> dict | None:
        t = NvfxDetailTimings()
        N.check(self.lib.nvfx_detail_get_timings(self.d, C.byref(t)), "nvfx_detail_get_timings")
        if t.frame_index == 0:
            return None
        return dict(pre=t.pre_ms, net=t.net_ms, post=t.post_ms, total=t.total_ms, sum=t.pre_ms + t.net_ms + t.post_ms,
                    cpu=t.cpu_ms, region=dict(x0=t.region[0], y0=t.region[1], w=t.region[2], h=t.region[3],
                                              crop_w=t.crop[0], crop_h=t.crop[1]),
                    frame_index=int(t.frame_index))

    def vram_bytes(self) -> int:
        return int(self.lib.nvfx_detail_vram_bytes(self.d))

    def wait_idle(self):
        N.check(self.lib.nvfx_context_wait_idle(self.ctx), "nvfx_context_wait_idle")

    def close(self):
        if getattr(self, "d", None):
            self.lib.nvfx_detail_destroy(self.d)
            self.d = None
        for t in getattr(self, "_textures", []):
            self.lib.nvfx_texture2d_release(t)
        self._textures = []
        if getattr(self, "ctx", None):
            self.lib.nvfx_context_destroy(self.ctx)
            self.ctx = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class QueueLoad:
    """Queues GPU work on a NativeDetail's context through the public solver API (for D4): a 64^3 solver with
    pressure_tol 0 and 256 PCG iteration slots, stepping the campfire scene. nvfx_solver_step never waits."""

    def __init__(self, nd: NativeDetail):
        from nvfx_scenes import get_scene
        self.lib = nd.lib
        cfg = N.NvfxSolverConfig()
        self.lib.nvfx_solver_config_default(C.byref(cfg))
        cfg.pressure_tol, cfg.pressure_max_iters, cfg.pressure_warm_start = 0.0, 256, 0
        self.s = C.c_void_p()
        N.check(self.lib.nvfx_solver_create(nd.ctx, 64, C.byref(cfg), C.byref(self.s)), "nvfx_solver_create")
        N.check(self.lib.nvfx_solver_reset(self.s, None, 0), "nvfx_solver_reset")
        self.sc = get_scene("campfire")
        self.k = 0

    def submit(self, n: int):
        for _ in range(n):
            fc = N.ForcesC(self.sc.forces(self.k))
            N.check(self.lib.nvfx_solver_step(self.s, C.byref(fc.s), float(N.NVFX_SIM_DT)), "nvfx_solver_step")
            self.k += 1

    def close(self):
        if getattr(self, "s", None):
            self.lib.nvfx_solver_destroy(self.s)
            self.s = None
