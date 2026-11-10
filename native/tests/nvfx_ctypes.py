"""ctypes binding of native/include/nvfx.h (the whole API) + a NativeSolver convenience wrapper.

OWNER: native test engineer. Written from the header only (no knowledge of native/core or native/sim).

DLL lookup: env var NVFX_DLL, else <worktree>/native/build/bin/nvfx.dll.
Env NVFX_DEBUG_LAYER=1 creates the context with the D3D12 debug layer.

NativeSolver interface (shared with backends.PyFakeSolver, so the gate checks run unchanged on both):
    NativeSolver(res=64, config=SolverConfig3D | None)
    .set_config(cfg)            .reset(obstacles)            .step(forces: Forces3D, dt)
    .project()                  .download() -> (6,N,N,N) f32 .download_solid() -> (N,N,N) f32
    .upload(data, solid=None)   .divergence() -> (N,N,N) f32 .get_timings() -> dict | None
    .wait_idle()                .set_timing(bool) (no-op)    .close()
"""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DLL = ROOT / "native" / "build" / "bin" / "nvfx.dll"

# ---------------------------------------------------------------- constants (nvfx.h)
NVFX_VERSION_MAJOR, NVFX_VERSION_MINOR = 0, 1
NVFX_DOMAIN_SIZE = 1.0
NVFX_AMBIENT_TEMP = 293.0
NVFX_SIM_DT = 1.0 / 30.0
NVFX_NUM_CHANNELS = 6
NVFX_MAX_EMITTERS = 32
NVFX_MAX_IMPULSES = 32
(NVFX_CH_DENSITY, NVFX_CH_VEL_X, NVFX_CH_VEL_Y, NVFX_CH_VEL_Z, NVFX_CH_TEMPERATURE, NVFX_CH_FUEL) = range(6)

NVFX_OK, NVFX_ERR_INVALID_ARG, NVFX_ERR_DEVICE, NVFX_ERR_UNSUPPORTED, NVFX_ERR_OUT_OF_MEMORY, NVFX_ERR_INTERNAL = range(6)
RESULT_NAMES = {0: "NVFX_OK", 1: "NVFX_ERR_INVALID_ARG", 2: "NVFX_ERR_DEVICE", 3: "NVFX_ERR_UNSUPPORTED",
                4: "NVFX_ERR_OUT_OF_MEMORY", 5: "NVFX_ERR_INTERNAL"}
NVFX_OBSTACLE_SPHERE, NVFX_OBSTACLE_BOX = 0, 1
NVFX_TEX_VEL_DENSITY, NVFX_TEX_TEMP_FUEL, NVFX_TEX_SOLID = 0, 1, 2

f3 = C.c_float * 3


# ---------------------------------------------------------------- structs (exact field order / types)
class NvfxContextDesc(C.Structure):
    _fields_ = [("adapter_index", C.c_int32), ("enable_debug_layer", C.c_int32)]


class NvfxSolverConfig(C.Structure):
    _fields_ = [
        ("buoyancy_alpha", C.c_float), ("buoyancy_beta", C.c_float), ("vorticity_eps", C.c_float),
        ("density_dissipation", C.c_float), ("temperature_cooling", C.c_float), ("burn_rate", C.c_float),
        ("ignition_temp", C.c_float), ("burn_heat", C.c_float), ("smoke_from_fuel", C.c_float),
        ("pressure_tol", C.c_float), ("pressure_max_iters", C.c_int32), ("pressure_warm_start", C.c_int32),
        ("max_temperature", C.c_float), ("wind_marker_temp", C.c_float), ("heat_diffusion", C.c_float),
        ("confinement_floor", C.c_float),
    ]


class NvfxEmitter(C.Structure):
    _fields_ = [("pos", f3), ("radius", C.c_float), ("density_rate", C.c_float), ("temperature", C.c_float),
                ("fuel_rate", C.c_float), ("velocity", f3)]


class NvfxImpulse(C.Structure):
    _fields_ = [("pos", f3), ("radius", C.c_float), ("velocity", f3)]


class NvfxForces(C.Structure):
    _fields_ = [("emitters", C.POINTER(NvfxEmitter)), ("num_emitters", C.c_int32),
                ("impulses", C.POINTER(NvfxImpulse)), ("num_impulses", C.c_int32),
                ("wind", f3)]


class NvfxObstacle(C.Structure):
    _fields_ = [("kind", C.c_int32), ("pos", f3), ("size", f3)]


class NvfxSolverTimings(C.Structure):
    _fields_ = [("advection_ms", C.c_float), ("forces_ms", C.c_float), ("projection_ms", C.c_float),
                ("total_ms", C.c_float), ("pcg_iters", C.c_int32), ("step_index", C.c_uint64)]


# expected x64 layouts (checked by test_harness_controls.py)
EXPECTED_SIZES = {NvfxContextDesc: 8, NvfxSolverConfig: 64, NvfxEmitter: 40, NvfxImpulse: 28,
                  NvfxForces: 40, NvfxObstacle: 28, NvfxSolverTimings: 32}

_ctx_p = C.c_void_p     # opaque NvfxContext*
_slv_p = C.c_void_p     # opaque NvfxSolver*
_fp = C.POINTER(C.c_float)

_PROTOS = {
    "nvfx_last_error": (C.c_char_p, []),
    "nvfx_version_string": (C.c_char_p, []),
    "nvfx_context_create": (C.c_int, [C.POINTER(NvfxContextDesc), C.POINTER(_ctx_p)]),
    "nvfx_context_create_from_device": (C.c_int, [C.c_void_p, C.c_void_p, C.POINTER(_ctx_p)]),
    "nvfx_context_destroy": (None, [_ctx_p]),
    "nvfx_context_wait_idle": (C.c_int, [_ctx_p]),
    "nvfx_context_device": (C.c_void_p, [_ctx_p]),
    "nvfx_context_queue": (C.c_void_p, [_ctx_p]),
    "nvfx_solver_config_default": (None, [C.POINTER(NvfxSolverConfig)]),
    "nvfx_solver_config_runtime": (None, [C.POINTER(NvfxSolverConfig)]),
    "nvfx_solver_create": (C.c_int, [_ctx_p, C.c_int32, C.POINTER(NvfxSolverConfig), C.POINTER(_slv_p)]),
    "nvfx_solver_destroy": (None, [_slv_p]),
    "nvfx_solver_res": (C.c_int32, [_slv_p]),
    "nvfx_solver_set_config": (C.c_int, [_slv_p, C.POINTER(NvfxSolverConfig)]),
    "nvfx_solver_reset": (C.c_int, [_slv_p, C.POINTER(NvfxObstacle), C.c_int32]),
    "nvfx_solver_step": (C.c_int, [_slv_p, C.POINTER(NvfxForces), C.c_float]),
    "nvfx_solver_record_step": (C.c_int, [_slv_p, C.c_void_p, C.POINTER(NvfxForces), C.c_float]),
    "nvfx_solver_project": (C.c_int, [_slv_p]),
    "nvfx_solver_get_timings": (C.c_int, [_slv_p, C.POINTER(NvfxSolverTimings)]),
    "nvfx_solver_download_state": (C.c_int, [_slv_p, _fp, _fp]),
    "nvfx_solver_upload_state": (C.c_int, [_slv_p, _fp, _fp]),
    "nvfx_solver_download_divergence": (C.c_int, [_slv_p, _fp]),
    "nvfx_solver_texture": (C.c_void_p, [_slv_p, C.c_int]),
}


class NvfxError(RuntimeError):
    pass


def dll_path() -> Path:
    return Path(os.environ.get("NVFX_DLL") or DEFAULT_DLL)


def dll_available() -> tuple[bool, str]:
    p = dll_path()
    if not p.is_file():
        return False, f"nvfx.dll not found at {p} (build with `powershell -File native/build.ps1` or set NVFX_DLL)"
    return True, str(p)


_LIB = None


def load_library():
    """Loads nvfx.dll once and declares every prototype of nvfx.h (a missing export raises)."""
    global _LIB
    if _LIB is not None:
        return _LIB
    p = dll_path()
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(p.parent))
    lib = C.CDLL(str(p))
    missing = []
    for name, (res, args) in _PROTOS.items():
        try:
            fn = getattr(lib, name)
        except AttributeError:
            missing.append(name)
            continue
        fn.restype, fn.argtypes = res, args
    if missing:
        raise NvfxError(f"{p} does not export {missing}")
    _LIB = lib
    return lib


def check(rc: int, what: str):
    if rc != NVFX_OK:
        msg = _LIB.nvfx_last_error().decode(errors="replace") if _LIB is not None else ""
        raise NvfxError(f"{what} -> {RESULT_NAMES.get(rc, rc)}: {msg}")


# ---------------------------------------------------------------- conversions from vfx dataclasses
CFG_FIELDS = [f for f, _ in NvfxSolverConfig._fields_]


def config_to_c(cfg) -> NvfxSolverConfig:
    """vfx.sim3d.SolverConfig3D -> NvfxSolverConfig (same field names)."""
    c = NvfxSolverConfig()
    for f in CFG_FIELDS:
        v = getattr(cfg, f)
        setattr(c, f, int(v) if f in ("pressure_max_iters", "pressure_warm_start") else float(v))
    return c


def config_from_c(c: NvfxSolverConfig) -> dict:
    return {f: getattr(c, f) for f in CFG_FIELDS}


class ForcesC:
    """Keeps the arrays alive while the NvfxForces struct is in use."""

    def __init__(self, forces):
        em, im = list(forces.emitters), list(forces.impulses)
        if len(em) > NVFX_MAX_EMITTERS or len(im) > NVFX_MAX_IMPULSES:
            raise ValueError("too many emitters / impulses")
        self.em = (NvfxEmitter * max(1, len(em)))()
        for i, e in enumerate(em):
            self.em[i] = NvfxEmitter(f3(*e.pos), e.radius, e.density_rate, e.temperature, e.fuel_rate, f3(*e.velocity))
        self.im = (NvfxImpulse * max(1, len(im)))()
        for i, m in enumerate(im):
            self.im[i] = NvfxImpulse(f3(*m.pos), m.radius, f3(*m.velocity))
        self.s = NvfxForces(C.cast(self.em, C.POINTER(NvfxEmitter)) if em else None, len(em),
                            C.cast(self.im, C.POINTER(NvfxImpulse)) if im else None, len(im), f3(*forces.wind))


def obstacles_to_c(obstacles):
    obs = list(obstacles)
    arr = (NvfxObstacle * max(1, len(obs)))()
    for i, o in enumerate(obs):
        kind = {"sphere": NVFX_OBSTACLE_SPHERE, "box": NVFX_OBSTACLE_BOX}[o.kind]
        arr[i] = NvfxObstacle(kind, f3(*o.pos), f3(*o.size))
    return arr, len(obs)


def _fptr(a: np.ndarray):
    assert a.dtype == np.float32 and a.flags["C_CONTIGUOUS"]
    return a.ctypes.data_as(_fp)


# ---------------------------------------------------------------- wrapper
class NativeSolver:
    name = "native"

    def __init__(self, res: int = 64, config=None):
        self.lib = load_library()
        self.res = int(res)
        desc = NvfxContextDesc(-1, 1 if os.environ.get("NVFX_DEBUG_LAYER") == "1" else 0)
        self._ctx = _ctx_p()
        check(self.lib.nvfx_context_create(C.byref(desc), C.byref(self._ctx)), "nvfx_context_create")
        self._s = _slv_p()
        cc = config_to_c(config) if config is not None else None
        check(self.lib.nvfx_solver_create(self._ctx, self.res, C.byref(cc) if cc is not None else None,
                                          C.byref(self._s)), "nvfx_solver_create")
        assert self.lib.nvfx_solver_res(self._s) == self.res

    @property
    def version(self) -> str:
        return self.lib.nvfx_version_string().decode()

    def set_config(self, cfg):
        c = config_to_c(cfg)
        check(self.lib.nvfx_solver_set_config(self._s, C.byref(c)), "nvfx_solver_set_config")

    def reset(self, obstacles=()):
        arr, n = obstacles_to_c(obstacles)
        check(self.lib.nvfx_solver_reset(self._s, arr if n else None, n), "nvfx_solver_reset")

    def step(self, forces, dt: float):
        fc = ForcesC(forces)
        check(self.lib.nvfx_solver_step(self._s, C.byref(fc.s), float(dt)), "nvfx_solver_step")

    def project(self):
        check(self.lib.nvfx_solver_project(self._s), "nvfx_solver_project")

    def download(self) -> np.ndarray:
        N = self.res
        out = np.empty((NVFX_NUM_CHANNELS, N, N, N), np.float32)
        check(self.lib.nvfx_solver_download_state(self._s, _fptr(out), None), "nvfx_solver_download_state")
        return out

    def download_solid(self) -> np.ndarray:
        N = self.res
        out = np.empty((N, N, N), np.float32)
        check(self.lib.nvfx_solver_download_state(self._s, None, _fptr(out)), "nvfx_solver_download_state(solid)")
        return out

    def upload(self, data: np.ndarray, solid: np.ndarray | None = None):
        N = self.res
        d = np.ascontiguousarray(data, np.float32)
        assert d.shape == (NVFX_NUM_CHANNELS, N, N, N)
        s = None
        if solid is not None:
            s = np.ascontiguousarray(solid, np.float32)
            assert s.shape == (N, N, N)
        check(self.lib.nvfx_solver_upload_state(self._s, _fptr(d), _fptr(s) if s is not None else None),
              "nvfx_solver_upload_state")

    def divergence(self) -> np.ndarray:
        N = self.res
        out = np.empty((N, N, N), np.float32)
        check(self.lib.nvfx_solver_download_divergence(self._s, _fptr(out)), "nvfx_solver_download_divergence")
        return out

    def get_timings(self) -> dict | None:
        t = NvfxSolverTimings()
        check(self.lib.nvfx_solver_get_timings(self._s, C.byref(t)), "nvfx_solver_get_timings")
        if not t.total_ms > 0.0:
            return None
        return dict(advection=t.advection_ms, forces=t.forces_ms, projection=t.projection_ms, total=t.total_ms,
                    pcg_iters=t.pcg_iters, step_index=int(t.step_index))

    def texture(self, which: int):
        return self.lib.nvfx_solver_texture(self._s, which)

    def set_timing(self, on: bool):
        pass   # native timestamps are always on

    def wait_idle(self):
        check(self.lib.nvfx_context_wait_idle(self._ctx), "nvfx_context_wait_idle")

    def close(self):
        if getattr(self, "_s", None):
            self.lib.nvfx_solver_destroy(self._s)
            self._s = None
        if getattr(self, "_ctx", None):
            self.lib.nvfx_context_destroy(self._ctx)
            self._ctx = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def default_config_c() -> dict:
    c = NvfxSolverConfig()
    load_library().nvfx_solver_config_default(C.byref(c))
    return config_from_c(c)


def runtime_config_c() -> dict:
    c = NvfxSolverConfig()
    load_library().nvfx_solver_config_runtime(C.byref(c))
    return config_from_c(c)
