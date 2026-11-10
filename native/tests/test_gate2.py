"""Gate 2 tests (native/README.md, "Gate 2 - native detail inference"): D1, D2, D3, D4, T3 + controls.

    <venv python> -m pytest native/tests/test_gate2.py -q -s                       # everything (GPU)
    <venv python> -m pytest native/tests/test_gate2.py -q -s -k "unit"             # CPU-only unit tests (no GPU, no DLL)
    <venv python> -m pytest native/tests/test_gate2.py -q -s -k "HC2"              # Python-vs-Python controls (CUDA, no DLL)

Skips: no CUDA -> every GPU test skips; nvfx.dll absent, or it lacks nvfx_detail_* or the proposed texture helpers
(nvfx_detail_ctypes.py docstring) -> every native test skips with the reason. Errors (not skips): the ONNX model was
exported from a different checkpoint than the one the Python reference loads (re-run native/infer/export_onnx.py).
Env: NVFX_DLL, NVFX_DETAIL_MODEL (default %LOCALAPPDATA%/neural-vfx/native_ref/onnx/screennet_fp16.onnx),
NVFX_G2_CHECKPOINT (default %LOCALAPPDATA%/neural-vfx/checkpoints/screen3d/latest.pt), NVFX_DETAIL_FLAGS (default 0 =
shipping), NVFX_G2_NO_CACHE=1, NVFX_DEBUG_LAYER=1.
Results: native/build/test_results/gate2_native.json and .md (merged across partial -k runs).
"""
from __future__ import annotations

import ctypes as C
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pytest

import gate2_checks as G
import nvfx_detail_ctypes as ND

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "native" / "build" / "test_results"
SCENES = ("campfire", "plume")
R2: dict = {}


def _checkpoint() -> Path:
    if os.environ.get("NVFX_G2_CHECKPOINT"):
        return Path(os.environ["NVFX_G2_CHECKPOINT"])
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "neural-vfx" / "checkpoints" / "screen3d" / "latest.pt"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------------ results file
def _clean(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else str(o)
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if hasattr(o, "item"):
        return _clean(o.item())
    return o


def _write_results():
    checks = {k: v for k, v in R2.items() if not k.startswith("_")}
    if not checks:
        return
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / "gate2_native.json"
    old = {}
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except Exception:
            old = {}
    old.update(_clean(checks))
    meta = dict(old.get("_meta", {}))
    meta.update(_clean(R2.get("_meta", {})))
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    old["_meta"] = meta
    path.write_text(json.dumps(old, indent=1))
    lines = [f"Gate 2 results (native detail inference), {meta['finished']}"
             + (f", {meta.get('nvfx_version')}" if meta.get("nvfx_version") else "")
             + (f", flags {meta.get('flags')}" if "flags" in meta else ""), "",
             f"Model `{meta.get('model')}`, checkpoint `{meta.get('checkpoint')}` (sha256 {str(meta.get('checkpoint_sha256'))[:12]}).",
             "", "| check | expected | result | measured |", "|---|---|---|---|"]
    for k in sorted(x for x in old if not x.startswith("_")):
        v = old[k]
        exp = v.get("expected", "PASS")
        got = "ok" if v.get("pass") else "**NOT OK**"
        lines.append(f"| {k} | {exp} | {got} | {str(v.get('summary', '')).replace('|', '/')} |")
    lines += ["", "For expected-FAIL controls, `ok` means the check failed as it should (the comparator can fail)."]
    t3 = old.get("T3")
    if t3 and "native" in t3:
        f = lambda d: f"{d['median']:.3f} / {d['p10']:.3f}" if d.get("n") else "-"
        lines += ["", "T3, ms median / p10 (GPU timestamps; PyTorch: ScreenDetail.timings CUDA events, same session)", "",
                  "| bbox | native pre | net | post | pre+net+post | total | n | PyTorch total | GPU before / after |",
                  "|---|---|---|---|---|---|---|---|---|"]
        g0, g1 = t3["gpu_before"], t3["gpu_after"]
        for b in ("one_third", "full_frame"):
            n, p = t3["native"][b], t3["python"][b]
            lines.append(f"| {b} | {f(n['pre'])} | {f(n['net'])} | {f(n['post'])} | {f(n['sum'])} | {f(n['total'])} | "
                         f"{n['total'].get('n')} | {f(p['total'])} | {g0.get('util_pct')} % {g0.get('mem_used_mib')} MiB / "
                         f"{g1.get('util_pct')} % {g1.get('mem_used_mib')} MiB |")
    (RESULT_DIR / "gate2_native.md").write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def record():
    def _rec(check_id: str, result: dict):
        R2[check_id] = result
        exp = result.get("expected", "PASS")
        print(f"\n[gate2] {'ok' if result.get('pass') else 'NOT OK'} (expected {exp}) {result.get('summary', '')}")
    yield _rec
    _write_results()


def _run(record, check_id, result):
    record(check_id, result)
    assert result["pass"], result["summary"]


# ------------------------------------------------------------------ GPU fixtures
@pytest.fixture(scope="session")
def cuda():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for the PyTorch reference")
    import warp as wp
    wp.config.quiet = True
    return torch


@pytest.fixture(scope="session")
def native_api():
    """Checks the DLL / exports / model / checkpoint pairing BEFORE any frame is rendered."""
    ok, msg = ND.detail_api_status()
    if not ok:
        pytest.skip(msg)
    mp = ND.model_path()
    if not mp.is_file():
        pytest.skip(f"ONNX model not found at {mp} (run native/infer/export_onnx.py with native/.venv-export)")
    ck = _checkpoint()
    meta_p = mp.parent / "screennet.json"
    if not meta_p.is_file():
        pytest.fail(f"{meta_p} missing: cannot tell which checkpoint {mp.name} was exported from")
    meta = json.loads(meta_p.read_text())
    sha = _sha256(ck)
    if meta.get("checkpoint_sha256") != sha:
        pytest.fail(f"PRECONDITION: {mp.name} was exported from checkpoint sha256 {meta.get('checkpoint_sha256')}, but "
                    f"{ck} is {sha}. Re-export (native/infer/export_onnx.py) or set NVFX_G2_CHECKPOINT to the exported one.")
    lib = ND._lib()
    R2.setdefault("_meta", {}).update(nvfx_version=lib.nvfx_version_string().decode(), dll=msg, model=str(mp),
                                      flags=ND.detail_flags(), export_meta_run=meta.get("run"))
    return mp


@pytest.fixture(scope="session")
def frames(cuda):
    from detail_frames import SCENES as S, make_inputs
    out = {s: make_inputs(s) for s in S}
    R2.setdefault("_meta", {})["inputs"] = {s: v.meta for s, v in out.items()}
    return out


@pytest.fixture(scope="session")
def pyref(cuda):
    ck = _checkpoint()
    if not ck.is_file():
        pytest.skip(f"checkpoint not found: {ck}")
    R2.setdefault("_meta", {}).update(checkpoint=str(ck), checkpoint_sha256=_sha256(ck))
    return G.PyRef(str(ck))


@pytest.fixture(scope="session")
def native(native_api, frames, pyref):
    nd = ND.NativeDetail(native_api, G_W(frames), G_H(frames), process_scale=pyref.sd.scale, flags=ND.detail_flags())
    for s, inp in frames.items():
        nd.upload_inputs(s, inp)
        nd.upload_inputs(f"{s}_alpha0", inp, zero_alpha=True)
    nd.out_a, nd.out_b = nd.make_output(), nd.make_output()
    R2["_meta"].update(create_s=nd.create_s, vram_bytes=nd.vram_bytes(), gpu_at_start=G.gpu_snapshot())
    yield nd
    nd.close()


def G_W(frames):
    return next(iter(frames.values())).hw[1]


def G_H(frames):
    return next(iter(frames.values())).hw[0]


# ------------------------------------------------------------------ unit tests (no GPU, no DLL)
def test_unit_struct_layouts():
    for st, size in ND.EXPECTED_SIZES.items():
        assert C.sizeof(st) == size, (st.__name__, C.sizeof(st), size)
    assert ND.NvfxDetailTimings.frame_index.offset == 48
    assert ND.NvfxDetailFrame.bbox.offset == 40
    assert ND.NvfxDetailDesc.process_scale.offset == 16


def _synthetic(H=96, W=160, seed=0):
    rng = np.random.default_rng(seed)
    inp = rng.integers(0, 256, (H, W, 4), dtype=np.uint8)
    reg = dict(x0=40, y0=16, w=64, h=48, crop_w=64, crop_h=64)
    py = inp[..., :3].copy()
    m = G.region_mask(H, W, reg)
    py[m] = rng.integers(0, 256, (int(m.sum()), 3), dtype=np.uint8)
    nat = inp.copy()
    nat[..., :3][m] = py[m]
    return inp, py, nat, reg, m


def test_unit_comparator():
    inp, py, nat, reg, m = _synthetic()
    c = G.compare_d1(nat, py, inp, reg, [reg, reg])
    assert c["pass_"] and c["max_abs"] == 0 and c["a_passthrough"] and c["py_outside_equals_input"]
    n_in = int(m.sum())                                      # 3072 px
    # +-2 on every region pixel: max abs 2, but every pixel differs by > 1 -> fails on the fraction
    n2 = nat.copy()
    n2[..., 0][m] = np.clip(n2[..., 0][m].astype(int) + 2, 0, 255).astype(np.uint8)
    c = G.compare_d1(n2, py, inp, reg, [reg])
    assert not c["pass_"] and c["max_abs"] <= 2 and not c["pixel_ok"]
    # 1 LSB everywhere passes
    n1 = nat.copy()
    n1[..., 1][m] = np.where(n1[..., 1][m] < 255, n1[..., 1][m] + 1, 254).astype(np.uint8)
    assert G.compare_d1(n1, py, inp, reg, [reg])["pass_"]
    # one pixel off by 3 -> max abs fails
    n3 = nat.copy()
    ys, xs = np.nonzero(m)
    y, x = ys[5], xs[5]
    n3[y, x, 2] = (int(py[y, x, 2]) + 3) % 256 if py[y, x, 2] < 253 else int(py[y, x, 2]) - 3
    c = G.compare_d1(n3, py, inp, reg, [reg])
    assert not c["pass_"] and c["max_abs"] == 3
    # 0.01 % boundary: on a 1e6-px region, 100 px at 2 passes, 101 fails
    H, W = 1000, 1000
    big = np.zeros((H, W, 4), np.uint8)
    rb = dict(x0=0, y0=0, w=W, h=H, crop_w=W, crop_h=H)
    nb = big.copy()
    nb.reshape(-1, 4)[:100, 0] = 2
    assert G.compare_d1(nb, big[..., :3], big, rb, [rb])["pass_"]
    nb.reshape(-1, 4)[100, 0] = 2
    assert not G.compare_d1(nb, big[..., :3], big, rb, [rb])["pass_"]
    # region mismatch, outside pixel changed, A changed
    r2 = dict(reg, x0=reg["x0"] + 64)
    assert not G.compare_d1(nat, py, inp, reg, [reg, r2])["region_ok"]
    no = nat.copy()
    no[0, 0, 0] ^= 1
    c = G.compare_d1(no, py, inp, reg, [reg])
    assert not c["outside_ok"] and not c["pass_"]
    na = nat.copy()
    na[5, 5, 3] ^= 1
    c = G.compare_d1(na, py, inp, reg, [reg])
    assert c["pass_"] and not c["a_passthrough"]


def test_unit_bbox_plans():
    b = G.d3_bboxes(2560, 1440)
    assert b == G.d3_bboxes(2560, 1440) and len(b) == G.D3_N
    assert all(0 <= x0 < x1 <= 2560 and 0 <= y0 < y1 <= 1440 for x0, y0, x1, y1 in b)
    assert len(G.d3_sample_indices()) == G.D3_SAMPLE and G.d3_sample_indices()[0] == 0
    assert G.centred(100, 100, 1536, 832, 2560, 1440) == (0, 0, 1536, 832)
    assert G.centred(1280, 720, 1536, 832, 2560, 1440) == (512, 304, 2048, 1136)


# ------------------------------------------------------------------ harness controls (CUDA, no DLL)
@pytest.fixture(scope="module")
def pyref2(pyref):
    return G.PyRef(pyref.checkpoint)


@pytest.mark.parametrize("kind", ("self", "alpha0", "shift64"))
def test_HC2_python_controls(record, frames, pyref, pyref2, kind):
    _run(record, f"HC2_py_{kind}_campfire", G.check_py_control(pyref, pyref2, frames["campfire"], kind))


# ------------------------------------------------------------------ gate checks (native)
@pytest.mark.parametrize("scene", SCENES)
def test_D1_parity(record, native, pyref, frames, scene):
    _run(record, f"D1_{scene}", G.check_d1(native, pyref, frames[scene], scene, native.out_a))


def test_D1_region_sweep(record, native, pyref):
    _run(record, "D1_region_sweep", G.check_d1_region_sweep(native, pyref, native.H, native.W))


@pytest.mark.parametrize("kind", ("alpha0", "shift64"))
def test_D1_negative_control(record, native, pyref, frames, kind):
    _run(record, f"D1_NC_{kind}_campfire", G.check_d1_negative(native, pyref, frames["campfire"], kind, native.out_a))


@pytest.mark.parametrize("scene", SCENES)
def test_D2_bit_identical(record, native, frames, scene):
    _run(record, f"D2_{scene}", G.check_d2(native, frames[scene], native.out_a, native.out_b))


def test_D3_random_bboxes(record, native_api, pyref, frames):
    H, W = G_H(frames), G_W(frames)
    make = lambda: ND.NativeDetail(native_api, W, H, process_scale=pyref.sd.scale, flags=ND.detail_flags())
    _run(record, "D3", G.check_d3(make, pyref, frames))


def test_D4_no_host_sync(record, native, frames):
    _run(record, "D4", G.check_d4(native, frames["campfire"]))


def test_T3_timing(record, native, pyref, frames):
    _run(record, "T3", G.check_t3(native, pyref, frames["campfire"]))
