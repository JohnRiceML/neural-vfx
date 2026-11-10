"""Gate 3a tests (native/README.md, "Gate 3a - native ray-marcher"): R1-R6, T2 + negative and harness controls.

    <venv python> -m pytest native/tests/test_gate3a.py -q -s                 # everything (GPU; run it in its own process)
    <venv python> -m pytest native/tests/test_gate3a.py -q -s -k unit         # CPU-only unit tests (no GPU, no DLL)
    <venv python> -m pytest native/tests/test_gate3a.py -q -s -k "HC3 or PRE_closeup"   # Python-only (CUDA, no DLL)

Reference: vfx.render3d.VolumeRenderer((1440, 2560), march_scale=0.5), default RenderParams3D, draw(volume, camera,
solid=state.solid, return_aux=True, velocity=state.velocity, prev_camera), on the Python Solver3D state after 150 steps
(campfire; plume + sphere). The same state (data + solid) is uploaded to a native solver with nvfx_solver_upload_state,
which the native renderer draws. Cameras: Camera(), Camera() orbited 60 deg about the domain centre, and a close-up
(bench.camera_at, the largest distance reaching >= 1/3 Python alpha > 0.05 coverage). prev_camera = the camera orbited
by -3 deg about the domain centre (moving camera); R4 is also run with prev_camera None (moving volume only).

Skips: no CUDA -> every GPU test; nvfx.dll absent or without the nvfx_renderer_* exports -> every native test.
DLL: env NVFX_DLL, else native/build-render/bin/nvfx.dll, else native/build/bin/nvfx.dll (nvfx_render_ctypes.py).
NVFX_RENDER_DITHER_MODE must be UNSET for the gate (R1 is the default mode's); test_R1_dither_modes runs R1 in four
fresh processes (unset, 0, 1, 2) and reports them side by side.
Results: native/build/test_results/gate3a_native.json/.md (merged across -k runs); images in
native/build/test_results/gate3a_images/ (<scene>_<camera>_{python,native,diff_x8,nc_yaw1_diff_x8}.png, contact_sheet.png).
"""
from __future__ import annotations

import ast
import ctypes as C
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import gate3a_checks as G
import nvfx_render_ctypes as NR

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = G.RESULT_DIR
R3A: dict = {}
CASES = [f"{s}_{c}" for s in G.SCENES for c in G.CAMERAS]
DITHER_MODES = ("unset", "0", "1", "2")


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
    checks = {k: v for k, v in R3A.items() if not k.startswith("_")}
    if not checks:
        return
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / "gate3a_native.json"
    old = {}
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except Exception:
            old = {}
    old.update(_clean(checks))
    meta = dict(old.get("_meta", {}))
    meta.update(_clean(R3A.get("_meta", {})))
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    old["_meta"] = meta
    path.write_text(json.dumps(old, indent=1))
    lines = [f"Gate 3a results (native ray-marcher vs vfx.render3d.VolumeRenderer), {meta['finished']}"
             + (f", {meta.get('nvfx_version')}" if meta.get("nvfx_version") else ""), "",
             f"DLL `{meta.get('dll')}`; NVFX_RENDER_DITHER_MODE in the gate process: `{meta.get('dither_mode_env')}`; "
             f"images `{G.IMAGE_DIR}`.", "",
             "| check | expected | result | measured |", "|---|---|---|---|"]
    for k in sorted(x for x in old if not x.startswith("_")):
        v = old[k]
        exp = v.get("expected", "PASS")
        got = "ok" if v.get("pass") else "**NOT OK**"
        lines.append(f"| {k} | {exp} | {got} | {str(v.get('summary', '')).replace('|', '/')} |")
    lines += ["", "For expected-FAIL controls, `ok` means the check failed as it should (the comparator can fail).",
              "R1 gates only the two numeric thresholds. \"No structured error visible\" is NOT automated: the x8 "
              "difference images are flagged for the lead's visual review; `struct_energy` (share of the error energy that is smooth "
              "at the 8x8 block scale; ~0.016 for uncorrelated dither noise, ~1 for a smooth error; also given vs the "
              "same error with its pixels shuffled) is a reported proxy, not a gate."]
    dm = old.get("R1_dither_modes")
    if dm and dm.get("modes"):
        lines += ["", "R1 per NVFX_RENDER_DITHER_MODE (fresh process each): mean abs / p99, worst case over the six",
                  "(scene, camera) cases; `= in-process default` = native RGB bytes identical to the gate run in every case.", "",
                  "| mode | worst mean abs | worst p99 | R1 pass (all cases) | = in-process default |", "|---|---|---|---|---|"]
        for m, r in dm["modes"].items():
            lines.append(f"| {m} | {r.get('worst_mean', '-')} | {r.get('worst_p99', '-')} | {r.get('all_pass', '-')} | "
                         f"{r.get('same_as_default', '-')} |")
    for k in ("T2_full", "T2_view"):
        t = old.get(k)
        if t and "native" in t:
            f = lambda d: f"{d['median']:.3f} / {d['p10']:.3f} / {d['p95']:.3f}" if d.get("n") else "-"
            n, p, g0, g1 = t["native"], t["python"], t["gpu_before"], t["gpu_after"]
            if k == "T2_full":
                lines += ["", "T2, ms median / p10 / p95 (native: renderer GPU timestamps; Python: VolumeRenderer.timings, "
                          "CUDA events, same session)", "",
                          "| case | native prep | march | post | total | n | Python prep | view | total | GPU before / after |",
                          "|---|---|---|---|---|---|---|---|---|---|"]
            lines.append(f"| {k} | {f(n['prep'])} | {f(n['march'])} | {f(n['post'])} | {f(n['total'])} | "
                         f"{n['total'].get('n')} | {f(p['prep'])} | {f(p['view'])} | {f(p['total'])} | "
                         f"{g0.get('util_pct')} % {g0.get('mem_used_mib')} MiB / {g1.get('util_pct')} % "
                         f"{g1.get('mem_used_mib')} MiB |")
    (RESULT_DIR / "gate3a_native.md").write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def record():
    def _rec(check_id: str, result: dict):
        R3A[check_id] = result
        exp = result.get("expected", "PASS")
        print(f"\n[gate3a] {'ok' if result.get('pass') else 'NOT OK'} (expected {exp}) {result.get('summary', '')}")
    yield _rec
    _write_results()


def _run(record, check_id, result):
    record(check_id, result)
    assert result["pass"], result["summary"]


# ------------------------------------------------------------------ unit tests (no GPU, no DLL, no torch)
def _dataclass_defaults(path: Path, cls: str) -> list[tuple[str, object]]:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return [(s.target.id, ast.literal_eval(s.value)) for s in node.body
                    if isinstance(s, ast.AnnAssign) and s.value is not None]
    raise AssertionError(f"{cls} not found in {path}")


def test_unit_struct_layouts():
    for st, size in NR.EXPECTED_SIZES.items():
        assert C.sizeof(st) == size, (st.__name__, C.sizeof(st), size)
    assert NR.NvfxRenderTimings.draw_index.offset == 24
    assert NR.NvfxRenderParams.light_res.offset == 4
    assert NR.NvfxRenderParams.light_dir.offset == 4 * 24


def test_unit_params_fields_match_renderparams3d():
    py = [n for n, _ in _dataclass_defaults(ROOT / "vfx" / "render3d" / "renderer.py", "RenderParams3D")]
    nat = ["floor" if n == "floor_opacity" else n for n, _ in NR.NvfxRenderParams._fields_]
    assert nat == py


def test_unit_cameras():
    defaults = dict(_dataclass_defaults(ROOT / "vfx" / "core" / "fields3d.py", "Camera"))
    c = G.Cam()
    assert (c.position, c.target, c.up, c.fov_y_deg) == (defaults["position"], defaults["target"], defaults["up"],
                                                          defaults["fov_y_deg"])
    # camera_at == vfx/render3d/bench.py camera_at (executed from its source with a stub Camera)
    src = (ROOT / "vfx" / "render3d" / "bench.py").read_text()
    tree = ast.parse(src)
    keep = [n for n in tree.body if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "TARGET" for t in n.targets))
            or (isinstance(n, ast.FunctionDef) and n.name == "camera_at")]
    ns = {"np": np, "Camera": lambda **kw: kw}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "bench_camera_at", "exec"), ns)
    for d in G.CLOSEUP_DISTS:
        ref, mine = ns["camera_at"](d), G.camera_at(d)
        assert np.allclose(ref["position"], mine.position, atol=1e-12) and tuple(ref["target"]) == mine.target
        assert ref["fov_y_deg"] == mine.fov_y_deg
    # orbit about the domain centre: distance to the vertical axis, height and target kept; 60 deg turned
    o = G.orbit(c, 60.0)
    rel = lambda p: np.array([p[0] - 0.5, p[2] - 0.5])
    assert abs(np.linalg.norm(rel(o.position)) - np.linalg.norm(rel(c.position))) < 1e-12
    assert o.position[1] == c.position[1] and np.allclose(o.target, c.target)
    cosang = rel(o.position) @ rel(c.position) / np.linalg.norm(rel(c.position)) ** 2
    assert abs(math.degrees(math.acos(cosang)) - 60.0) < 1e-9
    assert abs(G.angle_between_views(c, G.yaw(c, 1.0)) - 1.0) < 1e-6
    assert abs(G.angle_between_views(c, G.prev_of(c)) - G.PREV_ORBIT_DEG) < 0.5
    assert G.yaw(c, 1.0).position == c.position


def test_unit_r1_metrics():
    rng = np.random.default_rng(0)
    H, W = 360, 640
    py = rng.integers(20, 236, (H, W, 3)).astype(np.uint8)
    assert G.r1_metrics(py, py)["bit_identical"]
    # dither-like noise: 1/3 of the values off by +-1 -> passes, proxy ~ shuffled baseline
    noise = rng.choice(np.array([-1, 0, 0, 0, 0, 1], np.int16), size=py.shape)
    r = G.r1_metrics((py.astype(np.int16) + noise).astype(np.uint8), py)
    assert r["pass"] and abs(r["mean_abs"] - 1 / 3) < 0.01 and r["p99"] == 1
    assert r["structure"]["struct_energy"] < 0.03 and 0.7 < r["structure"]["struct_ratio"] < 1.3
    assert r["structure"]["blocks_gt1"] == 0
    # structured error with a small mean: +4 in one rectangle -> proxy far above the baseline
    s = py.astype(np.int16).copy()
    s[100:180, 200:360] += 4
    r = G.r1_metrics(s.astype(np.uint8), py)
    assert r["structure"]["struct_energy"] > 0.9 and r["structure"]["struct_ratio"] > 3
    assert r["structure"]["blocks_gt1"] > 0.05
    # p99 boundary (inverted CDF): exactly 1 % of values at 9 -> p99 0; one more -> 9 -> FAIL
    z = np.zeros((100, 100, 3), np.uint8)
    n = z.copy()
    n.reshape(-1)[:300] = 9
    assert G.r1_metrics(n, z)["p99"] == 0
    n.reshape(-1)[300] = 9
    r = G.r1_metrics(n, z)
    assert r["p99"] == 9 and not r["pass"]
    # mean boundary: 1.5 passes, above fails
    n = np.full((10, 10, 3), 0, np.uint8)
    n[:5] = 3                                    # half the values 3 -> mean 1.5, p99 3
    assert G.r1_metrics(n, z[:10, :10])["pass"]
    n[5, 0, 0] = 1
    assert not G.r1_metrics(n, z[:10, :10])["pass"]


def test_unit_aux_metrics():
    rng = np.random.default_rng(1)
    H, W = 64, 96
    alpha = np.zeros((H, W), np.float32)
    alpha[10:50, 20:80] = 0.6
    r = G.r2_metrics(alpha + 0.009, alpha)
    assert r["pass_"]
    assert not G.r2_metrics(alpha + 0.011, alpha)["pass_"]
    depth = np.where(alpha > 0, 1.7, 0).astype(np.float32)
    assert G.r3_metrics(depth + 0.009, depth, alpha)["pass_"]
    assert not G.r3_metrics(depth + 0.011, depth, alpha)["pass_"]
    assert G.r3_metrics(depth + 5.0 * (alpha == 0), depth, alpha)["pass_"]          # outside the mask: ignored
    mot = np.stack([rng.normal(30, 5, (H, W)), rng.normal(-12, 3, (H, W))]).astype(np.float32)
    assert G.r4_metrics(mot + np.float32(0.3), mot, alpha)["pass_"]                # EPE 0.42
    assert not G.r4_metrics(mot + np.float32(0.4), mot, alpha)["pass_"]            # EPE 0.57 (components 0.4)
    assert not G.r4_metrics(G.motion_swapped(mot), mot, alpha)["pass_"]
    assert not G.r4_metrics(G.motion_negated(mot), mot, alpha)["pass_"]
    assert G.r4_metrics(mot, mot, np.zeros_like(alpha))["n_px"] == 0


def test_unit_png_and_sheet(tmp_path):
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, (37, 53, 3)).astype(np.uint8)
    G.write_png(tmp_path / "a.png", img)
    assert np.array_equal(G.read_png(tmp_path / "a.png"), img)
    d = G.diff_image(img, np.zeros_like(img))
    assert d.max() == 255 and np.array_equal(d[img < 32], img[img < 32] * 8)
    t = G.thumb(rng.integers(0, 256, (40, 80, 3)).astype(np.uint8), 4)
    assert t.shape == (10, 20, 3)
    sheet = G.contact_sheet([[t, t, t], [t, t, t]])
    assert sheet.shape == (2 * 14 + 4, 3 * 24 + 4, 3)


# ------------------------------------------------------------------ GPU fixtures
@pytest.fixture(scope="session")
def cuda():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for the Python reference")
    import warp as wp
    wp.config.quiet = True
    return torch


@pytest.fixture(scope="session")
def py_side(cuda):
    """Python scenes after 150 steps, the reference renderer, the cameras (close-up search) per scene."""
    pr = G.PyRender()
    scenes = {}
    for s in G.SCENES:
        ps = G.PyScene(s)
        ps.state_np = ps.numpy_state()
        cams, info = G.cameras_for(pr, ps)
        scenes[s] = dict(ps=ps, cams=cams, info=info)
    R3A.setdefault("_meta", {}).update(py_march_size=list(pr.march_size()), steps=G.STEPS,
                                       cameras={s: {c: G.cam_dict(v) for c, v in d["cams"].items()} for s, d in scenes.items()},
                                       closeup_search={s: d["info"]["closeup_search"] for s, d in scenes.items()},
                                       prev_orbit_deg=G.PREV_ORBIT_DEG, orbit_deg=G.ORBIT_DEG)
    return pr, scenes


@pytest.fixture(scope="session")
def native_api():
    ok, msg = NR.render_api_status()
    if not ok:
        pytest.skip(msg)
    R3A.setdefault("_meta", {}).update(dll=msg, nvfx_version=NR.lib().nvfx_version_string().decode(),
                                       dither_mode_env=G.env_dither_mode() or "unset")
    return msg


@pytest.fixture(scope="session")
def native(native_api, py_side):
    from nvfx_ctypes import NativeSolver
    ns = NativeSolver(64)
    t0 = time.perf_counter()
    nr = NR.NativeRenderer(ns, G.FRAME_W, G.FRAME_H, G.MARCH_SCALE)
    R3A["_meta"].update(create_s=time.perf_counter() - t0, native_march_size=list(nr.march_size()),
                        gpu_at_start=G.gpu_snapshot())
    yield ns, nr
    nr.close()
    ns.close()


@pytest.fixture(scope="session")
def gate(native, py_side):
    """Runs every (scene, camera) case once; tests below only read the results."""
    ns, nr = native
    pr, scenes = py_side
    res, infos, rows, cache = {}, {}, [], {}
    for s in G.SCENES:
        d = scenes[s]
        data, solid = d["ps"].state_np
        ns.upload(data, solid)
        bd, bs = G.bit_compare(ns.download(), data), G.bit_compare(ns.download_solid(), solid)
        res[f"PRE_upload_{s}"] = dict(pass_=bd["identical"] and bs["identical"], data=bd, solid=bs,
                                      solid_cells=int((solid > 0.5).sum()), summary=(
            f"PRE upload {s}: download after nvfx_solver_upload_state: data {'identical' if bd['identical'] else str(bd['n_differ']) + ' differ'}, "
            f"solid {'identical' if bs['identical'] else str(bs['n_differ']) + ' differ'} ({int((solid > 0.5).sum())} solid cells)"))
        cache[s] = dict(data=data, solid=solid, cases={})
        for c in G.CAMERAS:
            r, info, row = G.run_case(nr, pr, d["ps"], c, d["cams"][c])
            res.update(r)
            cache[s]["cases"][c] = dict(camera=info["camera"], prev_camera=info["prev_camera"], py_rgb=info.pop("py_rgb"))
            infos[f"{s}_{c}"] = info
            rows.append(row)
    G.write_png(G.IMAGE_DIR / "contact_sheet.png", G.contact_sheet(rows))
    G.write_dither_cache(G.CACHE_DIR / "dither_inputs.npz", cache)
    del cache
    R3A["_meta"].update(cases=infos, contact_sheet=str(G.IMAGE_DIR / "contact_sheet.png"),
                        contact_sheet_layout="rows: " + ", ".join(CASES) + "; columns: Python | native | |diff| x8 (max-pooled)")
    return res          # every entry carries `pass_`; _take turns it into `pass`


def _take(record, gate, check_id):
    v = dict(gate[check_id])
    v["pass"] = v.pop("pass_")
    _run(record, check_id, v)


# ------------------------------------------------------------------ Python-only (CUDA, no DLL)
@pytest.mark.parametrize("scene", G.SCENES)
def test_PRE_closeup_coverage(record, py_side, scene):
    pr, scenes = py_side
    tried = scenes[scene]["info"]["closeup_search"]
    last = tried[-1]
    ok = last["alpha_gt_005"] >= G.CLOSEUP_MIN_COVERAGE
    _run(record, f"PRE_closeup_coverage_{scene}", dict(tried=tried, summary=(
        f"PRE close-up {scene}: camera_at({last['dist']}) Python alpha > 0.05 covers {last['alpha_gt_005']:.3f} of the "
        f"frame (>= 1/3); alpha > 0.02: {last['alpha_gt_002']:.3f}; tried {[t['dist'] for t in tried]}"), **{"pass": ok}))


@pytest.mark.parametrize("scene", G.SCENES)
def test_HC3_python_controls(record, py_side, scene):
    pr, scenes = py_side
    d = scenes[scene]
    out = G.python_self_controls(pr, d["ps"], d["cams"]["default"])
    bad = []
    for k, v in out.items():
        record(k, v)
        if not v["pass"]:
            bad.append(v["summary"])
    assert not bad, bad


# ------------------------------------------------------------------ native API sanity
def test_N_api_defaults(record, native_api):
    pyd = dict(_dataclass_defaults(ROOT / "vfx" / "render3d" / "renderer.py", "RenderParams3D"))
    nat = NR.params_default()
    bad = {}
    for k, v in pyd.items():
        a = np.asarray(nat[k], np.float32)
        b = np.asarray(v, np.float32)
        if a.shape != b.shape or not np.array_equal(a, b):
            bad[k] = (nat[k], v)
    cam = NR.camera_default()
    camd = dict(_dataclass_defaults(ROOT / "vfx" / "core" / "fields3d.py", "Camera"))
    for k, v in camd.items():
        if not np.array_equal(np.asarray(cam[k], np.float32), np.asarray(v, np.float32)):
            bad[f"camera.{k}"] = (cam[k], v)
    _run(record, "N_api_defaults", {"pass": not bad, "mismatch": bad, "summary": (
        "nvfx_render_params_default == RenderParams3D() and nvfx_camera_default == Camera() (float32)" if not bad
        else f"defaults differ: {bad}")})


def test_N_march_size(record, native, py_side):
    ns, nr = native
    pr, _ = py_side
    a, b = nr.march_size(), pr.march_size()
    _run(record, "N_march_size", {"pass": a == b, "summary": f"march buffer native {a} vs Python {b} (w, h)"})


# ------------------------------------------------------------------ gate checks
@pytest.mark.parametrize("scene", G.SCENES)
def test_PRE_upload(record, gate, scene):
    _take(record, gate, f"PRE_upload_{scene}")


@pytest.mark.parametrize("case", CASES)
def test_R1_colour(record, gate, case):
    env = G.env_dither_mode()
    v = dict(gate[f"R1_{case}"])
    v["pass"] = v.pop("pass_") and env is None
    if env is not None:
        v["summary"] += f"; INVALID: NVFX_RENDER_DITHER_MODE={env} is set in the gate process (the gate is the default mode's)"
    _run(record, f"R1_{case}", v)


@pytest.mark.parametrize("case", CASES)
def test_R2_alpha(record, gate, case):
    _take(record, gate, f"R2_{case}")


@pytest.mark.parametrize("case", CASES)
def test_R3_depth(record, gate, case):
    _take(record, gate, f"R3_{case}")


@pytest.mark.parametrize("kind", ("movingcam", "static"))
@pytest.mark.parametrize("case", CASES)
def test_R4_motion(record, gate, case, kind):
    _take(record, gate, f"R4_{case}_{kind}")


@pytest.mark.parametrize("case", CASES)
def test_R5_aux_off_on(record, gate, case):
    _take(record, gate, f"R5_{case}")


@pytest.mark.parametrize("case", CASES)
def test_R6_repeat(record, gate, case):
    _take(record, gate, f"R6_{case}")


@pytest.mark.parametrize("case", CASES)
def test_NC_R1_yaw1(record, gate, case):
    _take(record, gate, f"NC_R1_yaw1_{case}")


@pytest.mark.parametrize("kind", ("swapxy", "negate"))
@pytest.mark.parametrize("case", CASES)
def test_NC_R4(record, gate, case, kind):
    _take(record, gate, f"NC_R4_{kind}_{case}")


def test_R1_dither_modes(record, gate):
    """REPORT (not the gate): R1 in a fresh process per NVFX_RENDER_DITHER_MODE (unset, 0, 1, 2), with the same
    states and Python images. Passes when every probe ran; says which mode the in-process default equals."""
    inputs = G.CACHE_DIR / "dither_inputs.npz"
    default_sha = {k: v["native_sha256"] for k, v in R3A["_meta"]["cases"].items()}
    probe = ROOT / "native" / "tools" / "gate3a_dither_probe.py"
    modes, errors = {}, []
    for m in DITHER_MODES:
        env = dict(os.environ)
        env["NVFX_DLL"] = str(NR.render_dll_path())
        env.pop("NVFX_RENDER_DITHER_MODE", None)
        if m != "unset":
            env["NVFX_RENDER_DITHER_MODE"] = m
        out = G.CACHE_DIR / f"dither_mode_{m}.json"
        out.unlink(missing_ok=True)
        p = subprocess.run([sys.executable, str(probe), "--inputs", str(inputs), "--out", str(out)], env=env,
                           capture_output=True, text=True, timeout=1800, cwd=str(ROOT))
        if p.returncode != 0 or not out.is_file():
            errors.append(f"mode {m}: rc {p.returncode}: {p.stderr[-800:]}")
            modes[m] = dict(error=p.stderr[-2000:])
            continue
        r = json.loads(out.read_text())
        cs = r["cases"]
        modes[m] = dict(per_case={k: dict(mean_abs=v["r1"]["mean_abs"], p99=v["r1"]["p99"], pass_=v["r1"]["pass"],
                                          struct_energy=v["r1"]["structure"]["struct_energy"],
                                          same_as_default=v["native_sha256"] == default_sha.get(k)) for k, v in cs.items()},
                        worst_mean=round(max(v["r1"]["mean_abs"] for v in cs.values()), 4),
                        worst_p99=max(v["r1"]["p99"] for v in cs.values()),
                        all_pass=all(v["r1"]["pass"] for v in cs.values()),
                        same_as_default=all(v["native_sha256"] == default_sha.get(k) for k, v in cs.items())
                        and set(cs) == set(default_sha))
    equiv = [m for m in ("0", "1", "2") if modes.get(m, {}).get("same_as_default")]
    fresh_default_ok = modes.get("unset", {}).get("same_as_default")
    summ = ("R1 per dither mode (worst mean / p99 over 6 cases): "
            + "; ".join(f"{m}: {v.get('worst_mean', 'ERR')} / {v.get('worst_p99', 'ERR')}" for m, v in modes.items())
            + f". In-process default == fresh-process default: {fresh_default_ok}; default is byte-identical to mode(s) "
            + (", ".join(equiv) if equiv else "none") + ". REPORT ONLY: the gate is R1 in the default mode.")
    _run(record, "R1_dither_modes", {"pass": not errors, "expected": "REPORT", "modes": modes, "errors": errors,
                                     "default_equivalent_to": equiv, "summary": summ})


@pytest.mark.parametrize("mode", ("full", "view"))
def test_T2_timing(record, native, py_side, mode):
    ns, nr = native
    pr, scenes = py_side
    d = scenes["campfire"]
    cam = d["cams"]["default"]
    g0 = G.gpu_snapshot()
    nat = G.native_t2(ns, nr, d["ps"].state_np, d["ps"].sc.forces, G.STEPS, cam, mode)
    pyt = G.python_t2(pr, d["ps"], cam, mode)
    g1 = G.gpu_snapshot()
    t = nat["total"]
    ok = t.get("n", 0) >= G.T2_N and t["median"] <= G.T2_BUDGET_MS and nat["flags_ok"]
    idle = (not g0.get("other_compute_apps")) and (not g1.get("other_compute_apps")) and g0.get("util_pct", 100) <= 5
    f = lambda x: f"{x['median']:.3f} (p10 {x['p10']:.3f}, p95 {x['p95']:.3f})" if x.get("n") else "-"
    what = "new volume every frame (native solver stepped between draws)" if mode == "full" else "view-only (REUSE_VOLUME)"
    summ = (f"T2 {mode}, {what}, aux on, campfire, 2560x1440 x0.5: native total {f(t)} ms, n {t.get('n')} "
            f"(<= {G.T2_BUDGET_MS} ms); prep {nat['prep'].get('median', math.nan):.3f}, march "
            f"{nat['march'].get('median', math.nan):.3f}, post {nat['post'].get('median', math.nan):.3f}; flags ok "
            f"{nat['flags_ok']} {nat['flags_seen']}; Python total {f(pyt['total'])}; GPU util {g0.get('util_pct')}/"
            f"{g1.get('util_pct')} %, mem {g0.get('mem_used_mib')}/{g1.get('mem_used_mib')} MiB; "
            f"{'idle GPU (figure of record)' if idle else 'GPU SHARED (not the figure of record)'}")
    _run(record, f"T2_{mode}", {"pass": bool(ok), "summary": summ, "native": nat, "python": pyt, "gpu_before": g0,
                                "gpu_after": g1, "idle_gpu": bool(idle)})
