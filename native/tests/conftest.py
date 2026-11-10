"""pytest setup for native/tests (OWNER: native test engineer).

Backend: --backend=native|pyfake (default: env NVFX_BACKEND, else native). In native mode every gate test
skips with a clear message when nvfx.dll is absent. Results of the gate tests are written at the end of the
session to native/build/test_results/gate1_<backend>.json and .md (native/build is git-ignored).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "native" / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

RESULTS: dict = {}
RESULT_DIR = ROOT / "native" / "build" / "test_results"


def pytest_addoption(parser):
    parser.addoption("--backend", default=os.environ.get("NVFX_BACKEND", "native"),
                     help="native (nvfx.dll) | pyfake (reference-backed fake, harness validation)")


@pytest.fixture(scope="session")
def backend_name(request) -> str:
    return request.config.getoption("--backend")


@pytest.fixture(scope="session")
def make(backend_name):
    """Factory make(res, config) -> solver with the NativeSolver interface."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for the Python reference")
    if backend_name == "native":
        from nvfx_ctypes import dll_available, load_library
        ok, msg = dll_available()
        if not ok:
            pytest.skip(f"native backend: {msg}")
        lib = load_library()
        RESULTS.setdefault("_meta", {})["nvfx_version"] = lib.nvfx_version_string().decode()
        RESULTS["_meta"]["dll"] = msg
    from backends import make_backend
    import warp as wp
    wp.config.quiet = True
    return lambda res, cfg: make_backend(backend_name, res, cfg)


@pytest.fixture(scope="session")
def record(backend_name):
    def _rec(check_id: str, result: dict):
        RESULTS[check_id] = result
        tag = "harness control" if check_id.startswith("HC_") else backend_name
        print(f"\n[{tag}] {'PASS' if result.get('pass') else 'FAIL'} {result.get('summary', '')}")
    return _rec


def _clean(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else str(o)
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item"):
        return _clean(o.item())
    return o


def pytest_sessionfinish(session, exitstatus):
    backend = session.config.getoption("--backend")
    hc = {k: v for k, v in RESULTS.items() if k.startswith("HC_")}
    if hc:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        (RESULT_DIR / "harness_controls.json").write_text(json.dumps(_clean(hc), indent=1))
        lines = ["| control | expected | check result | control ok | measured |", "|---|---|---|---|---|"]
        for k, v in sorted(hc.items()):
            lines.append(f"| {k} | {v.get('expected', 'PASS')} | {'PASS' if v.get('pass') else 'FAIL'} | "
                         f"{v.get('pass_control', v.get('pass'))} | {v.get('summary', '').replace('|', '/')} |")
        (RESULT_DIR / "harness_controls.md").write_text("\n".join(lines) + "\n")
    checks = {k: v for k, v in RESULTS.items() if not k.startswith("_") and not k.startswith("HC_")}
    if not checks:
        return
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    meta = RESULTS.get("_meta", {})
    meta.update(backend=backend, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    path = RESULT_DIR / f"gate1_{backend}.json"
    old = {}
    if path.exists():   # merge with earlier partial runs (-k) of the same backend
        try:
            old = json.loads(path.read_text())
        except Exception:
            old = {}
    old.update(_clean(checks))
    old["_meta"] = _clean(meta)
    path.write_text(json.dumps(old, indent=1))
    lines = [f"Gate 1 results, backend `{backend}`, {meta['finished']}"
             + (f", {meta.get('nvfx_version')}" if meta.get("nvfx_version") else ""), "",
             "| check | result | measured |", "|---|---|---|"]
    for k in sorted(x for x in old if not x.startswith("_")):
        v = old[k]
        lines.append(f"| {k} | {'PASS' if v.get('pass') else '**FAIL**'} | {v.get('summary', '').replace('|', '/')} |")
    t1 = sorted(k for k in old if k.startswith("T1") and "backend" in old[k] and "total" in old[k]["backend"])
    if t1:
        f = lambda d: f"{d['median']:.3f} ({d['p95']:.3f})"
        lines += ["", "T1, ms median (p95). Backend: GPU timestamps; Python: step_profiled (3 graphs + CUDA events) "
                  "and the single-graph step, same session.", "",
                  "| scene, config | backend adv | forces | proj | total | n | PCG it | Python adv | forces | proj "
                  "| total | graph step | GPU util / mem before, after |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in t1:
            b, p, g0, g1 = old[k]["backend"], old[k]["python"], old[k]["gpu_before"], old[k]["gpu_after"]
            lines.append(f"| {k[3:]} | {f(b['advection'])} | {f(b['forces'])} | {f(b['projection'])} | "
                         f"{f(b['total'])} | {b['total']['n']} | {b['pcg_iters']['median']:.0f} | {f(p['advection'])} | "
                         f"{f(p['forces'])} | {f(p['projection'])} | {f(p['total'])} | {f(p['step_graph'])} | "
                         f"{g0.get('util_pct')} % {g0.get('mem_used_mib')} MiB, {g1.get('util_pct')} % "
                         f"{g1.get('mem_used_mib')} MiB |")
    (RESULT_DIR / f"gate1_{backend}.md").write_text("\n".join(lines) + "\n")
