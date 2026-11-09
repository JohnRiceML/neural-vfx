"""Layer 5 harness self-tests + M0 gate (GPU detected; harness reports timings for a trivial kernel).
The full benchmark is `python -m bench.run`; it is not run inside pytest."""
import pytest
import torch

from bench import run as bench


def test_m0_gpu_detected():
    assert torch.cuda.is_available(), "M0 gate: CUDA GPU not detected"
    print(torch.cuda.get_device_name())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_m0_trivial_kernel_timing():
    b = bench.Bench(iters=50)
    bench.trivial(b)
    t = b.res["stages"]["trivial_torch"]
    assert 0 < t["p50_ms"] < 5.0
    if "trivial_warp" in b.res["stages"]:
        assert 0 < b.res["stages"]["trivial_warp"]["p50_ms"] < 5.0
    else:
        pytest.fail(f"warp trivial kernel failed: {b.res['skipped'].get('trivial_warp')}")


def _res(ms, src="solver", vram=1000.0):
    st = {k: {"mean_ms": v, "p50_ms": v, "p95_ms": v} for k, v in ms.items()}
    return {"scene_source": src, "stages": st,
            "vram": {"torch_peak_allocated_mb": vram, "device_used_peak_minus_start_mb": vram}}


def test_compare_flags_regressions_over_10pct():
    base = _res({"solver_step_128": 1.0, "render_draw_1024": 1.0})
    assert bench.compare(_res({"solver_step_128": 1.09, "render_draw_1024": 0.5}), base) == []
    regs = bench.compare(_res({"solver_step_128": 1.11, "render_draw_1024": 1.0}), base)
    assert len(regs) == 1 and "solver_step_128" in regs[0]
    assert bench.compare(_res({"solver_step_128": 1.0}, vram=1200.0), base)      # VRAM regression
    # different scene source: stage timings are not comparable, nothing flagged
    assert bench.compare(_res({"solver_step_128": 5.0}, src="synthetic"), base) == []
