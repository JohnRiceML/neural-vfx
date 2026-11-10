"""Gate 1 tests (native/README.md table): P1, P2, P3, S1-S4, T1. Backend chosen with --backend.

    <venv python> -m pytest native/tests/test_gate1.py -q -s                    # native (skips w/o DLL)
    <venv python> -m pytest native/tests/test_gate1.py -q -s --backend=pyfake   # harness validation
"""
from __future__ import annotations

import pytest

import gate1_checks as G

SCENES = ("campfire", "plume")


def _run(record, check_id, result):
    record(check_id, result)
    assert result["pass"], result["summary"]


@pytest.mark.parametrize("scene", SCENES)
def test_P1_single_step_parity(make, record, scene):
    _run(record, f"P1_{scene}", G.check_p1(make, scene))


def test_P2_projection_only(make, record):
    _run(record, "P2_project", G.check_p2_project(make))


@pytest.mark.parametrize("scene", SCENES)
def test_P2_zero_sources_step(make, record, scene):
    _run(record, f"P2_zero_sources_{scene}", G.check_p2_zero_sources(make, scene))


@pytest.mark.parametrize("scene", SCENES)
def test_P3_free_running(make, record, scene):
    _run(record, f"P3_{scene}", G.check_p3(make, scene))


@pytest.mark.parametrize("scene", SCENES)
def test_S1_divergence(make, record, scene):
    _run(record, f"S1_{scene}", G.check_s1(make, scene))


def test_S2_mass_closed_box(make, record):
    _run(record, "S2_mass_box", G.check_s2(make))


def test_S3_determinism(make, record):
    _run(record, "S3_determinism", G.check_s3(make))


def test_S3b_reset_reuse(make, record):
    _run(record, "S3b_reset_reuse", G.check_s3_reset_reuse(make))


def test_S4_stress(make, record):
    _run(record, "S4_stress_full", G.check_s4(make))


@pytest.mark.parametrize("cfg", ("default", "runtime"))
@pytest.mark.parametrize("scene", SCENES)
def test_T1_timing(make, record, scene, cfg):
    _run(record, f"T1_{scene}_{cfg}", G.check_t1(make, scene, cfg))
