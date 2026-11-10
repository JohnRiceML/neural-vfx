"""P3 NOISE FLOOR -- a MEASUREMENT, NOT A GATE (OWNER: native test engineer).

Question: is the backend-vs-Python difference of P3 (free running, 120 steps, default config) larger than the
spread Python shows against itself when its state is perturbed once, at rounding level, after step 1?
For campfire and plume it builds an ensemble of Python runs (16 velocity scalings 1 + k*1e-7, k = +-1..8;
16 seeded +-1-ulp nudges of vx, vy, vz, T at 256 random fluid cells; 8 'matched' members with multiplicative
noise of the backend's own step-1 rel_l2 per channel), plus a zero-perturbation control that must be
bit-identical to the unperturbed run. The only assertion is that control (and that perturbations took
effect). P3 and its thresholds are untouched.

    <venv python> -m pytest native/tests/test_p3_noise_floor.py -q -s            # backend native
Results: native/build/test_results/p3_noise_floor.json / .md (suffix _<backend> for other backends).
"""
from __future__ import annotations

import json

import pytest

import gate1_checks as G
from conftest import RESULT_DIR, _clean

SCENES = ("campfire", "plume")
_OUT: dict = {}


def _fmt_dist(d, unit="%"):
    f = (lambda x: f"{x:+.3f}") if unit == "%" else (lambda x: f"{x:.2e}")
    return f"{f(d['min'])} / {f(d['median'])} / {f(d['max'])} / {d['std']:.3g}"


def _write(backend):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "p3_noise_floor" + ("" if backend == "native" else f"_{backend}")
    meta = {"backend": backend, "gpu": G.gpu_snapshot()}
    (RESULT_DIR / f"{stem}.json").write_text(json.dumps(_clean({"_meta": meta, **_OUT}), indent=1))
    L = [f"P3 noise floor (MEASUREMENT, not a gate), backend `{backend}`, {meta['gpu']['time']}", "",
         "Signed differences at step 120 vs the unperturbed Python run. Distribution: min / median / max / std.",
         "Backend position: z = (value - ensemble mean) / ensemble std; k/N = members with |x| >= |backend|.", "",
         "| scene | ensemble | N | heat % | mass % | centroid m | backend heat % (z, k/N) | backend mass % (z) "
         "| backend centroid m (z) |", "|---|---|---|---|---|---|---|---|---|"]
    for sc, r in _OUT.items():
        for g, key, pos in (("rounding (vel_scale+ulp_nudge)", "rounding (vel_scale+ulp_nudge)", "rounding"),
                            ("vel_scale only", "vel_scale", None), ("ulp_nudge only", "ulp_nudge", None),
                            ("matched step-1 amplitude", "matched_amp", "matched_amp")):
            s = r["stats_at_120"][key]
            row = f"| {sc} | {g} | {s['heat_pct']['n']} | {_fmt_dist(s['heat_pct'])} | {_fmt_dist(s['mass_pct'])} | " \
                  f"{_fmt_dist(s['centroid_m'], 'm')} |"
            if pos:
                p = r["native_position_at_120"][pos]
                row += (f" {p['heat_pct']['value']:+.3f} (z {p['heat_pct']['z']:+.1f}, "
                        f"{p['heat_pct']['members_with_abs_ge']}/{p['heat_pct']['n']}) | "
                        f"{p['mass_pct']['value']:+.4f} (z {p['mass_pct']['z']:+.1f}) | "
                        f"{p['centroid_m']['value']:.2e} (z {p['centroid_m']['z']:+.1f}) |")
            else:
                row += " | | |"
            L.append(row)
    L += ["", "Heat % over steps: backend vs rounding-ensemble envelope [min, max] and velocity rel_l2 "
              "(backend / ensemble median).", "",
          "| scene | step | backend heat % | ensemble heat % [min, max] | matched [min, max] | backend vel rel_l2 "
          "| ensemble median vel rel_l2 | matched median |", "|---|---|---|---|---|---|---|---|"]
    for sc, r in _OUT.items():
        for s in r["sample_steps"]:
            n, e, m = r["native_curve"][s], r["envelope_rounding"][s], r["envelope_matched_amp"][s]
            L.append(f"| {sc} | {s} | {n['heat_pct']:+.4f} | [{e['heat_min']:+.4f}, {e['heat_max']:+.4f}] | "
                     f"[{m['heat_min']:+.4f}, {m['heat_max']:+.4f}] | {n['vel_rel_l2']:.2e} | "
                     f"{e['vel_rel_l2_median']:.2e} | {m['vel_rel_l2_median']:.2e} |")
    L += ["", "| scene | zero-perturbation control bit-identical | members changed at step 1 | backend step-1 rel_l2 "
              "(per channel) |", "|---|---|---|---|"]
    for sc, r in _OUT.items():
        L.append(f"| {sc} | {r['control_zero_perturbation']['bit_identical_all_samples']} | "
                 f"{r['members_changed_at_step1']}/{sum(r['n_members'].values())} | "
                 f"{', '.join(f'{k} {v:.1e}' for k, v in r['backend_step1_rel_l2_per_channel'].items())} |")
    (RESULT_DIR / f"{stem}.md").write_text("\n".join(L) + "\n")


@pytest.mark.parametrize("scene", SCENES)
def test_p3_noise_floor(make, backend_name, scene):
    r = G.measure_p3_noise_floor(make, scene)
    _OUT[scene] = r
    _write(backend_name)
    p = r["native_position_at_120"]["rounding"]["heat_pct"]
    s = r["stats_at_120"]["rounding (vel_scale+ulp_nudge)"]["heat_pct"]
    print(f"\n[measurement, not a gate] P3 noise floor {scene}: Python rounding ensemble heat % at 120 "
          f"min {s['min']:+.3f} median {s['median']:+.3f} max {s['max']:+.3f} std {s['std']:.3f} (N {s['n']}); "
          f"backend {p['value']:+.3f} %, z {p['z']:+.1f}, {p['members_with_abs_ge']}/{p['n']} members with |x| >= it")
    assert r["control_zero_perturbation"]["bit_identical_all_samples"], "zero perturbation must be bit-identical"
    assert r["members_changed_at_step1"] == sum(r["n_members"].values()), "a perturbation had no effect"
