"""Owner self-check for vfx.particles3d (not the gate tests):

    python -m vfx.particles3d.selfcheck

1. uniform flow, sub-grid off: every particle moves exactly u*dt per tick (RK2 of a constant field)
2. smoke plume past a sphere (preset 2), 150 ticks: particle smoke mass vs grid mass, restriction of the volume to 64^3
   vs the solver density, no live particle inside a solid, everything finite
3. campfire (preset 1): fire temperature carried (max of the volume temperature vs the grid's), smoke from fuel
4. determinism: particle state bit-identical on rerun; the deposited volume only differs at float-atomic rounding
"""
from __future__ import annotations

import torch
import warp as wp

from vfx.core.fields3d import CH3, SIM_DT, Emitter3D, Forces3D


def _run(preset, ticks, count=500_000, res=128, **kw):
    from vfx.app3d.presets import PRESETS
    from vfx.sim3d import Solver3D, runtime_config

    from .particles import DetailParticles
    pre = PRESETS[preset]
    cfg = runtime_config(**pre.sim)
    s = Solver3D(64, cfg)
    st = s.reset(pre.obstacles)
    dp = DetailParticles(count, res, config=cfg, **kw)
    f = Forces3D(emitters=list(pre.emitters), wind=tuple(pre.wind))
    for _ in range(ticks):
        st = s.step(st, f, SIM_DT)
        dp.step(st, f, SIM_DT)
    return st, dp


def main():
    ok = True

    def check(name, cond, info):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}: {info}")

    from vfx.sim3d import Solver3D, runtime_config

    from .particles import DetailParticles
    # 1. uniform flow
    s = Solver3D(64, runtime_config())
    st = s.reset([])
    st.data[CH3["vel_x"]] = 0.3
    st.data[CH3["vel_y"]] = 0.2
    dp = DetailParticles(20_000, 128, sg_amp=0.0, config=runtime_config(density_dissipation=0.0, temperature_cooling=0.0))
    f = Forces3D(emitters=[Emitter3D((0.3, 0.3, 0.5), 0.05, density_rate=1.0, temperature=400.0)])
    dp.step(st, f, SIM_DT)
    p0 = wp.to_torch(dp.pos).clone()
    alive0 = wp.to_torch(dp.q)[:, 3] > 0
    dp.step(st, Forces3D(), SIM_DT)
    d = (wp.to_torch(dp.pos) - p0)[alive0]
    err = (d - torch.tensor([0.3 * SIM_DT, 0.2 * SIM_DT, 0.0], device=d.device)).abs().max().item()
    check("uniform advection exact", err < 1e-6, f"max error {err:.2e} m over {int(alive0.sum())} particles")

    # 2. smoke plume past a sphere
    st, dp = _run(2, 150)
    dg = dp.diagnostics(st)
    q = wp.to_torch(dp.q)
    pos = wp.to_torch(dp.pos)
    alive = q[:, 3] > 0
    ijk = (pos[alive] * 64).long().clamp(0, 63)
    in_solid = int((st.solid[ijk[:, 2], ijk[:, 1], ijk[:, 0]] > 0.5).sum())
    ratio = dg["particle_mass"] / dg["grid_mass"]
    check("particle mass tracks grid mass", 0.75 < ratio < 1.25,
          f"particle {dg['particle_mass']:.4g} vs grid {dg['grid_mass']:.4g} (ratio {ratio:.3f}), alive {dg['alive']}")
    check("volume restricts to solver density", dg["restrict_rel_err"] < 0.15,
          f"rel L1 {dg['restrict_rel_err']:.3f} (trilinear ratio field; volume mass {dg['volume_mass']:.4g})")
    check("no particles in solids", in_solid == 0, f"{in_solid} live particles in solid cells")
    check("finite", bool(torch.isfinite(dp.dens).all() and torch.isfinite(dp.temp).all()), "density/temperature")

    # 3. campfire
    st, dp = _run(1, 120)
    dg = dp.diagnostics(st)
    tg = st.data[CH3["temperature"]].max().item()
    tv = dp.temp.max().item()
    tp = wp.to_torch(dp.q)[:, 3].max().item()
    check("fire temperature carried", abs(tv - tg) / tg < 0.2,
          f"max T: grid {tg:.0f} K, volume {tv:.0f} K, hottest particle {tp:.0f} K")
    ratio = dg["particle_mass"] / max(dg["grid_mass"], 1e-12)
    check("smoke from fuel", 0.5 < ratio < 1.5, f"particle smoke {dg['particle_mass']:.4g} vs grid {dg['grid_mass']:.4g} "
          f"(ratio {ratio:.3f}); unburnt particle fuel {dg['particle_fuel']:.4g}")

    # 4. determinism
    st1, a = _run(2, 40, count=200_000)
    pa, qa, da = wp.to_torch(a.pos).clone(), wp.to_torch(a.q).clone(), a.dens.clone()
    st2, b = _run(2, 40, count=200_000)
    same = torch.equal(pa, wp.to_torch(b.pos)) and torch.equal(qa, wp.to_torch(b.q))
    dd = ((da - b.dens).abs().max() / da.abs().max()).item()
    check("particle state bit-identical on rerun", same, "positions + payload")
    check("deposit equal up to atomic order", dd < 1e-4, f"max rel diff {dd:.2e}")
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
