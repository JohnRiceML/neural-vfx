"""Stand-ins used ONLY when the real vfx.sim / vfx.runtime imports fail, so the app always runs.

ToySolver: torch-only 2D smoke/fire (semi-Lagrangian advection, buoyancy, vorticity confinement,
Jacobi pressure projection). It follows the vfx.sim.Solver interface but is not the reference
solver: no accuracy or determinism guarantees beyond "looks like smoke".
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vfx.core.fields import (AMBIENT_TEMP, CH, DOMAIN_SIZE, HR_RES, FieldMeta, FieldState, Forces,
                             HRFields, rasterize_obstacles)

SOLVER_VERSION = "app-stub-0"


@dataclass
class SolverConfig:
    buoyancy_alpha: float = 0.08
    buoyancy_beta: float = 0.004
    vorticity_eps: float = 0.12
    density_dissipation: float = 0.25
    temperature_cooling: float = 1.6
    burn_rate: float = 4.0
    ignition_temp: float = 600.0
    burn_heat: float = 900.0
    smoke_from_fuel: float = 0.5
    pressure_tol: float = 1e-3
    pressure_max_iters: int = 50


class Solver:
    def __init__(self, res: int, config: SolverConfig | None = None, device="cuda"):
        self.res, self.cfg, self.device = res, config or SolverConfig(), torch.device(device)
        self.dx = DOMAIN_SIZE / res
        c = (torch.arange(res, device=self.device, dtype=torch.float32) + 0.5) * self.dx
        self.yy, self.xx = torch.meshgrid(c, c, indexing="ij")
        g = (torch.arange(res, device=self.device, dtype=torch.float32) * 2 + 1) / res - 1
        gy, gx = torch.meshgrid(g, g, indexing="ij")
        self.base_grid = torch.stack([gx, gy], -1)[None]
        self.solid = torch.zeros(res, res, device=self.device)
        self._cross = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]], device=self.device)[None, None]

    def reset(self, obstacles=()) -> FieldState:
        self.solid = rasterize_obstacles(list(obstacles), self.res, str(self.device))
        data = torch.zeros(5, self.res, self.res, device=self.device)
        data[CH["temperature"]] = AMBIENT_TEMP
        meta = FieldMeta.for_res(self.res)
        meta.solver_version = SOLVER_VERSION
        return FieldState(data, self.solid, meta)

    # ------------------------------------------------------------------ pieces
    def advect(self, field, velocity, dt):
        squeeze = field.dim() == 2
        f = field[None, None] if squeeze else field[None]
        s = 2.0 * dt / (self.dx * self.res)
        disp = torch.stack([velocity[0] * s, velocity[1] * s], -1)[None]
        out = F.grid_sample(f, self.base_grid - disp, mode="bilinear", padding_mode="border", align_corners=False)
        return out[0, 0] if squeeze else out[0]

    def _ddx(self, f):
        p = F.pad(f[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
        return (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * self.dx), (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * self.dx)

    def divergence(self, state):
        u = state.velocity
        dudx, _ = self._ddx(u[0])
        _, dvdy = self._ddx(u[1])
        mx = u.abs().amax()
        return ((dudx + dvdy).abs() * self.dx / (mx + 1e-8)) * (1 - state.solid)

    def project(self, state):
        """Jacobi pressure solve (one conv per iteration); solids handled by masking velocity."""
        u, fluid = state.velocity, 1 - state.solid
        u.mul_(fluid)
        u[0, :, 0] = 0; u[0, :, -1] = 0; u[1, 0, :] = 0; u[1, -1, :] = 0
        dudx, _ = self._ddx(u[0])
        _, dvdy = self._ddx(u[1])
        rhs = ((dudx + dvdy) * fluid * (self.dx * self.dx))[None, None]
        p = torch.zeros_like(rhs)
        k = self._cross
        for _ in range(self.cfg.pressure_max_iters):
            p = (F.conv2d(F.pad(p, (1, 1, 1, 1), mode="replicate"), k) - rhs) * 0.25
        gx, gy = self._ddx(p[0, 0])
        u[0] -= gx * fluid
        u[1] -= gy * fluid
        return state

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, state: FieldState, forces: Forces, dt: float) -> FieldState:
        c, d = self.cfg, state.data
        T = d[CH["temperature"]]
        for e in forces.emitters:
            w = torch.exp(-((self.xx - e.pos[0]) ** 2 + (self.yy - e.pos[1]) ** 2) / (e.radius ** 2))
            w = w * (w > 0.05)
            d[CH["density"]] += e.density_rate * dt * w
            d[CH["fuel"]] += e.fuel_rate * dt * w
            if e.temperature > 0:
                T.copy_(torch.maximum(T, AMBIENT_TEMP + (e.temperature - AMBIENT_TEMP) * w))
            if e.velocity != (0.0, 0.0):
                d[CH["vel_x"]] += (e.velocity[0] - d[CH["vel_x"]]) * w
                d[CH["vel_y"]] += (e.velocity[1] - d[CH["vel_y"]]) * w
        for im in forces.impulses:
            w = torch.exp(-((self.xx - im.pos[0]) ** 2 + (self.yy - im.pos[1]) ** 2) / (im.radius ** 2))
            d[CH["vel_x"]] += im.velocity[0] * w
            d[CH["vel_y"]] += im.velocity[1] * w
        # combustion
        fuel = d[CH["fuel"]]
        burn = torch.minimum(fuel, c.burn_rate * dt * fuel.clamp(max=1) * 4) * (T > c.ignition_temp)
        fuel -= burn
        T += burn * c.burn_heat
        d[CH["density"]] += burn * c.smoke_from_fuel
        # body forces
        d[CH["vel_y"]] += dt * (-c.buoyancy_alpha * d[CH["density"]] + c.buoyancy_beta * (T - AMBIENT_TEMP))
        d[CH["vel_x"]] += dt * forces.wind[0]
        d[CH["vel_y"]] += dt * forces.wind[1]
        # vorticity confinement
        if c.vorticity_eps > 0:
            _, dudy = self._ddx(d[CH["vel_x"]])
            dvdx, _ = self._ddx(d[CH["vel_y"]])
            w = dvdx - dudy
            gx, gy = self._ddx(w.abs())
            n = torch.sqrt(gx * gx + gy * gy) + 1e-6
            d[CH["vel_x"]] += dt * c.vorticity_eps * self.dx * (gy / n) * w * 30
            d[CH["vel_y"]] -= dt * c.vorticity_eps * self.dx * (gx / n) * w * 30
        self.project(state)
        state.data = self.advect(d, state.velocity.clone(), dt)
        d = state.data
        d[CH["density"]] *= (1 - c.density_dissipation * dt)
        d[CH["temperature"]] = AMBIENT_TEMP + (d[CH["temperature"]] - AMBIENT_TEMP) * torch.exp(
            torch.tensor(-c.temperature_cooling * dt, device=self.device))
        d[CH["density"]].clamp_(min=0).mul_(1 - self.solid)
        d[CH["fuel"]].clamp_(min=0)
        state.meta.step += 1
        return state


def classical_upscale(lr_state: FieldState, mode: str = "bicubic") -> HRFields:
    x = lr_state.data[[CH["density"], CH["temperature"]]][None]
    s = HR_RES // x.shape[-1]
    up = F.interpolate(x, scale_factor=s, mode=mode, **({} if mode == "nearest" else {"align_corners": False}))[0]
    return HRFields(up[0].clamp(min=0), up[1].clamp(min=AMBIENT_TEMP))
