"""Stand-ins used ONLY when the real vfx.sim3d / vfx.runtime3d imports fail, so the sandbox always runs.

ToySolver3D: torch-only 3D smoke/fire (semi-Lagrangian advection via grid_sample, buoyancy,
combustion, vorticity confinement, Jacobi pressure projection). It follows the Solver3D interface
but is NOT the reference solver: no accuracy or determinism guarantees beyond "looks like smoke".
classical_upscale_3d: plain trilinear F.interpolate.
There is deliberately no neural stand-in: missing neural modules are shown as unavailable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vfx.core.fields3d import (AMBIENT_TEMP, CH3, FieldMeta3D, FieldState3D, Forces3D, Volume,
                               rasterize_obstacles_3d)

SOLVER3D_VERSION = "app3d-stub-0"


@dataclass
class SolverConfig3D:
    buoyancy_alpha: float = 0.10
    buoyancy_beta: float = 0.0045
    vorticity_eps: float = 0.22
    density_dissipation: float = 0.12
    temperature_cooling: float = 1.4
    burn_rate: float = 5.0
    ignition_temp: float = 600.0
    burn_heat: float = 1100.0
    smoke_from_fuel: float = 0.45
    pressure_tol: float = 1e-3
    pressure_max_iters: int = 30
    heat_diffusion: float = 0.0
    confinement_floor: float = 0.0


class Solver3D:
    def __init__(self, res: int, config: SolverConfig3D | None = None, device="cuda"):
        self.res, self.cfg, self.device = res, config or SolverConfig3D(), torch.device(device)
        self.dx = 1.0 / res
        c = (torch.arange(res, device=self.device, dtype=torch.float32) + 0.5) * self.dx
        self.zz, self.yy, self.xx = torch.meshgrid(c, c, c, indexing="ij")
        g = c * 2 - 1
        gz, gy, gx = torch.meshgrid(g, g, g, indexing="ij")
        self.base_grid = torch.stack([gx, gy, gz], -1)[None]          # (1,D,H,W,3) xyz
        self.solid = torch.zeros(res, res, res, device=self.device)
        k = torch.zeros(3, 3, 3, device=self.device)
        k[0, 1, 1] = k[2, 1, 1] = k[1, 0, 1] = k[1, 2, 1] = k[1, 1, 0] = k[1, 1, 2] = 1.0
        self._nbr = k[None, None]

    def reset(self, obstacles=()) -> FieldState3D:
        self.solid = rasterize_obstacles_3d(list(obstacles), self.res, str(self.device))
        data = torch.zeros(6, self.res, self.res, self.res, device=self.device)
        data[CH3["temperature"]] = AMBIENT_TEMP
        meta = FieldMeta3D.for_res(self.res)
        meta.solver_version = SOLVER3D_VERSION
        return FieldState3D(data, self.solid, meta)

    # ------------------------------------------------------------------ pieces
    def advect(self, field, velocity, dt):
        squeeze = field.dim() == 3
        f = field[None, None] if squeeze else field[None]
        disp = (velocity * (2.0 * dt)).permute(1, 2, 3, 0)[None]      # m/s * dt -> normalized (2/1m)
        out = F.grid_sample(f, self.base_grid - disp, mode="bilinear", padding_mode="border", align_corners=False)
        return out[0, 0] if squeeze else out[0]

    def _grad(self, f):
        p = F.pad(f[None, None], (1, 1, 1, 1, 1, 1), mode="replicate")[0, 0]
        gx = (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) / (2 * self.dx)
        gy = (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) / (2 * self.dx)
        gz = (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) / (2 * self.dx)
        return gx, gy, gz

    def _div(self, u):
        return self._grad(u[0])[0] + self._grad(u[1])[1] + self._grad(u[2])[2]

    def divergence(self, state):
        u = state.velocity
        return (self._div(u).abs() * self.dx / (u.abs().amax() + 1e-8)) * (1 - state.solid)

    def _walls(self, u, fluid):
        u.mul_(fluid)
        u[0, :, :, 0] = 0; u[0, :, :, -1] = 0
        u[1, :, 0, :] = 0; u[1, :, -1, :] = 0
        u[2, 0, :, :] = 0; u[2, -1, :, :] = 0

    def project(self, state):
        u, fluid = state.velocity, 1 - state.solid
        self._walls(u, fluid)
        rhs = self._div(u) * (self.dx * self.dx)
        p = torch.zeros_like(rhs)
        for _ in range(self.cfg.pressure_max_iters):
            nb = F.conv3d(F.pad(p[None, None], (1, 1, 1, 1, 1, 1), mode="replicate"), self._nbr)[0, 0]
            p = (nb - rhs) / 6.0
        gx, gy, gz = self._grad(p)
        u[0] -= gx; u[1] -= gy; u[2] -= gz
        self._walls(u, fluid)
        return state

    def _confine(self, u, dt):
        ux, uy, uz = u[0], u[1], u[2]
        dux, duy, duz = self._grad(ux), self._grad(uy), self._grad(uz)
        wx = duz[1] - duy[2]
        wy = dux[2] - duz[0]
        wz = duy[0] - dux[1]
        mag = torch.sqrt(wx * wx + wy * wy + wz * wz)
        nx, ny, nz = self._grad(mag)
        nn = torch.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
        nx, ny, nz = nx / nn, ny / nn, nz / nn
        e = self.cfg.vorticity_eps * self.dx * dt
        u[0] += e * (ny * wz - nz * wy)
        u[1] += e * (nz * wx - nx * wz)
        u[2] += e * (nx * wy - ny * wx)

    def _falloff(self, pos, radius):
        r2 = (self.xx - pos[0]) ** 2 + (self.yy - pos[1]) ** 2 + (self.zz - pos[2]) ** 2
        return torch.exp(-r2 / (radius * radius))

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, state: FieldState3D, forces: Forces3D, dt: float) -> FieldState3D:
        c, d = self.cfg, state.data
        dens, temp, fuel = d[CH3["density"]], d[CH3["temperature"]], d[CH3["fuel"]]
        u = state.velocity
        for e in forces.emitters:
            w = self._falloff(e.pos, e.radius)
            if e.density_rate:
                dens.add_(w * (e.density_rate * dt * 4.0))
            if e.fuel_rate:
                fuel.add_(w * (e.fuel_rate * dt))
            if e.temperature > 0:
                temp.copy_(torch.maximum(temp, AMBIENT_TEMP + (e.temperature - AMBIENT_TEMP) * w))
            if any(e.velocity):
                wv = w * 0.5
                for i in range(3):
                    u[i].mul_(1 - wv).add_(wv * e.velocity[i])
        for im in forces.impulses:
            w = self._falloff(im.pos, im.radius)
            for i in range(3):
                u[i].add_(w * im.velocity[i])
        # combustion
        burn = torch.where(temp > c.ignition_temp, fuel * min(1.0, c.burn_rate * dt), torch.zeros_like(fuel))
        fuel.sub_(burn)
        temp.add_(burn * c.burn_heat)
        dens.add_(burn * c.smoke_from_fuel)
        # buoyancy + wind
        u[1].add_(dt * (c.buoyancy_beta * (temp - AMBIENT_TEMP) - c.buoyancy_alpha * dens))
        if any(forces.wind):
            for i in range(3):
                u[i].add_((forces.wind[i] - u[i]) * min(1.0, 0.6 * dt))
        if c.vorticity_eps > 0:
            self._confine(u, dt)
        self.project(state)
        adv = self.advect(d, u.clone(), dt)
        d.copy_(adv)
        fluid = 1 - state.solid
        dens.mul_(fluid * math.exp(-c.density_dissipation * dt))
        fuel.mul_(fluid * 0.97)
        temp.sub_(AMBIENT_TEMP).mul_(math.exp(-c.temperature_cooling * dt)).add_(AMBIENT_TEMP)
        self._walls(u, fluid)
        state.meta.step += 1
        return state


def classical_upscale_3d(state: FieldState3D, factor: int = 4, mode: str = "trilinear") -> Volume:
    x = torch.stack([state.data[CH3["density"]], state.data[CH3["temperature"]]])[None]
    y = F.interpolate(x, scale_factor=factor, mode=mode, align_corners=False)[0]
    return Volume(y[0], y[1])
