"""vfx/correct3d: learned large-scale velocity correction for the 64^3 solver (feasibility study).

    corr = VelocityCorrector(checkpoint)          # identity when checkpoint is None
    ...
    corr.apply(state, dt)      # call BEFORE solver.step(state, forces, dt): u += c(state), in place;
    solver.step(state, forces, dt)                # this step's projection removes the divergent part

See README.md and EXPERIMENTS.md.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from vfx.core.fields3d import SIM_DT

from .model import OUT_SCALE, CorrNet, features

__all__ = ["VelocityCorrector"]


class VelocityCorrector:
    """Opt-in velocity correction. Applies u += clamp(net(state)) * fluid, BEFORE the solver step.

    Limiter (all on GPU, no host sync): per-cell |c| <= cell_cap m/s; if the correction's rms exceeds
    max_ratio * rms(u) + abs_floor, or is non-finite, the whole correction is dropped this step
    (identity fallback). last_stats = GPU (3,) [rms c, rms u, fallback flag].
    """

    def __init__(self, checkpoint_path=None, device="cuda", cell_cap: float = 0.15, max_ratio: float = 0.2,
                 abs_floor: float = 0.01, use_graph: bool = True, gain: float = 1.0):
        self.device = device
        self.net = None
        self.cell_cap, self.max_ratio, self.abs_floor, self.gain = cell_cap, max_ratio, abs_floor, gain
        self.timings: dict[str, float] = {}
        self.use_graph = use_graph
        self._graphs: dict = {}
        self.last_stats = None
        self.profile = False
        if checkpoint_path is not None:
            ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
            self.net = CorrNet(ck.get("width", 16)).to(device).eval()
            self.net.load_state_dict(ck["state_dict"])
            self.net.half()
            self.meta = ck.get("meta", {})
        self._dt = torch.ones((), device=device)

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def correction(self, data: torch.Tensor, solid: torch.Tensor) -> torch.Tensor:
        """(3,N,N,N) limited correction for `data` (already scaled by dt / SIM_DT and fluid-masked)."""
        f = features(data, solid).half()
        c = self.net(f).float() * (OUT_SCALE * self.gain)
        c = F.interpolate(c, scale_factor=2, mode="trilinear", align_corners=False)[0]
        c = c * (self._dt / SIM_DT) * (solid < 0.5)
        mag = c.norm(dim=0, keepdim=True)
        c = c * torch.clamp(self.cell_cap / (mag + 1e-12), max=1.0)
        u = data[1:4]
        rc = c.square().mean().sqrt()
        ru = u.square().mean().sqrt()
        bad = (~torch.isfinite(rc)) | (rc > self.max_ratio * ru + self.abs_floor)
        c = torch.where(bad, torch.zeros_like(c), torch.nan_to_num(c))
        self.last_stats = torch.stack([torch.nan_to_num(rc, 1e9), ru, bad.float()])
        return c

    @torch.no_grad()
    def _apply_eager(self, data, solid):
        data[1:4] += self.correction(data, solid)

    @torch.no_grad()
    def apply(self, state, dt: float) -> None:
        if self.net is None:
            return
        self._dt.fill_(float(dt))
        if self.profile:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
        if not self.use_graph:
            self._apply_eager(state.data, state.solid)
        else:
            key = (state.data.data_ptr(), state.solid.data_ptr())
            g = self._graphs.get(key)
            if g is None:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                backup = state.data.clone()
                with torch.cuda.stream(s):
                    for _ in range(3):
                        self._apply_eager(state.data, state.solid)
                torch.cuda.current_stream().wait_stream(s)
                state.data.copy_(backup)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._apply_eager(state.data, state.solid)
                state.data.copy_(backup)     # capture does not run the work; restore anyway for safety
                self._graphs[key] = (g, self.last_stats)
                if len(self._graphs) > 4:
                    self._graphs.pop(next(iter(self._graphs)))
                g = self._graphs[key]
            g[0].replay()
            self.last_stats = g[1]
        if self.profile:
            e1.record()
            e1.synchronize()
            self.timings = {"total": e0.elapsed_time(e1)}
