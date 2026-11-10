"""Route A runtime: VolumeUpscaler — sparse-brick neural 3D upscale, FP16, CUDA graphs, no host sync.

Per frame (R = LR res, N = s*R HR res, bricks of b^3 LR cells = (s*b)^3 HR voxels, nb = R/b per axis):
  1. cast the LR state into a static fp16 buffer (outside the graphs).
  2. pre graph (fixed shape): replicate-pad LR by `halo`; trilinear base of (density, T) written into
     the HR output canvas (this is the classical-upscale floor); brick activity mask
     (density > thresh or T > ambient + thresh, dilated by `dilate` LR cells, max-pooled to bricks).
  3. net graph (one per bucketed active-brick count): gather LR bricks with halo, sample base and the
     warped previous HR output at the active bricks' HR voxels only (sparse warp), run VolumeNet on the
     batch of bricks, scatter the core predictions into the canvas.
     Above `dense_above` occupancy the net runs once on the whole volume (no halo overhead) instead.
  4. the mask is copied async to pinned memory; a later frame uses the newest COMPLETED mask
     (<= max_lag frames old, else dense) — the 2D Upscaler's scheme.
Two HR canvases alternate (output of frame t is the "prev" of frame t+1): no 256^3 copy per frame.
`infer(state, active=ids)` forces a brick set (bench / sandbox).
"""
from __future__ import annotations

import bisect

import numpy as np
import torch
import torch.nn.functional as F

from vfx.core.fields import AMBIENT_TEMP, DOMAIN_SIZE, SIM_DT
from vfx.core.fields3d import CH3, LR_RES_3D, NUM_CHANNELS_3D, FieldState3D, Volume
from vfx.model3d import VolumeNet, VolumeNetConfig, load_volume_net, param_count
from vfx.runtime3d.warp3d import BrickGrid, clamp_dt, sample_bricks, trilinear_up, warp_prev_3d

_DT = slice(CH3["density"], CH3["temperature"] + 1, CH3["temperature"] - CH3["density"])
_VEL = slice(CH3["vel_x"], CH3["vel_z"] + 1)

# Defaults = the config we would train next (README: ~1.1 ms at 10% / ~2.1 ms at 25% occupancy).
DEFAULT_ARCH = dict(width=32, n_blocks=4, block="full", hr_feats=4, hr_hidden=8, hr_kernel=1, scale=4)


def cudnn_tuned(tune: bool = True):
    """cudnn autotuning scoped to our own capture/eager calls (no global side effect on the caller's code).
    Autotune results are cached per exact conv shape (~1 s each for the 3D trunk), so only a fixed, small
    set of shapes may ever be tuned; tune=False uses cudnn's heuristic choice (instant)."""
    return torch.backends.cudnn.flags(enabled=True, benchmark=tune, deterministic=False, allow_tf32=True)


def capture_graph(fn, device, pool, warm: int = 2, tune: bool = True):
    """Warm up fn() on a side stream (cudnn autotune if `tune`; fn may override per call) and capture it.
    Returns (graph, outputs)."""
    s = torch.cuda.Stream(device)
    s.wait_stream(torch.cuda.current_stream(device))
    with cudnn_tuned(tune):
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(warm):
                fn()
        torch.cuda.current_stream(device).wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g, pool=pool):
            out = fn()
    return g, out


def bucket_sizes(lo: int, hi: int, step: int) -> list[int]:
    """Batch capacities: powers of two from lo up to `step`, then multiples of `step`, capped at hi (and hi itself).
    Padding waste is < 2x for tiny batches (cheap anyway) and < `step` bricks above."""
    out, n = [], lo
    while n < min(step, hi):
        out.append(n)
        n *= 2
    n = step
    while n < hi:
        out.append(n)
        n += step
    return sorted(set(out + [hi]))


class VolumeUpscaler:
    def __init__(self, checkpoint_path=None, device="cuda", *, lr_res: int = LR_RES_3D, brick: int = 8,
                 halo: int = 2, sparse: bool = True, dense_above: float = 0.5, cuda_graph: bool = True,
                 channels_last: bool = True, density_thresh: float = 1e-3, temp_thresh: float = 1.0,
                 dilate: int = 2, bucket: int = 32, min_bucket: int = 8, max_lag: int = 2, fused: bool = True,
                 chunk: int = 64, tune_min: int = 64, prewarm: bool = True, **arch):
        self.device = torch.device(device)
        if checkpoint_path:
            self.model = load_volume_net(checkpoint_path, device)
            self.untrained = False
        else:
            self.model = VolumeNet(VolumeNetConfig(**{**DEFAULT_ARCH, **arch})).to(self.device)
            self.untrained = True
        self.cl = channels_last
        fmt = torch.channels_last_3d if channels_last else torch.contiguous_format
        self.model = self.model.half().to(memory_format=fmt).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.params = param_count(self.model)
        self.fused = None
        if fused and self.model.cfg.hr_kernel == 1 and self.device.type == "cuda":
            from vfx.runtime3d.fused import FusedHR
            self.fused = FusedHR(self.model, self.device)
        self.s = self.model.cfg.scale
        self.R, self.N = lr_res, lr_res * self.s
        self.b, self.halo, self.sb = brick, halo, brick * self.s
        self.nb = lr_res // brick
        self.n_bricks = self.nb ** 3
        self.sparse, self.dense_above, self.bucket, self.max_lag = sparse, dense_above, bucket, max_lag
        self.max_sparse = min(self.n_bricks, int(dense_above * self.n_bricks))   # more active -> dense pass
        self.chunk, self.tune_min = chunk, tune_min
        self.buckets = bucket_sizes(min(min_bucket, self.max_sparse), self.max_sparse, bucket) if self.max_sparse else []
        self.cuda_graph = cuda_graph and self.device.type == "cuda"
        self.density_thresh, self.temp_thresh, self.dilate = density_thresh, temp_thresh, dilate
        self.dx_hr = DOMAIN_SIZE / self.N
        self.grid = BrickGrid(self.N, self.sb, self.device)
        R, N, dev, h = self.R, self.N, self.device, torch.half
        self._lr = torch.zeros(1, NUM_CHANNELS_3D, R, R, R, device=dev, dtype=h)
        self._lr[:, CH3["temperature"]] = AMBIENT_TEMP
        self._samp = torch.zeros(1, 5, R, R, R, device=dev, dtype=h)        # d, T, vel of frame t-1
        self._samp[:, 1] = AMBIENT_TEMP
        self._hr = [torch.zeros(1, 2, N, N, N, device=dev, dtype=h) for _ in range(2)]
        for c in self._hr:
            c[:, 1] = AMBIENT_TEMP
        self._ids = torch.zeros(self.n_bricks, device=dev, dtype=torch.long)
        # dt lives on the device so the graphs do not depend on it (no re-capture when dt changes)
        self._dt = torch.zeros((), device=dev, dtype=torch.float32)
        self._vscale = torch.zeros(1, device=dev, dtype=torch.float32)     # dt / dx_hr (fused kernel arg)
        self._ids_host = [torch.zeros(self.n_bricks, dtype=torch.long, pin_memory=True) for _ in range(4)]
        self._ring = [{"host": torch.zeros((self.nb,) * 3, dtype=torch.uint8, pin_memory=True),
                       "event": torch.cuda.Event(), "frame": -10 ** 9} for _ in range(max_lag + 2)]
        self._pool = torch.cuda.graph_pool_handle() if self.cuda_graph else None
        self._graphs: dict = {}
        self._ev = {k: torch.cuda.Event(enable_timing=True) for k in ("start", "pre", "net", "end")}
        self.reset()
        if prewarm and self.cuda_graph:
            self.warmup()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self._cold = True
        self._frame = 0
        self._last_n = None
        for r in self._ring:
            r["frame"] = -10 ** 9

    # ------------------------------------------------------------------ stages (graph bodies)
    def _pre_fn(self):
        lr, h = self._lr, self.halo
        self._samp[:, 0:2].copy_(clamp_dt(lr[:, _DT]))
        lr_pad = F.pad(lr, (h,) * 6, mode="replicate") if h else lr
        canvas = trilinear_up(self._samp[:, 0:2], self.s)          # the HR output canvas (floor cost)
        act = ((lr[:, CH3["density"]:CH3["density"] + 1] > self.density_thresh)
               | (lr[:, CH3["temperature"]:CH3["temperature"] + 1] > AMBIENT_TEMP + self.temp_thresh)).half()
        if self.dilate:
            act = F.max_pool3d(act, 2 * self.dilate + 1, stride=1, padding=self.dilate)
        act = F.max_pool3d(act, self.b)[0, 0].to(torch.uint8)
        return lr_pad, act, canvas

    def _net_fn(self, cur: int, n: int, lr_pad):
        """Graph body: gather active LR bricks + sparse warp/base at their HR voxels, net, scatter."""
        b, h, sb, nb = self.b, self.halo, self.sb, self.nb
        fmt = torch.channels_last_3d if self.cl else torch.contiguous_format
        out, prev = self._hr[cur], self._hr[1 - cur]
        ids = self._ids[:n]
        bz, by, bx = ids // (nb * nb), (ids // nb) % nb, ids % nb
        S = b + 2 * h
        lr_b = lr_pad[0].unfold(1, S, b).unfold(2, S, b).unfold(3, S, b)[:, bz, by, bx]   # (6,n,S,S,S)
        lr_b = lr_b.transpose(0, 1).contiguous(memory_format=fmt)
        if self.fused is not None:     # Warp kernels for the HR stages (see fused.py)
            return self._fused_bricks(ids.int(), lr_b, out, prev, b, nb, h)
        base, pw = sample_bricks(self._samp, prev, self.grid.grid(ids), n, sb, self.dx_hr, self._dt)
        pred = self._chunked(lambda x, p, q: self.model(x, p, q, h), lr_b, pw.contiguous(memory_format=fmt),
                             base.contiguous(memory_format=fmt))
        ov = out[0].view(2, nb, sb, nb, sb, nb, sb)
        ov[:, bz, :, by, :, bx, :] = pred.to(out.dtype)
        return out

    def _fused_bricks(self, ids32, lr_b, out, prev, b, nb, h):
        fmt = torch.channels_last_3d if self.cl else torch.contiguous_format
        n, s = ids32.shape[0], self.s
        cells = n * b ** 3
        P = torch.empty(n, b, b, b, 2 * s ** 3, device=self.device, dtype=torch.half)
        self.fused.prev_unshuffle(ids32, self._samp[0], prev[0], P.view(cells, -1), nb, b, self.R, self.N,
                                  self._vscale)
        t = self._chunked(lambda x, p: self.model.trunk(x, p, h), lr_b,
                          P.permute(0, 4, 1, 2, 3).contiguous(memory_format=fmt))
        t = t.contiguous(memory_format=torch.channels_last_3d).permute(0, 2, 3, 4, 1).reshape(cells, -1)
        self.fused.head(ids32, self._samp[0], t, P.view(cells, -1), out[0], nb, b, self.R)
        return out

    def _chunked(self, f, *xs):
        """Run the cuDNN part on fixed-size batch chunks, so cuDNN only ever sees batch sizes
        {min_bucket .. chunk} (whatever the bucket) or the dense volume. Autotune only inputs at least as
        large as tune_min bricks (the full chunk and dense: tuned once each, ~1 s total); smaller
        batches use cuDNN's heuristic pick, which measured as fast there."""
        n, c = xs[0].shape[0], self.chunk
        tune_numel = self.tune_min * xs[0][0].numel() if xs[0].shape[-1] == self.b + 2 * self.halo else 0
        outs = []
        for i in range(0, n, c):
            part = [x[i:i + c] for x in xs]
            with cudnn_tuned(part[0].numel() >= tune_numel):
                outs.append(f(*part))
        return outs[0] if len(outs) == 1 else torch.cat(outs)

    def _dense_fn(self, cur: int):
        fmt = torch.channels_last_3d if self.cl else torch.contiguous_format
        out, prev = self._hr[cur], self._hr[1 - cur]
        if self.fused is not None:     # the whole volume as one brick, no halo
            ids32 = torch.zeros(1, device=self.device, dtype=torch.int32)
            return self._fused_bricks(ids32, self._lr.contiguous(memory_format=fmt), out, prev, self.R, 1, 0)
        pw = warp_prev_3d(prev, self._samp[:, 2:5], self.dx_hr, self._dt)
        pred = self.model(self._lr.contiguous(memory_format=fmt), pw.contiguous(memory_format=fmt),
                          out.contiguous(memory_format=fmt), 0)
        out.copy_(pred)
        return out

    def _run(self, key, fn):
        if not self.cuda_graph:
            with torch.inference_mode(), cudnn_tuned():
                return fn()
        g = self._graphs.get(key)
        if g is None:
            saved = [c.clone() for c in self._hr]
            g = self._graphs[key] = capture_graph(fn, self.device, self._pool)
            for c, s in zip(self._hr, saved):
                c.copy_(s)
        g[0].replay()
        return g[1]

    # ------------------------------------------------------------------ bricks (host side, no sync)
    def _mask_ids(self):
        """Newest completed lagged mask -> np ids; None = run dense."""
        for r in sorted(self._ring, key=lambda r: -r["frame"]):
            if self._frame - r["frame"] > self.max_lag:
                return None
            if r["event"].query():
                return np.flatnonzero(r["host"].numpy().reshape(-1))
        return None

    def _upload(self, ids: np.ndarray) -> int:
        n = len(ids)
        nq = self.buckets[bisect.bisect_left(self.buckets, n)]      # fixed capacity >= n (few graphs total)
        buf = self._ids_host[self._frame % len(self._ids_host)]
        buf[:n] = torch.from_numpy(np.asarray(ids, dtype=np.int64))
        buf[n:nq] = int(ids[-1])                         # pad with a duplicate: same value scattered twice
        self._ids[:nq].copy_(buf[:nq], non_blocking=True)
        return nq

    # ------------------------------------------------------------------ main
    @torch.inference_mode()
    def infer(self, state: FieldState3D, active=None) -> Volume:
        """LR state -> HR Volume (fp16, N^3). active: optional brick ids (bench/sandbox) overriding the mask."""
        ev, cur = self._ev, self._frame % 2
        ev["start"].record()
        self._lr.copy_(state.data.unsqueeze(0))
        dt = float(state.meta.dt)
        self._dt.fill_(dt)
        self._vscale.fill_(dt / self.dx_hr)
        if self.cuda_graph and ("pre", 0) not in self._graphs:
            for p in (0, 1):            # the two canvases are the outputs of the two pre graphs
                self._run(("pre", p), self._pre_fn)
            self._hr = [self._graphs[("pre", p)][1][2] for p in (0, 1)]
        lr_pad, act, canvas = self._run(("pre", cur), self._pre_fn)
        self._hr[cur] = canvas
        if self._cold:
            self._hr[1 - cur].copy_(self._hr[cur])           # first frame: prev := trilinear base
        ev["pre"].record()
        if active is not None:
            ids = np.asarray(active.cpu() if torch.is_tensor(active) else active).reshape(-1)
        elif self.sparse:
            ids = self._mask_ids()
        else:
            ids = None
        if ids is None or len(ids) > self.max_sparse:
            out = self._run(("dense", cur), lambda: self._dense_fn(cur))
            self._last_n = self.n_bricks
        elif len(ids) == 0:
            out = self._hr[cur]
            self._last_n = 0
        else:
            nq = self._upload(ids)
            out = self._run(("net", cur, nq), lambda: self._net_fn(cur, nq, lr_pad))
            self._last_n = nq
        ev["net"].record()
        slot = self._ring[self._frame % len(self._ring)]
        slot["host"].copy_(act, non_blocking=True)
        slot["event"].record()
        slot["frame"] = self._frame
        self._samp[:, 2:5].copy_(self._lr[:, _VEL])         # velocity that warps this output next frame
        self._cold = False
        self._frame += 1
        ev["end"].record()
        return Volume(density=out[0, 0], temperature=out[0, 1])

    def warmup(self, dt: float = SIM_DT, counts=None) -> None:
        """Capture every graph (pre, dense, one net graph per bucket capacity, x2 canvases). Called by
        the constructor (prewarm=True), so live use never captures. Graphs do not depend on dt."""
        counts = list(counts) if counts is not None else self.buckets
        st = _dummy_state(self, dt)
        sparse, self.sparse = self.sparse, False
        for _ in range(2):
            self.infer(st)                        # dense graphs (also used before the first mask arrives)
        self.sparse = sparse
        for n in counts:
            for _ in range(2):
                self.infer(st, active=np.arange(n))
        torch.cuda.synchronize(self.device)
        self.reset()

    def load_state_dict(self, sd) -> None:
        """Load VolumeNet weights after construction. Graph-safe: copies into the existing parameter and
        fused-head buffers, so the graphs captured at construction see the new weights."""
        self.model.load_state_dict(sd)
        self.refresh_weights()

    def refresh_weights(self) -> None:
        """Call after changing self.model's parameters in place (the fused HR head keeps an fp32 copy)."""
        if self.fused is not None:
            self.fused.refresh(self.model)

    @property
    def timings(self) -> dict:
        """ms of the last infer: pre (cast + trilinear base + mask), net (gather + sparse warp + net +
        scatter), post, total; processed_fraction of bricks. Waits for the last frame: HUD only."""
        ev = self._ev
        ev["end"].synchronize()
        return {"pre": ev["start"].elapsed_time(ev["pre"]), "net": ev["pre"].elapsed_time(ev["net"]),
                "post": ev["net"].elapsed_time(ev["end"]), "total": ev["start"].elapsed_time(ev["end"]),
                "processed_fraction": (self._last_n or 0) / self.n_bricks}


def _dummy_state(up: VolumeUpscaler, dt: float) -> FieldState3D:
    from vfx.core.fields3d import FieldMeta3D
    d = torch.zeros(NUM_CHANNELS_3D, up.R, up.R, up.R, device=up.device)
    d[CH3["temperature"]] = AMBIENT_TEMP
    return FieldState3D(d, torch.zeros(up.R, up.R, up.R, device=up.device), FieldMeta3D.for_res(up.R, dt))
