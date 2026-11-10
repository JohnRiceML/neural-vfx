"""Regression check for the per-draw camera upload (data engineer's bug: with a fixed ring of 4 pinned
buffers, draws queued more than 4 deep without a host sync rendered another draw's camera).

    python -m vfx.render3d.camera_queue_check

Queues N draws with N different cameras behind a GPU busy-wait (so the host runs far ahead), clones each
frame in stream order, then compares every frame with the same camera drawn alone after a sync.
Also checks AdvectedDetail ticks queued the same way against ticks with a sync between them.
"""
from __future__ import annotations

import hashlib
import math

import torch


def main(n: int = 40) -> bool:
    from vfx.core.fields3d import AMBIENT_TEMP, Camera, Volume
    from vfx.render3d import VolumeRenderer
    g = torch.linspace(0, 1, 64, device="cuda")
    z, y, x = torch.meshgrid(g, g, g, indexing="ij")
    d = 3.0 * torch.exp(-((x - 0.45) ** 2 + (y - 0.5) ** 2 * 0.5 + (z - 0.55) ** 2) / 0.02)
    vol = Volume(d, AMBIENT_TEMP + 900.0 * (d > 1.5).float())

    def cam(i):
        a = math.radians(9.0 * i)
        return Camera(position=(0.5 + 1.8 * math.sin(a), 0.6, 0.5 - 1.8 * math.cos(a)), target=(0.5, 0.45, 0.5))
    ok = True
    for temporal in (False,):
        r = VolumeRenderer((180, 320), march_scale=0.5, temporal=temporal)
        r.draw(vol, cam(0))
        torch.cuda.synchronize()
        torch.cuda._sleep(200_000_000)                # ~0.1 s of GPU busy-wait: the host queues all draws
        queued = [r.draw(vol, cam(i), reuse_volume=i > 0).clone() for i in range(n)]
        torch.cuda.synchronize()
        bad = 0
        for i in range(n):
            ref = r.draw(vol, cam(i)).clone()
            torch.cuda.synchronize()
            if not torch.equal(ref, queued[i]):
                bad += 1
        print(f"renderer: {n} queued draws, {bad} differ from the synced draw of the same camera; "
              f"pinned slots grown to {len(r._cam_ring.bufs)}, forced waits {r._cam_ring.waits}")
        ok &= bad == 0
    # AdvectedDetail per-tick params (same ring pattern)
    from vfx.core.fields3d import SIM_DT, Forces3D
    from vfx.detail3d import AdvectedDetail
    from vfx.sim3d import Solver3D, runtime_config
    from vfx.app3d.presets import PRESETS
    pre = PRESETS[1]
    outs = []
    for sync in (True, False):
        s = Solver3D(64, runtime_config(**pre.sim))
        st = s.reset(pre.obstacles)
        det = AdvectedDetail(64)
        f = Forces3D(emitters=list(pre.emitters))
        torch.cuda.synchronize()
        if not sync:
            torch.cuda._sleep(200_000_000)
        dyns = []
        for i in range(n):
            st = s.step(st, f, SIM_DT)
            det.step(st, SIM_DT)
            dyns.append(det.dyn.clone())
            if sync:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        outs.append(hashlib.sha256(torch.stack(dyns).cpu().numpy().tobytes()).hexdigest())
    print("detail: queued ticks identical to synced ticks:", outs[0] == outs[1])
    ok &= outs[0] == outs[1]
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
