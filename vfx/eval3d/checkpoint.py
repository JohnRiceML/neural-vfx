"""Where the route B (ScreenDetail) checkpoint lives (owner: 3D test/eval engineer).

Resolution order (first hit wins), mirroring how the sandbox would find it:
  1. env VFX_SCREEN_CKPT (explicit override; an empty value means "no checkpoint").
  2. a default exported by the owning modules, if any: vfx.runtime3d / vfx.train3d / vfx.app3d.backends
     attributes `default_screen_checkpoint()` (callable) or `DEFAULT_SCREEN_CHECKPOINT` (path).
  3. the default of `checkpoint_b` in vfx.app3d.backends.load_backends (the sandbox passes it to ScreenDetail).
  4. conventional paths under vfx.core.paths.CHECKPOINTS: screen3d/latest.pt, route_b/latest.pt, screen/latest.pt.
Returns (path or None, source string). A path is only returned if the file exists.
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path

CANDIDATE_DIRS = ("screen3d", "route_b", "routeb", "screen", "screen_detail")


def screen_checkpoint() -> tuple[Path | None, str]:
    env = os.environ.get("VFX_SCREEN_CKPT")
    if env is not None:
        if not env:
            return None, "VFX_SCREEN_CKPT set empty"
        p = Path(env)
        return (p, "VFX_SCREEN_CKPT") if p.exists() else (None, f"VFX_SCREEN_CKPT={env} does not exist")
    for modname in ("vfx.runtime3d", "vfx.train3d", "vfx.app3d.backends", "vfx.app3d"):
        try:
            mod = __import__(modname, fromlist=["_"])
        except Exception:  # noqa: BLE001
            continue
        for attr in ("default_screen_checkpoint", "DEFAULT_SCREEN_CHECKPOINT", "SCREEN_CHECKPOINT"):
            v = getattr(mod, attr, None)
            if v is None:
                continue
            try:
                v = v() if callable(v) else v
            except Exception:  # noqa: BLE001
                continue
            if v and Path(v).exists():
                return Path(v), f"{modname}.{attr}"
    try:
        from vfx.app3d.backends import load_backends

        d = inspect.signature(load_backends).parameters.get("checkpoint_b")
        if d is not None and d.default not in (None, inspect.Parameter.empty) and Path(d.default).exists():
            return Path(d.default), "vfx.app3d.backends.load_backends(checkpoint_b=...) default"
    except Exception:  # noqa: BLE001
        pass
    from vfx.core.paths import CHECKPOINTS

    for d in CANDIDATE_DIRS:
        p = CHECKPOINTS / d / "latest.pt"
        if p.exists():
            return p, f"conventional path {p}"
    return None, ("no route B checkpoint: sandbox passes checkpoint_b=None (untrained); looked for "
                  + ", ".join(str(CHECKPOINTS / d / "latest.pt") for d in CANDIDATE_DIRS))
