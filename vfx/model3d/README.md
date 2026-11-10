# vfx/model3d — 3D-phase networks (owner: 3D ML/runtime engineer)

Untrained in this phase: these exist so the runtime can be timed and the sandbox can run them.
Both nets zero-initialise their last conv, so an untrained net returns exactly its "classical" input
(the trilinear upscale for route A, the input frame for route B).

- `VolumeNet(VolumeNetConfig)` — route A, 3D version of the 2D `DetailNet` recipe on bricks.
  `forward(lr, prev, base, halo)`: `lr (n,6,S,S,S)` physical LR bricks with `halo` cells of context,
  `prev`/`base (n,2,sb,sb,sb)` warped previous HR output / trilinear base for the brick core; returns
  `(n,2,sb,sb,sb)` density, T(K). The whole 64^3 volume is the special case n=1, halo=0.
  `trunk(...)` is the LR part, shared by the PyTorch forward and the fused runtime path
  (`vfx/runtime3d/fused.py`). Block kinds: `full` 3x3x3, `sep` (2+1)D, `mp` 2D convs over slices of
  a cycling axis (SPEC "multi-pass" idea), `dw` depthwise+pointwise. `pixel_shuffle3d` / `pixel_unshuffle3d`.
- `ScreenNet(ScreenNetConfig)` — route B. `forward(rgb, aux, prev) -> Δrgb`; rgb/prev in [0,1],
  aux = transmittance, depth, screen velocity (px/frame / 16). Trunk at 1/`trunk_stride` resolution via
  PixelUnshuffle, PixelShuffle back, light 1x1 HR head.
- `save_*/load_*` store `{"config", "state_dict"}` like the 2D checkpoints.
