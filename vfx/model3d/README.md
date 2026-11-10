# vfx/model3d — 3D-phase networks (owner: 3D ML/runtime engineer)

Route A (VolumeNet) is untrained; route B (ScreenNet) is trained by vfx/train3d.
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
  aux = alpha, depth (m), motion (processed px / SIM_DT) in physical units, normalized inside the model
  (`depth_scale`, `motion_scale`, clip `input_clip`). Trunk at 1/`trunk_stride` resolution via PixelUnshuffle,
  PixelShuffle back, HR head (k×k). Output constraints in the forward (config, saved with the checkpoint):
  `grey_lowfreq` k = the part of Δrgb coarser than k px is made grey; `alpha_gate` g = |Δrgb| ≤ g·alpha.
  Trained config: 64 wide × 8 blocks, stride 4, HR head 3×3 8→16→3, no prev, grey_lowfreq 8, alpha_gate 0.5 (vfx/train3d).
- `save_*/load_*` store `{"config", "state_dict"}` like the 2D checkpoints.
- Untrained-output check (2026-11-09): on the live campfire preset the sandbox's `neural_vol` output equals
  `classical_upscale_3d` to within fp16 rounding: |Δdensity| ≤ 2.4e-4, |ΔT| ≤ 1 K at 1870 K, over 200 steps. The
  rendered frame shows the same fire as trilinear (see `vfx/runtime3d/README.md`).
- The runtime captures its CUDA graphs at construction. To swap weights into a live `VolumeUpscaler`, use
  `VolumeUpscaler.load_state_dict`, not `up.model.load_state_dict`.
