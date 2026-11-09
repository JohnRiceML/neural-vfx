# vfx/runtime — inference (owner: ML/runtime engineer)

- `warp_prev(prev_hr, lr_velocity, dx_hr, dt)` — semi-Lagrangian: `out(x) = prev(x - u(x) dt)` with the
  LR velocity bilinearly upsampled, bilinear sampling, border clamp. Used by training AND runtime.
- `classical_upscale(lr_state, mode="bicubic"|"bilinear"|"nearest")` — baseline / fallback.
- `Upscaler(checkpoint_path=None)` — raises `FileNotFoundError` if no checkpoint
  (default `paths.default_checkpoint()`); the app should then fall back to `classical_upscale`.
  - `infer(lr_state) -> HRFields` (fp16, 512^2). Keeps the previous output as temporal state; after
    `reset()` the first frame uses the bicubic upsample as "previous".
  - Sparse bricks: 16x16-LR-cell bricks, activity (density > 1e-3 or T > ambient+1 K) dilated by one brick,
    computed on GPU and copied async to pinned memory; the next frame crops the network to the active
    bounding box + 8-cell halo (sizes quantized to 32 LR cells) and fills the rest with bicubic.
    No host sync: the mask is used only once its copy event has completed (`event.query()`).
    Uses the newest completed mask up to `max_lag=2` frames old, otherwise runs the full frame.
  - FP16, channels_last, cudnn.benchmark. CUDA graphs: one for the fixed-shape pre-stage (cast, bicubic,
    warp, brick mask) and one per crop shape for the network. Call `warmup()` once to capture all of them.
  - `timings` — ms of the last infer from CUDA events: warp (pre-stage) / net (+compose) / post / total, plus
    `processed_fraction`. Reading it waits for the last frame; read it in the HUD, not mid-frame.
- `bench_selfcheck.py` — `.venv/Scripts/python -m vfx.runtime.bench_selfcheck`.
