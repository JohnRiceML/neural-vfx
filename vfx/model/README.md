# vfx/model — DetailNet (owner: ML/runtime engineer)

`DetailNet.forward(lr (B,5,h,w), prev_hr_warped (B,2,4h,4w)) -> (B,2,4h,4w)` — physical units in/out
(density, temperature K). Normalization is inside: density/1, velocity/`vel_scale` (set from the
training data), (T-293)/1000, fuel/1. Output: density >= 0, T >= ambient.

Architecture (residual on a smooth upsample; heavy work at LR):
1. LR trunk @128^2: concat[normalized LR (5), PixelUnshuffle(4)(prev warped) (32)] -> conv3x3 64 ->
   8 residual blocks (64 ch) -> conv3x3 -> 128 ch -> PixelShuffle(4) -> 8 ch @512^2.
2. HR head @512^2: concat[8 feats, prev warped (2), bicubic base (2)] -> conv3x3 16 -> ReLU -> conv3x3 2.
3. out = bicubic(lr density, T) + delta (last conv zero-initialised so training starts at bicubic).

fp16 stability: residual branches scaled by 0.1 (EDSR), normalized trunk inputs clipped to +-20, outputs to [0, 50].

Default size: 688k params. `load_model(path)` / `save_model(model, path)` store config + weights.
