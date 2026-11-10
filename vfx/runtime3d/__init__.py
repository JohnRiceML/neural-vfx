"""3D-phase runtime (owner: 3D ML/runtime engineer). Route A VolumeUpscaler (untrained), route B ScreenDetail
(trained: loads default_screen_checkpoint() when present, else identity), classical 3D upscale baseline."""
from vfx.runtime3d.warp3d import classical_upscale_3d, sample_bricks, upscale_lr_tensor_3d, warp_prev_3d
from vfx.runtime3d.volume_upscaler import VolumeUpscaler
from vfx.runtime3d.screen_detail import ScreenDetail, default_screen_checkpoint

__all__ = ["classical_upscale_3d", "warp_prev_3d", "sample_bricks", "upscale_lr_tensor_3d",
           "VolumeUpscaler", "ScreenDetail", "default_screen_checkpoint"]
