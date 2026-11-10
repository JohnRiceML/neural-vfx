"""3D-phase runtime (owner: 3D ML/runtime engineer). Route A VolumeUpscaler, route B ScreenDetail,
classical 3D upscale baseline. Untrained weights: outputs equal the classical upscale / input frame."""
from vfx.runtime3d.warp3d import classical_upscale_3d, sample_bricks, upscale_lr_tensor_3d, warp_prev_3d
from vfx.runtime3d.volume_upscaler import VolumeUpscaler
from vfx.runtime3d.screen_detail import ScreenDetail

__all__ = ["classical_upscale_3d", "warp_prev_3d", "sample_bricks", "upscale_lr_tensor_3d",
           "VolumeUpscaler", "ScreenDetail"]
