/* nvfx_render.h - native volume ray-marcher (port of vfx/render3d VolumeRenderer).  OWNER: native render engineer.
 *
 * PROPOSED addition to include/nvfx.h (the lead merges it). Everything here is implemented in native/render.
 *
 * Per draw, exactly VolumeRenderer.draw(volume, camera, solid, reuse_volume, return_aux=..., velocity, prev_camera)
 * with volume = (density, temperature), velocity and solid read straight from the solver's textures on the GPU:
 *   volume prep (skipped with NVFX_RENDER_REUSE_VOLUME): pack to the half-float volume + 32^3 occupancy bricks,
 *     blurred solid mask, light volume (light_res^3), fire-glow pyramid (32^3 + 8^3);
 *   view: march at round(march_scale * output) pixels (+ aux sums), bloom, composite at output res (+ aux resolve).
 * One command list per draw, prebuilt pipelines, no CPU round-trip of volume or image data, no host sync.
 */
#ifndef NVFX_RENDER_H
#define NVFX_RENDER_H

#include "nvfx.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NvfxRenderer NvfxRenderer;

/* vfx.core.fields3d.Camera. World space, the domain cube spans [0, 1]^3, y up. */
typedef struct NvfxCamera {
    float position[3];   /* (0.5, 0.55, -1.4) */
    float target[3];     /* (0.5, 0.45, 0.5)  */
    float up[3];         /* (0, 1, 0)         */
    float fov_y_deg;     /* 40                */
} NvfxCamera;

/* vfx.render3d.RenderParams3D, field for field in the same order, same defaults (nvfx_render_params_default).
 * `floor_opacity` is RenderParams3D.floor. light_dir is normalized by the renderer (it points TOWARD the light). */
typedef struct NvfxRenderParams {
    float   steps;            /* 192   ray-march samples per metre */
    int32_t light_res;        /* 80    light volume resolution (changing it synchronizes once, see set_params) */
    float   k_abs;            /* 40    */
    float   k_shadow;         /* 120   */
    float   k_solid;          /* 60    */
    float   k_amb;            /* 14    */
    float   k_emit;           /* 3     */
    float   emit_min;         /* 0.35  */
    float   emit_scale;       /* 28    */
    float   soot;             /* 0.9   */
    float   fire_light;       /* 0.9   */
    float   phase_g;          /* 0.6   */
    float   rim;              /* 0.75  */
    float   ms_a;             /* 0.3   */
    float   ms_b;             /* 0.5   */
    float   ms_c;             /* 0.5   */
    float   exposure;         /* 1.0   */
    float   contrast;         /* 1.15  */
    float   bloom;            /* 0.7   */
    float   bloom_thresh;     /* 0.8   */
    float   t_lo;             /* 700   */
    float   t_hi;             /* 3000  */
    float   t_ref;            /* 1400  */
    float   d_eps;            /* 2e-3  */
    float   light_dir[3];     /* (-0.80, 0.50, -0.20) */
    float   key_col[3];       /* (1.35, 1.25, 1.08)   */
    float   amb_col[3];       /* (0.05, 0.065, 0.10)  */
    float   smoke_albedo[3];  /* (0.68, 0.68, 0.72)   */
    float   bg_top[3];        /* (0.040, 0.048, 0.070) */
    float   bg_bot[3];        /* (0.008, 0.008, 0.012) */
    float   floor_col[3];     /* (0.030, 0.031, 0.036) */
    float   grid_col[3];      /* (0.030, 0.034, 0.042) */
    float   obst_col[3];      /* (0.16, 0.155, 0.15)   */
    float   glow_coarse;      /* 2.0   */
    float   floor_opacity;    /* 1.0   (RenderParams3D.floor; 0 = no floor) */
} NvfxRenderParams;

/* draw flags */
enum {
    NVFX_RENDER_AUX          = 1,  /* also write ALPHA / DEPTH / MOTION (Python return_aux=True). Colour is
                                      bit-identical with and without it. */
    NVFX_RENDER_REUSE_VOLUME = 2,  /* camera-only redraw of the volume prepared by this renderer's last full draw
                                      (Python reuse_volume=True): skips pack, light volume and glow. Ignored (a full
                                      draw is done) if there was no full draw yet, or the solver resolution or the
                                      NO_SOLID flag differ from it. The caller guarantees the solver state did not
                                      change since that draw. */
    NVFX_RENDER_NO_SOLID     = 4,  /* ignore the solver's solid mask (Python solid=None) */
    NVFX_RENDER_NO_VELOCITY  = 8   /* motion from the camera change only (Python velocity=None) */
};

/* output textures: ID3D12Resource* Texture2D, output width x height, 1 mip, not AddRef'd, owned by the renderer,
 * left in D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE between draws, row 0 = top of the screen.
 *   COLOR   R8G8B8A8_UNORM  final display-encoded image (Python's uint8 frame), A = 255
 *   ALPHA   R16_FLOAT       1 - transmittance of the smoke/fire medium (vfx/render3d README "Aux buffers")
 *   DEPTH   R16_FLOAT       opacity-weighted mean distance from the camera along the ray, metres; 0 where alpha < 1e-3
 *   MOTION  R16G16_FLOAT    output pixels per SIM_DT, x right, y down (content at p was at p - motion one sim step
 *                           ago); 0 where alpha < 1e-3
 * Aux textures hold the last AUX draw's values. The pointers are fixed for the renderer's lifetime. */
typedef enum NvfxRenderTexture {
    NVFX_RENDER_TEX_COLOR = 0, NVFX_RENDER_TEX_ALPHA = 1, NVFX_RENDER_TEX_DEPTH = 2, NVFX_RENDER_TEX_MOTION = 3
} NvfxRenderTexture;

/* GPU ms (D3D12 timestamp queries in the draw's own command list) of the newest draw that has finished on the
 * GPU. Reading never blocks; values lag the newest draw by a frame or two. All 0 until a draw has finished.
 *   prep_ms   volume prep (pack, solid blur, occupancy, light volume, glow); ~0 for a REUSE_VOLUME draw
 *   march_ms  the ray march (+ aux sums)
 *   post_ms   bloom + composite (+ aux resolve)
 *   total_ms  first to last timestamp of the draw */
typedef struct NvfxRenderTimings {
    float    prep_ms, march_ms, post_ms, total_ms;
    int32_t  flags;          /* the flags the draw actually ran with (REUSE_VOLUME cleared if it fell back) */
    int32_t  reserved;
    uint64_t draw_index;     /* draws since create, first = 1; 0 = nothing finished yet */
} NvfxRenderTimings;

NVFX_API void       nvfx_camera_default(NvfxCamera* cam);               /* fields3d.Camera() */
NVFX_API void       nvfx_render_params_default(NvfxRenderParams* p);    /* RenderParams3D() */

/* VolumeRenderer((height, width), march_scale=march_scale) with default params. The march buffer is
 * max(8, round(height * scale)) x max(8, round(width * scale)). Creates pipelines and all view-sized resources;
 * volume-sized resources are created by the first draw (and recreated, synchronizing, if the solver resolution
 * changes). Synchronizes. */
NVFX_API NvfxResult nvfx_renderer_create(NvfxContext* ctx, int32_t width, int32_t height, float march_scale,
                                         NvfxRenderer** out);
NVFX_API void       nvfx_renderer_destroy(NvfxRenderer* r);             /* waits for the context queue */
/* Takes effect at the next draw. Changing light_res recreates the light volume and waits for the GPU once.
 * A change of t_lo / t_hi / t_ref rebuilds the blackbody LUT (Python builds it in __init__ only: the result equals
 * a VolumeRenderer constructed with these params). A REUSE_VOLUME draw after a change of a prep parameter
 * (light_res, k_shadow, k_solid, k_amb, k_emit, emit_*, t_*, d_eps, light_dir, glow_coarse) shows the old volume
 * prep, as in Python. */
NVFX_API NvfxResult nvfx_renderer_set_params(NvfxRenderer* r, const NvfxRenderParams* p);
NVFX_API NvfxResult nvfx_renderer_get_params(NvfxRenderer* r, NvfxRenderParams* out);
/* march buffer size in pixels */
NVFX_API void       nvfx_renderer_march_size(NvfxRenderer* r, int32_t* width, int32_t* height);

/* Renders the solver's current state (its textures as they are after the last recorded step) with `camera`.
 * prev_camera = the camera one sim step (SIM_DT) ago, for MOTION; NULL = same camera. flags = NVFX_RENDER_*.
 * Records one command list and submits it to the context queue; never waits for the GPU unless 16 draws are
 * still in flight. */
NVFX_API NvfxResult nvfx_renderer_draw(NvfxRenderer* r, NvfxSolver* solver, const NvfxCamera* camera,
                                       const NvfxCamera* prev_camera, int32_t flags);
/* Hosted path: records the same draw into the host's ID3D12GraphicsCommandList* (direct or compute) instead of
 * submitting. It sets its own descriptor heap, compute root signature and pipeline state: the host re-binds its
 * own afterwards. The list must execute on the context's queue, after the solver's step for this frame. Recorded
 * draws must be executed: at most 16 may be outstanding (the 17th waits up to 20 s for the GPU, then fails). */
NVFX_API NvfxResult nvfx_renderer_record_draw(NvfxRenderer* r, void* d3d12_command_list, NvfxSolver* solver,
                                              const NvfxCamera* camera, const NvfxCamera* prev_camera, int32_t flags);

NVFX_API void*      nvfx_renderer_texture(NvfxRenderer* r, NvfxRenderTexture which);
NVFX_API NvfxResult nvfx_renderer_get_timings(NvfxRenderer* r, NvfxRenderTimings* out);

/* For tests and tools. Synchronizes (waits for every submitted draw). Layouts, row 0 = top:
 *   COLOR   height*width*4 uint8, row-major RGBA (A = 255): out[:, :, :3] is Python's uint8 frame
 *   ALPHA / DEPTH  height*width float32 (the fp16 values, widened)
 *   MOTION  2*height*width float32, planar: x plane then y plane (Python aux["motion"] (2,h,w)) */
NVFX_API NvfxResult nvfx_renderer_download(NvfxRenderer* r, NvfxRenderTexture which, void* out);

/* Effect bbox for the route B detail pass: the projected domain cube [0,1]^3 in output pixels, exactly as
 * vfx/app3d/pipeline.py _screen_aux computes it (x0, y0 floored, x1, y1 ceiled, exclusive end, clipped to the frame;
 * the whole frame when a cube corner is behind or within 1e-3 m of the camera plane). Pure CPU, no GPU sync, valid
 * before the draw is recorded. */
NVFX_API void       nvfx_render_screen_bbox(const NvfxCamera* camera, int32_t width, int32_t height,
                                            int32_t out_bbox[4]);

#ifdef __cplusplus
}
#endif
#endif /* NVFX_RENDER_H */
