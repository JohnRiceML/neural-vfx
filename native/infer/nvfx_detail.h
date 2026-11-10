/* nvfx_detail.h - route B screen-space detail (ScreenDetail) on D3D12 + DirectML.  OWNER: native inference engineer.
 *
 * PROPOSED addition to include/nvfx.h (the lead merges it). Everything here is implemented in native/infer.
 *
 * Per frame, exactly vfx/runtime3d/screen_detail.py ScreenDetail.infer():
 *   1. region: bbox grown per axis to the next of 6 fixed sizes (ScreenDetail.sizes: crop_fracs 0.25..1.0 of the frame
 *      side rounded up to the 64 px crop quantum) and shifted to stay inside the frame (whole axis if it does not fit).
 *   2. "pre" HLSL pass: crop, uint8 -> fp16, 2x2 average pool (rgb mean; alpha mean; depth / motion alpha-weighted
 *      means; motion x process_scale), aux normalisation, written straight into the D3D12 buffers the network reads.
 *   3. the network on DirectML, on the context's device, recorded into the same command list (default backend), or
 *      ONNX Runtime's DirectML EP on the context queue (NVFX_DETAIL_BACKEND_ORT). No CPU copy either way.
 *   4. "post" HLSL passes: grey_lowfreq + alpha_gate (the end of ScreenNet.forward), bilinear upsample of delta
 *      (torch F.interpolate align_corners=False), add, clamp, the same fp16 roundings and uint8 truncation as Python;
 *      pixels outside the region are the input frame.
 * Every crop bucket (36 at 2560x1440) is compiled in nvfx_detail_create: a moving bbox never hitches.
 */
#ifndef NVFX_DETAIL_H
#define NVFX_DETAIL_H

#include "nvfx.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NvfxDetail NvfxDetail;

typedef struct NvfxDetailDesc {
    const char* model_path;     /* UTF-8 path of screennet_fp16.onnx from native/infer/export_onnx.py */
    int32_t     width, height;  /* frame (output) size, pixels */
    float       process_scale;  /* 0.5 (what the shipped net was trained at) or 1.0 */
    int32_t     flags;          /* NVFX_DETAIL_* below; 0 = the shipping configuration */
} NvfxDetailDesc;

enum {
    /* backend selection */
    NVFX_DETAIL_BACKEND_ORT      = 1,   /* ONNX Runtime DML EP, one session per bucket (~2.2 GB VRAM at 1440p) */
    /* DirectML backend experiments (all off = shipping) */
    NVFX_DETAIL_DML_HALF_COMPUTE = 2,   /* DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION */
    NVFX_DETAIL_DML_NO_METACMD   = 4,   /* DML_EXECUTION_FLAG_DISABLE_META_COMMANDS */
    NVFX_DETAIL_DML_NHWC         = 8,   /* intermediate tensors NHWC strided instead of NCHW */
    /* ORT backend experiments */
    NVFX_DETAIL_ORT_NO_FUSION    = 16,  /* ep.dml.disable_graph_fusion */
    NVFX_DETAIL_LAZY_BUCKETS     = 32   /* build a bucket on first use (fast create, hitch on first use) */
};

/* Inputs / output of one frame. ID3D12Resource* Texture2D, width x height, 1 mip, created on the context's device.
 * Required states (on entry, left unchanged on exit):
 *   color  R8G8B8A8_UNORM  NON_PIXEL_SHADER_RESOURCE   the tonemapped frame (= Python's uint8 frame; A is passed through)
 *   alpha  R16_FLOAT       NON_PIXEL_SHADER_RESOURCE   renderer aux (vfx/render3d README "Aux buffers"); NULL = zeros
 *   depth  R16_FLOAT       NON_PIXEL_SHADER_RESOURCE   metres; NULL = zeros
 *   motion R16G16_FLOAT    NON_PIXEL_SHADER_RESOURCE   output px per SIM_DT (x right, y down); NULL = zeros
 *   output R8G8B8A8_UNORM  UNORDERED_ACCESS (ALLOW_UNORDERED_ACCESS)   must not alias color; fully written
 * bbox = effect bbox in pixels (x0, y0, x1, y1), exclusive end; x1 <= x0 or y1 <= y0 = whole frame. */
typedef struct NvfxDetailFrame {
    void*   color;
    void*   alpha;
    void*   depth;
    void*   motion;
    void*   output;
    int32_t bbox[4];
} NvfxDetailFrame;

/* GPU ms from D3D12 timestamp queries placed in the frame's own command list(s), of the newest frame that has
 * finished on the GPU; reading never blocks.
 *   pre_ms   crop + convert + pool + normalise + unshuffle (HLSL)
 *   net_ms   the network (DirectML dispatch; ORT backend: end of pre -> start of post, incl. submission gaps)
 *   post_ms  grey_lowfreq + alpha gate + upsample + add + round + copy of the pixels outside the region (HLSL)
 *   total_ms first -> last timestamp of the frame
 * cpu_ms = CPU wall time inside nvfx_detail_run / nvfx_detail_record for that frame. */
typedef struct NvfxDetailTimings {
    float    pre_ms, net_ms, post_ms, total_ms;
    float    cpu_ms;
    int32_t  region[4];     /* processed pixels: x0, y0, width, height (inside the frame) */
    int32_t  crop[2];       /* bucket: crop width, height in output pixels (>= region when the frame is smaller) */
    uint64_t frame_index;   /* 1 = first frame; 0 = nothing finished yet */
} NvfxDetailTimings;

/* Parses the ONNX weights, compiles the DirectML graph of every crop bucket (ORT backend: one session each),
 * creates the HLSL pipelines and buffers, uploads + initializes. DirectML.dll (and onnxruntime.dll for the ORT
 * backend) are loaded from the directory of nvfx.dll. Synchronizes; takes ~1-3 s at 1440p. */
NVFX_API NvfxResult nvfx_detail_create(NvfxContext* ctx, const NvfxDetailDesc* desc, NvfxDetail** out);
NVFX_API void       nvfx_detail_destroy(NvfxDetail* d);      /* waits for the context queue */

/* Stand-alone: records the frame into an internal command list and submits it to the context queue. Never waits
 * for the GPU unless 8 frames are still in flight. */
NVFX_API NvfxResult nvfx_detail_run(NvfxDetail* d, const NvfxDetailFrame* frame);

/* Hosted (DirectML backend only): records the whole frame (pre, network, post, timestamps) into the host's
 * ID3D12GraphicsCommandList* (direct or compute). It sets its own descriptor heap, root signature and pipeline state:
 * the host re-binds its own afterwards. The list must execute on the context's queue (the network's scratch memory is
 * shared between buckets, so detail frames must not run concurrently), and at most 8 recorded frames may be
 * outstanding. Returns NVFX_ERR_UNSUPPORTED with the ORT backend (ORT submits its own command lists). */
NVFX_API NvfxResult nvfx_detail_record(NvfxDetail* d, void* d3d12_command_list, const NvfxDetailFrame* frame);

NVFX_API NvfxResult nvfx_detail_get_timings(NvfxDetail* d, NvfxDetailTimings* out);

/* The region a frame with this bbox processes: x0, y0, w, h, crop_w, crop_h. Pure CPU. */
NVFX_API NvfxResult nvfx_detail_region(NvfxDetail* d, const int32_t bbox[4], int32_t out_region[6]);

/* GPU memory this object owns (buffers, DirectML persistent + shared temporary resources), bytes. ORT backend:
 * buffers only (ORT's own allocations are not visible; measure with DXGI QueryVideoMemoryInfo). */
NVFX_API uint64_t   nvfx_detail_vram_bytes(NvfxDetail* d);

#ifdef __cplusplus
}
#endif
#endif /* NVFX_DETAIL_H */
