/* Shared between C++ (render.cpp) and HLSL (shaders/ *.hlsl): descriptor-heap slots of the renderer.
 * Every shader reaches its resources through ResourceDescriptorHeap[R_*] (SM 6.6 dynamic resources):
 * one shader-visible heap per renderer, bound once per command list, no descriptor tables.
 * OWNER: native render engineer. */
#ifndef NVFX_RENDER_SLOTS_H
#define NVFX_RENDER_SLOTS_H

/* internal 3D textures (NON_PIXEL_SHADER_RESOURCE between draws, UNORDERED_ACCESS while prep writes them) */
#define R_VOL_SRV      0    /* Texture3D<float2>  RG16F   (density, temperature) at volume res: Python's half2 texture */
#define R_VOL_UAV      1
#define R_VEL_SRV      2    /* Texture3D<float4>  RGBA16F (vel_x, vel_y, vel_z, 0): Python's velocity texture */
#define R_VEL_UAV      3
#define R_SOLID8_SRV   4    /* Texture3D<float>   R8_UNORM blurred solid mask */
#define R_SOLID8_UAV   5
#define R_LIGHT_SRV    6    /* Texture3D<float2>  RG16F   (transmittance to key light, sky occlusion), light_res^3 */
#define R_LIGHT_UAV    7
#define R_GLOW_SRV     8    /* Texture3D<float4>  RGBA16F fire glow 32^3 */
#define R_GLOW_UAV     9
/* internal buffers (always UNORDERED_ACCESS) */
#define R_S1          10    /* float  N^3: solid blur scratch (z pass) */
#define R_S2          11    /* float  N^3: solid blur scratch (y pass) */
#define R_OCCD        12    /* uint   32^3: density occupancy bricks */
#define R_OCCS        13    /* uint   32^3: solid occupancy bricks */
#define R_G0          14    /* float4 32^3: adaptive-pooled glow source */
#define R_G1          15    /* float4 32^3 */
#define R_G2          16    /* float4 32^3 */
#define R_G3          17    /* float4 32^3: blurred narrow glow (g32b) */
#define R_H0          18    /* float4 8^3: avg_pool(g32, 4) */
#define R_H1          19    /* float4 8^3 */
#define R_H2          20    /* float4 8^3 */
#define R_HDR         21    /* float4 mh*mw: premultiplied radiance + transmittance (Python self.hdr) */
#define R_AUXHDR      22    /* float4 mh*mw: (alpha, alpha*depth, alpha*mx, alpha*my) */
#define R_B1          23    /* float4 bh*bw: bloom level 1 (blurred pool) */
#define R_P           24    /* float4 bh2*bw2: avg_pool2d(b1, 4, ceil_mode) */
#define R_U2          25    /* float4 bh2*bw2: x-blurred P */
#define R_BLOOM       26    /* float4 bh*bw: bloom buffer (Python self.bloom_buf) */
#define R_LUT         27    /* StructuredBuffer<float4> 256: blackbody LUT */
/* public output textures (NON_PIXEL_SHADER_RESOURCE between draws) */
#define R_COLOR_UAV   28    /* RWTexture2D<unorm float4> R8G8B8A8_UNORM */
#define R_ALPHA_UAV   29    /* RWTexture2D<float>  R16_FLOAT */
#define R_DEPTH_UAV   30    /* RWTexture2D<float>  R16_FLOAT */
#define R_MOTION_UAV  31    /* RWTexture2D<float2> R16G16_FLOAT */
/* per draw-ring slot k: SRVs of the solver textures used by that draw (the solver ping-pongs, so
 * they are rewritten per draw, into a slot the GPU is no longer reading) */
#define R_IN0         32    /* + 3k + 0: VEL_DENSITY (RGBA32F), +1: TEMP_FUEL (RG32F), +2: SOLID (R32F) */
#define R_RING        16
#define R_COUNT       (R_IN0 + 3 * R_RING)

#define R_BRICKS      32    /* empty-space skipping grid (kernels.py BRICKS) */
#define R_GLOW        32    /* glow pyramid fine level */
#define R_GLOW_C      8     /* glow pyramid coarse level */
#define R_LUT_N       256

#endif
