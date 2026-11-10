/* Shared between C++ (solver.cpp) and HLSL: descriptor-heap slots and scalar-buffer slots.
 * Every shader reaches its resources through ResourceDescriptorHeap[SL_*] (SM 6.6 dynamic resources):
 * one shader-visible heap per solver, bound once per command list, no descriptor tables. */
#ifndef NVFX_SIM_SLOTS_H
#define NVFX_SIM_SLOTS_H

/* public state textures (two ping-pong sets: +0 / +1) */
#define SL_VD_SRV0     0    /* Texture3D<float4> vel_x, vel_y, vel_z, density */
#define SL_TF_SRV0     2    /* Texture3D<float2> temperature, fuel */
#define SL_SOLID_SRV   4    /* Texture3D<float>  solid */
#define SL_VD_UAV0     5
#define SL_TF_UAV0     7
/* internal textures (always UNORDERED_ACCESS) */
#define SL_HAT_VD     10
#define SL_HAT_TF     11
#define SL_ADV_VD     12
#define SL_ADV_TF     13
#define SL_OMEGA      14    /* float4: curl (xyz) */
#define SL_OMEGA_S    15
#define SL_FORCE      16
#define SL_TA         17    /* float4: separable smoothing scratch (3 channels + weight) */
#define SL_TB         18
/* pressure buffers (RWStructuredBuffer, flat index (z*N + y)*N + x) */
#define SL_FM         20    /* int: open-face mask */
#define SL_PR         21
#define SL_PZ         22
#define SL_PP         23
#define SL_PQ         24
#define SL_PHI        25
#define SL_Z0         26
#define SL_RR         27
#define SL_ZT         28
#define SL_PART       29    /* 2*NB partials (sum | max) */
#define SL_PART_PQ    30
#define SL_MPART      31    /* 2*NBM mass partials */
#define SL_SCAL       32    /* RWByteAddressBuffer, S_* words */
#define SL_LX         33    /* multigrid level arrays, all levels in one buffer each */
#define SL_LR         34
#define SL_LD         35
#define SL_LO         36
#define SL_LEVELS     37    /* StructuredBuffer<Level> */
#define SL_PP2        39    /* fused path: p is ping-ponged between PCG iterations */
#define SL_PRED0      64    /* predicate buffer k (k = 1..NVFX_MAX_PRED) at SL_PRED0 + k */
#define NVFX_MAX_PRED 256
#define SL_COUNT      (SL_PRED0 + NVFX_MAX_PRED + 1)

/* scalar slots (32-bit words in SL_SCAL) */
#define S_MASS_RATIO 0
#define S_MAXU       1
#define S_RZ         2
#define S_ALPHA      3
#define S_BETA       4
#define S_MAXR       5
#define S_THR        6
#define S_BEST       7
#define S_IT         8      /* uint */
/* fused path: per-iteration CG scalars live in two banks (read bank j-1, write bank j) so every
 * group of p_cg_update can read the old values while group 0 writes the new ones */
#define S_BANK(b, s) (16 * (1 + (b)) + (s))
#define S_WORDS      48

#define NVFX_MG_OMEGA 0.8f  /* outer damped Jacobi on the exact K (pressure.py OMEGA) */
#define NVFX_CB       512   /* threads of the single-group coarse V-cycle kernel */

#endif
