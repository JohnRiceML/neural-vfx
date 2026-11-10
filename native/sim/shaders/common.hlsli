// Shared HLSL declarations for the sim3d port. Root signature (solver.cpp):
//   b0 = 16 root constants (k0..k14 per dispatch, n = grid size, set once per list)
//   b1 = root CBV with the per-step parameter block (kernels.py `prm`)
//   + CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED (all resources via ResourceDescriptorHeap[SL_*])
#include "slots.h"

struct RootConsts {
    uint k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, n;
};
ConstantBuffer<RootConsts> RC : register(b0);

// Same content as kernels.py prm (P_* slots); layout mirrored by struct Params in solver.cpp.
struct Params {
    float dt, wind_x, wind_y, wind_z;
    float n_em, n_imp, dx, amb;
    float kappa, cfloor, warm, inv_dx;
    float alpha, beta, eps, ddiss;
    float cool, burn, ign, bheat;
    float smoke, tol, maxit, tmax;
    float wmark, nsub, pad0, pad1;
    float4 em[96];    // emitter i: [3i] pos.xyz radius, [3i+1] density_rate temperature fuel_rate 0, [3i+2] vel.xyz 0
    float4 imp[64];   // impulse i: [2i] pos.xyz radius, [2i+1] vel.xyz 0
};
ConstantBuffer<Params> P : register(b1);

struct Level {        // one multigrid level (pressure.py self.lv[l])
    uint m, ox, orr, od, oo, pad0, pad1, pad2;   // size, offsets into SL_LX/LR/LD/LO
    float dmin, f1, f2, f3;
};

static const float HS = 0.35355339059327373f;   // 1/sqrt(8)
