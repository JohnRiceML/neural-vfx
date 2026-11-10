/* nvfx.h — C API of the neural-vfx native runtime (D3D12).  OWNER: lead.
 *
 * This header is the contract. Changing it needs the lead's approval and a matching edit of
 * native/README.md. Everything not declared here is internal to its owner's directory.
 *
 * Conventions (identical to vfx/core/fields3d.py):
 *   - Cubic domain of side NVFX_DOMAIN_SIZE metres, res^3 cells, dx = NVFX_DOMAIN_SIZE / res.
 *   - x right, y UP (buoyancy +y), z depth. World units: metres, m/s, kelvin, per-second rates.
 *   - CPU layout (C, D, H, W), index [c, z, y, x], float32, C order (x fastest):
 *       offset = ((c * D + z) * H + y) * W + x
 *     Channel order: density, vel_x, vel_y, vel_z, temperature, fuel.
 *   - Cell (x, y, z) has its centre at ((x + .5) dx, (y + .5) dx, (z + .5) dx).
 *   - GPU textures: Texture3D, texel (x, y, z) = cell (x, y, z); width = W, height = H, depth = D.
 *   - Velocity is collocated (cell-centred). The six box faces are walls. `solid` marks obstacles.
 *
 * Threading: one context = one thread at a time. No function blocks on the GPU unless its
 * comment says "synchronizes".
 */
#ifndef NVFX_H
#define NVFX_H

#include <stdint.h>

#ifdef _WIN32
#  ifdef NVFX_BUILD
#    define NVFX_API __declspec(dllexport)
#  else
#    define NVFX_API __declspec(dllimport)
#  endif
#else
#  define NVFX_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NVFX_VERSION_MAJOR 0
#define NVFX_VERSION_MINOR 1

#define NVFX_DOMAIN_SIZE   1.0f
#define NVFX_AMBIENT_TEMP  293.0f
#define NVFX_SIM_DT        (1.0f / 30.0f)
#define NVFX_NUM_CHANNELS  6
#define NVFX_MAX_EMITTERS  32
#define NVFX_MAX_IMPULSES  32

enum { NVFX_CH_DENSITY = 0, NVFX_CH_VEL_X, NVFX_CH_VEL_Y, NVFX_CH_VEL_Z, NVFX_CH_TEMPERATURE, NVFX_CH_FUEL };

typedef enum NvfxResult {
    NVFX_OK = 0,
    NVFX_ERR_INVALID_ARG = 1,
    NVFX_ERR_DEVICE = 2,          /* D3D12 call failed; see nvfx_last_error() */
    NVFX_ERR_UNSUPPORTED = 3,
    NVFX_ERR_OUT_OF_MEMORY = 4,
    NVFX_ERR_INTERNAL = 5
} NvfxResult;

/* Human-readable text for the last failure on this thread ("" if none). Never NULL. */
NVFX_API const char* nvfx_last_error(void);
NVFX_API const char* nvfx_version_string(void);     /* e.g. "nvfx 0.1 sim3d-d3d12-0.1.0" */

/* ---------------------------------------------------------------- context */
typedef struct NvfxContext NvfxContext;

typedef struct NvfxContextDesc {
    int32_t adapter_index;        /* -1 = the high-performance adapter */
    int32_t enable_debug_layer;   /* 1 = D3D12 debug layer + break on error (tests) */
} NvfxContextDesc;

/* Stand-alone: creates its own ID3D12Device and a direct command queue. */
NVFX_API NvfxResult nvfx_context_create(const NvfxContextDesc* desc, NvfxContext** out);
/* Hosted (engine plugin): adopts the host's ID3D12Device* and ID3D12CommandQueue* (AddRef'd). */
NVFX_API NvfxResult nvfx_context_create_from_device(void* d3d12_device, void* d3d12_command_queue,
                                                    NvfxContext** out);
NVFX_API void       nvfx_context_destroy(NvfxContext* ctx);
/* Blocks until all work submitted through this context has finished on the GPU. Synchronizes. */
NVFX_API NvfxResult nvfx_context_wait_idle(NvfxContext* ctx);
/* ID3D12Device* / ID3D12CommandQueue* (not AddRef'd). */
NVFX_API void*      nvfx_context_device(NvfxContext* ctx);
NVFX_API void*      nvfx_context_queue(NvfxContext* ctx);

/* ---------------------------------------------------------------- solver */
typedef struct NvfxSolver NvfxSolver;

/* Field names and defaults are those of vfx.sim3d.SolverConfig3D. */
typedef struct NvfxSolverConfig {
    float   buoyancy_alpha;        /* 0.6    */
    float   buoyancy_beta;         /* 0.004  */
    float   vorticity_eps;         /* 6.0    */
    float   density_dissipation;   /* 0.05   */
    float   temperature_cooling;   /* 4.0    */
    float   burn_rate;             /* 8.0    */
    float   ignition_temp;         /* 550    */
    float   burn_heat;             /* 4000   */
    float   smoke_from_fuel;       /* 0.35   */
    float   pressure_tol;          /* 2e-5; 0 = always run pressure_max_iters */
    int32_t pressure_max_iters;    /* 40 (capped at 256). On D3D12 every unused iteration slot still costs
                                      ~10 us, so a host that wants the adaptive stop should set ~12. */
    int32_t pressure_warm_start;   /* 1      */
    float   max_temperature;       /* 3000   */
    float   wind_marker_temp;      /* 300    */
    float   heat_diffusion;        /* 5e-5   */
    float   confinement_floor;     /* 0.5    */
} NvfxSolverConfig;

typedef struct NvfxEmitter {
    float pos[3]; float radius; float density_rate; float temperature; float fuel_rate; float velocity[3];
} NvfxEmitter;
typedef struct NvfxImpulse { float pos[3]; float radius; float velocity[3]; } NvfxImpulse;

typedef struct NvfxForces {
    const NvfxEmitter* emitters; int32_t num_emitters;   /* <= NVFX_MAX_EMITTERS */
    const NvfxImpulse* impulses; int32_t num_impulses;   /* <= NVFX_MAX_IMPULSES */
    float wind[3];
} NvfxForces;

typedef enum NvfxObstacleKind { NVFX_OBSTACLE_SPHERE = 0, NVFX_OBSTACLE_BOX = 1 } NvfxObstacleKind;
/* sphere: size[0] = radius.  box: size = half-extents.  Rasterized as rasterize_obstacles_3d does. */
typedef struct NvfxObstacle { int32_t kind; float pos[3]; float size[3]; } NvfxObstacle;

/* GPU ms of the last step that has finished on the GPU, from D3D12 timestamp queries placed in the
 * step's own command list. Reading them never blocks: values lag the newest step by a frame or two. */
typedef struct NvfxSolverTimings {
    float   advection_ms;   /* advection + mass correction */
    float   forces_ms;      /* sources, diffusion, vorticity, body forces */
    float   projection_ms;
    float   total_ms;
    int32_t pcg_iters;      /* iterations the projection actually ran */
    uint64_t step_index;    /* which step these numbers belong to: steps completed since create/reset
                               (first step = 1); all fields are 0 until a step has finished.
                               nvfx_solver_project writes no timings. */
} NvfxSolverTimings;

NVFX_API void       nvfx_solver_config_default(NvfxSolverConfig* cfg);   /* SolverConfig3D() */
NVFX_API void       nvfx_solver_config_runtime(NvfxSolverConfig* cfg);   /* runtime_config(): tol 0, 4 iters */

/* res: cubic grid size, even; 64 is the target. cfg NULL = defaults. State starts as reset() leaves it. */
NVFX_API NvfxResult nvfx_solver_create(NvfxContext* ctx, int32_t res, const NvfxSolverConfig* cfg, NvfxSolver** out);
NVFX_API void       nvfx_solver_destroy(NvfxSolver* s);
NVFX_API int32_t    nvfx_solver_res(const NvfxSolver* s);
NVFX_API NvfxResult nvfx_solver_set_config(NvfxSolver* s, const NvfxSolverConfig* cfg);  /* takes effect next step */

/* Ambient, empty state (density 0, velocity 0, T = ambient, fuel 0), obstacles rasterized into `solid`,
 * pressure warm-start cleared, step counter 0. */
NVFX_API NvfxResult nvfx_solver_reset(NvfxSolver* s, const NvfxObstacle* obstacles, int32_t num_obstacles);

/* One solver step, same scheme and order as Solver3D.step(). Submits to the context queue and returns
 * without waiting. forces NULL = none. Deterministic: identical inputs give bit-identical state. */
NVFX_API NvfxResult nvfx_solver_step(NvfxSolver* s, const NvfxForces* forces, float dt);
/* Hosted path: records the same step into the host's ID3D12GraphicsCommandList* instead of submitting.
 * The host executes the list on the queue the context was created with.
 * record_step sets its own descriptor heap, compute root signature and pipeline state on the list: the host
 * must re-bind its own afterwards. Recorded steps must be executed: at most 16 may be outstanding (the 17th
 * waits up to 20 s for the GPU, then fails). */
NVFX_API NvfxResult nvfx_solver_record_step(NvfxSolver* s, void* d3d12_command_list, const NvfxForces* forces, float dt);
/* Pressure projection only (Solver3D.project()). */
NVFX_API NvfxResult nvfx_solver_project(NvfxSolver* s);

NVFX_API NvfxResult nvfx_solver_get_timings(NvfxSolver* s, NvfxSolverTimings* out);

/* --- CPU access for tests and tools. All of these synchronize. --- */
/* data: 6*res^3 floats, layout (C,D,H,W). solid: res^3 floats (1 inside obstacles). Either may be NULL. */
NVFX_API NvfxResult nvfx_solver_download_state(NvfxSolver* s, float* data, float* solid);
/* Replaces the state (and/or the obstacle mask). Clears the pressure warm-start. */
NVFX_API NvfxResult nvfx_solver_upload_state(NvfxSolver* s, const float* data, const float* solid);
/* out: res^3 floats = Solver3D.divergence(): |div u| * dx / (max|u| + 1e-8) over fluid cells, 0 in solids,
 * div u = the face-averaged central divergence. */
NVFX_API NvfxResult nvfx_solver_download_divergence(NvfxSolver* s, float* out);

/* --- GPU access for the renderer / engine. Valid after create; contents current after each step. ---
 * Both are ID3D12Resource* Texture3D (res x res x res, 1 mip), not AddRef'd, left in
 * D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE between steps. The pointers may change from step to
 * step (ping-pong): query them after every step.
 *   NVFX_TEX_VEL_DENSITY  R32G32B32A32_FLOAT  (vel_x, vel_y, vel_z, density)
 *   NVFX_TEX_TEMP_FUEL    R32G32_FLOAT        (temperature K, fuel)
 *   NVFX_TEX_SOLID        R32_FLOAT           (1 inside obstacles)                                   */
typedef enum NvfxSolverTexture { NVFX_TEX_VEL_DENSITY = 0, NVFX_TEX_TEMP_FUEL = 1, NVFX_TEX_SOLID = 2 } NvfxSolverTexture;
NVFX_API void*      nvfx_solver_texture(NvfxSolver* s, NvfxSolverTexture which);

#ifdef __cplusplus
}
#endif
#endif /* NVFX_H */
