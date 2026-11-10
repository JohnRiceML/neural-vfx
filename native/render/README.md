# native/render: the volume ray-marcher on D3D12 (owner: native render engineer)

This is a port of `vfx/render3d` `VolumeRenderer` (the look pass and the route B aux buffers) to C++20 and HLSL compute
shaders inside `nvfx.dll`. It reads the native solver's state textures directly on the GPU.

> **Status (2026-11-09): written and compiled, NOT RUN.** A GPU pause was in effect, so nothing here has created a
> D3D12 device yet: no pipeline creation, no draw, no parity number, no timing. See "Not yet run" below.

## Files
- `nvfx_render.h`: the proposed C API, for the lead to merge into `include/nvfx.h`.
- `render.cpp`: the API implementation, resources, command recording, the camera basis, the blackbody LUT and
  the bbox.
- `rslots.h`: descriptor-heap slots, shared by the C++ and the HLSL.
- `shaders/render_common.hlsli`: the root signature layout, the parameter block (`RParams`) and the helpers
  from `kernels.py` (box_hit, safe_inv, hg, smoothstep, ign, LUT, emission).
- `shaders/prep.hlsl`: volume prep in 6 stages: pack + occupancy, solid blur, light volume, glow pyramid.
- `shaders/march.hlsl`: the march. It is compiled twice, as `march_c` (AUX=0) and `march_aux` (AUX=1).
- `shaders/post.hlsl`: bloom (3 dispatches), then composite with the aux resolve.
- `CMakeLists.txt`: compiles the shaders with dxc (`cs_6_6 -O3 -Gis -denorm preserve -WX`), embeds them in
  `nvfx.dll`, and builds `nvfx_render_selfcheck.exe`.
- `selfcheck.cpp` (`nvfx_render_selfcheck.exe`): the owner self-check and bench. It uses the C API only.
- `compare.py`: the owner's ctypes diff against the Python `VolumeRenderer` on the same solver state.

## API (proposed; full comments in `nvfx_render.h`)
```c
nvfx_camera_default(&cam);  nvfx_render_params_default(&p);            // fields3d.Camera(), RenderParams3D()
nvfx_renderer_create(ctx, 2560, 1440, 0.5f, &r);                        // VolumeRenderer((1440, 2560), march_scale=.5)
nvfx_renderer_set_params(r, &p);                                        // RenderParams3D field for field
nvfx_renderer_draw(r, solver, &cam, &prev_cam_or_NULL, NVFX_RENDER_AUX [| NVFX_RENDER_REUSE_VOLUME]);  // submits, no wait
nvfx_renderer_record_draw(r, host_cmd_list, solver, &cam, prev, flags); // hosted (Unreal render graph)
nvfx_renderer_texture(r, NVFX_RENDER_TEX_COLOR | _ALPHA | _DEPTH | _MOTION);   // ID3D12Resource*, NPSR between draws
nvfx_renderer_get_timings(r, &t);                                       // prep / march / post / total GPU ms, lagged
nvfx_renderer_download(r, which, out);                                  // tests; synchronizes
nvfx_render_screen_bbox(&cam, w, h, bbox);                              // CPU, projected domain cube
```
- Outputs are Texture2D at output resolution, row 0 = top, left in NON_PIXEL_SHADER_RESOURCE (the state
  `nvfx_detail.h` expects):
  - COLOR is R8G8B8A8_UNORM. RGB is Python's uint8 frame and A = 255.
  - ALPHA and DEPTH are R16_FLOAT.
  - MOTION is R16G16_FLOAT, in output px per SIM_DT, x right, y down.
- Flags:
  - `NO_SOLID` = Python `solid=None`.
  - `NO_VELOCITY` = Python `velocity=None`.
  - Otherwise the solver's `solid` and velocity are used. This is what the Gate 3a reference
    (`solid=state.solid`, `velocity=state.velocity`) corresponds to.
  - With an all-zero solid (campfire), the image is the same as with `NO_SOLID`.
- **Bbox: CPU, from the projected domain cube.** This is exactly `vfx/app3d/pipeline.py::_screen_aux`. I chose it
  over an alpha > threshold bbox read back from the GPU for three reasons:
  - It needs no GPU→CPU readback and no lag.
  - It is valid before the draw is even recorded, so the detail pass can be recorded in the same frame.
  - It is what the Python app feeds `ScreenDetail` today.

  The cost is that the box is looser than the visible effect.

## Design
- **One command list per draw.** It has 6 prep dispatches (skipped under REUSE_VOLUME) and 5 view dispatches:
  march, bloom1, bloom2, bloom3 and composite.
  - Every PSO is built at create.
  - There is one shader-visible heap and one root signature: 8 root constants, a root CBV and a static
    trilinear/clamp sampler, with `ResourceDescriptorHeap[]` (SM 6.6).
  - Only global UAV barriers are used, and only between dependent dispatches. Transitions are batched into them.
- **Inputs.** For every draw, the solver's current textures get SRVs written into the descriptor slots of a
  16-slot draw ring, because the solver ping-pongs. The parameter block is in the same ring slot of an upload
  heap, read directly as the root CBV. A slot is reused only after the draw 16 back has finished (a
  WriteBufferImmediate marker), so a draw blocks only when 16 are in flight.
- **Volume prep** reads the solver's fp32 textures with `Load` and does the same work as Python, in 6 stages:
  - It packs (max(d,0), T) into an RG16F texture, the velocity into RGBA16F (Python's half textures), and the
    occupancy bricks with apron from the fp32 values.
  - It blurs the solid mask (torch's 7-tap Gaussian, replicate padding, order z, y, x) into an R8_UNORM texture
    plus the solid bricks.
  - It builds the 80³ light volume (RG16F).
  - It builds the fire-glow pyramid with torch's exact pooling windows and divisors: adaptive avg-pool to 32³,
    7-tap blur, avg_pool 4 → 8³, blur, `upsample_trilinear3d` evaluation order, then an RGBA16F texture.
  - Independent jobs share a dispatch: each stage is a flat dispatch running up to 3 job ranges.
- **Sampling** uses hardware trilinear on the half / unorm8 textures, the same formats Python samples. All fp16
  roundings are explicit (`f32tof16`).
- **March:** a line-by-line port of `_make_march`: brick skipping on the jittered lattice, dual-lobe phase,
  3 scattering octaves, obstacle iso-surface with bisection and gradient normals, floor with grid.
  - The AUX variant adds the opacity-weighted depth/velocity sums. As in Python's `wp.static`, they are removed at
    compile time from the colour variant.
  - `-Gis` (IEEE strict, no contraction) keeps the colour arithmetic of both variants identical (R5).
    `NVFX_RENDER_SINGLE_MARCH=1` (read at create) runs colour-only draws through the AUX pipeline with the aux
    write disabled. This is a fallback if R5 ever fails.
- **Bloom:**
  - `bloom1` is a 16×16 LDS tile: adaptive pool of relu(hdr − thresh), x-blur, y-blur (= b1), and
    avg_pool2d(b1, 4, ceil_mode) of its own 4×4 blocks.
  - `bloom2` x-blurs that.
  - `bloom3` does the y-blur and `upsample_bilinear2d`, then adds 1.5×.
- **Composite:** Catmull-Rom clamped to the 2×2 range, background and vignette, bloom, exposure, contrast, ACES,
  sRGB, dither. When AUX is set, it also does `aux_resolve` (bilinear + un-premultiply), so the aux resolve has no
  dispatch of its own.
- **Timestamps:** 4 per draw (start, after prep, after march, end) go into a 64-slot readback ring.
  `get_timings` never blocks.
- **Determinism:** no atomics. The occupancy scatter only ever stores 1. There are no float reductions across
  threads except fixed-order LDS loops.

## What matches Python, and what deliberately differs
Same stages, constants and evaluation order: every kernel of `kernels.py`, `RenderParams3D` defaults,
`camera_basis` (float64, aspect = march w/h), the blackbody LUT (`blackbody.py` re-implemented in double), the
torch pool / blur / interpolate formulas, `round()` of the march size, and the bloom and pool output sizes.

Deviations and expected small differences:
1. **Floating point.**
   - Warp builds with `--fmad=true` (FMA contraction). These shaders are IEEE-strict.
   - HLSL `exp`/`pow` are the driver's approximations, not CUDA's.
   - torch's cuDNN conv sums in an unspecified order; here it is sequential.

   All of this is far below 8-bit and R1 granularity, but it means colours are not bit-exact against Python.
2. **Dither.** Python hashes `sin(r·12.9898 + c·78.233)·43758.5453` with arguments up to ~2e5 rad, where 1 ulp of
   the argument already decorrelates the dither. The native side matches it this way:
   - It forms the argument as one exactly emulated FMA (in double).
   - It evaluates sin with double range reduction and a Taylor series, i.e. close to CUDA's accurate `sinf`.
   - Which product nvcc fused is unknown. `NVFX_RENDER_DITHER_MODE=0|1|2` (default 0) selects the hypothesis, and
     `compare.py --dither-modes` tells which one matches.

   If none matches, expect about 1/3 of the pixels of every flat area to be off by 1 level: mean abs ≈ 0.33, within
   R1. A device without double shader ops gets a float-only composite (`composite_f`) whose dither differs.
3. **Solid mask resolution.**
   - Native always uses the solver's own mask (64³, `solid_res` = solver res).
   - The Python app passes 128³ obstacles; the Gate 3a reference passes the 64³ state solid.
4. **Glow source.** The glow pool job recomputes each light voxel's emission from the volume instead of reading a
   `glow_src` array written by the light kernel. It is the same code on the same inputs, so the values are the same,
   and the pool no longer waits for the light march.
5. **LUT rebuild.** `set_params` rebuilds the blackbody LUT when t_lo / t_hi / t_ref change, which synchronizes
   once. Python builds the LUT only in `__init__`, so the native result equals a renderer *constructed* with those
   params. Changing `light_res` also synchronizes once.
6. **Volume source.** The native renderer draws the solver's state, not an arbitrary `Volume`. Route A (a 256³
   neural volume) would need a texture-input variant of `draw`.
7. **REUSE_VOLUME fallback.** REUSE_VOLUME falls back to a full draw (reported in `timings.flags`) when:
   - there was no full draw yet,
   - the solver resolution changed, or
   - `NO_SOLID` differs from the last full draw.

   Python does the same for the missing-volume and has-solid cases.
8. **Timings.** They are split prep / march / post (bloom + composite + aux resolve) / total. Python reports
   prep / view / total.

## Not yet run (GPU pause), and how to run it
Nothing below has been executed. Everything compiles cleanly (no warnings) in `native/build-render`.
- Device-side creation: root signature, the 13 PSOs, the double-precision composite on this driver, and the
  format support checks.
- Every draw: prep, march, bloom, composite, aux. Resource states were reasoned by hand; the D3D12 debug layer is
  not installed on this machine (see `core/README.md`).
- R1–R6 parity, T2 timing, the dither-mode question, the hosted `record_draw` path, `download`, and `get_timings`.

To run, once the GPU is free (out_dir on D:, outside the repo):
```
native\build-render\bin\nvfx_render_selfcheck.exe D:\nvfx_render_out            # images, R5/R6/reuse/hosted checks, bench
<venv python> native/render/compare.py --out D:\nvfx_render_cmp --dither-modes     # vs Python, from the worktree root
```
