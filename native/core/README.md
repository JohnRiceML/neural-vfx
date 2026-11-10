# native/core — device, queue, command recording (owner: native solver engineer)

`context.cpp` / `nvfx_core.h`: the `NvfxContext` behind `nvfx_context_*`, plus helpers that `sim/` uses.

- **Stand-alone** (`nvfx_context_create`): own `ID3D12Device` on the high-performance adapter (or `adapter_index`)
  and a DIRECT queue. **Hosted** (`nvfx_context_create_from_device`): AddRefs the host device and queue (direct or
  compute). It never touches a swap chain.
- Feature checks at creation: shader model 6.6 (dynamic resources), resource binding tier 3, typed UAV loads
  for R32G32B32A32 / R32G32. If any is missing the call returns `NVFX_ERR_UNSUPPORTED`.
- Internal submissions (reset, upload, download, the stand-alone step) use a ring of 16 command allocators on
  the context queue, with one fence. `begin()` waits only when 16 submissions are still in flight.
- `nvfx_last_error()` is thread-local.
- Resource helpers: committed buffers (default / upload / readback; upload and readback are persistently mapped)
  and 3D textures with UAV access.

## Caveats
- `enable_debug_layer = 1` returns `NVFX_ERR_UNSUPPORTED` on this machine. The D3D12 debug layer (Windows
  optional feature "Graphics Tools") is not installed, and we install nothing system-wide. So **nothing in
  this runtime has been validated by the debug layer**. Resource states were reasoned by hand (see
  `sim/README.md`).
- The DLL links the MSVC runtime dynamically (`/MD`: `MSVCP140`, `VCRUNTIME140`). Shaders are embedded, and
  there are no other dependencies beyond `d3d12.dll` and `dxgi.dll`.
