// Texture2D helpers for tests and tools (nvfx_texture2d_*). All synchronous: every call that records GPU work
// submits it on the context queue and waits for it before returning. Subresource 0 (mip 0, slice 0) only.
#include "nvfx_core.h"

#include <cstring>

using namespace nvfx;

namespace {

// Validates ctx/resource and fetches the subresource-0 copy footprint. Sets the error text on failure.
NvfxResult prepare(NvfxContext* ctx, void* resource, const void* data, uint32_t row_pitch, ID3D12Resource** out_res,
                   D3D12_PLACED_SUBRESOURCE_FOOTPRINT* fp, UINT* rows, UINT64* row_size, UINT64* total) {
    if (!ctx || !resource || !data) { set_error("NULL argument"); return NVFX_ERR_INVALID_ARG; }
    if (ctx->recording) { set_error("internal: context is already recording"); return NVFX_ERR_INTERNAL; }
    auto* res = static_cast<ID3D12Resource*>(resource);
    D3D12_RESOURCE_DESC rd = res->GetDesc();
    if (rd.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
        set_error("resource is not a Texture2D (dimension %d)", static_cast<int>(rd.Dimension));
        return NVFX_ERR_INVALID_ARG;
    }
    ComPtr<ID3D12Device> dev;
    if (FAILED(res->GetDevice(IID_PPV_ARGS(&dev))) || dev.Get() != ctx->device.Get()) {
        set_error("resource was not created on the context's device");
        return NVFX_ERR_INVALID_ARG;
    }
    ctx->device->GetCopyableFootprints(&rd, 0, 1, 0, fp, rows, row_size, total);
    if (*total == UINT64_MAX || *rows == 0) {
        set_error("format %d has no copyable footprint", static_cast<int>(rd.Format));
        return NVFX_ERR_UNSUPPORTED;
    }
    if (row_pitch < *row_size) {
        set_error("row_pitch %u is smaller than a texture row (%llu bytes)", row_pitch,
                  static_cast<unsigned long long>(*row_size));
        return NVFX_ERR_INVALID_ARG;
    }
    *out_res = res;
    return NVFX_OK;
}

void transition(ID3D12GraphicsCommandList* l, ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
    if (from == to) return;
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    l->ResourceBarrier(1, &b);
}

// Copies `src` subresource 0 <-> `buf` (placed footprint) between transitions `state` -> copy_state -> `state`.
bool record_copy(NvfxContext* ctx, ID3D12Resource* tex, D3D12_RESOURCE_STATES state, ID3D12Resource* buf,
                 const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& fp, bool to_texture) {
    ID3D12GraphicsCommandList2* l = ctx->begin();
    if (!l) return false;
    D3D12_TEXTURE_COPY_LOCATION t = {};
    t.pResource = tex;
    t.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    t.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION b = {};
    b.pResource = buf;
    b.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    b.PlacedFootprint = fp;
    D3D12_RESOURCE_STATES cs = to_texture ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COPY_SOURCE;
    transition(l, tex, state, cs);
    if (to_texture) l->CopyTextureRegion(&t, 0, 0, 0, &b, nullptr);
    else            l->CopyTextureRegion(&b, 0, 0, 0, &t, nullptr);
    transition(l, tex, cs, state);
    uint64_t v = ctx->submit();
    return v && ctx->wait(v);
}

}  // namespace

extern "C" {

NVFX_API NvfxResult nvfx_texture2d_create(NvfxContext* ctx, int32_t width, int32_t height, uint32_t dxgi_format,
                                          uint32_t resource_flags, uint32_t state, void** out_resource) {
    clear_error();
    if (!ctx || !out_resource) { set_error("NULL argument"); return NVFX_ERR_INVALID_ARG; }
    *out_resource = nullptr;
    if (width <= 0 || height <= 0) { set_error("invalid size %dx%d", width, height); return NVFX_ERR_INVALID_ARG; }
    D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = static_cast<UINT64>(width);
    rd.Height = static_cast<UINT>(height);
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = static_cast<DXGI_FORMAT>(dxgi_format);
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags = static_cast<D3D12_RESOURCE_FLAGS>(resource_flags);
    ID3D12Resource* r = nullptr;
    HRESULT hr = ctx->device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, static_cast<D3D12_RESOURCE_STATES>(state),
                                                      nullptr, IID_PPV_ARGS(&r));
    if (hr == E_OUTOFMEMORY) { set_error("out of memory creating a %dx%d texture", width, height); return NVFX_ERR_OUT_OF_MEMORY; }
    if (!ok(hr, "CreateCommittedResource(Texture2D)")) return NVFX_ERR_DEVICE;
    r->SetName(L"nvfx.texture2d");
    *out_resource = r;
    return NVFX_OK;
}

NVFX_API void nvfx_texture2d_release(void* resource) {
    if (resource) static_cast<IUnknown*>(resource)->Release();
}

NVFX_API NvfxResult nvfx_texture2d_upload(NvfxContext* ctx, void* resource, uint32_t state, const void* data, uint32_t row_pitch) {
    clear_error();
    ID3D12Resource* tex = nullptr;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
    UINT rows = 0;
    UINT64 row_size = 0, total = 0;
    NvfxResult pr = prepare(ctx, resource, data, row_pitch, &tex, &fp, &rows, &row_size, &total);
    if (pr != NVFX_OK) return pr;
    Buffer up;
    if (!ctx->create_buffer(up, total, D3D12_HEAP_TYPE_UPLOAD, false, "texture2d_upload")) return NVFX_ERR_DEVICE;
    auto* dst = static_cast<uint8_t*>(up.map) + fp.Offset;
    auto* src = static_cast<const uint8_t*>(data);
    for (UINT y = 0; y < rows; ++y)
        std::memcpy(dst + size_t(y) * fp.Footprint.RowPitch, src + size_t(y) * row_pitch, size_t(row_size));
    if (!record_copy(ctx, tex, static_cast<D3D12_RESOURCE_STATES>(state), up.res.Get(), fp, true)) return NVFX_ERR_DEVICE;
    return NVFX_OK;
}

NVFX_API NvfxResult nvfx_texture2d_download(NvfxContext* ctx, void* resource, uint32_t state, void* data, uint32_t row_pitch) {
    clear_error();
    ID3D12Resource* tex = nullptr;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
    UINT rows = 0;
    UINT64 row_size = 0, total = 0;
    NvfxResult pr = prepare(ctx, resource, data, row_pitch, &tex, &fp, &rows, &row_size, &total);
    if (pr != NVFX_OK) return pr;
    if (!ctx->wait_idle()) return NVFX_ERR_DEVICE;   // the texture may still be written by queued work (e.g. a renderer)
    Buffer rb;
    if (!ctx->create_buffer(rb, total, D3D12_HEAP_TYPE_READBACK, false, "texture2d_readback")) return NVFX_ERR_DEVICE;
    if (!record_copy(ctx, tex, static_cast<D3D12_RESOURCE_STATES>(state), rb.res.Get(), fp, false)) return NVFX_ERR_DEVICE;
    auto* src = static_cast<const uint8_t*>(rb.map) + fp.Offset;
    auto* dst = static_cast<uint8_t*>(data);
    for (UINT y = 0; y < rows; ++y)
        std::memcpy(dst + size_t(y) * row_pitch, src + size_t(y) * fp.Footprint.RowPitch, size_t(row_size));
    return NVFX_OK;
}

}  // extern "C"
