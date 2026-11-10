// Context: device + queue ownership (or adoption), fence, command allocator ring, error text.
#include "nvfx_core.h"

#include <cstdarg>
#include <cstdio>
#include <new>

namespace nvfx {

static thread_local std::string g_err;

void set_error(const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    g_err = buf;
}
void clear_error() { g_err.clear(); }

bool ok(HRESULT hr, const char* what) {
    if (SUCCEEDED(hr)) return true;
    set_error("%s failed: HRESULT 0x%08lx", what, static_cast<unsigned long>(hr));
    return false;
}

bool init_context(NvfxContext* c) {
    if (!ok(c->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&c->fence)), "CreateFence")) return false;
    c->event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!c->event) { set_error("CreateEvent failed"); return false; }
    D3D12_COMMAND_QUEUE_DESC qd = c->queue->GetDesc();
    c->list_type = qd.Type;
    if (qd.Type != D3D12_COMMAND_LIST_TYPE_DIRECT && qd.Type != D3D12_COMMAND_LIST_TYPE_COMPUTE) {
        set_error("queue must be a direct or compute queue");
        return false;
    }
    for (auto& s : c->slots)
        if (!ok(c->device->CreateCommandAllocator(qd.Type, IID_PPV_ARGS(&s.alloc)), "CreateCommandAllocator")) return false;
    ComPtr<ID3D12GraphicsCommandList> l;
    if (!ok(c->device->CreateCommandList(0, qd.Type, c->slots[0].alloc.Get(), nullptr, IID_PPV_ARGS(&l)), "CreateCommandList"))
        return false;
    if (!ok(l.As(&c->list), "ID3D12GraphicsCommandList2")) return false;
    c->list->Close();
    UINT64 f = 0;
    if (!ok(c->queue->GetTimestampFrequency(&f), "GetTimestampFrequency")) return false;
    c->ts_freq = static_cast<double>(f);
    return true;
}

// feature checks the solver depends on
static bool check_features(ID3D12Device* d) {
    D3D12_FEATURE_DATA_SHADER_MODEL sm = {D3D_SHADER_MODEL_6_6};
    if (FAILED(d->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof sm)) || sm.HighestShaderModel < D3D_SHADER_MODEL_6_6) {
        set_error("device does not support shader model 6.6 (dynamic resources)");
        return false;
    }
    D3D12_FEATURE_DATA_D3D12_OPTIONS o = {};
    d->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &o, sizeof o);
    if (o.ResourceBindingTier < D3D12_RESOURCE_BINDING_TIER_3) {
        set_error("device needs resource binding tier 3");
        return false;
    }
    if (!o.TypedUAVLoadAdditionalFormats) {
        set_error("device lacks typed UAV loads for R32G32B32A32_FLOAT / R32G32_FLOAT");
        return false;
    }
    return true;
}

}  // namespace nvfx

using namespace nvfx;

ID3D12GraphicsCommandList2* NvfxContext::begin() {
    if (recording) { set_error("internal: nested command recording"); return nullptr; }
    Slot& s = slots[slot_i];
    if (s.fence && !wait(s.fence)) return nullptr;   // only waits if kCmdRing submissions are in flight
    if (!ok(s.alloc->Reset(), "CommandAllocator::Reset")) return nullptr;
    if (!ok(list->Reset(s.alloc.Get(), nullptr), "CommandList::Reset")) return nullptr;
    recording = true;
    return list.Get();
}

uint64_t NvfxContext::submit() {
    recording = false;
    if (!ok(list->Close(), "CommandList::Close")) return 0;
    ID3D12CommandList* l = list.Get();
    queue->ExecuteCommandLists(1, &l);
    uint64_t v = fence_next++;
    if (!ok(queue->Signal(fence.Get(), v), "Queue::Signal")) return 0;
    slots[slot_i].fence = v;
    slot_i = (slot_i + 1) % kCmdRing;
    return v;
}

bool NvfxContext::wait(uint64_t value) {
    if (fence->GetCompletedValue() >= value) return true;
    if (!ok(fence->SetEventOnCompletion(value, event), "SetEventOnCompletion")) return false;
    if (WaitForSingleObject(event, 30000) != WAIT_OBJECT_0) {
        HRESULT r = device->GetDeviceRemovedReason();
        set_error("GPU wait timed out (device removed reason 0x%08lx)", static_cast<unsigned long>(r));
        return false;
    }
    return true;
}

bool NvfxContext::wait_idle() {
    uint64_t v = fence_next++;
    if (!ok(queue->Signal(fence.Get(), v), "Queue::Signal")) return false;
    return wait(v);
}

bool NvfxContext::create_buffer(Buffer& b, uint64_t size, D3D12_HEAP_TYPE heap, bool uav, const char* name) {
    D3D12_HEAP_PROPERTIES hp = {heap};
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = size < 256 ? 256 : size;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    D3D12_RESOURCE_STATES st = heap == D3D12_HEAP_TYPE_UPLOAD     ? D3D12_RESOURCE_STATE_GENERIC_READ
                               : heap == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST
                                                                  : D3D12_RESOURCE_STATE_COMMON;
    HRESULT hr = device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, st, nullptr, IID_PPV_ARGS(&b.res));
    if (hr == E_OUTOFMEMORY) { set_error("out of memory creating %s", name); return false; }
    if (!ok(hr, name)) return false;
    b.size = rd.Width;
    if (name) {
        wchar_t w[128];
        swprintf(w, 128, L"nvfx.%hs", name);
        b.res->SetName(w);
    }
    if (heap != D3D12_HEAP_TYPE_DEFAULT) {
        D3D12_RANGE r = {0, 0};
        if (!ok(b.res->Map(0, heap == D3D12_HEAP_TYPE_READBACK ? nullptr : &r, &b.map), "Map")) return false;
    }
    return true;
}

bool NvfxContext::create_tex3d(ComPtr<ID3D12Resource>& t, DXGI_FORMAT fmt, int n, D3D12_RESOURCE_STATES state,
                               const char* name) {
    D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    rd.Width = static_cast<UINT64>(n);
    rd.Height = static_cast<UINT>(n);
    rd.DepthOrArraySize = static_cast<UINT16>(n);
    rd.MipLevels = 1;
    rd.Format = fmt;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    HRESULT hr = device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, IID_PPV_ARGS(&t));
    if (hr == E_OUTOFMEMORY) { set_error("out of memory creating %s", name); return false; }
    if (!ok(hr, name)) return false;
    wchar_t w[128];
    swprintf(w, 128, L"nvfx.%hs", name);
    t->SetName(w);
    return true;
}

// ------------------------------------------------------------------ C API
extern "C" {

NVFX_API const char* nvfx_last_error(void) { return g_err.c_str(); }
NVFX_API const char* nvfx_version_string(void) { return "nvfx 0.1 sim3d-d3d12-0.1.0"; }

NVFX_API NvfxResult nvfx_context_create(const NvfxContextDesc* desc, NvfxContext** out) {
    clear_error();
    if (!out) { set_error("out is NULL"); return NVFX_ERR_INVALID_ARG; }
    *out = nullptr;
    int adapter_index = desc ? desc->adapter_index : -1;
    bool debug = desc && desc->enable_debug_layer;
    if (debug) {
        ComPtr<ID3D12Debug> dbg;
        if (FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) {
            set_error("D3D12 debug layer not available (install the Windows optional feature 'Graphics Tools')");
            return NVFX_ERR_UNSUPPORTED;
        }
        dbg->EnableDebugLayer();
    }
    ComPtr<IDXGIFactory6> f;
    if (!ok(CreateDXGIFactory2(debug ? DXGI_CREATE_FACTORY_DEBUG : 0, IID_PPV_ARGS(&f)), "CreateDXGIFactory2"))
        return NVFX_ERR_DEVICE;
    ComPtr<IDXGIAdapter1> ad;
    HRESULT hr = adapter_index < 0
                     ? f->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&ad))
                     : f->EnumAdapters1(static_cast<UINT>(adapter_index), &ad);
    if (FAILED(hr)) { set_error("adapter %d not found", adapter_index); return NVFX_ERR_INVALID_ARG; }
    auto* c = new (std::nothrow) NvfxContext();
    if (!c) return NVFX_ERR_OUT_OF_MEMORY;
    if (!ok(D3D12CreateDevice(ad.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&c->device)), "D3D12CreateDevice")) {
        delete c;
        return NVFX_ERR_DEVICE;
    }
    if (debug) {
        ComPtr<ID3D12InfoQueue> iq;
        if (SUCCEEDED(c->device.As(&iq))) {
            iq->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
            iq->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
        }
    }
    if (!check_features(c->device.Get())) { delete c; return NVFX_ERR_UNSUPPORTED; }
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (!ok(c->device->CreateCommandQueue(&qd, IID_PPV_ARGS(&c->queue)), "CreateCommandQueue") || !init_context(c)) {
        nvfx_context_destroy(c);
        return NVFX_ERR_DEVICE;
    }
    c->queue->SetName(L"nvfx.queue");
    *out = c;
    return NVFX_OK;
}

NVFX_API NvfxResult nvfx_context_create_from_device(void* d3d12_device, void* d3d12_command_queue, NvfxContext** out) {
    clear_error();
    if (!out || !d3d12_device || !d3d12_command_queue) { set_error("NULL argument"); return NVFX_ERR_INVALID_ARG; }
    *out = nullptr;
    auto* c = new (std::nothrow) NvfxContext();
    if (!c) return NVFX_ERR_OUT_OF_MEMORY;
    c->hosted = true;
    c->device = static_cast<ID3D12Device*>(d3d12_device);          // ComPtr assignment AddRefs
    c->queue = static_cast<ID3D12CommandQueue*>(d3d12_command_queue);
    if (!check_features(c->device.Get())) { delete c; return NVFX_ERR_UNSUPPORTED; }
    if (!init_context(c)) { nvfx_context_destroy(c); return NVFX_ERR_DEVICE; }
    *out = c;
    return NVFX_OK;
}

NVFX_API void nvfx_context_destroy(NvfxContext* ctx) {
    if (!ctx) return;
    if (ctx->fence) ctx->wait_idle();
    if (ctx->event) CloseHandle(ctx->event);
    delete ctx;
}

NVFX_API NvfxResult nvfx_context_wait_idle(NvfxContext* ctx) {
    clear_error();
    if (!ctx) return NVFX_ERR_INVALID_ARG;
    return ctx->wait_idle() ? NVFX_OK : NVFX_ERR_DEVICE;
}

NVFX_API void* nvfx_context_device(NvfxContext* ctx) { return ctx ? ctx->device.Get() : nullptr; }
NVFX_API void* nvfx_context_queue(NvfxContext* ctx) { return ctx ? ctx->queue.Get() : nullptr; }

}  // extern "C"
