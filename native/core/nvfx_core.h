// Internal core of the native runtime: device / queue / fence ownership, command recording ring,
// resource helpers, error reporting. Not part of the public API (see include/nvfx.h).
#pragma once
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdint>
#include <string>

#include "nvfx.h"

using Microsoft::WRL::ComPtr;

namespace nvfx {

void set_error(const char* fmt, ...);
void clear_error();
// false (and sets the error text) if FAILED(hr)
bool ok(HRESULT hr, const char* what);

constexpr int kCmdRing = 16;

struct Buffer {
    ComPtr<ID3D12Resource> res;
    uint64_t size = 0;
    void* map = nullptr;   // persistently mapped for upload / readback heaps
    D3D12_GPU_VIRTUAL_ADDRESS va() const { return res->GetGPUVirtualAddress(); }
};

}  // namespace nvfx

struct NvfxContext {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> fence;
    HANDLE event = nullptr;
    uint64_t fence_next = 1;     // value the next submit signals
    bool hosted = false;
    double ts_freq = 1.0;        // timestamp ticks per second of `queue`
    D3D12_COMMAND_LIST_TYPE list_type = D3D12_COMMAND_LIST_TYPE_DIRECT;

    struct Slot { ComPtr<ID3D12CommandAllocator> alloc; uint64_t fence = 0; };
    Slot slots[nvfx::kCmdRing];
    int slot_i = 0;
    ComPtr<ID3D12GraphicsCommandList2> list;
    bool recording = false;

    // ---- internal command recording (own allocators, submitted on `queue`)
    ID3D12GraphicsCommandList2* begin();          // nullptr on failure
    uint64_t submit();                            // close + execute + signal; 0 on failure
    bool wait(uint64_t value);                    // CPU wait for fence value
    uint64_t completed() const { return fence->GetCompletedValue(); }
    bool wait_idle();

    // ---- resources
    bool create_buffer(nvfx::Buffer& b, uint64_t size, D3D12_HEAP_TYPE heap, bool uav, const char* name);
    bool create_tex3d(ComPtr<ID3D12Resource>& t, DXGI_FORMAT fmt, int n, D3D12_RESOURCE_STATES state, const char* name);
};

namespace nvfx {
bool init_context(NvfxContext* c);   // fence, allocators, list, timestamp frequency (device + queue set)
}
