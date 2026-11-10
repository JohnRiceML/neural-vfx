// nvfx_detail_bench: parity dumps and timing of the native route B runner (C API only + plain D3D12 for test I/O).
//
//   nvfx_detail_bench --frame-dir DIR --model screennet_fp16.onnx --mode parity --bbox x0,y0,x1,y1[;...] --out DIR
//   nvfx_detail_bench ... --mode time  --bbox ... [--frames 300] [--warmup 60] [--flags N]
//   nvfx_detail_bench ... --mode sync                      (does nvfx_detail_run wait for earlier GPU work?)
// Input files (detail_bench.py make-frame): frame_rgb.u8 (H,W,3), alpha.f16, depth.f16 (H,W), motion.f16 (2,H,W).
// Output: one JSON object per line on stdout; parity mode also writes <out>/native_<i>.rgba (H,W,4 uint8).
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "nvfx.h"
#include "nvfx_detail.h"

using Microsoft::WRL::ComPtr;

static void die(const char* what) {
    fprintf(stderr, "error: %s: %s\n", what, nvfx_last_error());
    exit(1);
}
#define CHECK(x) do { if ((x) != NVFX_OK) die(#x); } while (0)
#define HR(x) do { HRESULT hr_ = (x); if (FAILED(hr_)) { fprintf(stderr, "error: %s: 0x%08lx\n", #x, (unsigned long)hr_); exit(1); } } while (0)

static std::vector<char> read_file(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "error: cannot open %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> v(static_cast<size_t>(n));
    if (fread(v.data(), 1, v.size(), f) != v.size()) { fprintf(stderr, "error: short read %s\n", p.c_str()); exit(1); }
    fclose(f);
    return v;
}

struct Gpu {
    ID3D12Device* dev;
    ID3D12CommandQueue* q;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE ev;
    uint64_t fv = 0;
    void init(NvfxContext* ctx) {
        dev = static_cast<ID3D12Device*>(nvfx_context_device(ctx));
        q = static_cast<ID3D12CommandQueue*>(nvfx_context_queue(ctx));
        HR(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc)));
        HR(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&list)));
        HR(list->Close());
        HR(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    }
    ID3D12GraphicsCommandList* begin() {
        HR(alloc->Reset());
        HR(list->Reset(alloc.Get(), nullptr));
        return list.Get();
    }
    uint64_t submit() {
        HR(list->Close());
        ID3D12CommandList* l = list.Get();
        q->ExecuteCommandLists(1, &l);
        HR(q->Signal(fence.Get(), ++fv));
        return fv;
    }
    void wait(uint64_t v) {
        if (fence->GetCompletedValue() >= v) return;
        HR(fence->SetEventOnCompletion(v, ev));
        WaitForSingleObject(ev, INFINITE);
    }
    ComPtr<ID3D12Resource> buffer(uint64_t size, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES st) {
        D3D12_HEAP_PROPERTIES hp = {heap};
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = size;
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> r;
        HR(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, st, nullptr, IID_PPV_ARGS(&r)));
        return r;
    }
    ComPtr<ID3D12Resource> tex(int w, int h, DXGI_FORMAT f, bool uav) {
        D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = UINT64(w);
        rd.Height = UINT(h);
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = f;
        rd.SampleDesc.Count = 1;
        rd.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
        ComPtr<ID3D12Resource> r;
        HR(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&r)));
        return r;
    }
    static void barrier(ID3D12GraphicsCommandList* l, ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
        D3D12_RESOURCE_BARRIER br = {};
        br.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        br.Transition.pResource = r;
        br.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        br.Transition.StateBefore = a;
        br.Transition.StateAfter = b;
        l->ResourceBarrier(1, &br);
    }
    // texture <- tightly packed rows of `bpp` bytes, then transition to `after`
    void upload(ID3D12Resource* t, const void* data, int w, int h, int bpp, D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_DESC rd = t->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
        UINT64 total = 0;
        dev->GetCopyableFootprints(&rd, 0, 1, 0, &fp, nullptr, nullptr, &total);
        auto up = buffer(total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        char* m = nullptr;
        HR(up->Map(0, nullptr, reinterpret_cast<void**>(&m)));
        for (int y = 0; y < h; ++y)
            memcpy(m + fp.Offset + size_t(y) * fp.Footprint.RowPitch, static_cast<const char*>(data) + size_t(y) * w * bpp, size_t(w) * bpp);
        up->Unmap(0, nullptr);
        auto* l = begin();
        D3D12_TEXTURE_COPY_LOCATION dst = {t, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        dst.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION src = {up.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        src.PlacedFootprint = fp;
        l->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        barrier(l, t, D3D12_RESOURCE_STATE_COPY_DEST, after);
        wait(submit());
    }
    std::vector<uint8_t> download(ID3D12Resource* t, int w, int h, int bpp, D3D12_RESOURCE_STATES state) {
        D3D12_RESOURCE_DESC rd = t->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
        UINT64 total = 0;
        dev->GetCopyableFootprints(&rd, 0, 1, 0, &fp, nullptr, nullptr, &total);
        auto rb = buffer(total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        auto* l = begin();
        barrier(l, t, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION dst = {rb.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        dst.PlacedFootprint = fp;
        D3D12_TEXTURE_COPY_LOCATION src = {t, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        src.SubresourceIndex = 0;
        l->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        barrier(l, t, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
        wait(submit());
        std::vector<uint8_t> out(size_t(w) * h * bpp);
        char* m = nullptr;
        HR(rb->Map(0, nullptr, reinterpret_cast<void**>(&m)));
        for (int y = 0; y < h; ++y)
            memcpy(out.data() + size_t(y) * w * bpp, m + fp.Offset + size_t(y) * fp.Footprint.RowPitch, size_t(w) * bpp);
        rb->Unmap(0, nullptr);
        return out;
    }
};

static double vram_mb(ID3D12Device* dev) {
    LUID luid = dev->GetAdapterLuid();
    ComPtr<IDXGIFactory4> f;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&f)))) return -1;
    ComPtr<IDXGIAdapter3> a;
    if (FAILED(f->EnumAdapterByLuid(luid, IID_PPV_ARGS(&a)))) return -1;
    DXGI_QUERY_VIDEO_MEMORY_INFO mi = {};
    if (FAILED(a->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mi))) return -1;
    return double(mi.CurrentUsage) / (1024.0 * 1024.0);
}

struct Stats { double med, p10, p90, mn, mean; };
static Stats stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    Stats s = {};
    if (v.empty()) return s;
    auto at = [&](double p) { return v[std::min(v.size() - 1, size_t(p * double(v.size() - 1) + 0.5))]; };
    s.med = at(0.5); s.p10 = at(0.1); s.p90 = at(0.9); s.mn = v.front();
    double t = 0;
    for (double x : v) t += x;
    s.mean = t / double(v.size());
    return s;
}
static std::string js(const char* name, const Stats& s) {
    char b[256];
    snprintf(b, sizeof b, "\"%s\":{\"med\":%.4f,\"p10\":%.4f,\"p90\":%.4f,\"min\":%.4f,\"mean\":%.4f}", name, s.med, s.p10, s.p90, s.mn, s.mean);
    return b;
}

static std::vector<std::array<int, 4>> parse_bboxes(const std::string& s) {
    std::vector<std::array<int, 4>> v;
    size_t p = 0;
    while (p < s.size()) {
        size_t e = s.find(';', p);
        std::string t = s.substr(p, e == std::string::npos ? std::string::npos : e - p);
        std::array<int, 4> b = {0, 0, 0, 0};
        if (sscanf(t.c_str(), "%d,%d,%d,%d", &b[0], &b[1], &b[2], &b[3]) == 4) v.push_back(b);
        if (e == std::string::npos) break;
        p = e + 1;
    }
    return v;
}

int main(int argc, char** argv) {
    std::string dir, model, mode = "parity", bbs = "0,0,0,0", out = ".";
    int frames = 300, warmup = 60, flags = 0, use_record = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "--frame-dir") dir = v;
        else if (a == "--model") model = v;
        else if (a == "--mode") mode = v;
        else if (a == "--bbox") bbs = v;
        else if (a == "--out") out = v;
        else if (a == "--frames") frames = atoi(v.c_str());
        else if (a == "--warmup") warmup = atoi(v.c_str());
        else if (a == "--flags") flags = atoi(v.c_str());
        else if (a == "--record") use_record = atoi(v.c_str());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (dir.empty() || model.empty()) { fprintf(stderr, "need --frame-dir and --model\n"); return 2; }
    // meta.json: only H and W are needed
    std::vector<char> meta = read_file(dir + "/meta.json");
    meta.push_back(0);
    int H = 0, W = 0;
    if (const char* p = strstr(meta.data(), "\"H\":")) H = atoi(p + 4);
    if (const char* p = strstr(meta.data(), "\"W\":")) W = atoi(p + 4);
    if (H <= 0 || W <= 0) { fprintf(stderr, "bad meta.json\n"); return 1; }

    NvfxContext* ctx = nullptr;
    CHECK(nvfx_context_create(nullptr, &ctx));
    Gpu g;
    g.init(ctx);
    double vram0 = vram_mb(g.dev);

    std::vector<char> rgb = read_file(dir + "/frame_rgb.u8"), al = read_file(dir + "/alpha.f16"),
                      de = read_file(dir + "/depth.f16"), mo = read_file(dir + "/motion.f16");
    size_t n = size_t(W) * H;
    std::vector<uint8_t> rgba(n * 4);
    for (size_t i = 0; i < n; ++i) {
        rgba[i * 4 + 0] = uint8_t(rgb[i * 3 + 0]);
        rgba[i * 4 + 1] = uint8_t(rgb[i * 3 + 1]);
        rgba[i * 4 + 2] = uint8_t(rgb[i * 3 + 2]);
        rgba[i * 4 + 3] = 255;
    }
    std::vector<uint16_t> mrg(n * 2);
    const uint16_t* m16 = reinterpret_cast<const uint16_t*>(mo.data());
    for (size_t i = 0; i < n; ++i) { mrg[i * 2] = m16[i]; mrg[i * 2 + 1] = m16[n + i]; }

    const auto NPSR = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, UAVS = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    auto t_col = g.tex(W, H, DXGI_FORMAT_R8G8B8A8_UNORM, false);
    auto t_al = g.tex(W, H, DXGI_FORMAT_R16_FLOAT, false);
    auto t_de = g.tex(W, H, DXGI_FORMAT_R16_FLOAT, false);
    auto t_mo = g.tex(W, H, DXGI_FORMAT_R16G16_FLOAT, false);
    auto t_out = g.tex(W, H, DXGI_FORMAT_R8G8B8A8_UNORM, true);
    g.upload(t_col.Get(), rgba.data(), W, H, 4, NPSR);
    g.upload(t_al.Get(), al.data(), W, H, 2, NPSR);
    g.upload(t_de.Get(), de.data(), W, H, 2, NPSR);
    g.upload(t_mo.Get(), mrg.data(), W, H, 4, NPSR);
    std::vector<uint8_t> zero(n * 4, 0);
    g.upload(t_out.Get(), zero.data(), W, H, 4, UAVS);
    double vram1 = vram_mb(g.dev);

    NvfxDetailDesc dd = {};
    dd.model_path = model.c_str();
    dd.width = W;
    dd.height = H;
    dd.process_scale = 0.5f;
    dd.flags = flags;
    auto c0 = std::chrono::steady_clock::now();
    NvfxDetail* det = nullptr;
    CHECK(nvfx_detail_create(ctx, &dd, &det));
    double create_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
    double vram2 = vram_mb(g.dev);
    printf("{\"event\":\"create\",\"W\":%d,\"H\":%d,\"flags\":%d,\"create_s\":%.3f,\"vram_mb_ctx_inputs\":%.1f,"
           "\"vram_mb_after_create\":%.1f,\"vram_mb_detail\":%.1f,\"vram_mb_process_start\":%.1f,\"owned_mb\":%.1f}\n",
           W, H, flags, create_s, vram1, vram2, vram2 - vram1, vram0, double(nvfx_detail_vram_bytes(det)) / 1048576.0);
    fflush(stdout);

    NvfxDetailFrame fr = {};
    fr.color = t_col.Get();
    fr.alpha = t_al.Get();
    fr.depth = t_de.Get();
    fr.motion = t_mo.Get();
    fr.output = t_out.Get();
    auto boxes = parse_bboxes(bbs);
    auto set_box = [&](const std::array<int, 4>& b) { for (int i = 0; i < 4; ++i) fr.bbox[i] = b[size_t(i)]; };
    // --record 1: the hosted path, nvfx_detail_record into the bench's own command list (one allocator: each
    // frame waits for the previous one, so only serialized numbers are meaningful in this mode)
    uint64_t last = 0;
    auto run_frame = [&]() -> NvfxResult {
        if (!use_record) return nvfx_detail_run(det, &fr);
        g.wait(last);
        ID3D12GraphicsCommandList* l = g.begin();
        NvfxResult r = nvfx_detail_record(det, l, &fr);
        if (r != NVFX_OK) return r;
        last = g.submit();
        return NVFX_OK;
    };

    if (mode == "parity") {
        for (size_t i = 0; i < boxes.size(); ++i) {
            // a different region first, submitted back to back: checks queue ordering pre -> net -> post
            std::array<int, 4> other = (boxes[i][2] > boxes[i][0]) ? std::array<int, 4>{0, 0, 0, 0}
                                                                  : std::array<int, 4>{100, 100, 400, 300};
            set_box(other);
            CHECK(run_frame());
            set_box(boxes[i]);
            CHECK(run_frame());
            CHECK(nvfx_context_wait_idle(ctx));
            int32_t reg[6];
            CHECK(nvfx_detail_region(det, fr.bbox, reg));
            auto img = g.download(t_out.Get(), W, H, 4, UAVS);
            std::string p = out + "/native_" + std::to_string(i) + ".rgba";
            FILE* f = fopen(p.c_str(), "wb");
            fwrite(img.data(), 1, img.size(), f);
            fclose(f);
            printf("{\"event\":\"parity\",\"i\":%zu,\"bbox\":[%d,%d,%d,%d],\"region\":[%d,%d,%d,%d],\"crop\":[%d,%d],\"file\":\"%s\"}\n",
                   i, fr.bbox[0], fr.bbox[1], fr.bbox[2], fr.bbox[3], reg[0], reg[1], reg[2], reg[3], reg[4], reg[5], p.c_str());
            fflush(stdout);
        }
    } else if (mode == "time") {
        for (size_t i = 0; i < boxes.size(); ++i) {
            set_box(boxes[i]);
            for (int w = 0; w < warmup; ++w) {
                CHECK(run_frame());
                if (w % 4 == 3) CHECK(nvfx_context_wait_idle(ctx));
            }
            CHECK(nvfx_context_wait_idle(ctx));
            // (a) serialized: one frame, wait, read its timestamps -> exact per-frame GPU split
            std::vector<double> pre, net, post, tot, cpu;
            for (int k = 0; k < frames; ++k) {
                CHECK(run_frame());
                CHECK(nvfx_context_wait_idle(ctx));
                NvfxDetailTimings t;
                CHECK(nvfx_detail_get_timings(det, &t));
                pre.push_back(t.pre_ms); net.push_back(t.net_ms); post.push_back(t.post_ms);
                tot.push_back(t.total_ms); cpu.push_back(t.cpu_ms);
            }
            // (b) pipelined: frames submitted back to back (the runner's 8-frame ring is the only throttle)
            std::vector<double> cpu_p;
            auto p0 = std::chrono::steady_clock::now();
            for (int k = 0; k < frames; ++k) {
                auto a = std::chrono::steady_clock::now();
                CHECK(run_frame());
                cpu_p.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
            }
            CHECK(nvfx_context_wait_idle(ctx));
            double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - p0).count() / frames;
            int32_t reg[6];
            CHECK(nvfx_detail_region(det, fr.bbox, reg));
            printf("{\"event\":\"time\",\"bbox\":[%d,%d,%d,%d],\"region\":[%d,%d,%d,%d],\"crop\":[%d,%d],\"frames\":%d,%s,%s,%s,%s,%s,%s,"
                   "\"pipelined_ms_per_frame\":%.4f,\"vram_mb\":%.1f}\n",
                   fr.bbox[0], fr.bbox[1], fr.bbox[2], fr.bbox[3], reg[0], reg[1], reg[2], reg[3], reg[4], reg[5], frames,
                   js("pre", stats(pre)).c_str(), js("net", stats(net)).c_str(), js("post", stats(post)).c_str(),
                   js("total", stats(tot)).c_str(), js("cpu_run", stats(cpu)).c_str(), js("cpu_run_pipelined", stats(cpu_p)).c_str(),
                   wall, vram_mb(g.dev));
            fflush(stdout);
        }
    } else if (mode == "firstuse") {
        // Every crop bucket: with ~20+ ms of copies queued ahead, CPU ms of the first call that uses the bucket vs the
        // second and third. A lazy per-bucket init shows as first >> second; a GPU wait shows as ~the queued ms.
        uint64_t sz = 256ull << 20;
        auto a = g.buffer(sz, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
        auto bb = g.buffer(sz, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
        set_box({0, 0, 0, 0});
        for (int w = 0; w < 10; ++w) CHECK(run_frame());
        CHECK(nvfx_context_wait_idle(ctx));
        g.wait(last);
        const double fr6[6] = {0.25, 0.375, 0.5, 0.625, 0.75, 1.0};
        auto sizes = [](int L, const double* f) {
            std::vector<int> v;
            for (int i = 0; i < 6; ++i) v.push_back(std::min(L, (std::max(int(L * f[i]), 1) + 63) / 64 * 64));
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            return v;
        };
        for (int hh : sizes(H, fr6))
            for (int ww : sizes(W, fr6)) {
                std::array<int, 4> box = {(W - ww) / 2, (H - hh) / 2, (W - ww) / 2 + ww, (H - hh) / 2 + hh};
                if (ww == W && hh == H) box = {1, 1, W, H};   // whole frame, but not the warm-up call's exact bbox
                set_box(box);
                auto* l = g.begin();
                for (int c = 0; c < 16; ++c) l->CopyBufferRegion(c % 2 ? a.Get() : bb.Get(), 0, c % 2 ? bb.Get() : a.Get(), 0, sz);
                auto s0 = std::chrono::steady_clock::now();
                uint64_t v = g.submit();
                double ms[3];
                for (int k = 0; k < 3; ++k) {
                    auto k0 = std::chrono::steady_clock::now();
                    CHECK(nvfx_detail_run(det, &fr));
                    ms[k] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - k0).count();
                }
                g.wait(v);
                double busy = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
                CHECK(nvfx_context_wait_idle(ctx));
                int32_t reg[6];
                CHECK(nvfx_detail_region(det, fr.bbox, reg));
                printf("{\"event\":\"firstuse\",\"crop\":[%d,%d],\"first_ms\":%.3f,\"second_ms\":%.3f,\"third_ms\":%.3f,\"queued_ms\":%.1f}\n",
                       reg[4], reg[5], ms[0], ms[1], ms[2], busy);
                fflush(stdout);
            }
    } else if (mode == "sync") {
        // Queue ~N ms of copy work, then call nvfx_detail_run: if the runtime waited for the GPU, the call takes ~N ms.
        uint64_t sz = 256ull << 20;
        auto a = g.buffer(sz, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
        auto b = g.buffer(sz, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
        set_box(boxes[0]);
        for (int w = 0; w < 10; ++w) CHECK(nvfx_detail_run(det, &fr));
        CHECK(nvfx_context_wait_idle(ctx));
        for (int rep = 0; rep < 6; ++rep) {
            int copies = 24;
            auto* l = g.begin();
            for (int c = 0; c < copies; ++c) l->CopyBufferRegion(c % 2 ? a.Get() : b.Get(), 0, c % 2 ? b.Get() : a.Get(), 0, sz);
            auto s0 = std::chrono::steady_clock::now();
            uint64_t v = g.submit();
            auto s1 = std::chrono::steady_clock::now();
            CHECK(nvfx_detail_run(det, &fr));
            auto s2 = std::chrono::steady_clock::now();
            g.wait(v);
            auto s3 = std::chrono::steady_clock::now();
            CHECK(nvfx_context_wait_idle(ctx));
            auto s4 = std::chrono::steady_clock::now();
            auto ms = [](auto x, auto y) { return std::chrono::duration<double, std::milli>(y - x).count(); };
            printf("{\"event\":\"sync\",\"rep\":%d,\"busy_work_done_after_ms\":%.3f,\"run_cpu_ms\":%.3f,\"all_done_after_ms\":%.3f,"
                   "\"run_returned_before_busy_work_finished\":%s}\n",
                   rep, ms(s0, s3), ms(s1, s2), ms(s0, s4), ms(s1, s2) < ms(s0, s3) * 0.5 ? "true" : "false");
            fflush(stdout);
        }
    }
    nvfx_detail_destroy(det);
    nvfx_context_destroy(ctx);
    return 0;
}
