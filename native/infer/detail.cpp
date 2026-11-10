// Route B ScreenDetail runner: HLSL pre/post + the network on DirectML (default) or ONNX Runtime's DML EP.
// See nvfx_detail.h for the contract and native/infer/README.md for the design and the measurements.
//
// DirectML backend: the ONNX file is only a weight container here. detail.cpp parses its initializers (a minimal
// protobuf reader, no protobuf library) and builds the ScreenNet trunk + HR head as one DML graph per crop bucket
// (static shapes). All buckets share ONE temporary resource (only one bucket runs at a time on the queue); each has
// its own small persistent resource (weights in DML's preferred layout). Everything of a frame is recorded into one
// command list: pre (HLSL) -> DML dispatch -> lowfreq / fin / post (HLSL), with timestamps in between.
// The output constraints at the end of ScreenNet.forward (grey_lowfreq, alpha_gate) run in HLSL (fin), with the
// same fp16 roundings as PyTorch, rather than as DML element-wise ops.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <new>
#include <numeric>
#include <string>
#include <vector>

#include <DirectML.h>
#include <onnxruntime_c_api.h>
#include <dml_provider_factory.h>

#include "nvfx_core.h"
#include "nvfx_detail.h"

namespace {

#include "detail_shader_table.inc"   // kDetailShaders[] = {name, bytecode, size}

using nvfx::Buffer;
using nvfx::ok;
using nvfx::set_error;

constexpr int kFrames = 8;            // frames in flight (descriptor + timestamp slots)
constexpr int kStaticDesc = 8;        // see Slots
constexpr int kFrameDesc = 5;         // color, alpha, depth, motion SRV + output UAV
constexpr int kRbStride = 64;         // readback bytes per frame slot: 4 timestamps + marker
constexpr int kConsts = 28;
constexpr double kCropFracs[6] = {0.25, 0.375, 0.5, 0.625, 0.75, 1.0};   // ScreenDetail defaults
enum Slots { D_A = 0, D_B, D_DELTA, D_ALPHA, D_DRAW, D_COARSE };

// ------------------------------------------------------------------ runtime DLLs (loaded once, by full path)
struct Runtime {
    HMODULE dml = nullptr, ort = nullptr;
    decltype(&DMLCreateDevice1) create_dml = nullptr;
    const OrtApi* api = nullptr;
    const OrtDmlApi* dml_api = nullptr;
    OrtEnv* env = nullptr;
    int env_users = 0;
};
Runtime g_rt;

std::wstring module_dir() {
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&module_dir), &self);
    wchar_t buf[MAX_PATH];
    DWORD n = GetModuleFileNameW(self, buf, MAX_PATH);
    std::wstring p(buf, n);
    size_t s = p.find_last_of(L"\\/");
    return s == std::wstring::npos ? L"." : p.substr(0, s);
}

bool ort_ok(OrtStatus* s, const char* what) {
    if (!s) return true;
    set_error("%s: %s", what, g_rt.api->GetErrorMessage(s));
    g_rt.api->ReleaseStatus(s);
    return false;
}
#define ORT(expr) ort_ok((expr), #expr)

bool load_dml() {
    if (g_rt.create_dml) return true;
    if (!g_rt.dml) {
        g_rt.dml = LoadLibraryExW((module_dir() + L"\\DirectML.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!g_rt.dml) { set_error("cannot load DirectML.dll next to nvfx.dll (error %lu)", GetLastError()); return false; }
    }
    g_rt.create_dml = reinterpret_cast<decltype(&DMLCreateDevice1)>(GetProcAddress(g_rt.dml, "DMLCreateDevice1"));
    if (!g_rt.create_dml) { set_error("DirectML.dll has no DMLCreateDevice1"); return false; }
    return true;
}

bool ort_acquire() {
    if (!load_dml()) return false;   // ORT must bind to the same DirectML.dll
    if (!g_rt.ort) {
        g_rt.ort = LoadLibraryExW((module_dir() + L"\\onnxruntime.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!g_rt.ort) { set_error("cannot load onnxruntime.dll next to nvfx.dll (error %lu)", GetLastError()); return false; }
        auto get = reinterpret_cast<const OrtApiBase*(ORT_API_CALL*)()>(GetProcAddress(g_rt.ort, "OrtGetApiBase"));
        if (!get) { set_error("onnxruntime.dll has no OrtGetApiBase"); return false; }
        g_rt.api = get()->GetApi(ORT_API_VERSION);
        if (!g_rt.api) { set_error("onnxruntime.dll does not provide ORT API %d", ORT_API_VERSION); return false; }
        if (!ORT(g_rt.api->GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&g_rt.dml_api))))
            return false;
    }
    if (g_rt.env_users == 0 && !ORT(g_rt.api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "nvfx_detail", &g_rt.env))) return false;
    ++g_rt.env_users;
    return true;
}

void ort_release() {
    if (g_rt.env_users > 0 && --g_rt.env_users == 0) {
        g_rt.api->ReleaseEnv(g_rt.env);
        g_rt.env = nullptr;
    }   // the DLLs stay loaded (ORT keeps threads / statics alive; unloading is not safe)
}

// ------------------------------------------------------------------ minimal ONNX (protobuf) reader
struct Pb {
    const uint8_t* p;
    const uint8_t* e;
    bool varint(uint64_t& v) {
        v = 0;
        for (int s = 0; s < 64 && p < e; s += 7) {
            uint8_t b = *p++;
            v |= uint64_t(b & 0x7f) << s;
            if (!(b & 0x80)) return true;
        }
        return false;
    }
    bool key(uint32_t& num, uint32_t& wt) {
        if (p >= e) return false;
        uint64_t k;
        if (!varint(k)) return false;
        num = uint32_t(k >> 3);
        wt = uint32_t(k & 7);
        return true;
    }
    bool bytes(Pb& sub) {
        uint64_t n;
        if (!varint(n) || n > uint64_t(e - p)) return false;
        sub = {p, p + n};
        p += n;
        return true;
    }
    bool skip(uint32_t wt) {
        uint64_t v;
        Pb s;
        switch (wt) {
            case 0: return varint(v);
            case 1: if (e - p < 8) return false; p += 8; return true;
            case 2: return bytes(s);
            case 5: if (e - p < 4) return false; p += 4; return true;
            default: return false;
        }
    }
};

struct Init {
    std::vector<int64_t> dims;
    int dtype = 0;   // 1 float, 10 float16
    const uint8_t* data = nullptr;
    size_t size = 0;
    size_t count() const { size_t n = 1; for (auto d : dims) n *= size_t(d); return n; }
};

bool parse_tensor(Pb t, std::string& name, Init& out) {
    uint32_t f, wt;
    while (t.key(f, wt)) {
        Pb s;
        uint64_t v;
        if (f == 1 && wt == 0) { if (!t.varint(v)) return false; out.dims.push_back(int64_t(v)); }
        else if (f == 1 && wt == 2) {
            if (!t.bytes(s)) return false;
            while (s.p < s.e) { if (!s.varint(v)) return false; out.dims.push_back(int64_t(v)); }
        }
        else if (f == 2 && wt == 0) { if (!t.varint(v)) return false; out.dtype = int(v); }
        else if (f == 8 && wt == 2) { if (!t.bytes(s)) return false; name.assign(reinterpret_cast<const char*>(s.p), size_t(s.e - s.p)); }
        else if (f == 9 && wt == 2) { if (!t.bytes(s)) return false; out.data = s.p; out.size = size_t(s.e - s.p); }
        else if (!t.skip(wt)) return false;
    }
    return true;
}

bool parse_model(const std::vector<char>& bytes, std::map<std::string, Init>& inits, std::map<std::string, std::string>& meta) {
    const uint8_t* b0 = reinterpret_cast<const uint8_t*>(bytes.data());
    Pb m = {b0, b0 + bytes.size()};
    uint32_t f, wt;
    while (m.key(f, wt)) {
        Pb s;
        if (f == 7 && wt == 2) {            // GraphProto
            if (!m.bytes(s)) return false;
            uint32_t gf, gwt;
            while (s.key(gf, gwt)) {
                Pb t;
                if (gf == 5 && gwt == 2) {   // initializer
                    if (!s.bytes(t)) return false;
                    std::string name;
                    Init in;
                    if (!parse_tensor(t, name, in)) return false;
                    inits[name] = in;
                } else if (!s.skip(gwt)) return false;
            }
        } else if (f == 14 && wt == 2) {     // metadata_props
            if (!m.bytes(s)) return false;
            std::string k, v;
            uint32_t ef, ewt;
            while (s.key(ef, ewt)) {
                Pb t;
                if ((ef == 1 || ef == 2) && ewt == 2) {
                    if (!s.bytes(t)) return false;
                    (ef == 1 ? k : v).assign(reinterpret_cast<const char*>(t.p), size_t(t.e - t.p));
                } else if (!s.skip(ewt)) return false;
            }
            meta[k] = v;
        } else if (!m.skip(wt)) return false;
    }
    return true;
}

uint16_t f2h(float f) {   // round to nearest even
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000u, a = x & 0x7fffffffu, h;
    if (a >= 0x47800000u) h = a > 0x7f800000u ? 0x7e00u : 0x7c00u;
    else if (a >= 0x38800000u) h = (a - 0x38000000u + 0x0fffu + ((a >> 13) & 1u)) >> 13;
    else { float v; std::memcpy(&v, &a, 4); h = uint32_t(std::nearbyint(v * 16777216.0f)); }
    return uint16_t(s | h);
}

// ------------------------------------------------------------------ DirectML graph construction
struct TDesc {
    DML_BUFFER_TENSOR_DESC b = {};
    DML_TENSOR_DESC d = {};
    UINT sz[4] = {};
    UINT st[4] = {};
};

struct Val {           // a tensor in the graph: graph input `input` or output 0 of node `node`
    int node = -1;
    int input = -1;
    TDesc* desc = nullptr;
};

struct Graph {
    IDMLDevice* dev = nullptr;
    bool nhwc = false;
    std::deque<TDesc> tds;
    std::deque<std::vector<DML_TENSOR_DESC>> arrays;
    std::vector<ComPtr<IDMLOperator>> ops;
    std::vector<DML_INPUT_GRAPH_EDGE_DESC> ie;
    std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> me;
    std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> oe;
    bool failed = false;

    TDesc* td(UINT n, UINT c, UINT h, UINT w, bool nhwc_layout, bool owned) {
        TDesc& t = tds.emplace_back();
        t.sz[0] = n; t.sz[1] = c; t.sz[2] = h; t.sz[3] = w;
        t.b.DataType = DML_TENSOR_DATA_TYPE_FLOAT16;
        t.b.Flags = owned ? DML_TENSOR_FLAG_OWNED_BY_DML : DML_TENSOR_FLAG_NONE;
        t.b.DimensionCount = 4;
        t.b.Sizes = t.sz;
        if (nhwc_layout) {
            t.st[0] = h * w * c; t.st[1] = 1; t.st[2] = w * c; t.st[3] = c;
            t.b.Strides = t.st;
        }
        t.b.TotalTensorSizeInBytes = (UINT64(n) * c * h * w * 2 + 3) & ~UINT64(3);
        t.d = {DML_TENSOR_TYPE_BUFFER, &t.b};
        return &t;
    }
    int add(const DML_OPERATOR_DESC& d) {
        ComPtr<IDMLOperator> op;
        if (!ok(dev->CreateOperator(&d, IID_PPV_ARGS(&op)), "IDMLDevice::CreateOperator")) { failed = true; return 0; }
        ops.push_back(op);
        return int(ops.size()) - 1;
    }
    void connect(const Val& v, int to, UINT idx) {
        if (v.input >= 0) ie.push_back({UINT(v.input), UINT(to), idx, nullptr});
        else me.push_back({UINT(v.node), 0, UINT(to), idx, nullptr});
    }
    // conv k x k, pad k/2, + bias, optional fused ReLU; weights = graph inputs widx, widx + 1
    Val conv(const Val& x, UINT cout, UINT k, UINT widx, bool relu, bool out_nhwc) {
        const UINT* s = x.desc->sz;
        TDesc* f = td(cout, s[1], k, k, false, true);
        TDesc* b = td(1, cout, 1, 1, false, true);
        TDesc* o = td(1, cout, s[2], s[3], out_nhwc, false);
        DML_ACTIVATION_RELU_OPERATOR_DESC rd = {};
        DML_OPERATOR_DESC act = {DML_OPERATOR_ACTIVATION_RELU, &rd};
        UINT one[2] = {1, 1}, pad[2] = {k / 2, k / 2}, zero[2] = {0, 0};
        DML_CONVOLUTION_OPERATOR_DESC c = {&x.desc->d, &f->d, &b->d, &o->d, DML_CONVOLUTION_MODE_CROSS_CORRELATION,
                                           DML_CONVOLUTION_DIRECTION_FORWARD, 2, one, one, pad, pad, zero, 1,
                                           relu ? &act : nullptr};
        int n = add({DML_OPERATOR_CONVOLUTION, &c});
        connect(x, n, 0);
        connect({-1, int(widx), f}, n, 1);
        connect({-1, int(widx) + 1, b}, n, 2);
        return {n, -1, o};
    }
    Val scale(const Val& x, float s) {
        const UINT* z = x.desc->sz;
        TDesc* o = td(z[0], z[1], z[2], z[3], nhwc, false);
        DML_SCALE_BIAS sb = {s, 0.0f};
        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC d = {&x.desc->d, &o->d, &sb};
        int n = add({DML_OPERATOR_ELEMENT_WISE_IDENTITY, &d});
        connect(x, n, 0);
        return {n, -1, o};
    }
    Val addv(const Val& a, const Val& b) {
        const UINT* z = a.desc->sz;
        TDesc* o = td(z[0], z[1], z[2], z[3], nhwc, false);
        DML_ELEMENT_WISE_ADD_OPERATOR_DESC d = {&a.desc->d, &b.desc->d, &o->d};
        int n = add({DML_OPERATOR_ELEMENT_WISE_ADD, &d});
        connect(a, n, 0);
        connect(b, n, 1);
        return {n, -1, o};
    }
    Val depth_to_space(const Val& x, UINT r) {   // PixelShuffle = DepthToSpace CRD
        const UINT* z = x.desc->sz;
        TDesc* o = td(z[0], z[1] / (r * r), z[2] * r, z[3] * r, nhwc, false);
        DML_DEPTH_TO_SPACE1_OPERATOR_DESC d = {&x.desc->d, &o->d, r, DML_DEPTH_SPACE_ORDER_COLUMN_ROW_DEPTH};
        int n = add({DML_OPERATOR_DEPTH_TO_SPACE1, &d});
        connect(x, n, 0);
        return {n, -1, o};
    }
    Val join(const Val& a, const Val& b) {       // concat on channels
        const UINT* z = a.desc->sz;
        TDesc* o = td(z[0], z[1] + b.desc->sz[1], z[2], z[3], nhwc, false);
        auto& arr = arrays.emplace_back(std::vector<DML_TENSOR_DESC>{a.desc->d, b.desc->d});
        DML_JOIN_OPERATOR_DESC d = {2, arr.data(), &o->d, 1};
        int n = add({DML_OPERATOR_JOIN, &d});
        connect(a, n, 0);
        connect(b, n, 1);
        return {n, -1, o};
    }
    ComPtr<IDMLCompiledOperator> compile(IDMLDevice1* d1, UINT n_inputs, const Val& out, DML_EXECUTION_FLAGS flags) {
        oe.push_back({UINT(out.node), 0, 0, nullptr});
        std::vector<DML_OPERATOR_GRAPH_NODE_DESC> on(ops.size());
        std::vector<DML_GRAPH_NODE_DESC> nodes(ops.size());
        for (size_t i = 0; i < ops.size(); ++i) {
            on[i] = {ops[i].Get(), nullptr};
            nodes[i] = {DML_GRAPH_NODE_TYPE_OPERATOR, &on[i]};
        }
        std::vector<DML_GRAPH_EDGE_DESC> ein(ie.size()), emid(me.size()), eout(oe.size());
        for (size_t i = 0; i < ie.size(); ++i) ein[i] = {DML_GRAPH_EDGE_TYPE_INPUT, &ie[i]};
        for (size_t i = 0; i < me.size(); ++i) emid[i] = {DML_GRAPH_EDGE_TYPE_INTERMEDIATE, &me[i]};
        for (size_t i = 0; i < oe.size(); ++i) eout[i] = {DML_GRAPH_EDGE_TYPE_OUTPUT, &oe[i]};
        DML_GRAPH_DESC g = {n_inputs, 1, UINT(nodes.size()), nodes.data(), UINT(ein.size()), ein.data(),
                            UINT(eout.size()), eout.data(), UINT(emid.size()), emid.data()};
        ComPtr<IDMLCompiledOperator> c;
        if (!ok(d1->CompileGraph(&g, flags, IID_PPV_ARGS(&c)), "IDMLDevice1::CompileGraph")) return nullptr;
        return c;
    }
};

// ScreenNet architecture, from the weights + the scalar metadata written by export_onnx.py
struct Arch {
    int r = 4, width = 64, n_blocks = 0, hr_feats = 8, hr_hidden = 16, hr_k = 3, cin = 7;
    float res_scale = 0.1f, depth_scale = 0.5f, motion_scale = 0.125f, input_clip = 8.0f, alpha_gate = 0.0f;
    int grey_lowfreq = 0;
    struct W { std::string name; std::vector<int64_t> dims; std::vector<uint16_t> data; };
    std::vector<W> w;   // weight, bias per conv in graph-input order: head, (c1, c2) x n_blocks, tail, hr1, hr2
};

struct Bucket {
    int ch = 0, cw = 0, ph = 0, pw = 0;
    bool ready = false;
    // ORT backend
    OrtSession* sess = nullptr;
    OrtIoBinding* bind = nullptr;
    OrtValue* v_rgb = nullptr;
    OrtValue* v_aux = nullptr;
    OrtValue* v_delta = nullptr;
    // DML backend
    ComPtr<IDMLCompiledOperator> op;
    ComPtr<IDMLBindingTable> bt;
    Buffer persist;
    UINT desc_base = 0, desc_count = 0;
    UINT64 temp_size = 0, init_temp_size = 0;
};

void uav_barrier(ID3D12GraphicsCommandList* l) {
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    l->ResourceBarrier(1, &b);
}

}  // namespace

struct Consts {   // mirrors cbuffer K in shaders/detail.hlsl
    uint32_t color_srv, alpha_srv, depth_srv, motion_srv;
    uint32_t buf_a, buf_b, delta_uav, out_uav;
    int32_t y0, x0, vh, vw;
    uint32_t ph, pw, fw, fh;
    uint32_t alpha_buf, draw_buf, coarse_buf, grey_k;
    float gate, clip, depth_scale, motion_scale;
    uint32_t r, pad0, pad1, pad2;
};
static_assert(sizeof(Consts) == kConsts * 4, "root constants");

struct NvfxDetail {
    NvfxContext* ctx = nullptr;
    int W = 0, H = 0, k = 2, q = 64;
    float scale = 0.5f;
    int flags = 0;
    bool use_ort = false, ort_held = false;
    std::vector<int> hs, ws;
    std::vector<Bucket> buckets;     // hs.size() * ws.size(), row-major in (h index, w index)
    std::vector<char> model;         // ONNX bytes
    Arch arch;

    ComPtr<IDMLDevice> dml;
    ComPtr<IDMLDevice1> dml1;
    ComPtr<IDMLCommandRecorder> recorder;
    OrtMemoryInfo* mem = nullptr;
    OrtRunOptions* run_opts = nullptr;
    void* a_rgb = nullptr;
    void* a_aux = nullptr;
    void* a_delta = nullptr;

    Buffer buf_a, buf_b, delta, alpha, draw, coarse, temp, weights;
    uint64_t vram = 0;

    ComPtr<ID3D12RootSignature> rs;
    ComPtr<ID3D12PipelineState> pso_pre, pso_post, pso_lowfreq, pso_fin;
    ComPtr<ID3D12DescriptorHeap> heap;
    UINT inc = 0;
    ComPtr<ID3D12QueryHeap> qheap;
    Buffer readback;                 // kFrames x kRbStride

    struct Slot {
        uint64_t fence = 0;          // run(): ctx fence value of the frame's last submission
        uint64_t index = 0;          // frame index
        float cpu_ms = 0;
        int32_t region[4] = {}, crop[2] = {};
    } slots[kFrames];
    uint64_t frame = 0;

    D3D12_CPU_DESCRIPTOR_HANDLE cpu(UINT i) const {
        D3D12_CPU_DESCRIPTOR_HANDLE h = heap->GetCPUDescriptorHandleForHeapStart();
        h.ptr += SIZE_T(i) * inc;
        return h;
    }
    D3D12_GPU_DESCRIPTOR_HANDLE gpu(UINT i) const {
        D3D12_GPU_DESCRIPTOR_HANDLE h = heap->GetGPUDescriptorHandleForHeapStart();
        h.ptr += UINT64(i) * inc;
        return h;
    }

    std::vector<int> sizes(int L) const {
        std::vector<int> v;
        for (double f : kCropFracs) {
            int n = std::max(int(L * f), 1);
            v.push_back((n + q - 1) / q * q);
        }
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        return v;
    }

    // ScreenDetail._region along one axis: offset, bucket index, valid length
    void fit(int a, int b, int L, const std::vector<int>& sz, int& off, int& valid, int& idx) const {
        a = std::max(a, 0);
        b = std::min(b, L);
        idx = int(sz.size()) - 1;
        for (size_t i = 0; i < sz.size(); ++i)
            if (sz[i] >= b - a) { idx = int(i); break; }
        int n = sz[size_t(idx)];
        if (n >= L) { off = 0; valid = L; return; }
        off = std::min(std::max(a - (n - (b - a)) / 2, 0), L - n);   // n >= b - a: same as Python's floor division
        valid = n;
    }

    void region(const int32_t* bbox, int& y0, int& x0, int& vh, int& vw, int& bi) const {
        int bx0 = 0, by0 = 0, bx1 = W, by1 = H;
        if (bbox && bbox[2] > bbox[0] && bbox[3] > bbox[1]) { bx0 = bbox[0]; by0 = bbox[1]; bx1 = bbox[2]; by1 = bbox[3]; }
        int ih, iw;
        fit(bx0, bx1, W, ws, x0, vw, iw);
        fit(by0, by1, H, hs, y0, vh, ih);
        bi = ih * int(ws.size()) + iw;
    }

    bool init(const NvfxDetailDesc* desc);
    bool load_arch();
    bool build_dml(Bucket& b);
    bool init_dml_all();
    bool make_ort(Bucket& b);
    bool prepare(const NvfxDetailFrame* f, int& si, Bucket*& b, Consts& c);
    void record(ID3D12GraphicsCommandList2* l, int si, Bucket& b, const Consts& c);
    void finish_frame(ID3D12GraphicsCommandList2* l, int si);
    void release();
};

// ------------------------------------------------------------------ architecture + weights from the ONNX file
bool NvfxDetail::load_arch() {
    std::map<std::string, Init> inits;
    std::map<std::string, std::string> meta;
    if (!parse_model(model, inits, meta)) { set_error("model: not a readable ONNX protobuf"); return false; }
    auto num = [&](const char* key, double def) {
        auto it = meta.find(key);
        return it == meta.end() ? def : std::strtod(it->second.c_str(), nullptr);
    };
    if (meta.find("nvfx.arch") == meta.end()) {
        set_error("model has no nvfx.* metadata: export it with native/infer/export_onnx.py");
        return false;
    }
    if (num("nvfx.use_prev", 0) != 0 || num("nvfx.use_aux", 1) != 1) {
        set_error("model: only use_aux=1, use_prev=0 is implemented");
        return false;
    }
    Arch& a = arch;
    a.r = int(num("nvfx.trunk_stride", 4));
    a.res_scale = float(num("nvfx.res_scale", 0.1));
    a.depth_scale = float(num("nvfx.depth_scale", 0.5));
    a.motion_scale = float(num("nvfx.motion_scale", 0.125));
    a.input_clip = float(num("nvfx.input_clip", 8.0));
    a.alpha_gate = float(num("nvfx.alpha_gate", 0.0));
    a.grey_lowfreq = int(num("nvfx.grey_lowfreq", 0));
    std::string pre = inits.count("m.head.weight") ? "m." : "";
    auto push = [&](const std::string& n) {
        auto it = inits.find(pre + n);
        if (it == inits.end()) { set_error("model: initializer %s missing", (pre + n).c_str()); return false; }
        const Init& in = it->second;
        if (!in.data || (in.dtype != 10 && in.dtype != 1) || in.size != in.count() * (in.dtype == 10 ? 2 : 4)) {
            set_error("model: initializer %s is not raw fp16/fp32 data", (pre + n).c_str());
            return false;
        }
        Arch::W w;
        w.name = n;
        w.dims = in.dims;
        w.data.resize(in.count());
        if (in.dtype == 10) std::memcpy(w.data.data(), in.data, in.size);
        else for (size_t i = 0; i < w.data.size(); ++i) { float v; std::memcpy(&v, in.data + 4 * i, 4); w.data[i] = f2h(v); }
        a.w.push_back(std::move(w));
        return true;
    };
    a.n_blocks = 0;
    while (inits.count(pre + "blocks." + std::to_string(a.n_blocks) + ".c1.weight")) ++a.n_blocks;
    if (!push("head.weight") || !push("head.bias")) return false;
    for (int i = 0; i < a.n_blocks; ++i)
        for (const char* c : {".c1", ".c2"})
            if (!push("blocks." + std::to_string(i) + c + ".weight") || !push("blocks." + std::to_string(i) + c + ".bias"))
                return false;
    if (!push("tail.weight") || !push("tail.bias") || !push("hr1.weight") || !push("hr1.bias") ||
        !push("hr2.weight") || !push("hr2.bias"))
        return false;
    const auto& hd = a.w[0].dims;
    a.width = int(hd[0]);
    if (hd.size() != 4 || hd[1] != int64_t(a.cin) * a.r * a.r || hd[2] != 3) {
        set_error("model: head conv shape does not match 7 inputs x stride %d", a.r);
        return false;
    }
    const auto& tdm = a.w[size_t(2 + 4 * a.n_blocks)].dims;
    a.hr_feats = int(tdm[0]) / (a.r * a.r);
    const auto& h1 = a.w[size_t(4 + 4 * a.n_blocks)].dims;
    a.hr_hidden = int(h1[0]);
    a.hr_k = int(h1[2]);
    if (h1[1] != a.hr_feats + 3) { set_error("model: hr1 expects %lld inputs", static_cast<long long>(h1[1])); return false; }
    return true;
}

// ------------------------------------------------------------------ DML backend: one compiled graph per bucket
bool NvfxDetail::build_dml(Bucket& b) {
    const Arch& a = arch;
    bool nhwc = (flags & NVFX_DETAIL_DML_NHWC) != 0;
    Graph g;
    g.dev = dml.Get();
    g.nhwc = nhwc;
    UINT th = UINT(b.ph / a.r), tw = UINT(b.pw / a.r);
    Val x0 = {-1, 0, g.td(1, UINT(a.cin * a.r * a.r), th, tw, false, false)};
    Val rgb = {-1, 1, g.td(1, 3, UINT(b.ph), UINT(b.pw), false, false)};
    UINT wi = 2;
    Val x = g.conv(x0, UINT(a.width), 3, wi, true, nhwc);
    wi += 2;
    for (int i = 0; i < a.n_blocks; ++i) {
        Val h = g.conv(x, UINT(a.width), 3, wi, true, nhwc);
        wi += 2;
        Val h2 = g.conv(h, UINT(a.width), 3, wi, false, nhwc);
        wi += 2;
        x = g.addv(x, g.scale(h2, a.res_scale));                 // x + res_scale * c2(relu(c1(x)))
    }
    Val t = g.conv(x, UINT(a.hr_feats * a.r * a.r), 3, wi, true, nhwc);   // relu(shuffle(y)) == shuffle(relu(y))
    wi += 2;
    Val f = g.depth_to_space(t, UINT(a.r));
    Val cat = g.join(f, rgb);
    Val h1 = g.conv(cat, UINT(a.hr_hidden), UINT(a.hr_k), wi, true, nhwc);
    wi += 2;
    Val d = g.conv(h1, 3, UINT(a.hr_k), wi, false, false);       // graph output: NCHW for the HLSL passes
    wi += 2;
    if (g.failed) return false;
    DML_EXECUTION_FLAGS ef = DML_EXECUTION_FLAG_NONE;
    if (flags & NVFX_DETAIL_DML_HALF_COMPUTE) ef |= DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
    if (flags & NVFX_DETAIL_DML_NO_METACMD) ef |= DML_EXECUTION_FLAG_DISABLE_META_COMMANDS;
    b.op = g.compile(dml1.Get(), wi, d, ef);
    if (!b.op) return false;
    DML_BINDING_PROPERTIES p = b.op->GetBindingProperties();
    b.temp_size = p.TemporaryResourceSize;
    b.desc_count = p.RequiredDescriptorCount;
    if (p.PersistentResourceSize &&
        !ctx->create_buffer(b.persist, p.PersistentResourceSize, D3D12_HEAP_TYPE_DEFAULT, true, "detail_dml_persistent"))
        return false;
    vram += b.persist.size;
    return true;
}

// weights upload, shared temp, descriptor heap ranges, per-bucket initialization, execution binding tables
bool NvfxDetail::init_dml_all() {
    ID3D12Device* dev = ctx->device.Get();
    const Arch& a = arch;
    // weights: one buffer, 256-byte aligned tensors in graph-input order
    std::vector<UINT64> off;
    UINT64 total = 0;
    for (const auto& w : a.w) {
        off.push_back(total);
        total += (w.data.size() * 2 + 255) & ~UINT64(255);
    }
    Buffer up;
    if (!ctx->create_buffer(weights, total, D3D12_HEAP_TYPE_DEFAULT, true, "detail_weights") ||
        !ctx->create_buffer(up, total, D3D12_HEAP_TYPE_UPLOAD, false, "detail_weights_up"))
        return false;
    vram += weights.size;
    for (size_t i = 0; i < a.w.size(); ++i)
        std::memcpy(static_cast<char*>(up.map) + off[i], a.w[i].data.data(), a.w[i].data.size() * 2);
    ID3D12GraphicsCommandList2* l = ctx->begin();
    if (!l) return false;
    l->CopyBufferRegion(weights.res.Get(), 0, up.res.Get(), 0, total);
    if (!ctx->submit() || !ctx->wait_idle()) return false;

    // initializers: descriptor needs and temp sizes
    std::vector<ComPtr<IDMLOperatorInitializer>> inits(buckets.size());
    UINT64 tmax = 0;
    UINT dbase = kStaticDesc + kFrames * kFrameDesc;
    for (size_t i = 0; i < buckets.size(); ++i) {
        Bucket& b = buckets[i];
        IDMLCompiledOperator* ops[] = {b.op.Get()};
        if (!ok(dml->CreateOperatorInitializer(1, ops, IID_PPV_ARGS(&inits[i])), "CreateOperatorInitializer")) return false;
        DML_BINDING_PROPERTIES ip = inits[i]->GetBindingProperties();
        b.init_temp_size = ip.TemporaryResourceSize;
        b.desc_count = std::max({b.desc_count, ip.RequiredDescriptorCount, 1u});
        b.desc_base = dbase;
        dbase += b.desc_count;
        tmax = std::max({tmax, b.temp_size, b.init_temp_size});
    }
    if (tmax && !ctx->create_buffer(temp, tmax, D3D12_HEAP_TYPE_DEFAULT, true, "detail_dml_temp")) return false;
    vram += temp.size;

    // shader-visible heap: static + per-frame + DML binding tables
    D3D12_DESCRIPTOR_HEAP_DESC hd = {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, dbase, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    if (!ok(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "CreateDescriptorHeap")) return false;

    const size_t nin = a.w.size() + 2;
    for (size_t i = 0; i < buckets.size(); ++i) {
        Bucket& b = buckets[i];
        DML_BINDING_TABLE_DESC tdsc = {inits[i].Get(), cpu(b.desc_base), gpu(b.desc_base), b.desc_count};
        if (!ok(dml->CreateBindingTable(&tdsc, IID_PPV_ARGS(&b.bt)), "CreateBindingTable")) return false;
        std::vector<DML_BUFFER_BINDING> wb(nin, DML_BUFFER_BINDING{nullptr, 0, 0});   // x0, rgb: not owned by DML
        for (size_t j = 0; j < a.w.size(); ++j) wb[j + 2] = {weights.res.Get(), off[j], a.w[j].data.size() * 2};
        DML_BUFFER_ARRAY_BINDING arr = {UINT(nin), wb.data()};
        DML_BINDING_DESC in = {DML_BINDING_TYPE_BUFFER_ARRAY, &arr};
        b.bt->BindInputs(1, &in);
        DML_BUFFER_BINDING pb = {b.persist.res.Get(), 0, b.persist.size};
        DML_BINDING_DESC out = {b.persist.res ? DML_BINDING_TYPE_BUFFER : DML_BINDING_TYPE_NONE, b.persist.res ? &pb : nullptr};
        b.bt->BindOutputs(1, &out);
        DML_BUFFER_BINDING tb = {temp.res.Get(), 0, b.init_temp_size};
        DML_BINDING_DESC tbd = {DML_BINDING_TYPE_BUFFER, &tb};
        if (b.init_temp_size) b.bt->BindTemporaryResource(&tbd);
        l = ctx->begin();
        if (!l) return false;
        ID3D12DescriptorHeap* h = heap.Get();
        l->SetDescriptorHeaps(1, &h);
        recorder->RecordDispatch(l, inits[i].Get(), b.bt.Get());
        if (!ctx->submit() || !ctx->wait_idle()) return false;

        // execution bindings
        DML_BINDING_TABLE_DESC edsc = {b.op.Get(), cpu(b.desc_base), gpu(b.desc_base), b.desc_count};
        if (!ok(b.bt->Reset(&edsc), "IDMLBindingTable::Reset")) return false;
        UINT64 n_x0 = UINT64(a.cin) * b.ph * b.pw * 2, n3 = UINT64(3) * b.ph * b.pw * 2;
        DML_BUFFER_BINDING bx = {buf_a.res.Get(), 0, (n_x0 + 3) & ~3ull}, brgb = {buf_b.res.Get(), 0, (n3 + 3) & ~3ull};
        std::vector<DML_BINDING_DESC> ins(nin, DML_BINDING_DESC{DML_BINDING_TYPE_NONE, nullptr});
        ins[0] = {DML_BINDING_TYPE_BUFFER, &bx};
        ins[1] = {DML_BINDING_TYPE_BUFFER, &brgb};
        b.bt->BindInputs(UINT(nin), ins.data());
        DML_BUFFER_BINDING bo = {draw.res.Get(), 0, (n3 + 3) & ~3ull};
        DML_BINDING_DESC outs = {DML_BINDING_TYPE_BUFFER, &bo};
        b.bt->BindOutputs(1, &outs);
        DML_BUFFER_BINDING te = {temp.res.Get(), 0, b.temp_size};
        DML_BINDING_DESC ted = {DML_BINDING_TYPE_BUFFER, &te};
        if (b.temp_size) b.bt->BindTemporaryResource(&ted);
        DML_BINDING_DESC pd = {DML_BINDING_TYPE_BUFFER, &pb};
        if (b.persist.res) b.bt->BindPersistentResource(&pd);
        b.ready = true;
    }
    return true;
}

// ------------------------------------------------------------------ ORT backend: one session per bucket
bool NvfxDetail::make_ort(Bucket& b) {
    const OrtApi* api = g_rt.api;
    OrtSessionOptions* so = nullptr;
    if (!ORT(api->CreateSessionOptions(&so))) return false;
    bool good = ORT(api->DisableMemPattern(so)) && ORT(api->SetSessionExecutionMode(so, ORT_SEQUENTIAL)) &&
                ORT(api->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL)) &&
                ORT(api->AddFreeDimensionOverrideByName(so, "H", b.ph)) &&
                ORT(api->AddFreeDimensionOverrideByName(so, "W", b.pw)) &&
                ORT(api->AddSessionConfigEntry(so, "session.disable_cpu_ep_fallback", "1")) &&
                ORT(api->SetIntraOpNumThreads(so, 1));
    if (good && (flags & NVFX_DETAIL_ORT_NO_FUSION))
        good = ORT(api->AddSessionConfigEntry(so, "ep.dml.disable_graph_fusion", "1"));
    if (good) good = ORT(g_rt.dml_api->SessionOptionsAppendExecutionProvider_DML1(so, dml.Get(), ctx->queue.Get()));
    if (good) good = ORT(api->CreateSessionFromArray(g_rt.env, model.data(), model.size(), so, &b.sess));
    api->ReleaseSessionOptions(so);
    if (!good) return false;
    const int64_t s3[4] = {1, 3, b.ph, b.pw}, s4[4] = {1, 4, b.ph, b.pw};
    size_t n3 = size_t(3) * b.ph * b.pw * 2, n4 = size_t(4) * b.ph * b.pw * 2;
    const ONNXTensorElementDataType f16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    if (!ORT(api->CreateTensorWithDataAsOrtValue(mem, a_rgb, n3, s3, 4, f16, &b.v_rgb)) ||
        !ORT(api->CreateTensorWithDataAsOrtValue(mem, a_aux, n4, s4, 4, f16, &b.v_aux)) ||
        !ORT(api->CreateTensorWithDataAsOrtValue(mem, a_delta, n3, s3, 4, f16, &b.v_delta)) ||
        !ORT(api->CreateIoBinding(b.sess, &b.bind)) || !ORT(api->BindInput(b.bind, "rgb", b.v_rgb)) ||
        !ORT(api->BindInput(b.bind, "aux", b.v_aux)) || !ORT(api->BindOutput(b.bind, "delta", b.v_delta)))
        return false;
    // first run: weight upload + DML graph compile happen here, not on the first real frame
    if (!ORT(api->RunWithBinding(b.sess, run_opts, b.bind))) return false;
    b.ready = true;
    return true;
}

// ------------------------------------------------------------------ creation
bool NvfxDetail::init(const NvfxDetailDesc* desc) {
    W = desc->width;
    H = desc->height;
    scale = desc->process_scale;
    flags = desc->flags;
    use_ort = (flags & NVFX_DETAIL_BACKEND_ORT) != 0;
    if (W < 64 || H < 64 || W > 16384 || H > 16384) { set_error("frame size %dx%d out of range", W, H); return false; }
    if (scale == 0.5f) k = 2;
    else if (scale == 1.0f) k = 1;
    else { set_error("process_scale must be 0.5 or 1.0"); return false; }
    ID3D12Device* dev = ctx->device.Get();
    D3D12_FEATURE_DATA_D3D12_OPTIONS4 o4 = {};
    if (FAILED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &o4, sizeof o4)) || !o4.Native16BitShaderOpsSupported) {
        set_error("device lacks native 16-bit shader ops (SM 6.2)");
        return false;
    }

    std::wstring wp;
    {
        int n = MultiByteToWideChar(CP_UTF8, 0, desc->model_path, -1, nullptr, 0);
        wp.resize(size_t(std::max(n, 1)));
        MultiByteToWideChar(CP_UTF8, 0, desc->model_path, -1, wp.data(), n);
    }
    FILE* f = nullptr;
    if (_wfopen_s(&f, wp.c_str(), L"rb") || !f) { set_error("cannot open model %s", desc->model_path); return false; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    model.resize(size_t(std::max(sz, 0L)));
    size_t rd = fread(model.data(), 1, model.size(), f);
    fclose(f);
    if (sz <= 0 || rd != model.size()) { set_error("cannot read model %s", desc->model_path); return false; }
    if (!load_arch()) return false;
    // ScreenDetail quantum: max(64, lcm(trunk_stride, grey_lowfreq) / process_scale), doubled until it divides
    {
        int r = arch.r, g = arch.grey_lowfreq > 0 ? arch.grey_lowfreq : 1;
        int l = r / std::gcd(r, g) * g;
        q = std::max(64, l * k);
        while ((q / k) % l) q *= 2;
    }
    hs = sizes(H);
    ws = sizes(W);

    if (use_ort) {
        if (!ort_acquire()) return false;
        ort_held = true;
    } else if (!load_dml()) {
        return false;
    }
    if (!ok(g_rt.create_dml(dev, DML_CREATE_DEVICE_FLAG_NONE, DML_FEATURE_LEVEL_5_0, IID_PPV_ARGS(&dml)), "DMLCreateDevice1"))
        return false;
    if (!ok(dml.As(&dml1), "IDMLDevice1")) return false;
    if (!ok(dml->CreateCommandRecorder(IID_PPV_ARGS(&recorder)), "CreateCommandRecorder")) return false;

    // buffers sized for the largest bucket, shared by all (only one bucket runs at a time)
    uint64_t pmax = uint64_t(hs.back() / k) * uint64_t(ws.back() / k);
    auto mk = [&](Buffer& b, uint64_t bytes, const char* name) {
        if (!ctx->create_buffer(b, (bytes + 3) & ~3ull, D3D12_HEAP_TYPE_DEFAULT, true, name)) return false;
        vram += b.size;
        return true;
    };
    if (!mk(buf_a, uint64_t(use_ort ? 3 : arch.cin) * pmax * 2, "detail_in_a") ||
        !mk(buf_b, uint64_t(use_ort ? 4 : 3) * pmax * 2, "detail_in_b") || !mk(delta, 3 * pmax * 2, "detail_delta"))
        return false;
    if (!use_ort) {
        uint64_t gk = uint64_t(std::max(arch.grey_lowfreq, 1));
        if (!mk(alpha, pmax * 2, "detail_alpha") || !mk(draw, 3 * pmax * 2, "detail_draw") ||
            !mk(coarse, 3 * (pmax / (gk * gk) + 1) * 4, "detail_coarse"))
            return false;
    }

    // root signature: constants only, heap directly indexed (same scheme as native/sim)
    D3D12_ROOT_PARAMETER1 rp = {};
    rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rp.Constants = {0, 0, kConsts};
    rp.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC rsd = {};
    rsd.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    rsd.Desc_1_1.NumParameters = 1;
    rsd.Desc_1_1.pParameters = &rp;
    rsd.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED;
    ComPtr<ID3DBlob> blob, err;
    if (FAILED(D3D12SerializeVersionedRootSignature(&rsd, &blob, &err))) {
        set_error("root signature: %s", err ? static_cast<const char*>(err->GetBufferPointer()) : "?");
        return false;
    }
    if (!ok(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&rs)), "CreateRootSignature"))
        return false;
    auto pso = [&](const char* name, ComPtr<ID3D12PipelineState>& out) {
        for (const auto& s : kDetailShaders) {
            if (std::strcmp(s.name, name) != 0) continue;
            D3D12_COMPUTE_PIPELINE_STATE_DESC d = {};
            d.pRootSignature = rs.Get();
            d.CS = {s.data, s.size};
            return ok(dev->CreateComputePipelineState(&d, IID_PPV_ARGS(&out)), name);
        }
        set_error("internal: shader %s not embedded", name);
        return false;
    };
    const char* pre_name = use_ort ? (k == 2 ? "pre_s2" : "pre_s1") : (k == 2 ? "pre_dml_s2" : "pre_dml_s1");
    if (!pso(pre_name, pso_pre) || !pso(k == 2 ? "post_s2" : "post_s1", pso_post) || !pso("lowfreq", pso_lowfreq) ||
        !pso("fin", pso_fin))
        return false;
    inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    D3D12_QUERY_HEAP_DESC qd = {D3D12_QUERY_HEAP_TYPE_TIMESTAMP, kFrames * 4, 0};
    if (!ok(dev->CreateQueryHeap(&qd, IID_PPV_ARGS(&qheap)), "CreateQueryHeap")) return false;
    if (!ctx->create_buffer(readback, kFrames * kRbStride, D3D12_HEAP_TYPE_READBACK, false, "detail_timestamps")) return false;
    std::memset(readback.map, 0, kFrames * kRbStride);

    buckets.resize(hs.size() * ws.size());
    for (size_t i = 0; i < hs.size(); ++i)
        for (size_t j = 0; j < ws.size(); ++j) {
            Bucket& b = buckets[i * ws.size() + j];
            b.ch = hs[i];
            b.cw = ws[j];
            b.ph = hs[i] / k;
            b.pw = ws[j] / k;
        }

    if (use_ort) {
        D3D12_DESCRIPTOR_HEAP_DESC hd = {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, UINT(kStaticDesc + kFrames * kFrameDesc),
                                         D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
        if (!ok(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "CreateDescriptorHeap")) return false;
        const OrtApi* api = g_rt.api;
        if (!ORT(api->CreateMemoryInfo("DML", OrtDeviceAllocator, 0, OrtMemTypeDefault, &mem))) return false;
        if (!ORT(api->CreateRunOptions(&run_opts))) return false;
        if (!ORT(api->AddRunConfigEntry(run_opts, "disable_synchronize_execution_providers", "1"))) return false;
        if (!ORT(g_rt.dml_api->CreateGPUAllocationFromD3DResource(buf_a.res.Get(), &a_rgb)) ||
            !ORT(g_rt.dml_api->CreateGPUAllocationFromD3DResource(buf_b.res.Get(), &a_aux)) ||
            !ORT(g_rt.dml_api->CreateGPUAllocationFromD3DResource(delta.res.Get(), &a_delta)))
            return false;
        if (!(flags & NVFX_DETAIL_LAZY_BUCKETS))
            for (auto& b : buckets)
                if (!make_ort(b)) return false;
    } else {
        for (auto& b : buckets)
            if (!build_dml(b)) return false;
        if (!init_dml_all()) return false;
    }
    // static descriptors (raw UAVs of the network / scratch buffers)
    Buffer* bufs[6] = {&buf_a, &buf_b, &delta, &alpha, &draw, &coarse};
    for (int i = 0; i < 6; ++i) {
        if (!bufs[i]->res) continue;
        D3D12_UNORDERED_ACCESS_VIEW_DESC d = {};
        d.Format = DXGI_FORMAT_R32_TYPELESS;
        d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        d.Buffer.NumElements = UINT(bufs[i]->size / 4);
        d.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        dev->CreateUnorderedAccessView(bufs[i]->res.Get(), nullptr, &d, cpu(UINT(i)));
    }
    return ctx->wait_idle();
}

void NvfxDetail::release() {
    if (ctx) ctx->wait_idle();
    if (g_rt.api) {
        const OrtApi* api = g_rt.api;
        for (auto& b : buckets) {
            if (b.bind) api->ReleaseIoBinding(b.bind);
            if (b.v_rgb) api->ReleaseValue(b.v_rgb);
            if (b.v_aux) api->ReleaseValue(b.v_aux);
            if (b.v_delta) api->ReleaseValue(b.v_delta);
            if (b.sess) api->ReleaseSession(b.sess);
        }
        if (a_rgb) g_rt.dml_api->FreeGPUAllocation(a_rgb);
        if (a_aux) g_rt.dml_api->FreeGPUAllocation(a_aux);
        if (a_delta) g_rt.dml_api->FreeGPUAllocation(a_delta);
        if (run_opts) api->ReleaseRunOptions(run_opts);
        if (mem) api->ReleaseMemoryInfo(mem);
    }
    buckets.clear();
    a_rgb = a_aux = a_delta = nullptr;
    run_opts = nullptr;
    mem = nullptr;
    recorder.Reset();
    dml1.Reset();
    dml.Reset();
    if (ort_held) ort_release();
    ort_held = false;
}

// ------------------------------------------------------------------ per frame
bool NvfxDetail::prepare(const NvfxDetailFrame* f, int& si, Bucket*& bp, Consts& c) {
    ID3D12Device* dev = ctx->device.Get();
    int y0, x0, vh, vw, bi;
    region(f->bbox, y0, x0, vh, vw, bi);
    Bucket& b = buckets[size_t(bi)];
    if (!b.ready) {
        if (!use_ort) { set_error("internal: bucket not built"); return false; }
        if (!make_ort(b)) return false;
    }
    bp = &b;
    si = int(frame % kFrames);
    UINT base = UINT(kStaticDesc + si * kFrameDesc);
    auto srv = [&](void* r, DXGI_FORMAT fmt, UINT i) {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
        sd.Format = fmt;
        sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.Texture2D.MipLevels = 1;
        dev->CreateShaderResourceView(static_cast<ID3D12Resource*>(r), &sd, cpu(base + i));   // NULL -> reads 0
    };
    srv(f->color, DXGI_FORMAT_R8G8B8A8_UNORM, 0);
    srv(f->alpha, DXGI_FORMAT_R16_FLOAT, 1);
    srv(f->depth, DXGI_FORMAT_R16_FLOAT, 2);
    srv(f->motion, DXGI_FORMAT_R16G16_FLOAT, 3);
    D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
    ud.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    dev->CreateUnorderedAccessView(static_cast<ID3D12Resource*>(f->output), nullptr, &ud, cpu(base + 4));
    c = {};
    c.color_srv = base;
    c.alpha_srv = base + 1;
    c.depth_srv = base + 2;
    c.motion_srv = base + 3;
    c.out_uav = base + 4;
    c.buf_a = D_A;
    c.buf_b = D_B;
    c.delta_uav = D_DELTA;
    c.alpha_buf = D_ALPHA;
    c.draw_buf = D_DRAW;
    c.coarse_buf = D_COARSE;
    c.y0 = y0; c.x0 = x0; c.vh = vh; c.vw = vw;
    c.ph = uint32_t(b.ph); c.pw = uint32_t(b.pw);
    c.fw = uint32_t(W); c.fh = uint32_t(H);
    c.grey_k = uint32_t(std::max(arch.grey_lowfreq, 0));
    c.gate = arch.alpha_gate;
    c.clip = arch.input_clip;
    c.depth_scale = arch.depth_scale;
    c.motion_scale = arch.motion_scale;
    c.r = uint32_t(arch.r);
    Slot& s = slots[si];
    s.region[0] = x0; s.region[1] = y0; s.region[2] = vw; s.region[3] = vh;
    s.crop[0] = b.cw; s.crop[1] = b.ch;
    return true;
}

// DML backend: the whole frame into one list
void NvfxDetail::record(ID3D12GraphicsCommandList2* l, int si, Bucket& b, const Consts& c) {
    UINT q0 = UINT(si * 4);
    ID3D12DescriptorHeap* h = heap.Get();
    auto bind = [&]() {
        l->SetDescriptorHeaps(1, &h);
        l->SetComputeRootSignature(rs.Get());
        l->SetComputeRoot32BitConstants(0, kConsts, &c, 0);
    };
    bind();
    uav_barrier(l);
    l->EndQuery(qheap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, q0 + 0);
    l->SetPipelineState(pso_pre.Get());
    l->Dispatch(UINT((b.pw / 2 + 7) / 8), UINT((b.ph + 7) / 8), 1);
    uav_barrier(l);
    l->EndQuery(qheap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, q0 + 1);
    recorder->RecordDispatch(l, b.op.Get(), b.bt.Get());
    uav_barrier(l);
    l->EndQuery(qheap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, q0 + 2);
    bind();   // DirectML set its own root signature / pipeline
    if (c.grey_k) {
        l->SetPipelineState(pso_lowfreq.Get());
        l->Dispatch(UINT((b.pw / int(c.grey_k) + 7) / 8), UINT((b.ph / int(c.grey_k) + 7) / 8), 1);
        uav_barrier(l);
    }
    l->SetPipelineState(pso_fin.Get());
    l->Dispatch(UINT((b.pw / 2 + 7) / 8), UINT((b.ph + 7) / 8), 1);
    uav_barrier(l);
    l->SetPipelineState(pso_post.Get());
    l->Dispatch(UINT((W + 7) / 8), UINT((H + 7) / 8), 1);
    l->EndQuery(qheap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, q0 + 3);
}

void NvfxDetail::finish_frame(ID3D12GraphicsCommandList2* l, int si) {
    UINT q0 = UINT(si * 4);
    l->ResolveQueryData(qheap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, q0, 4, readback.res.Get(), uint64_t(si) * kRbStride);
    D3D12_WRITEBUFFERIMMEDIATE_PARAMETER p = {readback.va() + uint64_t(si) * kRbStride + 32, uint32_t(frame + 1)};
    D3D12_WRITEBUFFERIMMEDIATE_MODE m = D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT;
    l->WriteBufferImmediate(1, &p, &m);   // marker: slot si now holds the timestamps of frame `frame + 1`
}

// ------------------------------------------------------------------ C API
extern "C" {

NVFX_API NvfxResult nvfx_detail_create(NvfxContext* ctx, const NvfxDetailDesc* desc, NvfxDetail** out) {
    nvfx::clear_error();
    if (!ctx || !desc || !desc->model_path || !out) { set_error("NULL argument"); return NVFX_ERR_INVALID_ARG; }
    *out = nullptr;
    auto* d = new (std::nothrow) NvfxDetail();
    if (!d) return NVFX_ERR_OUT_OF_MEMORY;
    d->ctx = ctx;
    if (!d->init(desc)) {
        std::string e = nvfx_last_error();
        d->release();
        delete d;
        set_error("%s", e.c_str());
        return NVFX_ERR_DEVICE;
    }
    *out = d;
    return NVFX_OK;
}

NVFX_API void nvfx_detail_destroy(NvfxDetail* d) {
    if (!d) return;
    d->release();
    delete d;
}

NVFX_API NvfxResult nvfx_detail_region(NvfxDetail* d, const int32_t bbox[4], int32_t out_region[6]) {
    if (!d || !out_region) return NVFX_ERR_INVALID_ARG;
    int y0, x0, vh, vw, bi;
    d->region(bbox, y0, x0, vh, vw, bi);
    const Bucket& b = d->buckets[size_t(bi)];
    int32_t r[6] = {x0, y0, vw, vh, b.cw, b.ch};
    std::memcpy(out_region, r, sizeof r);
    return NVFX_OK;
}

NVFX_API uint64_t nvfx_detail_vram_bytes(NvfxDetail* d) { return d ? d->vram + d->readback.size : 0; }

NVFX_API NvfxResult nvfx_detail_run(NvfxDetail* d, const NvfxDetailFrame* f) {
    nvfx::clear_error();
    if (!d || !f || !f->color || !f->output) { set_error("NULL argument"); return NVFX_ERR_INVALID_ARG; }
    if (f->color == f->output) { set_error("output must not alias color"); return NVFX_ERR_INVALID_ARG; }
    auto t0 = std::chrono::steady_clock::now();
    NvfxContext* ctx = d->ctx;
    int si = int(d->frame % kFrames);
    if (d->slots[si].fence && !ctx->wait(d->slots[si].fence)) return NVFX_ERR_DEVICE;   // only with kFrames in flight
    Bucket* b = nullptr;
    Consts c;
    if (!d->prepare(f, si, b, c)) return NVFX_ERR_DEVICE;
    ID3D12GraphicsCommandList2* l = ctx->begin();
    if (!l) return NVFX_ERR_DEVICE;
    if (!d->use_ort) {
        d->record(l, si, *b, c);
    } else {
        // pre list -> ORT (its own command lists, same queue) -> post list
        UINT q0 = UINT(si * 4);
        ID3D12DescriptorHeap* h = d->heap.Get();
        auto bind = [&](ID3D12PipelineState* p) {
            l->SetDescriptorHeaps(1, &h);
            l->SetComputeRootSignature(d->rs.Get());
            l->SetPipelineState(p);
            l->SetComputeRoot32BitConstants(0, kConsts, &c, 0);
        };
        l->EndQuery(d->qheap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, q0 + 0);
        bind(d->pso_pre.Get());
        l->Dispatch(UINT((b->pw / 2 + 7) / 8), UINT((b->ph + 7) / 8), 1);
        l->EndQuery(d->qheap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, q0 + 1);
        if (!ctx->submit()) return NVFX_ERR_DEVICE;
        if (!ORT(g_rt.api->RunWithBinding(b->sess, d->run_opts, b->bind))) return NVFX_ERR_DEVICE;
        l = ctx->begin();
        if (!l) return NVFX_ERR_DEVICE;
        l->EndQuery(d->qheap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, q0 + 2);
        bind(d->pso_post.Get());
        l->Dispatch(UINT((d->W + 7) / 8), UINT((d->H + 7) / 8), 1);
        l->EndQuery(d->qheap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, q0 + 3);
    }
    d->finish_frame(l, si);
    uint64_t fv = ctx->submit();
    if (!fv) return NVFX_ERR_DEVICE;
    NvfxDetail::Slot& s = d->slots[si];
    s.fence = fv;
    s.index = ++d->frame;
    s.cpu_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return NVFX_OK;
}

NVFX_API NvfxResult nvfx_detail_record(NvfxDetail* d, void* cmd_list, const NvfxDetailFrame* f) {
    nvfx::clear_error();
    if (!d || !cmd_list || !f || !f->color || !f->output) { set_error("NULL argument"); return NVFX_ERR_INVALID_ARG; }
    if (f->color == f->output) { set_error("output must not alias color"); return NVFX_ERR_INVALID_ARG; }
    if (d->use_ort) { set_error("nvfx_detail_record: not available with the ORT backend"); return NVFX_ERR_UNSUPPORTED; }
    auto t0 = std::chrono::steady_clock::now();
    ComPtr<ID3D12GraphicsCommandList2> l;
    if (FAILED(static_cast<IUnknown*>(cmd_list)->QueryInterface(IID_PPV_ARGS(&l)))) {
        set_error("command list does not support ID3D12GraphicsCommandList2");
        return NVFX_ERR_INVALID_ARG;
    }
    int si;
    Bucket* b = nullptr;
    Consts c;
    if (!d->prepare(f, si, b, c)) return NVFX_ERR_DEVICE;
    d->record(l.Get(), si, *b, c);
    d->finish_frame(l.Get(), si);
    NvfxDetail::Slot& s = d->slots[si];
    s.fence = 0;   // completion is seen through the marker only
    s.index = ++d->frame;
    s.cpu_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return NVFX_OK;
}

NVFX_API NvfxResult nvfx_detail_get_timings(NvfxDetail* d, NvfxDetailTimings* out) {
    if (!d || !out) return NVFX_ERR_INVALID_ARG;
    std::memset(out, 0, sizeof *out);
    int best = -1;
    uint32_t best_marker = 0;
    for (int i = 0; i < kFrames; ++i) {
        const auto& s = d->slots[i];
        const volatile uint32_t* mk = reinterpret_cast<const volatile uint32_t*>(
            static_cast<const char*>(d->readback.map) + i * kRbStride + 32);
        uint32_t m = *mk;
        if (s.index && m == uint32_t(s.index) && m > best_marker) { best = i; best_marker = m; }
    }
    if (best < 0) return NVFX_OK;
    const auto& s = d->slots[best];
    const uint64_t* t = reinterpret_cast<const uint64_t*>(static_cast<const char*>(d->readback.map) + best * kRbStride);
    double ms = 1000.0 / d->ctx->ts_freq;
    out->pre_ms = float(double(t[1] - t[0]) * ms);
    out->net_ms = float(double(t[2] - t[1]) * ms);
    out->post_ms = float(double(t[3] - t[2]) * ms);
    out->total_ms = float(double(t[3] - t[0]) * ms);
    out->cpu_ms = s.cpu_ms;
    std::memcpy(out->region, s.region, sizeof s.region);
    std::memcpy(out->crop, s.crop, sizeof s.crop);
    out->frame_index = s.index;
    return NVFX_OK;
}

}  // extern "C"
