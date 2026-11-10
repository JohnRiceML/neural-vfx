// nvfx_demo.exe - Gate 3 stand-alone native demo. OWNER: native app engineer.
//
// A Win32 window + D3D12 flip-model swap chain that this app creates itself and hands to nvfx through
// nvfx_context_create_from_device (the hosted path an engine uses). Per frame, ONE command list on ONE direct queue:
//   [ts0] solver step(s) (nvfx_solver_record_step, 30 Hz)            -- skipped on non-sim frames
//         renderer draw  (nvfx_renderer_record_draw, AUX, REUSE_VOLUME on non-sim frames)
//         detail         (nvfx_detail_record, neural mode)              -- raw mode: passthrough
//   [ts1] present pass   (full-screen triangle: copy frame -> back buffer + HUD overlay)
//   [ts2] resolve timestamps
// The only per-frame CPU->GPU data are small constants (solver forces, camera, HUD text: all inside the libraries'
// and this app's upload rings). The single frame-pacing wait: WaitForSingleObjectEx on the swap chain's
// frame-latency waitable object (max latency 2) at the top of the frame. See README.md.
#include <windows.h>
#include <windowsx.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwctype>
#include <deque>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "nvfx.h"
#include "nvfx_detail.h"
#include "nvfx_render.h"

#include "hud_font.h"
#include "present_ps.h"
#include "present_vs.h"

using Microsoft::WRL::ComPtr;

namespace {

// ------------------------------------------------------------------ constants
constexpr int kFrames = 3;            // per-frame-slot resources: allocator, HUD constants, timestamp slots
constexpr int kBackBuffers = 3;
constexpr int kMaxLatency = 2;        // frames the CPU may run ahead of the display
constexpr int kHudCols = 64, kHudRows = 16;
constexpr uint32_t kCbSlot = 1792;    // sizeof(HudConsts) rounded up to 256
constexpr float kBudgetSim = 2.0f, kBudgetRender = 1.5f, kBudgetDetail = 2.5f, kBudgetTotal = 6.0f;
constexpr double kPi = 3.14159265358979323846;

enum SrvSlot { SRV_NULL = 0, SRV_RENDER_COLOR = 1, SRV_DETAIL_OUT = 2, SRV_COUNT = 3 };

struct HudConsts {                    // mirrors cbuffer Hud in present.hlsl
    uint32_t font[32][4];
    uint32_t text[64][4];
    float palette[4][4];
    float backdrop[4];
    float clear[4];
    int32_t box[4];
    int32_t grid[4];
    int32_t dims[4];
};
static_assert(sizeof(HudConsts) == 1680, "HudConsts layout");
static_assert(sizeof(HudConsts) <= kCbSlot, "HudConsts slot");

// ------------------------------------------------------------------ small helpers
double now_ms() {
    static LARGE_INTEGER f = [] { LARGE_INTEGER x; QueryPerformanceFrequency(&x); return x; }();
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return double(c.QuadPart) * 1000.0 / double(f.QuadPart);
}

std::string to_utf8(const std::wstring& w) {
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), int(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(size_t(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), int(w.size()), s.data(), n, nullptr, nullptr);
    return s;
}

const char* result_name(NvfxResult r) {
    switch (r) {
    case NVFX_OK: return "NVFX_OK";
    case NVFX_ERR_INVALID_ARG: return "NVFX_ERR_INVALID_ARG";
    case NVFX_ERR_DEVICE: return "NVFX_ERR_DEVICE";
    case NVFX_ERR_UNSUPPORTED: return "NVFX_ERR_UNSUPPORTED";
    case NVFX_ERR_OUT_OF_MEMORY: return "NVFX_ERR_OUT_OF_MEMORY";
    case NVFX_ERR_INTERNAL: return "NVFX_ERR_INTERNAL";
    }
    return "NVFX_ERR_?";
}

// every library call goes through this: prints the call, the result and nvfx_last_error()
bool nv_ok(NvfxResult r, const char* what) {
    if (r == NVFX_OK) return true;
    std::fprintf(stderr, "[nvfx_demo] %s failed: %s (%d): %s\n", what, result_name(r), int(r), nvfx_last_error());
    std::fflush(stderr);
    return false;
}

bool hr_ok(HRESULT hr, const char* what) {
    if (SUCCEEDED(hr)) return true;
    std::fprintf(stderr, "[nvfx_demo] %s failed: HRESULT 0x%08lx\n", what, static_cast<unsigned long>(hr));
    std::fflush(stderr);
    return false;
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
    D3D12_RESOURCE_BARRIER x = {};
    x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    x.Transition.pResource = r;
    x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    x.Transition.StateBefore = a;
    x.Transition.StateAfter = b;
    return x;
}

// ------------------------------------------------------------------ statistics
struct Stat { size_t n = 0; double mean = 0, p10 = 0, p50 = 0, p95 = 0, p99 = 0, max = 0; };

double pct_sorted(const std::vector<float>& s, double p) {
    if (s.empty()) return 0.0;
    double x = p / 100.0 * double(s.size() - 1);
    size_t i = size_t(x);
    double f = x - double(i);
    return i + 1 < s.size() ? s[i] * (1.0 - f) + s[i + 1] * f : s[i];
}

Stat stat_of(std::vector<float> v) {
    Stat st;
    st.n = v.size();
    if (v.empty()) return st;
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (float x : v) sum += x;
    st.mean = sum / double(v.size());
    st.p10 = pct_sorted(v, 10);
    st.p50 = pct_sorted(v, 50);
    st.p95 = pct_sorted(v, 95);
    st.p99 = pct_sorted(v, 99);
    st.max = v.back();
    return st;
}

// mean of the slowest 1 % of frame times -> fps (Python app3d convention)
double one_pct_low_fps(std::vector<float> ms) {
    if (ms.empty()) return 0.0;
    std::sort(ms.begin(), ms.end(), std::greater<float>());
    size_t k = std::max<size_t>(1, ms.size() / 100);
    double s = 0;
    for (size_t i = 0; i < k; ++i) s += ms[i];
    return 1000.0 / (s / double(k));
}

struct Track {                        // one GPU-ms series: rolling window for the HUD, full record for the bench
    std::deque<float> recent;
    std::vector<float> rec;
    void add(float v, bool recording) {
        recent.push_back(v);
        if (recent.size() > 90) recent.pop_front();
        if (recording) rec.push_back(v);
    }
    float median_recent() const {
        if (recent.empty()) return -1.0f;
        std::vector<float> v(recent.begin(), recent.end());
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
    }
};

// ------------------------------------------------------------------ args
struct Args {
    int width = 2560, height = 1440;
    bool size_given = false, fullscreen = false;
    std::wstring model;
    int preset = 1;
    bool neural = true;
    double bench_s = 0.0;
    std::wstring out;
    int frames = 0;
    std::wstring screenshot;
    double warmup_s = 3.0;
    bool sim_every_frame = false, no_detail = false, no_render_aux = false, solver_only = false, no_hud = false;
    bool vsync = true, debug_layer = false, orbit = false;
    int adapter = -1;
    float march_scale = 0.5f;
    double bench_dist = 2.15;
    // scripted-run / diagnostic switches (correctness runs without a human)
    std::wstring script;              // "K[-L]:event[=args];..." (see usage)
    bool no_torch = false;            // scripted runs: no fixed torch
    bool no_auto_orbit = false;       // scripted runs: fixed camera
    double yaw = 0.0;                 // initial orbit yaw (rad)
    std::vector<int> shot_at;         // extra screenshots at these frames
    std::wstring dump_aux;            // path prefix: alpha / motion images + stats of the last frame
    bool compare = false;             // last frame: back buffer vs a stand-alone nvfx_renderer_draw of the same state
};

void usage() {
    std::printf(
        "nvfx_demo.exe [options]\n"
        "  --width N --height N     client size (default 2560x1440, shrunk to fit the display)\n"
        "  --fullscreen             borderless window covering the primary monitor\n"
        "  --model PATH             screennet_fp16.onnx (default %%LOCALAPPDATA%%\\neural-vfx\\native_ref\\onnx\\)\n"
        "  --preset 1-4             campfire / plume past a sphere / twin torches + wind / box obstacle\n"
        "  --mode raw|neural        start mode (N toggles)\n"
        "  --bench SECONDS --out JSON   benchmark: vsync off, scripted input, auto-orbit, warm-up excluded\n"
        "  --warmup SECONDS         bench warm-up (default 3)\n"
        "  --bench-dist M           bench orbit distance (default 2.15: domain ~1/3 of the frame)\n"
        "  --frames N [--screenshot PATH.bmp|.ppm]   run N scripted frames, save the last back buffer\n"
        "  --sim-every-frame        sim step every frame instead of 30 Hz\n"
        "  --no-detail --no-render-aux --solver-only --no-hud   bisection switches\n"
        "  --novsync --orbit --march-scale S --adapter N --debug-layer\n"
        "scripted runs (--frames / --bench):\n"
        "  --script \"EV;EV;...\"   EV = K[-L]:name[=args], frames K..L (inclusive; L defaults to K):\n"
        "      mode | preset=P | reset | pause | hud | vsync          (key presses, at frame K)\n"
        "      puff=X,Y                                               (Shift+LMB at pixel X,Y, at frame K)\n"
        "      fire=X,Y                                               (LMB held at pixel X,Y during K..L)\n"
        "      drag=X0,Y0,X1,Y1                                       (LMB held, moving X0,Y0 -> X1,Y1 during K..L)\n"
        "      orbit=DX,DY | zoom=TICKS                               (RMB drag px per frame / wheel at K)\n"
        "  --no-torch --no-auto-orbit --yaw RAD\n"
        "  --shot-at K1,K2,...      also save <screenshot>_f<K>.<ext> at those frames\n"
        "  --dump-aux PREFIX        last frame: PREFIX_alpha.bmp, PREFIX_motion.bmp + alpha/motion/bbox stats\n"
        "  --compare                last frame: back buffer vs stand-alone nvfx_renderer_draw + download (raw)\n");
}

bool parse_args(int argc, wchar_t** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::wstring s = argv[i];
        auto val = [&]() -> const wchar_t* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "[nvfx_demo] %s needs a value\n", to_utf8(s).c_str());
                return nullptr;
            }
            return argv[++i];
        };
        const wchar_t* v = nullptr;
        if (s == L"--help" || s == L"-h") { usage(); std::exit(0); }
        else if (s == L"--width") { if (!(v = val())) return false; a.width = _wtoi(v); a.size_given = true; }
        else if (s == L"--height") { if (!(v = val())) return false; a.height = _wtoi(v); a.size_given = true; }
        else if (s == L"--fullscreen") a.fullscreen = true;
        else if (s == L"--model") { if (!(v = val())) return false; a.model = v; }
        else if (s == L"--preset") { if (!(v = val())) return false; a.preset = _wtoi(v); }
        else if (s == L"--mode") {
            if (!(v = val())) return false;
            std::wstring m = v;
            if (m == L"raw") a.neural = false;
            else if (m == L"neural") a.neural = true;
            else { std::fprintf(stderr, "[nvfx_demo] --mode raw|neural\n"); return false; }
        }
        else if (s == L"--bench") { if (!(v = val())) return false; a.bench_s = _wtof(v); }
        else if (s == L"--out") { if (!(v = val())) return false; a.out = v; }
        else if (s == L"--warmup") { if (!(v = val())) return false; a.warmup_s = _wtof(v); }
        else if (s == L"--bench-dist") { if (!(v = val())) return false; a.bench_dist = _wtof(v); }
        else if (s == L"--frames") { if (!(v = val())) return false; a.frames = _wtoi(v); }
        else if (s == L"--screenshot") { if (!(v = val())) return false; a.screenshot = v; }
        else if (s == L"--sim-every-frame") a.sim_every_frame = true;
        else if (s == L"--no-detail") a.no_detail = true;
        else if (s == L"--no-render-aux") a.no_render_aux = true;
        else if (s == L"--solver-only") a.solver_only = true;
        else if (s == L"--no-hud") a.no_hud = true;
        else if (s == L"--novsync") a.vsync = false;
        else if (s == L"--orbit") a.orbit = true;
        else if (s == L"--march-scale") { if (!(v = val())) return false; a.march_scale = float(_wtof(v)); }
        else if (s == L"--adapter") { if (!(v = val())) return false; a.adapter = _wtoi(v); }
        else if (s == L"--debug-layer") a.debug_layer = true;
        else if (s == L"--script") { if (!(v = val())) return false; a.script = v; }
        else if (s == L"--no-torch") a.no_torch = true;
        else if (s == L"--no-auto-orbit") a.no_auto_orbit = true;
        else if (s == L"--yaw") { if (!(v = val())) return false; a.yaw = _wtof(v); }
        else if (s == L"--shot-at") {
            if (!(v = val())) return false;
            for (const wchar_t* p = v; *p;) {
                a.shot_at.push_back(_wtoi(p));
                while (*p && *p != L',') ++p;
                if (*p == L',') ++p;
            }
        }
        else if (s == L"--dump-aux") { if (!(v = val())) return false; a.dump_aux = v; }
        else if (s == L"--compare") a.compare = true;
        else { std::fprintf(stderr, "[nvfx_demo] unknown argument %s (--help)\n", to_utf8(s).c_str()); return false; }
    }
    if (a.preset < 1 || a.preset > 4) { std::fprintf(stderr, "[nvfx_demo] --preset 1-4\n"); return false; }
    if (a.width < 64 || a.height < 64) { std::fprintf(stderr, "[nvfx_demo] size too small\n"); return false; }
    if (a.bench_s < 0 || a.frames < 0 || a.march_scale <= 0.0f || a.march_scale > 1.0f) {
        std::fprintf(stderr, "[nvfx_demo] invalid --bench / --frames / --march-scale\n");
        return false;
    }
    if (!a.screenshot.empty() && a.frames == 0) a.frames = 300;
    return true;
}

// ------------------------------------------------------------------ scripted input
struct ScriptEvent {
    int k0 = 0, k1 = 0;
    std::string name;
    float v[4] = {0, 0, 0, 0};
    int nv = 0;
};

bool parse_script(const std::string& s, std::vector<ScriptEvent>& out) {
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(';', i);
        if (j == std::string::npos) j = s.size();
        std::string e = s.substr(i, j - i);
        i = j + 1;
        if (e.empty()) continue;
        ScriptEvent ev;
        size_t c = e.find(':');
        if (c == std::string::npos) { std::fprintf(stderr, "[nvfx_demo] script event '%s': missing ':'\n", e.c_str()); return false; }
        std::string fr = e.substr(0, c), rest = e.substr(c + 1);
        size_t d = fr.find('-');
        ev.k0 = std::atoi(fr.c_str());
        ev.k1 = d == std::string::npos ? ev.k0 : std::atoi(fr.c_str() + d + 1);
        size_t q = rest.find('=');
        ev.name = rest.substr(0, q);
        if (q != std::string::npos) {
            const char* p = rest.c_str() + q + 1;
            while (*p && ev.nv < 4) {
                ev.v[ev.nv++] = float(std::atof(p));
                while (*p && *p != ',') ++p;
                if (*p == ',') ++p;
            }
        }
        static const char* known[] = {"mode", "preset", "reset", "pause", "hud", "vsync", "puff", "fire", "drag", "orbit", "zoom"};
        static const int need[] = {0, 1, 0, 0, 0, 0, 2, 2, 4, 2, 1};
        bool ok = false;
        for (int k = 0; k < 11; ++k)
            if (ev.name == known[k]) { ok = ev.nv >= need[k]; break; }
        if (!ok || ev.k1 < ev.k0) {
            std::fprintf(stderr, "[nvfx_demo] script event '%s': unknown name, missing args or bad frame range\n", e.c_str());
            return false;
        }
        out.push_back(ev);
    }
    return true;
}

// ------------------------------------------------------------------ scene (vfx/app3d/presets.py)
NvfxEmitter fire_emitter(const float p[3], float radius = 0.05f, float fuel = 7.0f) {
    return NvfxEmitter{{p[0], p[1], p[2]}, radius, 0.0f, 1300.0f, fuel, {0.0f, 0.35f, 0.0f}};
}
NvfxEmitter smoke_emitter(const float p[3], float radius = 0.07f, float rate = 2.5f) {
    return NvfxEmitter{{p[0], p[1], p[2]}, radius, rate, 620.0f, 0.0f, {0.0f, 0.6f, 0.0f}};
}
NvfxEmitter puff_emitter(const float p[3], float radius = 0.06f) {
    return NvfxEmitter{{p[0], p[1], p[2]}, radius, 8.0f, 520.0f, 0.0f, {0.0f, 0.0f, 0.0f}};
}
NvfxObstacle box_obstacle(float x, float y, float z, float hx, float hy, float hz) {
    return NvfxObstacle{NVFX_OBSTACLE_BOX, {x, y, z}, {hx, hy, hz}};
}
NvfxObstacle sphere_obstacle(float x, float y, float z, float r) {
    return NvfxObstacle{NVFX_OBSTACLE_SPHERE, {x, y, z}, {r, r, r}};
}

struct Preset {
    const char* name;
    std::vector<NvfxEmitter> emitters;
    std::vector<NvfxObstacle> obstacles;
    float wind[3];
    float density_dissipation, temperature_cooling;   // GAME_SIM
};

Preset make_preset(int k) {
    const float a[3] = {0.5f, 0.09f, 0.5f}, b[3] = {0.46f, 0.08f, 0.5f}, c1[3] = {0.28f, 0.1f, 0.45f},
                c2[3] = {0.62f, 0.1f, 0.58f}, d1[3] = {0.4f, 0.08f, 0.5f}, d2[3] = {0.6f, 0.08f, 0.45f};
    switch (k) {
    case 2: return {"smoke plume past a sphere", {smoke_emitter(b, 0.08f, 4.0f)},
                    {sphere_obstacle(0.52f, 0.48f, 0.5f, 0.12f)}, {0, 0, 0}, 1.2f, 5.0f};
    case 3: return {"twin torches + wind", {fire_emitter(c1, 0.05f, 10.0f), fire_emitter(c2, 0.05f, 10.0f)}, {},
                    {0.45f, 0.0f, 0.1f}, 1.2f, 5.0f};
    case 4: return {"box obstacle", {smoke_emitter(d1, 0.07f, 3.2f), fire_emitter(d2, 0.045f, 10.0f)},
                    {box_obstacle(0.48f, 0.55f, 0.5f, 0.2f, 0.03f, 0.16f)}, {0, 0, 0}, 1.2f, 5.0f};
    default: return {"campfire", {fire_emitter(a, 0.065f, 10.0f)},
                     {box_obstacle(0.5f, 0.025f, 0.5f, 0.11f, 0.025f, 0.11f)}, {0, 0, 0}, 1.2f, 5.0f};
    }
}

struct V3 { double x, y, z; };
V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V3 operator*(V3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
V3 normalize(V3 a) { double n = std::sqrt(dot(a, a)); return n > 1e-12 ? a * (1.0 / n) : V3{1, 0, 0}; }

// vfx/app3d/main.py Orbit
struct Orbit {
    V3 target{0.5, 0.42, 0.5};
    double yaw = 0.0, pitch = 0.20, dist = 2.15, fov = 40.0;

    NvfxCamera camera() const {
        double cp = std::cos(pitch);
        V3 pos = target + V3{cp * std::sin(yaw), std::sin(pitch), -cp * std::cos(yaw)} * dist;
        NvfxCamera c = {};
        c.position[0] = float(pos.x); c.position[1] = float(pos.y); c.position[2] = float(pos.z);
        c.target[0] = float(target.x); c.target[1] = float(target.y); c.target[2] = float(target.z);
        c.up[0] = 0.0f; c.up[1] = 1.0f; c.up[2] = 0.0f;
        c.fov_y_deg = float(fov);
        return c;
    }
    // World point under the cursor on the plane through the orbit target facing the camera (Orbit.cursor_point),
    // camera basis as render.cpp camera_basis (right = up x forward).
    void cursor_point(double px, double py, int w, int h, float out[3]) const {
        NvfxCamera c = camera();
        V3 o{c.position[0], c.position[1], c.position[2]};
        V3 fw = normalize(target - o);
        V3 rt = normalize(cross(V3{0, 1, 0}, fw));
        V3 up = cross(fw, rt);
        double th = std::tan(fov * kPi / 180.0 * 0.5), asp = double(w) / double(h);
        double sx = (2.0 * (px + 0.5) / w - 1.0) * th * asp;
        double sy = (1.0 - 2.0 * (py + 0.5) / h) * th;
        V3 d = normalize(fw + rt * sx + up * sy);
        double t = dot(target - o, fw) / std::max(dot(d, fw), 1e-6);
        V3 p = o + d * t;
        out[0] = float(std::clamp(p.x, 0.03, 0.97));
        out[1] = float(std::clamp(p.y, 0.03, 0.97));
        out[2] = float(std::clamp(p.z, 0.03, 0.97));
    }
};

// vfx/app3d/main.py Controls: host input state -> NvfxForces for the next sim tick
struct Controls {
    std::vector<std::pair<NvfxEmitter, int>> puffs;   // emitter, ticks left
    std::vector<NvfxImpulse> impulses;
    bool torch = false;               // cursor torch (LMB)
    float torch_pos[3] = {0.5f, 0.2f, 0.5f};
    bool fixed_torch = false;         // scripted runs: a fire held at a fixed point
    float fixed_pos[3] = {0.40f, 0.14f, 0.50f};
    std::vector<NvfxEmitter> em;                       // storage of the forces built for one tick
    std::vector<NvfxImpulse> imp;

    void clear() { puffs.clear(); impulses.clear(); torch = false; }   // fixed_torch is a run setting: kept
    void push(const float p0[3], const float p1[3], double dt) {
        double v[3], m = 0;
        for (int i = 0; i < 3; ++i) { v[i] = (p1[i] - p0[i]) / std::max(dt, 1.0 / 240.0) * 0.6; m += v[i] * v[i]; }
        m = std::sqrt(m);
        if (m > 1e-4) {
            double s = m > 2.0 ? 2.0 / m : 1.0;
            impulses.push_back(NvfxImpulse{{p1[0], p1[1], p1[2]}, 0.07f, {float(v[0] * s), float(v[1] * s), float(v[2] * s)}});
        }
    }
    void puff(const float p[3]) { puffs.push_back({puff_emitter(p), 8}); }
    NvfxForces forces(const Preset& pr) {
        em = pr.emitters;
        if (fixed_torch) em.push_back(fire_emitter(fixed_pos, 0.04f));
        if (torch) em.push_back(fire_emitter(torch_pos, 0.04f));
        for (auto& p : puffs) em.push_back(p.first);
        if (em.size() > size_t(NVFX_MAX_EMITTERS)) em.resize(NVFX_MAX_EMITTERS);
        for (auto& p : puffs) --p.second;
        puffs.erase(std::remove_if(puffs.begin(), puffs.end(), [](auto& p) { return p.second <= 0; }), puffs.end());
        size_t k = std::min<size_t>(impulses.size(), 8);
        imp.assign(impulses.end() - ptrdiff_t(k), impulses.end());
        impulses.clear();
        NvfxForces f = {};
        f.emitters = em.empty() ? nullptr : em.data();
        f.num_emitters = int32_t(em.size());
        f.impulses = imp.empty() ? nullptr : imp.data();
        f.num_impulses = int32_t(imp.size());
        f.wind[0] = pr.wind[0]; f.wind[1] = pr.wind[1]; f.wind[2] = pr.wind[2];
        return f;
    }
};

// ------------------------------------------------------------------ HUD text grid
struct HudText {
    uint8_t cell[kHudRows][kHudCols];
    int used_cols = 0, used_rows = 0;
    void clear() { std::memset(cell, 0, sizeof cell); used_cols = used_rows = 0; }
    // colour: 0 white, 1 green, 2 red, 3 amber. Returns the column after the text.
    int put(int row, int col, int color, const char* fmt, ...) {
        char buf[256];
        va_list ap;
        va_start(ap, fmt);
        std::vsnprintf(buf, sizeof buf, fmt, ap);
        va_end(ap);
        if (row < 0 || row >= kHudRows) return col;
        for (const char* p = buf; *p && col < kHudCols; ++p, ++col) {
            int ch = static_cast<unsigned char>(*p);
            if (ch >= 'a' && ch <= 'z') ch -= 32;
            if (ch < 32 || ch > 95) ch = '?';
            cell[row][col] = uint8_t((ch - 32) | (color << 6));
        }
        used_cols = std::max(used_cols, col);
        used_rows = std::max(used_rows, row + 1);
        return col;
    }
};

// ------------------------------------------------------------------ window input
struct InputState {
    bool quit = false, minimized = false;
    bool lmb = false, rmb = false;
    bool have_last = false;
    int last_x = 0, last_y = 0;
    double rmb_dx = 0, rmb_dy = 0, wheel = 0;
    bool puff_click = false;
    std::vector<UINT> keys;
};
InputState g_in;

LRESULT CALLBACK wnd_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
    case WM_CLOSE: g_in.quit = true; return 0;
    case WM_DESTROY: PostQuitMessage(0); return 0;
    case WM_SIZE: g_in.minimized = (w == SIZE_MINIMIZED); return 0;
    case WM_KILLFOCUS: g_in.lmb = g_in.rmb = false; return 0;
    case WM_LBUTTONDOWN:
        SetCapture(h);
        if (w & MK_SHIFT) g_in.puff_click = true;
        else g_in.lmb = true;
        return 0;
    case WM_LBUTTONUP:
        g_in.lmb = false;
        if (!g_in.rmb) ReleaseCapture();
        return 0;
    case WM_RBUTTONDOWN:
        SetCapture(h);
        g_in.rmb = true;
        g_in.last_x = GET_X_LPARAM(l);
        g_in.last_y = GET_Y_LPARAM(l);
        g_in.have_last = true;
        return 0;
    case WM_RBUTTONUP:
        g_in.rmb = false;
        if (!g_in.lmb) ReleaseCapture();
        return 0;
    case WM_MOUSEMOVE: {
        int x = GET_X_LPARAM(l), y = GET_Y_LPARAM(l);
        if (g_in.rmb && g_in.have_last) { g_in.rmb_dx += x - g_in.last_x; g_in.rmb_dy += y - g_in.last_y; }
        g_in.last_x = x; g_in.last_y = y; g_in.have_last = true;
        return 0;
    }
    case WM_MOUSEWHEEL: g_in.wheel += GET_WHEEL_DELTA_WPARAM(w) / 120.0; return 0;
    case WM_KEYDOWN:
        if (!(l & (1 << 30))) g_in.keys.push_back(UINT(w));   // ignore auto-repeat
        return 0;
    default: break;
    }
    return DefWindowProcW(h, m, w, l);
}

// ------------------------------------------------------------------ image output
bool write_image(const std::wstring& path, const std::vector<uint8_t>& rgba, int w, int h) {
    FILE* f = _wfopen(path.c_str(), L"wb");
    if (!f) { std::fprintf(stderr, "[nvfx_demo] cannot open %s\n", to_utf8(path).c_str()); return false; }
    std::wstring ext = path.size() >= 4 ? path.substr(path.size() - 4) : L"";
    for (auto& ch : ext) ch = wchar_t(towlower(ch));
    if (ext == L".ppm") {
        std::fprintf(f, "P6\n%d %d\n255\n", w, h);
        std::vector<uint8_t> row(size_t(w) * 3);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < 3; ++c) row[size_t(x) * 3 + c] = rgba[(size_t(y) * w + x) * 4 + c];
            std::fwrite(row.data(), 1, row.size(), f);
        }
    } else {                          // 24-bit BMP, bottom-up
        uint32_t stride = (uint32_t(w) * 3 + 3) & ~3u, img = stride * uint32_t(h);
        uint8_t hd[54] = {'B', 'M'};
        auto put32 = [&](int o, uint32_t v) { std::memcpy(hd + o, &v, 4); };
        auto put16 = [&](int o, uint16_t v) { std::memcpy(hd + o, &v, 2); };
        put32(2, 54 + img); put32(10, 54); put32(14, 40); put32(18, uint32_t(w)); put32(22, uint32_t(h));
        put16(26, 1); put16(28, 24); put32(34, img);
        std::fwrite(hd, 1, 54, f);
        std::vector<uint8_t> row(stride, 0);
        for (int y = h - 1; y >= 0; --y) {
            for (int x = 0; x < w; ++x) {
                const uint8_t* s = &rgba[(size_t(y) * w + x) * 4];
                row[size_t(x) * 3 + 0] = s[2]; row[size_t(x) * 3 + 1] = s[1]; row[size_t(x) * 3 + 2] = s[0];
            }
            std::fwrite(row.data(), 1, row.size(), f);
        }
    }
    bool ok = std::ferror(f) == 0;
    std::fclose(f);
    return ok;
}

// ------------------------------------------------------------------ the app
struct SlotMeta {
    uint64_t fence = 0;
    bool pending = false;
    int sim_steps = 0;
    bool neural = false;
    bool recording = false;
};

struct Demo {
    Args a;
    HWND hwnd = nullptr;
    int W = 0, H = 0;

    // D3D12 objects owned by the app (the "engine")
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter3> adapter;
    std::string adapter_name;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<IDXGISwapChain3> swap;
    HANDLE latency_wait = nullptr;
    bool tearing = false;
    ComPtr<ID3D12DescriptorHeap> rtv_heap, srv_heap;
    UINT rtv_inc = 0, srv_inc = 0;
    ComPtr<ID3D12Resource> back[kBackBuffers];
    ComPtr<ID3D12CommandAllocator> alloc[kFrames];
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE fence_event = nullptr;
    uint64_t fence_value = 0;
    ComPtr<ID3D12Resource> cb_upload;
    uint8_t* cb_map = nullptr;
    ComPtr<ID3D12QueryHeap> qheap;
    ComPtr<ID3D12Resource> ts_readback;
    uint64_t* ts_map = nullptr;
    double ts_freq = 1.0;
    ComPtr<ID3D12RootSignature> rs;
    ComPtr<ID3D12PipelineState> pso;
    ComPtr<ID3D12Resource> detail_out;
    SlotMeta meta[kFrames];
    uint32_t font_packed[64][2];

    // nvfx
    NvfxContext* ctx = nullptr;
    NvfxSolver* solver = nullptr;
    NvfxRenderer* renderer = nullptr;
    NvfxDetail* detail = nullptr;
    std::string detail_status = "not created";
    bool detail_broken = false;       // a record call failed: raw only from then on
    std::string model_utf8;
    int march_w = 0, march_h = 0;

    // scene / state
    Preset preset;
    int preset_id = 1;
    Orbit orbit;
    Controls ctl;
    bool neural = true, paused = false, hud_on = true, auto_orbit = false, vsync = true;
    bool need_full_draw = true;
    double sim_acc = NVFX_SIM_DT;
    uint64_t tick_id = 0, sim_steps_total = 0, frame_no = 0;
    std::map<uint64_t, std::vector<NvfxCamera>> cam_hist;   // tick -> cameras drawn during it (Pipeline.prev_camera)
    float last_cursor[3] = {0, 0, 0};
    bool have_last_cursor = false;
    std::string message;              // transient red HUD line (5 s)
    double message_t = -1e9;
    bool scripted = false;
    std::vector<ScriptEvent> script;
    // what the last frame drew (end-of-run diagnostics)
    NvfxCamera last_cam = {}, last_prev = {};
    bool last_have_prev = false;
    int32_t last_flags = 0, last_bbox[4] = {0, 0, 0, 0};
    bool last_neural = false;
    int32_t hud_box[4] = {0, 0, 0, 0};
    bool hud_drawn = false;

    // timings
    Track t_sim, t_render_full, t_render_reuse, t_detail, t_frame_sim, t_frame_other, t_present;
    std::vector<float> rec_frame_ms, rec_frame_gpu;
    std::deque<std::pair<double, float>> recent_frame_ms;  // (time, ms) for fps / 1% low over the last 3 s
    int pcg_iters = 0;
    uint64_t seen_step = 0, seen_draw = 0, seen_detail = 0;
    int32_t detail_region[4] = {0, 0, 0, 0}, detail_crop[2] = {0, 0};
    uint64_t slot_waits_blocked = 0;
    double vram_now_mb = 0, vram_peak_mb = 0, vram_budget_mb = 0;
    std::vector<float> rec_bbox_frac;
    uint64_t rec_frames_neural = 0, rec_frames_raw = 0, rec_sim_steps = 0;

    // HUD values (refreshed 4x per second so they are readable)
    double hud_t = -1e9;
    float h_sim = -1, h_rf = -1, h_rr = -1, h_det = -1, h_fs = -1, h_fo = -1, h_pres = -1, h_fps = 0, h_low = 0;

    // ---------------------------------------------------------------- setup
    bool create_window() {
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        HINSTANCE inst = GetModuleHandleW(nullptr);
        WNDCLASSEXW wc = {sizeof wc};
        wc.style = CS_HREDRAW | CS_VREDRAW;
        wc.lpfnWndProc = wnd_proc;
        wc.hInstance = inst;
        wc.hCursor = LoadCursorW(nullptr, IDC_CROSS);
        wc.lpszClassName = L"nvfx_demo";
        if (!RegisterClassExW(&wc)) { std::fprintf(stderr, "[nvfx_demo] RegisterClassEx failed\n"); return false; }

        HMONITOR mon = MonitorFromPoint(POINT{0, 0}, MONITOR_DEFAULTTOPRIMARY);
        MONITORINFO mi = {sizeof mi};
        GetMonitorInfoW(mon, &mi);
        const RECT& mr = mi.rcMonitor;
        const RECT& wr = mi.rcWork;
        int mw = mr.right - mr.left, mh = mr.bottom - mr.top;
        DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
        int x = 0, y = 0, ww = 0, wh = 0;
        W = a.width; H = a.height;
        bool borderless = a.fullscreen;
        if (a.fullscreen) {
            if (!a.size_given) { W = mw; H = mh; }
            else if (W != mw || H != mh)
                std::printf("[nvfx_demo] --fullscreen: window %dx%d, render %dx%d (stretched by DXGI)\n", mw, mh, W, H);
        } else {
            RECT r = {0, 0, W, H};
            AdjustWindowRect(&r, style, FALSE);
            int ow = r.right - r.left, oh = r.bottom - r.top;
            int aw = wr.right - wr.left, ah = wr.bottom - wr.top;
            if (ow > aw || oh > ah) {
                if (W <= mw && H <= mh) {
                    borderless = true;   // e.g. 2560x1440 on a 1440p display: borderless at native size
                    std::printf("[nvfx_demo] %dx%d + frame does not fit the work area: borderless window\n", W, H);
                } else {
                    double s = std::min(double(aw - (ow - W)) / W, double(ah - (oh - H)) / H);
                    int nw = (int(W * s) / 2) * 2, nh = (int(H * s) / 2) * 2;
                    std::printf("[nvfx_demo] %dx%d does not fit the display: using %dx%d\n", W, H, nw, nh);
                    W = nw; H = nh;
                    r = {0, 0, W, H};
                    AdjustWindowRect(&r, style, FALSE);
                    ow = r.right - r.left; oh = r.bottom - r.top;
                }
            }
            if (!borderless) {
                ww = ow; wh = oh;
                x = wr.left + std::max(0, (aw - ow) / 2);
                y = wr.top + std::max(0, (ah - oh) / 2);
            }
        }
        if (borderless) {
            style = WS_POPUP;
            if (a.fullscreen) { x = mr.left; y = mr.top; ww = mw; wh = mh; }
            else { x = mr.left + (mw - W) / 2; y = mr.top + (mh - H) / 2; ww = W; wh = H; }
        }
        hwnd = CreateWindowExW(0, wc.lpszClassName, L"nvfx_demo", style, x, y, ww, wh, nullptr, nullptr, inst, nullptr);
        if (!hwnd) { std::fprintf(stderr, "[nvfx_demo] CreateWindowEx failed\n"); return false; }
        ShowWindow(hwnd, SW_SHOW);
        UpdateWindow(hwnd);
        if (!a.fullscreen || !a.size_given) {
            RECT cr;
            GetClientRect(hwnd, &cr);
            int cw = cr.right - cr.left, ch = cr.bottom - cr.top;
            if (cw > 0 && ch > 0 && (cw != W || ch != H)) {
                std::printf("[nvfx_demo] client area is %dx%d (asked %dx%d): rendering at the client size\n", cw, ch, W, H);
                W = cw; H = ch;
            }
        }
        return true;
    }

    bool create_device() {
        UINT fflags = 0;
        if (a.debug_layer) {
            ComPtr<ID3D12Debug> dbg;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) {
                dbg->EnableDebugLayer();
                fflags = DXGI_CREATE_FACTORY_DEBUG;
            } else {
                std::printf("[nvfx_demo] D3D12 debug layer not available (Graphics Tools not installed)\n");
            }
        }
        if (!hr_ok(CreateDXGIFactory2(fflags, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2")) return false;
        ComPtr<IDXGIAdapter1> ad;
        HRESULT hr = a.adapter < 0
                         ? factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&ad))
                         : factory->EnumAdapters1(UINT(a.adapter), &ad);
        if (!hr_ok(hr, "enumerate adapter") || !hr_ok(ad.As(&adapter), "IDXGIAdapter3")) return false;
        DXGI_ADAPTER_DESC1 ds = {};
        adapter->GetDesc1(&ds);
        adapter_name = to_utf8(ds.Description);
        if (!hr_ok(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)), "D3D12CreateDevice"))
            return false;
        D3D12_COMMAND_QUEUE_DESC qd = {};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (!hr_ok(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)), "CreateCommandQueue")) return false;
        queue->SetName(L"nvfx_demo.queue");
        UINT64 f = 0;
        if (!hr_ok(queue->GetTimestampFrequency(&f), "GetTimestampFrequency")) return false;
        ts_freq = double(f);

        ComPtr<IDXGIFactory5> f5;
        BOOL allow = FALSE;
        if (SUCCEEDED(factory.As(&f5)) &&
            SUCCEEDED(f5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allow, sizeof allow)))
            tearing = allow == TRUE;

        DXGI_SWAP_CHAIN_DESC1 sd = {};
        sd.Width = UINT(W);
        sd.Height = UINT(H);
        sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;   // renderer COLOR is display-encoded RGBA8: copied as is
        sd.SampleDesc.Count = 1;
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.BufferCount = kBackBuffers;
        sd.Scaling = DXGI_SCALING_STRETCH;
        sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        sd.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
        sd.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT | (tearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0);
        ComPtr<IDXGISwapChain1> sc1;
        if (!hr_ok(factory->CreateSwapChainForHwnd(queue.Get(), hwnd, &sd, nullptr, nullptr, &sc1), "CreateSwapChainForHwnd") ||
            !hr_ok(sc1.As(&swap), "IDXGISwapChain3"))
            return false;
        factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
        if (!hr_ok(swap->SetMaximumFrameLatency(kMaxLatency), "SetMaximumFrameLatency")) return false;
        latency_wait = swap->GetFrameLatencyWaitableObject();

        D3D12_DESCRIPTOR_HEAP_DESC hd = {};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        hd.NumDescriptors = kBackBuffers;
        if (!hr_ok(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtv_heap)), "CreateDescriptorHeap(RTV)")) return false;
        rtv_inc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        for (int i = 0; i < kBackBuffers; ++i) {
            if (!hr_ok(swap->GetBuffer(UINT(i), IID_PPV_ARGS(&back[i])), "GetBuffer")) return false;
            D3D12_CPU_DESCRIPTOR_HANDLE h = rtv_heap->GetCPUDescriptorHandleForHeapStart();
            h.ptr += SIZE_T(i) * rtv_inc;
            device->CreateRenderTargetView(back[i].Get(), nullptr, h);
        }
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors = SRV_COUNT;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (!hr_ok(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&srv_heap)), "CreateDescriptorHeap(SRV)")) return false;
        srv_inc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        write_srv(nullptr, SRV_NULL);
        write_srv(nullptr, SRV_RENDER_COLOR);
        write_srv(nullptr, SRV_DETAIL_OUT);

        for (int i = 0; i < kFrames; ++i)
            if (!hr_ok(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc[i])),
                       "CreateCommandAllocator"))
                return false;
        if (!hr_ok(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc[0].Get(), nullptr, IID_PPV_ARGS(&list)),
                   "CreateCommandList"))
            return false;
        list->Close();
        if (!hr_ok(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence")) return false;
        fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!fence_event) return false;

        // HUD constants ring (upload heap, persistently mapped) and the frame timestamp ring
        D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_UPLOAD};
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = UINT64(kCbSlot) * kFrames;
        rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1; rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (!hr_ok(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                   IID_PPV_ARGS(&cb_upload)), "CreateCommittedResource(HUD constants)"))
            return false;
        D3D12_RANGE none = {0, 0};
        if (!hr_ok(cb_upload->Map(0, &none, reinterpret_cast<void**>(&cb_map)), "Map(HUD constants)")) return false;
        hp.Type = D3D12_HEAP_TYPE_READBACK;
        rd.Width = UINT64(kFrames) * 3 * sizeof(uint64_t);
        if (!hr_ok(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                   IID_PPV_ARGS(&ts_readback)), "CreateCommittedResource(timestamps)"))
            return false;
        if (!hr_ok(ts_readback->Map(0, nullptr, reinterpret_cast<void**>(&ts_map)), "Map(timestamps)")) return false;
        D3D12_QUERY_HEAP_DESC qh = {};
        qh.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        qh.Count = kFrames * 3;
        if (!hr_ok(device->CreateQueryHeap(&qh, IID_PPV_ARGS(&qheap)), "CreateQueryHeap")) return false;

        return create_present_pipeline();
    }

    void write_srv(ID3D12Resource* r, int slot) {
        D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
        d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Texture2D.MipLevels = 1;
        D3D12_CPU_DESCRIPTOR_HANDLE h = srv_heap->GetCPUDescriptorHandleForHeapStart();
        h.ptr += SIZE_T(slot) * srv_inc;
        device->CreateShaderResourceView(r, &d, h);
    }

    bool create_present_pipeline() {
        D3D12_DESCRIPTOR_RANGE range = {};
        range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        range.NumDescriptors = 1;
        range.BaseShaderRegister = 0;
        range.OffsetInDescriptorsFromTableStart = 0;
        D3D12_ROOT_PARAMETER p[2] = {};
        p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        p[0].Descriptor.ShaderRegister = 0;
        p[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p[1].DescriptorTable.NumDescriptorRanges = 1;
        p[1].DescriptorTable.pDescriptorRanges = &range;
        p[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_ROOT_SIGNATURE_DESC rsd = {};
        rsd.NumParameters = 2;
        rsd.pParameters = p;
        rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS | D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
                    D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;
        ComPtr<ID3DBlob> blob, err;
        if (FAILED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err))) {
            std::fprintf(stderr, "[nvfx_demo] D3D12SerializeRootSignature: %s\n",
                         err ? static_cast<const char*>(err->GetBufferPointer()) : "?");
            return false;
        }
        if (!hr_ok(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&rs)),
                   "CreateRootSignature"))
            return false;
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = rs.Get();
        pd.VS = {g_present_vs, sizeof g_present_vs};
        pd.PS = {g_present_ps, sizeof g_present_ps};
        D3D12_RENDER_TARGET_BLEND_DESC& bt = pd.BlendState.RenderTarget[0];
        bt.BlendEnable = FALSE;
        bt.LogicOpEnable = FALSE;
        bt.SrcBlend = D3D12_BLEND_ONE; bt.DestBlend = D3D12_BLEND_ZERO; bt.BlendOp = D3D12_BLEND_OP_ADD;
        bt.SrcBlendAlpha = D3D12_BLEND_ONE; bt.DestBlendAlpha = D3D12_BLEND_ZERO; bt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        bt.LogicOp = D3D12_LOGIC_OP_NOOP;
        bt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pd.SampleMask = UINT_MAX;
        pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pd.RasterizerState.DepthClipEnable = TRUE;
        pd.DepthStencilState.DepthEnable = FALSE;
        pd.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        pd.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        pd.DepthStencilState.StencilEnable = FALSE;
        D3D12_DEPTH_STENCILOP_DESC so = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP,
                                         D3D12_COMPARISON_FUNC_ALWAYS};
        pd.DepthStencilState.FrontFace = so;
        pd.DepthStencilState.BackFace = so;
        pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pd.NumRenderTargets = 1;
        pd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pd.SampleDesc.Count = 1;
        return hr_ok(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pso)), "CreateGraphicsPipelineState(present)");
    }

    std::wstring default_model_path() const {
        wchar_t buf[MAX_PATH] = {};
        DWORD n = GetEnvironmentVariableW(L"LOCALAPPDATA", buf, MAX_PATH);
        if (n == 0 || n >= MAX_PATH) return L"";
        return std::wstring(buf) + L"\\neural-vfx\\native_ref\\onnx\\screennet_fp16.onnx";
    }

    bool create_nvfx() {
        std::printf("[nvfx_demo] %s\n", nvfx_version_string());
        if (!nv_ok(nvfx_context_create_from_device(device.Get(), queue.Get(), &ctx), "nvfx_context_create_from_device"))
            return false;
        NvfxSolverConfig cfg;
        nvfx_solver_config_runtime(&cfg);
        if (!nv_ok(nvfx_solver_create(ctx, 64, &cfg, &solver), "nvfx_solver_create(64)")) return false;
        if (!load_preset(a.preset)) return false;
        if (a.solver_only) {
            detail_status = "off (--solver-only)";
            return true;
        }
        if (!nv_ok(nvfx_renderer_create(ctx, W, H, a.march_scale, &renderer), "nvfx_renderer_create")) return false;
        nvfx_renderer_march_size(renderer, &march_w, &march_h);
        auto* color = static_cast<ID3D12Resource*>(nvfx_renderer_texture(renderer, NVFX_RENDER_TEX_COLOR));
        if (!color) { std::fprintf(stderr, "[nvfx_demo] renderer COLOR texture is NULL\n"); return false; }
        write_srv(color, SRV_RENDER_COLOR);

        if (a.no_detail) { detail_status = "off (--no-detail)"; return true; }
        std::wstring mp = a.model.empty() ? default_model_path() : a.model;
        model_utf8 = to_utf8(mp);
        if (mp.empty() || GetFileAttributesW(mp.c_str()) == INVALID_FILE_ATTRIBUTES) {
            detail_status = "model not found: " + (model_utf8.empty() ? std::string("(LOCALAPPDATA unset)") : model_utf8);
            std::printf("[nvfx_demo] NEURAL UNAVAILABLE: %s -> raw 64^3 only (export it with native/infer/export_onnx.py "
                        "or pass --model)\n", detail_status.c_str());
            return true;
        }
        // detail output: the app's own RGBA8 UAV texture (must not alias the renderer's COLOR)
        D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = UINT64(W); rd.Height = UINT(H); rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        rd.SampleDesc.Count = 1;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (!hr_ok(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                   nullptr, IID_PPV_ARGS(&detail_out)), "CreateCommittedResource(detail output)"))
            return false;
        detail_out->SetName(L"nvfx_demo.detail_out");
        write_srv(detail_out.Get(), SRV_DETAIL_OUT);
        NvfxDetailDesc dd = {};
        dd.model_path = model_utf8.c_str();
        dd.width = W;
        dd.height = H;
        dd.process_scale = 0.5f;
        dd.flags = 0;   // shipping: raw DirectML graph (the only backend that supports nvfx_detail_record)
        double t0 = now_ms();
        NvfxResult r = nvfx_detail_create(ctx, &dd, &detail);
        if (r != NVFX_OK) {
            detail_status = std::string("nvfx_detail_create failed: ") + result_name(r) + ": " + nvfx_last_error();
            std::printf("[nvfx_demo] NEURAL UNAVAILABLE: %s -> raw 64^3 only\n", detail_status.c_str());
            detail = nullptr;
            return true;
        }
        detail_status = "ok";
        std::printf("[nvfx_demo] detail: %s (%.0f ms to create, %.0f MB owned)\n", model_utf8.c_str(), now_ms() - t0,
                    double(nvfx_detail_vram_bytes(detail)) / 1048576.0);
        return true;
    }

    bool detail_usable() const { return detail != nullptr && !detail_broken; }

    bool load_preset(int k) {
        preset_id = k;
        preset = make_preset(k);
        ctl.clear();
        NvfxSolverConfig cfg;
        nvfx_solver_config_runtime(&cfg);
        cfg.density_dissipation = preset.density_dissipation;
        cfg.temperature_cooling = preset.temperature_cooling;
        if (!nv_ok(nvfx_solver_set_config(solver, &cfg), "nvfx_solver_set_config")) return false;
        if (!nv_ok(nvfx_solver_reset(solver, preset.obstacles.empty() ? nullptr : preset.obstacles.data(),
                                     int32_t(preset.obstacles.size())), "nvfx_solver_reset"))
            return false;
        need_full_draw = true;
        cam_hist.clear();
        tick_id = 0;
        sim_acc = NVFX_SIM_DT;
        seen_step = 0;
        return true;
    }

    // Pipeline.prev_camera: the camera of the draw at the same position within the previous sim tick
    NvfxCamera prev_camera(const NvfxCamera& cam, bool& have) {
        auto& cur = cam_hist[tick_id];
        auto it = cam_hist.find(tick_id - 1);
        NvfxCamera pc = cam;
        have = false;
        if (tick_id > 0 && it != cam_hist.end() && !it->second.empty()) {
            pc = it->second[std::min(cur.size(), it->second.size() - 1)];
            have = true;
        }
        cur.push_back(cam);
        for (auto e = cam_hist.begin(); e != cam_hist.end();) {
            if (e->first + 1 < tick_id) e = cam_hist.erase(e);
            else ++e;
        }
        return pc;
    }

    // ---------------------------------------------------------------- per-frame bookkeeping
    void wait_fence(uint64_t v) {
        if (fence->GetCompletedValue() >= v) return;
        fence->SetEventOnCompletion(v, fence_event);
        WaitForSingleObject(fence_event, 30000);
    }

    void harvest_slot(int slot) {
        SlotMeta& m = meta[slot];
        if (!m.pending) return;
        if (fence->GetCompletedValue() < m.fence) {   // never expected after the latency wait; counted
            ++slot_waits_blocked;
            wait_fence(m.fence);
        }
        m.pending = false;
        const uint64_t* t = ts_map + size_t(slot) * 3;
        if (t[2] <= t[0] || t[1] < t[0]) return;
        float frame_ms = float(double(t[2] - t[0]) * 1000.0 / ts_freq);
        float pres_ms = float(double(t[2] - t[1]) * 1000.0 / ts_freq);
        t_present.add(pres_ms, m.recording);
        (m.sim_steps > 0 ? t_frame_sim : t_frame_other).add(frame_ms, m.recording);
        if (m.recording) rec_frame_gpu.push_back(frame_ms);
    }

    void poll_library_timings(bool recording) {
        NvfxSolverTimings st = {};
        if (nvfx_solver_get_timings(solver, &st) == NVFX_OK && st.step_index != 0 && st.step_index != seen_step) {
            seen_step = st.step_index;
            t_sim.add(st.total_ms, recording);
            pcg_iters = st.pcg_iters;
        }
        if (renderer) {
            NvfxRenderTimings rt = {};
            if (nvfx_renderer_get_timings(renderer, &rt) == NVFX_OK && rt.draw_index != 0 && rt.draw_index != seen_draw) {
                seen_draw = rt.draw_index;
                ((rt.flags & NVFX_RENDER_REUSE_VOLUME) ? t_render_reuse : t_render_full).add(rt.total_ms, recording);
            }
        }
        if (detail) {
            NvfxDetailTimings dt = {};
            if (nvfx_detail_get_timings(detail, &dt) == NVFX_OK && dt.frame_index != 0 && dt.frame_index != seen_detail) {
                seen_detail = dt.frame_index;
                t_detail.add(dt.total_ms, recording);
                std::memcpy(detail_region, dt.region, sizeof detail_region);
                std::memcpy(detail_crop, dt.crop, sizeof detail_crop);
            }
        }
    }

    void sample_vram() {
        DXGI_QUERY_VIDEO_MEMORY_INFO vi = {};
        if (SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &vi))) {
            vram_now_mb = double(vi.CurrentUsage) / 1048576.0;
            vram_budget_mb = double(vi.Budget) / 1048576.0;
            vram_peak_mb = std::max(vram_peak_mb, vram_now_mb);
        }
    }

    // ---------------------------------------------------------------- input
    // one frame of pointer input, from the window (interactive) or from --script
    struct Pointer {
        bool inside = false;
        double x = 0, y = 0;          // client pixels
        bool lmb = false, puff_click = false;
        double rmb_dx = 0, rmb_dy = 0, wheel = 0;
    };
    bool ptr_inside = false;
    float ptr_world[3] = {0.5f, 0.3f, 0.5f};

    bool handle_key(UINT k) {        // returns false on a fatal library error
        switch (k) {
        case 'N':
            if (!neural && !detail_usable()) { message = "N: NEURAL UNAVAILABLE, STAYING RAW"; message_t = now_ms(); }
            else neural = !neural;
            break;
        case 'C': auto_orbit = !auto_orbit; break;
        case 'R': return load_preset(preset_id);
        case VK_SPACE: paused = !paused; break;
        case 'V': vsync = !vsync; break;
        case 'H': hud_on = !hud_on; break;
        case 'G': if (ptr_inside) ctl.puff(ptr_world); break;
        case '1': case '2': case '3': case '4': return load_preset(int(k - '0'));
        default: break;
        }
        return true;
    }

    Pointer window_pointer() {
        Pointer p;
        POINT pt;
        if (GetCursorPos(&pt) && ScreenToClient(hwnd, &pt)) {
            p.x = pt.x; p.y = pt.y;
            p.inside = GetForegroundWindow() == hwnd && pt.x >= 0 && pt.y >= 0 && pt.x < W && pt.y < H;
        }
        p.lmb = g_in.lmb;
        p.puff_click = g_in.puff_click;
        p.rmb_dx = g_in.rmb_dx; p.rmb_dy = g_in.rmb_dy; p.wheel = g_in.wheel;
        g_in.puff_click = false;
        g_in.rmb_dx = g_in.rmb_dy = g_in.wheel = 0;
        return p;
    }

    // --script: pointer state and key presses of frame f
    Pointer script_pointer(uint64_t f, std::vector<UINT>& keys) {
        Pointer p;
        for (const ScriptEvent& e : script) {
            if (int64_t(f) < e.k0 || int64_t(f) > e.k1) continue;
            bool first = int64_t(f) == e.k0;
            const std::string& n = e.name;
            if (n == "mode" && first) keys.push_back('N');
            else if (n == "preset" && first) keys.push_back(UINT('0' + std::clamp(int(e.v[0]), 1, 4)));
            else if (n == "reset" && first) keys.push_back('R');
            else if (n == "pause" && first) keys.push_back(VK_SPACE);
            else if (n == "hud" && first) keys.push_back('H');
            else if (n == "vsync" && first) keys.push_back('V');
            else if (n == "puff" && first) { p.inside = true; p.x = e.v[0]; p.y = e.v[1]; p.puff_click = true; }
            else if (n == "fire") { p.inside = true; p.x = e.v[0]; p.y = e.v[1]; p.lmb = true; }
            else if (n == "drag") {
                double t = e.k1 > e.k0 ? double(int64_t(f) - e.k0) / double(e.k1 - e.k0) : 1.0;
                p.inside = true; p.lmb = true;
                p.x = e.v[0] + (e.v[2] - e.v[0]) * t;
                p.y = e.v[1] + (e.v[3] - e.v[1]) * t;
            }
            else if (n == "orbit") { p.rmb_dx += e.v[0]; p.rmb_dy += e.v[1]; }
            else if (n == "zoom" && first) p.wheel += e.v[0];
        }
        return p;
    }

    void apply_pointer(const Pointer& in, double fdt) {
        orbit.yaw -= in.rmb_dx * 0.006;
        orbit.pitch = std::clamp(orbit.pitch + in.rmb_dy * 0.004, -0.15, 1.35);
        if (in.wheel != 0) orbit.dist = std::clamp(orbit.dist * std::pow(0.9, in.wheel), 0.9, 4.5);
        ptr_inside = in.inside;
        if (in.inside) orbit.cursor_point(in.x, in.y, W, H, ptr_world);
        if (in.puff_click && in.inside) ctl.puff(ptr_world);
        ctl.torch = in.lmb && in.inside;              // LMB hold = fire at the cursor
        if (ctl.torch) std::memcpy(ctl.torch_pos, ptr_world, sizeof ptr_world);
        if (in.lmb && in.inside) {                    // LMB drag = push along the drag
            if (have_last_cursor) ctl.push(last_cursor, ptr_world, fdt);
            std::memcpy(last_cursor, ptr_world, sizeof ptr_world);
            have_last_cursor = true;
        } else {
            have_last_cursor = false;
        }
    }

    // ---------------------------------------------------------------- HUD
    void refresh_hud_values(double t) {
        if (t - hud_t < 250.0) return;
        hud_t = t;
        h_sim = t_sim.median_recent(); h_rf = t_render_full.median_recent(); h_rr = t_render_reuse.median_recent();
        h_det = t_detail.median_recent(); h_fs = t_frame_sim.median_recent(); h_fo = t_frame_other.median_recent();
        h_pres = t_present.median_recent();
        while (!recent_frame_ms.empty() && t - recent_frame_ms.front().first > 3000.0) recent_frame_ms.pop_front();
        std::vector<float> v;
        double sum = 0;
        for (auto& e : recent_frame_ms) { v.push_back(e.second); sum += e.second; }
        h_fps = v.empty() ? 0.0f : float(1000.0 * double(v.size()) / sum);
        h_low = float(one_pct_low_fps(v));
    }

    void ms_cell(HudText& tx, int row, int col, float v, float budget) {
        if (v < 0) { tx.put(row, col, 0, "   -  "); return; }
        tx.put(row, col, budget > 0 ? (v <= budget ? 1 : 2) : 0, "%6.2f", v);
    }

    void build_hud(HudText& tx, bool neural_now, int sim_steps) {
        tx.clear();
        if (!hud_on) return;
        int c = tx.put(0, 0, 0, "NVFX NATIVE DEMO  %dX%d  MARCH %dX%d  MODE ", W, H, march_w, march_h);
        tx.put(0, c, 3, "%s", a.solver_only ? "SOLVER ONLY" : neural_now ? "NEURAL" : "RAW 64^3");
        c = tx.put(1, 0, 0, "PRESET %d %s", preset_id, preset.name);
        if (paused) c = tx.put(1, c, 3, "  [PAUSED]");
        if (auto_orbit || scripted) c = tx.put(1, c, 0, "  [ORBIT]");
        if (scripted) tx.put(1, c, 3, a.bench_s > 0 ? "  [BENCH]" : "  [SCRIPTED]");
        c = tx.put(2, 0, h_low >= 60.0f || vsync ? 1 : 2, "FPS %6.1f  1%% LOW %6.1f", h_fps, h_low);
        tx.put(2, c, 0, "  VSYNC %s  SIM %s%s", vsync ? "ON" : "OFF", a.sim_every_frame ? "EVERY FRAME" : "30 HZ",
               sim_steps > 0 ? " *" : "");
        tx.put(3, 0, 3, "GPU MS (MEDIAN)     NOW  BUDGET");
        tx.put(4, 0, 0, "SIM STEP 64^3   ");     ms_cell(tx, 4, 16, h_sim, kBudgetSim);
        tx.put(4, 22, 0, "  2.00   PCG %d", pcg_iters);
        tx.put(5, 0, 0, "RENDER NEW VOL  ");     ms_cell(tx, 5, 16, h_rf, kBudgetRender);
        c = tx.put(5, 22, 0, "  1.50   REUSE");     ms_cell(tx, 5, c, h_rr, kBudgetRender);
        tx.put(6, 0, 0, "DETAIL          ");
        if (neural_now) {
            ms_cell(tx, 6, 16, h_det, kBudgetDetail);
            tx.put(6, 22, 0, "  2.50   %dX%d", detail_region[2], detail_region[3]);
        } else {
            tx.put(6, 16, 0, "   -    2.50");
        }
        tx.put(7, 0, 0, "TOTAL SIM FRAME ");     ms_cell(tx, 7, 16, h_fs, kBudgetTotal);
        c = tx.put(7, 22, 0, "  6.00   OTHER");     ms_cell(tx, 7, c, h_fo, kBudgetTotal);
        tx.put(8, 0, 0, "PRESENT+HUD     ");     ms_cell(tx, 8, 16, h_pres, 0);
        tx.put(9, 0, 0, "VRAM %.0f MB  STEP %llu  FRAME %llu", vram_now_mb, (unsigned long long)sim_steps_total,
               (unsigned long long)frame_no);
        int row = 10;
        if (!detail_usable() && !a.solver_only && !a.no_detail) tx.put(row++, 0, 2, "NEURAL UNAVAILABLE: %.44s", detail_status.c_str());
        if (!message.empty() && now_ms() - message_t < 5000.0) tx.put(row++, 0, 2, "%.62s", message.c_str());
        if (slot_waits_blocked) tx.put(row++, 0, 2, "FENCE WAITS %llu", (unsigned long long)slot_waits_blocked);
        tx.put(row++, 0, 0, "RMB ORBIT  WHEEL ZOOM  LMB FIRE+PUSH  SHIFT+LMB/G PUFF");
        tx.put(row, 0, 0, "N MODE  1-4 PRESET  C ORBIT  R RESET  SPC PAUSE  V VSYNC  H HUD");
    }

    void fill_hud_consts(int slot, const HudText& tx, bool has_source) {
        HudConsts hc = {};
        for (int g = 0; g < 64; ++g) {
            hc.font[g / 2][(g % 2) * 2 + 0] = font_packed[g][0];
            hc.font[g / 2][(g % 2) * 2 + 1] = font_packed[g][1];
        }
        for (int i = 0; i < kHudRows * kHudCols; ++i) {
            uint32_t b = tx.cell[i / kHudCols][i % kHudCols];
            hc.text[i >> 4][(i >> 2) & 3] |= b << ((i & 3) * 8);
        }
        const float pal[4][4] = {{0.88f, 0.90f, 0.93f, 1}, {0.55f, 0.87f, 0.55f, 1}, {1.0f, 0.47f, 0.43f, 1}, {1.0f, 0.82f, 0.47f, 1}};
        std::memcpy(hc.palette, pal, sizeof pal);
        hc.backdrop[0] = hc.backdrop[1] = hc.backdrop[2] = 0.0f;
        hc.backdrop[3] = 0.6f;
        hc.clear[0] = 0.03f; hc.clear[1] = 0.035f; hc.clear[2] = 0.05f; hc.clear[3] = 1.0f;
        int s = std::max(1, H / 720);
        int ox = 10 * s, oy = 10 * s, pad = 5 * s;
        hc.grid[0] = ox; hc.grid[1] = oy; hc.grid[2] = s;
        hc.grid[3] = (hud_on && tx.used_rows > 0 ? 1 : 0) | (has_source ? 2 : 0);
        hc.box[0] = ox - pad; hc.box[1] = oy - pad;
        hc.box[2] = ox + tx.used_cols * 6 * s + pad; hc.box[3] = oy + tx.used_rows * 9 * s + pad;
        std::memcpy(hud_box, hc.box, sizeof hud_box);
        hud_drawn = (hc.grid[3] & 1) != 0;
        hc.dims[0] = kHudCols; hc.dims[1] = kHudRows;
        std::memcpy(cb_map + size_t(slot) * kCbSlot, &hc, sizeof hc);
    }

    // ---------------------------------------------------------------- one frame
    // returns false on a fatal error
    bool frame(double fdt, bool recording) {
        const int slot = int(frame_no % kFrames);
        harvest_slot(slot);
        poll_library_timings(recording);
        if (frame_no % 30 == 0) sample_vram();

        // --- input -> camera, forces
        std::vector<UINT> keys;
        for (UINT k : g_in.keys) {
            if (k == VK_ESCAPE) g_in.quit = true;
            else if (!scripted) keys.push_back(k);     // scripted runs ignore the keyboard (except Esc)
        }
        g_in.keys.clear();
        Pointer ptr = scripted ? script_pointer(frame_no, keys) : window_pointer();
        apply_pointer(ptr, fdt);                       // before keys: G puffs at this frame's cursor
        for (UINT k : keys)
            if (!handle_key(k)) return false;
        if (scripted) ctl.fixed_torch = !a.no_torch;   // a fire held at a fixed point (deterministic)
        if (auto_orbit) orbit.yaw += 0.35 * fdt;

        int steps = 0;
        if (!paused) {
            if (scripted) steps = (a.sim_every_frame || frame_no % 2 == 0) ? 1 : 0;   // 60 Hz virtual timeline
            else if (a.sim_every_frame) steps = 1;
            else {
                sim_acc = std::min(sim_acc + fdt, 2.0 * NVFX_SIM_DT);
                while (sim_acc >= NVFX_SIM_DT - 1e-9) { ++steps; sim_acc -= NVFX_SIM_DT; }
            }
        }
        const bool neural_now = neural && detail_usable() && !a.solver_only;

        // --- record the frame
        if (!hr_ok(alloc[slot]->Reset(), "CommandAllocator::Reset") || !hr_ok(list->Reset(alloc[slot].Get(), nullptr), "CommandList::Reset"))
            return false;
        ID3D12GraphicsCommandList* l = list.Get();
        l->EndQuery(qheap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, UINT(slot * 3 + 0));
        bool fatal = false;

        for (int i = 0; i < steps && !fatal; ++i) {
            NvfxForces f = ctl.forces(preset);
            if (!nv_ok(nvfx_solver_record_step(solver, l, &f, NVFX_SIM_DT), "nvfx_solver_record_step")) fatal = true;
            ++tick_id;
            ++sim_steps_total;
            if (recording) ++rec_sim_steps;
        }

        NvfxCamera cam = orbit.camera();
        ID3D12Resource* src = nullptr;
        D3D12_RESOURCE_STATES src_state = D3D12_RESOURCE_STATE_COMMON;
        UINT src_srv = SRV_NULL;
        if (renderer && !fatal) {
            bool have_prev = false;
            NvfxCamera prev = prev_camera(cam, have_prev);
            int32_t flags = a.no_render_aux ? 0 : NVFX_RENDER_AUX;
            if (steps == 0 && !need_full_draw) flags |= NVFX_RENDER_REUSE_VOLUME;
            if (!nv_ok(nvfx_renderer_record_draw(renderer, l, solver, &cam, have_prev ? &prev : nullptr, flags),
                       "nvfx_renderer_record_draw"))
                fatal = true;
            need_full_draw = false;
            src = static_cast<ID3D12Resource*>(nvfx_renderer_texture(renderer, NVFX_RENDER_TEX_COLOR));
            src_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            src_srv = SRV_RENDER_COLOR;

            int32_t bbox[4] = {0, 0, W, H};
            nvfx_render_screen_bbox(&cam, W, H, bbox);
            last_cam = cam; last_prev = prev; last_have_prev = have_prev; last_flags = flags;
            std::memcpy(last_bbox, bbox, sizeof bbox);
            last_neural = false;
            if (recording)
                rec_bbox_frac.push_back(float(double(std::max(0, bbox[2] - bbox[0])) * std::max(0, bbox[3] - bbox[1]) /
                                              (double(W) * H)));
            if (neural_now && !fatal) {
                NvfxDetailFrame df = {};
                df.color = src;
                if (!a.no_render_aux) {
                    df.alpha = nvfx_renderer_texture(renderer, NVFX_RENDER_TEX_ALPHA);
                    df.depth = nvfx_renderer_texture(renderer, NVFX_RENDER_TEX_DEPTH);
                    df.motion = nvfx_renderer_texture(renderer, NVFX_RENDER_TEX_MOTION);
                }
                df.output = detail_out.Get();
                std::memcpy(df.bbox, bbox, sizeof bbox);
                if (nv_ok(nvfx_detail_record(detail, l, &df), "nvfx_detail_record")) {
                    src = detail_out.Get();
                    src_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                    src_srv = SRV_DETAIL_OUT;
                    last_neural = true;
                } else {
                    // keep running raw; whatever the runner recorded before failing still executes (its rings expect it)
                    // (not destroyed here: this list may reference its resources; destroyed at exit)
                    detail_status = std::string("nvfx_detail_record failed: ") + nvfx_last_error();
                    detail_broken = true;
                    neural = false;
                }
            }
        }
        l->EndQuery(qheap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, UINT(slot * 3 + 1));

        // --- present pass: frame -> back buffer + HUD (the app re-binds its own heap / root signature)
        HudText tx;
        build_hud(tx, neural_now && detail_usable(), steps);
        fill_hud_consts(slot, tx, src != nullptr);
        const UINT bb = swap->GetCurrentBackBufferIndex();
        D3D12_RESOURCE_STATES read_state = src_state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                                               ? D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                                               : D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
        D3D12_RESOURCE_BARRIER pre[2] = {transition(back[bb].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET)};
        UINT npre = 1;
        if (src) pre[npre++] = transition(src, src_state, read_state);
        l->ResourceBarrier(npre, pre);
        ID3D12DescriptorHeap* heaps[] = {srv_heap.Get()};
        l->SetDescriptorHeaps(1, heaps);
        l->SetGraphicsRootSignature(rs.Get());
        l->SetPipelineState(pso.Get());
        l->SetGraphicsRootConstantBufferView(0, cb_upload->GetGPUVirtualAddress() + UINT64(slot) * kCbSlot);
        D3D12_GPU_DESCRIPTOR_HANDLE gh = srv_heap->GetGPUDescriptorHandleForHeapStart();
        gh.ptr += UINT64(src_srv) * srv_inc;
        l->SetGraphicsRootDescriptorTable(1, gh);
        D3D12_VIEWPORT vp = {0.0f, 0.0f, float(W), float(H), 0.0f, 1.0f};
        D3D12_RECT sc = {0, 0, W, H};
        l->RSSetViewports(1, &vp);
        l->RSSetScissorRects(1, &sc);
        D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        rtv.ptr += SIZE_T(bb) * rtv_inc;
        l->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        l->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        l->DrawInstanced(3, 1, 0, 0);
        D3D12_RESOURCE_BARRIER post[2] = {transition(back[bb].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT)};
        UINT npost = 1;
        if (src) post[npost++] = transition(src, read_state, src_state);
        l->ResourceBarrier(npost, post);
        l->EndQuery(qheap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, UINT(slot * 3 + 2));
        l->ResolveQueryData(qheap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, UINT(slot * 3), 3, ts_readback.Get(),
                            UINT64(slot) * 3 * sizeof(uint64_t));
        if (!hr_ok(l->Close(), "CommandList::Close")) return false;

        // executed even after a library failure: recorded library work must run (their rings wait for it)
        ID3D12CommandList* lists[] = {l};
        queue->ExecuteCommandLists(1, lists);
        if (!hr_ok(queue->Signal(fence.Get(), ++fence_value), "Queue::Signal")) return false;
        meta[slot] = SlotMeta{fence_value, true, steps, neural_now, recording};
        if (recording) (neural_now ? rec_frames_neural : rec_frames_raw) += 1;
        return !fatal;
    }

    bool present() {
        UINT interval = vsync ? 1 : 0;
        UINT flags = (!vsync && tearing) ? DXGI_PRESENT_ALLOW_TEARING : 0;
        HRESULT hr = swap->Present(interval, flags);
        if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) {
            std::fprintf(stderr, "[nvfx_demo] Present: device removed (reason 0x%08lx)\n",
                         static_cast<unsigned long>(device->GetDeviceRemovedReason()));
            return false;
        }
        if (FAILED(hr)) return hr_ok(hr, "Present");
        return true;
    }

    void wait_idle() {
        if (!queue || !fence) return;
        if (SUCCEEDED(queue->Signal(fence.Get(), ++fence_value))) wait_fence(fence_value);
    }

    // the back buffer of the frame just executed, still in PRESENT state (called before Present). Synchronizes.
    bool download_back(std::vector<uint8_t>& rgba) {
        const UINT bb = swap->GetCurrentBackBufferIndex();
        rgba.assign(size_t(W) * H * 4, 0);
        return nv_ok(nvfx_texture2d_download(ctx, back[bb].Get(), uint32_t(D3D12_RESOURCE_STATE_PRESENT), rgba.data(),
                                             uint32_t(W) * 4), "nvfx_texture2d_download(back buffer)");
    }

    bool screenshot(const std::wstring& path) {
        std::vector<uint8_t> rgba;
        if (!download_back(rgba) || !write_image(path, rgba, W, H)) return false;
        std::printf("[nvfx_demo] frame %llu screenshot -> %s\n", (unsigned long long)frame_no, to_utf8(path).c_str());
        return true;
    }

    std::wstring shot_path(uint64_t f) const {
        std::wstring p = a.screenshot.empty() ? L"D:\\nvfx_demo_out\\shot.bmp" : a.screenshot;
        size_t dot = p.find_last_of(L'.');
        size_t sl = p.find_last_of(L"\\/");
        if (dot == std::wstring::npos || (sl != std::wstring::npos && dot < sl)) dot = p.size();
        wchar_t suf[32];
        swprintf(suf, 32, L"_f%llu", (unsigned long long)f);
        return p.substr(0, dot) + suf + p.substr(dot);
    }

    // last frame: the renderer's aux buffers as images + stats; is the bbox handed to detail sane?
    bool dump_aux_files() {
        if (!renderer) return true;
        std::vector<float> al(size_t(W) * H), de(size_t(W) * H), mo(size_t(W) * H * 2);
        if (!nv_ok(nvfx_renderer_download(renderer, NVFX_RENDER_TEX_ALPHA, al.data()), "nvfx_renderer_download(ALPHA)") ||
            !nv_ok(nvfx_renderer_download(renderer, NVFX_RENDER_TEX_DEPTH, de.data()), "nvfx_renderer_download(DEPTH)") ||
            !nv_ok(nvfx_renderer_download(renderer, NVFX_RENDER_TEX_MOTION, mo.data()), "nvfx_renderer_download(MOTION)"))
            return false;
        const size_t N = size_t(W) * H;
        int x0 = W, y0 = H, x1 = -1, y1 = -1;
        size_t n_vis = 0, n_out = 0, n_motion_clear = 0, n_nan = 0;
        double mmax = 0, msum = 0, mxs = 0, mys = 0, dmin = 1e9, dmax = 0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                size_t i = size_t(y) * W + x;
                float A = al[i], D = de[i], mx = mo[i], my = mo[N + i];
                if (!std::isfinite(A) || !std::isfinite(D) || !std::isfinite(mx) || !std::isfinite(my)) { ++n_nan; continue; }
                if (A < 1e-3f && (mx != 0.0f || my != 0.0f)) ++n_motion_clear;
                if (A <= 0.01f) continue;
                ++n_vis;
                x0 = std::min(x0, x); y0 = std::min(y0, y); x1 = std::max(x1, x + 1); y1 = std::max(y1, y + 1);
                if (x < last_bbox[0] || x >= last_bbox[2] || y < last_bbox[1] || y >= last_bbox[3]) ++n_out;
                double m = std::sqrt(double(mx) * mx + double(my) * my);
                mmax = std::max(mmax, m); msum += m; mxs += mx; mys += my;
                dmin = std::min(dmin, double(D)); dmax = std::max(dmax, double(D));
            }
        double nv = double(std::max<size_t>(n_vis, 1));
        std::printf("[aux] alpha>0.01: %zu px (%.1f %% of frame), extent [%d,%d)-[%d,%d)\n", n_vis, 100.0 * n_vis / N, x0, y0, x1, y1);
        std::printf("[aux] projected bbox handed to detail: [%d,%d)-[%d,%d) = %.1f %% of frame; visible px outside it: %zu\n",
                    last_bbox[0], last_bbox[1], last_bbox[2], last_bbox[3],
                    100.0 * (last_bbox[2] - last_bbox[0]) * (last_bbox[3] - last_bbox[1]) / double(N), n_out);
        std::printf("[aux] motion over visible px (px per SIM_DT, y down): |m| max %.2f mean %.2f, mean (x %.2f, y %.2f); "
                    "nonzero motion where alpha<1e-3: %zu px; non-finite: %zu\n", mmax, msum / nv, mxs / nv, mys / nv,
                    n_motion_clear, n_nan);
        std::printf("[aux] depth over visible px: %.3f .. %.3f m (camera distance to target %.3f m); draw flags 0x%x, prev cam %s\n",
                    n_vis ? dmin : 0.0, dmax, orbit.dist, unsigned(last_flags), last_have_prev ? "yes" : "no (same camera)");
        if (detail) {
            NvfxDetailTimings dt = {};
            int32_t reg[6] = {};
            nvfx_detail_get_timings(detail, &dt);
            nvfx_detail_region(detail, last_bbox, reg);
            std::printf("[aux] detail region for this bbox: x %d y %d %dx%d (crop %dx%d); last finished detail frame %llu region %dx%d\n",
                        reg[0], reg[1], reg[2], reg[3], reg[4], reg[5], (unsigned long long)dt.frame_index, dt.region[2], dt.region[3]);
        }
        // images: alpha grey; motion R = x, G = y (128 = 0, +-scale to the max), B = alpha
        double ms = mmax > 1e-6 ? 127.0 / mmax : 1.0;
        std::vector<uint8_t> img(N * 4, 255);
        for (size_t i = 0; i < N; ++i) {
            uint8_t g = uint8_t(std::clamp(al[i], 0.0f, 1.0f) * 255.0f + 0.5f);
            img[i * 4 + 0] = img[i * 4 + 1] = img[i * 4 + 2] = g;
        }
        bool ok = write_image(a.dump_aux + L"_alpha.bmp", img, W, H);
        for (size_t i = 0; i < N; ++i) {
            img[i * 4 + 0] = uint8_t(std::clamp(128.0 + mo[i] * ms, 0.0, 255.0));
            img[i * 4 + 1] = uint8_t(std::clamp(128.0 + mo[N + i] * ms, 0.0, 255.0));
            img[i * 4 + 2] = uint8_t(std::clamp(al[i], 0.0f, 1.0f) * 255.0f);
        }
        ok = write_image(a.dump_aux + L"_motion.bmp", img, W, H) && ok;
        std::printf("[aux] -> %s_alpha.bmp, %s_motion.bmp (motion scale: 127 = %.2f px)\n", to_utf8(a.dump_aux).c_str(),
                    to_utf8(a.dump_aux).c_str(), mmax);
        return ok;
    }

    // last frame, raw mode: (1) back buffer vs the renderer's COLOR texture (the app's copy path), (2) back buffer vs a
    // stand-alone nvfx_renderer_draw of the same solver state + camera + prev camera (the hosted record path)
    bool compare_library() {
        if (!renderer) return true;
        if (last_neural) { std::printf("[compare] skipped: last frame was NEURAL (use --mode raw)\n"); return true; }
        std::vector<uint8_t> bb, col(size_t(W) * H * 4);
        if (!download_back(bb)) return false;
        auto diff = [&](const char* what) {
            size_t n = 0, nd = 0;
            int mx = 0;
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    if (hud_drawn && x >= hud_box[0] && x < hud_box[2] && y >= hud_box[1] && y < hud_box[3]) continue;
                    size_t i = (size_t(y) * W + x) * 4;
                    int d = 0;
                    for (int c = 0; c < 3; ++c) d = std::max(d, std::abs(int(bb[i + c]) - int(col[i + c])));
                    ++n;
                    if (d) ++nd;
                    mx = std::max(mx, d);
                }
            std::printf("[compare] %s: %zu px compared%s, %zu differ, max abs %d -> %s\n", what, n,
                        hud_drawn ? " (HUD box excluded)" : "", nd, mx, nd == 0 ? "IDENTICAL" : "DIFFERENT");
        };
        if (!nv_ok(nvfx_renderer_download(renderer, NVFX_RENDER_TEX_COLOR, col.data()), "nvfx_renderer_download(COLOR)")) return false;
        diff("back buffer vs renderer COLOR (app copy path)");
        int32_t fl = last_flags & ~NVFX_RENDER_REUSE_VOLUME;
        if (!nv_ok(nvfx_renderer_draw(renderer, solver, &last_cam, last_have_prev ? &last_prev : nullptr, fl), "nvfx_renderer_draw"))
            return false;
        if (!nv_ok(nvfx_renderer_download(renderer, NVFX_RENDER_TEX_COLOR, col.data()), "nvfx_renderer_download(COLOR)")) return false;
        std::printf("[compare] app frame drew with flags 0x%x; stand-alone redraw with 0x%x\n", unsigned(last_flags), unsigned(fl));
        diff("back buffer vs stand-alone nvfx_renderer_draw (same state/camera)");
        return true;
    }

    // ---------------------------------------------------------------- run
    int run() {
        scripted = a.bench_s > 0 || a.frames > 0;
        neural = a.neural;
        vsync = a.vsync && a.bench_s <= 0;
        hud_on = !a.no_hud;
        auto_orbit = a.orbit || (scripted && !a.no_auto_orbit);
        orbit.yaw = a.yaw;
        if (a.bench_s > 0) orbit.dist = a.bench_dist;
        if (!a.script.empty() && !parse_script(to_utf8(a.script), script)) return 2;
        if (a.neural && !detail_usable() && !a.solver_only)
            std::printf("[nvfx_demo] mode neural requested, detail unavailable (%s): running RAW\n", detail_status.c_str());
        std::printf("[nvfx_demo] adapter: %s | %dx%d | march %dx%d | tearing %s\n", adapter_name.c_str(), W, H, march_w,
                    march_h, tearing ? "yes" : "no");
        std::fflush(stdout);

        double t_start = now_ms(), t_last_loop = t_start, t_last_present = 0, t_title = 0;
        bool ok = true;
        int exit_code = 0;
        while (!g_in.quit) {
            MSG msg;
            while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
                if (msg.message == WM_QUIT) g_in.quit = true;
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            if (g_in.quit) break;
            if (g_in.minimized && !scripted) { Sleep(10); t_last_loop = now_ms(); continue; }

            // ===== THE frame-pacing wait: at most kMaxLatency frames queued ahead of the display =====
            WaitForSingleObjectEx(latency_wait, 1000, TRUE);

            double t = now_ms();
            double fdt = scripted ? 1.0 / 60.0 : std::clamp((t - t_last_loop) / 1000.0, 0.0, 0.1);
            t_last_loop = t;
            double el = (t - t_start) / 1000.0;
            bool recording = a.bench_s > 0 && el >= a.warmup_s;
            if (a.bench_s > 0 && el >= a.warmup_s + a.bench_s) break;

            if (!frame(fdt, recording)) { ok = false; exit_code = 3; }
            bool last = a.frames > 0 && frame_no + 1 >= uint64_t(a.frames);
            if (ok && std::find(a.shot_at.begin(), a.shot_at.end(), int(frame_no)) != a.shot_at.end() &&
                !screenshot(shot_path(frame_no)))
                exit_code = 4;
            if (ok && last && !a.screenshot.empty() && !screenshot(a.screenshot)) exit_code = 4;
            if (ok && last && !a.dump_aux.empty() && !dump_aux_files()) exit_code = 4;
            if (ok && last && a.compare && !compare_library()) exit_code = 4;
            if (!present()) { ok = false; exit_code = 3; }
            double tp = now_ms();
            if (t_last_present > 0) {
                float ms = float(tp - t_last_present);
                recent_frame_ms.push_back({tp, ms});
                if (recording) rec_frame_ms.push_back(ms);
            }
            t_last_present = tp;
            refresh_hud_values(tp);
            ++frame_no;
            if (tp - t_title > 500.0) {
                t_title = tp;
                wchar_t title[256];
                swprintf(title, 256, L"nvfx_demo - %hs - %.1f fps (1%% low %.1f) - frame GPU %.2f ms", neural && detail_usable() ? "NEURAL" : "RAW",
                         h_fps, h_low, h_fs);
                SetWindowTextW(hwnd, title);
            }
            if (!ok || last) break;
        }
        wait_idle();
        for (int s = 0; s < kFrames; ++s) harvest_slot(s);
        poll_library_timings(a.bench_s > 0);
        sample_vram();
        if (a.bench_s > 0) write_bench();
        return ok ? exit_code : (exit_code ? exit_code : 3);
    }

    // ---------------------------------------------------------------- bench report
    static void json_stat(FILE* f, const char* name, const Stat& s, bool comma = true) {
        std::fprintf(f, "    \"%s\": {\"n\": %zu, \"mean\": %.4f, \"p10\": %.4f, \"median\": %.4f, \"p95\": %.4f, \"p99\": %.4f, \"max\": %.4f}%s\n",
                     name, s.n, s.mean, s.p10, s.p50, s.p95, s.p99, s.max, comma ? "," : "");
    }

    void write_bench() {
        Stat fm = stat_of(rec_frame_ms), sim = stat_of(t_sim.rec), rf = stat_of(t_render_full.rec), rr = stat_of(t_render_reuse.rec),
             det = stat_of(t_detail.rec), fs = stat_of(t_frame_sim.rec), fo = stat_of(t_frame_other.rec),
             fa = stat_of(rec_frame_gpu), pr = stat_of(t_present.rec), bb = stat_of(rec_bbox_frac);
        double fps = fm.mean > 0 ? 1000.0 / fm.mean : 0.0, low = one_pct_low_fps(rec_frame_ms);
        bool neural_eff = rec_frames_neural > 0 && rec_frames_raw == 0;
        double stage_sum = sim.p50 + rf.p50 + (neural_eff ? det.p50 : 0.0);
        uint64_t rec_frames = rec_frames_neural + rec_frames_raw;
        double sim_share = rec_frames ? std::min(1.0, double(rec_sim_steps) / double(rec_frames)) : 0.0;
        double amortized = sim_share * (sim.p50 + rf.p50) + (1.0 - sim_share) * rr.p50 + (neural_eff ? det.p50 : 0.0);
        double total_gate = fs.n ? fs.p50 : fa.p50;

        std::printf("\n=== nvfx_demo bench: preset %d (%s), mode %s%s, %dx%d (march %dx%d), %.1f s after %.1f s warm-up ===\n",
                    preset_id, preset.name, neural_eff ? "NEURAL" : "RAW",
                    a.neural && !neural_eff ? " (neural requested, unavailable)" : "", W, H, march_w, march_h, a.bench_s,
                    a.warmup_s);
        std::printf("frames %zu  avg %.1f fps  1%% low %.1f fps  frame ms median %.3f p99 %.3f\n", fm.n, fps, low, fm.p50, fm.p99);
        std::printf("GPU ms median [p10, p95]   (budget)\n");
        std::printf("  sim step       %6.3f [%6.3f, %6.3f]  (2.0)   PCG iters %d\n", sim.p50, sim.p10, sim.p95, pcg_iters);
        std::printf("  render new vol %6.3f [%6.3f, %6.3f]  (1.5)\n", rf.p50, rf.p10, rf.p95);
        std::printf("  render reuse   %6.3f [%6.3f, %6.3f]  (1.5)\n", rr.p50, rr.p10, rr.p95);
        if (neural_eff) std::printf("  detail         %6.3f [%6.3f, %6.3f]  (2.5)\n", det.p50, det.p10, det.p95);
        std::printf("  frame, sim     %6.3f [%6.3f, %6.3f]  (6.0)  %s\n", fs.p50, fs.p10, fs.p95, total_gate <= kBudgetTotal ? "WITHIN" : "OVER");
        std::printf("  frame, no sim  %6.3f [%6.3f, %6.3f]\n", fo.p50, fo.p10, fo.p95);
        std::printf("  present+HUD    %6.3f\n", pr.p50);
        std::printf("VRAM process local %.0f MB (peak %.0f, budget %.0f)  bbox %.1f %% of the frame  slot waits %llu\n",
                    vram_now_mb, vram_peak_mb, vram_budget_mb, bb.mean * 100.0, (unsigned long long)slot_waits_blocked);

        if (a.out.empty()) return;
        FILE* f = _wfopen(a.out.c_str(), L"wb");
        if (!f) { std::fprintf(stderr, "[nvfx_demo] cannot write %s\n", to_utf8(a.out).c_str()); return; }
        std::fprintf(f, "{\n");
        std::fprintf(f, "  \"app\": \"nvfx_demo\", \"nvfx\": \"%s\",\n", nvfx_version_string());
        std::fprintf(f, "  \"adapter\": \"%s\",\n", adapter_name.c_str());
        std::fprintf(f, "  \"width\": %d, \"height\": %d, \"march\": [%d, %d], \"march_scale\": %.3f,\n", W, H, march_w, march_h, a.march_scale);
        std::fprintf(f, "  \"preset\": %d, \"preset_name\": \"%s\",\n", preset_id, preset.name);
        std::fprintf(f, "  \"mode_requested\": \"%s\", \"mode_effective\": \"%s\", \"detail_status\": \"", a.neural ? "neural" : "raw",
                     neural_eff ? "neural" : (rec_frames_neural ? "mixed" : "raw"));
        for (char ch : detail_status) { if (ch == '"' || ch == '\\') std::fputc('\\', f); std::fputc(ch >= 32 ? ch : ' ', f); }
        std::fprintf(f, "\",\n");
        std::fprintf(f, "  \"switches\": {\"sim_every_frame\": %s, \"no_detail\": %s, \"no_render_aux\": %s, \"solver_only\": %s, \"hud\": %s},\n",
                     a.sim_every_frame ? "true" : "false", a.no_detail ? "true" : "false", a.no_render_aux ? "true" : "false",
                     a.solver_only ? "true" : "false", hud_on ? "true" : "false");
        std::fprintf(f, "  \"timeline\": \"virtual 60 Hz: camera yaw += 0.35 rad/s * 1/60 per frame, sim step on every %s frame, scripted torch at (0.40, 0.14, 0.50)\",\n",
                     a.sim_every_frame ? "" : "second");
        std::fprintf(f, "  \"orbit_dist\": %.3f, \"bbox_frac_mean\": %.4f,\n", orbit.dist, bb.mean);
        std::fprintf(f, "  \"vsync\": false, \"tearing\": %s, \"seconds\": %.3f, \"warmup_s\": %.3f,\n", tearing ? "true" : "false", a.bench_s, a.warmup_s);
        std::fprintf(f, "  \"frames\": %zu, \"sim_steps\": %llu, \"frames_neural\": %llu, \"frames_raw\": %llu,\n", fm.n,
                     (unsigned long long)rec_sim_steps, (unsigned long long)rec_frames_neural, (unsigned long long)rec_frames_raw);
        std::fprintf(f, "  \"fps_avg\": %.3f, \"fps_1pct_low\": %.3f,\n", fps, low);
        std::fprintf(f, "  \"cpu_frame_ms\": {\"n\": %zu, \"mean\": %.4f, \"median\": %.4f, \"p95\": %.4f, \"p99\": %.4f, \"max\": %.4f},\n",
                     fm.n, fm.mean, fm.p50, fm.p95, fm.p99, fm.max);
        std::fprintf(f, "  \"gpu_ms\": {\n");
        json_stat(f, "sim", sim);
        json_stat(f, "render_new_volume", rf);
        json_stat(f, "render_reuse", rr);
        json_stat(f, "detail", det);
        json_stat(f, "frame_sim", fs);
        json_stat(f, "frame_no_sim", fo);
        json_stat(f, "frame_all", fa);
        json_stat(f, "present_hud", pr, false);
        std::fprintf(f, "  },\n");
        std::fprintf(f, "  \"pcg_iters\": %d,\n", pcg_iters);
        std::fprintf(f, "  \"budget_ms\": {\"sim\": 2.0, \"render\": 1.5, \"detail\": 2.5, \"total\": 6.0},\n");
        std::fprintf(f, "  \"total_gpu_ms\": {\"measured_sim_frame_median\": %.4f, \"stage_sum_sim_frame\": %.4f, \"amortized_per_frame\": %.4f, \"within_6ms\": %s},\n",
                     total_gate, stage_sum, amortized, total_gate <= kBudgetTotal ? "true" : "false");
        std::fprintf(f, "  \"within_stage_budgets\": {\"sim\": %s, \"render\": %s, \"detail\": %s},\n", sim.p50 <= kBudgetSim ? "true" : "false",
                     rf.p50 <= kBudgetRender ? "true" : "false", !neural_eff ? "null" : det.p50 <= kBudgetDetail ? "true" : "false");
        std::fprintf(f, "  \"detail_region\": [%d, %d, %d, %d], \"detail_crop\": [%d, %d], \"detail_owned_mb\": %.1f,\n", detail_region[0],
                     detail_region[1], detail_region[2], detail_region[3], detail_crop[0], detail_crop[1],
                     detail ? double(nvfx_detail_vram_bytes(detail)) / 1048576.0 : 0.0);
        std::fprintf(f, "  \"vram_mb\": {\"process_local_end\": %.1f, \"process_local_peak\": %.1f, \"budget\": %.1f},\n", vram_now_mb,
                     vram_peak_mb, vram_budget_mb);
        std::fprintf(f, "  \"slot_fence_waits_blocked\": %llu,\n", (unsigned long long)slot_waits_blocked);
        std::fprintf(f, "  \"cpu_frame_ms_series\": [");
        for (size_t i = 0; i < rec_frame_ms.size(); ++i) std::fprintf(f, "%s%.3f", i ? "," : "", rec_frame_ms[i]);
        std::fprintf(f, "],\n  \"gpu_frame_ms_series\": [");
        for (size_t i = 0; i < rec_frame_gpu.size(); ++i) std::fprintf(f, "%s%.3f", i ? "," : "", rec_frame_gpu[i]);
        std::fprintf(f, "]\n}\n");
        std::fclose(f);
        std::printf("[nvfx_demo] bench -> %s\n", to_utf8(a.out).c_str());
    }

    // ---------------------------------------------------------------- teardown
    void destroy() {
        wait_idle();
        if (detail) nvfx_detail_destroy(detail);
        if (renderer) nvfx_renderer_destroy(renderer);
        if (solver) nvfx_solver_destroy(solver);
        if (ctx) nvfx_context_destroy(ctx);
        detail = nullptr; renderer = nullptr; solver = nullptr; ctx = nullptr;
        if (fence_event) CloseHandle(fence_event);
        fence_event = nullptr;
        if (hwnd) DestroyWindow(hwnd);
        hwnd = nullptr;
    }
};

}  // namespace

int wmain(int argc, wchar_t** argv) {
    Demo d;
    if (!parse_args(argc, argv, d.a)) { usage(); return 2; }
    hud_font_pack(d.font_packed);
    int rc = 1;
    if (d.create_window() && d.create_device() && d.create_nvfx()) rc = d.run();
    else std::fprintf(stderr, "[nvfx_demo] start-up failed\n");
    d.destroy();
    return rc;
}
