// nvfx_selfcheck: owner self-check of the native sim3d solver, through the public C API only.
//   nvfx_selfcheck [--quick] [--debug] [--bench-only] [--tex-only]
// Timings (GPU timestamps, median of >= 200 steps after warm-up, split by stage), PCG iteration cost
// and skipped-iteration cost, divergence, projection of a random field, mass drift over 1000 closed-box
// steps, rerun determinism, and the 1000-step stress run. Not the gate tests (native/tests/).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <d3d12.h>

#include "nvfx.h"

static const double SIM_DT = 1.0 / 30.0;
static const float AMB = NVFX_AMBIENT_TEMP;

#define CHECK(x)                                                                                    \
    do {                                                                                            \
        NvfxResult r_ = (x);                                                                        \
        if (r_ != NVFX_OK) {                                                                        \
            std::fprintf(stderr, "%s:%d %s -> %d: %s\n", __FILE__, __LINE__, #x, r_, nvfx_last_error()); \
            std::exit(1);                                                                           \
        }                                                                                           \
    } while (0)

struct Scene {
    std::vector<NvfxEmitter> em;
    std::vector<NvfxImpulse> imp;
    float wind[3] = {0, 0, 0};
    NvfxForces forces() const {
        NvfxForces f = {};
        f.emitters = em.data();
        f.num_emitters = int32_t(em.size());
        f.impulses = imp.data();
        f.num_impulses = int32_t(imp.size());
        std::memcpy(f.wind, wind, sizeof wind);
        return f;
    }
};

static Scene campfire(double t) {
    Scene s;
    const double spots[5][2] = {{0.5, 0.5}, {0.44, 0.47}, {0.56, 0.47}, {0.47, 0.56}, {0.55, 0.55}};
    for (int k = 0; k < 5; ++k) {
        double sc = 1.0 + 0.35 * std::sin(2.3 * t + 1.7 * k);
        NvfxEmitter e = {{float(spots[k][0]), 0.07f, float(spots[k][1])}, 0.045f, 0.0f, 900.0f, float(8.0 * sc), {0.0f, 0.4f, 0.0f}};
        s.em.push_back(e);
    }
    return s;
}
static Scene plume(double) {
    Scene s;
    s.em.push_back({{0.42f, 0.1f, 0.5f}, 0.06f, 1.0f, 900.0f, 0.0f, {0.0f, 1.2f, 0.0f}});
    s.wind[0] = 0.4f;
    return s;
}
static const NvfxObstacle SPHERE = {NVFX_OBSTACLE_SPHERE, {0.5f, 0.55f, 0.5f}, {0.12f, 0.12f, 0.12f}};

static std::string gpu_status() {
    FILE* p = _popen("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader", "r");
    if (!p) return "nvidia-smi failed";
    char buf[256] = {0};
    std::string s;
    while (std::fgets(buf, sizeof buf, p)) s += buf;
    _pclose(p);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return "GPU util, memory used (all processes) = " + s;
}

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[size_t(q * double(v.size() - 1))];
}

static double max_of(const std::vector<float>& v) {
    double m = 0.0;
    for (float x : v) m = std::max(m, double(x));
    return m;
}

struct Bench { double adv, forces, proj, total, iters; size_t samples; double total_p10, proj_p10; };

static Bench bench(NvfxContext* ctx, Scene (*fn)(double), const std::vector<NvfxObstacle>& obs, const NvfxSolverConfig& cfg,
                   int warm = 60, size_t want = 240) {
    NvfxSolver* s;
    CHECK(nvfx_solver_create(ctx, 64, &cfg, &s));
    CHECK(nvfx_solver_reset(s, obs.data(), int32_t(obs.size())));
    int i = 0;
    for (; i < warm; ++i) {
        Scene sc = fn(i * SIM_DT);
        NvfxForces f = sc.forces();
        CHECK(nvfx_solver_step(s, &f, float(SIM_DT)));
    }
    std::vector<double> a, fo, p, t, it;
    uint64_t last = 0;
    for (; a.size() < want && i < warm + 4000; ++i) {
        Scene sc = fn(i * SIM_DT);
        NvfxForces f = sc.forces();
        CHECK(nvfx_solver_step(s, &f, float(SIM_DT)));
        NvfxSolverTimings tm;
        CHECK(nvfx_solver_get_timings(s, &tm));
        if (tm.step_index > last && tm.step_index > uint64_t(warm)) {
            last = tm.step_index;
            a.push_back(tm.advection_ms); fo.push_back(tm.forces_ms); p.push_back(tm.projection_ms);
            t.push_back(tm.total_ms); it.push_back(tm.pcg_iters);
        }
    }
    CHECK(nvfx_context_wait_idle(ctx));
    nvfx_solver_destroy(s);
    return {median(a), median(fo), median(p), median(t), median(it), a.size(), pct(t, 0.1), pct(p, 0.1)};
}

static void step_n(NvfxSolver* s, Scene (*fn)(double), int n, int start = 0) {
    for (int i = start; i < start + n; ++i) {
        Scene sc = fn(i * SIM_DT);
        NvfxForces f = sc.forces();
        CHECK(nvfx_solver_step(s, &f, float(SIM_DT)));
    }
}

static bool finite_all(const std::vector<float>& v) {
    for (float x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

// nvfx_texture2d_* round trip: odd size, caller row pitch padded past the texture row (and != footprint pitch).
static bool tex_roundtrip(NvfxContext* ctx) {
    struct Case { const char* name; DXGI_FORMAT fmt; uint32_t bpp; uint32_t flags; uint32_t state; } cases[] = {
        {"R8G8B8A8_UNORM", DXGI_FORMAT_R8G8B8A8_UNORM, 4, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE},
        {"R32G32B32A32_FLOAT", DXGI_FORMAT_R32G32B32A32_FLOAT, 16, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
         D3D12_RESOURCE_STATE_UNORDERED_ACCESS}};
    const int W = 67, H = 33;
    bool all = true;
    for (auto& c : cases) {
        uint32_t pitch = W * c.bpp + 12;
        std::vector<uint8_t> in(size_t(pitch) * H, 0), out(size_t(pitch) * H, 0xCD);
        for (int y = 0; y < H; ++y)
            for (uint32_t x = 0; x < W * c.bpp; ++x) {
                uint8_t* p = &in[size_t(y) * pitch + x];
                *p = uint8_t((x * 7 + y * 13) & 0xFF);
                if (c.bpp == 16 && x % 4 == 3) *p = 0x3F;   // keep every float finite and normal (no NaN canonicalisation)
            }
        void* t = nullptr;
        CHECK(nvfx_texture2d_create(ctx, W, H, c.fmt, c.flags, c.state, &t));
        CHECK(nvfx_texture2d_upload(ctx, t, c.state, in.data(), pitch));
        CHECK(nvfx_texture2d_download(ctx, t, c.state, out.data(), pitch));
        nvfx_texture2d_release(t);
        bool same = true;
        for (int y = 0; y < H; ++y)
            same = same && std::memcmp(&in[size_t(y) * pitch], &out[size_t(y) * pitch], size_t(W) * c.bpp) == 0;
        std::printf("texture2d round trip %-18s %dx%d pitch %u: identical = %d\n", c.name, W, H, pitch, int(same));
        all = all && same;
    }
    return all;
}

int main(int argc, char** argv) {
    bool quick = false, debug = false, bench_only = false, tex_only = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--quick")) quick = true;
        if (!std::strcmp(argv[i], "--debug")) debug = true;
        if (!std::strcmp(argv[i], "--bench-only")) bench_only = true;
        if (!std::strcmp(argv[i], "--tex-only")) tex_only = true;
    }
    std::printf("%s\n", nvfx_version_string());
    NvfxContextDesc cd = {-1, debug ? 1 : 0};
    NvfxContext* ctx;
    CHECK(nvfx_context_create(&cd, &ctx));
    if (tex_only) {
        bool good = tex_roundtrip(ctx);
        nvfx_context_destroy(ctx);
        return good ? 0 : 1;
    }
    tex_roundtrip(ctx);
    const int N = 64;
    const size_t n3 = size_t(N) * N * N;
    NvfxSolverConfig def, rt;
    nvfx_solver_config_default(&def);
    nvfx_solver_config_runtime(&rt);
    std::vector<NvfxObstacle> none, sph = {SPHERE};
    struct { const char* name; Scene (*fn)(double); std::vector<NvfxObstacle>* obs; } scenes[] = {
        {"campfire", campfire, &none}, {"plume", plume, &sph}};

    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "--profile-run")) {   // with env NVFX_SIM_PROFILE=1: per-kernel table on stderr
            NvfxSolverConfig c = def;
            c.pressure_tol = 0.0f;
            c.pressure_max_iters = 8;
            NvfxSolver* s;
            CHECK(nvfx_solver_create(ctx, N, &c, &s));
            CHECK(nvfx_solver_reset(s, &SPHERE, 1));
            step_n(s, plume, 150);
            nvfx_solver_destroy(s);
            nvfx_context_destroy(ctx);
            return 0;
        }
    // ---------------------------------------------------------------- timing
    std::printf("GPU before bench: %s\n", gpu_status().c_str());
    for (auto* cfgp : {&def, &rt}) {
        for (auto& sc : scenes) {
            Bench b = bench(ctx, sc.fn, *sc.obs, *cfgp);
            std::printf("bench 64^3 %-7s %-8s total %.3f ms (p10 %.3f) | adv %.3f forces %.3f proj %.3f (p10 %.3f) | iters %.0f | samples %zu\n",
                        cfgp == &def ? "full" : "runtime", sc.name, b.total, b.total_p10, b.adv, b.forces, b.proj, b.proj_p10,
                        b.iters, b.samples);
        }
    }
    // projection cost vs executed / skipped PCG iterations (plume + sphere, warm state)
    {
        std::vector<std::pair<double, double>> pts;
        for (int k : {0, 1, 2, 4, 8, 16}) {
            NvfxSolverConfig c = def;
            c.pressure_tol = 0.0f;
            c.pressure_max_iters = k;
            Bench b = bench(ctx, plume, sph, c, 60, 200);
            pts.push_back({b.iters, b.proj});
            std::printf("fixed iters %2d: proj %.3f ms (p10 %.3f, ran %.0f), total %.3f ms\n", k, b.proj, b.proj_p10, b.iters, b.total);
        }
        double n = double(pts.size()), sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (auto& q : pts) { sx += q.first; sy += q.second; sxx += q.first * q.first; sxy += q.first * q.second; }
        double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        std::printf("=> %.4f ms per executed PCG iteration, %.3f ms fixed (fit over 0..16)\n", slope, (sy - slope * sx) / n);
        for (int k : {1, 40, 200}) {
            NvfxSolverConfig c = def;
            c.pressure_tol = 1e30f;   // stopping test passes immediately: all k recorded iterations predicated off
            c.pressure_max_iters = k;
            Bench b = bench(ctx, plume, sph, c, 60, 200);
            std::printf("skipped iters %3d: proj %.3f ms (p10 %.3f, ran %.0f)\n", k, b.proj, b.proj_p10, b.iters);
        }
    }
    std::printf("GPU after bench: %s\n", gpu_status().c_str());
    if (bench_only) { nvfx_context_destroy(ctx); return 0; }

    std::vector<float> data(6 * n3), sol(n3), div(n3);
    // ---------------------------------------------------------------- divergence, steps 1..120
    for (auto& sc : scenes) {
        NvfxSolver* s;
        CHECK(nvfx_solver_create(ctx, N, &def, &s));
        CHECK(nvfx_solver_reset(s, sc.obs->data(), int32_t(sc.obs->size())));
        double worst = 0;
        int steps = quick ? 30 : 120;
        for (int i = 0; i < steps; ++i) {
            step_n(s, sc.fn, 1, i);
            CHECK(nvfx_solver_download_divergence(s, div.data()));
            worst = std::max(worst, max_of(div));
        }
        CHECK(nvfx_solver_download_state(s, data.data(), sol.data()));
        double m = 0, h = 0, Tm = 0;
        for (size_t i = 0; i < n3; ++i) {
            m += data[i];
            h += std::max(0.0, double(data[4 * n3 + i]) - AMB);
            Tm = std::max(Tm, double(data[4 * n3 + i]));
        }
        NvfxSolverTimings tm;
        CHECK(nvfx_solver_get_timings(s, &tm));
        std::printf("%-8s %d steps default: max norm div steps 1..%d %.2e [< 1e-4] | smoke %.4f heat %.2f K m^3 Tmax %.0f | last iters %d | finite %d\n",
                    sc.name, steps, steps, worst, m / n3, h / n3, Tm, tm.pcg_iters, int(finite_all(data)));
        nvfx_solver_destroy(s);
    }
    // ---------------------------------------------------------------- projection of a smoothed random field
    for (int with_sphere = 0; with_sphere < 2; ++with_sphere) {
        NvfxSolver* s;
        CHECK(nvfx_solver_create(ctx, N, &def, &s));
        CHECK(nvfx_solver_reset(s, with_sphere ? &SPHERE : nullptr, with_sphere));
        CHECK(nvfx_solver_download_state(s, data.data(), sol.data()));
        uint64_t st = 12345;
        std::vector<float> noise(3 * n3);
        for (auto& x : noise) {   // sum of 4 uniforms ~ normal-ish
            double acc = 0;
            for (int k = 0; k < 4; ++k) { st = st * 6364136223846793005ull + 1442695040888963407ull; acc += double(st >> 11) / 9007199254740992.0; }
            x = float((acc - 2.0) * 1.7320508);
        }
        for (int c = 0; c < 3; ++c)
            for (int z = 0; z < N; ++z)
                for (int y = 0; y < N; ++y)
                    for (int x = 0; x < N; ++x) {
                        double acc = 0;
                        for (int dz = -1; dz <= 1; ++dz)
                            for (int dy = -1; dy <= 1; ++dy)
                                for (int dx = -1; dx <= 1; ++dx) {
                                    int a = x + dx, b = y + dy, e = z + dz;
                                    if (a < 0 || b < 0 || e < 0 || a >= N || b >= N || e >= N) continue;
                                    acc += noise[c * n3 + (size_t(e) * N + b) * N + a];
                                }
                        size_t i = (size_t(z) * N + y) * N + x;
                        data[(1 + c) * n3 + i] = sol[i] < 0.5f ? float(acc / 27.0) : 0.0f;
                    }
        CHECK(nvfx_solver_upload_state(s, data.data(), nullptr));
        CHECK(nvfx_solver_project(s));
        CHECK(nvfx_solver_download_divergence(s, div.data()));
        std::printf("projection of a smoothed random field, sphere=%d: max norm div %.2e [< 1e-4]\n", with_sphere, max_of(div));
        nvfx_solver_destroy(s);
    }
    // ---------------------------------------------------------------- mass drift, closed box + sphere, 1000 steps
    {
        NvfxSolverConfig c = def;
        c.density_dissipation = 0.0f;
        NvfxSolver* s;
        CHECK(nvfx_solver_create(ctx, N, &c, &s));
        NvfxObstacle o = {NVFX_OBSTACLE_SPHERE, {0.6f, 0.6f, 0.55f}, {0.1f, 0.1f, 0.1f}};
        CHECK(nvfx_solver_reset(s, &o, 1));
        CHECK(nvfx_solver_download_state(s, data.data(), sol.data()));
        for (int z = 0; z < N; ++z)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x) {
                    size_t i = (size_t(z) * N + y) * N + x;
                    float xx = (x + 0.5f) / N, yy = (y + 0.5f) / N, zz = (z + 0.5f) / N;
                    float blob = std::exp(-((xx - 0.35f) * (xx - 0.35f) + (yy - 0.3f) * (yy - 0.3f) + (zz - 0.45f) * (zz - 0.45f)) / 0.01f) *
                                 (sol[i] < 0.5f ? 1.0f : 0.0f);
                    data[i] = blob;
                    data[4 * n3 + i] = AMB + 400.0f * blob;
                    data[1 * n3 + i] = 0.5f * std::sin(3.1f * yy);
                    data[2 * n3 + i] = 0.0f;
                    data[3 * n3 + i] = 0.3f * std::cos(2.7f * xx);
                    data[5 * n3 + i] = 0.0f;
                }
        CHECK(nvfx_solver_upload_state(s, data.data(), nullptr));
        CHECK(nvfx_solver_project(s));
        CHECK(nvfx_solver_download_state(s, data.data(), nullptr));
        double m0 = 0;
        for (size_t i = 0; i < n3; ++i) m0 += data[i];
        int steps = quick ? 200 : 1000;
        for (int i = 0; i < steps; ++i) CHECK(nvfx_solver_step(s, nullptr, float(SIM_DT)));
        CHECK(nvfx_solver_download_state(s, data.data(), nullptr));
        double m1 = 0;
        for (size_t i = 0; i < n3; ++i) m1 += data[i];
        std::printf("mass %d steps closed box + sphere: %+.5f %%  finite %d  [|.| < 0.1 %%]\n", steps, 100.0 * (m1 / m0 - 1.0),
                    int(finite_all(data)));
        nvfx_solver_destroy(s);
    }
    // ---------------------------------------------------------------- determinism, 200 steps with impulses
    {
        std::vector<float> out[2];
        for (int run = 0; run < 2; ++run) {
            NvfxSolver* s;
            CHECK(nvfx_solver_create(ctx, N, &def, &s));
            CHECK(nvfx_solver_reset(s, &SPHERE, 1));
            for (int i = 0; i < 200; ++i) {
                Scene sc = campfire(i * SIM_DT);
                if (i % 10 == 0) sc.imp.push_back({{0.5f, 0.3f, 0.5f}, 0.08f, {1.0f, 0.5f, -0.5f}});
                NvfxForces f = sc.forces();
                CHECK(nvfx_solver_step(s, &f, float(SIM_DT)));
            }
            out[run].resize(6 * n3);
            CHECK(nvfx_solver_download_state(s, out[run].data(), nullptr));
            nvfx_solver_destroy(s);
        }
        std::printf("determinism 200 steps (campfire + impulses + sphere): bit-identical = %d\n",
                    int(std::memcmp(out[0].data(), out[1].data(), out[0].size() * 4) == 0));
    }
    // ---------------------------------------------------------------- hosted path: adopted device + record_step
    {
        auto* dev = static_cast<ID3D12Device*>(nvfx_context_device(ctx));
        auto* queue = static_cast<ID3D12CommandQueue*>(nvfx_context_queue(ctx));
        NvfxContext* hctx;
        CHECK(nvfx_context_create_from_device(dev, queue, &hctx));
        NvfxSolver *a, *b;
        CHECK(nvfx_solver_create(ctx, N, &def, &a));
        CHECK(nvfx_solver_create(hctx, N, &def, &b));
        CHECK(nvfx_solver_reset(a, &SPHERE, 1));
        CHECK(nvfx_solver_reset(b, &SPHERE, 1));
        ID3D12CommandAllocator* al = nullptr;
        ID3D12GraphicsCommandList* hl = nullptr;
        dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), reinterpret_cast<void**>(&al));
        dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, al, nullptr, __uuidof(ID3D12GraphicsCommandList),
                               reinterpret_cast<void**>(&hl));
        hl->Close();
        for (int i = 0; i < 40; ++i) {
            Scene sc = campfire(i * SIM_DT);
            if (i % 10 == 0) sc.imp.push_back({{0.5f, 0.3f, 0.5f}, 0.08f, {1.0f, 0.5f, -0.5f}});
            NvfxForces f = sc.forces();
            CHECK(nvfx_solver_step(a, &f, float(SIM_DT)));
            CHECK(nvfx_context_wait_idle(hctx));   // the allocator is reused every step
            al->Reset();
            hl->Reset(al, nullptr);
            CHECK(nvfx_solver_record_step(b, hl, &f, float(SIM_DT)));
            hl->Close();
            ID3D12CommandList* lists[] = {hl};
            queue->ExecuteCommandLists(1, lists);
        }
        CHECK(nvfx_context_wait_idle(hctx));
        std::vector<float> da(6 * n3), db(6 * n3);
        CHECK(nvfx_solver_download_state(a, da.data(), nullptr));
        CHECK(nvfx_solver_download_state(b, db.data(), nullptr));
        NvfxSolverTimings tb;
        CHECK(nvfx_solver_get_timings(b, &tb));
        std::printf("hosted path (adopted device, record_step into a host list), 40 steps: bit-identical to nvfx_solver_step = %d"
                    " | timings step %llu total %.3f ms iters %d | texture %p\n",
                    int(std::memcmp(da.data(), db.data(), da.size() * 4) == 0), static_cast<unsigned long long>(tb.step_index),
                    tb.total_ms, tb.pcg_iters, nvfx_solver_texture(b, NVFX_TEX_VEL_DENSITY));
        hl->Release();
        al->Release();
        nvfx_solver_destroy(a);
        nvfx_solver_destroy(b);
        nvfx_context_destroy(hctx);
    }
    // ---------------------------------------------------------------- stress
    for (auto* cfgp : {&def, &rt}) {
        NvfxSolver* s;
        CHECK(nvfx_solver_create(ctx, N, cfgp, &s));
        NvfxObstacle obs[2] = {SPHERE, {NVFX_OBSTACLE_BOX, {0.25f, 0.7f, 0.3f}, {0.1f, 0.02f, 0.15f}}};
        CHECK(nvfx_solver_reset(s, obs, 2));
        uint64_t st = 7;
        auto uni = [&](double a, double b) {
            st = st * 6364136223846793005ull + 1442695040888963407ull;
            return a + (b - a) * (double(st >> 11) / 9007199254740992.0);
        };
        double vmax = 0;
        int steps = quick ? 200 : 1000;
        for (int i = 0; i < steps; ++i) {
            Scene sc = campfire(i * SIM_DT);
            for (auto& e : sc.em) { e.fuel_rate *= 4.0f; e.temperature = 1500.0f; }
            sc.em.push_back({{0.3f, 0.2f, 0.7f}, 0.08f, 6.0f, 1200.0f, 20.0f, {1.5f, 2.0f, 0.0f}});
            if (i % 7 == 0) {
                NvfxImpulse m = {{float(uni(0.15, 0.85)), float(uni(0.15, 0.85)), float(uni(0.15, 0.85))}, 0.07f,
                                 {float(uni(-6, 6)), float(uni(-6, 6)), float(uni(-6, 6))}};
                sc.imp.push_back(m);
            }
            sc.wind[0] = float(std::sin(0.01 * i));
            sc.wind[2] = float(std::cos(0.013 * i));
            NvfxForces f = sc.forces();
            CHECK(nvfx_solver_step(s, &f, float(SIM_DT)));
            if (i % 100 == 99) {
                CHECK(nvfx_solver_download_state(s, data.data(), nullptr));
                for (size_t k = n3; k < 4 * n3; ++k) vmax = std::max(vmax, double(std::fabs(data[k])));
            }
        }
        CHECK(nvfx_solver_download_state(s, data.data(), sol.data()));
        CHECK(nvfx_solver_download_divergence(s, div.data()));
        double Tm = 0, wc = 0;
        for (size_t i = 0; i < n3; ++i) {
            Tm = std::max(Tm, double(data[4 * n3 + i]));
            if (sol[i] > 0.5f)
                for (int c = 0; c < 4; ++c) wc = std::max(wc, double(std::fabs(data[c * n3 + i])));
        }
        std::printf("stress %s %d steps: finite %d  max|u| %.2f m/s  Tmax %.0f K  div %.2e  solid clean %.1e\n",
                    cfgp == &def ? "full" : "runtime", steps, int(finite_all(data)), vmax, Tm, max_of(div), wc);
        nvfx_solver_destroy(s);
    }
    nvfx_context_destroy(ctx);
    return 0;
}
