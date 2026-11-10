// nvfx_render_selfcheck: owner self-check + bench of the native ray-marcher, through the C API only.
//   nvfx_render_selfcheck <out_dir> [--frames N] [--steps N] [--no-bench] [--no-images] [--width W --height H]
//                                   [--scale S]
// Steps the native solver (campfire; plume + sphere; vfx/sim3d/selfcheck.py scenes, default config) for --steps
// (150) steps, renders at 2560x1440, march_scale 0.5, aux on, with three cameras (default, orbit 60 deg, close-up;
// the previous camera is 1 deg of orbit back so motion has a camera part), writes PPM/PGM images to out_dir, and checks:
//   determinism (two draws bit-identical), aux on/off colour bit-identical, REUSE_VOLUME == full draw with the same
//   camera, hosted record_draw == draw.
// Bench: GPU timestamps of >= --frames draws (300): "full" = solver step + new volume every frame, "view" =
// REUSE_VOLUME camera-only redraws; median and p10 per stage. Not the gate tests (native/tests/).
#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "nvfx.h"
#include "nvfx_render.h"

using Microsoft::WRL::ComPtr;

static const double SIM_DT = 1.0 / 30.0;

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
    float wind[3] = {0, 0, 0};
    NvfxForces forces() const {
        NvfxForces f = {};
        f.emitters = em.data();
        f.num_emitters = int32_t(em.size());
        std::memcpy(f.wind, wind, sizeof wind);
        return f;
    }
};
static Scene campfire(double t) {
    Scene s;
    const double spots[5][2] = {{0.5, 0.5}, {0.44, 0.47}, {0.56, 0.47}, {0.47, 0.56}, {0.55, 0.55}};
    for (int k = 0; k < 5; ++k) {
        double sc = 1.0 + 0.35 * std::sin(2.3 * t + 1.7 * k);
        s.em.push_back({{float(spots[k][0]), 0.07f, float(spots[k][1])}, 0.045f, 0.0f, 900.0f, float(8.0 * sc), {0.0f, 0.4f, 0.0f}});
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

// orbit of the default camera about the vertical axis through its target
static NvfxCamera orbit(double deg) {
    NvfxCamera c;
    nvfx_camera_default(&c);
    double a = deg * 3.141592653589793 / 180.0;
    double ox = c.position[0] - c.target[0], oz = c.position[2] - c.target[2];
    c.position[0] = float(c.target[0] + std::cos(a) * ox + std::sin(a) * oz);
    c.position[2] = float(c.target[2] - std::sin(a) * ox + std::cos(a) * oz);
    return c;
}
static NvfxCamera closeup() {
    NvfxCamera c;
    nvfx_camera_default(&c);
    c.position[0] = 0.5f; c.position[1] = 0.5f; c.position[2] = -0.45f;
    c.target[0] = 0.5f; c.target[1] = 0.4f; c.target[2] = 0.5f;
    return c;
}

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

static double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[size_t(q * double(v.size() - 1) + 0.5)];
}

struct Frame {
    std::vector<uint8_t> color;
    std::vector<float> alpha, depth, motion;
};
static Frame download(NvfxRenderer* r, int w, int h, bool aux) {
    Frame f;
    f.color.resize(size_t(w) * h * 4);
    CHECK(nvfx_renderer_download(r, NVFX_RENDER_TEX_COLOR, f.color.data()));
    if (aux) {
        f.alpha.resize(size_t(w) * h);
        f.depth.resize(size_t(w) * h);
        f.motion.resize(size_t(w) * h * 2);
        CHECK(nvfx_renderer_download(r, NVFX_RENDER_TEX_ALPHA, f.alpha.data()));
        CHECK(nvfx_renderer_download(r, NVFX_RENDER_TEX_DEPTH, f.depth.data()));
        CHECK(nvfx_renderer_download(r, NVFX_RENDER_TEX_MOTION, f.motion.data()));
    }
    return f;
}
static bool same_bits(const Frame& a, const Frame& b, bool aux) {
    if (a.color != b.color) return false;
    if (!aux) return true;
    auto eq = [](const std::vector<float>& x, const std::vector<float>& y) {
        return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * 4) == 0;
    };
    return eq(a.alpha, b.alpha) && eq(a.depth, b.depth) && eq(a.motion, b.motion);
}
static size_t ndiff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) n += a[i] != b[i];
    return n;
}

static void write_ppm(const std::string& path, int w, int h, const std::vector<uint8_t>& rgb3) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); return; }
    std::fprintf(f, "P6\n%d %d\n255\n", w, h);
    std::fwrite(rgb3.data(), 1, rgb3.size(), f);
    std::fclose(f);
}
static void write_pgm(const std::string& path, int w, int h, const std::vector<uint8_t>& g) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); return; }
    std::fprintf(f, "P5\n%d %d\n255\n", w, h);
    std::fwrite(g.data(), 1, g.size(), f);
    std::fclose(f);
}
static uint8_t u8(double v) { return uint8_t(std::clamp(int(std::lround(v)), 0, 255)); }

static void save_images(const std::string& dir, const std::string& tag, int w, int h, const Frame& f) {
    size_t n = size_t(w) * h;
    std::vector<uint8_t> rgb(n * 3), a(n), d(n), m(n * 3);
    for (size_t i = 0; i < n; ++i) {
        rgb[3 * i + 0] = f.color[4 * i + 0];
        rgb[3 * i + 1] = f.color[4 * i + 1];
        rgb[3 * i + 2] = f.color[4 * i + 2];
        a[i] = u8(f.alpha[i] * 255.0);
        d[i] = u8(f.depth[i] / 3.0 * 255.0);                 // 0..3 m
        m[3 * i + 0] = u8(128.0 + 4.0 * f.motion[i]);         // 1/4 px per level
        m[3 * i + 1] = u8(128.0 + 4.0 * f.motion[n + i]);
        m[3 * i + 2] = u8(f.alpha[i] * 255.0);
    }
    write_ppm(dir + "\\" + tag + "_color.ppm", w, h, rgb);
    write_pgm(dir + "\\" + tag + "_alpha.pgm", w, h, a);
    write_pgm(dir + "\\" + tag + "_depth.pgm", w, h, d);
    write_ppm(dir + "\\" + tag + "_motion.ppm", w, h, m);
}

static void aux_stats(const Frame& f, int w, int h, const char* tag, const NvfxCamera& cam) {
    size_t n = size_t(w) * h, cov = 0;
    double dsum = 0, mx = 0, my = 0, mabs = 0;
    for (size_t i = 0; i < n; ++i)
        if (f.alpha[i] > 0.05f) {
            ++cov;
            dsum += f.depth[i];
            mx += f.motion[i];
            my += f.motion[n + i];
            mabs = std::max(mabs, double(std::hypot(f.motion[i], f.motion[n + i])));
        }
    int32_t bb[4];
    nvfx_render_screen_bbox(&cam, w, h, bb);
    double c = cov ? double(cov) : 1.0;
    std::printf("  %-22s alpha>0.05: %5.1f %% of frame, mean depth %.3f m, mean motion (%+.2f, %+.2f) px, max |motion| %.1f px, "
                "bbox (%d,%d)-(%d,%d)\n",
                tag, 100.0 * double(cov) / double(n), dsum / c, mx / c, my / c, mabs, bb[0], bb[1], bb[2], bb[3]);
}

int main(int argc, char** argv) {
    std::string dir;
    int frames = 300, steps = 150, W = 2560, H = 1440;
    float scale = 0.5f;
    bool bench = true, images = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--frames" && i + 1 < argc) frames = std::atoi(argv[++i]);
        else if (a == "--steps" && i + 1 < argc) steps = std::atoi(argv[++i]);
        else if (a == "--width" && i + 1 < argc) W = std::atoi(argv[++i]);
        else if (a == "--height" && i + 1 < argc) H = std::atoi(argv[++i]);
        else if (a == "--scale" && i + 1 < argc) scale = float(std::atof(argv[++i]));
        else if (a == "--no-bench") bench = false;
        else if (a == "--no-images") images = false;
        else if (a[0] != '-') dir = a;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (dir.empty() && images) {
        std::fprintf(stderr, "usage: nvfx_render_selfcheck <out_dir (outside the repo)> [--frames N] [--steps N] "
                             "[--no-bench] [--no-images] [--width W --height H --scale S]\n");
        return 2;
    }
    if (images) CreateDirectoryA(dir.c_str(), nullptr);
    std::printf("%s\n%s\n", nvfx_version_string(), gpu_status().c_str());

    NvfxContextDesc cd = {-1, 0};
    NvfxContext* ctx;
    CHECK(nvfx_context_create(&cd, &ctx));
    NvfxRenderer* r;
    CHECK(nvfx_renderer_create(ctx, W, H, scale, &r));
    int mw, mh;
    nvfx_renderer_march_size(r, &mw, &mh);
    std::printf("renderer %dx%d, march %dx%d\n", W, H, mw, mh);

    struct { const char* name; Scene (*fn)(double); bool sphere; } scenes[] = {{"campfire", campfire, false},
                                                                                  {"plume", plume, true}};
    struct { const char* name; NvfxCamera cam, prev; } cams[] = {
        {"default", orbit(0.0), orbit(-1.0)}, {"orbit60", orbit(60.0), orbit(59.0)}, {"closeup", closeup(), closeup()}};
    int fails = 0;

    for (auto& sc : scenes) {
        NvfxSolver* s;
        CHECK(nvfx_solver_create(ctx, 64, nullptr, &s));
        CHECK(nvfx_solver_reset(s, sc.sphere ? &SPHERE : nullptr, sc.sphere ? 1 : 0));
        for (int i = 0; i < steps; ++i) {
            Scene f = sc.fn(i * SIM_DT);
            NvfxForces fo = f.forces();
            CHECK(nvfx_solver_step(s, &fo, float(SIM_DT)));
        }
        std::printf("\n[%s] after %d steps\n", sc.name, steps);
        for (auto& c : cams) {
            CHECK(nvfx_renderer_draw(r, s, &c.cam, &c.prev, NVFX_RENDER_AUX));
            Frame a = download(r, W, H, true);
            std::string tag = std::string(sc.name) + "_" + c.name;
            aux_stats(a, W, H, tag.c_str(), c.cam);
            if (images) save_images(dir, tag, W, H, a);
            // R6: same inputs twice
            CHECK(nvfx_renderer_draw(r, s, &c.cam, &c.prev, NVFX_RENDER_AUX));
            Frame b = download(r, W, H, true);
            bool det = same_bits(a, b, true);
            // R5: aux off -> same colour
            CHECK(nvfx_renderer_draw(r, s, &c.cam, &c.prev, 0));
            Frame no_aux = download(r, W, H, false);
            bool r5 = no_aux.color == a.color;
            // REUSE_VOLUME after a full draw of another camera == full draw of this camera
            const NvfxCamera& other = cams[(&c - cams + 1) % 3].cam;
            CHECK(nvfx_renderer_draw(r, s, &other, nullptr, 0));
            CHECK(nvfx_renderer_draw(r, s, &c.cam, &c.prev, NVFX_RENDER_AUX | NVFX_RENDER_REUSE_VOLUME));
            Frame re = download(r, W, H, true);
            bool reuse = same_bits(a, re, true);
            std::printf("    determinism %s, aux on/off colour %s (%zu bytes differ), reuse==full %s\n", det ? "OK" : "FAIL",
                        r5 ? "identical" : "DIFFERENT", ndiff(no_aux.color, a.color), reuse ? "OK" : "FAIL");
            fails += !det + !r5 + !reuse;
        }
        // hosted path: record_draw into our own list on the context queue == draw
        {
            auto* dev = static_cast<ID3D12Device*>(nvfx_context_device(ctx));
            auto* queue = static_cast<ID3D12CommandQueue*>(nvfx_context_queue(ctx));
            ComPtr<ID3D12CommandAllocator> al;
            ComPtr<ID3D12GraphicsCommandList> gl;
            bool okh = SUCCEEDED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&al))) &&
                       SUCCEEDED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, al.Get(), nullptr, IID_PPV_ARGS(&gl)));
            if (okh) {
                CHECK(nvfx_renderer_draw(r, s, &cams[0].cam, &cams[0].prev, NVFX_RENDER_AUX));
                Frame a = download(r, W, H, true);
                CHECK(nvfx_renderer_record_draw(r, gl.Get(), s, &cams[0].cam, &cams[0].prev, NVFX_RENDER_AUX));
                gl->Close();
                ID3D12CommandList* l = gl.Get();
                queue->ExecuteCommandLists(1, &l);
                CHECK(nvfx_context_wait_idle(ctx));
                Frame b = download(r, W, H, true);
                bool same = same_bits(a, b, true);
                std::printf("  hosted record_draw == draw: %s\n", same ? "OK" : "FAIL");
                fails += !same;
            } else {
                std::printf("  hosted record_draw: could not create a command list\n");
                ++fails;
            }
        }

        if (bench) {
            for (int mode = 0; mode < 2; ++mode) {
                std::vector<double> pr, ma, po, to;
                uint64_t last = 0;
                const int warm = 30;
                int i = 0;
                CHECK(nvfx_renderer_draw(r, s, &cams[0].cam, nullptr, NVFX_RENDER_AUX));   // a volume to reuse
                for (; int(to.size()) < frames && i < frames * 20; ++i) {
                    double ang = 0.5 * i;   // slow orbit, previous camera 0.5 deg back
                    NvfxCamera cam = orbit(ang), prev = orbit(ang - 0.5);
                    int flags = NVFX_RENDER_AUX;
                    if (mode == 0) {
                        Scene f = sc.fn((steps + i) * SIM_DT);
                        NvfxForces fo = f.forces();
                        CHECK(nvfx_solver_step(s, &fo, float(SIM_DT)));
                    } else {
                        flags |= NVFX_RENDER_REUSE_VOLUME;
                    }
                    CHECK(nvfx_renderer_draw(r, s, &cam, &prev, flags));
                    NvfxRenderTimings t;
                    CHECK(nvfx_renderer_get_timings(r, &t));
                    if (t.draw_index > last) {
                        last = t.draw_index;
                        if (i > warm && ((t.flags & NVFX_RENDER_REUSE_VOLUME) != 0) == (mode == 1)) {
                            pr.push_back(t.prep_ms); ma.push_back(t.march_ms); po.push_back(t.post_ms); to.push_back(t.total_ms);
                        }
                    }
                }
                CHECK(nvfx_context_wait_idle(ctx));
                std::printf("  bench %-4s (%zu draws, AUX, %dx%d scale %.2f): median / p10 ms  prep %.3f / %.3f  march %.3f / %.3f  "
                            "post %.3f / %.3f  total %.3f / %.3f\n",
                            mode == 0 ? "full" : "view", to.size(), W, H, double(scale), pct(pr, 0.5), pct(pr, 0.1),
                            pct(ma, 0.5), pct(ma, 0.1), pct(po, 0.5), pct(po, 0.1), pct(to, 0.5), pct(to, 0.1));
            }
        }
        nvfx_solver_destroy(s);
    }
    std::printf("\n%s\n", gpu_status().c_str());
    nvfx_renderer_destroy(r);
    nvfx_context_destroy(ctx);
    std::printf("self-check: %s (%d failures)\n", fails ? "FAIL" : "OK", fails);
    return fails ? 1 : 0;
}
