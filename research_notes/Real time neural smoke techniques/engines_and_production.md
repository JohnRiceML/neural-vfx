# Engines and Production Practice for Real-Time Volumetric Smoke/Fire (2018–2026)

Scope note: about 20 search/fetch calls. Primary sources were Epic docs, NVIDIA Flow docs, the JangaFX roadmap, Zibra blogs and FAQ, arXiv papers, and the Unity docs. Several studio talks (Frostbite, CoD, Remedy, Valve) could not be reached in this pass. They are listed under Gaps as leads, and none of their numbers appear in the Cited Findings.

## 1. Unreal Engine: Niagara Fluids, Heterogeneous Volumes, SVT, VDB, path-traced volumes

### Takeaway
Niagara Fluids 3D Gas is a dense, grid-based (Eulerian-style) GPU solver with iterative pressure relaxation. Epic positions it as a hero/cinematic tool and recommends baking to flipbooks for real-time use. Epic's docs publish no official cost figures. The only measured number found is a forum report of about 6 ms on an RTX 4080 for a cached 3D gas played back through Heterogeneous Volumes. Rendering goes either through the Niagara ray-march materials or through the still-Experimental Heterogeneous Volumes path, which ray-marches Sparse Volume Textures with a dense lighting cache. A 64^3 solver plus learned detail at 6 ms competes directly with the "hero" tier of Niagara Fluids, not with its everyday use.

### Cited Findings
- 3D Gas is grid-based: the gas is a grid of cells, each storing density, temperature and velocity. Controls include Pressure Relaxation (keep near 1), Pressure Solve Iterations (more is more accurate and more expensive), vorticity confinement, per-quantity dissipation, buoyancy and gravity. The docs do not name the advection scheme or a default resolution. — [Epic: Niagara Fluids Reference](https://dev.epicgames.com/documentation/unreal-engine/niagara-fluids-reference-in-unreal-engine)
- Resolution is set by "Resolution Max Axis" (cells along the longest side of the bounds) or by cell size. Smaller cells improve quality at a performance cost. Each axis has an open/closed boundary toggle. — [Epic: Niagara Fluids Reference](https://dev.epicgames.com/documentation/unreal-engine/niagara-fluids-reference-in-unreal-engine)
- Epic's guidance says 3D sims carry "higher memory and GPU cost", are "best for hero effects or cinematics", and should be baked into flipbooks for real-time use. — [Epic: Niagara Fluids Reference](https://dev.epicgames.com/documentation/unreal-engine/niagara-fluids-reference-in-unreal-engine)
- Rendering: Render Step Size Mult defaults to one cell. Density renders as Linear or Curve with an albedo setting, and temperature as Black Body or Curve. Shadows are baked by stepping through the volume, with Shadow Quality and Shadow Max Steps settings. Lighting comes from directional-light data interfaces. The Cine template advects texture coordinates in a simulation stage to modulate the render with noise. This is Epic's built-in "detail upres" trick, a texture-advection analogue of the learned-detail idea. — [Epic: Niagara Fluids Reference](https://dev.epicgames.com/documentation/unreal-engine/niagara-fluids-reference-in-unreal-engine)
- Sparse simulation: the Niagara Fluids reference does not mention sparse grids, so the solver is treated here as dense. — [Epic: Niagara Fluids Reference](https://dev.epicgames.com/documentation/unreal-engine/niagara-fluids-reference-in-unreal-engine) (absence of mention; unverified)
- Measured cost (anecdotal): a cached 3D gas sim rendered through Heterogeneous Volumes on an RTX 4080 dropped an empty scene from about 110 to about 50 fps. GPU profiling showed roughly 6 ms. Resolutions below the default of 200 lost too much quality. — [Epic forum: Low performance with cached gas simulations](https://forums.unrealengine.com/t/low-performance-with-cached-gas-simulations-niagara-fluids-3d-gas-sim/2177751)
- Unsourced blog numbers: 3D smoke at 64^3 costs about 1.8 ms, 3D smoke at 128^3 about 5.5 ms, and 3D fire at 64^3 about 2.4 ms. No hardware or methodology is given, so treat these as unverified. — [StraySpark blog](https://www.strayspark.studio/blog/niagara-fluids-for-gameplay-beyond-vfx-2026)
- Heterogeneous Volumes (HV) are **Experimental**. They render volume-domain materials sampling static or animated SVTs, using either real-time ray marching or the Path Tracer. Step Factor sets the march step as a multiple of the volume resolution. — [Epic: Heterogeneous Volumes](https://dev.epicgames.com/documentation/unreal-engine/heterogeneous-volumes-in-unreal-engine?lang=en-US)
- HV lighting uses an internal lighting cache, sized by Lighting Downsample Factor. The cache is **dense even though the SVT is sparse**, capped at 1024x1024x512. Indirect lighting is off by default (`r.HeterogeneousVolumes.IndirectLighting 0`). — [Epic: Heterogeneous Volumes](https://dev.epicgames.com/documentation/unreal-engine/heterogeneous-volumes-in-unreal-engine?lang=en-US); [Epic: Sparse Volume Textures](https://dev.epicgames.com/documentation/en-us/unreal-engine/sparse-volume-textures-in-unreal-engine)
- HV defaults and scalability cvars:

  | Cvar | Default |
  |---|---|
  | `MaxStepCount` | 512 |
  | `Shadows.Resolution` (volumetric shadow map) | 512 |
  | `DownsampleFactor` (viewport downsample) | 1.0 |
  | Ortho grid and frustum grid shading rate | 4 (cinematic recommendation: 1) |
  | Bottom-level memory | 128 MB (cinematic: 512) |
  | `FrustumGrid.DepthSliceCount` | 512 |

  If the memory limit is too small for the chosen shading rate, voxels go unallocated and parts of the volume disappear. — [Epic: Heterogeneous Volumes](https://dev.epicgames.com/documentation/unreal-engine/heterogeneous-volumes-in-unreal-engine?lang=en-US)
- HV known issues: overlapping volumes can look blocky because homogeneous-region detection works per volume, and path-tracer invalidation fails during animation, which causes a screen-door effect. For VDB-to-SVT import, Epic recommends 8-bit unorm density with other attributes in 16-bit float. — [Epic: Heterogeneous Volumes](https://dev.epicgames.com/documentation/unreal-engine/heterogeneous-volumes-in-unreal-engine?lang=en-US)
- SVT limits: the page table is capped at 2 GB, and the virtual resolution is at most 32k on X and Y. Streaming performance "usually starts to suffer" for animated SVT frames above 30–50 MB. Streaming Mip Bias lowers the operating resolution. — [Epic: Sparse Volume Textures](https://dev.epicgames.com/documentation/en-us/unreal-engine/sparse-volume-textures-in-unreal-engine)
- An academic stress test (GEOMAR, 2025): stock UE5 3D textures top out around 2048^3, with an allocation limit near 1 gigavoxel. SVT needed engine fixes for integer overflows to exceed about 2 gigavoxels compressed. About 6 billion voxels rendered at about 20 fps on an RTX 3500 Ada with the modified engine. — [arXiv 2504.07485](https://arxiv.org/pdf/2504.07485)
- Zibra reports that UE Heterogeneous Volume rendering is about 3x slower than its own custom ray-marcher for the same VDB content. This is a vendor claim and the comparison conditions are unclear. — [Zibra blog: ZibraVDB performance & quality](https://www.zibra.ai/blog-posts/zibravdb-performance-and-quality-optimization)
- Epic's legacy froxel Volumetric Fog: the docs say cost is driven mainly by volume texture resolution, and that lights with Cast Volumetric Shadow cost about 3x more than unshadowed lights. A third-party write-up quotes about 1 ms on PS4 (High) and 3 ms on a GTX 970 (Epic), but the primary source could not be checked. — [Epic: Volumetric Fog](https://dev.epicgames.com/documentation/en-us/unreal-engine/volumetric-fog-in-unreal-engine); [iRender summary](https://irendering.net/exploring-volumetric-fog-in-unreal-engine/)

### Inferences
- Built-in UE quality bar: dense grids, Max Axis around 128–256 for hero shots (the forum user found below 200 unacceptable), noise-advection detail in the Cine template, and directional-light shadow marching. HV adds a sparse SVT input but a dense lighting cache and many-ms costs. A 64^3 solve with learned upres that looks like 200+ Max Axis, at 6 ms or less including rendering, would beat the observed HV playback cost (about 6 ms at higher resolution).
- Integration targets inside UE: (a) a Niagara simulation stage or data interface producing a grid, rendered with HV materials or a custom ray-march material; (b) writing SVT-compatible output. The dense lighting cache in HV is a weakness a custom renderer could exploit.

### Gaps
- No Epic-published ms costs for Niagara Fluids at any resolution. No confirmation of the advection scheme (likely semi-Lagrangian or MacCormack, unverified) or of sparse support in UE 5.4–5.7+.
- The UE version in which HV and SVT shipped (believed to be 5.3 experimental) was not verified. Lumen interaction with HV is not documented on the page.
- The OpenVDB/"VDB playback" path in UE goes through SVT import, which is verified above. No separate real-time VDB playback feature was found.

## 2. NVIDIA Flow (1.x GameWorks and Omniverse-era Flow 2)

### Takeaway
Flow is the canonical real-time **sparse block** smoke/fire solver. It uses dynamically allocated 32x16x16 voxel blocks, optionally 16x8x8, chosen by emitters plus simulation feedback, with no bounding box and a capped block pool. It is HLSL on DX12/Vulkan, with a combustion model and NanoVDB export. NVIDIA publishes no timing numbers in the current docs.

### Cited Findings
- Flow 2 (omni.flowsdk, version 105.x, docs dated 2021–2022) "by default dynamically allocates 32x16x16 voxel blocks based on emitters and feedback from the simulation". Allocation is unbounded except for 32-bit precision and a configurable maximum block count. NvFlowSparse takes an unsorted list of block locations, enforces the maximum, and builds fast-access structures. — [NVIDIA Flow docs](https://nvidia-omniverse.github.io/PhysX/flow/index.html)
- Flow 2 runs on the GPU in HLSL targeting Vulkan and DX12 (DX11 untested). It includes a combustion model for fire and smoke, and NvFlowSparse can build NanoVDB structures for export. — [NVIDIA Flow docs](https://nvidia-omniverse.github.io/PhysX/flow/index.html)
- `enableSmallBlocks` switches to 16x8x8 density-cell blocks. `velocitySubSteps` improves fast-moving behavior at a performance cost. — [Omniverse Flow settings](https://docs.omniverse.nvidia.com/kit/docs/flow/latest/settings.html)
- Flow 1.x (GameWorks, UE4 integration) used D3D11.3 volume tiled resources with 64 KB tiles, which with typical formats gives 32x16x16-cell blocks. Blocks are made resident dynamically in regions of interest, allowing domains larger than a fully resident grid. — [NVIDIA FLOWUe4 Intro](https://docs.nvidia.com/gameworks/content/artisttools/Flow/FLOWUe4_Intro.html)

### Inferences
- 32x16x16 at 4 bytes per cell is 32 KB per channel, so the block size is tied to the hardware tile size. A team building a 64^3 solver could treat 64^3 as 8x16 such blocks and adopt the same emitter-driven activation for localized effects.
- Flow is open (the docs are on the PhysX GitHub pages, implying source in the PhysX repo), which makes it a realistic baseline to benchmark against at equal cost.

### Gaps
- No ms or fps numbers found for Flow 1 or Flow 2. GDC 2016–2018 Flow talks were not retrieved.
- The open-source license and repository status were not confirmed in this pass.
- The ray-march renderer details (Flow ships a self-shadowed ray-marcher) were not documented on the pages read.

## 3. JangaFX EmberGen

### Takeaway
EmberGen is an offline-authoring tool that runs in real time, not a game runtime. Its output reaches games as flipbooks (with motion vectors and normals) or as VDBs, which then go to UE SVT or ZibraVDB. EmberGen 1.x used dense grids up to about 512^3 at launch. 1.0 added upscaling to about 1 billion voxels. 2.0 adds a sparse solver: about 1 billion active voxels on 24 GB, with a public beta labeled Q4 2026. The roadmap lists no engine runtime or SDK plans.

### Cited Findings
- Beta (2019): voxel resolutions up to 512^3 with 256^3 as the target. It "runs fast on a 2080 Ti" and drops to 5–6 fps on a GTX 1080. Flipbook exports (EXR/PNG/TGA) include motion vectors, normal maps and depth. — [RealTimeVFX forum, EmberGen beta](https://realtimevfx.com/t/embergen-beta-released-real-time-fluid-sims-for-instant-flipbook-generation/9915)
- 1.0 (2023): simulation upscaling "to a billion voxels", solver optimizations, and Alembic particle export. VDB export is supported. — [CG Channel: EmberGen 1.0](https://www.cgchannel.com/2023/03/jangafx-releases-embergen-1-0/); [GameFromScratch](https://gamefromscratch.com/embergen-real-time-fluid-simulation/)
- 2.0 sparse solver (roadmap, marked completed internally Apr 2025): it simulates only where data exists, from visible smoke or an activity map plus a predefined region. The simulation space limit is 8192^3. Claimed capacity is 10–15x more voxels than 1.x.

  | VRAM | Active sparse voxels (2.0) |
  |---|---|
  | 6 GiB | 200M |
  | 8 GiB | 300M |
  | 16 GiB | 600M |
  | 24 GiB | 1B |
  | 48 GiB | 2B |

  The hard cap is 2,147,450,880 active voxels. Upscaled counts exceed 8B. The roadmap also lists a new optimized sparse renderer, HDRI/atmosphere indirect lighting with adaptive refinement, an experimental Gaussian-splat coarse renderer (about 64 MB VRAM), VDB import to resume simulation, and a scriptable API. The public beta is labeled Q4 2026 and production-ready 2027. "VDB compression" and "real-time path tracing" are marked delayed. Game-facing items are "4D volume slice exports" and "animated volume textures for games". — [JangaFX roadmap](https://jangafx.com/roadmap/embergen)
- Internal inconsistency: the roadmap claims 10–15x, but its own table's 1.x dense-equivalent column implies about 5x. — [JangaFX roadmap](https://jangafx.com/roadmap/embergen)
- Demo: about 1B voxels simulated and rendered "interactively" on an RTX PRO 6000 Blackwell, with the RTX 5090 described as similar but with less headroom. No fps was given, so this is marketing without measurement. — [80.lv, Oct 2025](https://80.lv/articles/upcoming-embergen-s-sparse-solver-handles-one-billion-voxels-in-real-time)
- Release timing conflicts: CG Channel (Jan 2025) said 2.0 was due in 2025 or 2026. The roadmap now says beta Q4 2026. — [CG Channel: EmberGen 2.0](https://www.cgchannel.com/2025/01/check-out-the-new-features-due-in-embergen-2-0/)

### Inferences
- EmberGen's look comes from high voxel counts (256^3 to 512^3 and up), upscaled components (for example 2x smoke resolution over 1x velocity, as the roadmap describes), and an offline-quality renderer with environment lighting. In games these reach the player as 2D flipbooks, so a runtime system competes with "EmberGen quality frozen into a flipbook".
- The upscaling (separate smoke and velocity resolutions) is the classical analogue of the learned-detail stage. Matching EmberGen's upscaled look at 64^3 velocity is a meaningful bar.

### Gaps
- No official per-frame ms numbers for EmberGen at a given resolution and GPU.
- The details of the EmberGen upscaling algorithm (noise-based vs. sub-step advection) were not found.
- No confirmed runtime or engine integration product.

## 4. Flipbooks, 6-way lightmaps, motion vectors, VDB playback, and shipped-title volumetrics

### Takeaway
Baked sprite smoke (flipbooks with six-way lightmaps and motion-vector frame blending) is still the default in AAA production. Unity ships six-way lighting as a first-class VFX Graph output in URP and HDRP. Real-time volumetric smoke in shipped titles is rare and gameplay-driven: CS2's voxel flood-fill smoke and Guerrilla's Nubis voxel clouds. Detailed official numbers for these were not retrieved. A 2026 arXiv paper on neural six-way lightmaps is the closest academic competitor to a neural real-time smoke system.

### Cited Findings
- Unity VFX Graph six-way lighting pre-bakes directional lighting into six lightmaps packed into two RGBA textures, which the shader blends at runtime by scene light direction. It is supported in URP and HDRP, has no specular from reflection probes, and Unity documents a tool to generate the textures. — [Unity docs: six-way lighting](https://docs.unity3d.com/Packages/com.unity.visualeffectgraph@17.2/manual/six-way-lighting.html); [Unity docs: generate six-way lightmaps](https://docs.unity3d.com/Packages/com.unity.visualeffectgraph@17.5/manual/use-tools-generate-six-way-lightmap-textures.html); [Unity forum announcement](https://discussions.unity.com/t/vfx-graph-introducing-six-way-lighting-for-smoke-dust-and-clouds-vfx/906564)
- "Real-time Neural Six-way Lightmaps" (arXiv 2604.03748, 2026) calls six-way lightmaps efficient and visually compelling "particularly in AAA game production". It says standard six-way billboards show artifacts for asymmetric smoke. — [arXiv 2604.03748](https://arxiv.org/html/2604.03748)
- That paper's method: an LBM solver with a 100^3 velocity grid and a 400^3 density grid. A coarse large-step ray-march produces a 512x512 guiding map, and a U-Net predicts six lightmaps plus transparency and emission. Reported timings:
  - guiding map about 2 ms, inference 1.2 ms (3 ms in another section, an internal inconsistency), final render under 0.3 ms, shadow 0.8 ms
  - simulation is expensive: velocity 4.0 ms, density advection 11.5 ms; jet demo 10 ms total (7 ms simulation, 3 ms rendering)
  - hardware: unnamed 24 GB, 16,384-core GPU, consistent with an RTX 4090
  - compared against MRPNN: about 4 ms inference plus over 170 ms precompute for 400^3

  — [arXiv 2604.03748](https://arxiv.org/html/2604.03748)
- CS2 smoke grenades grow to fill space and can be cut by gunshots and HE grenades. — [Wikipedia: Counter-Strike 2](https://en.wikipedia.org/wiki/Counter-Strike_2)
- Recreations of CS2 smoke use a voxelized level, a limited flood fill, and ray-marching. These are hobbyist reconstructions, not Valve documentation. — [GarrettGunnell/CS2-Smoke-Grenades](https://github.com/GarrettGunnell/CS2-Smoke-Grenades); [DX11 CS2 smoke recreation](https://github.com/lolgube010/DX11-TGA-Portfolio-CS2-Smoke)
- Guerrilla, "Nubis^3" (SIGGRAPH 2023 Advances): a voxel-based cloud renderer and modeling approach built in under 6 months for Horizon Forbidden West: Burning Shores. It works as a volumetric skybox with time of day and supports flying through clouds. Assets were authored offline as NVDF voxel clouds and released as TGA slices and VDB. This is baked voxel data rendered by real-time ray-marching, not a runtime sim. — [SIGGRAPH 2023 Advances course](https://advances.realtimerendering.com/s2023/index.html); [Schneider on X](https://x.com/vonschneidz/status/1689051134596718593)

### Inferences
- Shipped-game state of practice: (1) sprite flipbooks with six-way lighting and motion-vector blending for nearly all smoke and fire; (2) baked voxel assets ray-marched at runtime (Nubis NVDF, UE SVT/VDB playback); (3) cheap gameplay-driven voxel propagation without Navier–Stokes (CS2 flood fill, as reconstructed). Full Navier–Stokes smoke at runtime in shipped AAA games is rare. The neural system's realistic competition is flipbook quality at about 0.1–0.5 ms and Niagara Fluids hero sims at several ms.
- The neural six-way paper shows a learned stage of about 1–3 ms is acceptable to reviewers, but its 400^3 density advection (11.5 ms) is the bottleneck. A 64^3 solver plus learned detail directly attacks that.

### Gaps
- No Valve primary source (talk or blog) on CS2 smoke internals, voxel size or cost was found.
- No Nubis^3 ms figures were extracted. The slides exist on the course page but were not fetched. From memory and unverified: the original 2015 Nubis budget was about 2 ms on PS4.
- Not retrieved: Frostbite (Hillaire's 2015 physically based volumetrics, pre-2018), Call of Duty volumetrics, God of War (2018) volumetrics, Alan Wake 2 (Remedy Northlight volumetric fog/lighting), and id Tech. No numbers from them are claimed here.
- No primary source for motion-vector flipbook blending costs or its standard references (for example, the Guerrilla/Klemen talks). This is noted as a widely used technique without citation in this pass.

## 5. Unity, Zibra, and middleware

### Takeaway
Unity's core offers sprite-based six-way smoke and HDRP froxel volumetric fog, but no built-in 3D fluid solver. Zibra provides a third-party GPU smoke and fire simulation for Unity, plus ZibraVDB for compressed VDB playback in UE and Unity. By Zibra's own FAQ, ZibraVDB is **not** neural. Zibra's "AI" branding is mostly marketing, and its one measured smoke number is a single anecdotal 2 ms.

### Cited Findings
- Zibra Smoke & Fire is described as "AI-powered" real-time smoke and fire simulation for Unity. In a creator case study, a full-screen smoke simulation with scene interactions added "about 2 ms" to frame time. No hardware, resolution or method was given. The neural element named on that page is "Neural SDF" object representations for colliders, in the liquid section. — [Zibra case study](https://www.zibra.ai/case-studies/enhancing-game-visuals-with-zibra-liquid-and-zibra-smoke-fire-a-creators-journey)
- Zibra Effects claims to reduce collider memory by about 40x on average through neural object representations. — [ACM SIGGRAPH 2024 Real-Time Live / talk: Revolutionizing VFX Production with Real-Time Volumetric Effects](https://dl.acm.org/doi/fullHtml/10.1145/3641520.3665311)
- ZibraVDB:
  - Rate: default Quality 0.6 gives about 40x compression, with 20x on average and up to 100x elsewhere. An earlier test reported 16.46x at 37.80 dB PSNR.
  - Decompression: about 0.02–0.03 ms per million voxels on an RTX 3090.
  - Pipeline: a compute-shader decompressor plus a custom ray-marcher. It uses no tensor or RT cores, keeps sequences compressed, and decompresses on demand.
  - Memory: VRAM equal to 2–3x the frame size. A 180-frame effect takes 10 GB.
  - Lighting: illumination texture at 1/4 resolution, up to 15 lights.

  These are vendor numbers, partly shown only in charts. — [Zibra: ZibraVDB perf & quality](https://www.zibra.ai/blog-posts/zibravdb-performance-and-quality-optimization); [ZibraVDB product page](https://www.zibra.ai/zibra-vdb-compression); [CG Channel: ZibraVDB](https://www.cgchannel.com/2024/07/zibraai-unveils-zibra-vdb-compression/)
- Zibra's FAQ states that ZibraVDB does not use generative AI or neural networks. — [ZibraVDB product page](https://www.zibra.ai/zibra-vdb-compression) (as reported in search summary)
- The ZibraVDB UE5 test plugin reached 120 fps on an RTX 3090 (vendor claim). The Unreal plugin was made free in 2025. — [Zibra blog](https://www.zibra.ai/blog-posts/zibravdb-a-new-solution-by-zibraai?3f2a3f4a_page=1); [Digital Production](https://digitalproduction.com/2025/01/16/free-zibravdb-for-unreal-smoke-fire-and-simulated-chaos/)

### Inferences
- In this market, "AI/neural" smoke is so far mostly branding. A system with a genuinely learned detail stage and published per-pass ms numbers would stand out.
- The ZibraVDB decompression rate (about 0.025 ms per million voxels) is a useful reference: a 64^3 grid (0.26M voxels) is trivial at that rate, while a 256^3 upres (16.7M voxels) would be about 0.4 ms of decode alone.

### Gaps
- Grid resolutions and the solver type of Zibra Smoke & Fire (grid vs. particle, and what exactly is "AI") were not found.
- HDRP volumetric fog and local volumetric fog costs were not retrieved from Unity docs. No Unity-native 3D fluid package was found.

## 6. Sparse volumetric data structures and real-time use

### Takeaway
NanoVDB is the GPU-friendly, read-mostly VDB (2021). fVDB (2024) adds an IndexGrid that separates topology from data for sparse ML. NeuralVDB compresses 10x beyond NanoVDB and 100x beyond OpenVDB, but explicitly does not target fast random access, so it decodes to VDB before rendering. Real-time sparse smoke in practice means block or tiled sparsity (Flow's 32x16x16 bricks, EmberGen 2.0's activity map), not deep VDB trees.

### Cited Findings
- NanoVDB (Museth, SIGGRAPH 2021 Talks) is a portable C++11/C99/graphics-API VDB "for real-time rendering and simulation" of sparse volumes. — [ACM DL: NanoVDB](https://dl.acm.org/doi/10.1145/3450623.3464653)
- NeuralVDB (ACM TOG 2024, arXiv 2208.04448):
  - NanoVDB compression "rarely exceeds 6x" at low quantization noise.
  - NeuralVDB is typically one order of magnitude above NanoVDB and two above OpenVDB.
  - Peaks are 61.2x for SDFs and 140.9x for density volumes.
  - Faster rendering is an explicit non-goal. Decompression can be in-core or out-of-core.

  — [arXiv 2208.04448](https://arxiv.org/pdf/2208.04448); [ACM DL](https://dl.acm.org/doi/full/10.1145/3641817)
- A follow-up GPU volume-rendering paper says online inference over NeuralVDB is too slow for real-time, and recommends decoding to a regular VDB first. — [GPU Volume Rendering with Hierarchical Compression Using VDB](https://www.researchgate.net/publication/390569557_GPU_Volume_Rendering_with_Hierarchical_Compression_Using_VDB)
- fVDB (Williams et al. 2024, arXiv 2407.01781) is a PyTorch framework built on NanoVDB with a single novel "VDB index grid", GPU sparse-grid construction, and tensor-core sparse convolution. It claims to match or exceed other sparse frameworks. A 2026 TOG paper (Sifakis et al., dynamic grids) credits fVDB's IndexGrid with topology/data separation. — [NVIDIA fVDB page](https://research.nvidia.com/labs/toronto-ai/fVDB/); [arXiv 2407.01781](https://arxiv.org/html/2407.01781v2); [Sifakis et al. 2026 dynamic grids](https://research.nvidia.com/labs/prl/sifakis2026dynamic/dynamic_grids.pdf)
- DCGrid (TU Delft) is an adaptive GPU grid for smoke that reports about 1/3 of SPGrid's memory in a comparable setup. The paper itself could not be fetched. — [DCGrid PDF](https://graphics.tudelft.nl/~klaus/papers/DCGrid.pdf) (search snippet only)

### Inferences
- For a 64^3 localized effect, sparsity buys little on the solver: the whole domain is about 0.26M cells. Sparsity matters for the upres or render volume (for example, a 256^3 detail field) and for multiple effects sharing a pool. Flow-style brick pools (fixed-size bricks plus a page table) fit better than VDB trees for a per-frame solver.
- fVDB's IndexGrid and sparse convolution are the natural toolkit if the learned-detail network runs only on active bricks.

### Gaps
- The NanoVDB paper's own timing numbers, the SPGrid (Setaluri et al. 2014) speedups, and tall-cell grid costs were not retrieved. The DCGrid and Sifakis 2026 PDFs failed to load (connection refused, and over the size limit).
- No evidence was found of NanoVDB or fVDB used inside a shipped game's runtime simulation.

## 7. Production rendering tricks and frame budgets

### Takeaway
The documented numbers found are:
- UE volumetric fog: about 1 ms on PS4 and 3 ms on a GTX 970, from a secondary source.
- Hobbyist froxel passes: 0.05–0.5 ms.
- UE HV playback: about 6 ms, an anecdotal report.
- Neural six-way pipeline: about 4 ms of rendering.

Studio talks with authoritative budgets were not retrieved. A 6 ms budget for one effect is generous compared with typical whole-frame volumetric allocations (around 1–3 ms), so the system should be positioned as a hero-effect tier or should scale down.

### Cited Findings
- UE Volumetric Fog uses froxels. Its cost is dominated by volume texture resolution, and volumetric-shadowed lights cost about 3x more. — [Epic: Volumetric Fog](https://dev.epicgames.com/documentation/en-us/unreal-engine/volumetric-fog-in-unreal-engine)
- Epic's UE4-era volumetric fog figures, about 1 ms on PS4 (High) and 3 ms on a GTX 970 (Epic), come from a secondary source. — [iRender](https://irendering.net/exploring-volumetric-fog-in-unreal-engine/)
- UE HV exposes `r.HeterogeneousVolumes.DownsampleFactor` (reduced-resolution march), a 512-step cap, a 512 shadow-map resolution, and lighting-cache downsampling. These are Epic's scalability levers for volume marching. — [Epic: Heterogeneous Volumes](https://dev.epicgames.com/documentation/unreal-engine/heterogeneous-volumes-in-unreal-engine?lang=en-US)
- ZibraVDB renders illumination at 1/4 resolution by default. — [Zibra blog](https://www.zibra.ai/blog-posts/zibravdb-performance-and-quality-optimization)
- Neural six-way pipeline breakdown: coarse march to the guiding map about 2 ms, inference 1.2–3 ms, compose under 0.3 ms, screen-space shadow 0.8 ms. — [arXiv 2604.03748](https://arxiv.org/html/2604.03748)

### Inferences
- The levers proven in production are half- or quarter-resolution marching, a lighting cache or shadow volume at reduced resolution (HV, ZibraVDB's 1/4), step caps, and temporal accumulation. A 1440p, 6 ms design should march at half resolution, compute lighting into a coarse 3D cache once per frame, and upsample, consistent with HV and Zibra practice.

### Gaps
- Not retrieved and to be filled by another pass: Hillaire (Frostbite, SIGGRAPH 2015/2016), Wronski (AC4 froxels, 2014), Bauer (CoD, SIGGRAPH 2019 "volumetric lighting"), Remedy (Alan Wake 2 talks), and blue-noise and temporal-reprojection ms figures. No shipped title's total frame-time budget for volumetric effects was verified in this pass.
