# Niagara Fluids 3D Gas: Measured Real-Time Performance and Real-World Usage

Reference point being compared (from assignment, not researched): our system is about 2 ms GPU per effect on an RTX 5080 at 1440p (64^3 sim about 0.5-1.0 ms per step, detail stage about 0.8 ms, volume ray-march render about 0.85-1.1 ms).

Research date: 2026-11-10. About 15 tool calls. The main result: **Epic has not published any GPU timings for Niagara Fluids 3D gas.** The only numbers available are a few community reports and one vendor blog table that gives no methodology.

## Published GPU timings for Niagara Fluids 3D gas

### Takeaway
I found no Epic-published ms or fps numbers for 3D gas at any stated resolution or GPU. Community figures are scarce. The one first-hand number is about 6 ms of render cost on an RTX 4080 for a cached 3D gas effect at resolution 200, with the fps falling from about 110 to about 50. A 2026 game-studio blog lists 64^3 smoke at 1.8 ms and 128^3 smoke at 5.5 ms on PS5/XSX, but it gives no methodology and could not be verified.

### Cited Findings
- **Community, RTX 4080, UE version not stated.** A developer played back a *cached* Niagara Fluids 3D gas smoke effect in an empty scene at grid resolution 200 (the template default), with about 200 frames of animation. The fps fell from about 110 to about 50. `profilegpu` showed about 6 ms under "HeterogeneousVolumes", which is the render pass, not the sim. The poster said lowering resolution noticeably hurt quality. — [Epic forum: Low performance with cached gas simulations](https://forums.unrealengine.com/t/low-performance-with-cached-gas-simulations-niagara-fluids-3d-gas-sim/2177751)
  - A reply in the same thread (community user "Arkiras", not Epic staff) said raymarching the volume onto the screen is "extremely expensive on its own", separately from the sim. They called 3D gas better suited to tightly controlled cinematics and "more of a future-looking feature". They suggested raising `r.HeterogeneousVolumes.CLOD.Bias` and lowering `r.HeterogeneousVolumes.MaxStepCount`. No Epic staff replied. — [same thread](https://forums.unrealengine.com/t/low-performance-with-cached-gas-simulations-niagara-fluids-3d-gas-sim/2177751)
- **Vendor blog (StraySpark, 2026), PS5/Xbox Series X at 60 Hz, "measured on UE5.7.1".** The table gives:
  - 3D smoke 64^3: 1.8 ms
  - 3D smoke 128^3: 5.5 ms
  - 3D fire 64^3: 2.4 ms
  - 2D smoke 256^2: 0.4 ms
  - 2D water 512^2: 0.6 ms
  - SPH water 10k particles: 0.8 ms; 50k particles: 3.2 ms

  The post does not say whether these figures cover sim only or sim plus render. It also gives no scene content, channels, capture tool or run count. It recommends a "gameplay fluid budget" of 3-4 ms at 60 Hz (6-8 ms at 30 Hz), and says Steam Deck lacks the budget at 60 Hz. — [StraySpark: Niagara Fluids for Gameplay: Beyond VFX in 2026](https://www.strayspark.studio/blog/niagara-fluids-for-gameplay-beyond-vfx-2026)
- **Community, mid-range PC.** A search-result summary of a developer report said a *2D* gas effect added 4-6 ms of frame time. I could not trace this to a primary post, so treat it as unverified. — summarised in search results alongside [StraySpark](https://www.strayspark.studio/blog/niagara-fluids-for-gameplay-beyond-vfx-2026)
- **Editor slowness (UE 5.5, GPU not stated).** A user said 3D gas sims are "unbearably slow in editor" in 5.5 and the GPU "constantly maxes out". A reply said Niagara Fluids "runs very smoothly in UE5.3, but becomes nearly unusable in 5.5" with the same setup. No numbers were given. — [Epic forum: UE5.5 Niagara Gas Simulation are unbearably slow in editor](https://forums.unrealengine.com/t/ue5-5-niagara-gas-simulation-are-unbearably-slow-in-editor/2439193)
- **Speed claim (UE 5.8, creator Jesse Pitela).** An 80.lv article says Niagara Fluids is "up to 30% faster" in UE 5.8. The article promotes the creator's course, and I found no Epic benchmark behind the figure. — search-result summary of [80.lv: Exploring real-time water simulations in UE 5.8](https://80.lv/articles/exploring-real-time-water-simulations-in-unreal-engine-5-8/)

### Inferences
- The 4080 report suggests that for 3D gas the *render* (Heterogeneous Volumes ray-march) can cost more than the sim. About 6 ms for render alone is several times our 0.85-1.1 ms ray-march. However, its resolution (200, compared with our 64^3), screen resolution and UE version all differ or are unknown, so it is not a like-for-like comparison.
- If the StraySpark 64^3 number (1.8 ms on PS5) is real, it is broadly the same order as our approximately 2 ms total on an RTX 5080. The RTX 5080 is a much faster GPU than a PS5, so on equal hardware the StraySpark figure would favour Niagara. The source's reliability is low, though.

### Gaps
- No Epic-published timings, and no `stat GPU` / `ProfileGPU` breakdown from Epic talks or docs, for 3D gas at stated resolutions and GPUs.
- No community benchmark separates the sim cost from the render cost at a stated resolution on a stated RTX GPU.
- I found no YouTube breakdown with readable profiler numbers. Video content could not be searched effectively.
- The Epic bug tracker entry UE-213356 came up in results for 3D gas performance, but it returned HTTP 403, so I do not know its contents.

## What Epic says about real-time suitability

### Takeaway
Epic's own docs steer 3D gas toward "hero effects or cinematics". They call 2D sims the option for games and real time, and recommend baking 3D results to flipbooks for real-time use. The renderer Epic pairs with cached 3D gas (Heterogeneous Volumes plus Sparse Volume Textures) is labelled Experimental. Epic gives no numeric resolution or console guidance.

### Cited Findings
- From the Niagara Fluids Reference (UE 5.9 docs):
  - "2D simulations are more efficient and better suited for games and real-time use."
  - "3D simulations have a more realistic look, but a tradeoff in higher memory and GPU cost."
  - "3D simulations are best for hero effects or cinematics."
  - Results "can also be baked into a flipbook and applied to textures for increased performance in real-time use."
  - Higher resolution, smaller cells and more pressure iterations all reduce performance.
  - The page gives no recommended resolutions, console guidance or LOD discussion.

  — [Niagara Fluids Reference in Unreal Engine](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-reference-in-unreal-engine)
- The Fluid Simulation overview says 2D/grid and FLIP-style sims are much faster than 3D simulations and best suited for games and real time (search-result summary). — [Fluid Simulation in Unreal Engine - Overview](https://dev.epicgames.com/documentation/en-us/unreal-engine/fluid-simulation-in-unreal-engine---overview)
- From the Heterogeneous Volumes docs (UE 5.9):
  - The page carries the warning "Experimental feature, but use caution when shipping with it."
  - "The resolution of the imported SVT has a direct correlation to runtime performance."
  - Indirect lighting is "best used with high-end PCs and Cinematic scalability settings."
  - The streaming-block option is "not recommended for real-time playback".
  - The documented Niagara-to-Heterogeneous-Volume workflow (3D Gas template, loop duration under 5 s) is aimed at the Path Tracer.

  — [Heterogeneous Volumes in Unreal Engine](https://dev.epicgames.com/documentation/en-us/unreal-engine/heterogeneous-volumes-in-unreal-engine)
- Epic's Unreal Fest 2023 talk "Creating Visual Effects with Niagara Fluids, Sparse Volume Textures and Heterogeneous Volumes" presents caching sims into Sparse Volume Textures as the route to real-time playback, plus adapting the workflow to cinematics. The listing gives no performance numbers. — [Epic forum: Unreal Fest 2023 talk listing](https://forums.unrealengine.com/t/talks-and-demos-creating-visual-effects-with-niagara-fluids-sparse-volume-textures-and-heterogeneous-volumes-in-ue-unreal-fest-2023/1683078); video: [YouTube](https://www.youtube.com/watch?v=Z3n4vSrwo44)
- Unreal Fest 2023 "Let it Flow: Film Quality Fluids in UE" covers a Houdini-to-UE workflow for "High End Realtime" and "Rendered Cinematic" use, with no numbers. — [Epic forum listing](https://forums.unrealengine.com/t/talks-and-demos-let-it-flow-film-quality-fluids-in-ue/1312494)
- An AI-generated transcript note of an Epic Niagara Fluids talk says the speaker showed full 3D real-time *liquid* simulation in a game environment in the Electric Dreams GDC demo. This is secondary and unverified. — [lilys.ai notes](https://lilys.ai/notes/967872)
- **Experimental/Beta status is disputed.**
  - A forum user in September 2024 wrote "Be careful as it is Beta right now." — [Epic forum: Exploring the 3D Gas Emitter, p.2](https://forums.unrealengine.com/t/tutorial-niagara-fluids-exploring-the-3d-gas-emitter/888748?page=2)
  - StraySpark says the plugin left Experimental in 5.5 and became production in 5.6. Another StraySpark listing says production since 5.5, so the studio contradicts itself. Neither claim is confirmed against Epic release notes. — [StraySpark](https://www.strayspark.studio/blog/niagara-fluids-for-gameplay-beyond-vfx-2026)

### Inferences
- Epic's stated position matches our competitive angle. Epic itself treats *live* 3D gas in gameplay as a hero or cinematic tool and points to baking (flipbooks or SVT) for game use. A live 3D smoke system at about 2 ms per effect would be aiming at something Epic does not claim to offer.

### Gaps
- I found no Epic numeric guidance on resolution budgets or consoles (PS5/XSX/Series S/Switch 2) for 3D gas.
- Plugin maturity status per UE version was not verified against official release notes.

## Shipped games and demos: live versus baked

### Takeaway
I found no confirmed shipped game that runs Niagara Fluids 3D gas live at runtime, and no evidence that Fortnite, LEGO Fortnite or The Matrix Awakens use it live. Known uses are demos and portfolio or fan pieces, plus the baked workflows Epic itself promotes (flipbooks and SVT caches).

### Cited Findings
- A forum responder asked the original poster which game they had seen use 3D gas sims, implying none are known. — [Epic forum thread](https://forums.unrealengine.com/t/low-performance-with-cached-gas-simulations-niagara-fluids-3d-gas-sim/2177751)
- StraySpark: "not aware of a shipped game using this technique as of early 2026" (meaning gameplay-driven Niagara Fluids). — [StraySpark](https://www.strayspark.studio/blog/niagara-fluids-for-gameplay-beyond-vfx-2026)
- Epic's own Niagara Fluids showcase video (80.lv coverage, 2 June 2022) shows smoke, fire, explosions and water. It gives no real-time or cinematic labelling, hardware or numbers. — [80.lv: Epic Games Demonstrated Niagara Fluids' Abilities](https://80.lv/articles/epic-games-demonstrated-niagara-fluids-abilities)
- Redefine FX's fan-made Tesla Roadster game demo used Niagara Fluids for smoke, and Tim Sweeney reposted it. It is a demo, not a shipped game, and no perf data is given. — search-result summary of [Redefine FX: Tesla Roadster Game Demo](https://redefinefx.com/tesla-roadster-game/)
- Epic technical artist Asher Zhu's 80.lv interview covers Niagara Fluids *water* workflows and mentions no shipped game. — [80.lv: Working with Niagara Fluids to Create Water Simulations](https://80.lv/articles/working-with-niagara-fluids-to-create-water-simulations)
- A Gnomon workshop by a Gearbox lead VFX artist uses Niagara particles with Houdini Pyro flipbooks, which is the traditional baked approach rather than Niagara Fluids. — search-result summary via [Creative Bloq](https://www.creativebloq.com/3d/video-game-design/dont-miss-this-unreal-engine-5-vfx-course-while-its-free)

### Inferences
- Live 3D grid smoke in shipped UE games appears to be essentially absent. The industry norm remains flipbooks or baked volumes, so live 3D smoke is an unoccupied niche.

### Gaps
- I could not confirm or rule out Niagara Fluids use (live or baked) in Fortnite, LEGO Fortnite, The Matrix Awakens or specific 2023-2026 UE5 titles. No sources were found either way.
- The Electric Dreams real-time liquid claim is unverified against a primary Epic source.

## Known limitations and developer complaints

### Takeaway
The complaints centre on render cost (the Heterogeneous Volumes ray-march), quality collapsing below the default resolution, Experimental or Beta status, regressions between versions (5.3 to 5.5 editor slowdown), crashes at high resolution, and rendering-path incompatibilities such as forward rendering.

### Cited Findings
- **Render cost and quality:**
  - The ray-march render is "extremely expensive on its own".
  - Lowering resolution below 200 loses too much quality, which led the poster to question whether 3D gas is viable in games and to consider falling back to traditional particles.

  — [Epic forum](https://forums.unrealengine.com/t/low-performance-with-cached-gas-simulations-niagara-fluids-3d-gas-sim/2177751)
- **Version regression:** "runs very smoothly in UE5.3, but becomes nearly unusable in 5.5". The workaround is to recompile the system when lag starts. — [Epic forum](https://forums.unrealengine.com/t/ue5-5-niagara-gas-simulation-are-unbearably-slow-in-editor/2439193)
- **Crashes at high resolution:** there is a thread titled "how to fix crashing with Niagara Grid3D_Gas_Master_Emitter resolution too high" (December 2024). An Epic-side responder (FrostyJas) asked for specs and logs. The crash details are not visible. — [Epic forum](https://forums.unrealengine.com/t/how-to-fix-crashing-with-niagara-grid3d-gas-master-emitter-resolution-too-high/2087476/3)
- **Stability and usability issues (UE 5.4-5.5):**
  - The 3D gas emitter sometimes doesn't appear.
  - In 5.5, fire needs Render Temperature set to Black Body, which is not the default.
  - The preview is empty when the project uses forward rendering.
  - Meshes behind the emitter get white highlights (MetaHuman hair).

  — [Epic forum: Exploring the 3D Gas Emitter, p.2](https://forums.unrealengine.com/t/tutorial-niagara-fluids-exploring-the-3d-gas-emitter/888748?page=2)
- **Other threads:** titles found include "Niagara fluids doesnt (has never) work" and "Niagara Fluids 3D Gas does not show up (GPU warning)". I did not read their contents. — [forum 1](https://forums.unrealengine.com/t/niagara-fluids-doesnt-has-never-work/2681724), [forum 2](https://forums.unrealengine.com/t/niagara-fluids-3d-gas-does-not-show-up-gpu-warning/1231010)
- **Renderer:** Heterogeneous Volumes is Experimental. Environment shadow casting and path-traced volumes are also Experimental. — [Epic HV docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/heterogeneous-volumes-in-unreal-engine)
- **Bounded domain:** the reference describes a fixed world grid with set cell size and resolution. Smaller cells mean higher quality but a performance cost. — [Niagara Fluids Reference](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-reference-in-unreal-engine)
- **Cross-hardware determinism:** not guaranteed, according to a secondary source. The SPH solver is called "less mature". — [StraySpark](https://www.strayspark.studio/blog/niagara-fluids-for-gameplay-beyond-vfx-2026)
- **LOD:** the HV renderer has a CLOD bias cvar (`r.HeterogeneousVolumes.CLOD.Bias`) mentioned by the forum user. Beyond scalability overrides, I found no documented sim-level LOD for Niagara Fluids. — [Epic forum](https://forums.unrealengine.com/t/low-performance-with-cached-gas-simulations-niagara-fluids-3d-gas-sim/2177751); [Niagara Fluids Reference](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-reference-in-unreal-engine)

### Inferences
- Running several 3D gas sims at once appears untested and undocumented in public sources. Given single-effect costs of several ms, multiple live sims are probably impractical on mid-range hardware. This is an inference, not a measurement.

### Gaps
- I found no published data on the cost of multiple simultaneous 3D gas sims.
- I found no published per-light lighting or shadow cost numbers. The reference names Shadow Quality and Shadow Max Steps but gives no costs.

## Memory footprint per simulation

### Takeaway
I found no published per-simulation VRAM figure for Niagara Fluids 3D gas at any resolution. The only related numbers are the Heterogeneous Volumes renderer memory caps and an academic out-of-VRAM report for SVT rendering.

### Cited Findings
- Heterogeneous Volumes renderer memory:
  - `r.HeterogeneousVolumes.OrthoGrid.MaxBottomLevelMemoryInMegabytes` defaults to 128 MB (512 MB at cinematic). The FrustumGrid cap is the same.
  - The dense lighting cache will not allocate beyond 1024x1024x512 voxels.
  - Lowering the shading rate can exceed the memory limit, so parts of the volume go missing.

  — [Epic HV docs, UE 5.9](https://dev.epicgames.com/documentation/en-us/unreal-engine/heterogeneous-volumes-in-unreal-engine)
- An academic survey of large-volume rendering in UE5 reported out-of-VRAM crashes on a 12 GB RTX 3500 Ada when showing more than five chunks of about 384 max-dimension each. This concerns SVT/volume rendering of datasets, not Niagara sims. — [arXiv 2504.07485](https://arxiv.org/pdf/2504.07485)
- Epic says only that 3D has "higher memory and GPU cost" than 2D. — [Niagara Fluids Reference](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-reference-in-unreal-engine)
- StraySpark says cooking adds "a few hundred MB" for projects with many fluid systems, with no measurement details. — [StraySpark](https://www.strayspark.studio/blog/niagara-fluids-for-gameplay-beyond-vfx-2026)

### Inferences
- A lower bound can be derived from first principles rather than from a source. A 200^3 grid has 8M cells, so each float32 scalar channel takes about 32 MB, before velocity, pressure, temperature and double-buffering. This is arithmetic only, not a published figure.

### Gaps
- No measured VRAM per sim (for example at 64^3, 128^3 or 200^3) from Epic or the community.

## Context: NVIDIA Flow and JangaFX EmberGen

### Takeaway
NVIDIA Flow is a real-time sparse-grid GPU smoke and fire library with a UE4 integration (around 2017). I found no published ms numbers or confirmed shipped games in this pass. EmberGen is a standalone authoring tool. It simulates in real time at modest resolutions, but its output is baked (flipbooks or VDB) and it does not run inside shipped games.

### Cited Findings
- **NVIDIA Flow:**
  - It is a "sparse grid-based fluid simulation library for real-time applications", GPU-based and mainly for fire and smoke.
  - It uses D3D11.3 Volume Tiled Resources when available, and supports DX11/DX12.

  — [GitHub NVIDIAGameWorks/Flow](https://github.com/NVIDIAGameWorks/Flow); [NVIDIA FLOW UE4 docs](https://docs.nvidia.com/gameworks/content/artisttools/Flow/FLOWUe4_Intro.html), [Grid](https://docs.nvidia.com/gameworks/content/artisttools/Flow/FLOWUe4_Grid.html)
  - NVIDIA showed a DX12 demo in April 2017. — [ComputerBase](https://www.computerbase.de/2017-04/nvidia-gameworks-effekt-flow-directx-12-video/); [TechPowerUp](https://www.techpowerup.com/232664/nvidia-demonstrates-gameworks-flow-tech-under-directx-12?amp)
  - A UE4 forum user said the integration requires a static bounding box. — search-result summary of [UE forum GameWorks Integration thread](https://forums.unrealengine.com/t/nvidia-gameworks-integration/16686?page=183)
- **EmberGen:**
  - 2019 beta notes: voxel resolutions up to 512^3 "run fast on a 2080 Ti" but drop to 5-6 fps on a GTX 1080.
  - A 2021 developer post: on an RTX 3090, a high-res scene took 13.5 ms for simulation but 150-200 ms to render, depending on camera angle. At lower resolutions it runs at 30+ fps.

  These are search-result summaries of developer posts. — [realtimevfx.com EmberGen alpha thread](https://realtimevfx.com/t/embergen-alpha-real-time-volumetric-fluid-simulations-for-vfx-artists/9915); [r/Simulated mirror, Apr 2021](https://lr.us.psf.lt/r/Simulated/comments/ml1d9b/not_only_are_trees_burning_but_so_is_the_gpu)
  - Early coverage says EmberGen is not meant for creating 3D volumetric sims inside a game, and that it outputs flipbooks and sprite sheets. — [80.lv](https://80.lv/articles/embergen-upcoming-real-time-fluid-simulation-tool); [GameFromScratch](https://gamefromscratch.com/embergen-real-time-fluid-simulation-2/)

### Inferences
- Neither comparator is a documented, shipped, live in-game 3D smoke solution with public per-frame costs. The EmberGen 3090 data point repeats the Niagara pattern: rendering, not simulation, dominates the cost of volumetric smoke.

### Gaps
- I found no published NVIDIA Flow ms-per-frame numbers at stated grid sizes or GPUs. I also found no confirmed list of shipped games using Flow, and its status in Omniverse / Flow 2 was not researched.
- I found no 2025-2026 EmberGen benchmarks.
