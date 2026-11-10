# Niagara Fluids 3D Gas: Rendering, Lighting, Integration and Artist Workflow

Scope note: research done November 2026 with about 17 search and fetch calls. The current Epic docs are labelled UE 5.9, but many of their internal links still point to 5.5, so a page labelled 5.9 may contain older text. Labels used below: **[Epic]** means Epic documentation, release notes, or a stated Epic employee. **[Community]** means forum users, third-party blogs or academic users.

Our system, for comparison: a half-resolution ray-marcher, a shadow volume from one light, a blackbody lookup, a 1 m bounded box, sphere and box obstacles, mouse-driven emitters, and no integration, artist tools, LOD or caching.

## Rendering: how the gas volume is drawn

### Takeaway
Niagara 3D gas has two render paths:
- **Template path:** the template's own material-based ray-march with baked lighting, configured in its "Render" section.
- **Heterogeneous Volumes (HV) path:** Epic's newer volume renderer, which ray-marches through a world-space voxel grid and a frustum-aligned voxel grid. It has a viewport downsample factor, a step-count cap, a dense lighting cache and translucency compositing. Its sorting is weak.

HV is still marked **Experimental** in the current (5.9) docs. The 5.5 release notes called it Beta, which is a conflict.

### Cited Findings
- **[Epic]** The Grid 3D Gas Fire template has a "Render" section with these controls:
  - "Render Density": None, Linear or Curve.
  - "Render Temperature": None, Black Body or Curve, with range and gain.
  - "Shadow Quality" and "Shadow Max Steps", which control baked shadow sampling.
  - "Render Step Size Mult", which sets sampling density per cell.
  - "Render Debug Slice": density in red, temperature in green, light in blue.

  — [Niagara Fluids Reference (Grid 3D Gas)](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-reference-in-unreal-engine)
- **[Epic]** The Cine template variant has a "Texture" section that modulates the render with noise: Texture Scale, Value Band Min/Max, Loop Duration and Debug Texture. — [Niagara Fluids Reference](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-reference-in-unreal-engine)
- **[Epic]** The Niagara Fluids landing page offers 2D templates "for games" and 3D templates "for cinematics". It warns that fluid scenes are graphically intensive and can cause a GPU crash on Windows. — [Niagara Fluids (UE 5.9 docs)](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-in-unreal-engine)
- **[Epic]** UE 5.4 release notes: "Heterogeneous Volume Rendering provides for integrated rendering of volumetric phenomena such as fire, smoke, and fluids driven by Niagara Fluids or Open VDB files." HV was **Experimental** in 5.4. — [UE 5.4 Release Notes](https://dev.epicgames.com/documentation/unreal-engine/unreal-engine-5.4-release-notes?application_version=5.4)
- **[Epic]** UE 5.5 release notes list "Heterogeneous Volumes (Beta)". New items: importing VDB into Sparse Volume Textures for rendering, the "Ability to render multiple overlapping HVs", and self-shadowing when indirect lighting is included. — [UE 5.5 Release Notes](https://dev.epicgames.com/documentation/en-us/unreal-engine/unreal-engine-5-5-release-notes)
- **[Epic]** The current HV doc still says "Experimental", which conflicts with the 5.5 "Beta" label above. The doc describes a deferred ray-march path and these settings:
  - Step Factor, Shadow Step Factor and Shadow Bias Factor.
  - Lighting Downsample Factor, which scales an internal lighting cache. The cache is dense, capped at 1024 x 1024 x 512 voxels.
  - `r.HeterogeneousVolumes.DownsampleFactor` (default 1.0), which downsamples the rendered viewport.
  - `r.HeterogeneousVolumes.MaxStepCount` (default 512).
  - Voxel grids: OrthoGrid, a world-space grid (default on), and FrustumGrid, a frustum-aligned grid. Each has MaxBottomLevelMemoryInMegabytes (default 128, 512 recommended for cinematic) and ShadingRate (default 4, 1 recommended for cinematic). FrustumGrid.DepthSliceCount defaults to 512.
  - Known issues: shading rate 4 looks "blocky or blurry"; voxels go missing if the memory cap is exceeded; overlapping volumes turn blocky because homogeneous regions are aggregated (`...BottomLevelGrid.HomogeneousAggregation 0` disables this); a screen-door effect in the path tracer.

  — [Heterogeneous Volumes (UE 5.9 docs)](https://dev.epicgames.com/documentation/unreal-engine/heterogeneous-volumes-in-unreal-engine?lang=en-US)
- **[Epic]** Project setting "Composite with Translucency" enables compositing of heterogeneous volumes with translucency. The doc gives no details on sorting. — [Heterogeneous Volumes docs](https://dev.epicgames.com/documentation/unreal-engine/heterogeneous-volumes-in-unreal-engine?lang=en-US)
- **[Community]** Forum users on UE 5.3 to 5.4 report several sorting and compositing problems:
  - Separately simulated volumes do not blend. One user said they behave "more like a shaded VDB projected on to a plane", so only one volume shows in front.
  - Depth sorting "is still unsolved" (5.4 Preview 1).
  - Translucent materials need "Before DOF" or "After DOF" changes to composite correctly.
  - Niagara Fluids smoke was not obscured by dust particles (5.2).
  - Volumes rendered only in the first split-screen viewport (5.3).
  - Exponential height fog blending works in 5.4 but did not in 5.3.
  - An unnamed poster said "many rendering issues are fixed in 5.4.2".

  — [Forum: 5.3 Heterogeneous Volume Rendering tips and tricks, p.3](https://forums.unrealengine.com/t/5-3-heterogeneous-volume-rendering-tips-and-tricks/1291002?page=3)
- **[Community]** `r.HeterogeneousVolumes.Jitter 0` was suggested to fix temporal jitter in volumes, and popping remained when orbiting the camera. — [Forum thread p.3](https://forums.unrealengine.com/t/5-3-heterogeneous-volume-rendering-tips-and-tricks/1291002?page=3)
- **[Community]** An academic survey (GEOMAR, UE 5.3 and 5.4) compared four ways to render Sparse Volume Textures (SVTs): the SVT Viewer, Volumetric Fog ("blurry and ghostly"), Volumetric Cloud (noisy), and the Heterogeneous Volume actor. It judged the HV actor the most suitable. — [Schlüter et al. 2025, arXiv 2504.07485](https://arxiv.org/html/2504.07485v1)
- **[Epic]** The SVT doc says volumetric fog can render SVTs on meshes with volume materials, but the detail can exceed the resolution of the fog grid. — [Sparse Volume Textures docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/sparse-volume-textures-in-unreal-engine)

### Inferences
- Epic's real-time path resembles our half-resolution ray-marcher, but it adds several things we lack:
  - A world-space or frustum-space voxel grid of adjustable memory and shading rate.
  - A viewport downsample cvar.
  - A step cap.
  - A separate dense lighting cache.
  - A translucency composite option.
- The `Jitter` cvar implies Epic jitters ray steps and relies on temporal accumulation through TAA/TSR.

### Gaps
- No Epic statement says explicitly how HV interacts with TSR, DLSS or temporal reprojection; only the jitter cvar hints at it.
- The exact shader approach of the template ray-march material, for example whether it is a translucent box-mesh material, is not documented on the pages fetched.
- No official explanation of how volumes sort against translucency was found.

## Lighting: lights, shadows, GI, emission and path tracing

### Takeaway
- **Template path:** samples up to two lights connected as data interfaces, with baked shadow marching.
- **HV path:** has a lighting cache, an optional shadow-map resolution, experimental shadow casting onto the scene, off-by-default indirect lighting (high-end only), and path tracer support.
- **Not supported:** MegaLights does not support HV. No Epic source describes fire lighting the scene or a Lumen GI link.

### Cited Findings
- **[Epic]** Light handling in the 3D gas template:
  - Light1 and Light2 data interfaces read lights connected through Override Parameters.
  - With no lights connected, defaults are used: "Lgt1/2 Default" intensity, color and direction.

  — [Niagara Fluids Reference](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-reference-in-unreal-engine)
- **[Epic, via interview]** Asher Zhu (Senior Technical Artist, Epic, 2022) listed "Lighting injection Interfaces for gas" as a feature. — [80.lv interview, July 2022](https://80.lv/articles/working-with-niagara-fluids-to-create-water-simulations)
- **[Epic]** HV lighting settings and status:
  - Project setting "Shadow Casting" lets volumes cast shadows onto the environment. It is experimental.
  - `r.HeterogeneousVolumes.Shadows.Resolution` defaults to 512.
  - `r.HeterogeneousVolumes.IndirectLighting` defaults to 0. The doc says it is "best used with high-end PCs and Cinematic scalability settings."

  — [Heterogeneous Volumes docs](https://dev.epicgames.com/documentation/unreal-engine/heterogeneous-volumes-in-unreal-engine?lang=en-US)
- **[Epic]** UE 5.4 work on HV focused on shadow casting. Release notes said "Much of the remaining work for the deferred renderer includes adding support for global illumination." — [UE 5.4 Release Notes](https://dev.epicgames.com/documentation/unreal-engine/unreal-engine-5.4-release-notes?application_version=5.4)
- **[Epic]** Path tracer support for HV:
  - Requires DX12, hardware ray tracing and the Path Tracing project setting, plus `r.PathTracing.HeterogeneousVolumes 1`.
  - For a Niagara 3D gas template, set Loop Behavior to Infinite with Loop Duration under 5 s.
  - Niagara systems trigger path-tracing invalidation only while the Niagara Editor is open.

  — [Heterogeneous Volumes docs](https://dev.epicgames.com/documentation/unreal-engine/heterogeneous-volumes-in-unreal-engine?lang=en-US)
- **[Epic]** The SVT doc says the path tracer "offers more complete support for rendering volumes", including scattering, shadows and global illumination. It says the real-time path is "partly limited by deferred rendering features." — [SVT docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/sparse-volume-textures-in-unreal-engine)
- **[Epic]** MegaLights docs list "Water, Clouds, Heterogenous Volumes and Local Volumetrics" as not supported. MegaLights was Experimental in 5.5. — [MegaLights docs](https://dev.epicgames.com/documentation/unreal-engine/megalights-in-unreal-engine); [UE 5.5 Release Notes](https://dev.epicgames.com/documentation/en-us/unreal-engine/unreal-engine-5-5-release-notes)
- **[Community]** Shadow behaviour reported on UE 5.4 to late 2024:
  - Volumes cast shadows on the ground from a point light and onto themselves, but not onto each other (5.4 Preview 1).
  - Shadow casting was called unstable, and quality degraded with distance from the world origin (Nov 2024).
  - Raising Shadows.Resolution to 2048 helps, at a performance cost.
  - Coloured extinction tints the volume's own lighting but not light passing through it.

  — [Forum thread p.3](https://forums.unrealengine.com/t/5-3-heterogeneous-volume-rendering-tips-and-tricks/1291002?page=3)
- **[Community]** Gas sims rendering fully or partly black were fixed by setting `r.HeterogeneousVolumes.LightingCache 0`, at a performance cost. — [RealTimeVFX thread](https://realtimevfx.com/t/niagara-gas-sim-appears-partially-completely-black/30909)
- **[Community]** In a modified UE 5.4 build, HV lighting comes from a precomputed lighting cache, so the number and type of lights do not change cost. The authors disabled self-shadow casting because it flickered. — [Schlüter et al. 2025](https://arxiv.org/html/2504.07485v1)
- **[Community]** A user suggested recolouring cached smoke by changing light intensity and colour, using matching lighting channels. — [Arctiem blog](https://www.arctiem.com/post/adjusting-density-and-color-in-cached-niagara-3d-fluid-gas-simulations)

### Inferences
- In real time, both Niagara and our system are mostly limited to a few directional or point lights with a shadow march.
- Niagara adds these over our single shadow volume:
  - Experimental shadows cast onto the scene.
  - A second light.
  - Optional indirect lighting.
  - Path tracer parity for cinematics.

### Gaps
- No Epic source found that fire emission from a Niagara gas sim lights the scene through Lumen or emissive GI. Typical practice of adding a separate point light to the effect is not documented in what was fetched.
- No documented Lumen interaction with HV, in either direction.
- No published Epic cost numbers in milliseconds for HV lighting.

## Scene interaction: collision, bounds, forces, particles and gameplay hooks

### Takeaway
- **Collision:** artists opt in specific actors. Epic's interview names per-mesh distance fields, the global distance field, physics assets for characters, and Landscape and Skeletal Mesh interfaces. Geometry collection and geometry cache support was work in progress.
- **Sources:** particles from other emitters can splat density, temperature and velocity into the grid.
- **Bounds:** a world-space box with a local pivot and per-face open boundaries.

### Cited Findings
- **[Epic]** 3D gas template controls:
  - "Collide Against" adds data interfaces for actor collision; you pick actors in Override Parameters.
  - Grid: "World Space Size", "Resolution Max Axis", "Open Boundary +/- X/Y/Z" and "Local Pivot".
  - Turbulence: "Seed Turbulence" on the first frame only, and Turbulence 1 and 2 at runtime, with gain, frequency, speed, bias, local space and density/temperature bands.
  - Physics: Buoyancy (density and temperature), Gravity, Vorticity Confinement, Dissipation and Pressure Solve Iterations.

  — [Niagara Fluids Reference](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-reference-in-unreal-engine)
- **[Epic]** Particles can feed the grid:
  - "Particle Attribute Reader" reads another emitter.
  - "Scatter Density", "Scatter Temperature" and "Scatter Velocity" read particle attributes, using "Use Radius" and "Splat Size".
  - "Local Space Particles" sets the particle space.
  - The core emitter is Grid3D_Gas_Emitter, and child emitters inherit from it.

  — [Niagara Fluids Reference](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-reference-in-unreal-engine)
- **[Epic employee, 2022]** Collision interfaces named in the interview:
  - Static Mesh: uses each tagged mesh's own distance field. View-independent, gives position, normal and velocity. Cost scales with mesh count.
  - Global Distance Field: constant cost, but view-dependent so it can pop, and it has no mesh velocity.
  - Physics Asset: for characters.
  - Geometry Collection: "still WiP".
  - Geometry Cache: "experimental".
  - "There are also other collision types for Landscape and Skeletal Mesh."

  — [80.lv, Asher Zhu interview](https://80.lv/articles/working-with-niagara-fluids-to-create-water-simulations)
- **[Epic]** UE 5.4 added two-way coupling with GPU particles for 3D liquids (not gas). — [UE 5.4 Release Notes](https://dev.epicgames.com/documentation/unreal-engine/unreal-engine-5.4-release-notes?application_version=5.4)
- **[Community]** Other Niagara collision modes don't fit characters well. Distance-field collision only covers static meshes. Depth-buffer collision is screen-space and misses off-screen geometry. — [Niagara GPU distance field forum](https://forums.unrealengine.com/t/niagara-gpu-distance-field-collisions-dont-work/488103)

### Inferences
- Compared with our sphere and box obstacles, Niagara reads real scene geometry: per-mesh SDFs, the global SDF, physics-asset capsules for characters, and landscape. It also gets object velocity for two-way momentum transfer.
- Emitters can be any Niagara particle system rather than a mouse.
- Blueprint and user parameters (for example the "Set Sim Cache" node and Override Parameters) give gameplay hooks.

### Gaps
- No Epic doc confirms a moving or camera-following grid, or sparse or adaptive 3D gas grids.
- No doc confirms an explicit Wind actor or Physics Field link for gas.
- The gameplay data-interface list (reading density back to the CPU, Niagara Data Channels) was not documented on the pages fetched.

## Scalability: LOD, quality settings, platforms and multiple sims

### Takeaway
Scalability is mostly manual:
- Resolution through "Resolution Max Axis" or "Num Cells Max Axis".
- Scalability overrides that by default apply at Cinematic quality.
- HV cvars for step count, downsampling, shading rate and memory caps.

Epic positions 3D gas for cinematics, and SVTs are not supported on the mobile renderer.

### Cited Findings
- **[Epic]** 3D gas template overrides apply by default at Cinematic quality, tied to Movie Render Queue's "Game Overrides". — [Niagara Fluids Reference](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-reference-in-unreal-engine)
- **[Epic]** "2D templates for games and 3D templates for cinematics." — [Niagara Fluids docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-in-unreal-engine)
- **[Epic employee, 2022]** Advice on cost:
  - Fluids take a large share of the frame budget; profile in Standalone mode.
  - Test particle count, grid resolution and rendering at each scalability level.
  - Prefer art-directed supplementary detail over higher simulation resolution.

  — [80.lv](https://80.lv/articles/working-with-niagara-fluids-to-create-water-simulations)
- **[Epic]** SVT platform support:
  - Windows on DX11 and DX12 (SM5 and SM6).
  - VDB import works only in the Windows editor.
  - Cooked builds can run SVTs on other platforms if they meet the performance requirements.
  - "SVTs do not work with the Mobile Renderer."

  — [SVT docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/sparse-volume-textures-in-unreal-engine)
- **[Epic]** HV level-of-detail controls include Streaming Mip Bias (the SVT mip level), the Lighting Downsample Factor, shading rates and memory caps. — [HV docs](https://dev.epicgames.com/documentation/unreal-engine/heterogeneous-volumes-in-unreal-engine?lang=en-US)
- **[Community]** Suggested cvars for cost are `r.HeterogeneousVolumes.CLOD.Bias` (raise it) and `MaxStepCount` (lower it). — [Forum: Low performance with cached gas](https://forums.unrealengine.com/t/low-performance-with-cached-gas-simulations-niagara-fluids-3d-gas-sim/2177751)
- **[Community]** VRAM limits in academic testing:
  - Niagara Fluids allocates a sim grid on top of texture data.
  - On a 12 GB GPU, more than 5 chunks of about 384³ crashed from running out of VRAM. Total display was about 0.28 giga-voxels at about 20 fps.

  — [Schlüter et al. 2025](https://arxiv.org/html/2504.07485v1)
- **[Community]** Niagara Fluids not showing in packaged builds is a reported issue. — [Forum thread](https://forums.unrealengine.com/t/niagara-fluids-not-showing-in-packaged-project/1423822)

### Inferences
- `CLOD.Bias` suggests HV has a continuous level-of-detail term, but it is undocumented in the pages read.

### Gaps
- No official console support statement for 3D gas or HV was found.
- No automatic distance-based LOD or culling documentation for gas grids was found.
- No guidance on how many 3D sims can run at once was found.

## Artist workflow: templates, baking, caching and playback

### Takeaway
Artists get:
- Templates (fire, smoke and others) with grouped, no-code controls and debug slices.
- A Flipbook Baker that bakes to 2D textures usable in any material.
- A Niagara Sim Cache, recorded from the editor baker or from Sequencer, and played back with Blueprint.
- OpenVDB import into SVTs for the HV actor, with Blueprint playback controls.

No Epic one-step OpenVDB export from Niagara gas was found. Playback of a cached 3D gas sim through HV is not cheap.

### Cited Findings
- **[Epic]** Fluid templates cover fire, smoke and gas. Advanced users can modify them without code. Simulations can be baked to a flipbook with the Niagara Flipbook Baker for use on any material. — [Niagara Fluids docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-in-unreal-engine); [Flipbook Baker Quick Start](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-flipbook-baker-quick-start-guide-in-unreal-engine)
- **[Epic employee, 2022]** Templates exist for "fire, smoke, pools of water, splashes, and shallow water". — [80.lv](https://80.lv/articles/working-with-niagara-fluids-to-create-water-simulations)
- **[Epic]** UE 5.4 added caching support for 2D gas. — [UE 5.4 Release Notes](https://dev.epicgames.com/documentation/unreal-engine/unreal-engine-5.4-release-notes?application_version=5.4)
- **[Epic]** Unreal Fest 2023 talk: "Creating Visual Effects with Niagara Fluids, Sparse Volume Textures, and Heterogeneous Volumes". It describes SVTs as a compact cache format for real-time playback, with HV handling scene integration and lighting control, in a cinematic pipeline. — [Unreal Fest 2023 talk listing](https://forums.unrealengine.com/t/talks-and-demos-creating-visual-effects-with-niagara-fluids-sparse-volume-textures-and-heterogeneous-volumes-in-ue-unreal-fest-2023/1683078)
- **[Epic]** SVT import and playback:
  - Static or animated VDB import, with two attribute groups: A is 8-bit unorm (recommended for density) and B is 16-bit float.
  - The HV actor has Frame, Frame Rate, Start Frame, End Frame, Playing and Looping properties, with Blueprint "Play" and "Set Playing" nodes.
  - Playback defaults to 24 fps.

  — [HV docs](https://dev.epicgames.com/documentation/unreal-engine/heterogeneous-volumes-in-unreal-engine?lang=en-US)
- **[Epic]** SVT cost and streaming:
  - Memory is spent only on regions with "interesting" data, and unused channels cost nothing.
  - Animated playback "depends a lot on their resolution and sparseness".
  - Streaming suffers above roughly 30 to 50 MB per frame at 24 fps. Frames that arrive late fall back to coarser mips.
  - The Page Table is capped at 2 GB.

  — [SVT docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/sparse-volume-textures-in-unreal-engine)
- **[Community]** Caching through Niagara Sim Cache:
  - Caching 3D gas through Sequencer is possible.
  - The baker can save a simulation cache asset for runtime playback, but it works only inside the Niagara Editor.
  - Baked sims sometimes flicker or turn into blocky voxels. One user noted the VDB/SVT path has a Streaming Pool Size Factor fix, but the sim cache asset has no equivalent.

  — [Forum: Cached/baked fluid sim as in-game triggered FX](https://forums.unrealengine.com/t/cached-baked-fluid-sim-as-in-game-triggered-fx/1304091); [Forum: Caching Niagara Fluids with Sequencer](https://forums.unrealengine.com/t/tutorial-caching-niagara-fluids-with-sequencer/886259)
- **[Community]** Cost of cached playback versus a live sim (one data point, Dec 2024):
  - A cached 3D gas sim (about 200 frames, 200 resolution, "Set Sim Cache" Blueprint node) cost about 6 ms under the HeterogeneousVolumes GPU stat on an RTX 4080. The frame rate fell from about 110 to about 50 fps.
  - A responder called it "much more of a future-looking feature", suited mainly to tightly controlled cinematics.

  — [Forum: Low performance with cached gas](https://forums.unrealengine.com/t/low-performance-with-cached-gas-simulations-niagara-fluids-3d-gas-sim/2177751)
- **[Community]** Edits are limited after caching: you can't change cached smoke directly, so you adjust it with light colour and intensity. — [Arctiem blog](https://www.arctiem.com/post/adjusting-density-and-color-in-cached-niagara-3d-fluid-gas-simulations)
- **[Community]** OpenVDB export:
  - No direct Niagara-to-OpenVDB export was found.
  - Eidos-Montréal's unofficial plugin reads OpenVDB and NanoVDB into Unreal.
  - One forum thread asks about capturing a Niagara Fluids render-target volume into a volume texture from Blueprint, with no confirmed answer.

  — [eidosmontreal/unreal-vdb](https://github.com/eidosmontreal/unreal-vdb); [Forum: capturing sim results into volumetric textures](https://forums.unrealengine.com/t/niagara-fluids-capturing-simulation-results-into-volumetric-textures/1227743)

### Inferences
- The main lesson from caching is cost. Caching removes the solver cost but not the ray-march and lighting cost: the ~6 ms data point was all HV rendering.
- For our system, flipbook baking (2D) is the cheapest route to games.
- A cached 3D volume still needs a ray-marcher as efficient as ours or better.

### Gaps
- No Epic-published millisecond comparison of live sim versus sim-cache versus SVT playback was found.
- No Epic documentation of a Niagara-to-SVT bake path or an OpenVDB exporter for gas was found in the pages fetched.

## Status and roadmap by version

### Takeaway
**Confirmed labels:**

| Feature | Label | Version | Source |
|---|---|---|---|
| Niagara Fluids | Beta | 5.4 | 5.4 release notes |
| Heterogeneous Volumes | Experimental | 5.4 | 5.4 release notes |
| Heterogeneous Volumes | Beta | 5.5 | 5.5 release notes |
| Heterogeneous Volumes | Experimental | 5.9 docs page | current HV doc |
| Sparse Volume Textures | Experimental | 5.9 docs page | current SVT doc |
| MegaLights | Experimental | 5.5 | 5.5 release notes |

The 5.9 doc pages may contain stale text, so their labels may not reflect the current status. No Epic source found says Niagara Fluids became production-ready.

### Cited Findings
- **[Epic]** Niagara Fluids is listed as "(Beta)" in the 5.4 release notes. The same notes add 2D gas FFT pressure solve, cubic interpolation and caching, and 3D liquid two-way coupling. — [UE 5.4 Release Notes](https://dev.epicgames.com/documentation/unreal-engine/unreal-engine-5.4-release-notes?application_version=5.4)
- **[Epic]** HV is "Beta" in the 5.5 notes, which add overlapping HVs and VDB import into SVT. — [UE 5.5 Release Notes](https://dev.epicgames.com/documentation/en-us/unreal-engine/unreal-engine-5-5-release-notes)
- **[Epic]** The current HV and SVT doc pages, labelled 5.9, show the "Experimental" banner. — [HV docs](https://dev.epicgames.com/documentation/unreal-engine/heterogeneous-volumes-in-unreal-engine?lang=en-US); [SVT docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/sparse-volume-textures-in-unreal-engine)
- **[Epic]** A public roadmap card "Heterogeneous Volumes (Beta)" exists. A search snippet associated it with 5.8, but the page did not render, so this is unverified. — [Epic Public Roadmap](https://portal.productboard.com/epicgames/1-unreal-engine-public-roadmap/c/1710-heterogeneous-volumes-beta)
- **[Community]** The 5.7 coverage seen names PCG and Substrate as graduating to Production Ready. Niagara Fluids was not mentioned. — [CG Channel on UE 5.7](https://www.cgchannel.com/2025/11/unreal-engine-5-7-five-key-features-for-cg-artists/)
- **[Community, unreliable]** A third-party blog claims Niagara Fluids left Experimental in 5.5, became production in 5.6, and is shippable in 5.7. The blog contradicts its own dates and is unverified. — [StraySpark blog](https://www.strayspark.studio/blog/niagara-fluids-for-gameplay-beyond-vfx-2026)
- **[Community]** A forum user reported that `r.PathTracing.HeterogeneousVolumes` is absent in 5.7.4, possibly because it is now on by default. — [RealTimeVFX thread](https://realtimevfx.com/t/niagara-gas-sim-appears-partially-completely-black/30909)

### Gaps
- The 5.6 release-notes page did not render. 5.7, 5.8 and 5.9 release notes for Niagara Fluids were not retrieved.
- No Epic announcement of sparse or adaptive Niagara gas grids was found.
- The status of Niagara Fluids after 5.4 is unconfirmed.

## Machine learning or neural techniques for fluids and volumetrics

### Takeaway
No Epic machine-learning or neural feature for fluids, smoke detail or volume upscaling was found. Epic's related ML work is the Neural Network Engine (NNE), which went to Beta in 5.4, and an experimental NNE-based path-tracer denoiser. Neither targets fluids.

### Cited Findings
- **[Epic]** NNE moved from Experimental to Beta in 5.4. It gives a common API to run pre-trained models in the editor and at runtime. — [NNE Overview](https://dev.epicgames.com/documentation/en-us/unreal-engine/neural-network-engine-overview-in-unreal-engine); [UE 5.4 Release Notes](https://dev.epicgames.com/documentation/unreal-engine/unreal-engine-5.4-release-notes?application_version=5.4)
- **[Epic, per search snippet of the 5.4 notes]** The 5.4 notes include an experimental neural denoiser (NNEDenoiser) for the Path Tracer. Not verified in a fetched excerpt. — [UE 5.4 Release Notes](https://dev.epicgames.com/documentation/unreal-engine/unreal-engine-5.4-release-notes?application_version=5.4)
- **[Third party]** Intel's NNERuntimeOpenVINO plugin accelerates NNE inference. — [Intel guide](https://www.intel.com/content/www/us/en/developer/articles/guide/accelerated-ai-inference-for-unreal-engine-5.html)

### Inferences
- In principle NNE could host a custom neural fluid upscaler, but Epic provides none. A neural smoke-detail system would therefore have no direct Niagara counterpart.

### Gaps
- Searches found no Epic GDC, SIGGRAPH or Unreal Fest talk on neural volumetrics, an ML deformer-style tool for fluids, or neural volume upsampling.
