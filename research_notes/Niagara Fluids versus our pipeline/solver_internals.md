# Niagara Fluids 3D Gas: Solver Internals

Scope: how Unreal Engine's Niagara Fluids plugin simulates 3D gas (smoke and fire), for a technique-by-technique comparison with a 64^3 Eulerian GPU solver (MacCormack advection, MG-preconditioned CG pressure, vorticity confinement, combustion, optional 128^3 scalar upres, optional 1M-particle deposition to 128^3).

Source-quality legend: **[Epic]** = Epic documentation or Epic staff post. **[Community]** = user forum post, blog or third-party code reading (hearsay relative to Epic). **[Third-party marketing]** = commercial blog with unverified claims.

Overall caveat: Epic's public docs describe the gas templates at the artist-parameter level only. They do not document the numerical methods (advection scheme, grid staggering, pressure algorithm). Most numerical detail below comes from community code readings and must be treated as unverified. The engine source (which would settle these questions) sits behind Epic's GitHub login and was not accessed.

## 1. Grid types, layout (collocated vs staggered), dense vs sparse

### Takeaway
3D gas runs on dense, uniform, fixed-box Niagara Grid3D collections, sized by a single "Resolution Max Axis" over a user-set world box. I found no Epic source describing a sparse, adaptive or auto-resizing simulation grid. Sparse data (Sparse Volume Textures) is used only for caching and rendering. The one community code reading available (of the liquid template) shows cell-centred, collocated velocity with central-difference operators, not a MAC grid.

### Cited Findings
- The plugin is built on Niagara's Simulation Stages and Grid Collection data interfaces. [Community] — [iRender, Oct 2023](https://irendering.net/getting-started-with-niagara-fluids-in-unreal-engine-5/)
- "Gas simulations are represented by a grid ... each cell contains data representing the density", plus temperature and velocity. Smaller cells give higher quality at higher computational cost. [Epic] — [Fluid Simulation Overview](https://dev.epicgames.com/documentation/en-us/unreal-engine/fluid-simulation-in-unreal-engine---overview)
- Grid parameters in the 3D gas emitter Summary: "World Space Size" (the size of the container box), "Local Pivot", "Resolution Max Axis" (sets resolution from the longest side; higher is more accurate but slower), "Open Boundary +/- X/Y/Z" (toggles whether each face acts as a solid wall), "Grid Bounds Overlay", "Cell Size Guides". [Epic] — [Niagara Fluids Reference](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-reference-in-unreal-engine)
- The "Grid 3D Set Resolution" module derives `CellSize` = largest world extent / `NumCellsMaxAxis`. `NumCellsX/Y/Z` = floor(extent / CellSize), adjusted by one cell at edge cases, and the simulated box is recomputed as `Out_WorldGridExtents`. So cells are cubic and the box snaps to whole cells. [Community, 2022 code reading of the UE5 liquid template] — [songjihu.top mirror of CSDN analysis](http://songjihu.top/2022/05/13/NiagaraFluids%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%901/)
- That same reading shows velocity stored per cell (a `RasterizationGrid_velocity` with channels 0-2 = xyz and channel 3 = weight). `SimGrid` holds `Velocity` and `StartVelocity`, `TemporaryGrid` holds divergence, and `PressureGrid` holds pressure. Divergence is a central difference `/(2.*dx)` and the pressure gradient is `/(2.0*dx)`. Both are collocated, cell-centred stencils, not face-staggered. [Community; liquid template, not gas] — [songjihu.top](http://songjihu.top/2022/05/13/NiagaraFluids%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%901/)
- Staff: "the local pivot defines where the internal origin is for the simulation". Sources are transformed into the grid's UVW space. To move the effect, move the source particles, not the pivot. [Epic staff, Daniel Pearson] — [Forum: Exploring the 3D Gas Emitter](https://forums.unrealengine.com/t/tutorial-niagara-fluids-exploring-the-3d-gas-emitter/888748)
- A 2025 survey of large-volume rendering in UE5 reports VRAM limits with Niagara Fluids. Effective resolution per Niagara system must be low, the simulation grid stays allocated even when it is not updated, and grid data costs about 4x the memory of a scalar density value. [Academic] — [arXiv 2504.07485](https://arxiv.org/html/2504.07485v1)
- Sparse Volume Textures (SVT) are used to cache Niagara Fluids output for real-time playback, and Heterogeneous Volumes render them. [Epic, Unreal Fest 2023 talk abstract] — [Forum: UF2023 talk](https://forums.unrealengine.com/t/talks-and-demos-creating-visual-effects-with-niagara-fluids-sparse-volume-textures-and-heterogeneous-volumes-in-ue-unreal-fest-2023/1683078)

### Inferences
- The live simulation is a dense box, so it has the same memory scaling as a dense 64^3 solver. Sparsity comes in only after baking to SVT.
- If the gas template uses the same projection modules as the liquid template (plausible, since they share the Grid3D Fluid module library, but unverified), the gas solver is collocated with wide central-difference stencils. That is weaker than a MAC/staggered discretisation: it is prone to checkerboard (odd-even decoupling) pressure modes, and divergence is not exactly removed in the discrete sense.
- Moving the "Local Pivot" does not translate the grid through the world. There is no evidence of moving-window or follow-the-source bounds in the shipped templates.

### Gaps
- No Epic source states whether the 3D gas velocity is collocated or staggered. The only code reading found covers the FLIP liquid template. A second Chinese code reading of the gas template ([Zhihu, "UE5 Niagara Fluid Gas 速读"](https://zhuanlan.zhihu.com/p/692194304)) returned HTTP 403. Its search snippet describes gradient, divergence, curl, pressure-iteration and pressure-projection operators in the gas N-S module, but not their staggering.
- I found no Epic statement on sparse, adaptive or moving-bounds simulation for 3D gas in any 5.x version.
- Attribute formats (16-bit vs 32-bit floats per Grid3D attribute) were not found.

## 2. Advection and pressure projection

### Takeaway
Epic documents only "Pressure Solve Iterations" and "Pressure Relaxation" (0 to 1, "keep close to 1"). Community code readings and analyses say the pressure solve is a Jacobi-style or red-black SOR relaxation with a fixed iteration count, not multigrid or CG. Divergence is therefore not fully resolved at real-time iteration counts. I found no public statement of the advection scheme. Nothing found mentions MacCormack or BFECC. Advection is presumably semi-Lagrangian, but that is unverified.

### Cited Findings
- "Pressure Relaxation": a value between 0 and 1 controlling solver convergence, recommended "close to 1". "Pressure Solve Iterations": "The more iterations you perform in the solver, the more accurate, however the slower it will be." No numeric defaults are given. [Epic] — [Niagara Fluids Reference](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-reference-in-unreal-engine)
- Overview: pressure solving "involves solving a system of equations" and is iterative, with "more iterations resulting in a more accurate simulation". It also notes that pressure solving can "dampen" the effect of added forces. [Epic] — [Fluid Simulation Overview](https://dev.epicgames.com/documentation/en-us/unreal-engine/fluid-simulation-in-unreal-engine---overview)
- "The pressures are solved using an iterative Jacobi method". "better approximations were obtained with 10 iterations (instead of 6 as suggested in the videos)". The divergence-free velocity uses "Helmholtz decomposition". "The initial border conditions of pressure and temperature are set to 0." [Community user analysis, Jan 2023, no Epic confirmation] — [Forum: Injecting scalar fields](https://forums.unrealengine.com/t/niagara-fluids-injecting-scalar-fields-from-external-data-sources/740604)
- Liquid template code reading: the pressure solve is red-black SOR with **50 iterations**, update `Pressure = (1 - Weight) * P_center + Weight * JacobiPressure`. `FluidCellCount` starts at 6 and is reduced per solid or empty neighbour, `BoundaryAdd` injects solid-boundary relative velocity, and `PressureGrid` is initialised to 0. [Community; liquid template] — [songjihu.top](http://songjihu.top/2022/05/13/NiagaraFluids%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%901/)
- A summary of a Chinese gas-template reading says the pressure projection module takes the gradient of the relaxation-iterated pressure and projects the advected velocity to a divergence-free field. The listed N-S operators are gradient, divergence, curl, pressure iteration and pressure projection. [Community; search snippet only, page 403] — [Zhihu p/692194304](https://zhuanlan.zhihu.com/p/692194304)
- For water templates, the iteration count "determines Solve Pressure sim stage iterations count". The pressure solve is a repeated Niagara simulation stage. [Community/press] — [80.lv, water sims](https://80.lv/articles/working-with-niagara-fluids-to-create-water-simulations)
- Gas-adjacent advection: "Grid3D_Gas_CONTROLS_CINE_Emitter" includes a simulation stage that "advects texture coordinates" through the sim. Particles can be advected by the velocity field via the "Grid 3D Gas Particle Advection" template [Epic staff]. — [Niagara Fluids Reference](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-reference-in-unreal-engine); [Inside Unreal thread, Dan Pearson](https://forums.unrealengine.com/t/inside-unreal-creating-fluid-simulation-in-ue5/533695?page=2)

### Inferences
- The pressure solver is a relaxation-only fixed-iteration solver (Jacobi or red-black SOR with relaxation weight "Pressure Relaxation"). At 6-50 iterations on a 100-300 cell axis, low-frequency divergence is far from resolved. Expect visibly compressible behaviour at large scales, with good small-scale projection. An MG-preconditioned CG at 64^3 should give much better large-scale incompressibility per frame. Niagara's per-iteration cost is lower and its grid is larger.
- "Pressure Relaxation close to 1" plus red-black ordering matches a weighted SOR/Jacobi in which 1.0 means plain Gauss-Seidel/Jacobi. Values above 1 (over-relaxation) appear not to be exposed, given the 0-1 range.
- The CINE emitter advects texture coordinates, which is a procedural detail technique. It suggests the base density advection is diffusive enough to need help for cinematics.

### Gaps
- **Advection scheme:** no Epic source or code reading found that names it (semi-Lagrangian vs MacCormack/BFECC, linear vs cubic sampling). Not found.
- **Default iteration counts and relaxation value** for the shipped gas templates: not found in docs. "6" is attributed to Epic videos only second-hand by a user. "50" is from the liquid template code reading.
- Whether any multigrid or CG path exists anywhere in Niagara Fluids: none found. Searches for multigrid, conjugate gradient and red-black with Niagara Fluids returned only the community SOR reading.

## 3. Detail: vorticity confinement, turbulence noise, scalar upres, particles

### Takeaway
Niagara Fluids 3D gas adds detail with (a) a user-scaled "Vorticity Confinement" term, (b) curl/Perlin-style turbulence noise injection ("Seed Turbulence" at init plus "Turbulence 1/2" while running), and (c) in the CINE variant, advected texture coordinates. I found no evidence of a separate higher-resolution scalar grid (no "scalar resolution multiplier"), wavelet turbulence, or particle-to-grid density upres for gas.

### Cited Findings
- "Vorticity Confinement": "Enter a value to amplify the vorticity in the simulation by this amount." [Epic] — [Niagara Fluids Reference](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-reference-in-unreal-engine)
- Turbulence: "Seed Turbulence" (Gain, Frequency, Speed, Local Space) applies on the initialisation frame only. "Turbulence 1" and "Turbulence 2" apply during the sim, with Density/Temperature Gain, Density/Temperature Band, Frequency, Speed, Bias and Local Space. So noise can be masked by density and temperature bands. [Epic] — [Niagara Fluids Reference](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-reference-in-unreal-engine)
- "Turbulence is not really solved, only Perlin noise is applied at different frequencies." [Community] — [Forum: Injecting scalar fields](https://forums.unrealengine.com/t/niagara-fluids-injecting-scalar-fields-from-external-data-sources/740604)
- CINE emitter: "a slower, deterministic" particle-source variant for cinematics with "advanced texture coordinates". It advects texture coordinates through the sim, with Texture params "Gain, Scale, Remap To 0/1, Value Data, Value Band Min/Max/Sharpness, Loop Duration, Debug Texture" (Grid 3D Gas Explosion Cine only). [Epic] — [Niagara Fluids Reference](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-reference-in-unreal-engine)
- Source injection is particle-driven: "Grid3D_Gas_CONTROLS_Emitter" adds particle source support where density, temperature and velocity are injected from a second emitter. Attributes: `fluids_source_density`, `fluids_source_temperature`, `fluids_source_velocity`, `fluids_source_radius`, with multipliers and "Splat Size" per quantity, "Use Falloff", "Use Radius", "Scale Emission by dt". [Epic] — [Niagara Fluids Reference](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-reference-in-unreal-engine)
- Particles can be advected by the gas velocity field ("Grid 3D Gas Particle Advection" template) and then behave like any other Niagara particles. This is one-way advection for secondary particles, not deposition back to a render volume. [Epic staff] — [Inside Unreal thread](https://forums.unrealengine.com/t/inside-unreal-creating-fluid-simulation-in-ue5/533695?page=2)

### Inferences
- Our pipeline's 128^3 scalar field advected by 64^3 velocity, and the 1M-particle deposition to 128^3, have no documented Niagara Fluids equivalent. Niagara keeps density, temperature and velocity at one resolution and adds detail through noise, vorticity confinement and (cinematic only) advected UVs.
- Niagara's band-masked turbulence (noise gated by density and temperature ranges) is a cheap art-directable trick our solver could copy.

### Gaps
- No Epic statement found on a scalar-vs-velocity resolution multiplier, wavelet turbulence, or procedural upres in any version. Not found, which is not proof of absence.
- Vorticity confinement default value and its curl stencil: not found.

## 4. Combustion / fire model

### Takeaway
Epic's docs expose no fuel or burn chemistry for 3D gas. Fire is a temperature scalar rendered through a blackbody or curve map, with temperature buoyancy and density "buoyancy" (downward weight). A community analysis says temperature is an unphysical scalar, with a flame/smoke threshold near 0.02, and that density and temperature only dissipate (no diffusion). There is no documented fuel field, ignition temperature or gas expansion (divergence source) term.

### Cited Findings
- Simulation params: "Density Dissipation", "Temperature Dissipation", "Velocity Dissipation" (higher means faster fade). "Density Buoyancy" adds downward velocity from density, "Temperature Buoyancy" adds upward velocity, and "Gravity" sets direction and magnitude. [Epic] — [Niagara Fluids Reference](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-reference-in-unreal-engine)
- Quick-start example values (Grid3D_Gas_Explosion via the Grid3D_Gas_CONTROLS_Emitter): Density Dissipation **3.0**, Temperature Dissipation **1.5**, Velocity Dissipation **0.8**, Temperature Buoyancy **3.5**. Render Density **Linear** with Render Density Gain **0.5**, and Render Temperature **Curve**. The default curve is a black body ramp (black, red, orange, white). [Epic] — [Quick Start Guide](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-quick-start-guide-for-unreal-engine)
- Render Temperature modes: None, Black Body, Curve. With Black Body, the curve's alpha sets fire opacity. With Curve, the curve sets both colour and opacity. Further params: "Temp Color Gain", "Temp Opacity Gain". [Epic] — [Niagara Fluids Reference](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-reference-in-unreal-engine)
- "there is no direct interpretation of physic temperature". "A threshold is established to separate what is smoke and what is flame with temperature values close to 0.02". "It is not a complete thermo-fluid-dynamic model". "There is no real diffusion of density or temperature, just dissipation (decay) over time". Mass is unity, viscosity is not a force, and buoyancy acts only along the vertical/gravity axis. [Community, Jan 2023] — [Forum: Injecting scalar fields](https://forums.unrealengine.com/t/niagara-fluids-injecting-scalar-fields-from-external-data-sources/740604)

### Inferences
- Our combustion model (assuming it has a fuel field consumed into heat and smoke) is more physically structured than the documented Niagara model, which is source-injected density and temperature with exponential decay plus buoyancy.

### Gaps
- No fuel/burn/ignition/expansion parameter found for Grid 3D Gas Fire or Explosion. I could not determine whether the Explosion template uses a divergence source for expansion or only velocity injection from the particle source.

## 5. Default and typical resolutions; real-time guidance

### Takeaway
The Epic quick start sets ResolutionMaxAxis = **300** with WorldSpaceSize 300 x 300 x 600 for an explosion, which works out to about 150 x 150 x 300 cells at 2 units per cell. A user reports the shipped smoke template's base resolution as **200**. Epic positions 3D gas as "hero effects" and cinematics, recommends 2D gas or baking to flipbooks/SVT for general real-time use, and gives no numeric real-time maximum.

### Cited Findings
- Quick start: "ResolutionMaxAxis" set to **300** (to reduce pixelation, keeping it "as small as possible" for memory and performance), "WorldSpaceSize" **300, 300, 600**. The template is Grid3D_Gas_Explosion and the system is Grid3DGasExplosion. [Epic] — [Quick Start Guide](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-quick-start-guide-for-unreal-engine)
- A user's 3D smoke sim "uses all the base quality settings (200 resolution)", and going lower "loses too much quality". [Community] — [Forum: Low performance with cached gas sims](https://forums.unrealengine.com/t/low-performance-with-cached-gas-simulations-niagara-fluids-3d-gas-sim/2177751)
- 3D gas "simulates depth and a more complex flow at a higher memory and GPU cost". It is "best used for hero effects in real-time applications" and cinematics, and results "can also be baked into textures for increased performance in real time". 2D sims are "generally much faster than 3D" but "lack the complex turbulent flow". [Epic] — [Fluid Simulation Overview](https://dev.epicgames.com/documentation/en-us/unreal-engine/fluid-simulation-in-unreal-engine---overview)
- Templates exist for 2D Gas, 2D Liquid, 3D Gas, 3D Liquid and Shallow Water. The docs suggest creating a Fluids emitter and baking to a flipbook for use on any material. [Epic, via doc search summary] — [Niagara Fluids Reference](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-reference-in-unreal-engine)
- Users report crashes with "Grid3D_Gas_Master_Emitter" at high resolution (Dec 2024), with no Epic answer. [Community] — [Forum: crash at high resolution](https://forums.unrealengine.com/t/how-to-fix-crashing-with-niagara-grid3d-gas-master-emitter-resolution-too-high/2087476/2)
- The "Scalability" section overrides settings by default at Cinematic quality. [Epic] — [Niagara Fluids Reference](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-reference-in-unreal-engine)
- Emitter hierarchy: Grid3D_Gas_Emitter (core sim), then Grid3D_Gas_Controls_Base_Emitter (central controller, debug slice, render), then Grid3D_Gas_CONTROLS_Emitter (particle sources), then Grid3D_Gas_CONTROLS_CINE_Emitter. In 5.3 the core functionality "has been folded into" Grid 3D Gas Master Emitter ("the hub for all 3D Gas simulations"). The old emitter stays for backwards compatibility but is "no longer exposed". [Epic + Epic staff] — [Niagara Fluids Reference](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-reference-in-unreal-engine); [Forum: Exploring the 3D Gas Emitter](https://forums.unrealengine.com/t/tutorial-niagara-fluids-exploring-the-3d-gas-emitter/888748)

### Inferences
- A max-axis of 200-300 means the longest axis has 200-300 cells and the others scale with the box aspect. A 300 x 300 x 600 box at 300 gives about 3.4M cells, roughly 13x our 64^3 (262k) velocity grid and about 1.6x our 128^3 (2.1M) scalar grid. Niagara therefore spends its budget on a single high-resolution dense grid with a cheap solver, where ours uses a small velocity grid with a strong solver and upres.

### Gaps
- Per-template defaults (Smoke, Fire, Explosion, Gas Controls) for ResolutionMaxAxis, iterations and vorticity were not documented. Only the quick-start 300 and the user-reported 200 were found.
- No Epic-published frame timings or "max recommended for real time" numbers for 3D gas.

## 6. Collisions, character interaction, wind/forces

### Takeaway
Collision uses data interfaces that rasterise scene objects into a boundary mask: static meshes, geometry collections, physics assets (skeletal characters), mesh distance fields, and depth maps. Solid velocity enters the pressure solve. External wind and vector field injection are not documented as built-ins.

### Cited Findings
- Colliders are added as data interfaces "Collide_GeometryCollection", "Collide_PhysicsAsset", "Collide_StaticMesh". Level actors are picked by the actor tag "collider". [Epic] — [Quick Start Guide](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-quick-start-guide-for-unreal-engine)
- "Collide Against" options add data interfaces so the sim collides with actors chosen in Override Parameters. [Epic] — [Niagara Fluids Reference](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-reference-in-unreal-engine)
- Boundary conditions mask out object locations. Many object types, such as static meshes, Geometry Collections and depth maps, can be colliders. [Epic] — [Fluid Simulation Overview](https://dev.epicgames.com/documentation/en-us/unreal-engine/fluid-simulation-in-unreal-engine---overview)
- Staff: the resolution of distance fields used as collisions can get coarser with distance, so mesh distance fields are one collision path. [Epic staff] — [Forum: Exploring the 3D Gas Emitter](https://forums.unrealengine.com/t/tutorial-niagara-fluids-exploring-the-3d-gas-emitter/888748)
- Code reading (liquid): cells typed `FLUID_CELL`/`SOLID_CELL`/`EMPTY_CELL`, `SolidVelocity` stored per solid cell and added via `BoundaryAdd` in the pressure solve, and solid faces take the solid velocity in projection. This gives moving colliders proper two-way velocity coupling on the fluid side. [Community] — [songjihu.top](http://songjihu.top/2022/05/13/NiagaraFluids%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%901/)
- Limitations (staff, 2022): multi-GPU is unsupported, real-time sims are not network-replicated (run per client), and fluids could not yet be shadowed by scene geometry. [Epic staff] — [Inside Unreal thread](https://forums.unrealengine.com/t/inside-unreal-creating-fluid-simulation-in-ue5/533695?page=2)
- Users asked how to inject wind or external vector/scalar fields and got no Epic answer in the thread. [Community] — [Forum: Injecting scalar fields](https://forums.unrealengine.com/t/niagara-fluids-injecting-scalar-fields-from-external-data-sources/740604)

### Inferences
- Character interaction comes through physics-asset capsules and spheres (Collide_PhysicsAsset), not per-triangle skinned meshes.
- Wind is presumably done through Gravity/velocity sources or custom modules, since no built-in wind input is documented.

### Gaps
- GBuffer/scene-depth collision for 3D gas specifically (depth maps are mentioned generically in the overview): unconfirmed for 3D.
- Whether the fluid pushes back on rigid bodies or characters (true two-way coupling): not found.

## 7. Version history and maturity

### Takeaway
Epic's current docs (labelled "Unreal Engine 5.9 Documentation") still mark the plugin as **Beta**: "this plugin is still in Beta" and "use caution when shipping with it". The milestones I could confirm: the Inside Unreal stream on fluids in the UE5 era (2022); the 5.3 consolidation into the Grid 3D Gas Master Emitter, plus SVT caching and Heterogeneous Volume rendering; and a 5.4 fix for Heterogeneous Volume plus volumetric-fog compositing. A third-party blog's claim that it became "production-ready" in 5.6 contradicts Epic's own Beta label.

### Cited Findings
- "Learn to use this Beta feature, but use caution when shipping with it." The page is titled for UE 5.9 docs. [Epic] — [Fluid Simulation Overview](https://dev.epicgames.com/documentation/en-us/unreal-engine/fluid-simulation-in-unreal-engine---overview)
- "A warning message is displayed, because this plugin is still in Beta." (UE 5.9 docs) [Epic] — [Quick Start Guide](https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-quick-start-guide-for-unreal-engine)
- In 5.2 the master emitter was missing from the create dialog ("an oversight in 5.2"; workaround: untick Library Only). In 5.3 filtering was fixed and the Grid 3D Gas Emitter was folded into the Master Emitter. The 5.4 preview fixed a Heterogeneous Volumes vs volumetric fog compositing issue. Heterogeneous Volumes must be enabled in project settings. [Epic staff] — [Forum: Exploring the 3D Gas Emitter](https://forums.unrealengine.com/t/tutorial-niagara-fluids-exploring-the-3d-gas-emitter/888748)
- In 5.3 Niagara gained support for transforming fluid sims after caching (duplicate/offset caches), and SVT/OpenVDB import arrived as experimental. [Press] — [CG Channel, UE 5.3](https://www.cgchannel.com/2023/09/unreal-engine-5-3-five-key-features-for-cg-artists)
- A page that appears to reproduce the 5.3 notes says "Niagara fluids are now cached internally to SVTs resulting in significant performance and memory improvements". [Unverified aggregator; via search snippet] — [wildvisualizer](https://wildvisualizer.com/conversation/wildchat/2194402)
- Claims that Niagara Fluids became production-ready in 5.6 and that the 2D/3D solvers stabilised in 5.7. [Third-party marketing; contradicted by Epic's Beta label] — [StraySpark blog](https://www.strayspark.studio/blog/niagara-fluids-for-gameplay-beyond-vfx-2026)
- "up to 30% faster in Unreal Engine 5.8" (a creator's claim about liquids, not gas). [Community] — [80.lv, UE 5.8 water](https://80.lv/articles/exploring-real-time-water-simulations-in-unreal-engine-5-8/)
- An open bug: SVT/Heterogeneous Volumes and Niagara Fluid sims render only in the left eye in VR. [Epic issue tracker, via search snippet] — [UE-209789](https://issues.unrealengine.com/issue/UE-209789)

### Inferences
- As of late 2026 the solver appears architecturally unchanged since 5.0-5.3 (a dense Grid3D sim with a relaxation pressure solve). Later changes concern rendering (Heterogeneous Volumes), caching (SVT) and performance. No source describes a solver-algorithm upgrade.

### Gaps
- I could not confirm from a primary source the 4.26 experimental or early-preview origin (Epic showed fluids work around 4.26/5.0 EA in demos, but I found no citable release note), the exact version where the plugin went Experimental to Beta (believed 5.1, unconfirmed), or official 5.5-5.9 release-note entries for Niagara Fluids gas.
