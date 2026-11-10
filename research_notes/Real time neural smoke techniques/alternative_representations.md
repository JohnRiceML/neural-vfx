# Alternative Representations for Real-Time Smoke and Fire (2019–2026)

Context: the team's current pipeline is a 64^3 Eulerian GPU solver, then a volume ray-march, then a screen-space detail CNN, within 6 ms GPU per effect at 1440p on an RTX 5080. These notes cover representations other than "dense grid + image-space network" that could give fine wisps more cheaply. Research was done on 2026-11-09 with about 20 search/fetch calls. Many numbers below come from abstracts or secondary summaries, and each one is flagged where that applies.

## 1. Gaussian splatting / kernel primitives for dynamic volumetric media (smoke, fire)

### Takeaway
Almost all Gaussian work on smoke so far reconstructs smoke from multi-view video offline. None of it is a real-time forward-simulation detail layer for games. The useful idea to borrow is "Gaussians as passive tracers advected by a velocity field" (LagrangianSplats), together with ray-traced or exact volumetric primitive rendering (EVER, "Don't Splat"). Rendering cost and sorting/popping in semi-transparent emissive media are still the main risks. The best published exact method runs at about 30 fps at 720p on an RTX 4090, which is far over a 6 ms budget.

### Cited Findings
- **LagrangianSplats** (Tao, Chen, Chu, PKU; SIGGRAPH 2026; arXiv 2605.09299). It reconstructs smoke density and velocity from sparse multi-view video. The method couples Lagrangian 3DGS primitives with a continuous Divergence-Free-Kernel velocity field, so incompressibility holds by construction. Gaussian positions are not free variables: they are passive tracers advected by the velocity field. Initialisation is standard 3DGS on frame 1. The authors report that baselines suffer "severe density drift and dissipation" in re-simulation, while their method holds shape better. A secondary review describes it as faster than implicit neural representations but "not yet capable of real-time reconstruction". It is limited to incompressible flow (smoke) and struggles with highly chaotic dynamics. Code is on GitHub and data on Hugging Face. No fps numbers were found. — [ACM DL](https://dl.acm.org/doi/10.1145/3799902.3811188); [project page](https://taoningxiao.github.io/lagrangiansplats/); [arXiv PDF](https://arxiv.org/pdf/2605.09299); [Moonlight review](https://www.themoonlight.io/en/review/lagrangiansplats-divergence-free-transport-of-gaussian-primitives-for-fluid-reconstruction)
- **FluidGS** (ACM MM 2025). It does physics-informed Gaussian splatting for sparse-view dynamic fluid reconstruction, coupling density with velocity estimation through differentiable advection. The authors argue that vanilla 3DGS struggles with fluids. The test input is rising smoke filmed by 5 fixed cameras. — [ACM DL](https://dl.acm.org/doi/10.1145/3746027.3755500)
- **Gaussian Splashing** (arXiv 2401.15318; venue later listed as CVPR 2025). It combines 3DGS with position-based fluids (PBD), and the same Gaussians are used for both physics and rendering. It adds per-Gaussian normals, an anisotropy loss and PBR reflections, and supports two-way solid–fluid coupling. It targets liquids, not smoke or fire. No real-time claim was found. — [arXiv](https://arxiv.org/abs/2401.15318v1); [HF papers](https://huggingface.co/papers/2401.15318)
- **GASP** (arXiv 2409.05819). It couples Gaussian rendering with physics for objects. Its stated limitation is artifacts "when the simulation changes shapes essentially", which is the normal case for smoke. — [arXiv](https://arxiv.org/pdf/2409.05819)
- **Gaussian Fluids** (Xing, Wang, Chu, Chen; SIGGRAPH 2025; arXiv 2405.18133). It is a grid-free forward solver: velocity is represented as a weighted sum of Gaussians (Gaussian Spatial Representation) and solved with first-order physics-guided optimisation. The authors state that it "does not yet match the speed of fluid solvers using explicit representations" but reduces spatial discretisation error. Code is public. The follow-up **DDFKs** (Dynamic Divergence-Free Kernels, arXiv 2601.17693) comes from the same line of work. — [arXiv](https://arxiv.org/pdf/2405.18133); [project](https://xjr01.github.io/GaussianFluids); [DDFKs](https://arxiv.org/html/2601.17693v1)
- **GaussFluids** (Pacific Graphics 2025) reconstructs Lagrangian fluid particles from video with Gaussian splatting. Details were not fetched. — [EG DL](https://diglib.eg.org/handle/10.2312/pg20251286)
- **"Don't Splat your Gaussians"** (Condor et al., Meta; arXiv 2405.15425, v3 Sep 2025). Scattering and **emissive** media are represented as mixtures of 3D kernel primitives (Gaussian and Epanechnikov). The method uses hardware ray tracing to find primitive bounds and closed-form transmittance and free-flight sampling, so it fits into standard volumetric path tracers. This supports scattering, relighting and inverse rendering. Example: a cloud stored as 5.6k primitives takes 224 KB, compared with 4 GB for a 1024^3 voxel grid. The implementation is in Mitsuba (offline). — [arXiv](https://arxiv.org/abs/2405.15425v2); [code](https://github.com/facebookresearch/volumetric_primitives)
- **EVER** (Google; ICCV 2025). It uses constant-density ellipsoids that are ray-traced with entry and exit intersections, so the volume integral is exact even with unlimited overlap. This removes the popping and view-dependent density of 3DGS. The authors point out that 3DGS sorts once per frame by mean, StopThePop sorts only approximately per ray, and both StopThePop and 3DGRT ignore primitive overlap. Speed is **about 30 fps at 720p on an RTX 4090** (BVH, OptiX/Slang/CUDA). — [arXiv](https://arxiv.org/html/2410.01804v5); [radiancefields.com](https://radiancefields.com/google-open-sources-ever)
- The wider 4DGS literature reports very high fps for opaque-ish scenes, for example "Disentangled 4DGS" at 343 fps, but these figures are for captured scenes, not participating media. — [arXiv 2503.22159](https://arxiv.org/pdf/2503.22159)

### Inferences
- A practical hybrid is: keep the 64^3 solver for velocity and **advect N detail Gaussians/particles as passive tracers**, the same way LagrangianSplats uses them. Each tracer carries sub-grid density or emission and an anisotropic shape that is stretched by the velocity gradient. This gives view-consistent 3D detail, which is the main weakness of a screen-space CNN. The cost moves to sorting and compositing semi-transparent splats at 1440p.
- Rasterised 3DGS-style splatting with per-frame mean sorting will pop when smoke is dense and overlapping. Exact methods (EVER) are about 33 ms at 720p on a 4090, which is roughly 10x over the budget at 1440p. A likely compromise is to splat detail primitives into a froxel or low-res 3D texture and then ray-march that (no sorting needed), or to use them only as a density modulation inside the existing ray-march.
- Self-shadowing for Gaussians is not solved in the real-time papers found. The "Don't Splat" closed-form transmittance is the right math, but it was shown offline.

### Gaps
- No paper was found that uses Gaussians as a **real-time sub-grid detail layer driven by a live coarse sim**. This looks like an open niche, not established practice.
- Per-frame rendering timings for LagrangianSplats, FluidGS and Gaussian Splashing were not found. Emissive and fire-specific Gaussian work for real time was not found. PhysGaussian and "Physically embodied Gaussian" were not checked in this pass. They target solids/MPM and robotics, so they are probably low relevance.

## 2. Particle-carried detail on a coarse grid (vortex particles, turbulence particles, flow maps, advected textures)

### Takeaway
Lagrangian detail on a coarse grid is the most mature and cheapest route to stable, view-consistent fine detail. Pfaff et al. 2010 is explicitly built for GPUs and "millions of particles at high framerates". Advected or Lagrangian textures (Neyret) are real-time and resolution-independent. The 2023–2025 flow-map work (Neural Flow Maps, Vortex Particle Flow Maps, Eulerian Vortex on Flow Maps) shows how to keep vortical detail alive for a long time, but it is offline research quality.

### Cited Findings
- **Pfaff, Thuerey, Cohen, Tariq, Gross**, "Scalable Fluid Simulation using Anisotropic Turbulence Particles" (SIGGRAPH Asia 2010, NVIDIA/ETH). Small-scale detail is offloaded to particles, so "a very low grid resolution" is enough for the base flow. A k-ε turbulence model drives anisotropic turbulence, and detail scales with particle count. The paper describes the method as highly suitable for massively parallel hardware, giving "detailed turbulent simulations with millions of particles at high framerates". Exact frame times were not found in this pass. — [NVIDIA Research](https://research.nvidia.com/publication/2010-12_scalable-fluid-simulation-using-anisotropic-turbulence-particles); [SIGGRAPH archive](https://history.siggraph.org/?p=197301)
- Wavelet-noise turbulence on GPU (Fujisawa et al., PG 2011, SPH + curl of wavelet noise): about **15 ms/frame for about 40k particles** on 2011 hardware. Spectrum and wavelet steps take about 70% of the time. — [EG DL](https://diglib.eg.org/items/2e5a2147-52c0-4be4-a48a-72ef71becd33/full)
- **Vortex filaments**: Weißmann & Pinkall built a real-time interactive GPU smoke solver based on discrete integrable vortex filaments (pre-2019, foundational). — [Semantic Scholar](https://www.semanticscholar.org/paper/Real-time-Interactive-Simulation-of-Smoke-Using-Weißmann-Pinkall/c2bf8cbcc785a2da3735493e268abb27fec2472f)
- **Neural Flow Maps** (Deng, Yu, Zhang, Wu, Zhu; SIGGRAPH Asia 2023, TOG 42(6):244). Spatially Sparse Neural Fields (small MLPs over multi-resolution sparse grids) store long-term velocity and compute bidirectional flow maps and Jacobians. The paper reports low dissipation and preserved leapfrogging and reconnecting vortices. It is a forward simulation and offline. — [arXiv 2312.14635](https://arxiv.org/pdf/2312.14635); [project](https://yitongdeng-projects.github.io/neural_flow_maps_webpage)
- **Vortex Particle Flow Maps** (Wang et al., Bo Zhu group; SIGGRAPH 2025). Vortex-in-Cell is combined with particle flow maps. The key finding is that vorticity allows much longer stable flow-map distances than velocity or impulse. A 2D leapfrog stayed stable for more than 600 s, while Neural Flow Maps, Particle Flow Maps and the Eulerian Vortex Method degraded earlier. It supports dynamic solid boundaries. — [arXiv 2505.21946](https://arxiv.org/abs/2505.21946)
- **An Eulerian Vortex Method on Flow Maps** (SIGGRAPH Asia 2024) advects vorticity using flow-map transport of line elements with bidirectional marching. — [arXiv 2409.06201](https://arxiv.org/abs/2409.06201)
- **Advected Textures** (Neyret, SCA 2003). Layers of advected, periodically regenerated texture parameterisations are blended by local accumulated deformation, trading spatio-temporal continuity against preserving texture statistics. — [EG DL](https://diglib.eg.org/items/c4bf80cc-23f0-4107-958d-e03c0b9c8ac4/full)
- **Lagrangian Texture Advection** (Yu, Neyret, Bruneton, Holzschuch, IEEE TVCG 2011). Texture is advected per pixel and locally, keeps its Fourier spectrum, and runs in real time on very large or "potentially infinite" scenes. — [Search summary of abstract; core.ac.uk](https://core.ac.uk/download/93038924.pdf)
- Related: "Flow-noise en temps réel" (Augustin & Neyret 2006 tech report). Only metadata could be seen. — [INRIA](https://evasion.inrialpes.fr/Publications/2006/AN06)

### Inferences
- For the 6 ms budget, the strongest non-neural candidate is to **advect detail carriers with the 64^3 velocity**. The carriers can be: (a) turbulence/vortex particles that inject sub-grid velocity, as in Pfaff; (b) passive density tracers or Gaussians; or (c) a 3D advected noise texture coordinate field with Neyret-style dual-phase regeneration, sampled during the ray-march. Option (c) costs roughly one or two extra 3D texture taps per march step plus a 64^3 or 128^3 coordinate advection pass. That is probably well under 1 ms, and it is inherently view-consistent, unlike a screen-space CNN.
- Flow-map ideas such as long-range back-and-forth advection and vorticity on particles can cut numerical dissipation in the coarse solver itself. That keeps more "real" vortical structure at 64^3 before any detail stage runs.

### Gaps
- Exact Pfaff timings and modern-GPU numbers were not found. No game-shipped postmortem (GDC) on advected noise for volumetric smoke was found in this pass. Production Niagara Fluids and EmberGen internals were not found either.
- Runtimes of Neural Flow Maps and VPFM on GPU were not retrieved. They are presumed offline, which is unverified.

## 3. Neural volumetric representations and compressed playback of baked high-res sims

### Takeaway
Neural fields for smoke (PINF, HyFluid) are reconstruction tools that need hours to days of training, so they are not usable at runtime. For playback of baked high-res sims, ZibraVDB is the production option: about 40x compression at default quality with GPU decompression in Unreal. Its per-frame decode ms figures appear only in charts on the vendor's site. NeuralVDB reaches 10–100x or more, but decode times are minutes, so it is offline. No published work was found that does "bake a library of high-res sims + latent blending conditioned on a live coarse sim".

### Cited Findings
- **PINF** (Physics-informed neural fields for smoke reconstruction, arXiv 2206.06577). Training is slow because PDE residuals are computed by autodiff. The "Game" scene took **64 hours on one Quadro RTX 8000**. The PINN velocity model is about 3x slower than a plain NeRF. — [arXiv](https://arxiv.org/pdf/2206.06577)
- **HyFluid** (NeurIPS 2023). Density uses a 4D-extended iNGP hash grid optimised by differentiable volume rendering from multi-view video, which requires multiple views. No training time was retrieved. — [project](https://kovenyu.com/HyFluid/); [related 2024 paper](https://arxiv.org/pdf/2412.13897)
- **NeuralVDB** (NVIDIA, TOG 2024; arXiv 2208.04448). It gives 10x to more than 100x compression over already-compact VDB, for example the Disney Cloud goes from 1.5 GB to 25 MB. NanoVDB "rarely exceeds 6x" at low noise. Encoding takes about 5 min and decoding about 3 min for the Disney Cloud. The authors recommend it as an "efficient but lossy offline representation". A [Hash,5,4,NN(3)] configuration balances memory against random access, but no timings were retrieved for it. It assumes a fixed tree topology. — [arXiv](https://arxiv.org/pdf/2208.04448); [NVIDIA](https://research.nvidia.com/labs/prl/publication/neuralvdb)
- **ZibraVDB** (vendor, self-reported). Default Quality 0.6 gives **about 40x** compression. Marketing claims up to 100x, and an older page says 20x. A planned temporal mode targets 150x. Decompression runs on the GPU and needs about 2–3x the frame size in VRAM. Test frames were 45.5 MB single-channel and 106 MB three-channel. Test GPUs were an RTX 4090 Laptop, an RTX 3090 and an RX 5700 XT (as a PS5 proxy), at 1920x1080. The vendor claims Unreal's Heterogeneous Volumes path is "about 3x slower" than its custom renderer and that ZibraVDB is about 2x faster than Unreal SVT and Heterogeneous Volumes. The illumination texture defaults to 1/4 resolution, and at most 15 lights affect volumes. Per-frame decode ms appear only in chart images and could not be extracted. — [Zibra perf blog](https://www.zibra.ai/blog-posts/zibravdb-performance-and-quality-optimization); [CG Channel 2024](https://www.cgchannel.com/2024/07/zibraai-unveils-zibra-vdb-compression)
- **Real-time Neural Six-way Lightmaps** (arXiv 2604.03748, 2026). This is the most game-relevant neural result found. A coarse camera-space ray-march with a large step makes a 512^2 "guiding map", and a U-Net (NAFBlock adapters) predicts six-way lightmaps that plug into standard engine six-way smoke shading. It handles obstacles, camera motion and light changes, and it was demonstrated in Unreal (compared against Niagara). Numbers on an unnamed GPU with 16,384 cores and 24 GB (consistent with an RTX 4090, but not stated): sim at velocity 100^3 / density 400^3 takes 4.0 + 11.5 ms; guiding map about 2 ms; inference **1.2 ms** (TensorRT); final render under 0.3 ms; obstacle shadow 0.8 ms. The intro claims "less than 4 ms" for the lighting pipeline, which is consistent with 2 + 1.2 + 0.3, but the Performance section totals about 19 ms including the sim, so the two are internally inconsistent unless the sim is excluded. Temporal stability comes from a FlowNet flow loss and is not measured quantitatively. Limitations: weak far from the training distribution, a blocky coarse guiding map, and only screen-space obstacle shadows. Baselines: Volumetric ReSTIR at 1 spp takes more than 10 ms plus about 2 ms denoise; MRPNN takes more than 170 ms of precompute on 400^3 and about 4 ms to re-light plus about 4 ms inference. — [arXiv HTML](https://arxiv.org/html/2604.03748)

### Inferences
- For hero effects that can be authored ahead of time, **baked high-res VDB playback (ZibraVDB-class) completely avoids the detail-hallucination problem**. The cost is losing interactivity: no response to the live coarse sim.
- A "library + live-sim-conditioned blending" approach appears to be unpublished. The closest pieces are flow-map advection of baked detail and conditioned neural rendering (six-way lightmaps).
- Neural Six-way Lightmaps shows that a 1–2 ms neural pass at 512^2 is feasible and engine-integrable. Its contribution is **lighting/self-shadowing**, not geometric detail. This matters because lighting may give more perceived quality per ms than wisps (see section 6).

### Gaps
- ZibraVDB decode ms per frame is only in charts, and there are no independent benchmarks. NeuralVDB random-access query cost was not found.
- No neural decoder for volumetric animation with published ms decode on a consumer GPU was found besides Zibra (vendor).

## 4. Graph-network / neural-operator / generative-video simulators for smoke

### Takeaway
No neural operator, GNN or transformer smoke simulator was found that reaches interactive rates at useful 3D resolution on one consumer GPU with documented temporal stability. Generative video models are being used to *refine* coarse smoke sims (WonderPlay, FluidNexus), but only offline. The fastest learned-fluid work found is still 2017–2018 era (CNN pressure solver; tempoGAN super-resolution).

### Cited Findings
- No 2025 neural-operator or transformer work with real-time smoke fps was found in a targeted search. The closest real-time neural item is Neural Six-way Lightmaps (rendering, not simulation). — [arXiv 2604.03748](https://arxiv.org/pdf/2604.03748)
- **tempoGAN** (Xie, Franz, Chu, Thuerey, SIGGRAPH 2018) is a volumetric conditional GAN for smoke super-resolution with a temporal discriminator for coherent detail. This is the standard baseline for learned sub-grid smoke detail. — [arXiv 1801.09710](https://arxiv.org/pdf/1801.09710)
- Chu et al. 2020 (spatio-temporal GAN) is described by the Six-way Lightmaps authors as having "prohibitively high inference cost" for real time. — [arXiv 2604.03748](https://arxiv.org/html/2604.03748)
- Diffusion fluid super-resolution in 2025 (arXiv 2504.05443 unpaired SR; SG-Diff 2504.04375) targets 2D CFD turbulence. There is no smoke-specific or real-time claim, and no explicit temporal coherence. — [2504.05443](https://arxiv.org/pdf/2504.05443); [2504.04375](https://arxiv.org/html/2504.04375v3)
- **WonderPlay** (arXiv 2505.18151). A coarse physics solver drives smoke, which it describes as "too sticky" from numerical viscosity. A video generator then refines motion and appearance, and conditioning on both motion and appearance was necessary. It is offline. — [arXiv](https://arxiv.org/pdf/2505.18151)
- **FluidNexus** (CVPR 2025) uses a position-based sim plus generative video refinement of rough renders for single-video 3D fluid reconstruction and prediction. **SmokeSVD** (arXiv 2507.12156) uses a diffusion model for side views plus physics consistency to reconstruct smoke from one video. **VideoPDE** (arXiv 2506.13754) treats PDE solving as video inpainting with a video DiT. All are offline. — [FluidNexus supp](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Gao_FluidNexus_3D_Fluid_CVPR_2025_supplemental.pdf); [SmokeSVD](https://arxiv.org/abs/2507.12156); [VideoPDE](https://arxiv.org/pdf/2506.13754)
- Classical reference point: an open-source CUDA Stable-Fluids smoke solver runs at 165 fps using about 5% of the frame budget on an RTX 2070 Super (resolution not stated). — [GitHub FluidSim](https://github.com/GraphicsCodeBase/FluidSim)

### Inferences
- "Coarse sim → generative refinement" is validated conceptually by WonderPlay and FluidNexus, but only with heavy offline video models. Few-step diffusion at 1440p within 6 ms with frame-to-frame coherence is not demonstrated for fluids. It would carry the same view-inconsistency risk as the current screen-space CNN, and probably more flicker.

### Gaps
- GNS / MeshGraphNets / FNO timings for 3D smoke on consumer GPUs were not retrieved. No StreamDiffusion-class real-time application to smoke was found.

## 5. Real-time fire-specific techniques

### Takeaway
Real-time fire in games is mostly temperature-field grids mapped through a blackbody LUT and ray-marched (the EmberGen lineage). Little recent academic real-time fire work turned up, and no neural fire with real-time numbers was found.

### Cited Findings
- EmberGen exports temperature in Kelvin, which can exceed 10,000. It has a user-set "blackbody max temperature" (one user used 4000 K). Mismatched temperature mapping between tools produces wrong gradients. This comes from user forums, not official docs. — [JangaFX forum](https://forums.jangafx.com/t/embergen-blender-cant-get-it-to-render-properly/532); [forum 2](https://forums.jangafx.com/t/blender-import-looks-nothing-like-embergen-render/422)
- JangaFX says EmberGen is used by more than 200 game studios and simulates and renders volumes in real time on the GPU (vendor claim). — [JangaFX](https://jangafx.com/software/embergen?irgwc=1)
- Example of a real-time blackbody renderer (student/independent project): LUT over 800–3000 K, fixed-step ray-march, 96^3 grid, at most 256 steps. — [Ignis Divinus](https://yijun-xiang.github.io/ignis-divinus/final-full.html)
- Older academic real-time fire work: pseudorandom noise for volumetric fire (VG 2008) and a real-time fire paper submitted to I3D 2007. — [Vanzine & Vrajitoru](https://www.cs.iusb.edu/~danav/papers/yvdv_vg08.pdf); [Fuller et al.](https://www.cs.ucdavis.edu/~hamann/FullerKrishnanMahrousHamannJoyFirePaperFor_I3D2007AsSubmitted11012006.pdf)
- Neural Six-way Lightmaps includes a transparency/emissive channel adapter, so it handles emission as part of lightmap prediction. — [arXiv 2604.03748](https://arxiv.org/html/2604.03748)

### Inferences
- Fire is emission-dominated and largely self-luminous, so self-shadowing matters less than for smoke. Fine detail at the flame front plus correct temperature-to-colour mapping (and HDR/bloom) likely dominate. Advected noise along the flame front, or thin-flame level-set detail, is a cheaper target than general smoke wisps. This is an inference, not sourced.

### Gaps
- No sources were retrieved this pass on thin-flame/level-set flame fronts in real time, soot models in games, or neural fire. No perceptual study of fire was found.

## 6. Perceptual guidance: what makes real-time smoke/fire read as high quality

### Takeaway
There is very little smoke-specific perceptual evidence. The ScalarFlow 2AFC study found an old procedural method (wavelet turbulence) roughly tied with modern solvers for realism. That suggests plausible procedural or advected detail is perceptually competitive. A thesis on game smoke found quality can be cut until a threshold is crossed, after which perception drops sharply.

### Cited Findings
- **ScalarFlow** (Eckert, Um, Thuerey, SIGGRAPH Asia 2019; arXiv 2011.10284) ran 2AFC user studies against real captured smoke. **Wavelet turbulence performed about on par with advection-reflection and slightly beat MacCormack**. In one example comparison there was about a 64% preference for advection-reflection over MacCormack relative to real reconstructions. Resolution was also varied. — [arXiv](https://arxiv.org/pdf/2011.10284)
- Um, Hu, Thuerey, "Perceptual Evaluation of Liquid Simulation Methods" (TOG 2017) collected more than 53,000 crowd votes. Having a real reference video made judgements consistent. It is about liquids, but the methodology transfers. — [arXiv 2011.10257](https://arxiv.org/pdf/2011.10257); [TUM PDF](https://ge.in.tum.de/download/2017-sig-um/2017-sig-um.pdf)
- An Aalborg University thesis (2011) on smoke in virtual environments found quality can be reduced unnoticed up to a threshold, beyond which the "illusion was broken". — [AAU](https://projekter.aau.dk/projekter/files/56059960/REPORT.pdf)

### Inferences
- Because procedural wavelet detail scored as well as better solvers, cheap **view-consistent advected or procedural sub-grid detail** is likely as good perceptually as expensive learned detail. Lighting/self-shadowing (cf. the six-way lightmap investment) and temporal coherence are probably where compute pays off. This is suggestive only, since no study isolated lighting against detail.

### Gaps
- No study was found that separates the perceptual weight of fine turbulence, self-shadowing, temporal flicker and motion for real-time smoke or fire. This is a real evidence gap.

## Cross-cutting summary for the team (inferences)
- **Cheapest likely upgrade over a screen-space CNN**: Lagrangian or advected detail driven by the 64^3 velocity, using turbulence/vortex particles (Pfaff), advected 3D noise coordinates (Neyret), or passive detail tracers (LagrangianSplats idea). All of these are view-consistent by construction, which the CNN is not.
- **Gaussian primitives**: promising as a data structure for detail tracers, but exact rendering is too slow (EVER at about 30 fps at 720p on a 4090). Rasterised splats will pop. The safest path is to inject them into the existing ray-march volume.
- **Neural spending**: the best-evidenced real-time neural use is lighting (Neural Six-way Lightmaps: 1.2 ms inference plus about 2 ms guiding map), not detail generation.
- **Baked hero effects**: ZibraVDB playback (about 40x compression, GPU decode, vendor numbers only) gives real high-res detail with no hallucination, at the cost of interactivity.
- **Generative video and neural operators**: not real-time at useful resolution as of the sources found, and temporal stability is unproven.
