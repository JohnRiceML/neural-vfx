# Hyper-performant in-frame neural inference and pressure-solve acceleration for real-time smoke (2018–2026)

Scope note: research was done on 2026-11-09 with a limited tool budget (~17 calls). Items marked **[recalled, not re-fetched]** come from the researcher's prior knowledge of a well-known primary source; the URL is the canonical one, but the specific number was NOT re-verified this session and should be checked before quoting in a final report. Everything else was read from the cited page this session.

Target pipeline: 64^3 Warp/CUDA smoke, 0.8–1.0 ms/step, 75–88% in MGPCG with ~25 dependent launches/iteration; ~730k-param screen-space CNN, 1.3–2.2 ms FP16 (PyTorch + CUDA graphs, RTX 5080, 1440p, half-res bbox).

## A1. TensorRT vs PyTorch, FP16/INT8/FP8, and runtime-in-engine (NNE, DirectML, Windows ML, Sentis)

### Takeaway
The only hard in-engine number found is NVIDIA's own: TensorRT-for-RTX inside Unreal NNE (RDG path) ran a style-transfer net in 3.8 ms vs 5.7 ms with DirectML at 1080p on an RTX 5090 (~1.5x, vendor-measured). Runtime choice gives ~1.5–2x; it does not by itself turn a 1.3–2.2 ms net into a sub-0.5 ms one. Tiling small fixed-shape models across a frame adds CUDA↔graphics context switches.

### Cited Findings
- NVIDIA blog: TensorRT for RTX (TRT) in UE NNE took **3.8 ms vs 5.7 ms for DirectML** on a **GeForce RTX 5090 at 1080p**, measured with Unreal Insights on the neural post-processing enqueue tasks. The model was the ONNX-zoo "candy" style-transfer net (fixed 1x3x224x224 I/O; the author suggests re-exporting at 1x3x720x720 to avoid tiling, but gives no timing for that). Vendor-authored measurement. — [NVIDIA Technical Blog: Speed up UE NNE inference with TensorRT for RTX](https://developer.nvidia.com/blog/speed-up-unreal-engine-nne-inference-with-nvidia-tensorrt-for-rtx-runtime/)
- The same post says TRT for RTX uses a **just-in-time optimizer that builds a GPU-specific engine once on the user's machine**. It does not state build time or engine size. It supports RTX GPUs from Turing (CC 7.5) through Blackwell (CC 10.0). NNERuntimeTRT supports NNE's GPU and RDG interfaces. Using it needs an **engine source build**, because the neural-profile runtime list (`ENeuralProfileRuntimeType`, `kRuntimeNames`) is hard-coded. Warning quoted from the post: small models tiled across larger framebuffers spawn multiple inference tasks per frame, "causing frequent CUDA-graphics context switches." — [same NVIDIA blog](https://developer.nvidia.com/blog/speed-up-unreal-engine-nne-inference-with-nvidia-tensorrt-for-rtx-runtime/)
- NVIDIA's plugin listing claims **~1.5x–2x over DirectML** (vendor claim), for Windows x64 and UE 5.7. — [Fab listing (via search snippet)](https://www.fab.com/listings/ab269c0f-24cc-4486-a45a-212f34aa8024)
- Epic NNE docs: for the RDG interface, "input and output tensors have to be provided as RDG buffers", and evaluation is added to the provided RDG graph builder. This keeps inference GPU-resident within the frame, unlike the CPU-synced GPU interface. — [Epic: Neural Network Engine overview](https://dev.epicgames.com/documentation/en-us/unreal-engine/neural-network-engine-overview-in-unreal-engine)
- Field reports (UE 5.3 era, forum, unverified): NNERuntimeRDGHlsl failed to create a model because of unsupported layers. RDG DirectML ran but crashed the editor after graph execution. An older reply said GPU-buffer I/O was unsupported, and a later reply said the experimental NNE plugin supports RDG buffers directly. — [UE forum thread](https://forums.unrealengine.com/t/running-neural-network-inference-with-gpu-input-output-tensors/690222)
- FP8 in practice is a tensor-core data type (E4M3/E5M2) on Ada (SM 8.9), Hopper and Blackwell, exposed by TensorRT through explicit Q/DQ quantization (ModelOpt). Blackwell consumer GPUs such as the RTX 50 series add FP4. INT8 needs calibration (PTQ) or QAT. **[recalled, not re-fetched]** — [TensorRT developer guide: working with quantized types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)

### Inferences
- For the 730k-param CNN, a TRT FP16 engine with fused conv+bias+activation is plausibly 1.3–2x faster than PyTorch+CUDA-graphs. PyTorch with graphs has already removed launch overhead, so the remaining gain comes from kernel selection and fusion, not dispatch. Expect roughly 0.8–1.4 ms, not 0.3 ms. This is an estimate extrapolated from the 1.5x vendor figure, not measured.
- Each crossing between the engine's D3D12/Vulkan queue and a CUDA context (TRT, PyTorch) needs external-memory and semaphore interop plus queue synchronization. NVIDIA itself flags this as a cost when inference is invoked many times per frame. Run one inference per frame on a fixed bbox-sized tensor (pad to a few shape buckets) rather than tiles.
- FP8/INT8 for an image-to-image generator risks banding in smooth gradients like smoke density. Standard practice is QAT, or keeping the first and last layers in FP16. No measured banding study for smoke nets was found.

### Gaps
- No independent (non-vendor) benchmark of TRT vs PyTorch eager vs ONNX Runtime-DML for a small (<1M param) image CNN at 1440p was retrieved.
- No measured TRT-for-RTX JIT build times, and no INT8/FP8 speedups on an RTX 5080 for small convnets.
- Unity Sentis (reportedly renamed "Inference Engine" in Unity 6.2) and Windows ML (reportedly GA in 2025, built on ONNX Runtime execution providers, with DirectML de-emphasized): **not verified this session**.
- Quantitative cost of a CUDA↔D3D12 interop fence per frame was not found.

## A2. Inference inside the renderer: Cooperative Vectors, VK_NV_cooperative_vector, RTX Neural Shaders, neural texture compression

### Takeaway
Cooperative vectors (D3D12 Shader Model 6.9, with a preview Agility SDK announced for late April 2025; Vulkan VK_NV_cooperative_vector) give shaders tensor-core matrix-vector ops, so small MLPs run inside pixel and compute shaders with no API crossing. The flagship example, RTX Neural Texture Compression "inference on sample", is reported at roughly 0.5–2 ms/frame on RTX 50-series. Third-party reports conflict on the slowdown without cooperative vectors.

### Cited Findings
- Microsoft (GDC 2025): cooperative vectors are "coming soon in Shader Model 6.9" and add hardware-accelerated vector and matrix ops for neural rendering. A preview Agility SDK was announced for "late April 2025". Use cases named were neural block texture compression and neural supersampling and denoising. **Intel claimed a "10x speed up in inference performance"** for neural compression models with cooperative vectors (vendor claim). The page lists no data types and no hardware support matrix. — [Microsoft DirectX blog, GDC 2025](https://devblogs.microsoft.com/directx/announcing-directx-raytracing-1-2-pix-neural-rendering-and-more-at-gdc-2025/)
- NVIDIA said its neural shading SDK would support DirectX through cooperative vectors and expose tensor cores to shaders. — [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidia-teams-up-with-microsoft-to-put-neural-shading-into-directx-giving-devs-access-to-ai-tensor-cores)
- NTC SDK: recommends Ada or newer and supports cooperative-vector acceleration on Vulkan and D3D12 for Ada and Blackwell. The fallback path is "significantly slower" and intended only for functional validation. — [NVIDIA-RTX/RTXNTC GitHub](https://github.com/NVIDIA-RTX/RTXNTC)
- NTC third-party measurements, which conflict with each other:
  - Compusemble, RTX 4090, 4K TAA, inference-on-sample: ~1,500 FPS with cooperative vectors vs just under 650 FPS without. — [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidias-new-tech-reduces-vram-usage-by-up-to-96-percent-in-beta-demo-rtx-neural-texture-compression-looks-impressive)
  - GamesGPU summary of a similar test: ~1,500 FPS falling to ~1,000 FPS without them. — [GameGPU](https://en.gamegpu.com/news/zhelezo/rtx-neural-texture-compression-nvidia-pokazala-testy-novoj-tekhnologii-szhatiya-tekstur)
  - Texture memory 11.37 MB, a 95.8% reduction, at 1440p. — [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidias-new-tech-reduces-vram-usage-by-up-to-96-percent-in-beta-demo-rtx-neural-texture-compression-looks-impressive)
  - Reported render cost 0.045 ms (BC) vs 0.111 ms (NTC), rising to 5.7 ms without cooperative vectors (GPU not identified in the snippet). — [TweakTown](https://www.tweaktown.com/news/105895/ai-texture-decompression-will-breathe-new-life-into-gpus-with-only-8gb-of-vram/index.html)
  - Sample mode adds 0.5–2 ms/frame on RTX 50-series (secondary aggregator). — [Pulse summary](https://www.pulse.bot/hardware/news/benchmarking-nvidias-rtx-neural-texture-compression-tech-that-can-reduce-vram-usage-by-over-80-c55d38be-58c2-40b6-9ce3-3fbe745fd64d)
- NVIDIA released RTX neural rendering tech, including neural shaders, as an Unreal Engine branch and plugins. — [NVIDIA blog: RTX neural rendering for UE developers](https://developer.nvidia.com/blog/nvidia-releases-rtx-neural-rendering-tech-for-unreal-engine-developers)
- Neural Radiance Caching (Müller et al. 2021) used fully fused tiny MLPs (tiny-cuda-nn) trained and inferred online every frame. The reported overhead was a few ms at 1080p on an RTX 3090 (≈2.6 ms). **[recalled, not re-fetched]** — [Real-time Neural Radiance Caching for Path Tracing, arXiv 2106.12372](https://arxiv.org/abs/2106.12372)
- Instant-NGP (Müller et al. 2022): a multiresolution hash encoding (e.g. L=16 levels, F=2 features, table size up to 2^19 per level) feeds a tiny fully fused MLP (64 neurons wide). The fully fused MLP is reported at roughly an order of magnitude faster than a TensorFlow MLP. **[recalled, not re-fetched]** — [Instant Neural Graphics Primitives, arXiv 2201.05989](https://arxiv.org/abs/2201.05989)
- 2026 preprint relevant to smoke in games: "Real-time Neural Six-way Lightmaps" (Li et al., arXiv 2604.03748). It predicts six-way smoke lightmaps from a cheap ray-marched guiding map and claims the result "can be seamlessly used in existing game engine pipelines." The abstract gives no ms, network size or resolution. — [arXiv 2604.03748](https://arxiv.org/abs/2604.03748)

### Inferences
- Cooperative vectors accelerate per-pixel matrix-vector work, which suits MLPs. A 3x3 CNN can be expressed as per-pixel matvecs over gathered neighborhoods (im2col in registers or groupshared). Spatial weight reuse is worse than in a cuDNN implicit-GEMM, though. So a 730k-param CNN gains most if it is restructured into a pixel-unshuffled, low-res, mostly 1x1 / depthwise network, or a per-pixel MLP on hand-crafted features.
- Running the net in an HLSL or Slang compute pass inside RDG avoids all interop and context-switch cost. That is the cost NVIDIA flags for TRT when tiled.

### Gaps
- No published ms figure was found for a hand-written HLSL CNN (as opposed to an MLP) at 1080p/1440p on RTX 40/50.
- AMD and Intel cooperative-vector support status in shipping drivers, and the VK_NV_cooperative_vector vs a cross-vendor KHR extension status, were not verified this session.
- DLSS network size and cost per frame are not published by NVIDIA in parameter terms. Only NVIDIA's per-GPU execution-time tables in the DLSS programming guide exist, and those were not fetched.

## A3. Model-side efficiency (distillation, re-parameterisation, pruning, NTIRE efficient-SR scale)

### Takeaway
Structural re-parameterisation is the cheapest proven win for a deployed image-to-image CNN: train with multi-branch blocks, export as plain 3x3 convs, and it is free at inference. Knowledge distillation from the 730k teacher to a ~100–200k student is the standard route to a 3–5x FLOP reduction. Specific measured ratios were not re-verified this session.

### Cited Findings
- RepVGG (Ding et al., CVPR 2021) trains 3x3+1x1+identity branches and merges them into a single 3x3 conv at inference. This yields a plain VGG-style net with better accuracy/speed trade-offs than ResNets on GPU. **[recalled, not re-fetched]** — [RepVGG, arXiv 2101.03697](https://arxiv.org/abs/2101.03697)
- ECBSR (Zhang, Zeng & Zhang, ACM MM 2021) applies edge-oriented re-parameterisable blocks to super-resolution for mobile real-time. At inference it collapses to a plain conv stack. **[recalled, not re-fetched]** — [ECBSR paper PDF](https://www4.comp.polyu.edu.hk/~cslzhang/paper/MM21_ECBSR.pdf)
- NTIRE Efficient SR challenge reports (2022–2025) are the reference for sub-1M-param image restoration nets with measured runtime. Winners use re-parameterisation, pixel-shuffle, and attention-free conv blocks. **[recalled, not re-fetched]** — [NTIRE 2024 ESR report, arXiv 2404.10343](https://arxiv.org/abs/2404.10343)

### Inferences
- Concrete recipe for the 730k CNN:
  - Pixel-unshuffle the half-res bbox by 2 (working at 1/4 res with 4x channels).
  - Use re-parameterised 3x3 blocks plus 1x1 mixing, and distil to ≤200k params.
  - Export fused weights and use FP16 throughout.
- This is how efficient-SR nets reach a few ms at 4K on consumer GPUs. The estimate is not measured for this net.

### Gaps
- Exact teacher→student compression ratios and quality deltas for GAN/image-to-image distillation (e.g. GAN Compression, Li et al. CVPR 2020) were not fetched this session.

## A4. Sparse inference (sparse/submanifold convs on small 3D grids)

### Takeaway
Sparse conv libraries are tuned for large, very sparse LiDAR point clouds (≪5% occupancy, 100k–1M points). At a 64^3 grid (262k cells) with 5–30% occupancy, dense FP16 conv on tensor cores is likely as fast or faster. No benchmark was found showing sparse wins in that regime.

### Cited Findings
- TorchSparse (MLSys 2022): 1.6x and 1.5x end-to-end speedups over MinkowskiEngine and SpConv. — [TorchSparse, arXiv 2204.10319](https://arxiv.org/pdf/2204.10319)
- TorchSparse++ (MICRO 2023): on an A100, end-to-end speedups of 2.9x, 3.3x, 2.2x and 1.7x over MinkowskiEngine, SpConv 1.2, TorchSparse and SpConv v2. — [TorchSparse++](https://ar5iv.labs.arxiv.org/html/2311.12862)
- A CVPR 2026 supplemental benchmarking fVDB v0.3.1 places its inference latency in the same range as TorchSparse++, with higher memory use. It gave no ratio. — [CVPR 2026 supplemental (PointCNN)](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Li_PointCNN_Performant_Convolution_CVPR_2026_supplemental.pdf)

### Inferences
- Sparse convs pay index-map (kernel map) building plus gather/scatter overheads that are fixed costs per layer. For a dense-ish 64^3 grid at 30% occupancy, brick-batched dense conv over occupied 8^3 or 16^3 bricks (with halo) is the practical middle ground.
- This is the screen-space net's analogue of the bbox crop.

### Gaps
- No published numbers were found for sparse vs dense conv at 5–30% occupancy on ≤128^3 grids.

## B1. GPU multigrid / MGPCG for real-time smoke and launch-overhead tricks

### Takeaway
Multigrid is the right solver, and one V-cycle beats tens to hundreds of Jacobi iterations. At 64^3, though, the solve is latency- and launch-bound, not bandwidth-bound. Fusing the 25 launches per iteration and capturing the whole fixed-iteration solve in a CUDA graph is the biggest lever. A fixed small V-cycle budget (1–2 V-cycles, warm-started, possibly as a standalone solver rather than inside PCG) is what production real-time tools converge on.

### Cited Findings
- vassvik (EmberGen / JangaFX author) on a WIP sparse EmberGen:
  - 134M active voxels, with ~240 ms of a step spent in the multigrid solve (4 V-cycles), about half the step.
  - One V-cycle is "equivalent to about a thousand Jacobi iterations" in effect, at the cost of "around 10 Jacobi iterations".
  - Jacobi at 40–160 iterations still smears and gains volume, and only ~1000+ iterations approaches the MG reference.
  - 1 V-cycle already gives "decent results".
  - GPU model and per-cycle timings are not given.
  — [vassvik gist: Jacobi vs Multigrid](https://gist.github.com/vassvik/f06a453c18eae03a9ad4dc8cc011d2dc)
- Glimberg, Erleben & Bennetsen (VRIPHYS 2009), GPU smoke with multigrid: up to 350x over CPU solvers, and multigrid up to 3x more accurate than Jacobi in the same time. Grids up to 256^3. — [Smoke simulation for fire engineering using multigrid on graphics hardware](https://test01.ku.dk/:obvius/pureproxy/3974268/en/publications/smoke-simulation-for-fire-engineering-using-a-multigrid-method-on)
- McAdams, Sifakis & Teran (SCA 2010), "A parallel multigrid Poisson solver for fluids simulation on large grids", is the canonical MGPCG for smoke: a V-cycle preconditioner with irregular-boundary smoothing, on multi-core CPU, with iteration counts nearly independent of resolution. **[recalled, not re-fetched]** — [McAdams et al. 2010 PDF](https://pages.cs.wisc.edu/~sifakis/papers/mgpcg.pdf)
- Lan et al. (neural-preconditioned Poisson, below) report AMG spends ~90% of its time in setup. That favours geometric MG with no setup for changing obstacles. — [arXiv 2310.00177](https://arxiv.org/html/2310.00177v5)

### Inferences
- At 64^3 the finest level has 262k cells (~1 MB per FP32 field). An RTX 5080 streams that in roughly a microsecond, so ~25 dependent launches per iteration at ~2–5 µs each (launch plus tail) dominate.
- Concrete levers:
  - Capture the whole solve with a fixed iteration count in a CUDA graph (`wp.ScopedCapture`).
  - Fuse the smoother, residual and restriction into one kernel per level.
  - Run the coarse levels (≤16^3) in a single thread-block kernel entirely in shared memory.
  - Fuse the CG dot products into a single-pass reduction plus update.
  - Warm-start the pressure from the previous frame.
  - Consider 1–2 V-cycles as the solver (no CG) with a fixed budget.
- Plausible outcome: 2–4x on the solve, e.g. 0.7 ms down to ~0.2–0.35 ms. This is an estimate, not measured.
- Red-black Gauss-Seidel converges about 2x faster per sweep than weighted Jacobi, but needs 2 launches unless fused with a grid-sync or per-block approach. Chebyshev smoothing gives Jacobi-like single-pass parallelism with better smoothing.

### Gaps
- No published per-solve ms for GPU MGPCG at 64^3–256^3 on modern consumer GPUs was retrieved this session.
- Niagara Fluids' pressure solver (iteration counts, solver type) and NVIDIA Flow's solver details were not found. A search for Niagara Fluids pressure iterations returned nothing product-specific.

## B2. Learned pressure solvers and preconditioners (Tompson 2017, DCDM 2023, neural PCG, neural multigrid)

### Takeaway
The best learned preconditioners now beat AMGX/AMGCL/IC by about 3–12x at 128^3–256^3. The strongest 2026 result beats a highly tuned geometric multigrid by only 1.03–1.26x. Absolute times at 128^3 are still tens of ms on an RTX 3080. One-shot CNN solvers like FluidNet are fast but leave large residual divergence. For a 64^3 solve already under 1 ms, learned methods are unlikely to help, and network evaluation would dominate.

### Cited Findings
- Lan et al., "A Neural-preconditioned Poisson Solver for Mixed Dirichlet and Neumann Boundary Conditions" (ICLR 2024 / arXiv 2310.00177). Benchmarked on a **Ryzen 9 5950X + RTX 3080**, tolerance a 10^6 residual reduction, with a ~25k-param net (4 levels).
  - Average per-solve runtime (units unlabeled in the table; most likely seconds, as CG at 256^3 = 13.2):

    | Grid | Ours | AMGX | AMGCL | IC | CG |
    |---|---|---|---|---|---|
    | Smoke-solid 128^3 | 0.072 | 0.324 | 0.696 | 0.852 | 1.050 |
    | Smoke-solid 256^3 | 0.636 | 3.156 | 6.433 | 11.1 | 13.245 |

  - Overall averages: 0.267 (Ours) vs 0.819 (AMGX) vs 3.232 (CG), i.e. ~3x vs AMGX and ~12x vs CG.
  - About 79.4% of the solver's time is network evaluation.
  - Peak memory at 128^3: 1.5 GiB (Ours) vs 5.2 GiB (FluidNet) vs 8.5 GiB (DCDM).
  — [arXiv 2310.00177v5](https://arxiv.org/html/2310.00177v5)
- Same paper, FluidNet bunny test (pure Neumann):

  | Method | Iterations | Time | Residual |
  |---|---|---|---|
  | CG | 790 | 0.5615 | 1e-6 |
  | Ours | 19 | 0.0524 | |
  | **FluidNet (Tompson-style one-shot)** | 1 | 0.0308 | **2.035** (i.e. no real convergence) |
  | DCDM | 39 | 12.35 | |

  The authors state DCDM did not beat CG or IC against CUDA baselines, contrary to the original DCDM paper. They also say DCDM overflowed GPU memory at 256^3, and that DCDM and FluidNet "struggle" with Dirichlet BCs. — [arXiv 2310.00177v5](https://arxiv.org/html/2310.00177v5)
- Kaneda et al., "A Deep Conjugate Direction Method for Iteratively Solving Linear Systems" (ICML 2023, PMLR 202). A CNN approximates A^-1 up to scale to generate CG search directions. The network is nonlinear, so the method is not a strict preconditioner. — [PMLR](https://proceedings.mlr.press/v202/kaneda23a.html)
- Lyu, Cen, Wu & Du, "A Multigrid-inspired Neural Iterative Solver for Poisson Equations on Large Voxel Grids" (ICLR 2026). It uses a hybrid dual-channel (classical smoother + conv) multigrid preconditioner that is kept SPD. The authors claim it is **5–10x faster than prior neural preconditioners** and **1.03–1.26x faster than a highly tuned geometric multigrid**, on 6,000+ instances with 16–64M unknowns. Hardware and absolute times are not on the abstract page. — [ICLR 2026 page](https://iclr.cc/virtual/2026/10015904); [PDF](https://people.iiis.tsinghua.edu.cn/~taodu/multigrid/lyu2026multigrid.pdf)
- Tompson, Schlachter, Sprechmann & Perlin (ICML 2017), "Accelerating Eulerian Fluid Simulation with Convolutional Networks" (FluidNet): a one-shot CNN pressure projection trained unsupervised on divergence. It is fast, but residual divergence remains; see Lan et al.'s residual 2.035 above. **[speedup numbers recalled, not re-fetched]** — [arXiv 1607.03597](https://arxiv.org/abs/1607.03597)
- Related: learned sparse approximate inverse preconditioners for CG on GPU (NeurIPS 2025, co-author Tao Du). — [arXiv 2510.27517](https://arxiv.org/html/2510.27517v1)

### Inferences
- Learned preconditioners pay off where the baseline is AMG with heavy setup, or plain IC/CG, at ≥128^3. Against a tuned GMG their margin is ≤1.26x even at 64M unknowns, and at 64^3 the net's own launch and latency cost would likely exceed the savings.
- A one-shot CNN projection (FluidNet-style) breaks mass conservation over long rollouts, which shows as volume gain or loss and smearing. It is not a safe drop-in for shipping unless followed by 1 V-cycle correction.

### Gaps
- The full text of Lyu et al. 2026 (hardware, ms) could not be parsed: the PDF was binary and no text extraction was available.
- Long-rollout stability numbers for FluidNet beyond the above were not re-verified.

## B3. Alternatives avoiding a global solve (FFT/DCT, LBM, vortex methods) and what games ship

### Takeaway
Lattice Boltzmann gives local, solve-free updates and has shown interactive smoke on GPUs. The retrieved figures are 157 FPS at 64x64x128 and a 2026 interactive LBM smoke at 100^3 velocity / 400^3 density. No 256^3 real-time LBM smoke number was found. Production real-time tools (EmberGen) use multigrid rather than Jacobi.

### Cited Findings
- Wei, Li, Mueller & Kaufman: GPU LBM smoke at interactive ~30 FPS at low resolution (older work, pre-2018). — [Stony Brook TVCG 2004 PDF](https://www3.cs.stonybrook.edu/~mueller/papers/smokeTVCG04.pdf)
- A vorticity-preserving LBM smoke paper reports **157.3 FPS on a 64x64x128 fine grid** (hardware not captured in the snippet). — [Springer, The Visual Computer 2018 (s00371-018-1514-x)](https://link.springer.com/10.1007/s00371-018-1514-x)
- A 2026 interactive smoke work uses a 3D LBM (weakly compressible) velocity solver on a **100^3 velocity grid with a 400^3 density grid**. FPS was not captured. — [arXiv 2604.03748](https://arxiv.org/pdf/2604.03748)
- Jacobi-only pressure (10–40 iterations, a common game choice) leaves visible volume gain and smearing. Multigrid is preferred. — [vassvik gist](https://gist.github.com/vassvik/f06a453c18eae03a9ad4dc8cc011d2dc)

### Inferences
- FFT/DCT Poisson at 64^3 with cuFFT is a handful of launches and exactly solves box domains. Obstacles would need an immersed or penalty correction or a few PCG iterations preconditioned by the FFT solve. This is a reasonable fallback, but GMG with fused kernels is likely equally fast at 64^3.
- LBM needs 19–27 distributions per cell (~19–27x the memory traffic of a velocity field) and a small time step, so it is not obviously cheaper at 64^3. Its benefit is the absence of a dependent-launch chain.
- Decoupling velocity resolution (64^3) from density resolution (e.g. 4x upsampled advected density) is a cheap visual-fidelity lever, used in the 2026 LBM work.

### Gaps
- Niagara Fluids and NVIDIA Flow default pressure-iteration counts were not found. NVIDIA Flow was reportedly a sparse-grid solver with a fixed-iteration (multigrid-like) pressure solve; **unverified**.
- No 256^3 real-time LBM smoke benchmark was found.
- Vortex-method real-time smoke numbers were not researched within budget.
