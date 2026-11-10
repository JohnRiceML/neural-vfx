# Real-time Neural Rendering and Image Reconstruction Techniques Transferable to Volumetric Smoke/Fire Detail (2018–2026)

Verification note for the report writer: items marked **[V]** were read in a primary or near-primary source during this session. Items marked **[PK – not re-verified]** come from the researcher's prior knowledge of the primary source (URL given). Their numbers should be checked before they are quoted as fact. The research budget was 15 tool calls. The DLSS Programming Guide PDF could not be text-extracted (compressed PDF, no local PDF library), so no DLSS execution-time table was verified.

Context assumed from the brief: a 64^3 smoke sim is rendered at 1440p with alpha, depth and motion-vector buffers. The detail CNN is single-frame and only sharpens edges. Problem (1): fluid-velocity motion vectors do not match the apparent motion of the image. Problem (2): flicker metrics reward blur.

---

## Q1. Temporal upscalers (DLSS 2/3/3.5/4, FSR 2/3/4, XeSS, TSR, Arm/Qualcomm, academic): inputs, history design, cost, and how they handle transparency, particles and smoke

### Takeaway
Every production temporal upscaler is built around per-pixel motion vectors from opaque geometry plus history rectification (neighbourhood colour clamping and disocclusion tests). None of them has a real solution for alpha-blended smoke or fire. The documented approach is to *reduce trust in history* for those pixels (FSR's reactive mask and transparency & composition mask, DLSS's now-deprecated bias-current-colour mask, UE's "responsive AA"), which means falling back to the current frame. That is the same "no reprojection is better than wrong reprojection" conclusion the team reached. Costs: about 0.3–1.2 ms at 1440p for analytic FSR 2 on desktop AMD GPUs, and about 0.5–1.4 ms at 4K for DLSS CNN versus transformer on an RTX 4090.

### Cited Findings
**FSR 2 (analytic, open source): inputs and design**
- **[V]** Inputs: render-resolution colour (HDR flag optional), depth (inverted or infinite flags), 2×float motion vectors at render or display resolution, an optional R8_UNORM reactive mask, and a 1×1 R32 exposure (optional with auto-exposure). "All inputs that are provided at Render Resolution, except for motion vectors, should be rendered with jitter." FSR2 stores MVs internally as 16-bit, so extra precision gives nothing. — [AMD FidelityFX SDK FSR2 docs](https://gpuopen.com/manuals/fidelityfx_sdk/fidelityfx_sdk-page_techniques_super-resolution-temporal/)
- **[V]** The reactive mask is for areas that "do not leave a footprint in the depth buffer or include motion vectors", e.g. "particles, or alpha-blended objects which do not write depth or motion vectors". AMD guidance: "applications should write `alpha` to the reactive mask", with the maximum clamped to "around 0.9". Without the mask, "shading change detection logic will handle these cases as best it can." — [AMD FSR2 docs](https://gpuopen.com/manuals/fidelityfx_sdk/fidelityfx_sdk-page_techniques_super-resolution-temporal/)
- **[V]** Auto-generation (`ffxFsr2ContextGenerateReactiveMask`) takes two colour buffers, opaque-only and opaque+transparent, and builds the mask from a luminance-difference heuristic. Defaults: `autoReactiveScale` 5.0, `autoReactiveMax` 0.9. — [AMD FSR2 docs](https://gpuopen.com/manuals/fidelityfx_sdk/fidelityfx_sdk-page_techniques_super-resolution-temporal/)
- **[V]** The transparency & composition mask "adjusts the pixel history protection mechanisms" and "removes the effect of the luminance instability factor". 0 leaves locks unchanged and 1 removes locks. — [AMD FSR2 docs](https://gpuopen.com/manuals/fidelityfx_sdk/fidelityfx_sdk-page_techniques_super-resolution-temporal/)
- **[V]** History rectification: history is clamped to the current frame's YCoCg neighbourhood bounding box, and its weight is reduced in disoccluded areas and where it lies far outside the box. Disocclusion compares depth deltas against an "Akeley separation" value. "Locks" are created from 3×3 luminance analysis to protect thin features such as wires and fences. Accumulation upsamples the current frame with Lanczos and blends it into history "with a relatively low linear interpolation factor", at presentation resolution. — [AMD FSR2 docs](https://gpuopen.com/manuals/fidelityfx_sdk/fidelityfx_sdk-page_techniques_super-resolution-temporal/)
- **[V]** FSR 2.3.3 DX12 cost at **2560×1440 output**: RX 7900 XTX 0.3/0.3/0.3/0.2 ms (Quality/Balanced/Perf/Ultra-perf), RX 6800 XT 0.5/0.5/0.4/0.4 ms, RX 6700 XT 0.9/0.8/0.8/0.7 ms, RX 5700 XT 1.1/1.0/0.9/0.8 ms, Vega 56 1.9/1.7/1.5/1.2 ms, RX 590 2.3/2.1/1.9/1.7 ms. At **3840×2160**: 7900 XTX 0.7/0.6/0.6/0.5 ms, 6800 XT 1.2/1.1/1.0/0.9 ms, RX 590 5.4/4.9/4.4/3.7 ms. Figures are rounded to 0.1 ms, exclude sharpening, and are "subject to change". — [AMD FSR2 docs](https://gpuopen.com/manuals/fidelityfx_sdk/fidelityfx_sdk-page_techniques_super-resolution-temporal/)

**DLSS (2 → 4.5)**
- **[V, secondary]** The DLSS guide treats per-pixel MVs as core to the algorithm and lists objects with missing MVs ("many particle effects") as a problem case. The mask that marks problem pixels so DLSS "uses an alternate technique" (Bias Current Color) should match colour-input resolution. The 310.1.0 guide says R8G8B8A8_UNORM, while Unity's docs prefer R8_UNORM. — [DLSS Programming Guide 310.1.0 (raw PDF)](https://raw.githubusercontent.com/NVIDIA/DLSS/d489595a8c5cb19c7a8d770b0cb5bc198cc99018/doc/DLSS_Programming_Guide_Release.pdf); [Unity DLSSTextureTable](https://docs.unity3d.com/ScriptReference/NVIDIA.DLSSTextureTable.html)
- **[V, secondary; unconfirmed]** A third-party GitHub issue (Oct 2026) quotes DLSS guide 310.4.0 §3.15 as saying Bias Current Color "has not been introduced for use with latest models and should be avoided", and reports that the hint is ignored under the default transformer model. If this is true, current DLSS gives integrators no per-pixel lever for particles other than correct MVs or compositing after the upscaler. — [Eolan/forge-engine issue #360](https://github.com/Eolan/forge-engine/issues/360)
- **[V, vendor claim]** NVIDIA says the DLSS 4 transformer has "four times the computations and twice the number of parameters" of the CNN in a similar frame budget, with improved disocclusion handling and temporal stability. — [NVIDIA ADLR DLSS 4 page](https://research.nvidia.com/labs/adlr/DLSS4/)
- **[V, third-party blog measurement]** On an RTX 4090 at 4K Performance in Horizon Zero Dawn: DLSS 3.7 CNN 0.51 ms, DLSS 4 transformer (preset K) 1.06 ms, DLSS 4.5 "Transformer v2" (preset M) 1.39 ms. — [Opinali, Medium](https://medium.com/@opinali/dlss-strikes-again-transformer-v2-04e44e63d5fc)
- **[V, review]** The transformer costs about 3–5% FPS on an RTX 4080. On the RTX 30 series, transformer SR+RR was about 25% slower than CNN SR+RR. — [TechPowerUp DLSS 4 review](https://www.techpowerup.com/review/nvidia-dlss-4-transformers-image-quality/)
- **[PK – not re-verified]** The DLSS 2 architecture (GTC 2020, Edward Liu) is a convolutional autoencoder. Inputs: jittered low-res colour, MVs, depth, exposure, and the previous high-res output warped by the MVs. The network outputs the reconstruction directly, replacing hand-tuned clamping heuristics. — [GTC 2020 talk S22698](https://www.nvidia.com/en-us/on-demand/session/gtcspring20-s22698/)
- **[PK – not re-verified]** DLSS 3.5 Ray Reconstruction replaces the hand-tuned denoisers with one network trained on path-traced data. Reviews widely reported smearing, ghosting and an "oil-paint" look in motion with the CNN version. The transformer RR was marketed as fixing much of this. — [NVIDIA DLSS 3.5 article](https://www.nvidia.com/en-us/geforce/news/nvidia-dlss-3-5-ray-reconstruction/)

**XeSS, TSR, FSR 4, mobile**
- **[PK – not re-verified]** XeSS takes the same inputs as FSR 2 / DLSS (jittered colour, MVs, depth, optional exposure) and has an optional "responsive pixel mask" with the same semantics as FSR's reactive mask. It runs an XMX path on Arc and a slower DP4a path on other GPUs. — [Intel XeSS SDK on GitHub](https://github.com/intel/xess)
- **[PK – not re-verified]** UE5 TSR handles translucency through the material flag "Output Velocity" (translucent materials write MVs), the "Responsive AA" stencil, and the "Has Pixel Animation" flag. These either supply MVs or make TSR reject history more aggressively. — [UE TSR documentation](https://dev.epicgames.com/documentation/en-us/unreal-engine/temporal-super-resolution-in-unreal-engine)
- **[PK – not re-verified]** FSR 4 (March 2025) is ML-based and was initially limited to RDNA 4 (FP8). AMD has not published its architecture or per-GPU ms tables publicly to the same level of detail as FSR 2. — [AMD GPUOpen FSR page](https://gpuopen.com/fidelityfx-super-resolution-3/)
- **[PK – not re-verified]** Arm ASR is a mobile-optimised fork of FSR 2. Arm announced "Neural Super Sampling" (2025) for 2026 Mali GPUs with neural accelerators. Qualcomm Snapdragon GSR is single-pass spatial, with GSR 2 temporal. No independent ms numbers were found.

**Academic neural supersampling**
- **[V]** Xiao et al. 2020 (Facebook Reality Labs, SIGGRAPH 2020) do 4×4 (16×) upsampling, e.g. 960×540 → 3840×2160. Inputs: low-res colour, depth and dense MVs. The network is a "novel temporal network design" over several past frames. — [FB Research blog](https://research.facebook.com/blog/2020/7/introducing-neural-supersampling-for-real-time-rendering/); [publication page](https://research.facebook.com/publications/neural-supersampling-for-real-time-rendering/)
- **[PK – not re-verified]** Xiao et al. architecture: per-frame feature extraction on the current frame plus 4 previous frames, zero-upsampling, backward warping of previous frames by MVs into the current frame, a learned per-pixel feature-reweighting module that down-weights mis-warped history, and a U-Net reconstruction. The researcher recalls about 18–25 ms at 1080p target on a Titan V with TensorRT FP16, i.e. **not** within 1–3 ms. Check this against the paper's timing table.
- **[V]** STSS (He et al. 2023, arXiv 2312.10890) does joint super-sampling and frame extrapolation, 540p→1080p, with about 0.4 M parameters (417 K). Network time is 4.35 ms on an RTX 3090 at FP16, inside an 11.51 ms total pipeline, versus about 17 ms for the earlier two-stage ExtraNet+NSRR pipelines. Disocclusion holes are treated as "reshading regions", trained with random rectangle masks, and handled by masked local attention so invalid pixels contribute zero. Transparency, particles and volumetrics are not mentioned. — [arXiv 2312.10890](https://arxiv.org/html/2312.10890v1)
- **[V, link only]** Related follow-ups: Li et al. CVPR 2024 "Neural Super-Resolution for Real-time Rendering with Radiance Demodulation" (upscales demodulated lighting separately from texture). — [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Neural_Super-Resolution_for_Real-time_Rendering_with_Radiance_Demodulation_CVPR_2024_paper.pdf)

### Inferences
- The industry consensus for smoke and fire is: **do not reproject by geometry MVs; mark the pixels reactive and let the current frame dominate.** This is consistent with the team's finding. Smoke inside an upscaler gets almost no temporal super-resolution, only spatial upscaling plus whatever stability the clamp gives.
- Production upscalers reach 1–3 ms by (a) running at most a few conv layers or an analytic pass at presentation resolution, (b) keeping a single recurrent history buffer instead of stacking N past frames (Xiao-style multi-frame stacks cost 10–25 ms), and (c) using a low-res feature backbone with a cheap full-res final stage. A smoke detail net should copy this: one recurrent history, with most compute at sim-render resolution.
- Ghosting and smearing on particles (DLSS RR CNN, FSR without a reactive mask) is the documented failure mode. Its cause is history trusted under wrong MVs. The other failure mode, shimmer, comes from history rejected every frame. For smoke the right compromise is history weighted by a confidence derived from *appearance* consistency, not from MV validity alone.

### Gaps
- No exact DLSS execution-time table (ms per GPU and resolution) was verified. The PDF was not parseable in this session.
- No primary documentation describes how DLSS 4 or FSR 4 internally handle alpha-blended content, and no vendor document explicitly discusses smoke or fire failure cases.
- FSR 4, XeSS 2, PSSR and Arm NSS ms costs at 1440p are unverified or unpublished.
- The Xiao et al. 2020 runtime table was not read.

---

## Q2. Motion estimation for content without valid motion vectors (optical-flow frame generation, learned flow, volumetric reprojection in production)

### Takeaway
DLSS 3 frame generation pairs game MVs with a hardware optical-flow field precisely because MVs miss shading-driven and particle motion. FSR 3 uses compute-shader optical flow. Neither documents smoke-specific handling, and reviewers and guides attribute smoke/fire FG artifacts to "randomized" motion. In production volumetrics, reprojection is camera-only, either per froxel or by a representative depth. The medium's own velocity is generally not used. The team's observation is consistent with this practice.

### Cited Findings
- **[V, review]** DLSS 3 FG uses dedicated optical-flow hardware (the Ada OFA) plus a neural network to synthesise intermediate frames. FSR 3 uses an algorithmic (compute) optical flow with lower overhead, giving higher FPS gains from the same base frame rate. — [TechSpot FSR 3 analysis](https://www.techspot.com/article/2747-amd-fsr-3-tech/); [TechSpot DLSS 3](https://www.techspot.com/article/2546-dlss-3/)
- **[V, review]** DLSS 3 FG initially garbled UI because it reconstructs the whole image. FSR 3 can decouple UI from the FG loop. AMD recommends a base of at least 60 fps because interpolation artifacts grow at low frame rates. — [TechSpot FSR 3 analysis](https://www.techspot.com/article/2747-amd-fsr-3-tech/)
- **[V, weak source]** A general guide claims smoke and fire generate FG artifacts "because their movement is randomized". This is not a measured study. — [bottleneckcalculator FG article](https://bottleneckcalculator.us.com/knowledge-base/gaming-performance/frame-generation-impact/)
- **[V, Streamline doc link]** The DLSS-G (FG) programming guide is in the NVIDIA Streamline repo. Its inputs are depth, MVs and HUD-less colour. — [Streamline ProgrammingGuideDLSS_G.md](https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuideDLSS_G.md)
- **[V, secondary]** Froxel fog reprojection jitters the sample position along each froxel's depth and blends with a reprojected history buffer, which removes slice banding. Newly revealed froxels have no history and are the noisiest. — [Mattias Strand, Volumetric Fog](https://www.mattiasstrand.com/posts/volumetric-fog/); [Traverse blog, newly revealed froxels](https://traverse.nl/blog/shadowing-newly-revealed-froxels)
- **[PK – not re-verified]** Wronski (SIGGRAPH 2014, AC4) and Hillaire (SIGGRAPH 2015, "Physically Based and Unified Volumetric Rendering in Frostbite") reproject froxel *scattering volumes* by camera motion only, and blend with an exponential moving average (about 0.9–0.95 history weight) plus jitter. — [Hillaire 2015 slides, Frostbite](https://www.ea.com/frostbite/news/physically-based-unified-volumetric-rendering-in-frostbite)
- **[PK – not re-verified]** Cloud renderers (Nubis, Frostbite clouds) reproject by a representative cloud depth. For distant clouds this is effectively infinite depth plus wind offset. The ray-marched result is treated as quasi-static between frames.

### Inferences
- For problem (1), **compute "apparent" image-space motion instead of fluid velocity.** Options, roughly from cheapest to most expensive:
  - (a) Camera-only reprojection at a transmittance-weighted representative depth (depth where accumulated opacity crosses 0.5, or the opacity-weighted mean depth). This is the production cloud/fog practice and works because the plume shape is quasi-steady.
  - (b) Optical flow on the *low-res rendered smoke frames*, e.g. a tiny PWC/RAFT-style net or hardware OFA. This captures the motion of the visible shape, not the gas.
  - (c) Velocity of the density *iso-surface*, i.e. ∂ρ/∂t, which is near zero for a steady plume, rather than the velocity of the gas.
- Recommended framing: reproject the *volume* (or the representative depth) under camera motion only. Let the network handle residual shape change through a learned confidence or blend mask, as DLSS 2 and Xiao et al. do.

### Gaps
- No source found that measures DLSS-FG or FSR-FG on smoke or fire specifically.
- No primary-source description found of per-pixel motion-vector generation for rendered fluid volumes in a shipping game.

---

## Q3. Temporal accumulation for volumetrics (Nubis, Frostbite, UE clouds and fog, checkerboard, low-res plus bilateral upsampling): designs and costs

### Takeaway
Production volumetrics amortise ray-marching over time: quarter or lower resolution, about 1/16 of pixels updated per frame, jittered start offsets, reprojection plus history clamping, and a depth-aware upsample to full resolution. The total cost is about 1–2 ms on PS4-class hardware. This is the standard recipe the team's low-res smoke render can copy before any neural stage.

### Cited Findings
- **[V, secondary]** Guerrilla's Nubis updates 1 pixel in 16 per frame. The original PS4 implementation cost about 2 ms for the entire sky. Both figures come from secondary write-ups; check them against the Schneider slides. — [Cinevva blog, volumetric clouds](https://app.cinevva.com/blog/2026-05-04-volumetric-clouds-and-weather); [bevy issue #18151](https://github.com/bevyengine/bevy/issues/18151)
- **[PK – not re-verified]** Schneider and Vos, "The Real-Time Volumetric Cloudscapes of Horizon Zero Dawn" (SIGGRAPH 2015 Advances, pre-2018 but the canonical source), plus Schneider, "Nubis: Authoring Real-Time Volumetric Cloudscapes with the Decima Engine" (SIGGRAPH 2017). The later "Nubis^3" voxel clouds for Horizon Forbidden West (SIGGRAPH 2022/2023 Advances) moved to voxel-grid clouds with similar temporal amortisation. — [Advances in Real-Time Rendering course site](https://advances.realtimerendering.com/)
- **[PK – not re-verified]** Hillaire, "Physically Based Sky, Atmosphere and Cloud Rendering in Frostbite" (SIGGRAPH 2016 PBR course): clouds are ray-marched at reduced resolution with temporal reprojection and blue-noise or Bayer jitter. — [Frostbite publications](https://www.ea.com/frostbite/news/physically-based-sky-atmosphere-and-cloud-rendering)
- **[PK – not re-verified]** UE5 volumetric clouds: tracing at half or quarter resolution with temporal reconstruction (`r.VolumetricRenderTarget.*` cvars), then a depth-aware upsample. UE volumetric fog: a froxel grid (`r.VolumetricFog.GridPixelSize`, `GridSizeZ`) with temporal reprojection on by default. — [UE Volumetric Clouds documentation](https://dev.epicgames.com/documentation/en-us/unreal-engine/volumetric-cloud-component-properties-in-unreal-engine)
- **[V, secondary]** A practical engine setup uses a quarter-resolution ray-march with temporal reprojection "from day one" under a hard budget. No ms figure is given. — [kooch issue #731](https://github.com/lobinuxsoft/kooch/issues/731)

### Inferences
- The team can get a "free" quality gain without a neural net: ray-march the 64^3 volume at 1/2–1/4 resolution with per-frame jitter of the ray start and light-sample offsets. Accumulate in history reprojected by camera-only motion at the representative depth, clamp in YCoCg against the current neighbourhood, and upsample with depth/alpha-aware (joint bilateral) weights. The neural stage then only adds sub-grid detail on a stable base.
- Clamping and reactive logic designed for opaque surfaces over-rejects history on semi-transparent media with low local contrast. An *alpha- or transmittance-aware clamp* is a likely needed modification.

### Gaps
- No primary-source ms breakdown was found for the reprojection or upsample stages themselves. Primary slide decks were not fetched this session.
- No source was found for checkerboard rendering applied specifically to volumetrics with costs.

---

## Q4. Neural denoising and reconstruction for path-traced volumes / participating media (KPCN, OIDN, real-time volume denoisers, NRC, ReSTIR volumes)

### Takeaway
Real-time denoising and reconstruction of volumes exists in research. The most production-relevant pieces are ReSTIR-style resampling for volumes and DLSS Ray Reconstruction as a learned denoiser. OIDN and KPCN-class denoisers are interactive or offline-oriented, not 1–3 ms at 1440p on mid-range GPUs. Neural radiance caching uses tiny fully-fused MLPs, which is the most relevant precedent for per-sample MLP cost.

### Cited Findings
- **[PK – not re-verified]** Lin, Wyman and Yuksel, "Fast Volume Rendering with Spatiotemporal Reservoir Resampling" (SIGGRAPH Asia 2021): ReSTIR for heterogeneous volumes (explosions, clouds) at interactive or real-time rates with temporal and spatial reuse. — [NVIDIA Research page](https://research.nvidia.com/publication/2021-11_fast-volume-rendering-spatiotemporal-reservoir-resampling)
- **[PK – not re-verified]** Müller et al., "Real-time Neural Radiance Caching for Path Tracing" (SIGGRAPH 2021): a tiny MLP (about 7 layers × 64 neurons) via tiny-cuda-nn fully fused kernels, trained online every frame. The researcher recalls a few ms per frame at 1080p on an RTX 3090. — [NVIDIA Research NRC](https://research.nvidia.com/publication/2021-06_real-time-neural-radiance-caching-path-tracing)
- **[PK – not re-verified]** Iglesias-Guitian et al., "Real-time denoising of volumetric path tracing for direct volume rendering" (IEEE TVCG 2020) is a real-time volume-specific denoiser. Hofmann et al. 2020 cover neural denoising for medical volumetric path tracing. — [IEEE TVCG](https://ieeexplore.ieee.org/document/9123563)
- **[PK – not re-verified]** Intel Open Image Denoise is a U-Net that runs at interactive rates on GPU (OIDN 2.x). At 1440p on consumer GPUs it is typically well above 3 ms, so it is unsuitable for a 1–3 ms budget. — [OIDN](https://www.openimagedenoise.org/)

### Inferences
- The team's renderer is not noisy (deterministic ray-march of a 64^3 grid), so denoisers are mostly irrelevant *unless* they switch to stochastic ray-marching (jitter plus fewer steps). In that case, temporal accumulation (Q3) is the cheap denoiser and a learned denoiser is overkill.
- KPCN-style *kernel prediction* (predict per-pixel filter weights rather than colours) is a robust design pattern for the detail net. Outputs are bounded combinations of inputs, which limits hallucinated flicker. The catch is that KPCN cannot add frequency content that is not in its inputs.

### Gaps
- No ms numbers were verified this session for volume denoisers, NRC or ReSTIR-volumes.
- No DLSS Ray Reconstruction documentation on participating media or volumetric fog was found.

---

## Q5. Tiny-budget image-to-image detail synthesis and SR (ESPCN/FSRCNN, re-parameterised nets, edge-SR, NTIRE efficient SR, GAN/perceptual training, one-step diffusion)

### Takeaway
Efficient SR nets with about 0.1–0.4 M parameters run in roughly 10–15 ms for ×4 SR on an RTX 3090 at challenge resolutions. Production-relevant budgets need even smaller nets (ESPCN/FSRCNN/edge-SR scale, a few thousand to tens of thousands of parameters) or nets operating mostly at low resolution with pixel-shuffle output. Adversarial or perceptual training is what *adds* plausible detail instead of sharpening edges. Temporal GAN work on smoke (tempoGAN) and video (TecoGAN) directly addresses the team's problem (2), that flicker metrics reward blur. Diffusion SR, even one-step, is far from 1–3 ms.

### Cited Findings
- **[V]** NTIRE 2024 Efficient SR challenge: ×4 SR, 262 registrants, 34 valid submissions. Runtime was measured on an RTX 3090 and was the primary ranking metric. The RLFN baseline has 0.317 M parameters and runs in 13.54 ms on the validation/test sets. The per-team winner table was not extracted. — [NTIRE 2024 ESR report, arXiv 2404.10343](https://arxiv.org/abs/2404.10343); [CVPRW PDF](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Ren_The_Ninth_NTIRE_2024_Efficient_Super-Resolution_Challenge_Report_CVPRW_2024_paper.pdf)
- **[V, link only]** NTIRE 2022 ESR report and NTIRE 2026 (11th) ESR report also exist. — [arXiv 2205.05675](https://arxiv.org/pdf/2205.05675); [arXiv 2604.03198](https://arxiv.org/pdf/2604.03198)
- **[PK – not re-verified]** The core efficient-SR architectures:
  - ESPCN (Shi et al. 2016): sub-pixel convolution / pixel shuffle, with all convs at low resolution.
  - FSRCNN (Dong et al. 2016): shrink, map, expand, deconvolution.
  - ECBSR (Zhang et al., ACM MM 2021): edge-oriented convolution blocks that structurally re-parameterise into plain 3×3 convs at inference, aimed at mobile real time.
  - RepVGG (Ding et al., CVPR 2021): the re-parameterisation principle.
  - edge-SR (Michelini et al., CVPRW 2022): one-layer SR for edge devices.
  - [ESPCN arXiv 1609.05158](https://arxiv.org/abs/1609.05158); [ECBSR](https://github.com/xindongzhang/ECBSR); [edge-SR arXiv 2108.10335](https://arxiv.org/abs/2108.10335)
- **[PK – not re-verified]** The Mobile AI (MAI) 2021/2022 real-time SR challenges (Ignatov et al., CVPRW) target ×3 SR to 1080p on mobile NPUs. Winners ran in the tens of ms with tiny (a few-layer) INT8 networks. — [MAI 2022 report arXiv 2211.05910](https://arxiv.org/abs/2211.05910)
- **[PK – not re-verified]** Real-ESRGAN (Wang et al., ICCVW 2021) uses high-order degradation modelling plus GAN and perceptual losses. It is relevant as a *training recipe*, synthesising LR/HR pairs by degrading HR smoke renders. The network itself (RRDB, about 16 M parameters) is far too large. — [arXiv 2107.10833](https://arxiv.org/abs/2107.10833)
- **[PK – not re-verified, highly relevant]** tempoGAN (Xie, Franz, Chu, Thuerey, SIGGRAPH 2018) super-resolves 3D smoke *density fields* by 4× using a spatial discriminator plus a temporal discriminator fed advected (warped) consecutive frames. Low-res velocity is an input. This is the direct precedent for sub-grid smoke detail synthesis with temporal coherence. — [arXiv 1801.09710](https://arxiv.org/abs/1801.09710)
- **[PK – not re-verified, highly relevant to problem (2)]** TecoGAN (Chu et al., SIGGRAPH 2020) proposes the tOF and tLP temporal metrics and a "Ping-Pong" loss. It explicitly shows that pure temporal-difference or warping-error metrics are minimised by blurry results, and proposes LPIPS-based temporal metrics (tLP) and spatio-temporal discriminators that reward *consistent detail* rather than absent detail. — [arXiv 1811.09393](https://arxiv.org/abs/1811.09393)
- **[PK – not re-verified]** Transport-based neural style transfer for smoke (Kim et al., SIGGRAPH Asia 2019) stylises smoke volumes with temporal coherence by advecting style through the velocity field. It is offline. — [arXiv 1905.07442](https://arxiv.org/abs/1905.07442)
- **[PK – not re-verified]** One-step diffusion SR (e.g. OSEDiff, SinSR, adversarial-diffusion-distilled SR, 2024) still costs on the order of 100 ms or more per 512² image on data-centre GPUs. It is not viable at 1440p in 1–3 ms. — [OSEDiff arXiv 2406.08177](https://arxiv.org/abs/2406.08177)

### Inferences
- **For problem (2):** replace raw flicker metrics (frame-to-frame difference or warp error) with a pair of metrics. One is tLP/tOF-style, i.e. a temporal change in LPIPS or optical flow compared *against ground truth's* temporal change. The other is a sharpness or detail term (high-frequency energy relative to the reference). Then a blurry output is penalised for lacking the reference's temporal *and* spatial detail. Train with a temporal discriminator on triplets warped by apparent motion (tempoGAN / TecoGAN).
- Architecture: do the work at sim-render resolution (or ½ of 1440p), use 3×3 re-parameterisable convs with ReLU, and finish with a pixel-shuffle to 1440p. Budget: a 1440p output with about 8–16 channels at quarter resolution is in the same compute class as DLSS CNN (about 0.5 ms on a 4090 at 4K). The team must benchmark its own GPU.
- Adding a recurrent input (the previous output warped by apparent motion, plus a learned confidence) is what turns "edge sharpening" into stable detail accumulation.

### Gaps
- Exact NTIRE 2024 winner runtime and parameters were not extracted.
- No published real-time neural SR result specific to rendered volumetrics (screen space) was found.

---

## Q6. Neural textures, neural materials and in-shader MLPs: could a tiny per-sample MLP synthesise sub-grid smoke detail?

### Takeaway
NVIDIA has shown that small MLP decoders can run inline in real-time shaders: neural texture compression (Vaidyanathan et al. 2023, now RTX Neural Texture Compression) and real-time neural appearance models (Zeltner et al. 2023/24). Both rely on tensor-core cooperative vectors and evaluate the MLP roughly once per shaded pixel. A ray-marcher evaluates 32–128 samples per pixel, which multiplies that cost by one to two orders of magnitude. Per-sample MLP detail is therefore only plausible with very few march samples, small MLPs (e.g. 2 hidden layers × 16–32), or evaluation on a coarse froxel or brick grid that is cached and reused.

### Cited Findings
- **[V]** Vaidyanathan et al. 2023, "Random-Access Neural Compression of Material Textures": a small per-material MLP jointly compresses multiple material textures and their mip chains. Authors claim "16x more texels" at low bitrate versus AVIF/JPEG XL quality, and real-time random-access decompression like block compression. — [arXiv 2305.17105](https://arxiv.org/abs/2305.17105)
- **[V]** Zeltner et al., "Real-Time Neural Appearance Models" (arXiv 2023; ACM TOG 2024): learned hierarchical latent textures decoded by small neural decoders inlined into a real-time path tracer. §8.2 reports shading timings at 1 spp 1920×1080 (exact numbers not extracted). The paper cites prior neural-BRDF work at about 5 ms for one BRDF evaluation per pixel at 1080p on an RTX 2080 Ti. — [arXiv 2305.02678](https://arxiv.org/abs/2305.02678); [ACM TOG](https://dl.acm.org/doi/10.1145/3659577); [NVlabs code](https://github.com/NVlabs/neuralappearance)
- **[PK – not re-verified]** NVIDIA RTX Neural Texture Compression SDK (2025) and DirectX "Cooperative Vectors" (Shader Model 6.9 preview, 2025) expose tensor-core matrix-vector ops in HLSL, which is how in-shader MLPs reach practical speed. — [NVIDIA RTX Neural Texture Compression on GitHub](https://github.com/NVIDIA-RTX/RTXNTC)

### Inferences
- Rough cost model: 1440p is about 3.7 M pixels. At 64 march samples per pixel that is about 236 M MLP evaluations per frame. Even with tensor cores, a 2×32 MLP at that rate is likely several ms. **Cheaper alternative:** evaluate the MLP (or a neural or procedural noise decoder) on a mid-resolution detail grid around the plume (e.g. 128^3–256^3 bricks only where density > 0), advected with the sim velocity, then sample it trilinearly during ray-marching. This gives object-space, view-consistent detail that is temporally stable by construction. It sidesteps problem (1) entirely, because detail moves with the gas while the screen shows the integrated result.
- Object-space detail cannot "hallucinate" view-dependent sharpening of edges. A hybrid is likely best: object-space advected detail (stable) plus a light screen-space recurrent net (edges, anti-aliasing, upsampling).
- Classic texture/noise advection with dual-phase blending (Neyret 2003; pre-2018) handles detail stretching over time. A learned version is close to tempoGAN's setup.

### Gaps
- Exact NTC decode costs (ms at resolution and GPU) and Zeltner §8.2 numbers were not extracted.
- No published work was found that evaluates a neural field per ray-march sample for smoke in real time.

---

## Summary table: real-time viability and slot in a "low-res smoke render + detail" pipeline

| Technique | Real time on consumer GPU? | Where it slots | Main risk for smoke/fire |
|---|---|---|---|
| FSR 2-style analytic TAAU | Yes, 0.3–1.2 ms at 1440p **[V]** | After the low-res volume render: jitter + reproject + clamp | Clamp rejects history on low-contrast media; MVs wrong for gas |
| DLSS / XeSS / FSR 4 (ML TAAU) | Yes, about 0.5–1.4 ms at 4K on a 4090 **[V, third-party]** | Whole-frame upscaler, with smoke as reactive or composited after | Black box; particle hint deprecated (unconfirmed); ghosting |
| Xiao 2020-style multi-frame neural SR | No, roughly 4–25 ms **[V for STSS 4.35 ms 3090 1080p]** | Replace TAAU | Too expensive; MV-dependent |
| Optical flow (OFA / compute) | Yes (FG uses it every frame) | Compute apparent motion of the smoke image | Flow on semi-transparent layers is ambiguous |
| Volumetric temporal amortisation (Nubis/Frostbite/UE) | Yes, about 2 ms total sky on PS4 **[V, secondary]** | Inside the ray-marcher | Needs representative-depth reprojection |
| Efficient SR CNN (ESPCN/ECBSR scale) | Yes if tiny; NTIRE-class 0.3 M params about 13.5 ms (3090, ×4 challenge res) **[V]** | Screen-space detail stage | Sharpening, not detail, without GAN/perceptual loss |
| tempoGAN / TecoGAN training | Training-time only | Loss and metric design for the detail net | GAN instability; needs paired data |
| In-shader MLP (NTC / neural appearance) | Per-pixel yes; per march sample doubtful | Object-space detail grid | Cost × samples per pixel |
| One-step diffusion SR | No (100 ms or more) | n/a | n/a |
