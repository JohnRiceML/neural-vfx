# Fluid Detail Synthesis: Learned and Procedural Super-Resolution for Coarse Smoke/Fire Simulations (2018–2026)

Context for the reader: the target is a real-time pipeline with a 64^3 classical GPU solver plus a learned detail stage. The detail stage gets 2.5 ms of GPU time on an RTX 5080 at 1440p. The team has already tried two things: (a) a tempoGAN-style volumetric 4x upscale from 64^3 to 256^3, which was too slow unless sparse, and (b) a screen-space 2D detail network trained with L1, gradient and spectrum losses on guided 256^3/64^3 pairs, which sharpens edges but produces no wisps. Adversarial training of (b) is in progress.

Research budget note: these notes come from about 20 search and fetch calls. Primary papers were read through ar5iv HTML for tempoGAN, Multi-Pass GAN and Bai et al. Most other items were verified only at abstract or project-page level, and are marked that way.

---

## 1. tempoGAN, Multi-Pass GAN, dictionary/patch smoke upsampling and successors

### Takeaway
None of the published volumetric GAN or dictionary upsamplers is close to real time. Measured inference ranges from about 1 s to 60 s per frame on GTX 1080 Ti or TITAN X class GPUs, for 200^3 to 512^3 outputs. Two of their design choices are still worth copying. The first is a temporal discriminator that sees advection-aligned frame triplets. The second is a feature-space loss computed in the discriminator's layers. The pairing question also matters. tempoGAN and Multi-Pass GAN train on downsampled high-res data, while Bai et al. train on independently simulated coarse/fine pairs, and they argue that this choice determines whether the result generalizes to a real coarse solver.

### Cited Findings

**tempoGAN (Xie, Franz, Chu, Thuerey, SIGGRAPH 2018 / ACM TOG 37(4))**
- **Generator:** Fully convolutional with residual blocks. The ResNet variant "gave slightly sharper results" than a U-Net. The first two layers use nearest-neighbour interpolation rather than deconvolution. The 8 conv layers give a receptive field of about 16 cells. — [ar5iv tempoGAN](https://ar5iv.labs.arxiv.org/html/1801.09710)
- **Discriminators:** Each has 4 conv layers with leaky ReLU (slope 0.2) and a fully connected layer. The spatial discriminator D_s is conditioned on the low-res input x. The temporal discriminator D_t is unconditional. — [ar5iv tempoGAN](https://ar5iv.labs.arxiv.org/html/1801.09710)
- **Temporal discriminator inputs:** D_t sees three frames, with the neighbours advected onto frame t: {A(y^{t-1}, v^{t-1}), y^t, A(y^{t+1}, -v^{t+1})}. A is semi-Lagrangian advection using the low-res velocity. Fake triplets are built the same way from G's outputs. — [ar5iv tempoGAN](https://ar5iv.labs.arxiv.org/html/1801.09710)
- **Other losses:** A feature-space loss L_f = Σ_j λ_f^j ||F^j(G(x)) − F^j(y)||² uses D_s's intermediate features, and negative λ_f "also gave good results". A small-weight L1 term is added. A plain L2 temporal loss was rejected because it reduced overall density. — [ar5iv tempoGAN](https://ar5iv.labs.arxiv.org/html/1801.09710)
- **Inputs:** Density plus velocity (ρ, v). Vorticity gave only subtle gains. Noise-field inputs were ignored by the trained network. — [ar5iv tempoGAN](https://ar5iv.labs.arxiv.org/html/1801.09710)
- **Data pairing:** About 20 randomized Mantaflow smoke sims of 120 frames each. Low-res inputs are high-res frames downsampled ("typically by a factor of 4"), not separately simulated. Augmentation is scaling by 0.85–1.15 and rotation by ±90°. — [ar5iv tempoGAN](https://ar5iv.labs.arxiv.org/html/1801.09710)
- **Hardware and training cost:** GTX 1080 Ti (11 GB), TensorFlow. 2D training takes about 14 h on 1 GPU. 3D training takes about 9 days on 2 GPUs. — [ar5iv tempoGAN](https://ar5iv.labs.arxiv.org/html/1801.09710)
- **Inference timings, 1080 Ti:**
  - A 34^3 input tile to a 136^3 output tile takes about 2.2 s.
  - A 64^3 input to a 256^3 output takes 17.9 s on average.
  - For the 150×100×100 → 600×400×400 scene, all tiles take about 3.9 min, versus an estimated 31.5 min/frame for a full high-res sim.
  - Tiles overlap by about 3 cells because of GPU memory limits.
  - — [ar5iv tempoGAN](https://ar5iv.labs.arxiv.org/html/1801.09710)
- **Stated limitations:**
  - The upscale factor is fixed (4x).
  - It "can also be slower than … calculating the advection for a high resolution grid".
  - It was tested only on buoyant smoke.
  - Applying it recursively amplifies features.
  - Feature-loss weights depend on the data. The wavelet-turbulence dataset needed 2x the data and different weights.
  - — [ar5iv tempoGAN](https://ar5iv.labs.arxiv.org/html/1801.09710)

**Multi-Pass GAN (Werhahn, Xie, Chu, Thuerey, SCA 2019 / PACMCGIT)**
- **Architecture:** Two 2D GANs on slices.
  - G1 first linearly interpolates the input along Z, then upsamples XY slices 4x.
  - G2 refines YZ slices at full resolution.
  - A third XZ pass (G3) was found largely redundant.
  - Factors are 4x (tempoGAN-style losses) and 8x (progressive growing, WGAN-GP with λ_W = 10, λ_L1 = 20, no feature loss).
  - — [ar5iv Multi-Pass GAN](https://ar5iv.labs.arxiv.org/html/1906.01689)
- **Data:** 20 Mantaflow sims of 120 frames. The 64^3 inputs are downscaled from 256^3 or 512^3 references. — [ar5iv Multi-Pass GAN](https://ar5iv.labs.arxiv.org/html/1906.01689)
- **Timings (single GTX 1080 Ti, average over 100 frames):**

  | Method | 256^3 output (s) | 512^3 output (s) |
  |---|---|---|
  | Multi-Pass GAN | 10.14 | 59.65 (G1 7.58 s + G2 52.07 s) |
  | CPU regular solver | 116.90 | 1376.54 |
  | CPU multigrid solver | 41.90 | 463.81 |

  The paper notes that the solver times are CPU-based, so the comparison is not like for like. — [ar5iv Multi-Pass GAN](https://ar5iv.labs.arxiv.org/html/1906.01689)
- **Memory:** tempoGAN keeps up to 128 latent features per voxel, which forces tiling. The 2D slice networks fit full 256^3 slices without tiling. — [ar5iv Multi-Pass GAN](https://ar5iv.labs.arxiv.org/html/1906.01689)
- **Failure modes:**
  - At 8x, step-like artifacts from the low-res sampling appear.
  - Artifacts occur when input velocities fall outside the training range (rescaling velocities to the training time step helps).
  - On smooth, slow inputs the network adds few details but can create spurious density gradients at boundaries.
  - There were no obstacles in the training data.
  - Training took about 3 + 2 days (4x) and about 8 + 5 days (8x) on one GPU.
  - — [ar5iv Multi-Pass GAN](https://ar5iv.labs.arxiv.org/html/1906.01689)

**Dictionary-based smoke upsampling (Bai, Li, Desbrun, Liu, ACM TOG 40(1), 2020)**
- **Method:**
  - A LISTA-style learned sparse-coding network encodes coarse velocity patches (5^3) over a dictionary.
  - Coarse patches carry velocity and vorticity from the last 3 steps.
  - Output is the enlarged coarse patch plus a sparse combination of learned high-res residual atoms. Overlapping patches are blended with a 4D Gaussian.
  - The loss has velocity, Sobolev-gradient and divergence terms.
  - — [ar5iv Bai et al.](https://ar5iv.labs.arxiv.org/html/1910.09166)
- **Pairing:** Coarse and fine are separately simulated (mostly with the kinetic solver of Li et al. 2018), not downsampled. Patches are importance-sampled by density and strain rate, with 3M–22M training patches. The authors argue that downsampled-HR training (as in tempoGAN) is unrealistic because turbulent structures change at coarser resolution. — [ar5iv Bai et al.](https://ar5iv.labs.arxiv.org/html/1910.09166)
- **Timings:**
  - Mainly 4x per axis (8x was tried). Synthesis ran on a TITAN X (12 GB).
  - Upsampling takes 0.02–3.48 s/frame. Coarse sims take 0.02–0.12 s/frame and fine sims 0.07–75.9 s/frame, for speed-ups of 1.8–21.8x.
  - 50^3 → 200^3 takes "between 1 to 3 seconds". tempoGAN needed about 12 s versus about 3 s for this method at 200^3.
  - — [ar5iv Bai et al.](https://ar5iv.labs.arxiv.org/html/1910.09166)
- **Failure modes:**
  - Accuracy drops as Reynolds number rises.
  - Short-lived vortices are missed by importance sampling.
  - Output is sometimes crisper (less diffusive) than the real fine sim.
  - Spurious high-res wisps appear where the coarse sim has little smoke.
  - It works at a fixed ratio only.
  - The authors judge that tempoGAN looks sharper but misses the fine dynamics.
  - — [ar5iv Bai et al.](https://ar5iv.labs.arxiv.org/html/1910.09166)
- **Earlier patch method:** Bai et al. say that the nearest-neighbour patch search of Chu & Thuerey 2017 (CNN feature descriptors) restricts the synthesized structures and gives them a directional bias. — [ar5iv Bai et al.](https://ar5iv.labs.arxiv.org/html/1910.09166)

### Inferences
- **Scaling to the team's budget:** Scale tempoGAN's measured 17.9 s for 64^3 → 256^3 on a 1080 Ti linearly by throughput to an RTX 5080. Even assuming roughly 5–10x more effective FP16/tensor throughput plus TensorRT, it is still roughly 1–3 s/frame. That is 3 orders of magnitude over 2.5 ms. A 256^3 dense volumetric GAN is infeasible. Only a much smaller network evaluated on a sparse set of active bricks, or a 2x factor, could fit.
- **Most transferable ideas:** Three ideas carry over to the team's screen-space network:
  1. Advection-aligned temporal discrimination (warp the neighbouring frames with known motion before D_t).
  2. A feature-space loss taken from the discriminator.
  3. Velocity as a conditioning input.

  For a screen-space net, the "advection" would be reprojection using screen-space motion vectors from the 64^3 velocity field.
- **Why the guided pairs may lack wisps:** The team pairs a guided 256^3 sim with the 64^3 sim, which matches Bai et al.'s argument for physically simulated pairs over downsampled ones. But if guiding is strong, the 256^3 sim may itself be over-constrained and lack wisps (see Section 5). That would cap what any loss can learn.

### Gaps
- No direct successor (2021–2026) to tempoGAN, Multi-Pass GAN or Bai et al. reporting real-time 3D smoke super-resolution was found in this search. Absence is not confirmed; it was not searched exhaustively.
- Exact λ values for tempoGAN's L1 and feature terms were not captured; Appendix A was truncated.

---

## 2. Procedural upres without a network (wavelet turbulence, noise, flow-map advection) and production tools

### Takeaway
Wavelet turbulence and its descendants are the production standard for upres. Houdini Gas Upres exposes "Wavelets" and "Curl" turbulence sources. They inject curl-noise detail scaled by the energy missing from the coarse field. No pressure solve is needed, and the cost is embarrassingly parallel. Even the 2008 CPU implementation ran at about 2 min/frame for a 720×576×576 output. Flow-map, covector and neural-flow-map methods preserve vorticity much better. But they are full simulators (seconds per step), not detail post-processes, and none reports real-time numbers.

### Cited Findings
- **Wavelet turbulence (Kim, Thürey, James, Gross, SIGGRAPH 2008):** Uses the wavelet decomposition of a low-res sim to find where and how strongly high-frequency energy is missing, then synthesizes high-res detail as a post-process. There is no linear system to solve, so it parallelizes trivially. — [Cornell project page](https://www.cs.cornell.edu/~tedkim/WTURB)
- **Wavelet turbulence timings:**
  - A 720×576×576 output from an 80×64×64 input took under 2 min/frame on an 8-core workstation. — [Cornell project page](https://www.cs.cornell.edu/~tedkim/WTURB)
  - A 400^3 output from 50^3 took about 30 s/frame on a 4-core workstation. This figure was found only on a mirror of the project page. — [mirror](https://kelaenderkoe.pages.dev/patterns-http-tkim.graphics/WTURB)
- **Houdini Gas Up Res (DOP), inputs:** Requires the base sim's `vel` plus `density` (smoke) or `heat` (fire). The up-res sim develops "based on the velocity field from the base simulation" and may modify almost anything "other than the general motion (velocity vectors)". — [SideFX Gas Up Res docs](https://www.sidefx.com/docs/houdini/nodes/dop/gasupres.html)
- **Houdini Gas Up Res, turbulence sources:**
  - **Constant.**
  - **Wavelets:** "the power of the finest detail coefficients used as a multiplier for turbulence". It "will only introduce turbulence where there existed some in the coarser smoke simulation".
  - **Curl:** turbulence scaled by curl magnitude.
  - **Other controls:** an automatic frequency cutoff "that adds detail without changing the base sim's behavior", a blur radius on the amplitude to avoid velocity kinks, octaves/roughness/attenuation, a temporal frequency, and an auto-regenerated rest field so the noise tracks the smoke.
  - The docs page does not state whether pressure is re-projected.
  - — [SideFX Gas Up Res docs](https://www.sidefx.com/docs/houdini/nodes/dop/gasupres.html)
- **EmberGen upscaling (in the product since 2019):**
  - A JangaFX developer (informal forum comment) said they upscale the temperature and advected textures while keeping the initial velocities, with 2x, 3x or 4x factors. The upscaled sim can be exported as VDB. — [Reddit thread](https://reddit.birdcat.cafe/r/Simulated/comments/echd0j/embergen_upscalingupresing_tests_great_realtime/fbbcwtx/?context=3)
  - EmberGen 1.0 claimed "massive improvements" to upscaling, up to a billion voxels. — [CGChannel 2023](https://www.cgchannel.com/2023/03/jangafx-releases-embergen-1-0/)
- **EmberGen 2.0 sparse solver:**
  - About 10–15x more voxels than 1.x. A 6 GiB GPU handles about 200 M active voxels (≈1 B dense equivalent), "without any kind of upscaling". — [JangaFX roadmap](https://www.jangafx.com/roadmap)
  - Shown simulating and rendering 1 B voxels interactively on an RTX PRO 6000 Blackwell, with release expected in 2026. — [80.lv](https://80.lv/articles/upcoming-embergen-s-sparse-solver-handles-one-billion-voxels-in-real-time); [CGChannel 2025](https://www.cgchannel.com/2025/01/check-out-the-new-features-due-in-embergen-2-0)
- **Covector Fluids (Nabizadeh et al. 2022):** Introduced flow maps for advecting the covector variable, with state-of-the-art vorticity preservation. — [cited in arXiv 2405.09672 / search summary](https://arxiv.org/pdf/2405.09672)
- **Neural Flow Maps (Deng, Yu, Zhang, Wu, Zhu, SIGGRAPH Asia 2023, ACM TOG 42(6) Art. 248):**
  - Represents velocity with Spatially Sparse Neural Fields (SSNF) and marches them forward and backward for bidirectional flow maps, combined with an impulse-based solver.
  - Shown on leapfrogging vortices and vortex reconnection.
  - Code: Taichi 1.6 plus PyTorch 2.0, CUDA 11.8.
  - No per-step timing was found on the pages checked.
  - — [arXiv 2312.14635](https://arxiv.org/abs/2312.14635v1); [code](https://github.com/yitongdeng-projects/neural_flow_maps_code)
- **Particle Flow Maps (Zhou et al., SIGGRAPH 2024):** Uses particle trajectories as bidirectional flow maps on a hybrid impulse-based grid solver. Reports up to 49x less compute and 41% less memory than Neural Flow Maps, with better vorticity preservation. The abstract page gives no absolute timings. — [arXiv 2405.09672](https://arxiv.org/abs/2405.09672)
- **Vortex Particle Flow Maps (2025):** Extends vortex particles with flow maps for long-term vorticity preservation. — [arXiv 2505.21946](https://arxiv.org/pdf/2505.21946)
- **GPU follow-up:** A related GPU adaptive-grid paper reports a 1.5–2x speedup over Particle Flow Maps on the same hardware. Its authors were not confirmed. — [search summary; arXiv 2602.00499 likely](https://arxiv.org/pdf/2602.00499)
- **Fast Fluid Up-Res (Ramirez and Jones):** A SIGGRAPH talk exists under this title. Its content was not read. — [SIGGRAPH history](https://history.siggraph.org/?p=227366)

### Inferences
- **Best fit for the budget:** A GPU wavelet- or curl-noise upres fits the 2.5 ms budget comfortably. It is a handful of compute dispatches over a high-res brick set or over raymarch samples, with no solve. It matches what Houdini and, informally, EmberGen do.
  - It adds detail where the coarse sim already has energy. That is exactly the spatially plausible "wisp" placement the screen-space net lacks.
  - Its weaknesses are well known: the noise looks generic, it can "boil", and it does not create new large-scale structure.
- **Hybrid option:** Use procedural turbulence as the base. Then use a small learned network only to modulate amplitude, scale and orientation of the procedural detail per region. This is lower risk than synthesizing wisps from scratch.
- **Where flow-map methods belong:** BFECC/MacCormack-style higher-order advection is standard and cheap, a few extra advection passes on a 64^3 grid. It reduces numerical dissipation so the coarse solver keeps more of the energy that a detail stage can key on. Full flow-map methods (covector, NFM, PFM) are research simulators. Their cost relative to a 64^3 stable-fluids step is unknown and likely too high for real time. Flow maps belong in the solver stage, not the detail stage.

### Gaps
- Advection-reflection (Zehnder et al. 2018) and BFECC/MacCormack GPU costs were not verified in this search.
- No absolute per-frame timings for Neural Flow Maps or Particle Flow Maps were obtained.
- Niagara Fluids (Unreal) upres or detail methods were not researched; no Epic documentation was fetched.
- EmberGen's upscaling algorithm is documented only by an informal developer forum comment.
- Whether Houdini Gas Upres re-solves pressure was not stated on the docs page.

---

## 3. Diffusion and generative models for fluid/turbulence super-resolution (2022–2026)

### Takeaway
Diffusion and flow-matching super-resolution for turbulence is active, mostly in scientific computing (2D turbulence, flames, atmospheric downscaling). It gives better spectral fidelity than regression models. But the best distilled turbulence model found is a one-step consistency student, and even that reports only a 12x speedup over its own teacher on small 2D fields. No paper found demonstrates real-time 3D smoke super-resolution with diffusion.

### Cited Findings
- **Distilled one-step turbulence SR (May 2026):**
  - An optimal-transport flow-matching teacher is distilled into a one-step consistency model.
  - Tested on turbulent channel flow (64→192) and Kolmogorov flow (64→256).
  - The student keeps similar spectrum metrics to the teacher with about half the parameters and a 12x inference speedup.
  - Absolute latency was not obtained.
  - — [arXiv 2605.05975](https://arxiv.org/pdf/2605.05975)
- **Spectrally Decomposed Diffusion Models for Generative Turbulence Recovery (2023):** Exists. Details were not read. — [arXiv 2312.15029](https://arxiv.org/pdf/2312.15029)
- **3D flame SR with diffusion:**
  - Physically consistent diffusion models for 3D swirling-flame super-resolution: *Physics of Fluids* 36(9), 2024.
  - A "spaced" physically consistent diffusion variant: J. Comput. Phys., 2025.
  - Step counts and timings were not obtained.
  - — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S002199912500912X)
- **Diffusion SR on 3D flow fields:** One paper explicitly states that the cost of high-res 3D fields has kept diffusion super-resolution confined to low-res 3D settings. — [arXiv 2510.19971](https://arxiv.org/pdf/2510.19971)
- **CD-TVD (2025):** Contrastive diffusion for 3D super-resolution of time-varying volume data with scarce high-res data. — [arXiv 2508.08173](https://arxiv.org/pdf/2508.08173)
- **One-step image SR (not fluid-specific):**
  - CTMSR (ICCV 2025) does one-step generative SR via consistency trajectory matching. — [mlanthology](https://mlanthology.org/iccv/2025/you2025iccv-consistency/)
  - FlowSR does single-step SR via consistency rectified flow. — [arXiv 2605.12377](https://arxiv.org/html/2605.12377v1)
  - AirCast-SR (atmospheric) uses latent consistency diffusion with 4–25 steps. — [HF papers 2605.26130](https://huggingface.co/papers/2605.26130.md)
- **PDE-Refiner (Lippe et al., NeurIPS 2023):** Identifies "neglect of non-dominant spatial frequency information" as the main failure of MSE-trained neural PDE surrogates. Uses a diffusion-inspired multistep refinement to model all frequencies, and reports stable, accurate rollouts on fluid benchmarks. The number of refinement steps was not captured. — [arXiv 2308.05732](https://arxiv.org/abs/2308.05732)
- **Diffusion for smoke reconstruction:** SmokeSVD (2025) uses diffusion for single-view smoke reconstruction (reconstruction, not upsampling). — [arXiv 2507.12156](https://arxiv.org/pdf/2507.12156)

### Inferences
- **What fits the budget:** Only a one-step, distilled model could plausibly fit in 2.5 ms, and then only in screen space or on very sparse bricks. A one-step consistency or adversarial-distilled 2D image network at 1440p (possibly run at half resolution) is in the same cost class as the team's existing screen-space net. Its objective is better at hallucinating plausible high-frequency content than L1.
- **Temporal coherence:** Per-frame sampling of a generative model will flicker. Any adoption needs noise warped by motion vectors or temporal conditioning on the previous output, analogous to tempoGAN's D_t.

### Gaps
- No diffusion paper on 3D smoke super-resolution for graphics (as opposed to scientific flow) was found.
- No published absolute ms/frame for any fluid diffusion SR on consumer GPUs was found.
- Quantitative diffusion-vs-GAN comparisons on smoke were not found.

---

## 4. Latent-space / reduced-order / neural-operator simulation and learned corrections (solver-in-the-loop, ML-accelerated CFD)

### Takeaway
Learned corrections can make a coarse solver match a finer one. Kochkov et al. report 8–10x finer-resolution accuracy and 40–80x speedup, but in 2D turbulence. Solver-in-the-Loop shows stable rollouts of several hundred steps when trained through a differentiable solver. Neither has been demonstrated as a real-time 3D smoke system. Later work notes that direct corrections are prone to instabilities from perturbations. Deep Fluids-style reduced models are fast (700x vs CPU) but tied to their training parameterization.

### Cited Findings
- **Kochkov et al., PNAS 2021:**
  - Learned interpolation and correction components inside a classical solver are as accurate as baselines with 8–10x finer resolution per dimension, a 40–80x speedup.
  - For LES, 8x coarser grids gave about a 40x speedup.
  - Stable over long rollouts, and generalizes to unseen forcing and Reynolds numbers.
  - 2D only.
  - — [arXiv 2102.01010](https://arxiv.org/abs/2102.01010); [Google Research](https://research.google/pubs/machine-learning-accelerated-computational-fluid-dynamics/)
- **Solver-in-the-Loop (Um, Brand, Fei, Holl, Thuerey, NeurIPS 2020):**
  - Alternates network correction and coarse solver steps, with training through the differentiable solver.
  - Clearly outperforms supervised (pre-computed) training, "with stable rollouts of several hundred recurrent evaluation steps".
  - Examples range from advection-diffusion to 3D Navier-Stokes.
  - Code is at tum-pbs/Solver-in-the-Loop.
  - — [NeurIPS proceedings](https://papers.nips.cc/paper/2020/hash/43e4e6a6f341e00671e123714de019a8-Abstract.html); [arXiv 2007.00016](https://arxiv.org/abs/2007.00016v1)
- **Later caveat:** A 2025 Thuerey-group paper describes such "direct corrections" as improving rollout accuracy but "prone to instabilities caused by perturbations". — [arXiv 2511.12764](https://www.arxiv.org/pdf/2511.12764)
- **Deep Fluids (Kim, Azevedo, Thuerey, Kim, Gross, Solenthaler, Eurographics 2019):** A CNN generates divergence-free velocity fields from a few parameters (or a latent space). Up to 700x faster than re-simulating with the CPU solver, with up to about 1300x compression. It is limited to the parameterized scenario space it was trained on. — [arXiv 1806.02071](https://arxiv.org/pdf/1806.02071); [TUM portal](https://portal.fis.tum.de/en/publications/deep-fluids-a-generative-network-for-parameterized-fluid-simulati/)
- **PDE-Refiner:** MSE-trained surrogates neglect high-frequency, low-amplitude content, which destabilizes long rollouts. Refinement fixes this. — [arXiv 2308.05732](https://arxiv.org/abs/2308.05732)

### Inferences
- **Solver-in-the-loop for the 64^3 sim:** Corrections trained against a downsampled 256^3 sim are the most principled way to make a 64^3 sim "behave like" a finer one at large scales, for example by keeping vortices alive longer and plumes rolling up correctly.
  - The correction network runs at 64^3 (262k cells). A small 3D CNN there could plausibly cost well under 1 ms on an RTX 5080. This is an estimate, not a measurement.
  - This improves the coarse motion. It does not add sub-grid wisps, so it complements a detail stage rather than replacing one.
  - Risks: instability under perturbations (user interaction, sources outside the training distribution), and the need for a differentiable version of the team's GPU solver for training.
- **Deep Fluids / latent models:** Poorly suited to interactive, open-ended game scenarios because they are tied to their training parameterization.

### Gaps
- No 3D real-time smoke deployment of learned corrections was found.
- No measured GPU ms for a 64^3 correction network was found.
- FNO-family and "Latent Space Physics" (Wiewel et al. 2019) and subspace neural physics (Holden et al. 2019) were not researched in this pass; no sources were fetched.

---

## 5. Guided simulation / sim tracking (high-res following low-res)

### Takeaway
The standard formulation is Nielsen et al. 2009 and 2010. They couple the two resolutions variationally: the high-res velocity is projected to stay close to the low-res velocity in the low-frequency band and is free in high frequencies. The time-dependent version from 2010 is what allows more high-frequency detail to develop while keeping correspondence. Houdini's production equivalent keeps the base-sim velocity as the "general motion" and adds frequency-limited turbulence on top.

### Cited Findings
- **Nielsen et al. 2009:** Nielsen, Christensen, Bin Zafar, Roble, Museth, "Guiding of Smoke Animations Through Variational Coupling of Simulations at Different Resolutions", SCA 2009.
  - A variational formulation lets the high-res smoke adopt the low-frequency features of a low-res (or non-physical) simulation while developing higher frequencies.
  - Motivated by the fact that changing resolution can "result in dramatic changes in the behavior" of smoke.
  - — [EG digital library](https://diglib.eg.org/items/3d19a2fe-49ba-4e7e-a305-5f947d26b5ab); [author page](https://nafees.net/publications/2009-08-Variational-Smoke-Guiding/)
- **Nielsen & Christensen 2010, "Improved Variational Guiding of Smoke Animations" (CGF 29(2)):**
  - Steady guiding "often leads to a lack of sufficiently high detail".
  - Time-dependent guiding was expensive because the system matrix had to be recomputed every iteration. The new model avoids this.
  - Time-dependent guiding "allows more high frequency detail to develop without losing correspondence to the low resolution simulation".
  - — [EG digital library](https://diglib.eg.org/items/7fde3794-e5a3-4cb8-a861-379cef4121d8)
- **Houdini Gas Up Res:** Preserves "general motion (velocity vectors)" from the base sim. Turbulence is added with an automatic frequency cutoff chosen to "add detail without changing the base sim's behavior". — [SideFX docs](https://www.sidefx.com/docs/houdini/nodes/dop/gasupres.html)

### Inferences
- **Likely cause of the missing wisps:** If the team's guided 256^3 sim constrains velocity at all frequencies, or uses a strong blend or steady guiding, the training targets will lack the free high-frequency content the network is supposed to learn. Nielsen 2010 says exactly this about steady guiding. Two checks follow:
  1. Measure the energy spectrum of the 256^3 targets against an unguided 256^3 sim. If the high-k tail is suppressed, fix the data before the loss.
  2. Guide only below the coarse grid's Nyquist frequency (for example, low-pass-filter the difference and penalize only that), not the full field.
- **Pairing tension:** Guided pairs give spatial alignment, which regression losses need. Unguided or downsampled pairs give realistic detail statistics, which adversarial or distribution losses can use without exact alignment.

### Gaps
- Exact Nielsen energy-functional equations were not extracted; only abstracts were read.
- Bridson-style "guide shapes" and Houdini's "Gas Match Field" / guided pyro specifics were not covered.

---

## 6. Why regression losses blur fluid detail, and which objectives produce filaments

### Takeaway
Pixel- or voxel-wise L1/L2 losses predict the conditional mean of many plausible fine-scale realizations. Turbulent detail is not determined by the coarse input, so the mean averages out the filaments. Sharpening losses (gradient, spectrum) fix edge sharpness but not this ambiguity. Every method that produced convincing filaments used a distribution-matching term:
- adversarial spatial plus temporal discriminators with discriminator feature losses (tempoGAN, Multi-Pass GAN),
- a learned dictionary of real high-res atoms (Bai et al.),
- diffusion or flow-matching refinement (PDE-Refiner, consistency SR), or
- optimization-based style transfer against exemplar structure (Kim et al. 2019/2020).

### Cited Findings
- **Low-amplitude high frequencies:** PDE-Refiner identifies neglect of "non-dominant spatial frequency information" under MSE as the core failure. — [arXiv 2308.05732](https://arxiv.org/abs/2308.05732)
- **tempoGAN's loss design:**
  - The adversarial spatial discriminator plus a feature-space loss in discriminator layers carried the detail. L1 was kept at a small weight.
  - Wavelet-turbulence data needed 2x the data and a different feature-loss setting.
  - A plain L2 temporal loss lowered density, so it was dropped in favour of an adversarial temporal discriminator.
  - — [ar5iv tempoGAN](https://ar5iv.labs.arxiv.org/html/1801.09710)
- **Multi-Pass GAN, 8x model:** Used WGAN-GP plus L1 (λ_L1 = 20) without feature loss. — [ar5iv Multi-Pass GAN](https://ar5iv.labs.arxiv.org/html/1906.01689)
- **Bai et al. vs tempoGAN:** tempoGAN output looks "sharper" but does not capture the fine dynamics. The dictionary approach follows the fine sim more closely but occasionally places wisps where the coarse sim has little smoke. — [ar5iv Bai et al.](https://ar5iv.labs.arxiv.org/html/1910.09166)
- **Transport-based NST for smoke (Kim, Azevedo, Thuerey, Kim, Gross, Solenthaler, ACM TOG 2019):** Optimizes a velocity that transports density toward a stylized target. Temporal coherence comes from transporting and aligning successive stylized velocities. — [DeepAI listing](https://api.deepai.org/publication/transport-based-neural-style-transfer-for-smoke-simulations); [ar5iv 1905.07442](https://ar5iv.arxiv.org/html/1905.07442)
- **Lagrangian NST for fluids (Kim, Azevedo, Gross, Solenthaler, SIGGRAPH 2020):** Attributes on particles give intrinsic temporal consistency. Compute drops from about one day to about one hour, under an hour per sequence per the authors. It is offline optimization, not a feed-forward net. — [arXiv 2005.00803](https://arxiv.org/pdf/2005.00803)

### Inferences
- **For the team's screen-space network:**
  - Keep L1 only at low weight, or only on a low-passed version of the output, so the coarse structure is anchored.
  - Put the detail burden on a patch discriminator, on discriminator-feature matching, and on a temporal discriminator fed motion-compensated triplets.
  - Spectrum losses on the image alone match energy but not phase or structure, which explains "sharpened edges without wisps".
  - An alternative low-risk target is to predict the parameters of procedural detail (noise amplitude, scale, anisotropy) rather than pixels.
- **Style-transfer methods:** Useful offline for art direction and for authoring training targets (stylized exemplars). Too slow for runtime use as published.

### Gaps
- No quantitative ablation of loss types specifically for screen-space smoke detail was found.
- No published comparison of spectrum loss versus adversarial loss on smoke was found.

---

## 7. Real-time suitability and pipeline slotting (summary table)

### Takeaway
Within a 2.5 ms detail budget on an RTX 5080, the published approaches that plausibly fit are:
1. Procedural wavelet- or curl-noise upres keyed on coarse-sim energy.
2. A small 2D or screen-space network with adversarial, temporal and feature losses (tempoGAN losses applied to a lightweight 2D architecture), possibly one-step distilled.
3. A solver-in-the-loop correction at 64^3, which belongs in the solver budget and improves coarse motion only.

Dense volumetric GANs, dictionary methods, diffusion samplers and flow-map simulators all have published costs of seconds per frame or more.

### Cited Findings (timing anchors; see sections above for sources)
| Technique | Published cost | Hardware | Real time? |
|---|---|---|---|
| tempoGAN 64^3→256^3 | 17.9 s/frame | GTX 1080 Ti | No ([ar5iv](https://ar5iv.labs.arxiv.org/html/1801.09710)) |
| Multi-Pass GAN 64^3→256^3 / 512^3 | 10.14 s / 59.65 s | GTX 1080 Ti | No ([ar5iv](https://ar5iv.labs.arxiv.org/html/1906.01689)) |
| Bai et al. dictionary 50^3→200^3 | 1–3 s (0.02–3.48 s range) | TITAN X | No ([ar5iv](https://ar5iv.labs.arxiv.org/html/1910.09166)) |
| Wavelet turbulence 80×64×64→720×576×576 | <2 min/frame | 8-core CPU (2008) | CPU no; GPU port plausible, not verified ([Cornell](https://www.cs.cornell.edu/~tedkim/WTURB)) |
| EmberGen sparse solver | 1 B voxels interactive | RTX PRO 6000 Blackwell | Yes, but brute-force sim, not upres ([80.lv](https://80.lv/articles/upcoming-embergen-s-sparse-solver-handles-one-billion-voxels-in-real-time)) |
| Kochkov learned correction | 40–80x speedup vs equal-accuracy solver | TPU/GPU (2D) | 3D real-time unverified ([arXiv](https://arxiv.org/abs/2102.01010)) |
| Distilled 1-step flow-matching SR | 12x faster than teacher | not stated | Unverified ([arXiv](https://arxiv.org/pdf/2605.05975)) |
| Deep Fluids | up to 700x vs CPU re-sim | — | Fast, but scenario-bound ([arXiv](https://arxiv.org/pdf/1806.02071)) |
| Lagrangian NST | ~1 h per sequence | — | No ([arXiv](https://arxiv.org/pdf/2005.00803)) |

### Inferences
- **Suggested pipeline:** 64^3 solver with higher-order advection and an optional learned correction, then GPU curl/wavelet turbulence evaluated at raymarch sample points or on sparse high-res bricks (amplitude driven by coarse sub-grid energy, as in Houdini's "Wavelets" source), then optionally a screen-space adversarially trained refinement for final filaments.
- **Brute force as a comparison point:** EmberGen 2.0 shows that brute-force sparse simulation at high voxel counts is becoming feasible on top-end GPUs. A sparse 128^3–256^3 active-brick simulation is a non-learned baseline the team should benchmark against.

### Gaps
- No published RTX 40/50-series ms numbers were found for any of these techniques. All real-time judgements above are extrapolations.
