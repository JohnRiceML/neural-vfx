# Neural VFX Engine — SPEC

## Demo scope (v0, lead decision 2026-11-09)
A playable 2D slice through every stage of the pipeline, before committing to 3D:
- 2D Eulerian smoke + fire solver in **Warp** (open decision resolved for v0: Warp; MAC internally allowed, collocated at the boundary).
- LR grid 128², HR reference 512² (4× upscale), sim step 1/30 s, render 60 fps.
- Paired dataset: HR sim, LR = 4× average-pooled HR (risk-table mitigation), plus independent LR sims for eval. Stored as float16 `.npy` per scenario (Zarr deferred) under `%LOCALAPPDATA%/neural-vfx`.
- Detail network: small residual conv net with warped previous-HR input (M3-style, non-adversarial), FP16 inference in PyTorch (TensorRT deferred), sparse 16² bricks.
- Renderer: 2D shading with blackbody emission for fire. App: pygame-ce window, classical vs neural toggle, side-by-side, HUD with per-stage GPU ms.
- Demo exception to the no-CPU-copy rule: one final uint8 display blit per frame.
- Tests: Layer 1 solver tests, Layer 3 data integrity, Layer 4 model quality vs bicubic, Layer 5 bench.

---


Neural VFX Engine Spec
Nov 9, 2026 · @John Rice
Build a real-time neural VFX engine for smoke and fire, with water as a later phase: a fast classical GPU solver at low resolution plus a learned detail network, running at 60fps for localized effects on one RTX 5080.
Goals and success criteria
The v1 deliverable is an interactive demo where neural smoke and fire hold 60fps on the target machine and beat classical upsampling on every quality metric below.
Target machine: AMD Ryzen 7, NVIDIA RTX 5080 (16 GB VRAM), 1440p display.
Target
Metric
v1 threshold
Frame rate
Full demo, one localized effect on screen
≥ 60 fps sustained, 1% lows ≥ 50 fps
Sim + neural cost
GPU time per frame for one effect
≤ 6 ms total
Detail quality
Energy spectrum error vs high-res reference
≥ 50% lower than tricubic upsampling
Temporal stability
Warped-frame flicker metric
Within 1.2× of the high-res reference
Long-run stability
10-minute interactive session
No NaN, no blow-up, no visible drift
Generalization
Held-out scene types (new obstacles, forces)
Quality within 20% of in-distribution scores
Interactivity
Input to visible response
≤ 3 frames
Every threshold is measured by an automated test (see Testing and validation), not by eye alone.
Scope and non-goals
v1 is smoke and fire in a bounded box; water follows only after the smoke and fire gates pass.
In scope
2D and 3D Eulerian grid solver for smoke and fire (density, velocity, temperature, fuel)
Paired low-res / high-res dataset generator
Neural detail network (upscaler) with temporal coherence
Real-time runtime: TensorRT inference, sparse brick processing, volume renderer
Interactive playground app with a classical vs neural toggle and a performance HUD
Phase 4: hybrid particle water (FLIP or MPM) with a neural model and a classical fallback
Not in v1
Open-world scale or many simultaneous effects
Support for GPUs other than the target RTX 5080
Unreal or Unity plugin (stretch goal after Phase 5)
Destruction and character physics (future project reusing this stack)
Physical accuracy for engineering use; the bar is visual plausibility plus stability
Prior work to build on
Reproduce tempoGAN as the smoke baseline, copy the hybrid-with-fallback design for water, and treat real-time neural fire as the original contribution.
Work
What we take from it
Used in
tempoGAN (Xie, Franz, Chu, Thuerey, SIGGRAPH 2018)
Conditional GAN upscaler with a temporal discriminator; single-frame inference; the baseline to reproduce in PyTorch (code)
Phase 2
Multi-pass GAN (Werhahn et al., SCA 2019)
Splitting 3D upscaling into 2D passes to fit memory; 8× upscaling
Phase 2, if 3D VRAM is tight
Dictionary-based smoke upsampling (Bai et al.)
Patch-based synthesis that generalized from a ball obstacle to a dragon; a generalization benchmark idea
Phase 2 evaluation
Neural Flow Maps (Deng et al., SIGGRAPH Asia 2023)
Vortex preservation with sparse multi-resolution neural fields; quality reference, not v1
Later research
Learning to Simulate (GNS) (Sanchez-Gonzalez et al., ICML 2020)
Encode-process-decode graph network for particle fluids
Phase 4 water
Hybrid neural-MPM (2025)
Neural physics with a classical MPM fallback triggered by a cheap divergence monitor
Phase 4 water
GNN liquids with kinematic objects (2025)
Liquid interacting with moving rigid bodies
Phase 4 stretch
Tools
NVIDIA Warp: GPU simulation kernels in Python, zero-copy interop with PyTorch
PhiFlow: differentiable reference solver from the tempoGAN group, for cross-checking
PyTorch for training; ONNX + TensorRT for runtime inference at FP16/FP8
System architecture
The system has two pipelines that share one solver: an offline pipeline that builds the model, and a per-frame runtime that runs it.

system architecture · offline pipeline and per-frame runtime
The runtime reuses the exact solver code that generated the training data, so the network never sees a simulator it was not trained on. The trained model reaches the runtime only as an exported TensorRT engine.
Module
Responsibility
sim/
Classical GPU solvers in Warp: advection, pressure projection, buoyancy, vorticity confinement, combustion; water in Phase 4
datagen/
Randomized scenarios, paired LR / HR runs, dataset versions and cards
model/ + train/
Upscaler networks, losses, discriminators, config-driven training
runtime/
Distillation, ONNX and TensorRT export, sparse brick scheduling, CUDA graph
render/
Volume ray-marcher with blackbody emission for fire
app/
Interactive playground, classical vs neural toggle, performance HUD
tests/ + bench/
All six test layers and stored performance baselines
Interfaces and data contracts
One shared schema file (vfx/core/fields.py) defines every tensor that crosses a module boundary; changing it requires a spec update and the lead&apos;s approval.
Field state (the core contract)
Layout: (C, D, H, W) for 3D, (C, H, W) for 2D; batch dim added only inside training
Channels, fixed order: density, vel_x, vel_y, vel_z, temperature, fuel (fire adds the last two; unused channels are absent, not zero-filled)
Velocity on a collocated grid at the module boundary, even if the solver uses a staggered (MAC) grid internally
Units: grid spacing dx in meters, time step dt in seconds, temperature in kelvin; all stored in a FieldMeta record beside the tensor
Dtype: float32 in the solver, float16 at the network boundary
Device: tensors stay on the GPU; any CPU copy in the runtime loop is a bug
Module interfaces
Module
Input
Output
sim.step(state, forces, dt)
Low-res field state, user forces
Next low-res field state
datagen.run(scenario)
Scenario config with seed
Paired LR / HR sequences + metadata JSON
model.forward(lr_state, prev_hr)
LR state + previous HR output warped by velocity
HR density (and temperature for fire)
runtime.infer(lr_state)
LR state on GPU
HR detail for active bricks only
render.draw(hr_fields, camera)
HR density, temperature
Frame buffer
Dataset format
Zarr shards, float16, chunked per frame; one metadata JSON per scenario (seed, solver version, resolution, emitters, obstacles)
Splits are by scenario seed, never by frame, so near-identical frames cannot leak between train and test
Every dataset gets a version tag and a short dataset card (counts, parameter ranges, generator commit)
Performance budget
One effect gets 6 ms of GPU time per frame, leaving about 10 ms of a 16.7 ms frame for the rest of a game. These are starting targets; the benchmark harness in M0 confirms or revises them.
Stage
Resolution
Budget per frame
Classical solver step
64³ low-res grid
≤ 2.0 ms
Neural detail inference
Active bricks only, 4× upscale
≤ 2.5 ms
Volume rendering
1440p, ray-marched
≤ 1.5 ms
Total for one effect

≤ 6.0 ms
Peak VRAM, runtime

≤ 4 GB
Tactics that make the budget, in order of expected payoff
Sparse bricks: split the domain into 16³ bricks and run the network only on bricks with density or heat above a threshold
Distill the trained model into a small runtime model; export through ONNX to TensorRT at FP16, then test FP8
Keep everything on the GPU: Warp and PyTorch share memory, and the frame loop is captured as a CUDA graph
Decouple rates: simulate at 30 Hz, interpolate for 60 fps rendering
Fallback path: screen-space 2D detail on the rendered image if 3D inference cannot meet budget
Testing and validation
No module merges without tests, and the agent that writes tests is never the agent that wrote the code under test. A neural model trained on a broken solver learns the bug, so solver correctness is gated before any training run.
Layer 1: solver correctness (runs on every commit)
Divergence after pressure projection below 1e-4 (normalized) everywhere in the domain
Density mass conserved within 0.1% over 1,000 steps in a closed box with no sources
Pure advection of a known field matches the analytic result within tolerance
Buoyancy pushes hot, light fluid up; boundary conditions hold at walls and obstacles
Determinism: same seed and config give bit-identical output
Convergence: error against a reference shrinks as resolution increases
Layer 2: physics benchmarks (runs nightly)
Taylor–Green vortex decay against the analytic solution
Lid-driven cavity velocity profiles against published reference data (Ghia et al. 1982)
Vortex ring and leapfrogging rings: rings stay coherent for a set number of passes
Phase 4: dam break front position against published results
Cross-check smoke scenes against PhiFlow on identical inputs
Layer 3: data integrity (runs on every dataset build)
No NaN or Inf; shapes, dtypes and channel order match the schema
Per-channel statistics within expected ranges; outlier scenarios flagged, not silently kept
Splits by scenario seed, verified by a leakage test
Held-out test set includes scene types never seen in training (new obstacle shapes, stronger forces, larger emitters)
Layer 4: model quality (runs on every checkpoint)
Sanity: the model can overfit a single batch; a broken pipeline fails here in minutes
Baselines it must beat: trilinear and tricubic upsampling of the low-res sim
Metrics: energy spectrum error (the key detail metric), per-voxel L1, warped-frame flicker, LPIPS on fixed renders, divergence of predicted velocity if velocity is predicted
Long rollout: 10,000 frames with no blow-up, measured on held-out scenes
Out-of-distribution score reported separately from in-distribution score
Layer 5: performance (runs on every commit touching runtime or render)
Benchmark harness reports ms per stage and peak VRAM on fixed scenes
CI fails on a regression above 10% against the stored baseline
Layer 6: visual review (every milestone)
Fixed camera paths render side by side: classical low-res, neural, high-res reference
Results published to a review page with the metrics above; a human signs off at each milestone gate
Experiment hygiene
Every training run is config-driven, logged (TensorBoard), and tied to a dataset version and git commit
An experiment log records the hypothesis, result and decision for each run, so agents do not repeat dead ends
Milestones and exit gates
Milestones run in order; a milestone is done only when every exit gate passes in CI and the visual review is signed off.
M0 Foundation. Repo, environment (Python, PyTorch, Warp, TensorRT), CI, benchmark harness, schema file.
Gate: CI green on the target machine; GPU detected; harness reports timings for a trivial kernel.
M1 Classical smoke. 2D stable-fluids solver, then 3D, with vorticity confinement and obstacles.
Gate: all Layer 1 and Layer 2 tests pass; 64³ step ≤ 2 ms.
M2 Data factory v1. Randomized scenario generator, paired LR/HR runs, dataset card.
Gate: Layer 3 tests pass; at least 500 training scenarios and a held-out OOD set.
M3 Baseline upscaler. Non-adversarial 3D U-Net with previous-frame input; 2D first, then 3D.
Gate: beats tricubic on spectrum error and L1 on held-out scenes.
M4 tempoGAN-style model. Spatial and temporal discriminators added.
Gate: spectrum error at least 50% below tricubic; flicker within 1.2× of reference; 10,000-frame rollout stable.
M5 Real-time runtime. Distilled model, TensorRT, sparse bricks, CUDA graph, renderer, playground app.
Gate: 60 fps sustained with one effect; total ≤ 6 ms; no CPU copies in the frame loop.
M6 Fire. Temperature, fuel, combustion and blackbody emission; model retrained with the extra channels.
Gate: M4 and M5 gates re-pass for fire scenes.
M7 Water (hybrid). FLIP or MPM solver, GNS-style neural model, divergence monitor with classical fallback.
Gate: dam-break benchmark passes; fallback triggers before visible failure in OOD tests; 60 fps in a small tank.
M8 Engine plugin (stretch). Unreal or Unity integration of the runtime.
Agent team
Seven roles, each owning one module, with an independent test agent and a lead who owns this spec and every interface.
Role
Owns
Must not
Lead / architect
This spec, fields.py, milestone gates, merge decisions
Write module code without review
Solver engineer
sim/ (Warp kernels, boundary handling, combustion)
Change interfaces unilaterally
Data engineer
datagen/, dataset versions and cards
Train models
ML engineer
model/, train/, experiment log
Edit the solver to make metrics look better
Runtime engineer
runtime/ (export, TensorRT, bricks, CUDA graph)
Change model architecture without the ML engineer
Render and app engineer
render/, app/, performance HUD
Add CPU copies to the frame loop
Test and eval engineer
tests/, benchmarks, metrics, review page
Test code it wrote itself
Working rules
Each agent works in its own git worktree and branch; merges go through review by a second agent plus passing CI
An interface change needs a spec update first, approved by the lead
A repo-root CLAUDE.md holds the conventions: commands, layout, test commands, the no-CPU-copy rule
Small PRs: one feature or fix each, with tests in the same PR
Every training run and benchmark is logged; claims in a PR cite the logged numbers
Agents stop and report at a milestone gate rather than starting the next milestone
Repo layout
vfx/  core/       fields.py, config, logging  sim/        smoke, fire, water solvers  datagen/    scenarios, paired runs, dataset cards  model/      networks, losses, discriminators  train/      training loops, configs  runtime/    export, TensorRT engines, bricks  render/     volume ray-marcher  app/        interactive playgroundtests/        unit, benchmarks, data, model, perfbench/        harness and stored baselinesCLAUDE.mdSPEC.md       export of this doc
Risks and mitigations
The biggest risk is the gap between training data and game reality: a coarse in-game sim drifts away from its high-res twin.
Risk
Mitigation
LR and HR sims diverge, so pairs stop matching
Start with HR downsampled to LR for training; evaluate on independently simulated LR; try guiding HR large scales with LR
Temporal flicker
Previous-frame input warped by velocity, temporal discriminator, flicker metric in CI
GAN training instability
Non-adversarial baseline first (M3); adversarial loss added with a small weight and early stopping on spectrum error
3D training exceeds 16 GB VRAM
Patch training, mixed precision, gradient checkpointing; multi-pass 2D fallback
Overfitting to the scenario generator
Wide parameter randomization; held-out OOD scene types reported separately
Runtime misses 6 ms
Sparse bricks, distillation, FP8; screen-space fallback path
Agents optimize the metric, not the look
Human visual sign-off at every gate; test agent independent of model agent
Open decisions
☐ OS: native Windows or Linux / WSL2 for development (recommended: WSL2 or Linux for tooling; confirm TensorRT setup on the target machine)
☐ Solver framework: Warp (recommended for PyTorch zero-copy and CUDA graphs) or Taichi
☐ Grid type inside the solver: collocated or staggered MAC (recommended: MAC internally, collocated at the boundary)
☐ Renderer: custom ray-marcher in Warp or CUDA, or an existing engine for the demo
☐ Water method for M7: FLIP or MPM
Sources
tempoGAN paper and code
A Multi-Pass GAN for Fluid Flow Super-Resolution
Dynamic Upsampling of Smoke through Dictionary-based Learning
Fluid Simulation on Neural Flow Maps
Learning to Simulate Complex Physics with Graph Networks
Hybrid real-time neural-MPM fluid simulation
GNNs for liquid simulations with kinematic objects
NVIDIA Warp, PhiFlow
