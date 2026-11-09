# Neural VFX Engine — conventions

Full design: `SPEC.md`. This repo is currently building the **v0 2D demo** (see "Demo scope" in SPEC.md).

## Environment
- Windows 11, RTX 5080 (sm_120), Python 3.12 venv at `.venv/`.
- Always run Python as `.venv/Scripts/python` (from the repo root). Never install into the global Python.
- Packages: torch (cu128), warp-lang, pygame-ce, numpy, scipy, pytest.
- Datasets/checkpoints are large: they live OUTSIDE OneDrive, under `vfx.core.paths.DATA_ROOT`
  (default `%LOCALAPPDATA%/neural-vfx`). Never write big files inside the repo.

## Commands
- Tests: `.venv/Scripts/python -m pytest tests -q`
- Bench: `.venv/Scripts/python -m bench.run`
- Datagen: `.venv/Scripts/python -m vfx.datagen.build --version v1`
- Train: `.venv/Scripts/python -m vfx.train.train --config vfx/train/configs/demo.json`
- Demo app: `.venv/Scripts/python -m vfx.app.main`

## Layout and ownership (one owner per directory — do not edit other owners' files)
| Path | Owner |
|---|---|
| `vfx/core/` (fields.py, paths.py) | lead — interface changes need lead approval |
| `vfx/sim/` | solver engineer |
| `vfx/datagen/` | data engineer |
| `vfx/model/`, `vfx/train/`, `vfx/runtime/` | ML/runtime engineer |
| `vfx/render/`, `vfx/app/` | render/app engineer |
| `tests/`, `bench/`, `vfx/eval/` | test/eval engineer |

If you need something from another module that does not exist yet, code against the interface
below; do not implement it yourself inside their directory. If an interface is wrong, report it.

## Module interfaces (v0, 2D)
All types come from `vfx.core.fields`.

```python
# vfx/sim/__init__.py
class SolverConfig: ...            # dataclass of physical params, sensible defaults
class Solver:
    def __init__(self, res: int, config: SolverConfig | None = None, device="cuda"): ...
    def reset(self, obstacles: list[Obstacle] = ()) -> FieldState      # ambient, empty
    def step(self, state: FieldState, forces: Forces, dt: float) -> FieldState
    # step may update state in place and return it. Deterministic for identical inputs.
    # --- hooks for tests (public, stable) ---
    def divergence(self, state: FieldState) -> torch.Tensor  # (H,W) |div u|*dx/(max|u|+1e-8), fluid cells only (0 in solids)
    def advect(self, field: torch.Tensor, velocity: torch.Tensor, dt: float) -> torch.Tensor
    #   field (H,W) or (C,H,W) float32 cuda, velocity (2,H,W) m/s; returns the advected field (same scheme step() uses)
    def project(self, state: FieldState) -> FieldState      # pressure projection only, in place ok
SOLVER_VERSION: str
# SolverConfig field names are fixed (tests set them to 0 to isolate terms):
#   buoyancy_alpha (density sinks), buoyancy_beta (heat lifts), vorticity_eps, density_dissipation,
#   temperature_cooling, burn_rate, ignition_temp, burn_heat, smoke_from_fuel,
#   pressure_tol, pressure_max_iters
# All in world units, so 128^2 and 512^2 runs of the same Scenario look alike at large scales.

# vfx/datagen/__init__.py
def sample_scenario(seed: int, split: str) -> Scenario          # split in {"train","test","ood"}
def run(scenario: Scenario, out_dir=None, frames=None) -> dict   # writes one scenario dir, returns metadata
def load_dataset(version: str, split="train") -> "PairedDataset"  # test/ood items also carry `lr_sim`

# vfx/model/__init__.py
class DetailNet(torch.nn.Module):
    def forward(self, lr: Tensor, prev_hr_warped: Tensor) -> Tensor
    # lr (B,5,h,w) physical units; prev_hr_warped (B,2,4h,4w) density,temperature(K);
    # returns (B,2,4h,4w) density,temperature(K). Normalization happens INSIDE the model.
def load_model(path) -> DetailNet

# vfx/runtime/__init__.py
def warp_prev(prev_hr, lr_velocity, dx_hr, dt) -> Tensor   # (2,H,W) prev HR advected by upsampled LR velocity
#   used by BOTH train/ and runtime/ so training and inference see identical warping
class Upscaler:                    # neural; owns temporal state (previous HR output)
    def __init__(self, checkpoint_path: str | None = None, device="cuda"): ...
    def reset(self) -> None
    def infer(self, lr_state: FieldState) -> HRFields               # HR_RES x HR_RES
    timings: dict[str, float]      # ms of last infer, measured with CUDA events
def classical_upscale(lr_state: FieldState, mode="bicubic") -> HRFields   # baseline

# vfx/render/__init__.py
class Renderer:
    def __init__(self, out_size: tuple[int, int], device="cuda"): ...
    def draw(self, fields: HRFields, solid: torch.Tensor | None = None) -> torch.Tensor
    # returns (out_h, out_w, 3) uint8 on cuda, already flipped so row 0 = top of screen

# vfx/eval/metrics.py
def spectrum_error(pred, ref) -> float      # log radial power-spectrum error of density
def flicker(seq, velocities, dx, dt) -> float
def l1(pred, ref) -> float
```

## Dataset format (v0) — `DATA_ROOT/datasets/<version>/`
```
card.json        version, counts per split, param ranges, solver_version, created
splits.json      {"train": [seeds], "test": [seeds], "ood": [seeds]}   # split by scenario seed
<split>/scn_<seed:06d>/
  meta.json      scenario (dataclasses.asdict), solver_version, lr_res, hr_res, dt, start_step, tags
  lr.npy         (T, 5, 128, 128) float16   HR state 4x average-pooled (training input)
  hr.npy         (T, 2, 512, 512) float16   HR density, temperature (target)
  solid_lr.npy   (128, 128) uint8           solid_hr.npy (512, 512) uint8
  lr_sim.npy     (T, 5, 128, 128) float16   independent 128^2 sim of the same scenario (test/ood only)
```
Frames are consecutive solver steps (frame t-1 -> t is one SIM_DT). `load_dataset(version)` returns a
torch Dataset whose items are dicts of float32 tensors: `lr` (5,128,128), `hr_prev` (2,512,512),
`hr` (2,512,512), `solid_lr`, `solid_hr`, plus `seed`, `t`. Whole dataset must stay under ~6 GB.

## Git
Five engineers share this one checkout. Do NOT run git commands (no add/commit/stash/checkout);
the lead reviews and commits.

## Rules
- No CPU copies (`.cpu()`, `.item()`, `.numpy()`, `print(tensor)`) in the frame loop. The ONLY
  exception is the final display blit in `vfx/app/`.
- Time GPU work with `torch.cuda.Event` / `wp.ScopedTimer(synchronize=True)`, never wall-clock without sync.
- Small, readable modules. Each owner adds a short `README.md` in their directory.
- The test agent never tests code it wrote; owners may add quick self-checks under their own dir
  (`vfx/<mod>/selfcheck.py`) but the gate tests live in `tests/`.
- Report numbers you measured; do not claim numbers you did not run.
