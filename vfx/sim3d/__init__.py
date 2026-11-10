"""3D smoke + fire solver (Warp). See README.md."""
from .config import (SOLVER3D_REVISION, SOLVER3D_VERSION, SolverConfig3D, fused_config, fused_fixed_config,
                     runtime_config, solver_version_tag)
from .solver import Solver3D

__all__ = ["Solver3D", "SolverConfig3D", "SOLVER3D_VERSION", "SOLVER3D_REVISION", "runtime_config",
           "fused_config", "fused_fixed_config", "solver_version_tag"]
