"""3D smoke + fire solver (Warp). See README.md."""
from .config import SOLVER3D_VERSION, SolverConfig3D, runtime_config
from .solver import Solver3D

__all__ = ["Solver3D", "SolverConfig3D", "SOLVER3D_VERSION", "runtime_config"]
