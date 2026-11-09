"""2D smoke + fire solver (Warp). Interface: see CLAUDE.md "Module interfaces"."""
from .config import SOLVER_VERSION, SolverConfig
from .solver import Solver

__all__ = ["Solver", "SolverConfig", "SOLVER_VERSION"]
