"""Sub-grid smoke detail carried on Lagrangian particles advected by the 64^3 velocity and deposited into a
`Volume` (feasibility study; see README.md). OWNER: particle-detail engineer."""
from .particles import DEFAULTS, DetailParticles

__all__ = ["DetailParticles", "DEFAULTS"]
