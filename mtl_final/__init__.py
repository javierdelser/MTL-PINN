"""MTL-PINN experiments for parametric PDEs."""

__version__ = "1.0.0"

from mtl_final.config import PDE_NAMES, RUN_MODES
from mtl_final.problems import get_problem

__all__ = ["__version__", "PDE_NAMES", "RUN_MODES", "get_problem"]
