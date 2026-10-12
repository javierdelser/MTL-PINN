#!/usr/bin/env python3
"""Show interpolation and extrapolation for every baseline, grouped by PDE."""

import sys
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "--show" not in sys.argv:
    matplotlib.use("Agg")

from mtl_final.plot_comparison import main

if __name__ == "__main__":
    main()
