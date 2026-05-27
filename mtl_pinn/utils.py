"""Shared utilities for experiments."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 50) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None = None) -> str:
    if requested is None or requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def configure_matplotlib(interactive: bool = False) -> None:
    import matplotlib

    if interactive:
        for backend in ("Qt5Agg", "TkAgg"):
            try:
                matplotlib.use(backend)
                break
            except ImportError:
                continue
    else:
        matplotlib.use("Agg")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
