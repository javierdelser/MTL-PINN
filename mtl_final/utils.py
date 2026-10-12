"""Shared helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None = None) -> torch.device:
    if requested is None or requested == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        name = requested
    return torch.device(name)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def mlp(in_dim: int, hidden_dim: int, layers: int) -> nn.Sequential:
    modules: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
    for _ in range(layers - 1):
        modules.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
    return nn.Sequential(*modules)


def as_col(value, like: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(dtype=like.dtype, device=like.device).reshape(-1, 1).expand_as(like)
    return torch.full_like(like, float(value))


def full(n: int, value: float, device: torch.device) -> torch.Tensor:
    return torch.full((n, 1), float(value), dtype=torch.float32, device=device)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def make_scheduler(optimizer, epochs: int, schedule: str, lr_min: float = 1e-5):
    if schedule == "constant":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=lr_min
    )
