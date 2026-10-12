"""PDE problem protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from mtl_final.solvers import GridSnapshot


@dataclass
class EvalGrid:
    coords: dict[str, np.ndarray]
    coord_tensors: tuple[torch.Tensor, ...]
    param_tensor: torch.Tensor
    time_tensor: torch.Tensor | None
    shape: tuple[int, ...]


@dataclass
class ProblemSpec:
    name: str
    coord_names: tuple[str, ...]
    param_names: tuple[str, ...]
    field_names: tuple[str, ...]
    task_params: list[tuple[float, ...]]
    query_params: list[tuple[float, ...]]
    query_labels: dict[tuple[float, ...], str]
    snapshot_param: tuple[float, ...]
    has_time: bool
    t_min: float = 0.0
    t_max: float = 1.0
    parametric_trunk: bool = False
    hidden_dim: int = 128
    layers: int = 5
    attn_hidden: int = 64
    n_res: int = 400
    n_ic: int = 100
    n_bc: int = 100
    extra: dict = field(default_factory=dict)

    @property
    def n_coord(self) -> int:
        return len(self.coord_names)

    @property
    def n_param(self) -> int:
        return len(self.param_names)

    @property
    def n_fields(self) -> int:
        return len(self.field_names)

    @property
    def time_index(self) -> int | None:
        if "t" in self.coord_names:
            return self.coord_names.index("t")
        return None


class PDEProblem:
    spec: ProblemSpec

    def sample_batch(self, n: int, device: torch.device, kind: str = "residual"):
        raise NotImplementedError

    def physics_loss(
        self,
        model,
        mu: Sequence[float],
        device: torch.device,
        *,
        head_idx: int | None = None,
        attn_reg: float = 0.0,
        task_index: int | None = None,
        hybrid=None,
    ) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def reference(self, coords: dict[str, np.ndarray], mu: Sequence[float]) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def eval_grid(self, mu: Sequence[float], device: torch.device, **grid_kw) -> EvalGrid:
        raise NotImplementedError

    def numerical_snapshot(self, mu: Sequence[float]) -> GridSnapshot:
        raise NotImplementedError

    def mu_for_split(self, split: str) -> tuple[float, ...]:
        """Query parameter labelled interpolation or extrapolation."""
        wanted = split.strip().lower()
        for mu, label in self.spec.query_labels.items():
            if label.lower() == wanted or label.lower().startswith(wanted):
                return tuple(float(v) for v in mu)
        raise KeyError(f"{self.spec.name} has no query labelled {split!r}")

    def example_inputs(self, device: torch.device, n: int = 8):
        mu = self.spec.task_params[0]
        coords, t = self._dummy_coords(n, device)
        params = torch.tensor([mu], dtype=torch.float32, device=device).repeat(n, 1)
        return coords, params, t

    def _dummy_coords(self, n: int, device: torch.device):
        coords = torch.rand(n, self.spec.n_coord, device=device)
        t = None
        if self.spec.has_time:
            t_idx = self.spec.time_index
            t = coords[:, t_idx : t_idx + 1] if t_idx is not None else torch.rand(n, 1, device=device)
        return coords, t

    def pack_params(self, n: int, mu: Sequence[float], device: torch.device) -> torch.Tensor:
        return torch.tensor(mu, dtype=torch.float32, device=device).view(1, -1).repeat(n, 1)

    def voter_time(self, coords: torch.Tensor) -> torch.Tensor | None:
        if not self.spec.has_time:
            return None
        idx = self.spec.time_index
        if idx is None:
            return None
        return coords[:, idx : idx + 1]

    def split_grad_coords(self, packed: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        cols = [
            packed[:, i : i + 1].detach().clone().requires_grad_(True)
            for i in range(self.spec.n_coord)
        ]
        return cols, torch.cat(cols, dim=1)
