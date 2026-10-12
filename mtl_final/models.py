"""Shared backbone, linear task heads, and a pairwise voter on (μ, μ_ℓ)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from mtl_final.utils import mlp


class MTLPINN(nn.Module):
    """Multi-head PINN whose voter mixes heads from the query parameter.

    The backbone sees coordinates (and, if ``parametric_trunk``, the PDE
    parameter). The voter sees pairs ``(μ_query, μ_task)``. When
    ``time_conditioned_voter`` is true, physical time is concatenated so the
    mixing weights may vary along a trajectory:

        w_ℓ(μ, t) = softmax( v(μ, μ_ℓ, t) )   or raw logits if softmax is off.
    """

    def __init__(
        self,
        *,
        n_coord: int,
        n_param: int,
        n_fields: int,
        task_params: Sequence[Sequence[float]],
        hidden_dim: int = 128,
        layers: int = 5,
        attn_hidden: int = 64,
        softmax: bool = True,
        time_conditioned_voter: bool = False,
        parametric_trunk: bool = False,
        t_max: float = 1.0,
        coord_scale: Sequence[float] | None = None,
        param_offset: Sequence[float] | None = None,
        param_scale: Sequence[float] | None = None,
    ):
        super().__init__()
        tasks = torch.tensor(task_params, dtype=torch.float32)
        if tasks.ndim == 1:
            tasks = tasks.view(-1, 1)
        self.register_buffer("task_params", tasks.view(1, -1, n_param))
        self.n_coord = n_coord
        self.n_param = n_param
        self.n_fields = n_fields
        self.softmax = softmax
        self.time_conditioned_voter = time_conditioned_voter
        self.parametric_trunk = parametric_trunk
        self.t_max = float(t_max)
        if coord_scale is None:
            coord_scale = [1.0] * n_coord
        if param_offset is None:
            param_offset = [0.0] * n_param
        if param_scale is None:
            param_scale = [1.0] * n_param
        self.register_buffer("coord_scale", torch.tensor(coord_scale, dtype=torch.float32).view(1, -1))
        self.register_buffer("param_offset", torch.tensor(param_offset, dtype=torch.float32).view(1, -1))
        self.register_buffer("param_scale", torch.tensor(param_scale, dtype=torch.float32).view(1, -1))

        trunk_in = n_coord + (n_param if parametric_trunk else 0)
        self.shared = mlp(trunk_in, hidden_dim, layers)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, n_fields) for _ in range(tasks.shape[0])]
        )
        voter_in = 2 * n_param + (1 if time_conditioned_voter else 0)
        self.attn_net = nn.Sequential(
            nn.Linear(voter_in, attn_hidden),
            nn.ReLU(),
            nn.Linear(attn_hidden, attn_hidden),
            nn.ReLU(),
            nn.Linear(attn_hidden, 1),
        )

    @property
    def n_heads(self) -> int:
        return self.task_params.shape[1]

    def _scaled_coords(self, coords: torch.Tensor) -> torch.Tensor:
        return coords / self.coord_scale

    def _scaled_params(self, params: torch.Tensor) -> torch.Tensor:
        return (params - self.param_offset) / self.param_scale.clamp_min(1e-12)

    def encode(self, coords: torch.Tensor, params: torch.Tensor | None = None) -> torch.Tensor:
        xc = self._scaled_coords(coords)
        if self.parametric_trunk:
            if params is None:
                raise ValueError("parametric trunk requires params")
            return self.shared(torch.cat([xc, self._scaled_params(params)], dim=1))
        return self.shared(xc)

    def head_outputs(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(hidden) for head in self.heads], dim=2)

    def voter_logits(self, params: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        n = params.shape[0]
        n_tasks = self.n_heads
        query = params.view(n, 1, self.n_param).expand(n, n_tasks, self.n_param)
        tasks = self.task_params.expand(n, -1, -1)
        pieces = [query, tasks]
        if self.time_conditioned_voter:
            if t is None:
                t = torch.zeros(n, 1, device=params.device, dtype=params.dtype)
            t_norm = (t / max(self.t_max, 1e-12)).view(n, 1, 1).expand(n, n_tasks, 1)
            pieces.append(t_norm)
        attn_in = torch.cat(pieces, dim=2).reshape(n * n_tasks, -1)
        return self.attn_net(attn_in).view(n, n_tasks)

    def get_attention_weights(self, params: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.voter_logits(params, t)
        if self.softmax:
            return torch.softmax(logits, dim=1)
        return logits

    def mix(self, hidden: torch.Tensor, params: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        heads = self.head_outputs(hidden)
        weights = self.get_attention_weights(params, t)
        return (heads * weights.unsqueeze(1)).sum(dim=2)

    def forward(
        self,
        coords: torch.Tensor,
        params: torch.Tensor,
        t: torch.Tensor | None = None,
        head_idx: int | None = None,
    ) -> torch.Tensor:
        hidden = self.encode(coords, params)
        if head_idx is None:
            return self.mix(hidden, params, t)
        return self.heads[head_idx](hidden)

    def expand_voter_for_extra_task(self, extra_param: Sequence[float]) -> None:
        """Append a snapshot μ so a hybrid voter can mix an extra atom."""
        extra = torch.tensor(extra_param, dtype=self.task_params.dtype, device=self.task_params.device)
        extra = extra.view(1, 1, self.n_param)
        self.register_buffer("task_params", torch.cat([self.task_params, extra], dim=1))


def mlp_parameter_count(in_dim: int, hidden_dim: int, layers: int, out_dim: int) -> int:
    n = (in_dim + 1) * hidden_dim
    n += (layers - 1) * (hidden_dim + 1) * hidden_dim
    n += (hidden_dim + 1) * out_dim
    return n


def hidden_dim_for_budget(in_dim: int, layers: int, out_dim: int, target: int) -> int:
    """Hidden width whose MLP size is closest to ``target`` parameters."""
    best_h, best_diff = 8, abs(mlp_parameter_count(in_dim, 8, layers, out_dim) - target)
    for hidden in range(8, 513):
        diff = abs(mlp_parameter_count(in_dim, hidden, layers, out_dim) - target)
        if diff < best_diff:
            best_h, best_diff = hidden, diff
    return best_h


class ParametricPINN(nn.Module):
    """Single network u = net(x, μ). No task heads and no voter.

    Width is chosen so the parameter count is close to the MTL model it is
    compared against. Coordinates and parameters use the same scaling as MTL.
    """

    def __init__(
        self,
        *,
        n_coord: int,
        n_param: int,
        n_fields: int,
        hidden_dim: int,
        layers: int,
        coord_scale: Sequence[float] | None = None,
        param_offset: Sequence[float] | None = None,
        param_scale: Sequence[float] | None = None,
    ):
        super().__init__()
        self.n_coord = n_coord
        self.n_param = n_param
        self.n_fields = n_fields
        self.hidden_dim = hidden_dim
        self.layers = layers
        if coord_scale is None:
            coord_scale = [1.0] * n_coord
        if param_offset is None:
            param_offset = [0.0] * n_param
        if param_scale is None:
            param_scale = [1.0] * n_param
        self.register_buffer("coord_scale", torch.tensor(coord_scale, dtype=torch.float32).view(1, -1))
        self.register_buffer("param_offset", torch.tensor(param_offset, dtype=torch.float32).view(1, -1))
        self.register_buffer("param_scale", torch.tensor(param_scale, dtype=torch.float32).view(1, -1))
        in_dim = n_coord + n_param
        modules: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(layers - 1):
            modules.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        modules.append(nn.Linear(hidden_dim, n_fields))
        self.net = nn.Sequential(*modules)

    def forward(self, coords, params, t=None, head_idx=None):
        del t, head_idx
        xc = coords / self.coord_scale
        pc = (params - self.param_offset) / self.param_scale.clamp_min(1e-12)
        return self.net(torch.cat([xc, pc], dim=1))
