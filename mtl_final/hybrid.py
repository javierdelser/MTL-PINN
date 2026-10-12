"""Frozen PINN dictionary plus one classical-solver snapshot, mixed by a voter."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from mtl_final.models import MTLPINN
from mtl_final.solvers import GridSnapshot


class HybridMTL(nn.Module):
    """PINN heads plus one interpolated numerical atom.

    Backbone and PINN heads stay in ``self.mtl``. The voter is expanded by one
    task so it can mix the solver snapshot. Solver outputs are detached.
    """

    def __init__(self, mtl: MTLPINN, snapshot: GridSnapshot, extra_param: Sequence[float], field_names: Sequence[str]):
        super().__init__()
        self.mtl = mtl
        self.snapshot = snapshot
        self.field_names = list(field_names)
        self.extra_param = tuple(float(v) for v in extra_param)
        self.n_pinn_heads = len(mtl.heads)
        if mtl.task_params.shape[1] == self.n_pinn_heads:
            mtl.expand_voter_for_extra_task(self.extra_param)
        self._freeze_pinn_dictionary()

    def _freeze_pinn_dictionary(self) -> None:
        for param in self.mtl.shared.parameters():
            param.requires_grad = False
        for head in self.mtl.heads:
            for param in head.parameters():
                param.requires_grad = False
        self.mtl.shared.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.mtl.shared.eval()
        return self

    def get_attention_weights(self, params: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        return self.mtl.get_attention_weights(params, t)

    def solver_fields(self, *coord_tensors: torch.Tensor) -> torch.Tensor:
        cols = [self.snapshot.eval(name, *coord_tensors) for name in self.field_names]
        return torch.cat(cols, dim=1)

    def solver_deriv(self, name: str, *coord_tensors: torch.Tensor) -> torch.Tensor:
        return self.snapshot.eval(name, *coord_tensors)

    def pinn_fields(self, coords: torch.Tensor, params: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.mtl.encode(coords, params)
        heads = self.mtl.head_outputs(hidden)
        return heads

    def forward(
        self,
        coords: torch.Tensor,
        params: torch.Tensor,
        t: torch.Tensor | None = None,
        solver_coords: tuple[torch.Tensor, ...] | None = None,
        head_idx: int | None = None,
    ) -> torch.Tensor:
        if head_idx is not None:
            return self.mtl(coords, params, t=t, head_idx=head_idx)
        hidden = self.mtl.encode(coords, params)
        pinn = self.mtl.head_outputs(hidden)
        if solver_coords is None:
            ndim = len(self.snapshot.axes)
            solver_coords = tuple(coords[:, i : i + 1] for i in range(ndim))
        snap = self.solver_fields(*solver_coords)
        fields = torch.cat([pinn, snap.unsqueeze(2)], dim=2)
        weights = self.get_attention_weights(params, t)
        return (fields * weights.unsqueeze(1)).sum(dim=2)

    @property
    def shared(self):
        return self.mtl.shared

    @property
    def heads(self):
        return self.mtl.heads

    @property
    def attn_net(self):
        return self.mtl.attn_net

    @property
    def task_params(self):
        return self.mtl.task_params

    @property
    def softmax(self):
        return self.mtl.softmax

    @property
    def time_conditioned_voter(self):
        return self.mtl.time_conditioned_voter

    @property
    def parametric_trunk(self):
        return self.mtl.parametric_trunk

    @property
    def n_fields(self):
        return self.mtl.n_fields

    def encode(self, coords, params=None):
        return self.mtl.encode(coords, params)

    def voter_logits(self, params, t=None):
        return self.mtl.voter_logits(params, t)
