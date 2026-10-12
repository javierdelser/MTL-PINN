"""Poisson / λx²: -u_xx + 2μ = 0, u(x;μ)=μ x² on [-1,1]."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.autograd as autograd

from mtl_final.problems.base import EvalGrid, PDEProblem, ProblemSpec
from mtl_final.solvers import poisson_fd


class PoissonProblem(PDEProblem):
    def __init__(self):
        self.spec = ProblemSpec(
            name="poisson",
            coord_names=("x",),
            param_names=("mu",),
            field_names=("u",),
            task_params=[(0.5,), (1.5,), (2.5,), (3.5,)],
            query_params=[(2.0,), (8.0,)],
            query_labels={(2.0,): "interpolation", (8.0,): "extrapolation"},
            snapshot_param=(2.0,),
            has_time=False,
            hidden_dim=64,
            layers=3,
            attn_hidden=32,
            n_res=100,
            n_ic=0,
            n_bc=100,
        )

    def sample_batch(self, n, device, kind="residual"):
        if kind == "left":
            return -torch.ones(n, 1, device=device)
        if kind == "right":
            return torch.ones(n, 1, device=device)
        return torch.rand(n, 1, device=device) * 2.0 - 1.0

    def physics_loss(self, model, mu, device, *, head_idx=None, attn_reg=0.0, task_index=None, hybrid=None):
        n_res, n_bc = self.spec.n_res, self.spec.n_bc
        x = self.sample_batch(n_res, device).requires_grad_(True)
        p = self.pack_params(n_res, mu, device)
        u = _forward(model, x, p, None, head_idx, hybrid, (x,))
        u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        if hybrid is not None:
            w = hybrid.get_attention_weights(p, None)
            u_xx = u_xx + w[:, -1:] * hybrid.solver_deriv("u_xx", x)
        loss_r = torch.mean((-u_xx + 2.0 * p) ** 2)

        x_l = self.sample_batch(n_bc, device, "left")
        p_bc = self.pack_params(n_bc, mu, device)
        u_l = _forward(model, x_l, p_bc, None, head_idx, hybrid, (x_l,))
        loss_left = torch.mean((u_l - p_bc) ** 2)

        x_r = self.sample_batch(n_bc, device, "right").requires_grad_(True)
        u_r = _forward(model, x_r, p_bc, None, head_idx, hybrid, (x_r,))
        u_rx = autograd.grad(u_r, x_r, torch.ones_like(u_r), create_graph=True)[0]
        if hybrid is not None:
            w_bc = hybrid.get_attention_weights(p_bc, None)
            u_rx = u_rx + w_bc[:, -1:] * hybrid.solver_deriv("u_x", x_r)
        loss_right = torch.mean((u_rx - 2.0 * p_bc) ** 2)

        attn = _attn_reg(model, p, None, attn_reg, task_index, head_idx)
        physics = loss_r + loss_left + loss_right
        return physics + attn, {"physics": physics.detach(), "attn_reg": attn.detach()}

    def reference(self, coords, mu):
        x = np.asarray(coords["x"])
        return {"u": float(mu[0]) * x ** 2}

    def eval_grid(self, mu, device, nx=100, **_):
        x = torch.linspace(-1.0, 1.0, nx, device=device).view(-1, 1)
        p = self.pack_params(nx, mu, device)
        x_np = x.detach().cpu().numpy().reshape(-1)
        return EvalGrid(
            coords={"x": x_np},
            coord_tensors=(x,),
            param_tensor=p,
            time_tensor=None,
            shape=(nx,),
        )

    def numerical_snapshot(self, mu):
        return poisson_fd(float(mu[0]))


def _forward(model, coords, params, t, head_idx, hybrid, solver_coords):
    if hybrid is not None:
        return hybrid(coords, params, t=t, solver_coords=solver_coords, head_idx=head_idx)
    return model(coords, params, t=t, head_idx=head_idx)


def _attn_reg(model, params, t, attn_reg, task_index, head_idx):
    if attn_reg <= 0 or task_index is None or head_idx is not None:
        return torch.zeros((), device=params.device)
    weights = model.get_attention_weights(params, t)
    target = torch.zeros_like(weights)
    target[:, task_index] = 1.0
    return attn_reg * torch.mean((weights - target) ** 2)
