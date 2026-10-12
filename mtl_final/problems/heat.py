"""1-D heat: u_t = p u_xx, u(x,0)=sin(πx), Dirichlet 0."""

from __future__ import annotations

import numpy as np
import torch
import torch.autograd as autograd

from mtl_final.problems.base import EvalGrid, PDEProblem, ProblemSpec
from mtl_final.problems.poisson import _attn_reg, _forward
from mtl_final.solvers import heat_crank_nicolson


class HeatProblem(PDEProblem):
    def __init__(self):
        self.spec = ProblemSpec(
            name="heat",
            coord_names=("x", "t"),
            param_names=("p",),
            field_names=("u",),
            task_params=[(0.05,), (0.1,), (0.2,), (0.3,)],
            query_params=[(0.15,), (0.8,)],
            query_labels={(0.15,): "interpolation", (0.8,): "extrapolation"},
            snapshot_param=(0.15,),
            has_time=True,
            t_max=1.0,
            n_res=400,
            n_ic=100,
            n_bc=100,
        )

    def sample_batch(self, n, device, kind="residual"):
        if kind == "ic":
            x = torch.rand(n, 1, device=device)
            t = torch.zeros(n, 1, device=device)
        elif kind == "bc":
            t = torch.rand(n, 1, device=device) * self.spec.t_max
            x = torch.randint(0, 2, (n, 1), device=device, dtype=torch.float32)
        else:
            x = torch.rand(n, 1, device=device)
            t = torch.rand(n, 1, device=device) * self.spec.t_max
        return torch.cat([x, t], dim=1)

    def physics_loss(self, model, mu, device, *, head_idx=None, attn_reg=0.0, task_index=None, hybrid=None):
        xt = self.sample_batch(self.spec.n_res, device)
        (x, t), xt = self.split_grad_coords(xt)
        p = self.pack_params(self.spec.n_res, mu, device)
        u = _forward(model, xt, p, t, head_idx, hybrid, (x, t))
        ones = torch.ones_like(u)
        u_x = autograd.grad(u, x, ones, create_graph=True)[0]
        u_t = autograd.grad(u, t, ones, create_graph=True)[0]
        u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        if hybrid is not None:
            w = hybrid.get_attention_weights(p, t)
            u_t = u_t + w[:, -1:] * hybrid.solver_deriv("u_t", x, t)
            u_xx = u_xx + w[:, -1:] * hybrid.solver_deriv("u_xx", x, t)
        loss_r = torch.mean((u_t - float(mu[0]) * u_xx) ** 2)

        xt_ic = self.sample_batch(self.spec.n_ic, device, "ic")
        p_ic = self.pack_params(self.spec.n_ic, mu, device)
        u_ic = _forward(model, xt_ic, p_ic, xt_ic[:, 1:2], head_idx, hybrid, (xt_ic[:, 0:1], xt_ic[:, 1:2]))
        loss_ic = torch.mean((u_ic - torch.sin(np.pi * xt_ic[:, 0:1])) ** 2)

        t_bc = torch.rand(self.spec.n_bc, 1, device=device) * self.spec.t_max
        p_bc = self.pack_params(self.spec.n_bc, mu, device)
        xl = torch.cat([torch.zeros_like(t_bc), t_bc], dim=1)
        xr = torch.cat([torch.ones_like(t_bc), t_bc], dim=1)
        u_l = _forward(model, xl, p_bc, t_bc, head_idx, hybrid, (xl[:, 0:1], t_bc))
        u_r = _forward(model, xr, p_bc, t_bc, head_idx, hybrid, (xr[:, 0:1], t_bc))
        loss_bc = torch.mean(u_l ** 2 + u_r ** 2)

        attn = _attn_reg(model, p, t, attn_reg, task_index, head_idx)
        physics = loss_r + loss_ic + loss_bc
        return physics + attn, {"physics": physics.detach(), "attn_reg": attn.detach()}

    def reference(self, coords, mu):
        x = np.asarray(coords["x"])
        t = np.asarray(coords["t"])
        return {"u": np.sin(np.pi * x) * np.exp(-(np.pi ** 2) * float(mu[0]) * t)}

    def eval_grid(self, mu, device, nx=80, nt=80, **_):
        x = torch.linspace(0.0, 1.0, nx, device=device)
        t = torch.linspace(0.0, self.spec.t_max, nt, device=device)
        X, T = torch.meshgrid(x, t, indexing="ij")
        coords = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)
        p = self.pack_params(coords.shape[0], mu, device)
        return EvalGrid(
            coords={"x": X.detach().cpu().numpy(), "t": T.detach().cpu().numpy()},
            coord_tensors=(coords,),
            param_tensor=p,
            time_tensor=coords[:, 1:2],
            shape=(nx, nt),
        )

    def numerical_snapshot(self, mu):
        return heat_crank_nicolson(float(mu[0]), t_max=self.spec.t_max)
