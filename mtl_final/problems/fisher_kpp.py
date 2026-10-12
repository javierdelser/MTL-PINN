"""2-D Fisher–KPP traveling wave. Parameter r, D=1."""

from __future__ import annotations

import numpy as np
import torch
import torch.autograd as autograd

from mtl_final.problems.base import EvalGrid, PDEProblem, ProblemSpec
from mtl_final.problems.poisson import _attn_reg, _forward
from mtl_final.solvers import fisher_kpp_exact

L_DOMAIN = 10.0
D_FIXED = 1.0


def wave_numbers(r, D=D_FIXED):
    k = np.sqrt(r / D) / (2.0 * np.sqrt(3.0))
    omega = 5.0 * r / 6.0
    return k, omega


def analytical_numpy(x, y, t, r, D=D_FIXED):
    k, omega = wave_numbers(r, D)
    return (1.0 + np.exp(k * (x + y) - omega * t)) ** (-2)


class FisherKPPProblem(PDEProblem):
    def __init__(self):
        self.spec = ProblemSpec(
            name="fisher_kpp",
            coord_names=("x", "y", "t"),
            param_names=("r",),
            field_names=("u",),
            task_params=[(0.5,), (1.0,), (1.5,), (2.0,)],
            query_params=[(1.25,), (3.5,)],
            query_labels={(1.25,): "interpolation", (3.5,): "extrapolation"},
            snapshot_param=(1.25,),
            has_time=True,
            t_max=3.0,
            n_res=400,
            n_ic=200,
            n_bc=200,
        )

    def sample_batch(self, n, device, kind="residual"):
        if kind == "ic":
            x = (torch.rand(n, 1, device=device) * 2 - 1) * L_DOMAIN
            y = (torch.rand(n, 1, device=device) * 2 - 1) * L_DOMAIN
            t = torch.zeros(n, 1, device=device)
            return torch.cat([x, y, t], dim=1)
        if kind == "bc":
            n_edge = max(n // 4, 1)
            t = torch.rand(4 * n_edge, 1, device=device) * self.spec.t_max
            coord = (torch.rand(n_edge, 1, device=device) * 2 - 1) * L_DOMAIN
            x = torch.cat(
                [
                    -L_DOMAIN * torch.ones(n_edge, 1, device=device),
                    L_DOMAIN * torch.ones(n_edge, 1, device=device),
                    coord,
                    coord,
                ]
            )
            y = torch.cat(
                [
                    coord,
                    coord,
                    -L_DOMAIN * torch.ones(n_edge, 1, device=device),
                    L_DOMAIN * torch.ones(n_edge, 1, device=device),
                ]
            )
            return torch.cat([x, y, t], dim=1)
        x = (torch.rand(n, 1, device=device) * 2 - 1) * L_DOMAIN
        y = (torch.rand(n, 1, device=device) * 2 - 1) * L_DOMAIN
        t = torch.rand(n, 1, device=device) * self.spec.t_max
        return torch.cat([x, y, t], dim=1)

    def _exact(self, x, y, t, r):
        k = torch.sqrt(r / D_FIXED) / (2.0 * np.sqrt(3.0))
        omega = 5.0 * r / 6.0
        return (1.0 + torch.exp(k * (x + y) - omega * t)) ** (-2)

    def physics_loss(self, model, mu, device, *, head_idx=None, attn_reg=0.0, task_index=None, hybrid=None):
        xyz = self.sample_batch(self.spec.n_res, device)
        (x, y, t), xyz = self.split_grad_coords(xyz)
        p = self.pack_params(self.spec.n_res, mu, device)
        u = _forward(model, xyz, p, t, head_idx, hybrid, (x, y, t))
        ones = torch.ones_like(u)
        u_x = autograd.grad(u, x, ones, create_graph=True)[0]
        u_y = autograd.grad(u, y, ones, create_graph=True)[0]
        u_t = autograd.grad(u, t, ones, create_graph=True)[0]
        u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        u_yy = autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
        if hybrid is not None:
            w = hybrid.get_attention_weights(p, t)
            u_t = u_t + w[:, -1:] * hybrid.solver_deriv("u_t", x, y, t)
            u_xx = u_xx + w[:, -1:] * hybrid.solver_deriv("u_xx", x, y, t)
            u_yy = u_yy + w[:, -1:] * hybrid.solver_deriv("u_yy", x, y, t)
        r = p
        loss_r = torch.mean((u_t - D_FIXED * (u_xx + u_yy) - r * u * (1.0 - u)) ** 2)

        ic = self.sample_batch(self.spec.n_ic, device, "ic")
        p_ic = self.pack_params(ic.shape[0], mu, device)
        u_ic = _forward(model, ic, p_ic, ic[:, 2:3], head_idx, hybrid, (ic[:, 0:1], ic[:, 1:2], ic[:, 2:3]))
        loss_ic = torch.mean((u_ic - self._exact(ic[:, 0:1], ic[:, 1:2], ic[:, 2:3], p_ic)) ** 2)

        bc = self.sample_batch(self.spec.n_bc, device, "bc")
        p_bc = self.pack_params(bc.shape[0], mu, device)
        u_bc = _forward(model, bc, p_bc, bc[:, 2:3], head_idx, hybrid, (bc[:, 0:1], bc[:, 1:2], bc[:, 2:3]))
        loss_bc = torch.mean((u_bc - self._exact(bc[:, 0:1], bc[:, 1:2], bc[:, 2:3], p_bc)) ** 2)

        attn = _attn_reg(model, p, t, attn_reg, task_index, head_idx)
        physics = loss_r + loss_ic + loss_bc
        return physics + attn, {"physics": physics.detach(), "attn_reg": attn.detach()}

    def reference(self, coords, mu):
        return {"u": analytical_numpy(np.asarray(coords["x"]), np.asarray(coords["y"]), np.asarray(coords["t"]), float(mu[0]))}

    def eval_grid(self, mu, device, nx=48, ny=48, snapshot_time=1.5, **_):
        x = torch.linspace(-L_DOMAIN, L_DOMAIN, nx, device=device)
        y = torch.linspace(-L_DOMAIN, L_DOMAIN, ny, device=device)
        X, Y = torch.meshgrid(x, y, indexing="xy")
        T = torch.full_like(X, float(snapshot_time))
        coords = torch.stack([X.reshape(-1), Y.reshape(-1), T.reshape(-1)], dim=1)
        p = self.pack_params(coords.shape[0], mu, device)
        return EvalGrid(
            coords={
                "x": X.detach().cpu().numpy(),
                "y": Y.detach().cpu().numpy(),
                "t": T.detach().cpu().numpy(),
            },
            coord_tensors=(coords,),
            param_tensor=p,
            time_tensor=coords[:, 2:3],
            shape=(ny, nx),
        )

    def numerical_snapshot(self, mu):
        return fisher_kpp_exact(float(mu[0]), t_max=self.spec.t_max)
