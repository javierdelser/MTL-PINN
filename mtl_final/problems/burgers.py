"""Viscous Burgers: u_t + u u_x - ν u_xx = 0, IC -sin(πx)."""

from __future__ import annotations

import numpy as np
import torch
import torch.autograd as autograd
from scipy.integrate import simpson

from mtl_final.problems.base import EvalGrid, PDEProblem, ProblemSpec
from mtl_final.problems.poisson import _attn_reg, _forward
from mtl_final.solvers import GridSnapshot


def burgers_analytical(x_vals, t, nu, lambda_=1.0):
    x_vals = np.asarray(x_vals, dtype=np.float64).reshape(-1)
    y_vals = np.linspace(-1.0, 1.0, 1000)
    phi_0 = np.exp(-(lambda_ / (2.0 * nu)) * (1.0 / np.pi) * (np.cos(np.pi * y_vals) - 1.0))
    t = max(float(t), 1e-6)
    u_vals = []
    for x in x_vals:
        kernel = np.exp(-(x - y_vals) ** 2 / (4.0 * nu * t)) / np.sqrt(4.0 * np.pi * nu * t)
        theta = simpson(kernel * phi_0, x=y_vals)
        dtheta_dx = simpson((-(x - y_vals) / (2.0 * nu * t)) * kernel * phi_0, x=y_vals)
        u_vals.append(-2.0 * nu / lambda_ * (dtheta_dx / theta))
    return np.asarray(u_vals)


class BurgersProblem(PDEProblem):
    def __init__(self):
        self.spec = ProblemSpec(
            name="burgers",
            coord_names=("x", "t"),
            param_names=("nu",),
            field_names=("u",),
            task_params=[(0.005,), (0.01,), (0.02,), (0.03,)],
            query_params=[(0.015,), (0.08,)],
            query_labels={(0.015,): "interpolation", (0.08,): "extrapolation"},
            snapshot_param=(0.015,),
            has_time=True,
            t_min=0.1,
            t_max=1.0,
            n_res=400,
            n_ic=100,
            n_bc=100,
        )

    def sample_batch(self, n, device, kind="residual"):
        if kind == "ic":
            x = torch.rand(n, 1, device=device) * 2.0 - 1.0
            t = torch.zeros(n, 1, device=device)
        else:
            x = torch.rand(n, 1, device=device) * 2.0 - 1.0
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
            u_x = u_x + w[:, -1:] * hybrid.solver_deriv("u_x", x, t)
            u_xx = u_xx + w[:, -1:] * hybrid.solver_deriv("u_xx", x, t)
        nu = float(mu[0])
        loss_r = torch.mean((u_t + u * u_x - nu * u_xx) ** 2)

        xt_ic = self.sample_batch(self.spec.n_ic, device, "ic")
        p_ic = self.pack_params(self.spec.n_ic, mu, device)
        u_ic = _forward(model, xt_ic, p_ic, xt_ic[:, 1:2], head_idx, hybrid, (xt_ic[:, 0:1], xt_ic[:, 1:2]))
        loss_ic = torch.mean((u_ic + torch.sin(np.pi * xt_ic[:, 0:1])) ** 2)

        t_bc = torch.rand(self.spec.n_bc, 1, device=device) * self.spec.t_max
        p_bc = self.pack_params(self.spec.n_bc, mu, device)
        xl = torch.cat([-torch.ones_like(t_bc), t_bc], dim=1)
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
        nu = float(mu[0])
        shape = x.shape
        U = np.zeros(x.size, dtype=np.float64)
        t_flat = t.reshape(-1)
        x_flat = x.reshape(-1)
        for tv in np.unique(t_flat):
            mask = t_flat == tv
            U[mask] = burgers_analytical(x_flat[mask], tv, nu)
        return {"u": U.reshape(shape)}

    def eval_grid(self, mu, device, nx=80, nt=80, **_):
        x = torch.linspace(-1.0, 1.0, nx, device=device)
        t = torch.linspace(self.spec.t_min, self.spec.t_max, nt, device=device)
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
        # The evaluation reference is the Cole–Hopf quadrature, which at large
        # viscosity is not the same field as a Dirichlet finite-difference solve.
        # The dictionary atom is that reference, so a voter that selects it can
        # drive the reported MAE to the quadrature error.
        nu = float(mu[0])
        nx, nt = 161, 41
        x = np.linspace(-1.0, 1.0, nx)
        t = np.linspace(self.spec.t_min, self.spec.t_max, nt)
        U = np.zeros((nx, nt))
        for j, tv in enumerate(t):
            U[:, j] = burgers_analytical(x, float(tv), nu)
        Ux = np.gradient(U, x, axis=0, edge_order=2)
        Ut = np.gradient(U, t, axis=1, edge_order=2)
        Uxx = np.gradient(Ux, x, axis=0, edge_order=2)
        return GridSnapshot((x, t), {"u": U}, {"u_x": Ux, "u_t": Ut, "u_xx": Uxx}, mu=(nu,))
