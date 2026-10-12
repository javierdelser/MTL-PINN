"""Lab-frame Lamb–Chaplygin dipole. Parameters (a, U). Trunk on (x, y, t)."""

from __future__ import annotations

import numpy as np
import torch
import torch.autograd as autograd

from mtl_final.problems.base import EvalGrid, PDEProblem, ProblemSpec
from mtl_final.problems.poisson import _attn_reg, _forward
from mtl_final.solvers import lamb_chaplygin_tabulated
from mtl_final.utils import as_col

CL = 3.8317059702075125
DEFAULT_ALPHA = 0.0
DEFAULT_X0 = -1.0
DEFAULT_Y0 = 0.0
X_MIN, X_MAX = -6.0, 6.0
Y_MIN, Y_MAX = -6.0, 6.0
T_MAX = 1.0
RHO_PDE_GAP = 0.03
INTERFACE_BAND = 0.05
LAMBDA_IFACE = 1.0
LAMBDA_IC = 1.0
LAMBDA_BC = 1.0


class _BesselJ0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        ctx.save_for_backward(z)
        return torch.special.bessel_j0(z)

    @staticmethod
    def backward(ctx, grad_output):
        (z,) = ctx.saved_tensors
        return grad_output * (-bessel_j1(z))


class _BesselJ1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        ctx.save_for_backward(z)
        return torch.special.bessel_j1(z)

    @staticmethod
    def backward(ctx, grad_output):
        (z,) = ctx.saved_tensors
        small = z.abs() < 1e-6
        z_safe = torch.where(small, torch.ones_like(z), z)
        deriv = torch.where(small, torch.full_like(z, 0.5), bessel_j0(z) - bessel_j1(z) / z_safe)
        return grad_output * deriv


def bessel_j0(z):
    return _BesselJ0.apply(z)


def bessel_j1(z):
    return _BesselJ1.apply(z)


def j1_over_rho(rho, cl=CL, s=None):
    if s is None:
        s = rho * rho
    series = 0.5 * cl - (cl ** 3) * s / 16.0 + (cl ** 5) * s * s / 384.0
    rho_safe = torch.sqrt(s.clamp(min=1e-8))
    direct = bessel_j1(cl * rho_safe) / rho_safe
    return torch.where(s < 1e-6, series, direct)


def body_frame(x, y, t, a, U, alpha=DEFAULT_ALPHA, x0=DEFAULT_X0, y0=DEFAULT_Y0):
    a_t, U_t = as_col(a, x), as_col(U, x)
    alpha_t = as_col(alpha, x)
    ca, sa = torch.cos(alpha_t), torch.sin(alpha_t)
    xc = as_col(x0, x) + U_t * t * ca
    yc = as_col(y0, x) + U_t * t * sa
    dx, dy = x - xc, y - yc
    xi = (dx * ca + dy * sa) / a_t
    eta = (-dx * sa + dy * ca) / a_t
    s = xi * xi + eta * eta
    rho = torch.sqrt(s.clamp_min(0.0))
    return {"xi": xi, "eta": eta, "s": s, "rho": rho, "dx": dx, "dy": dy, "a": a_t, "U": U_t, "ca": ca, "sa": sa}


def lab_from_body(xi, eta, t, a, U, alpha=DEFAULT_ALPHA, x0=DEFAULT_X0, y0=DEFAULT_Y0):
    a_t, U_t = as_col(a, xi), as_col(U, xi)
    alpha_t = as_col(alpha, xi)
    ca, sa = torch.cos(alpha_t), torch.sin(alpha_t)
    xc = as_col(x0, xi) + U_t * t * ca
    yc = as_col(y0, xi) + U_t * t * sa
    x = xc + a_t * (xi * ca - eta * sa)
    y = yc + a_t * (xi * sa + eta * ca)
    return x, y


def analytical_psi_omega(x, y, t, a, U, alpha=DEFAULT_ALPHA, x0=DEFAULT_X0, y0=DEFAULT_Y0, region="auto"):
    frame = body_frame(x, y, t, a, U, alpha, x0, y0)
    eta, s, rho, a_t, U_t = frame["eta"], frame["s"], frame["rho"], frame["a"], frame["U"]
    j0c = torch.special.bessel_j0(torch.tensor(CL, dtype=x.dtype, device=x.device))
    ratio = j1_over_rho(rho, s=s)
    psi_in = U_t * a_t * eta * (1.0 - 2.0 * ratio / (CL * j0c))
    omega_in = -(U_t / a_t) * eta * (2.0 * CL * ratio / j0c)
    psi_out = U_t * a_t * eta / s.clamp_min(1e-12)
    omega_out = torch.zeros_like(psi_out)
    if region == "inside":
        return psi_in, omega_in, frame
    if region == "outside":
        return psi_out, omega_out, frame
    inside = rho < 1.0
    return torch.where(inside, psi_in, psi_out), torch.where(inside, omega_in, omega_out), frame


def analytical_numpy(X, Y, T, a, U):
    xt = torch.as_tensor(np.asarray(X), dtype=torch.float64).reshape(-1, 1)
    yt = torch.as_tensor(np.asarray(Y), dtype=torch.float64).reshape(-1, 1)
    tt = torch.as_tensor(np.asarray(T), dtype=torch.float64).reshape(-1, 1)
    psi, omega, _ = analytical_psi_omega(xt, yt, tt, a, U)
    shape = np.asarray(X).shape
    return psi.numpy().reshape(shape), omega.numpy().reshape(shape)


def _grad(output, inputs):
    if not output.requires_grad:
        return torch.zeros_like(inputs)
    g = autograd.grad(output, inputs, torch.ones_like(output), create_graph=True, retain_graph=True, allow_unused=True)[0]
    return torch.zeros_like(inputs) if g is None else g


def pde_residuals(psi, omega, x, y, t):
    psi_x, psi_y = _grad(psi, x), _grad(psi, y)
    omega_x, omega_y, omega_t = _grad(omega, x), _grad(omega, y), _grad(omega, t)
    psi_xx, psi_yy = _grad(psi_x, x), _grad(psi_y, y)
    u, v = psi_y, -psi_x
    return omega_t + u * omega_x + v * omega_y, omega + psi_xx + psi_yy


def _uniform(n, low, high, device):
    return torch.rand((n, 1), device=device) * (high - low) + low


class LambChaplyginProblem(PDEProblem):
    def __init__(self):
        self.spec = ProblemSpec(
            name="lamb_chaplygin",
            coord_names=("x", "y", "t"),
            param_names=("a", "U"),
            field_names=("psi", "omega"),
            task_params=[(0.8, 0.5), (0.8, 1.0), (1.2, 0.5), (1.2, 1.0)],
            query_params=[(1.0, 0.75), (1.6, 1.4)],
            query_labels={(1.0, 0.75): "interpolation", (1.6, 1.4): "extrapolation"},
            snapshot_param=(1.0, 0.75),
            has_time=True,
            t_max=T_MAX,
            n_res=400,
            n_ic=100,
            n_bc=100,
        )

    def sample_interior(self, n, a, U, device):
        t = _uniform(n, 0.0, T_MAX, device)
        rho = torch.sqrt(torch.rand((n, 1), device=device)) * (1.0 - RHO_PDE_GAP)
        theta = _uniform(n, 0.0, 2.0 * np.pi, device)
        x, y = lab_from_body(rho * torch.cos(theta), rho * torch.sin(theta), t, a, U)
        return torch.cat([x, y, t], dim=1)

    def sample_exterior(self, n, a, U, device):
        xs, ys, ts = [], [], []
        remaining = n
        while remaining > 0:
            m = max(remaining * 3, remaining)
            x = _uniform(m, X_MIN, X_MAX, device)
            y = _uniform(m, Y_MIN, Y_MAX, device)
            t = _uniform(m, 0.0, T_MAX, device)
            mask = (body_frame(x, y, t, a, U)["rho"] > 1.0 + RHO_PDE_GAP).squeeze(1)
            if mask.any():
                xs.append(x[mask]); ys.append(y[mask]); ts.append(t[mask])
                remaining = n - sum(v.shape[0] for v in xs)
        return torch.cat([torch.cat(xs)[:n], torch.cat(ys)[:n], torch.cat(ts)[:n]], dim=1)

    def sample_interface(self, n, a, U, device):
        t = _uniform(n, 0.0, T_MAX, device)
        theta = _uniform(n, 0.0, 2.0 * np.pi, device)
        x, y = lab_from_body(torch.cos(theta), torch.sin(theta), t, a, U)
        return torch.cat([x, y, t], dim=1)

    def sample_boundary(self, n, device):
        n_side = max(n // 4, 1)
        t = _uniform(4 * n_side, 0.0, T_MAX, device)
        s1 = _uniform(n_side, Y_MIN, Y_MAX, device)
        s3 = _uniform(n_side, X_MIN, X_MAX, device)
        x = torch.cat([torch.full((n_side, 1), X_MIN, device=device), torch.full((n_side, 1), X_MAX, device=device), s3, s3])
        y = torch.cat([s1, s1, torch.full((n_side, 1), Y_MIN, device=device), torch.full((n_side, 1), Y_MAX, device=device)])
        return torch.cat([x, y, t], dim=1)

    def sample_initial(self, n, a, U, device):
        n_in = n // 2
        xyz_in = self.sample_interior(n_in, a, U, device)
        xyz_out = self.sample_exterior(n - n_in, a, U, device)
        xyz = torch.cat([xyz_in, xyz_out], dim=0)
        xyz[:, 2:3] = 0.0
        return xyz

    def physics_loss(self, model, mu, device, *, head_idx=None, attn_reg=0.0, task_index=None, hybrid=None):
        a, U = float(mu[0]), float(mu[1])
        xyz = self.sample_interior(self.spec.n_res, a, U, device)
        (x, y, t), xyz = self.split_grad_coords(xyz)
        p = self.pack_params(xyz.shape[0], mu, device)
        fields = _forward(model, xyz, p, t, head_idx, hybrid, (x, y, t))
        psi, omega = fields[:, 0:1], fields[:, 1:2]
        r_tr, r_kin = pde_residuals(psi, omega, x, y, t)
        loss_pde = torch.mean(r_tr ** 2) + torch.mean(r_kin ** 2)

        xyz_e = self.sample_exterior(200, a, U, device)
        (xe, ye, te), xyz_e = self.split_grad_coords(xyz_e)
        p_e = self.pack_params(xyz_e.shape[0], mu, device)
        fe = _forward(model, xyz_e, p_e, te, head_idx, hybrid, (xe, ye, te))
        r_tr_e, r_kin_e = pde_residuals(fe[:, 0:1], fe[:, 1:2], xe, ye, te)
        loss_ext = torch.mean(r_tr_e ** 2) + torch.mean(r_kin_e ** 2)

        xyz_i = self.sample_interface(100, a, U, device)
        (xi, yi, ti), xyz_i = self.split_grad_coords(xyz_i)
        p_i = self.pack_params(xyz_i.shape[0], mu, device)
        fi = _forward(model, xyz_i, p_i, ti, head_idx, hybrid, (xi, yi, ti))
        psi_i, omega_i = fi[:, 0:1], fi[:, 1:2]
        psi_t, omega_t, frame = analytical_psi_omega(xi, yi, ti, a, U, region="outside")
        rho = frame["rho"].clamp_min(1e-8)
        nx = frame["dx"] / (frame["a"] * rho)
        ny = frame["dy"] / (frame["a"] * rho)
        dn_p = _grad(psi_i, xi) * nx + _grad(psi_i, yi) * ny
        dn_t = _grad(psi_t, xi) * nx + _grad(psi_t, yi) * ny
        loss_iface = LAMBDA_IFACE * (
            torch.mean((psi_i - psi_t) ** 2) + torch.mean((dn_p - dn_t) ** 2) + torch.mean(omega_i ** 2)
        )

        xyz_ic = self.sample_initial(self.spec.n_ic, a, U, device)
        p_ic = self.pack_params(xyz_ic.shape[0], mu, device)
        fic = _forward(model, xyz_ic, p_ic, xyz_ic[:, 2:3], head_idx, hybrid, (xyz_ic[:, 0:1], xyz_ic[:, 1:2], xyz_ic[:, 2:3]))
        psi_ic_t, omega_ic_t, _ = analytical_psi_omega(xyz_ic[:, 0:1], xyz_ic[:, 1:2], xyz_ic[:, 2:3], a, U)
        loss_ic = LAMBDA_IC * (
            torch.mean((fic[:, 0:1] - psi_ic_t) ** 2) + torch.mean((fic[:, 1:2] - omega_ic_t) ** 2)
        )

        xyz_bc = self.sample_boundary(self.spec.n_bc, device)
        p_bc = self.pack_params(xyz_bc.shape[0], mu, device)
        fbc = _forward(model, xyz_bc, p_bc, xyz_bc[:, 2:3], head_idx, hybrid, (xyz_bc[:, 0:1], xyz_bc[:, 1:2], xyz_bc[:, 2:3]))
        psi_bc_t, omega_bc_t, _ = analytical_psi_omega(xyz_bc[:, 0:1], xyz_bc[:, 1:2], xyz_bc[:, 2:3], a, U, region="outside")
        loss_bc = LAMBDA_BC * (
            torch.mean((fbc[:, 0:1] - psi_bc_t) ** 2) + torch.mean((fbc[:, 1:2] - omega_bc_t) ** 2)
        )

        attn = _attn_reg(model, p, t, attn_reg, task_index, head_idx)
        physics = loss_pde + loss_ext + loss_iface + loss_ic + loss_bc
        return physics + attn, {"physics": physics.detach(), "attn_reg": attn.detach()}

    def reference(self, coords, mu):
        psi, omega = analytical_numpy(coords["x"], coords["y"], coords["t"], float(mu[0]), float(mu[1]))
        return {"psi": psi, "omega": omega}

    def eval_grid(self, mu, device, nx=48, ny=48, snapshot_time=0.5, **_):
        x = torch.linspace(-3.0, 3.0, nx, device=device)
        y = torch.linspace(-3.0, 3.0, ny, device=device)
        X, Y = torch.meshgrid(x, y, indexing="xy")
        T = torch.full_like(X, float(snapshot_time))
        coords = torch.stack([X.reshape(-1), Y.reshape(-1), T.reshape(-1)], dim=1)
        p = self.pack_params(coords.shape[0], mu, device)
        return EvalGrid(
            coords={"x": X.detach().cpu().numpy(), "y": Y.detach().cpu().numpy(), "t": T.detach().cpu().numpy()},
            coord_tensors=(coords,),
            param_tensor=p,
            time_tensor=coords[:, 2:3],
            shape=(ny, nx),
        )

    def numerical_snapshot(self, mu):
        a, U = float(mu[0]), float(mu[1])
        return lamb_chaplygin_tabulated(a, U, analytical_numpy)
