"""Classical numerical snapshots used as extra MTL dictionary atoms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import solve_banded


@dataclass
class GridSnapshot:
    """Interpolant of a precomputed numerical field (and optional derivatives)."""

    axes: tuple[np.ndarray, ...]
    fields: dict[str, np.ndarray]
    derivs: dict[str, np.ndarray] | None = None
    mu: tuple[float, ...] = ()

    def __post_init__(self):
        self._interps: dict[str, RegularGridInterpolator] = {}
        payload = dict(self.fields)
        if self.derivs:
            payload.update(self.derivs)
        for name, values in payload.items():
            self._interps[name] = RegularGridInterpolator(
                self.axes,
                np.asarray(values, dtype=np.float64),
                bounds_error=False,
                fill_value=None,
            )

    def eval(self, name: str, *coords: torch.Tensor) -> torch.Tensor:
        pts = np.stack([c.detach().cpu().numpy().reshape(-1) for c in coords], axis=1)
        vals = self._interps[name](pts)
        ref = coords[0]
        return torch.tensor(vals, device=ref.device, dtype=ref.dtype).reshape(-1, 1)


def poisson_fd(mu: float, nx: int = 401) -> GridSnapshot:
    x = np.linspace(-1.0, 1.0, nx)
    h = x[1] - x[0]
    n = nx
    rhs = np.full(n, 2.0 * mu * h * h)
    rhs[0] = mu
    mat = np.zeros((n, n))
    mat[0, 0] = 1.0
    for i in range(1, n - 1):
        mat[i, i - 1] = 1.0
        mat[i, i] = -2.0
        mat[i, i + 1] = 1.0
    mat[-1, -1] = 3.0
    mat[-1, -2] = -4.0
    mat[-1, -3] = 1.0
    rhs[-1] = 4.0 * mu * h
    u = np.linalg.solve(mat, rhs)
    ux = np.gradient(u, x, edge_order=2)
    uxx = np.gradient(ux, x, edge_order=2)
    return GridSnapshot((x,), {"u": u}, {"u_x": ux, "u_xx": uxx}, mu=(mu,))


def heat_crank_nicolson(p: float, nx: int = 257, nt: int = 257, t_max: float = 1.0) -> GridSnapshot:
    x = np.linspace(0.0, 1.0, nx)
    t = np.linspace(0.0, t_max, nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    r = p * dt / (2.0 * dx * dx)
    interior = nx - 2
    ab = np.zeros((3, interior))
    ab[0, 1:] = -r
    ab[1, :] = 1.0 + 2.0 * r
    ab[2, :-1] = -r
    u = np.sin(np.pi * x)
    U = np.zeros((nx, nt))
    U[:, 0] = u
    for k in range(nt - 1):
        rhs = (1.0 - 2.0 * r) * u[1:-1] + r * u[:-2] + r * u[2:]
        u[1:-1] = solve_banded((1, 1), ab, rhs)
        u[0] = 0.0
        u[-1] = 0.0
        U[:, k + 1] = u
    Ux = np.gradient(U, x, axis=0, edge_order=2)
    Ut = np.gradient(U, t, axis=1, edge_order=2)
    Uxx = np.gradient(Ux, x, axis=0, edge_order=2)
    return GridSnapshot((x, t), {"u": U}, {"u_x": Ux, "u_t": Ut, "u_xx": Uxx}, mu=(p,))


def burgers_rk4(nu: float, nx: int = 401, t_max: float = 1.0, cfl: float = 0.35) -> GridSnapshot:
    x = np.linspace(-1.0, 1.0, nx)
    dx = x[1] - x[0]
    dt = min(cfl * dx * dx / max(2.0 * nu, 1e-12), 0.45 * dx, 2.5e-3)
    nsteps = int(np.ceil(t_max / dt))
    dt = t_max / nsteps
    t = np.linspace(0.0, t_max, nsteps + 1)
    u = -np.sin(np.pi * x)
    u[0] = u[-1] = 0.0
    U = np.zeros((nx, nsteps + 1))
    U[:, 0] = u

    def rhs(state):
        dudt = np.zeros_like(state)
        ux = (state[2:] - state[:-2]) / (2.0 * dx)
        uxx = (state[2:] - 2.0 * state[1:-1] + state[:-2]) / (dx * dx)
        dudt[1:-1] = -state[1:-1] * ux + nu * uxx
        return dudt

    for step in range(nsteps):
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        u[0] = u[-1] = 0.0
        U[:, step + 1] = u
    Ux = np.gradient(U, x, axis=0, edge_order=2)
    Ut = np.gradient(U, t, axis=1, edge_order=2)
    Uxx = np.gradient(Ux, x, axis=0, edge_order=2)
    return GridSnapshot((x, t), {"u": U}, {"u_x": Ux, "u_t": Ut, "u_xx": Uxx}, mu=(nu,))


def fisher_kpp_exact(r: float, D: float = 1.0, L: float = 10.0, t_max: float = 3.0, nx: int = 97, nt: int = 49) -> GridSnapshot:
    """Tabulated Ablowitz–Zeppetella wave.

    The explicit RK4 grid is unstable at the diffusivities and front speeds
    used here, so the dictionary atom is the exact field and its derivatives.
    """
    x = np.linspace(-L, L, nx)
    y = np.linspace(-L, L, nx)
    t = np.linspace(0.0, t_max, nt)
    X, Y, T = np.meshgrid(x, y, t, indexing="ij")
    k = np.sqrt(r / D) / (2.0 * np.sqrt(3.0))
    omega = 5.0 * r / 6.0
    s = k * (X + Y) - omega * T
    e = np.exp(s)
    den = 1.0 + e
    U = den ** (-2)
    du = -2.0 * e * den ** (-3)
    d2 = -2.0 * e * (1.0 - 2.0 * e) * den ** (-4)
    return GridSnapshot(
        (x, y, t),
        {"u": U},
        {
            "u_x": du * k,
            "u_y": du * k,
            "u_t": du * (-omega),
            "u_xx": d2 * k * k,
            "u_yy": d2 * k * k,
        },
        mu=(r,),
    )


def fisher_kpp_rk4(r: float, D: float = 1.0, L: float = 10.0, t_max: float = 3.0, nx: int = 81, nt: int = 121) -> GridSnapshot:
    x = np.linspace(-L, L, nx)
    y = np.linspace(-L, L, nx)
    t = np.linspace(0.0, t_max, nt)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    k = np.sqrt(r / D) / (2.0 * np.sqrt(3.0))
    omega = 5.0 * r / 6.0

    def exact(tt):
        return (1.0 + np.exp(k * (X + Y) - omega * tt)) ** (-2)

    U = np.zeros((nx, nx, nt))
    U[:, :, 0] = exact(0.0)
    dt = t[1] - t[0]

    def lap(field):
        out = np.zeros_like(field)
        out[1:-1, 1:-1] = (
            field[2:, 1:-1] + field[:-2, 1:-1] + field[1:-1, 2:] + field[1:-1, :-2] - 4.0 * field[1:-1, 1:-1]
        ) / (dx * dx)
        return out

    def rhs(state):
        return D * lap(state) + r * state * (1.0 - state)

    u = U[:, :, 0].copy()
    for n in range(nt - 1):
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        u[0, :] = exact(t[n + 1])[0, :]
        u[-1, :] = exact(t[n + 1])[-1, :]
        u[:, 0] = exact(t[n + 1])[:, 0]
        u[:, -1] = exact(t[n + 1])[:, -1]
        U[:, :, n + 1] = u
    Ux = np.gradient(U, x, axis=0, edge_order=2)
    Uy = np.gradient(U, y, axis=1, edge_order=2)
    Ut = np.gradient(U, t, axis=2, edge_order=2)
    Uxx = np.gradient(Ux, x, axis=0, edge_order=2)
    Uyy = np.gradient(Uy, y, axis=1, edge_order=2)
    return GridSnapshot(
        (x, y, t),
        {"u": U},
        {"u_x": Ux, "u_y": Uy, "u_t": Ut, "u_xx": Uxx, "u_yy": Uyy},
        mu=(r,),
    )


def lamb_chaplygin_tabulated(a: float, U: float, evaluate_fn: Callable, nx: int = 97, nt: int = 33) -> GridSnapshot:
    x = np.linspace(-6.0, 6.0, nx)
    y = np.linspace(-6.0, 6.0, nx)
    t = np.linspace(0.0, 1.0, nt)
    X, Y, T = np.meshgrid(x, y, t, indexing="ij")
    psi, omega = evaluate_fn(X, Y, T, a, U)
    return GridSnapshot((x, y, t), {"psi": psi, "omega": omega}, mu=(a, U))


def save_snapshot(path, snap: GridSnapshot) -> None:
    folder = __import__("pathlib").Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "mu.npy", np.asarray(snap.mu, dtype=np.float64))
    for i, axis in enumerate(snap.axes):
        np.save(folder / f"axis{i}.npy", np.asarray(axis))
    for name, values in snap.fields.items():
        np.save(folder / f"field_{name}.npy", np.asarray(values))
    derivs = snap.derivs or {}
    (folder / "deriv_names.txt").write_text("\n".join(derivs))
    for name, values in derivs.items():
        np.save(folder / f"deriv_{name}.npy", np.asarray(values))


def load_snapshot(path) -> GridSnapshot | None:
    folder = __import__("pathlib").Path(path)
    if not (folder / "mu.npy").exists():
        return None
    mu = tuple(float(v) for v in np.load(folder / "mu.npy"))
    axes = []
    i = 0
    while (folder / f"axis{i}.npy").exists():
        axes.append(np.load(folder / f"axis{i}.npy"))
        i += 1
    fields = {p.name[6:-4]: np.load(p) for p in sorted(folder.glob("field_*.npy"))}
    deriv_file = folder / "deriv_names.txt"
    derivs = {}
    if deriv_file.exists() and deriv_file.read_text().strip():
        for name in deriv_file.read_text().splitlines():
            derivs[name] = np.load(folder / f"deriv_{name}.npy")
    return GridSnapshot(tuple(axes), fields, derivs or None, mu=mu)
