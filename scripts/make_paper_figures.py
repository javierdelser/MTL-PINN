#!/usr/bin/env python3
"""Reproduce the manuscript figures in outputs/paper_figures/.

Reads the stored seed-50 checkpoints and CSVs. Does not retrain.

    python scripts/make_paper_figures.py --all
    python scripts/make_paper_figures.py --fig f1 f3 weights

Keys: f1 Fisher–KPP solutions, f2 Lamb–Chaplygin solutions, f3 generalization
curves, f4 hybrid curves, f5 learned heads, f6 query fields, rq2all combined
prediction and error, f7 voter-only loss, d1 voter-weight CSV, d2 explained
variance CSV, weights voter-weight bars.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtl_final.config import RunSpec
from mtl_final.evaluate import _predict
from mtl_final.problems import get_problem
from mtl_final.problems.fisher_kpp import analytical_numpy as fkpp_exact
from mtl_final.problems.lamb_chaplygin import (
    DEFAULT_ALPHA,
    DEFAULT_X0,
    DEFAULT_Y0,
    analytical_numpy as lc_exact,
)
from mtl_final.train import build_baseline, load_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID = {"nx": 41, "ny": 41, "nt": 21}
COMP = ROOT / "outputs" / "comparison"
HYB = ROOT / "outputs" / "hybrid"

OUT = ROOT / "outputs" / "paper_figures"
CURVES = ROOT / "outputs" / "generalization" / "curves.csv"
CHECK_REFS = True

PDES = ("poisson", "heat", "burgers", "fisher_kpp", "lamb_chaplygin")
TITLES = {
    "poisson": "Poisson",
    "heat": "Heat",
    "burgers": "Burgers",
    "fisher_kpp": "Fisher–KPP",
    "lamb_chaplygin": "Lamb–Chaplygin",
}
XLABELS = {
    "poisson": r"Query parameter $\lambda$",
    "heat": r"Query parameter $\rho$",
    "burgers": r"Query parameter $\nu$",
    "fisher_kpp": r"Query parameter $r$",
    "lamb_chaplygin": r"Path coordinate $s$ along $(a,U)$",
}

# Okabe–Ito
C_ONE = "#0072B2"
C_TWO = "#009E73"
C_PINN = "#000000"
C_HY_V = "#D55E00"
C_HY_VE = "#E69F00"
C_HY_1 = "#CC79A7"
C_HY_2 = "#56B4E9"
C_HY_F = "#0072B2"
C_INTERP = "#0072B2"
C_EXTRAP = "#D55E00"
C_TRAIN = "#999999"
SOL_CMAP = matplotlib.cm.RdBu_r
NEW_PDES = {"fisher_kpp", "lamb_chaplygin"}

STYLE = {
    "pinn": dict(color=C_PINN, ls="-", lw=1.6, label="Standard PINN"),
    "one_soft": dict(color=C_ONE, ls="-", lw=1.5, label="One-phase, softmax"),
    "one_raw": dict(color=C_ONE, ls="--", lw=1.5, label="One-phase, no softmax"),
    "two_soft": dict(color=C_TWO, ls="-", lw=1.5, label="Two-phase, softmax"),
    "two_raw": dict(color=C_TWO, ls="--", lw=1.5, label="Two-phase, no softmax"),
    "hy_voter_i": dict(color=C_HY_V, ls="-", lw=1.5, label="Hybrid voter only, head at interpolation"),
    "hy_voter_e": dict(color=C_HY_V, ls="--", lw=1.35, label="Hybrid voter only, head at extrapolation"),
    "hy_frozen_i": dict(color=C_HY_F, ls="-", lw=1.5, label="Hybrid backbone frozen, head at interpolation"),
    "hy_frozen_e": dict(color=C_HY_F, ls="--", lw=1.35, label="Hybrid backbone frozen, head at extrapolation"),
    "hy_one_i": dict(color=C_HY_1, ls="-", lw=1.5, label="Hybrid one-phase, head at interpolation"),
    "hy_one_e": dict(color=C_HY_1, ls="--", lw=1.35, label="Hybrid one-phase, head at extrapolation"),
    "hy_two_i": dict(color=C_HY_2, ls="-", lw=1.5, label="Hybrid two-phase, head at interpolation"),
    "hy_two_e": dict(color=C_HY_2, ls="--", lw=1.35, label="Hybrid two-phase, head at extrapolation"),
}

HEAD_GRID = {
    "poisson": {"nx": 201},
    "heat": {"nx": 81, "nt": 41},
    "burgers": {"nx": 61, "nt": 31},
    "fisher_kpp": {"nx": 65, "ny": 65, "snapshot_time": 1.5},
    "lamb_chaplygin": {"nx": 65, "ny": 65, "snapshot_time": 0.5},
}

TABLE_MAE = {
    ("poisson", "pinn"): (1.201e-02, 1.580e00),
    ("poisson", "one_soft"): (7.453e-03, 1.574e00),
    ("poisson", "one_raw"): (5.195e-02, 2.838e00),
    ("poisson", "two_soft"): (6.889e-03, 1.592e00),
    ("poisson", "two_raw"): (1.216e-03, 2.523e-01),
    ("heat", "pinn"): (7.929e-04, 4.795e-02),
    ("heat", "one_soft"): (2.883e-02, 1.147e-01),
    ("heat", "one_raw"): (2.044e-02, 4.725e00),
    ("heat", "two_soft"): (1.859e-02, 1.202e-01),
    ("heat", "two_raw"): (1.767e-02, 4.001e-01),
    ("burgers", "pinn"): (2.544e-02, 2.604e-01),
    ("burgers", "one_soft"): (2.859e-02, 1.942e-01),
    ("burgers", "one_raw"): (4.272e-02, 4.465e-01),
    ("burgers", "two_soft"): (4.323e-02, 2.071e-01),
    ("burgers", "two_raw"): (1.167e-01, 1.887e-01),
    ("fisher_kpp", "pinn"): (4.792e-04, 1.444e-02),
    ("fisher_kpp", "one_soft"): (3.590e-03, 9.838e-02),
    ("fisher_kpp", "one_raw"): (4.312e-02, 8.458e-01),
    ("fisher_kpp", "two_soft"): (3.505e-03, 9.888e-02),
    ("fisher_kpp", "two_raw"): (1.479e-02, 6.480e-01),
    ("lamb_chaplygin", "pinn"): (4.733e-02, 2.761e-01),
    ("lamb_chaplygin", "one_soft"): (4.734e-02, 5.321e-01),
    ("lamb_chaplygin", "one_raw"): (7.959e-02, 5.909e-01),
    ("lamb_chaplygin", "two_soft"): (5.087e-02, 5.500e-01),
    ("lamb_chaplygin", "two_raw"): (1.788e-01, 8.660e-01),
}

SCORE_MTL = {
    ("poisson", "pinn"): (9.494e-03, 4.228e-01),
    ("poisson", "one_soft"): (3.544e-02, 5.686e-01),
    ("poisson", "one_raw"): (1.441e-02, 9.962e-01),
    ("poisson", "two_soft"): (2.145e-02, 6.142e-01),
    ("poisson", "two_raw"): (1.655e-03, 6.370e-02),
    ("heat", "pinn"): (6.153e-04, 1.595e-02),
    ("heat", "one_soft"): (1.307e-02, 6.550e-02),
    ("heat", "one_raw"): (1.402e-02, 1.805e00),
    ("heat", "two_soft"): (8.490e-03, 7.323e-02),
    ("heat", "two_raw"): (1.353e-02, 1.846e-01),
    ("burgers", "pinn"): (2.780e-02, 1.554e-01),
    ("burgers", "one_soft"): (3.080e-02, 1.303e-01),
    ("burgers", "one_raw"): (4.638e-02, 2.920e-01),
    ("burgers", "two_soft"): (4.248e-02, 1.412e-01),
    ("burgers", "two_raw"): (1.089e-01, 1.341e-01),
    ("fisher_kpp", "pinn"): (6.202e-04, 5.308e-03),
    ("fisher_kpp", "one_soft"): (8.761e-03, 5.194e-02),
    ("fisher_kpp", "one_raw"): (1.908e-02, 4.405e-01),
    ("fisher_kpp", "two_soft"): (6.131e-03, 5.248e-02),
    ("fisher_kpp", "two_raw"): (1.099e-02, 2.743e-01),
    ("lamb_chaplygin", "pinn"): (3.499e-02, 9.811e-02),
    ("lamb_chaplygin", "one_soft"): (4.747e-02, 2.160e-01),
    ("lamb_chaplygin", "one_raw"): (6.009e-02, 2.414e-01),
    ("lamb_chaplygin", "two_soft"): (4.992e-02, 2.298e-01),
    ("lamb_chaplygin", "two_raw"): (9.389e-02, 3.433e-01),
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )


def save(fig, name: str) -> Path:
    _apply_outside_letters(fig)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"saved {path}", flush=True)
    return path


def panel_label(ax, letter: str) -> None:
    """Remember a panel letter. save() draws it under the axes, below any xlabel."""
    ax._outside_letter = str(letter)


def _apply_outside_letters(fig) -> None:
    for ax in fig.axes:
        letter = getattr(ax, "_outside_letter", None)
        if not letter:
            continue
        tag = f"({letter})"
        current = ax.get_xlabel() or ""
        if tag not in current:
            ax.set_xlabel((current + "\n" if current else "") + tag)
        ax.xaxis.label.set_fontsize(9)
        ax.xaxis.label.set_linespacing(1.8)
        ax.xaxis.labelpad = 8


def _loosen(fig, h_pad=0.12, w_pad=0.08) -> None:
    engine = fig.get_layout_engine()
    if engine is not None and hasattr(engine, "set"):
        engine.set(h_pad=h_pad, w_pad=w_pad, hspace=0.04, wspace=0.03)


def require(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing {what}: {path}")
    return path


def load_curves() -> dict[str, list[dict]]:
    require(CURVES, "generalization curves")
    by = {name: [] for name in PDES}
    with CURVES.open() as handle:
        for row in csv.DictReader(handle):
            rec = {k: row[k] for k in row}
            rec["x"] = float(row["x"])
            rec["y"] = float(row["y"]) if row["y"] else None
            for key in STYLE:
                if key in row and row[key] != "":
                    rec[key] = float(row[key])
            by[row["pde"]].append(rec)
    return by


def family_meta(name: str) -> dict:
    problem = get_problem(name)
    train = [tuple(float(v) for v in mu) for mu in problem.spec.task_params]
    return {
        "name": name,
        "train": train,
        "interp": problem.mu_for_split("interpolation"),
        "extrap": problem.mu_for_split("extrapolation"),
        "problem": problem,
    }


def s_of(mu, fam) -> float:
    if fam["name"] == "lamb_chaplygin":
        a0, u0 = fam["interp"]
        a1, u1 = fam["extrap"]
        da, du = a1 - a0, u1 - u0
        return float(((mu[0] - a0) * da + (mu[1] - u0) * du) / (da * da + du * du))
    return float(mu[0])


def near(a, b) -> bool:
    return all(abs(float(x) - float(y)) <= 1e-6 for x, y in zip(a, b))


def parse_mu(text: str) -> tuple[float, ...]:
    return tuple(float(v) for v in text.split(","))


def inside_train(mu, train) -> bool:
    if len(mu) == 1:
        lo = min(row[0] for row in train)
        hi = max(row[0] for row in train)
        return lo <= mu[0] <= hi
    a_lo, a_hi = min(row[0] for row in train), max(row[0] for row in train)
    u_lo, u_hi = min(row[1] for row in train), max(row[1] for row in train)
    return a_lo <= mu[0] <= a_hi and u_lo <= mu[1] <= u_hi


def gmean(values) -> float:
    arr = np.clip(np.asarray(values, dtype=np.float64), 1e-30, None)
    return float(np.exp(np.mean(np.log(arr))))


def sig3(a: float, b: float) -> bool:
    if a == 0 and b == 0:
        return True
    return abs(a - b) <= 0.005 * max(abs(a), abs(b), 1e-30) or f"{a:.3e}" == f"{b:.3e}"


def check_pair(tag: str, got: float, ref: float, mismatches: list) -> None:
    if not CHECK_REFS:
        return
    if not sig3(got, ref):
        mismatches.append(f"{tag}: got {got:.6e}, table {ref:.6e}")
    else:
        print(f"  ok {tag}: {got:.3e}", flush=True)


def lamb_intra_s(fam) -> tuple[float, float]:
    a0, u0 = fam["interp"]
    a1, u1 = fam["extrap"]
    samples = []
    for s in np.linspace(-0.45, 1.2, 4001):
        mu = (a0 + s * (a1 - a0), u0 + s * (u1 - u0))
        if inside_train(mu, fam["train"]):
            samples.append(s)
    if not samples:
        return 0.0, 0.0
    return float(min(samples)), float(max(samples))


def ckpt_mtl(pde: str, mode: str, softmax: bool) -> Path:
    folder = "softmax" if softmax else "nosoftmax"
    return COMP / pde / "mtl" / mode / folder / "muvoter" / "seed50" / "model.pth"


def ckpt_pinn(pde: str) -> Path:
    return COMP / pde / "baseline" / "pinn" / "seed50" / "model.pth"


def ckpt_hybrid(pde: str, mode: str, split: str) -> Path:
    short = "interp" if split.startswith("interp") else "extrap"
    return HYB / pde / "hybrid" / short / mode / "softmax" / "muvoter" / "seed50" / "model.pth"


def load_pinn(problem):
    spec = RunSpec(pde=problem.spec.name, mode="baseline", seed=50)
    model = build_baseline(problem, spec).to(DEVICE)
    payload = torch.load(require(ckpt_pinn(problem.spec.name), "PINN checkpoint"), map_location=DEVICE, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def load_mtl(problem, mode: str, softmax: bool):
    spec = RunSpec(pde=problem.spec.name, mode=mode, softmax=softmax, seed=50)
    path = require(ckpt_mtl(problem.spec.name, mode, softmax), f"{mode} softmax={softmax}")
    model, _ = load_checkpoint(path, problem, spec, DEVICE, hybrid=False)
    model.eval()
    return model


def load_hybrid(problem, mode: str, split: str):
    spec = RunSpec(pde=problem.spec.name, mode=mode, softmax=True, hybrid=True, snapshot_split=split, seed=50)
    path = require(ckpt_hybrid(problem.spec.name, mode, split), f"hybrid {mode} {split}")
    model, _ = load_checkpoint(path, problem, spec, DEVICE, hybrid=True)
    model.eval()
    return model


@torch.no_grad()
def mae_of(problem, model, mu, grid_kw) -> float:
    ev = problem.eval_grid(mu, DEVICE, **grid_kw)
    true = problem.reference(ev.coords, mu)
    pred = _predict(model, ev).detach().cpu().numpy()
    errs = []
    for j, name in enumerate(problem.spec.field_names):
        arr = pred[:, j].reshape(ev.shape)
        errs.append(float(np.mean(np.abs(arr - np.asarray(true[name])))))
    return float(np.mean(errs))


@torch.no_grad()
def head_field(problem, model, mu, head_idx, grid_kw):
    ev = problem.eval_grid(mu, DEVICE, **grid_kw)
    packed = ev.coord_tensors[0]
    pred = model(packed, ev.param_tensor, t=ev.time_tensor, head_idx=head_idx)
    pred_np = pred.detach().cpu().numpy()
    fields = {name: pred_np[:, j].reshape(ev.shape) for j, name in enumerate(problem.spec.field_names)}
    true = problem.reference(ev.coords, mu)
    return ev, fields, true


@torch.no_grad()
def mix_field(problem, model, mu, grid_kw):
    ev = problem.eval_grid(mu, DEVICE, **grid_kw)
    pred = _predict(model, ev).detach().cpu().numpy()
    fields = {name: pred[:, j].reshape(ev.shape) for j, name in enumerate(problem.spec.field_names)}
    true = problem.reference(ev.coords, mu)
    return ev, fields, true


PLOT_GRID = {
    "heat": {"nx": 161, "nt": 81},
    "burgers": {"nx": 161, "nt": 81},
    "fisher_kpp": {"nx": 161, "ny": 161, "snapshot_time": 1.5},
    "lamb_chaplygin": {"nx": 161, "ny": 161, "snapshot_time": 0.5},
}
N_CONTOUR = 80


def image_extent(problem, ev, values):
    coords = ev.coords
    names = problem.spec.coord_names
    x = np.asarray(coords["x"])
    if "y" in names:
        y = np.asarray(coords["y"])
        extent = [float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))]
        return values, extent, "x", "y"
    t = np.asarray(coords["t"])
    extent = [float(np.min(x)), float(np.max(x)), float(np.min(t)), float(np.max(t))]
    return np.asarray(values).T, extent, "x", "t"


def field_mesh(problem, ev, values):
    x = np.asarray(ev.coords["x"])
    values = np.asarray(values)
    if "y" in problem.spec.coord_names:
        return x, np.asarray(ev.coords["y"]), values, "x", "y"
    return x, np.asarray(ev.coords["t"]), values, "x", "t"


def smooth_field(ax, xx, yy, zz, levels, cmap, vmin, vmax):
    cf = ax.contourf(
        xx,
        yy,
        zz,
        levels=levels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extend="both",
        antialiased=False,
    )
    cf.set_rasterized(True)
    ax.set_rasterization_zorder(1)
    cf.set_zorder(0)
    return cf


def ref_limits(arr, diverging=False):
    arr = np.asarray(arr)
    if diverging:
        m = float(np.max(np.abs(arr)))
        if m == 0:
            m = 1e-6
        return -m, m
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


# ---------------------------------------------------------------------------
# F1
# ---------------------------------------------------------------------------

def apply_solution_style() -> None:
    """Larger type used by the exact-solution panels."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )


def fig_f1() -> Path:
    """Four exact Fisher–KPP fields u(x, y, t=1.5; r) on [-10, 10]^2."""
    apply_solution_style()
    rates = (0.5, 1.0, 1.5, 2.0)
    n = 201
    x = np.linspace(-10.0, 10.0, n)
    y = np.linspace(-10.0, 10.0, n)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    fields = [fkpp_exact(xx, yy, 1.5, rate) for rate in rates]
    print(
        "Fisher–KPP formula: u=(1+exp(k(x+y)-ω t))^{-2}, "
        "k=sqrt(r/D)/(2 sqrt(3)), ω=5r/6, D=1 at t=1.5",
        flush=True,
    )
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = matplotlib.cm.RdBu_r
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    plt.subplots_adjust(top=0.9, bottom=0.195, left=0.068, right=0.95, hspace=0.2, wspace=0.185)
    for i, (ax, rate, field) in enumerate(zip(axes, rates, fields)):
        ax.contourf(xx, yy, field, levels=100, cmap=cmap, norm=norm)
        ax.set_title(rf"$r={rate:g}$")
        ax.set_xlabel(r"$x$")
        if i == 0:
            ax.set_ylabel(r"$y$")
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=12)
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes.tolist(),
        location="right",
        orientation="vertical",
        pad=0.02,
    )
    cbar.ax.tick_params(labelsize=12)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig_fkpp_sols-crop.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}", flush=True)
    return path


def fig_f2() -> Path:
    """Lamb–Chaplygin stream function (top) and vorticity (bottom) at t=0.5."""
    apply_solution_style()
    corners = ((0.8, 0.5), (0.8, 1.0), (1.2, 0.5), (1.2, 1.0))
    n = 201
    x = np.linspace(-3.0, 3.0, n)
    y = np.linspace(-3.0, 3.0, n)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    tt = np.full_like(xx, 0.5)
    sols = [lc_exact(xx, yy, tt, a, speed) for a, speed in corners]
    rows = ([pair[0] for pair in sols], [pair[1] for pair in sols])
    lims = [max(float(np.max(np.abs(field))) for field in row) for row in rows]
    print("Lamb–Chaplygin conventions (mtl_final/problems/lamb_chaplygin.py):", flush=True)
    print("  transport:  ω_t + u ω_x + v ω_y = 0,  with  u = ψ_y,  v = −ψ_x", flush=True)
    print("  kinematic:  ω + ψ_xx + ψ_yy = 0,  i.e.  ω = −Δψ", flush=True)
    print(f"  Initial centre: (x0, y0) = ({DEFAULT_X0:g}, {DEFAULT_Y0:g}).", flush=True)
    print(f"  Heading α = {DEFAULT_ALPHA:g} (along +x). Centre at t is (−1 + U t, 0).", flush=True)
    print(f"  Top row stream function, bottom row vorticity.  max|ψ|={lims[0]:.4g}, max|ω|={lims[1]:.4g}", flush=True)

    ax_h = 0.705 * 4.0
    fig_h = 0.4 + 2.0 * ax_h + 0.2 * ax_h + 0.78
    cmap = matplotlib.cm.RdBu_r
    fig, axes = plt.subplots(2, 4, figsize=(12, fig_h))
    plt.subplots_adjust(
        top=1.0 - 0.4 / fig_h,
        bottom=0.78 / fig_h,
        left=0.068,
        right=0.95,
        hspace=0.2,
        wspace=0.185,
    )
    for row, fields in enumerate(rows):
        vmax = lims[row]
        norm = matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)
        for col, ((a, speed), field) in enumerate(zip(corners, fields)):
            ax = axes[row, col]
            ax.contourf(xx, yy, field, levels=100, cmap=cmap, norm=norm)
            if row == 0:
                ax.set_title(rf"$(a,U)=({a:g},{speed:g})$")
            ax.set_xlabel(r"$x$")
            if col == 0:
                ax.set_ylabel(r"$y$")
            ax.tick_params(labelsize=12)
        cbar = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axes[row].tolist(),
            location="right",
            orientation="vertical",
            pad=0.02,
        )
        cbar.ax.tick_params(labelsize=12)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig_lc_sols-crop.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}", flush=True)
    return path


# ---------------------------------------------------------------------------
# F3 / F4
# ---------------------------------------------------------------------------

def _plot_curve_panel(ax, name, rows, keys, fam, mark_heads=True, hybrid_head_colors=None):
    xs = [s_of(parse_mu(row["mu"]), fam) for row in rows]
    order = np.argsort(xs)
    xs = np.asarray(xs)[order]
    train_set = {tuple(np.round(mu, 8)) for mu in fam["train"]}
    for key in keys:
        ys = np.asarray([rows[i][key] for i in order])
        ax.plot(xs, ys, **STYLE[key])
        if name != "lamb_chaplygin":
            for i in order:
                mu = parse_mu(rows[i]["mu"])
                if tuple(np.round(mu, 8)) in train_set:
                    ax.plot(
                        s_of(mu, fam),
                        rows[i][key],
                        marker="o",
                        mfc="none",
                        mec=STYLE[key]["color"],
                        ms=4.5,
                        ls="none",
                        zorder=5,
                    )
    if name == "lamb_chaplygin":
        lo, hi = lamb_intra_s(fam)
        ax.axvspan(lo, hi, color="#0072B2", alpha=0.08, zorder=0)
    else:
        for i, mu in enumerate(fam["train"]):
            ax.axvline(s_of(mu, fam), color=C_TRAIN, lw=0.7, ls="--", zorder=0, label="Training parameter" if i == 0 else None)
    if mark_heads:
        ax.axvline(s_of(fam["interp"], fam), color=C_INTERP, lw=0.8, ls=":", zorder=0, label="Interpolation query")
        ax.axvline(s_of(fam["extrap"], fam), color=C_EXTRAP, lw=0.8, ls=":", zorder=0, label="Extrapolation query")
    if hybrid_head_colors:
        for mu, color, label in hybrid_head_colors:
            ax.axvline(s_of(mu, fam), color=color, lw=0.9, ls=":", zorder=0, label=label)
    ax.set_yscale("log")
    ax.set_ylabel("MAE")
    ax.set_xlabel(XLABELS[name])
    ax.set_title(TITLES[name])
    ax.grid(True, which="both", axis="y", lw=0.3, color="#dddddd")
    if name == "lamb_chaplygin":
        ticks = [-0.4, 0.0, 0.4, 0.8, 1.0, 1.2]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{s:g}" for s in ticks])
    ys_all = []
    for key in keys:
        ys_all.extend(float(row[key]) for row in rows if key in row)
    visible = [y for y in ys_all if y > 1e-6]
    if visible:
        ymin = 10 ** np.floor(np.log10(min(visible)) - 0.15)
        ymax = 10 ** np.ceil(np.log10(max(ys_all)) + 0.05)
        ax.set_ylim(ymin, ymax)


def _score_from_rows(rows, key, fam, snapshot=None) -> tuple[float, float]:
    intra, extra = [], []
    for row in rows:
        mu = parse_mu(row["mu"])
        if any(near(mu, t) for t in fam["train"]):
            continue
        if snapshot is not None and near(mu, snapshot):
            continue
        if inside_train(mu, fam["train"]):
            intra.append(row[key])
        else:
            extra.append(row[key])
    return gmean(intra), gmean(extra)


def fig_f3() -> Path:
    apply_style()
    by = load_curves()
    mismatches = []
    keys = ["pinn", "one_soft", "one_raw", "two_soft", "two_raw"]
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 6.7), layout="constrained")
    _loosen(fig, h_pad=0.16, w_pad=0.10)
    axes = axes.ravel()
    for i, name in enumerate(PDES):
        fam = family_meta(name)
        _plot_curve_panel(axes[i], name, by[name], keys, fam, mark_heads=True)
        if name == "burgers":
            axes[i].set_title("Burgers'")
        panel_label(axes[i], chr(ord("a") + i))
        for key in keys:
            intra, extra = _score_from_rows(by[name], key, fam)
            ref = SCORE_MTL[(name, key)]
            check_pair(f"F3 {name} {key} intra", intra, ref[0], mismatches)
            check_pair(f"F3 {name} {key} extra", extra, ref[1], mismatches)
    axes[5].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[5].legend(handles, labels, loc="center", frameon=False, fontsize=8, borderpad=0.6, labelspacing=0.45)
    path = save(fig, "fig_generalization_curves-crop.pdf")
    if mismatches:
        print("F3 MAE mismatches:", *mismatches, sep="\n  ", flush=True)
    else:
        print("F3: all intra/extra geometric means match the stored scores to 3 s.f.", flush=True)
    return path


def fig_f4() -> Path:
    apply_style()
    by = load_curves()
    mismatches = []
    keys = [
        "pinn",
        "two_soft",
        "hy_voter_i",
        "hy_voter_e",
        "hy_frozen_i",
        "hy_frozen_e",
        "hy_one_i",
        "hy_one_e",
        "hy_two_i",
        "hy_two_e",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 6.7), layout="constrained")
    _loosen(fig, h_pad=0.16, w_pad=0.10)
    axes = axes.ravel()
    for i, name in enumerate(PDES):
        fam = family_meta(name)
        heads = [
            (fam["interp"], C_HY_V, "Numerical head, interpolation"),
            (fam["extrap"], C_HY_VE, "Numerical head, extrapolation"),
        ]
        _plot_curve_panel(axes[i], name, by[name], keys, fam, mark_heads=False, hybrid_head_colors=heads)
        if name == "burgers":
            axes[i].set_title("Burgers'")
        panel_label(axes[i], chr(ord("a") + i))
        snaps = {
            "pinn": None,
            "two_soft": None,
            "hy_voter_i": fam["interp"],
            "hy_voter_e": fam["extrap"],
            "hy_frozen_i": fam["interp"],
            "hy_frozen_e": fam["extrap"],
            "hy_one_i": fam["interp"],
            "hy_one_e": fam["extrap"],
            "hy_two_i": fam["interp"],
            "hy_two_e": fam["extrap"],
        }
        for key in keys:
            intra, extra = _score_from_rows(by[name], key, fam, snapshot=snaps[key])
            print(f"  F4 {name} {key} intra={intra:.3e} extra={extra:.3e}", flush=True)
            if key in SCORE_MTL:
                ref = SCORE_MTL[(name, key)]
                check_pair(f"F4 {name} {key} intra", intra, ref[0], mismatches)
                check_pair(f"F4 {name} {key} extra", extra, ref[1], mismatches)
    axes[5].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[5].legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        fontsize=6.4,
        borderpad=0.3,
        labelspacing=0.28,
        handlelength=2.4,
    )
    path = save(fig, "fig_hybrid_curves-crop.pdf")
    if mismatches:
        print("F4 MAE mismatches (PINN / two-phase only):", *mismatches, sep="\n  ", flush=True)
    else:
        print("F4: PINN and two-phase softmax intra/extra match the stored scores to 3 s.f.", flush=True)
    return path


# ---------------------------------------------------------------------------
# F5 heads
# ---------------------------------------------------------------------------

def _head_mae(pred, true, field) -> float:
    return float(np.mean(np.abs(np.asarray(pred[field]) - np.asarray(true[field]))))


def fig_f5_poisson() -> list[Path]:
    apply_style()
    problem = get_problem("poisson")
    mus = [tuple(float(v) for v in mu) for mu in problem.spec.task_params]
    paths = []
    mismatches = []
    for mode, fname in (("two-phase", "fig_poisson_learned_head_spec-crop.pdf"), ("one-phase", "fig_poisson_learned_head_no_spec-crop.pdf")):
        model = load_mtl(problem, mode, True)
        fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.05), layout="constrained", sharey=False)
        _loosen(fig, h_pad=0.08, w_pad=0.10)
        for i, (ax, mu) in enumerate(zip(axes, mus)):
            ev, pred, true = head_field(problem, model, mu, i, HEAD_GRID["poisson"])
            x = np.asarray(ev.coords["x"]).reshape(-1)
            order = np.argsort(x)
            ax.plot(x[order], np.asarray(true["u"]).reshape(-1)[order], color="black", lw=2.0, label="reference")
            ax.plot(
                x[order],
                pred["u"].reshape(-1)[order],
                color=C_TWO if mode == "two-phase" else C_ONE,
                lw=1.2,
                ls="--",
            )
            mae = _head_mae(pred, true, "u")
            ax.set_title(rf"$\mu={mu[0]:g}$" + f"\nMAE {mae:.2e}")
            ax.set_xlabel(r"$x$")
            if i == 0:
                ax.set_ylabel(r"$u$")
            panel_label(ax, chr(ord("a") + i))
            ax.grid(True, alpha=0.3)
            print(f"  F5 Poisson {mode} head {i+1} MAE {mae:.3e}", flush=True)
        paths.append(save(fig, fname))
    return paths


# Axes box shared with the Fisher–KPP solution figure, in inches.
HEAD_AX_W = 192.85883644346865 / 100.0
HEAD_AX_H = 2.82
HEAD_GAP_W = 0.185 * HEAD_AX_W
HEAD_GAP_H = 0.2 * HEAD_AX_H
HEAD_LEFT = 1.55
HEAD_BOTTOM = 0.85
HEAD_TOP = 0.85
HEAD_CBAR_GAP = 0.18
HEAD_CBAR_W = 0.16
HEAD_RIGHT = 0.55


@torch.no_grad()
def _head_columns(problem, mus, field: str, grid_kw: dict):
    two = load_mtl(problem, "two-phase", True)
    one = load_mtl(problem, "one-phase", True)
    vertical = "y" if "y" in problem.spec.coord_names else "t"
    index = problem.spec.field_names.index(field)
    columns = []
    for col, mu in enumerate(mus):
        ev = problem.eval_grid(mu, DEVICE, **grid_kw)
        true = np.asarray(problem.reference(ev.coords, mu)[field])

        def predict(model, head_idx=col, grid=ev):
            packed = grid.coord_tensors[0]
            pred = model(packed, grid.param_tensor, t=grid.time_tensor, head_idx=head_idx)
            return pred.detach().cpu().numpy()[:, index].reshape(grid.shape)

        columns.append((ev.coords["x"], ev.coords[vertical], predict(two), predict(one), true))
    return columns


def _head_symmetric_norm(columns) -> matplotlib.colors.Normalize:
    peak = max(float(np.max(np.abs(true))) for *_, true in columns)
    if peak == 0.0:
        peak = 1.0
    return matplotlib.colors.Normalize(vmin=-peak, vmax=peak)


def _render_head_grid(columns, titles, vertical_symbol: str, norm, fname: str, right: float = 0.70) -> Path:
    """3×4 heads: two-phase, one-phase, analytical. One shared colorbar, no text label."""
    apply_style()
    ylabels = (
        "Two-phase\n" + vertical_symbol,
        "One-phase\n" + vertical_symbol,
        "Analytical\n" + vertical_symbol,
    )
    n_col, n_row = 4, 3
    fig_w = HEAD_LEFT + n_col * HEAD_AX_W + (n_col - 1) * HEAD_GAP_W + HEAD_CBAR_GAP + HEAD_CBAR_W + right
    fig_h = HEAD_TOP + n_row * HEAD_AX_H + (n_row - 1) * HEAD_GAP_H + HEAD_BOTTOM
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = []
    for row in range(n_row):
        y = HEAD_BOTTOM + (n_row - 1 - row) * (HEAD_AX_H + HEAD_GAP_H)
        axes.append([])
        for col in range(n_col):
            x = HEAD_LEFT + col * (HEAD_AX_W + HEAD_GAP_W)
            axes[row].append(fig.add_axes([x / fig_w, y / fig_h, HEAD_AX_W / fig_w, HEAD_AX_H / fig_h]))

    cmap = matplotlib.cm.RdBu_r
    for col, (xx, yy, pred_two, pred_one, true) in enumerate(columns):
        for row, values in enumerate((pred_two, pred_one, true)):
            ax = axes[row][col]
            ax.contourf(xx, yy, values, levels=100, cmap=cmap, norm=norm)
            if col == 0:
                ax.set_ylabel(ylabels[row])
            else:
                ax.tick_params(labelleft=False)
            if row == 0:
                ax.set_title(titles[col] + "\n" + rf"Head {col + 1}")
                ax.tick_params(labelbottom=False)
            elif row == 1:
                ax.set_title(rf"Head {col + 1}")
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(r"$x$")

    cax = fig.add_axes(
        [
            (HEAD_LEFT + n_col * HEAD_AX_W + (n_col - 1) * HEAD_GAP_W + HEAD_CBAR_GAP) / fig_w,
            HEAD_BOTTOM / fig_h,
            HEAD_CBAR_W / fig_w,
            (n_row * HEAD_AX_H + (n_row - 1) * HEAD_GAP_H) / fig_h,
        ]
    )
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="vertical")
    cbar.ax.tick_params(labelsize=9)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / fname
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}", flush=True)
    return path


def fig_f5() -> list[Path]:
    paths = fig_f5_poisson()
    heat = get_problem("heat")
    rhos = (0.05, 0.1, 0.2, 0.3)
    paths.append(
        _render_head_grid(
            _head_columns(heat, [(rho,) for rho in rhos], "u", {"nx": 201, "nt": 101}),
            [rf"$\rho={rho:g}$" for rho in rhos],
            r"$t$",
            matplotlib.colors.Normalize(vmin=0.0, vmax=1.0),
            "fig_heat_learned_head_spec_no_spec_analytic-crop.pdf",
            right=HEAD_RIGHT,
        )
    )
    burgers = get_problem("burgers")
    nus = (0.005, 0.01, 0.02, 0.03)
    burgers_cols = _head_columns(burgers, [(nu,) for nu in nus], "u", {"nx": 201, "nt": 81})
    paths.append(
        _render_head_grid(
            burgers_cols,
            [rf"$\nu={nu:g}$" for nu in nus],
            r"$t$",
            _head_symmetric_norm(burgers_cols),
            "fig_burgers_learned_head_spec_no_spec_analytic-crop.pdf",
        )
    )
    fkpp = get_problem("fisher_kpp")
    rates = (0.5, 1.0, 1.5, 2.0)
    paths.append(
        _render_head_grid(
            _head_columns(fkpp, [(rate,) for rate in rates], "u", {"nx": 161, "ny": 161, "snapshot_time": 1.5}),
            [rf"$r={rate:g}$" for rate in rates],
            r"$y$",
            matplotlib.colors.Normalize(vmin=0.0, vmax=1.0),
            "fig_fkpp_learned_head_spec_no_spec_analytic-crop.pdf",
        )
    )
    lamb = get_problem("lamb_chaplygin")
    corners = ((0.8, 0.5), (0.8, 1.0), (1.2, 0.5), (1.2, 1.0))
    titles = [rf"$(a,U)=({a:g},{speed:g})$" for a, speed in corners]
    grid = {"nx": 161, "ny": 161, "snapshot_time": 0.5}
    for field, fname in (
        ("psi", "fig_lc_psi_learned_head_spec_no_spec_analytic-crop.pdf"),
        ("omega", "fig_lc_learned_head_spec_no_spec_analytic-crop.pdf"),
    ):
        columns = _head_columns(lamb, list(corners), field, grid)
        paths.append(_render_head_grid(columns, titles, r"$y$", _head_symmetric_norm(columns), fname))
    return paths


# ---------------------------------------------------------------------------
# F6 RQ2
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_poisson_rq2(out_dir: Path | None = None) -> Path:
    """Two-phase and one-phase predictions against u = λ x² at λ = 2 and λ = 8."""
    apply_style()
    out_dir = OUT if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    problem = get_problem("poisson")
    models = {mode: load_mtl(problem, mode, True) for mode in ("two-phase", "one-phase")}
    colors = {"two-phase": "#1B4F9E", "one-phase": "#0C582A"}
    labels = {"two-phase": "Predicted (two-phase)", "one-phase": "Predicted (one-phase)"}
    gap_w, left, right, bottom, top = 0.62, 0.62, 0.15, 0.55, 0.35
    fig_w = left + 4 * HEAD_AX_W + 3 * gap_w + right
    fig_h = bottom + HEAD_AX_H + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    x_fine = np.linspace(-1.0, 1.0, 201)
    col = 0
    for lam in (2.0, 8.0):
        grid = problem.eval_grid((lam,), DEVICE, nx=31)
        x = np.asarray(grid.coords["x"]).reshape(-1)
        order = np.argsort(x)
        for mode in ("two-phase", "one-phase"):
            pred = _predict(models[mode], grid).detach().cpu().numpy().reshape(-1)
            x0 = left + col * (HEAD_AX_W + gap_w)
            ax = fig.add_axes([x0 / fig_w, bottom / fig_h, HEAD_AX_W / fig_w, HEAD_AX_H / fig_h])
            color = colors[mode]
            ax.plot(x[order], pred[order], ls="--", marker="o", ms=3.5, lw=1.4, color=color, label=labels[mode])
            ax.plot(x_fine, lam * x_fine**2, ls="-", lw=1.4, color=color, label="Analytical solution")
            ax.set_title(rf"$\lambda={lam:g}$")
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$u(x;\lambda)$")
            ax.set_xlim(-1.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper center", fontsize=8, frameon=True, framealpha=0.9)
            col += 1
    path = out_dir / "fig_RQ2_Poisson-crop.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}", flush=True)
    return path


def _fig_f6_maps(pde: str, field: str, fname: str, diverging=False, mismatches=None) -> Path:
    apply_style()
    problem = get_problem(pde)
    one = load_mtl(problem, "one-phase", True)
    two = load_mtl(problem, "two-phase", True)
    queries = [
        ("interpolation", problem.mu_for_split("interpolation")),
        ("extrapolation", problem.mu_for_split("extrapolation")),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.05), layout="constrained")
    _loosen(fig, h_pad=0.16, w_pad=0.05)
    letters = "abcd"
    cmap_s = SOL_CMAP
    pname = problem.spec.param_names[0] if len(problem.spec.param_names) == 1 else None
    last_top = [None, None]
    last_bot = [None, None]
    for q, (split, mu) in enumerate(queries):
        ev, p1, true = mix_field(problem, one, mu, GRID)
        _, p2, _ = mix_field(problem, two, mu, GRID)
        ref = np.asarray(true[field])
        lo, hi = ref_limits(ref, diverging=diverging)
        mae1 = float(np.mean(np.abs(p1[field] - ref)))
        mae2 = float(np.mean(np.abs(p2[field] - ref)))
        if pde != "lamb_chaplygin":
            check_pair(
                f"F6 {pde} one-phase {split}",
                mae1,
                TABLE_MAE[(pde, "one_soft")][0 if split.startswith("interp") else 1],
                mismatches,
            )
            check_pair(
                f"F6 {pde} two-phase {split}",
                mae2,
                TABLE_MAE[(pde, "two_soft")][0 if split.startswith("interp") else 1],
                mismatches,
            )
        else:
            print(f"  F6 {pde} {split} omega-only MAE one={mae1:.3e} two={mae2:.3e} (table is mae_mean)", flush=True)
            ev_full, m1, tfull = mix_field(problem, one, mu, GRID)
            _, m2, _ = mix_field(problem, two, mu, GRID)
            mean1 = float(np.mean([np.mean(np.abs(m1[n] - np.asarray(tfull[n]))) for n in problem.spec.field_names]))
            mean2 = float(np.mean([np.mean(np.abs(m2[n] - np.asarray(tfull[n]))) for n in problem.spec.field_names]))
            check_pair(f"F6 {pde} one-phase {split} mae_mean", mean1, TABLE_MAE[(pde, "one_soft")][0 if split.startswith("interp") else 1], mismatches)
            check_pair(f"F6 {pde} two-phase {split} mae_mean", mean2, TABLE_MAE[(pde, "two_soft")][0 if split.startswith("interp") else 1], mismatches)
        err1 = np.abs(p1[field] - ref)
        err2 = np.abs(p2[field] - ref)
        err_hi = max(float(np.max(err1)), float(np.max(err2)), 1e-12)
        if pname:
            mu_txt = rf"${pname}={mu[0]:g}$"
        else:
            mu_txt = rf"$({mu[0]:g},{mu[1]:g})$"
        top = ((p1[field], "one-phase", mae1), (p2[field], "two-phase", mae2))
        bot = ((err1, "one-phase error"), (err2, "two-phase error"))
        for j, (values, name, mae) in enumerate(top):
            ax = axes[0, 2 * q + j]
            img, extent, xl, yl = image_extent(problem, ev, values)
            last_top[q] = ax.imshow(img, origin="lower", extent=extent, aspect="auto", vmin=lo, vmax=hi, cmap=cmap_s)
            short = "1φ" if name.startswith("one") else "2φ"
            ax.set_title(f"{short}, {split}\n{mu_txt}, MAE {mae:.2e}", fontsize=8)
            if 2 * q + j == 0:
                ax.set_ylabel(rf"${yl}$")
            panel_label(ax, letters[2 * q + j])
            ax.tick_params(length=2, labelsize=7)
        for j, (values, name) in enumerate(bot):
            ax = axes[1, 2 * q + j]
            img, extent, xl, yl = image_extent(problem, ev, values)
            last_bot[q] = ax.imshow(img, origin="lower", extent=extent, aspect="auto", vmin=0.0, vmax=err_hi, cmap=SOL_CMAP)
            ax.set_title(name, fontsize=8)
            ax.set_xlabel(rf"${xl}$")
            if 2 * q + j == 0:
                ax.set_ylabel(rf"$|\,\cdot\,|$, ${yl}$")
            ax.tick_params(length=2, labelsize=7)
        fig.colorbar(last_top[q], ax=axes[0, 2 * q : 2 * q + 2].tolist(), fraction=0.046, pad=0.02)
        fig.colorbar(last_bot[q], ax=axes[1, 2 * q : 2 * q + 2].tolist(), fraction=0.046, pad=0.02)
    return save(fig, fname)


def fig_f6() -> list[Path]:
    mismatches = []
    paths = [plot_poisson_rq2(OUT)]
    paths.append(_fig_f6_maps("heat", "u", "fig_RQ2_Heat-crop.pdf", mismatches=mismatches))
    paths.append(_fig_f6_maps("burgers", "u", "fig_RQ2_Burgers-crop.pdf", diverging=True, mismatches=mismatches))
    paths.append(_fig_f6_maps("fisher_kpp", "u", "fig_RQ2_FKPP-crop.pdf", mismatches=mismatches))
    paths.append(_fig_f6_maps("lamb_chaplygin", "omega", "fig_RQ2_LC-crop.pdf", diverging=True, mismatches=mismatches))
    if mismatches:
        print("F6 MAE mismatches:", *mismatches, sep="\n  ", flush=True)
    else:
        print("F6: query MAE matches the stored evaluation to 3 s.f.", flush=True)
    return paths


PRED_BLOCKS = (
    ("heat", "u", False, "Heat", r"$u$"),
    ("burgers", "u", True, "Burgers'", r"$u$"),
    ("fisher_kpp", "u", False, "Fisher–KPP", r"$u$"),
    ("lamb_chaplygin", "psi", True, r"LC $\psi$", r"$\psi$"),
    ("lamb_chaplygin", "omega", True, r"LC $\omega$", r"$\omega$"),
)

PRED_COLS = (
    ("two-phase", "interpolation", "Two-phase, interp."),
    ("one-phase", "interpolation", "One-phase, interp."),
    ("two-phase", "extrapolation", "Two-phase, extrap."),
    ("one-phase", "extrapolation", "One-phase, extrap."),
)


def fig_pred_error_all() -> Path:
    """Prediction and |error| for all PDEs except Poisson: 10x4 (LC has ψ and ω)."""
    apply_style()
    models = {}
    for pde in ("heat", "burgers", "fisher_kpp", "lamb_chaplygin"):
        problem = get_problem(pde)
        models[pde] = {
            "one-phase": load_mtl(problem, "one-phase", True),
            "two-phase": load_mtl(problem, "two-phase", True),
            "problem": problem,
        }
    cache = {}
    mae_cache = {}
    for pde, pack in models.items():
        for mode in ("one-phase", "two-phase"):
            for split in ("interpolation", "extrapolation"):
                mu = pack["problem"].mu_for_split(split)
                cache[(pde, mode, split)] = mix_field(pack["problem"], pack[mode], mu, PLOT_GRID[pde])
                ev_m, pred_m, true_m = mix_field(pack["problem"], pack[mode], mu, GRID)
                mae_cache[(pde, mode, split)] = {
                    name: float(np.mean(np.abs(np.asarray(pred_m[name]) - np.asarray(true_m[name]))))
                    for name in pack["problem"].spec.field_names
                }

    n_blocks = len(PRED_BLOCKS)
    height_ratios = []
    for b in range(n_blocks):
        height_ratios.extend([1.0, 1.0])
        if b < n_blocks - 1:
            height_ratios.append(0.42)
    height_ratios.append(0.22)
    nrows = len(height_ratios)
    fig = plt.figure(figsize=(7.0, 10.1))
    gs = fig.add_gridspec(
        nrows,
        5,
        width_ratios=[1, 1, 1, 1, 0.045],
        height_ratios=height_ratios,
        wspace=0.08,
        hspace=0.14,
        left=0.11,
        right=0.94,
        top=0.955,
        bottom=0.055,
    )
    last_err_row = 3 * (n_blocks - 1) + 1
    err_axes_blocks = []
    colorbars = []
    for b, (pde, field, diverging, row_name, _field_tex) in enumerate(PRED_BLOCKS):
        problem = models[pde]["problem"]
        pred_row = 3 * b
        err_row = 3 * b + 1
        refs = []
        preds = []
        errs = []
        maes = []
        ev0 = None
        for mode, split, _title in PRED_COLS:
            ev, pred, true = cache[(pde, mode, split)]
            ev0 = ev
            ref = np.asarray(true[field])
            hat = np.asarray(pred[field])
            refs.append(ref)
            preds.append(hat)
            errs.append(np.abs(hat - ref))
            maes.append(mae_cache[(pde, mode, split)][field])
            print(f"  pred-error {pde} {field} {mode} {split} MAE {maes[-1]:.3e}", flush=True)
        if diverging:
            vmax = max(float(np.max(np.abs(r))) for r in refs)
            lo, hi = -vmax, vmax
        elif pde == "fisher_kpp":
            lo, hi = 0.0, 1.0
        else:
            lo = min(float(np.min(r)) for r in refs)
            hi = max(float(np.max(r)) for r in refs)
        err_hi = max(float(np.max(e)) for e in errs)
        last_pred = last_err = None
        err_axes = []
        for col, (_mode, _split, col_title) in enumerate(PRED_COLS):
            ax_p = fig.add_subplot(gs[pred_row, col])
            ax_e = fig.add_subplot(gs[err_row, col])
            xx, yy, _, xl, yl = field_mesh(problem, ev0, preds[col])
            last_pred = smooth_field(ax_p, xx, yy, preds[col], np.linspace(lo, hi, N_CONTOUR + 1), SOL_CMAP, lo, hi)
            last_err = smooth_field(ax_e, xx, yy, errs[col], np.linspace(0.0, err_hi, N_CONTOUR + 1), "YlOrRd", 0.0, err_hi)
            ax_p.set_aspect("auto")
            ax_e.set_aspect("auto")
            if pred_row == 0:
                ax_p.set_title(col_title, fontsize=7.5, pad=2)
            ax_p.tick_params(length=1.5, labelsize=6)
            ax_e.tick_params(length=1.5, labelsize=6)
            if col != 0:
                ax_p.set_yticklabels([])
                ax_e.set_yticklabels([])
            else:
                ax_p.set_ylabel(row_name, fontsize=7)
                ax_e.set_ylabel("MAE", fontsize=7)
            ax_p.set_xticklabels([])
            if err_row != last_err_row:
                ax_e.set_xticklabels([])
            else:
                ax_e.set_xlabel(rf"${xl}$", fontsize=7)
            err_axes.append(ax_e)
        err_axes_blocks.append(err_axes)
        cax_p = fig.add_subplot(gs[pred_row, 4])
        cax_e = fig.add_subplot(gs[err_row, 4])
        cb_p = fig.colorbar(last_pred, cax=cax_p)
        cb_e = fig.colorbar(last_err, cax=cax_e)
        cb_p.ax.tick_params(labelsize=5.5, length=1.5)
        cb_e.ax.tick_params(labelsize=5.5, length=1.5)
        colorbars.append((cax_p, cb_p, lo, hi))
        colorbars.append((cax_e, cb_e, 0.0, err_hi))
    fig.canvas.draw()
    for cax, cb, vmin, vmax in colorbars:
        pos = cax.get_position()
        pad = min(0.007, 0.16 * pos.height)
        cax.set_position([pos.x0, pos.y0 + pad, pos.width, pos.height - 2.0 * pad])
        ticks = np.linspace(vmin, vmax, 3)
        cb.set_ticks(ticks)
        labels = []
        scale = max(abs(vmin), abs(vmax), 1e-12)
        for t in ticks:
            if abs(t) < 1e-3 * scale:
                labels.append("0")
            else:
                labels.append(f"{t:.2g}")
        cb.set_ticklabels(labels)
    for b, axes_e in enumerate(err_axes_blocks):
        x0 = min(ax.get_position().x0 for ax in axes_e)
        x1 = max(ax.get_position().x1 for ax in axes_e)
        y0 = min(ax.get_position().y0 for ax in axes_e)
        offset = 0.040 if b == n_blocks - 1 else 0.016
        fig.text((x0 + x1) / 2.0, y0 - offset, f"({chr(ord('a') + b)})", ha="center", va="top", fontsize=9)
    return save(fig, "fig_RQ2_all-crop.pdf")


# ---------------------------------------------------------------------------
# F7
# ---------------------------------------------------------------------------

def fig_f7() -> Path:
    apply_style()
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 6.7), layout="constrained")
    _loosen(fig, h_pad=0.16, w_pad=0.12)
    axes = axes.ravel()
    for i, name in enumerate(PDES):
        ax = axes[i]
        for split, color, label in (
            ("interp", C_HY_V, "head at interpolation"),
            ("extrap", C_HY_VE, "head at extrapolation"),
        ):
            path = HYB / name / "hybrid" / split / "only-voter" / "softmax" / "muvoter" / "seed50" / "losses.csv"
            require(path, f"voter-only losses {name} {split}")
            ys = []
            with path.open() as handle:
                for row in csv.DictReader(handle):
                    ys.append(float(row["loss"]))
            # CSV "epoch" is 0,2,...,39998 (2 × step). Use the step index: 20 000 optimizer steps.
            xs = np.arange(len(ys))
            step = np.ones(len(ys), dtype=bool)
            step[2000:] = False
            step[2000::20] = True
            ax.plot(xs[step], np.asarray(ys)[step], color=color, ls="-" if split == "interp" else "-.", lw=1.1, label=label)
        ax.axvline(500, color=C_TRAIN, lw=0.8, ls="--")
        ax.set_xlim(0, 20_000)
        ax.set_xticks([0, 5_000, 10_000, 15_000, 20_000])
        ax.set_xticklabels(["0", "5000", "10000", "15000", "20000"])
        ax.set_yscale("log")
        ax.set_ylabel("training loss")
        ax.set_title(TITLES[name])
        ax.grid(True, which="both", axis="y", lw=0.3, color="#dddddd")
        panel_label(ax, chr(ord("a") + i))
        if i >= 3:
            ax.set_xlabel("epoch")
    axes[5].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=C_TRAIN, ls="--", lw=0.8, label="epoch 500"))
    labels.append("epoch 500")
    axes[5].legend(handles, labels, loc="center", frameon=False, borderpad=0.7, labelspacing=0.45)
    return save(fig, "fig_voter_only_loss-crop.pdf")


# ---------------------------------------------------------------------------
# D1 voter weights
# ---------------------------------------------------------------------------

@torch.no_grad()
def fig_d1() -> list[Path]:
    rows = []
    paths = []
    variants = [
        ("one-phase", True, "One-phase, softmax"),
        ("one-phase", False, "One-phase, no softmax"),
        ("two-phase", True, "Two-phase, softmax"),
        ("two-phase", False, "Two-phase, no softmax"),
    ]
    for pde in PDES:
        problem = get_problem(pde)
        for mode, softmax, label in variants:
            model = load_mtl(problem, mode, softmax)
            for split in ("interpolation", "extrapolation"):
                mu = problem.mu_for_split(split)
                p = problem.pack_params(1, mu, DEVICE)
                w = model.get_attention_weights(p, None).detach().cpu().numpy().reshape(-1)
                rec = {
                    "pde": pde,
                    "variant": label,
                    "mode": mode,
                    "softmax": softmax,
                    "split": split,
                    "mu": ",".join(f"{v:g}" for v in mu),
                    "sum": float(np.sum(w)),
                }
                for i, wi in enumerate(w):
                    rec[f"w{i}"] = float(wi)
                rows.append(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "voter_weights.csv"
    fields = ["pde", "variant", "mode", "softmax", "split", "mu"] + [f"w{i}" for i in range(4)] + ["sum"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {csv_path}", flush=True)
    paths.append(csv_path)
    return paths


HEAD_COLORS = ("#0072B2", "#009E73", "#E69F00", "#D55E00")


def fig_voter_weights() -> Path:
    """One figure: softmax voter weights, all PDEs, one-phase vs two-phase, both queries."""
    apply_style()
    csv_path = require(OUT / "voter_weights.csv", "voter weights")
    by = {}
    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            if row["softmax"] != "True":
                continue
            key = (row["pde"], row["mode"], row["split"])
            by[key] = np.array([float(row[f"w{i}"]) for i in range(4)])
    fig, axes = plt.subplots(2, 5, figsize=(7.0, 4.15), sharey=True)
    fig.subplots_adjust(left=0.10, right=0.995, top=0.86, bottom=0.16, wspace=0.32, hspace=0.78)
    splits = ("interpolation", "extrapolation")
    for col, pde in enumerate(PDES):
        problem = get_problem(pde)
        for row, split in enumerate(splits):
            ax = axes[row, col]
            w_one = by[(pde, "one-phase", split)]
            w_two = by[(pde, "two-phase", split)]
            bottom = np.zeros(2)
            for i, color in enumerate(HEAD_COLORS):
                vals = np.array([w_one[i], w_two[i]])
                ax.bar((0, 1), vals, bottom=bottom, width=0.62, color=color, edgecolor="white", linewidth=0.4)
                bottom = bottom + vals
            ax.set_xticks((0, 1), ("One-phase", "Two-phase"))
            ax.set_ylim(0.0, 1.02)
            ax.tick_params(axis="x", length=2, labelsize=6)
            ax.tick_params(axis="y", length=2, labelsize=7)
            query = _mu_tex(problem, problem.mu_for_split(split))
            name = "Burgers'" if pde == "burgers" else TITLES[pde]
            if row == 0:
                ax.set_title(f"{name}\n{query}", fontsize=8)
            else:
                ax.set_title(query, fontsize=8)
            if col == 0:
                ax.set_ylabel(("Interpolation" if split == "interpolation" else "Extrapolation") + "\n" + r"$w_\ell$")
            ax.set_xlabel(f"({chr(ord('a') + row * 5 + col)})", fontsize=10, labelpad=2)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=f"Head {i + 1}")
        for i, color in enumerate(HEAD_COLORS)
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.52, 0.075))
    return save(fig, "fig_voter_weights-crop.pdf")


_PARAM_TEX = {"mu": r"\lambda", "p": r"\rho", "nu": r"\nu", "r": r"r", "a": r"a", "U": r"U"}


def _mu_tex(problem, mu) -> str:
    names = [_PARAM_TEX.get(n, n) for n in problem.spec.param_names]
    if len(mu) == 1:
        return rf"${names[0]}={mu[0]:g}$"
    return rf"$({names[0]},{names[1]})=({mu[0]:g},{mu[1]:g})$"


# ---------------------------------------------------------------------------
# D2 explained variance
# ---------------------------------------------------------------------------

def _ev_param_sample(problem) -> list[tuple[float, ...]]:
    name = problem.spec.name
    if name == "lamb_chaplygin":
        a = np.linspace(0.8, 1.2, 7)
        u = np.linspace(0.5, 1.0, 7)
        return [(float(ai), float(uj)) for ai in a for uj in u]
    lo = min(mu[0] for mu in problem.spec.task_params)
    hi = max(mu[0] for mu in problem.spec.task_params)
    n = 17 if name == "burgers" else 31
    return [(float(v),) for v in np.linspace(lo, hi, n)]


def _ev_grid_kw(name: str) -> dict:
    if name == "poisson":
        return {"nx": 201}
    if name in {"fisher_kpp", "lamb_chaplygin"}:
        return dict(GRID)
    return dict(GRID)


@torch.no_grad()
def _snapshot_vector(problem, mu, grid_kw) -> np.ndarray:
    ev = problem.eval_grid(mu, torch.device("cpu"), **grid_kw)
    true = problem.reference(ev.coords, mu)
    parts = [np.asarray(true[name], dtype=np.float64).reshape(-1) for name in problem.spec.field_names]
    return np.concatenate(parts)


@torch.no_grad()
def _head_matrix(problem, model, grid_kw) -> np.ndarray:
    mu0 = tuple(float(v) for v in problem.spec.task_params[0])
    ev = problem.eval_grid(mu0, DEVICE, **grid_kw)
    packed = ev.coord_tensors[0]
    cols = []
    n_fields = len(problem.spec.field_names)
    for i in range(len(model.heads)):
        pred = model(packed, ev.param_tensor, t=ev.time_tensor, head_idx=i).detach().cpu().numpy()
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        parts = [pred[:, j].reshape(-1) for j in range(n_fields)]
        cols.append(np.concatenate(parts))
    return np.stack(cols, axis=1)


def fig_d2() -> list[Path]:
    rows = []
    for pde in PDES:
        problem = get_problem(pde)
        grid_kw = _ev_grid_kw(pde)
        mus = _ev_param_sample(problem)
        print(f"D2 {pde}: {len(mus)} parameter samples, grid {grid_kw}", flush=True)
        snaps = np.stack([_snapshot_vector(problem, mu, grid_kw) for mu in mus], axis=1)
        xc = snaps - snaps.mean(axis=1, keepdims=True)
        total = float(np.linalg.norm(xc, ord="fro") ** 2)
        _, s, _ = np.linalg.svd(xc, full_matrices=False)
        eta_star = float(np.sum(s[:4] ** 2) / np.sum(s ** 2)) if total > 0 else 1.0
        rec = {"pde": pde, "n_mu": len(mus), "eta_star": eta_star}
        for mode, key in (("one-phase", "one_soft"), ("two-phase", "two_soft")):
            model = load_mtl(problem, mode, True)
            H = _head_matrix(problem, model, grid_kw)
            q, _r = np.linalg.qr(H, mode="reduced")
            eta = float(np.linalg.norm(q.T @ xc, ord="fro") ** 2 / total) if total > 0 else 1.0
            rec[key] = eta
            print(f"  {mode} η={eta:.6f}  bound={eta_star:.6f}", flush=True)
        rows.append(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "explained_variance.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pde", "n_mu", "one_soft", "two_soft", "eta_star"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {csv_path}", flush=True)
    return [csv_path]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

FIGURES = {
    "f1": fig_f1,
    "f2": fig_f2,
    "f3": fig_f3,
    "f4": fig_f4,
    "f5": fig_f5,
    "f6": fig_f6,
    "rq2all": fig_pred_error_all,
    "f7": fig_f7,
    "d1": fig_d1,
    "d2": fig_d2,
    "weights": fig_voter_weights,
}


def main(argv=None) -> None:
    global OUT
    parser = argparse.ArgumentParser(description="Build seed-50 paper figures from existing outputs.")
    parser.add_argument("--fig", nargs="+", choices=list(FIGURES), default=None, help="Subset of figures.")
    parser.add_argument("--all", action="store_true", help="Build every figure.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.out is not None:
        OUT = args.out
    names = list(FIGURES) if args.all or not args.fig else args.fig
    for name in names:
        print(f"=== {name} ===", flush=True)
        try:
            FIGURES[name]()
        except FileNotFoundError as exc:
            print(f"STOPPED {name}: {exc}", flush=True)


if __name__ == "__main__":
    main()
