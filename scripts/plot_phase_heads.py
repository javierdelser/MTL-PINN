#!/usr/bin/env python3
"""Fields of the one-phase and two-phase PINN heads, one figure per PDE.

Softmax, parameter-only voter, seed 50: the same dictionary the hybrid runs
start from. Each head is drawn at the parameter it was trained for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtl_final.config import PDE_NAMES, RunSpec
from mtl_final.problems import get_problem
from mtl_final.train import load_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMP = ROOT / "outputs" / "comparison"
OUT = COMP / "figures" / "heads"

TITLES = {
    "poisson": "Poisson",
    "heat": "Heat",
    "burgers": "Burgers",
    "fisher_kpp": "Fisher–KPP",
    "lamb_chaplygin": "Lamb–Chaplygin",
}

GRIDS = {
    "poisson": {"nx": 201},
    "heat": {"nx": 81, "nt": 41},
    "burgers": {"nx": 61, "nt": 31},
    "fisher_kpp": {"nx": 65, "ny": 65, "snapshot_time": 1.5},
    "lamb_chaplygin": {"nx": 65, "ny": 65, "snapshot_time": 0.5},
}

SCHEDULES = ("one-phase", "two-phase")
LINE_COLOR = {"one-phase": "#1f4e79", "two-phase": "#2a9d8f"}


def _mu_text(names: tuple[str, ...], mu: tuple[float, ...]) -> str:
    return ", ".join(f"{name}={value:.4g}" for name, value in zip(names, mu))


def _checkpoint(pde: str, mode: str) -> Path:
    return COMP / pde / "mtl" / mode / "softmax" / "muvoter" / "seed50" / "model.pth"


def _load(problem, pde: str, mode: str):
    spec = RunSpec(pde=pde, mode=mode, softmax=True, seed=50)
    model, _ = load_checkpoint(_checkpoint(pde, mode), problem, spec, DEVICE, hybrid=False)
    model.eval()
    return model


def _field_on_grid(problem, model, mu: tuple[float, ...], head_idx: int):
    grid = problem.eval_grid(mu, DEVICE, **GRIDS[problem.spec.name])
    packed = grid.coord_tensors[0]
    with torch.no_grad():
        pred = model(packed, grid.param_tensor, t=grid.time_tensor, head_idx=head_idx)
    pred_np = pred.detach().cpu().numpy()
    fields = {
        name: pred_np[:, j].reshape(grid.shape)
        for j, name in enumerate(problem.spec.field_names)
    }
    truth = problem.reference(grid.coords, mu)
    return {"grid": grid, "pred": fields, "true": truth}


def _image(values: np.ndarray, coords: dict[str, np.ndarray], names: tuple[str, ...]):
    x = np.asarray(coords["x"])
    if "y" in names:
        y = np.asarray(coords["y"])
        extent = [float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))]
        return values, extent, "x", "y"
    t = np.asarray(coords["t"])
    extent = [float(np.min(x)), float(np.max(x)), float(np.min(t)), float(np.max(t))]
    return values.T, extent, "x", "t"


def _limits(reference: np.ndarray) -> tuple[float, float]:
    lo = float(np.min(reference))
    hi = float(np.max(reference))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def plot_pde(pde: str) -> Path:
    problem = get_problem(pde)
    names = problem.spec.param_names
    coord_names = problem.spec.coord_names
    fields = problem.spec.field_names
    models = {mode: _load(problem, pde, mode) for mode in SCHEDULES}
    n_pinn = len(models["one-phase"].heads)
    columns = []
    for index in range(n_pinn):
        mu = tuple(float(v) for v in problem.spec.task_params[index])
        columns.append((f"PINN head {index + 1}", mu, index))
    n_rows = len(SCHEDULES) * len(fields)
    n_cols = n_pinn
    line = problem.spec.n_coord == 1
    if line:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.35 * n_rows), squeeze=False, sharex=True)
    else:
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(2.35 * n_cols, 2.15 * n_rows), squeeze=False, layout="constrained"
        )

    row = 0
    for mode in SCHEDULES:
        model = models[mode]
        drawn = []
        for label, mu, head_idx in columns:
            drawn.append((label, mu, _field_on_grid(problem, model, mu, head_idx)))
        for field in fields:
            ax_row = axes[row]
            for col, (label, mu, payload) in enumerate(drawn):
                ax = ax_row[col]
                pred = payload["pred"][field]
                truth = np.asarray(payload["true"][field])
                mae = float(np.mean(np.abs(pred - truth)))
                title = f"{label}\n{_mu_text(names, mu)}\nMAE {mae:.2e}"
                if line:
                    x = np.asarray(payload["grid"].coords["x"]).reshape(-1)
                    order = np.argsort(x)
                    ax.plot(x[order], truth.reshape(-1)[order], color="black", lw=1.6)
                    ax.plot(x[order], pred.reshape(-1)[order], color=LINE_COLOR[mode], lw=1.3)
                    ax.grid(True, alpha=0.3)
                    ax.set_title(title, fontsize=8)
                    if col == 0:
                        ax.set_ylabel(f"{mode}\n{field}", fontsize=8)
                else:
                    image, extent, xlabel, ylabel = _image(pred, payload["grid"].coords, coord_names)
                    lo, hi = _limits(truth)
                    im = ax.imshow(image, origin="lower", extent=extent, aspect="auto", vmin=lo, vmax=hi, cmap="RdBu_r")
                    ax.set_title(title, fontsize=8)
                    ax.set_xlabel(xlabel, fontsize=8)
                    if col == 0:
                        ax.set_ylabel(f"{mode}, {field}\n{ylabel}", fontsize=8)
                    ax.tick_params(labelsize=7)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            row += 1
    note = (
        "Black is the reference at that head's parameter."
        if line
        else "Color scale is the reference range at that head's parameter."
    )
    fig.suptitle(f"{TITLES[pde]}: one-phase and two-phase, softmax. {note}", fontsize=11)
    if line:
        fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{pde}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}", flush=True)
    return path


def main() -> None:
    for pde in PDE_NAMES:
        plot_pde(pde)


if __name__ == "__main__":
    main()
