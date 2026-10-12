#!/usr/bin/env python3
"""Fields of the backbone-frozen hybrid heads, one figure per PDE.

Each checkpoint has four retrained PINN heads and one numerical snapshot.
Both placements of that snapshot (interpolation and extrapolation) are shown.
A PINN head is drawn on the parameter it was trained for. The numerical head
is the classical snapshot at the extra parameter.
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
HYB = ROOT / "outputs" / "hybrid"
OUT = HYB / "figures" / "heads"

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


def _mu_text(names: tuple[str, ...], mu: tuple[float, ...]) -> str:
    return ", ".join(f"{name}={value:.4g}" for name, value in zip(names, mu))


def _checkpoint(pde: str, split: str) -> Path:
    short = "interp" if split.startswith("interp") else "extrap"
    return (
        HYB
        / pde
        / "hybrid"
        / short
        / "backbone-frozen-retrain-rest"
        / "softmax"
        / "muvoter"
        / "seed50"
        / "model.pth"
    )


def _load(problem, pde: str, split: str):
    spec = RunSpec(
        pde=pde,
        mode="backbone-frozen-retrain-rest",
        softmax=True,
        hybrid=True,
        snapshot_split=split,
        seed=50,
    )
    model, _ = load_checkpoint(_checkpoint(pde, split), problem, spec, DEVICE, hybrid=True)
    model.eval()
    return model


def _field_on_grid(problem, model, mu: tuple[float, ...], head_idx: int | None) -> tuple[dict, dict[str, np.ndarray]]:
    grid = problem.eval_grid(mu, DEVICE, **GRIDS[problem.spec.name])
    packed = grid.coord_tensors[0]
    with torch.no_grad():
        if head_idx is None:
            cols = tuple(packed[:, i : i + 1] for i in range(len(model.snapshot.axes)))
            pred = model.solver_fields(*cols)
        else:
            pred = model.mtl(packed, grid.param_tensor, t=grid.time_tensor, head_idx=head_idx)
    pred_np = pred.detach().cpu().numpy()
    fields = {
        name: pred_np[:, j].reshape(grid.shape)
        for j, name in enumerate(problem.spec.field_names)
    }
    truth = problem.reference(grid.coords, mu)
    return {"grid": grid, "pred": fields, "true": truth}, fields


def _image(problem, values: np.ndarray, coords: dict[str, np.ndarray]):
    names = problem.spec.coord_names
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
    fields = problem.spec.field_names
    placements = ("interpolation", "extrapolation")
    models = {split: _load(problem, pde, split) for split in placements}
    n_pinn = len(models["interpolation"].heads)
    columns = []
    for index in range(n_pinn):
        mu = tuple(float(v) for v in problem.spec.task_params[index])
        columns.append((f"PINN head {index + 1}", mu, index))
    n_rows = len(placements) * len(fields)
    n_cols = n_pinn + 1
    spatial = problem.spec.n_coord == 1
    if spatial:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.35 * n_rows), squeeze=False, sharex=True)
    else:
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(2.35 * n_cols, 2.15 * n_rows), squeeze=False, layout="constrained"
        )

    row = 0
    for split in placements:
        model = models[split]
        snap_mu = tuple(float(v) for v in model.extra_param)
        head_cols = columns + [("numerical head", snap_mu, None)]
        drawn = []
        for label, mu, head_idx in head_cols:
            payload, _ = _field_on_grid(problem, model, mu, head_idx)
            drawn.append((label, mu, payload))
        place = "interpolation" if split.startswith("interp") else "extrapolation"
        for field_i, field in enumerate(fields):
            ax_row = axes[row]
            for col, (label, mu, payload) in enumerate(drawn):
                ax = ax_row[col]
                pred = payload["pred"][field]
                truth = np.asarray(payload["true"][field])
                mae = float(np.mean(np.abs(pred - truth)))
                title = f"{label}\n{_mu_text(names, mu)}\nMAE {mae:.2e}"
                if spatial:
                    x = np.asarray(payload["grid"].coords["x"]).reshape(-1)
                    order = np.argsort(x)
                    ax.plot(x[order], truth.reshape(-1)[order], color="black", lw=1.6)
                    ax.plot(x[order], pred.reshape(-1)[order], color="#1f4e79", lw=1.3)
                    ax.grid(True, alpha=0.3)
                    ax.set_title(title, fontsize=8)
                else:
                    image, extent, xlabel, ylabel = _image(problem, pred, payload["grid"].coords)
                    lo, hi = _limits(truth)
                    im = ax.imshow(image, origin="lower", extent=extent, aspect="auto", vmin=lo, vmax=hi, cmap="RdBu_r")
                    ax.set_title(title, fontsize=8)
                    ax.set_xlabel(xlabel, fontsize=8)
                    if col == 0:
                        ax.set_ylabel(f"{place}, {field}\n{ylabel}", fontsize=8)
                    ax.tick_params(labelsize=7)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if spatial and field_i == 0:
                ax_row[0].set_ylabel(f"{place}\n{field}", fontsize=8)
            row += 1
    if spatial:
        fig.suptitle(
            f"{TITLES[pde]}: backbone frozen. Black is the reference at that head's parameter.",
            fontsize=11,
        )
        fig.tight_layout()
    else:
        fig.suptitle(
            f"{TITLES[pde]}: backbone frozen. Color scale is the reference range at that head's parameter.",
            fontsize=11,
        )
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
