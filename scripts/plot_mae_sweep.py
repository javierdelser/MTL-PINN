#!/usr/bin/env python3
"""MAE versus query parameter for the plain PINN and the voter-only hybrid."""

from __future__ import annotations

import csv
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

from mtl_final.config import RunSpec
from mtl_final.evaluate import _predict
from mtl_final.problems import get_problem
from mtl_final.train import build_baseline, load_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID = {"nx": 41, "ny": 41, "nt": 21}
COMP = ROOT / "outputs" / "comparison"
HYB = ROOT / "outputs" / "hybrid"


def _mu_key(mu) -> str:
    return ",".join(f"{float(v):.8g}" for v in mu)


def _sweep_1d(lo, hi, n, specials):
    vals = list(np.linspace(lo, hi, n)) + [float(v) for v in specials]
    vals = sorted(set(np.round(np.asarray(vals, dtype=np.float64), decimals=8)))
    return [float(v) for v in vals]


def queries_for(name: str):
    problem = get_problem(name)
    train = [tuple(float(v) for v in mu) for mu in problem.spec.task_params]
    interp = problem.mu_for_split("interpolation")
    extrap = problem.mu_for_split("extrapolation")
    if name == "poisson":
        grid = [(v,) for v in _sweep_1d(0.25, 9.0, 37, [m[0] for m in train] + [interp[0], extrap[0]])]
        xlabel = r"query parameter $\mu$"
    elif name == "heat":
        grid = [(v,) for v in _sweep_1d(0.02, 0.95, 37, [m[0] for m in train] + [interp[0], extrap[0]])]
        xlabel = r"query parameter $p$"
    elif name == "burgers":
        grid = [(v,) for v in _sweep_1d(0.004, 0.10, 29, [m[0] for m in train] + [interp[0], extrap[0]])]
        xlabel = r"query parameter $\nu$"
    elif name == "fisher_kpp":
        grid = [(v,) for v in _sweep_1d(0.35, 4.0, 33, [m[0] for m in train] + [interp[0], extrap[0]])]
        xlabel = r"query parameter $r$"
    elif name == "lamb_chaplygin":
        # Line through the interpolation and extrapolation queries. The four
        # training corners sit off this line; they are marked in the panel.
        a0, u0 = interp
        a1, u1 = extrap
        grid = []
        for s in np.linspace(-0.45, 1.2, 29):
            grid.append((float(a0 + s * (a1 - a0)), float(u0 + s * (u1 - u0))))
        xlabel = r"query $(a, U)$ along the line from $(1.0, 0.75)$ to $(1.6, 1.4)$"
    else:
        raise ValueError(name)
    return problem, grid, train, interp, extrap, xlabel


def _load_pinn(problem):
    spec = RunSpec(pde=problem.spec.name, mode="baseline", seed=50)
    model = build_baseline(problem, spec).to(DEVICE)
    path = COMP / problem.spec.name / "baseline" / "pinn" / "seed50" / "model.pth"
    payload = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _load_hybrid(problem, split: str):
    spec = RunSpec(
        pde=problem.spec.name,
        mode="only-voter",
        softmax=True,
        hybrid=True,
        snapshot_split=split,
        seed=50,
    )
    short = "interp" if split.startswith("interp") else "extrap"
    path = HYB / problem.spec.name / "hybrid" / short / "only-voter" / "softmax" / "muvoter" / "seed50" / "model.pth"
    model, _ = load_checkpoint(path, problem, spec, DEVICE, hybrid=True)
    model.eval()
    return model


@torch.no_grad()
def mae_of(problem, model, grid_np, true):
    pred = _predict(model, grid_np)
    pred_np = pred.detach().cpu().numpy()
    errs = []
    for j, name in enumerate(problem.spec.field_names):
        arr = pred_np[:, j].reshape(grid_np.shape)
        errs.append(float(np.mean(np.abs(arr - np.asarray(true[name])))))
    return float(np.mean(errs))


def evaluate_family(name: str):
    problem, grid, train, interp, extrap, xlabel = queries_for(name)
    print(f"{name}: {len(grid)} queries", flush=True)
    models = {
        "pinn": _load_pinn(problem),
        "hybrid_interp": _load_hybrid(problem, "interpolation"),
        "hybrid_extrap": _load_hybrid(problem, "extrapolation"),
    }
    rows = []
    for mu in grid:
        ev = problem.eval_grid(mu, DEVICE, **GRID)
        true = problem.reference(ev.coords, mu)
        rec = {"mu": _mu_key(mu), "x": mu[0]}
        if len(mu) > 1:
            rec["y"] = mu[1]
        for key, model in models.items():
            rec[key] = mae_of(problem, model, ev, true)
        rows.append(rec)
        print(f"  {_mu_key(mu)} pinn={rec['pinn']:.3e} hI={rec['hybrid_interp']:.3e} hE={rec['hybrid_extrap']:.3e}", flush=True)
    return {
        "name": name,
        "xlabel": xlabel,
        "train": train,
        "interp": interp,
        "extrap": extrap,
        "rows": rows,
    }


def _x_of(mu, family):
    if family["name"] == "lamb_chaplygin":
        a0, u0 = family["interp"]
        a1, u1 = family["extrap"]
        da, du = a1 - a0, u1 - u0
        return ((mu[0] - a0) * da + (mu[1] - u0) * du) / (da * da + du * du)
    return mu[0]


def plot(families, path: Path):
    titles = {
        "poisson": "Poisson",
        "heat": "Heat",
        "burgers": "Burgers",
        "fisher_kpp": "Fisher–KPP",
        "lamb_chaplygin": "Lamb–Chaplygin",
    }
    fig, axes = plt.subplots(len(families), 1, figsize=(8.4, 12.2), layout="constrained")
    for ax, fam in zip(axes, families):
        xs = [_x_of(tuple(float(v) for v in row["mu"].split(",")), fam) for row in fam["rows"]]
        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        for key, label, color, style in (
            ("pinn", "Plain PINN", "#222222", "-"),
            ("two_phase", "Two-phase, softmax", "#2a9d8f", "-"),
            ("hybrid_interp", "Hybrid voter only, head at interpolation", "#1f4e79", "-"),
            ("frozen_interp", "Hybrid backbone frozen, head at interpolation", "#1f4e79", "--"),
            ("hybrid_extrap", "Hybrid voter only, head at extrapolation", "#b85c38", "-"),
            ("frozen_extrap", "Hybrid backbone frozen, head at extrapolation", "#b85c38", "--"),
        ):
            ys = np.asarray([fam["rows"][i][key] for i in order])
            ax.plot(xs, ys, color=color, lw=1.6, ls=style, label=label)
        train_x = [_x_of(mu, fam) for mu in fam["train"]]
        head_i = _x_of(fam["interp"], fam)
        head_e = _x_of(fam["extrap"], fam)
        for i, xv in enumerate(train_x):
            ax.axvline(xv, color="#9aa0a6", lw=0.8, ls="--", zorder=0, label="Training parameter" if i == 0 else None)
        ax.axvline(head_i, color="#1f4e79", lw=0.9, ls=":", zorder=0, label="Numerical head, interpolation")
        ax.axvline(head_e, color="#b85c38", lw=0.9, ls=":", zorder=0, label="Numerical head, extrapolation")
        ax.set_yscale("log")
        ax.set_ylabel("MAE")
        ax.set_xlabel(fam["xlabel"])
        ax.set_title(titles[fam["name"]])
        ax.grid(True, which="both", axis="y", lw=0.3, color="#dddddd")
        if fam["name"] == "lamb_chaplygin":
            ticks = [-0.4, 0.0, 0.4, 0.8, 1.0, 1.2]
            a0, u0 = fam["interp"]
            a1, u1 = fam["extrap"]
            labels = [f"({a0 + s * (a1 - a0):.2f}, {u0 + s * (u1 - u0):.2f})" for s in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.text(
                0.01,
                0.04,
                "Training corners, off this line:\n(0.8, 0.5), (0.8, 1.0), (1.2, 0.5), (1.2, 1.0)",
                transform=ax.transAxes,
                fontsize=7,
                va="bottom",
                color="#444444",
            )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=7.5, frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    print(f"saved {path}")


def write_csv(families, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pde", "mu", "x", "y", "pinn", "two_phase", "hybrid_interp", "hybrid_extrap", "frozen_interp", "frozen_extrap"],
        )
        writer.writeheader()
        for fam in families:
            for row in fam["rows"]:
                writer.writerow({
                    "pde": fam["name"],
                    "mu": row["mu"],
                    "x": row["x"],
                    "y": row.get("y", ""),
                    "pinn": row["pinn"],
                    "two_phase": row["two_phase"],
                    "hybrid_interp": row["hybrid_interp"],
                    "hybrid_extrap": row["hybrid_extrap"],
                    "frozen_interp": row["frozen_interp"],
                    "frozen_extrap": row["frozen_extrap"],
                })


def main():
    names = ["poisson", "heat", "burgers", "fisher_kpp", "lamb_chaplygin"]
    families = [evaluate_family(name) for name in names]
    out = HYB / "figures"
    write_csv(families, out / "mae_vs_query.csv")
    plot(families, out / "mae_vs_query.png")


if __name__ == "__main__":
    main()
