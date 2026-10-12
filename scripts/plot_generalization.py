#!/usr/bin/env python3
"""MAE versus query parameter, one figure per PDE.

Intra score: geometric mean of MAE on queries inside the training parameter
range. Extra score: the same mean outside that range. Training parameters,
and the numerical-head parameter of a hybrid, are left out of both means.
"""

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
from scripts.plot_mae_sweep import COMP, DEVICE, GRID, HYB, _mu_key, _x_of, queries_for
from mtl_final.train import build_baseline, load_checkpoint

OUT = ROOT / "outputs" / "generalization"

SERIES = [
    ("pinn", "Plain PINN", "#222222", "-", None),
    ("one_soft", "One-phase, softmax", "#1d4e89", "-", None),
    ("one_raw", "One-phase, no softmax", "#7eb6d9", "--", None),
    ("two_soft", "Two-phase, softmax", "#0b6e4f", "-", None),
    ("two_raw", "Two-phase, no softmax", "#7dcea0", "--", None),
    ("hy_one_i", "Hybrid one-phase, head at interpolation", "#6a4c93", "-", "interpolation"),
    ("hy_one_e", "Hybrid one-phase, head at extrapolation", "#6a4c93", "--", "extrapolation"),
    ("hy_two_i", "Hybrid two-phase, head at interpolation", "#b85c38", "-", "interpolation"),
    ("hy_two_e", "Hybrid two-phase, head at extrapolation", "#b85c38", "--", "extrapolation"),
    ("hy_voter_i", "Hybrid voter only, head at interpolation", "#c1121f", "-", "interpolation"),
    ("hy_voter_e", "Hybrid voter only, head at extrapolation", "#c1121f", "--", "extrapolation"),
    ("hy_frozen_i", "Hybrid backbone frozen, head at interpolation", "#e09f3e", "-", "interpolation"),
    ("hy_frozen_e", "Hybrid backbone frozen, head at extrapolation", "#e09f3e", "--", "extrapolation"),
]


def _near(a, b) -> bool:
    return all(abs(float(x) - float(y)) <= 1e-6 for x, y in zip(a, b))


def _inside(mu, train) -> bool:
    if len(mu) == 1:
        lo = min(row[0] for row in train)
        hi = max(row[0] for row in train)
        return lo <= mu[0] <= hi
    a_lo, a_hi = min(row[0] for row in train), max(row[0] for row in train)
    u_lo, u_hi = min(row[1] for row in train), max(row[1] for row in train)
    return a_lo <= mu[0] <= a_hi and u_lo <= mu[1] <= u_hi


def _region(mu, train, snapshot) -> str:
    if any(_near(mu, row) for row in train):
        return "atom"
    if snapshot is not None and _near(mu, snapshot):
        return "snapshot"
    if _inside(mu, train):
        return "intra"
    return "extra"


def _gmean(values: list[float]) -> float:
    arr = np.clip(np.asarray(values, dtype=np.float64), 1e-30, None)
    return float(np.exp(np.mean(np.log(arr))))


def _load_pinn(problem):
    spec = RunSpec(pde=problem.spec.name, mode="baseline", seed=50)
    model = build_baseline(problem, spec).to(DEVICE)
    path = COMP / problem.spec.name / "baseline" / "pinn" / "seed50" / "model.pth"
    payload = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _load_mtl(problem, mode: str, softmax: bool):
    spec = RunSpec(pde=problem.spec.name, mode=mode, softmax=softmax, seed=50)
    folder = "softmax" if softmax else "nosoftmax"
    path = COMP / problem.spec.name / "mtl" / mode / folder / "muvoter" / "seed50" / "model.pth"
    model, _ = load_checkpoint(path, problem, spec, DEVICE, hybrid=False)
    model.eval()
    return model


def _load_hybrid(problem, mode: str, split: str):
    spec = RunSpec(
        pde=problem.spec.name,
        mode=mode,
        softmax=True,
        hybrid=True,
        snapshot_split=split,
        seed=50,
    )
    short = "interp" if split.startswith("interp") else "extrap"
    path = HYB / problem.spec.name / "hybrid" / short / mode / "softmax" / "muvoter" / "seed50" / "model.pth"
    model, _ = load_checkpoint(path, problem, spec, DEVICE, hybrid=True)
    model.eval()
    return model


def _models(problem):
    return {
        "pinn": (_load_pinn(problem), None),
        "one_soft": (_load_mtl(problem, "one-phase", True), None),
        "one_raw": (_load_mtl(problem, "one-phase", False), None),
        "two_soft": (_load_mtl(problem, "two-phase", True), None),
        "two_raw": (_load_mtl(problem, "two-phase", False), None),
        "hy_one_i": (_load_hybrid(problem, "one-phase", "interpolation"), problem.mu_for_split("interpolation")),
        "hy_one_e": (_load_hybrid(problem, "one-phase", "extrapolation"), problem.mu_for_split("extrapolation")),
        "hy_two_i": (_load_hybrid(problem, "two-phase", "interpolation"), problem.mu_for_split("interpolation")),
        "hy_two_e": (_load_hybrid(problem, "two-phase", "extrapolation"), problem.mu_for_split("extrapolation")),
        "hy_voter_i": (_load_hybrid(problem, "only-voter", "interpolation"), problem.mu_for_split("interpolation")),
        "hy_voter_e": (_load_hybrid(problem, "only-voter", "extrapolation"), problem.mu_for_split("extrapolation")),
        "hy_frozen_i": (
            _load_hybrid(problem, "backbone-frozen-retrain-rest", "interpolation"),
            problem.mu_for_split("interpolation"),
        ),
        "hy_frozen_e": (
            _load_hybrid(problem, "backbone-frozen-retrain-rest", "extrapolation"),
            problem.mu_for_split("extrapolation"),
        ),
    }


@torch.no_grad()
def _mae(problem, model, grid, true) -> float:
    pred = _predict(model, grid).detach().cpu().numpy()
    errs = []
    for j, name in enumerate(problem.spec.field_names):
        arr = pred[:, j].reshape(grid.shape)
        errs.append(float(np.mean(np.abs(arr - np.asarray(true[name])))))
    return float(np.mean(errs))


def evaluate_family(name: str) -> dict:
    problem, grid, train, interp, extrap, xlabel = queries_for(name)
    print(f"{name}: {len(grid)} queries", flush=True)
    loaded = _models(problem)
    rows = []
    buckets = {key: {"intra": [], "extra": []} for key, *_rest in SERIES}
    for mu in grid:
        ev = problem.eval_grid(mu, DEVICE, **GRID)
        true = problem.reference(ev.coords, mu)
        rec = {"mu": _mu_key(mu), "x": mu[0]}
        if len(mu) > 1:
            rec["y"] = mu[1]
        for key, (model, snapshot) in loaded.items():
            err = _mae(problem, model, ev, true)
            rec[key] = err
            region = _region(mu, train, snapshot)
            if region in {"intra", "extra"}:
                buckets[key][region].append(err)
        rows.append(rec)
        print(f"  {_mu_key(mu)}", flush=True)
    scores = {}
    for key, *_rest in SERIES:
        scores[key] = {
            "intra": _gmean(buckets[key]["intra"]) if buckets[key]["intra"] else float("nan"),
            "extra": _gmean(buckets[key]["extra"]) if buckets[key]["extra"] else float("nan"),
            "n_intra": len(buckets[key]["intra"]),
            "n_extra": len(buckets[key]["extra"]),
        }
    return {
        "name": name,
        "xlabel": xlabel,
        "train": train,
        "interp": interp,
        "extrap": extrap,
        "rows": rows,
        "scores": scores,
    }


def plot_family(fam: dict, path: Path) -> None:
    titles = {
        "poisson": "Poisson",
        "heat": "Heat",
        "burgers": "Burgers",
        "fisher_kpp": "Fisher–KPP",
        "lamb_chaplygin": "Lamb–Chaplygin",
    }
    fig, ax = plt.subplots(figsize=(9.6, 6.6), layout="constrained")
    xs = [_x_of(tuple(float(v) for v in row["mu"].split(",")), fam) for row in fam["rows"]]
    order = np.argsort(xs)
    xs = np.asarray(xs)[order]
    for key, label, color, style, _split in SERIES:
        ys = np.asarray([fam["rows"][i][key] for i in order])
        ax.plot(xs, ys, color=color, lw=1.5, ls=style, label=label)
    for i, mu in enumerate(fam["train"]):
        ax.axvline(_x_of(mu, fam), color="#9aa0a6", lw=0.8, ls="--", zorder=0, label="Training parameter" if i == 0 else None)
    ax.axvline(_x_of(fam["interp"], fam), color="#1f4e79", lw=0.9, ls=":", zorder=0, label="Numerical head, interpolation")
    ax.axvline(_x_of(fam["extrap"], fam), color="#b85c38", lw=0.9, ls=":", zorder=0, label="Numerical head, extrapolation")
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
    fig.legend(loc="outside lower center", ncol=2, fontsize=7, frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}", flush=True)


def write_curves(families, path: Path) -> None:
    fields = ["pde", "mu", "x", "y"] + [key for key, *_rest in SERIES]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for fam in families:
            for row in fam["rows"]:
                out = {"pde": fam["name"], "mu": row["mu"], "x": row["x"], "y": row.get("y", "")}
                for key, *_rest in SERIES:
                    out[key] = row[key]
                writer.writerow(out)


def write_scores(families, path: Path) -> None:
    titles = {
        "poisson": "Poisson",
        "heat": "Heat",
        "burgers": "Burgers",
        "fisher_kpp": "Fisher–KPP",
        "lamb_chaplygin": "Lamb–Chaplygin",
    }
    rows = []
    for fam in families:
        scores = fam["scores"]
        for key, label, *_rest in SERIES:
            rows.append(
                {
                    "pde": fam["name"],
                    "family": titles[fam["name"]],
                    "variant": label,
                    "key": key,
                    "intra": f"{scores[key]['intra']:.6e}",
                    "extra": f"{scores[key]['extra']:.6e}",
                }
            )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pde", "family", "variant", "key", "intra", "extra"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}", flush=True)


def main() -> None:
    names = ["poisson", "heat", "burgers", "fisher_kpp", "lamb_chaplygin"]
    families = []
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        fam = evaluate_family(name)
        plot_family(fam, OUT / f"{name}.png")
        families.append(fam)
    write_curves(families, OUT / "curves.csv")
    write_scores(families, OUT / "scores.csv")


if __name__ == "__main__":
    main()
