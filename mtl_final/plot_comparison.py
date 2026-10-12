"""Plot interpolation and extrapolation fields for every baseline in a family."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mtl_final.config import PDE_NAMES
from mtl_final.utils import package_root

PDE_TITLE = {
    "poisson": "Poisson",
    "heat": "Heat",
    "burgers": "Burgers",
    "fisher_kpp": "Fisher–KPP",
    "lamb_chaplygin": "Lamb–Chaplygin",
}

# Column order matches the comparison table.
VARIANTS = [
    ("baseline", "pinn", None, None, "PINN"),
    ("mtl", "one-phase", "softmax", "muvoter", "1φ soft μ"),
    ("mtl", "one-phase", "nosoftmax", "muvoter", "1φ raw μ"),
    ("mtl", "two-phase", "softmax", "muvoter", "2φ soft μ"),
    ("mtl", "two-phase", "nosoftmax", "muvoter", "2φ raw μ"),
]

FIELDS = {
    "poisson": ("u",),
    "heat": ("u",),
    "burgers": ("u",),
    "fisher_kpp": ("u",),
    "lamb_chaplygin": ("psi", "omega"),
}


def _read_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty field file: {path}")
    skip = {"pde", "mode", "softmax", "time_conditioned_voter", "hybrid", "seed", "mu", "split"}
    out = {}
    for key in rows[0]:
        if key in skip:
            out[key] = rows[0][key]
            continue
        out[key] = np.array([float(row[key]) for row in rows], dtype=np.float64)
    return out


def _on_axes(x: np.ndarray, y: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xu, ix = np.unique(np.round(x, 8), return_inverse=True)
    yu, iy = np.unique(np.round(y, 8), return_inverse=True)
    grid = np.empty((yu.size, xu.size), dtype=np.float64)
    grid[iy, ix] = values
    return xu, yu, grid


def _variant_dirs(root: Path, pde: str, seed: int) -> list[tuple[str, Path]]:
    found = []
    for kind, mode, softmax, voter, label in VARIANTS:
        if kind == "baseline":
            path = root / pde / "baseline" / "pinn" / f"seed{seed}"
        else:
            path = root / pde / "mtl" / mode / softmax / voter / f"seed{seed}"
        if (path / "fields").is_dir():
            found.append((label, path))
    if not found:
        raise FileNotFoundError(f"no baselines for {pde} under {root} (seed {seed})")
    return found


def _split_file(run: Path, split: str) -> Path:
    matches = sorted((run / "fields").glob(f"{split}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"no {split} field in {run / 'fields'}")
    return matches[0]


def _mae(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(true - pred)))


def plot_family(root: Path, pde: str, seed: int, save_dir: Path | None, show: bool) -> list[Path]:
    runs = _variant_dirs(root, pde, seed)
    saved = []
    for field in FIELDS[pde]:
        panels = []
        for label, run in runs:
            interp = _read_columns(_split_file(run, "interpolation"))
            extrap = _read_columns(_split_file(run, "extrapolation"))
            panels.append((label, interp, extrap))
        ref_interp, ref_extrap = panels[0][1], panels[0][2]
        spatial = "y" in ref_interp
        if spatial or "t" in ref_interp and pde != "poisson":
            path = _plot_grids(pde, field, panels, ref_interp, ref_extrap, spatial, save_dir)
        else:
            path = _plot_lines(pde, field, panels, save_dir)
        if path is not None:
            saved.append(path)
        if show:
            plt.show()
        else:
            plt.close("all")
    return saved


def _plot_lines(pde: str, field: str, panels, save_dir: Path | None) -> Path | None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(panels), 1)))
    for ax, split, key in (
        (axes[0], "interpolation", "interp"),
        (axes[1], "extrapolation", "extrap"),
    ):
        sample = panels[0][1 if split == "interpolation" else 2]
        x = sample["x"]
        order = np.argsort(x)
        true = sample[f"{field}_true"]
        ax.plot(x[order], true[order], color="black", lw=2.2, label="reference", zorder=5)
        for color, (label, interp, extrap) in zip(colors, panels):
            data = interp if split == "interpolation" else extrap
            pred = data[f"{field}_pred"]
            mae = _mae(data[f"{field}_true"], pred)
            ax.plot(data["x"][order], pred[order], color=color, lw=1.3, label=f"{label}  {mae:.2e}")
        mu = sample.get("mu", "")
        ax.set_title(f"{split}  μ={mu}")
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(field)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.suptitle(f"{PDE_TITLE[pde]}: all baselines")
    fig.tight_layout()
    return _save(fig, save_dir, f"{pde}_{field}")


def _plot_grids(pde, field, panels, ref_interp, ref_extrap, spatial: bool, save_dir: Path | None) -> Path | None:
    n = len(panels)
    fig, axes = plt.subplots(
        2, n + 1, figsize=(2.15 * (n + 1), 5.6), squeeze=False, layout="constrained"
    )
    y_name = "y" if spatial else "t"
    rows = (
        (0, "interpolation", ref_interp),
        (1, "extrapolation", ref_extrap),
    )
    for row, split, ref in rows:
        x, y, truth = _on_axes(ref["x"], ref[y_name], ref[f"{field}_true"])
        vmin, vmax = float(np.min(truth)), float(np.max(truth))
        if vmin == vmax:
            vmax = vmin + 1e-6
        images = []
        extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]
        im = axes[row, 0].imshow(
            truth, origin="lower", extent=extent, aspect="auto", vmin=vmin, vmax=vmax, cmap="RdBu_r"
        )
        images.append(im)
        mu = ref.get("mu", "")
        axes[row, 0].set_title(f"reference\n{split[:5]} μ={mu}", fontsize=8)
        for col, (label, interp, extrap) in enumerate(panels, start=1):
            data = interp if split == "interpolation" else extrap
            _, _, pred = _on_axes(data["x"], data[y_name], data[f"{field}_pred"])
            mae = _mae(data[f"{field}_true"], data[f"{field}_pred"])
            im = axes[row, col].imshow(
                pred, origin="lower", extent=extent, aspect="auto", vmin=vmin, vmax=vmax, cmap="RdBu_r"
            )
            images.append(im)
            axes[row, col].set_title(f"{label}\nMAE {mae:.2e}", fontsize=8)
        for ax in axes[row]:
            ax.set_xlabel("x", fontsize=7)
            ax.tick_params(labelsize=6)
        axes[row, 0].set_ylabel(y_name, fontsize=8)
        fig.colorbar(images[-1], ax=axes[row, :].tolist(), fraction=0.015, pad=0.01)
    fig.suptitle(f"{PDE_TITLE[pde]}: {field}, interpolation (top) and extrapolation (bottom)")
    return _save(fig, save_dir, f"{pde}_{field}")


def _save(fig, save_dir: Path | None, stem: str) -> Path | None:
    if save_dir is None:
        return None
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{stem}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"saved {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show interpolation and extrapolation fields for every baseline, one figure per PDE family.",
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=package_root() / "outputs" / "comparison",
        help="Root written by `python -m mtl_final compare`.",
    )
    parser.add_argument("--pdes", nargs="+", default=list(PDE_NAMES), choices=PDE_NAMES)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory for PNG figures. Default: <comparison-dir>/figures.",
    )
    parser.add_argument("--show", action="store_true", help="Open an interactive window as well as saving.")
    parser.add_argument("--no-save", action="store_true", help="Do not write PNG files.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path(args.comparison_dir)
    save_dir = None if args.no_save else (args.save_dir or root / "figures")
    for pde in args.pdes:
        plot_family(root, pde, args.seed, save_dir, args.show)


if __name__ == "__main__":
    main()
