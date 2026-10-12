"""Evaluation: MAE, voter weights, field CSVs, per-query latency."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from mtl_final.csv_io import append_csv, write_csv, write_field_csv
from mtl_final.hybrid import HybridMTL


def _predict(model, grid, hybrid_solver_coords=None):
    coords = grid.coord_tensors[0]
    params = grid.param_tensor
    t = grid.time_tensor
    if isinstance(model, HybridMTL):
        sc = hybrid_solver_coords
        if sc is None:
            names = model.snapshot.axes
            # split coords to match snapshot dimensionality
            if len(names) == 1:
                sc = (coords[:, 0:1],)
            elif len(names) == 2:
                sc = (coords[:, 0:1], coords[:, 1:2])
            else:
                sc = tuple(coords[:, i : i + 1] for i in range(len(names)))
        return model(coords, params, t=t, solver_coords=sc)
    return model(coords, params, t=t)


@torch.no_grad()
def evaluate_run(problem, model, device, out: Path, spec_meta: dict, grid_kw: dict | None = None) -> dict:
    model.eval()
    grid_kw = dict(grid_kw or {})
    rows = []
    weight_rows = []
    query_times = []
    mus = list(problem.spec.task_params) + list(problem.spec.query_params)
    labels = {}
    for mu in problem.spec.task_params:
        labels[mu] = "train"
    labels.update(problem.spec.query_labels)

    for mu in mus:
        grid = problem.eval_grid(mu, device, **grid_kw)
        t0 = time.perf_counter()
        pred = _predict(model, grid)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n_pts = pred.shape[0]
        query_times.append(dt / max(n_pts, 1))
        pred_np = pred.detach().cpu().numpy()
        true = problem.reference(grid.coords, mu)
        pred_fields = {}
        mae = {}
        for j, name in enumerate(problem.spec.field_names):
            arr = pred_np[:, j].reshape(grid.shape)
            pred_fields[name] = arr
            mae[name] = float(np.mean(np.abs(arr - np.asarray(true[name]))))
        split = labels.get(mu, "query")
        mu_key = ",".join(f"{v:g}" for v in mu)
        rows.append(
            {
                **spec_meta,
                "mu": mu_key,
                "split": split,
                **{f"mae_{k}": v for k, v in mae.items()},
                "mae_mean": float(np.mean(list(mae.values()))),
                "test_time_per_query_s": dt,
                "test_time_per_point_s": dt / max(n_pts, 1),
            }
        )
        write_field_csv(
            out / "fields" / f"{split}_{mu_key.replace(',', '_')}.csv",
            coords=grid.coords,
            true_fields=true,
            pred_fields=pred_fields,
            extra={**spec_meta, "mu": mu_key, "split": split},
        )

        t_for_w = grid.time_tensor
        if problem.spec.has_time and t_for_w is not None:
            times = [
                float(t_for_w.min().item()),
                float(t_for_w.median().item()),
                float(t_for_w.max().item()),
            ]
            unique_t = sorted(set(times))
        else:
            unique_t = [None]
        has_voter = hasattr(model, "get_attention_weights") and spec_meta.get("mode") != "baseline"
        for tv in unique_t:
            row = {
                **spec_meta,
                "mu": mu_key,
                "split": split,
                "t": "" if tv is None else tv,
                "has_voter": int(has_voter),
            }
            if has_voter:
                p = grid.param_tensor[:1]
                t_in = None if tv is None else torch.full((1, 1), tv, device=device)
                w = model.get_attention_weights(p, t_in).detach().cpu().numpy().reshape(-1)
                row.update({f"w{i}": float(w[i]) for i in range(len(w))})
                row["weights"] = w
            weight_rows.append(row)

    write_csv(out / "evaluation.csv", rows)
    write_csv(out / "voter_weights.csv", weight_rows)
    mean_test = float(np.mean(query_times)) if query_times else float("nan")
    summary = {
        **spec_meta,
        "mean_test_time_per_point_s": mean_test,
        "n_eval_queries": len(mus),
    }
    write_csv(out / "eval_summary.csv", [summary], list(summary.keys()))
    return {"rows": rows, "weights": weight_rows, "mean_test_time_per_point_s": mean_test}


SUMMARY_FIELDS = [
    "pde",
    "mode",
    "softmax",
    "time_conditioned_voter",
    "hybrid",
    "snapshot_split",
    "snapshot_mu",
    "seed",
    "n_parameters",
    "n_trainable_parameters",
    "flops_per_forward",
    "peak_memory_mb",
    "n_epochs",
    "time_per_epoch_s",
    "total_train_time_s",
    "mean_test_time_per_point_s",
    "output_dir",
]


def append_master_summary(root: Path, row: dict) -> None:
    append_csv(root / "summary_metrics.csv", [row], SUMMARY_FIELDS)
