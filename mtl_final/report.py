"""Join per-run CSVs into one comparison table."""

from __future__ import annotations

import csv
from pathlib import Path


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_comparison_report(root: Path) -> Path:
    root = Path(root)
    score_rows = []
    weight_rows = []
    complex_rows = []
    for metrics_path in sorted(root.rglob("train_metrics.csv")):
        run_dir = metrics_path.parent
        metrics = _read(metrics_path)
        if not metrics:
            continue
        meta = metrics[0]
        complex_rows.append(meta)
        for row in _read(run_dir / "evaluation.csv"):
            if row.get("split") not in {"interpolation", "extrapolation"}:
                continue
            merged = dict(meta)
            merged.update({k: row[k] for k in row if k not in merged or k in {"mu", "split", "mae_mean", "test_time_per_query_s"}})
            for key in row:
                if key.startswith("mae_") or key in {"mu", "split", "test_time_per_query_s", "test_time_per_point_s"}:
                    merged[key] = row[key]
            score_rows.append(merged)
        weight_rows.extend(_read(run_dir / "voter_weights.csv"))

    scores_path = root / "comparison_scores.csv"
    weights_path = root / "comparison_weights.csv"
    complex_path = root / "comparison_complexity.csv"
    _write(scores_path, score_rows)
    _write(weights_path, weight_rows)
    _write(complex_path, complex_rows)
    return scores_path


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
