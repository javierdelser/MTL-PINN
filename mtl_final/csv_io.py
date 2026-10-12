"""CSV writers used by every experiment."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def write_csv(path: Path, rows: Sequence[Mapping], fieldnames: Sequence[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return path
    names = list(fieldnames or rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in names})
    return path


def append_csv(path: Path, rows: Sequence[Mapping], fieldnames: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in fieldnames})
    return path


def write_field_csv(
    path: Path,
    *,
    coords: Mapping[str, np.ndarray],
    true_fields: Mapping[str, np.ndarray],
    pred_fields: Mapping[str, np.ndarray],
    extra: Mapping | None = None,
) -> Path:
    """Flatten grids so true and predicted fields can be plotted later."""
    extra = dict(extra or {})
    keys = list(coords.keys())
    n = int(np.asarray(next(iter(coords.values()))).size)
    rows = []
    for i in range(n):
        row = dict(extra)
        for name in keys:
            row[name] = np.asarray(coords[name]).reshape(-1)[i]
        for name, arr in true_fields.items():
            row[f"{name}_true"] = np.asarray(arr).reshape(-1)[i]
        for name, arr in pred_fields.items():
            row[f"{name}_pred"] = np.asarray(arr).reshape(-1)[i]
        rows.append(row)
    fieldnames = list(extra.keys()) + keys
    fieldnames += [f"{n}_true" for n in true_fields]
    fieldnames += [f"{n}_pred" for n in pred_fields]
    return write_csv(path, rows, fieldnames)


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value).reshape(-1)
        return ",".join(f"{float(v):.8g}" for v in arr)
    if isinstance(value, float):
        return f"{value:.8g}"
    return value
