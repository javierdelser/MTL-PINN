#!/usr/bin/env python3
"""Write query and training FLOPs for every variant to a CSV."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtl_final.complexity import all_costs


def main() -> None:
    rows = all_costs()
    path = ROOT / "outputs" / "complexity.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["pde", "variant", "group", "n_parameters", "query_flops", "train_flops"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
