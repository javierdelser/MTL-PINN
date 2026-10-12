#!/usr/bin/env bash
# Seed-50 comparison and hybrid retraining.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

DEVICE="${DEVICE:-auto}"

python -m mtl_final compare \
  --epochs 20000 \
  --seed 50 \
  --device "$DEVICE"

python -m mtl_final hybrid \
  --epochs 20000 \
  --seed 50 \
  --device "$DEVICE"
