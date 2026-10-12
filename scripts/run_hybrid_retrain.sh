#!/usr/bin/env bash
# Hybrid retraining from the stored two-phase softmax checkpoints.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

DEVICE="${DEVICE:-auto}"

python -m mtl_final hybrid \
  --epochs 20000 \
  --seed 50 \
  --device "$DEVICE" \
  --checkpoint-root "$ROOT/outputs/comparison"
