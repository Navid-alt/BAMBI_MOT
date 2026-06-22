#!/usr/bin/env bash
# Train the RF-DETR track associator (frozen detector, MOTRv2-style).
# Reuses the detection_models uv env (torch + rfdetr + mlflow) and the repo's
# annotations/ + data/yolo_data/ in place. Start MLflow first if you want logging:
#   (cd detection_models && docker compose up -d)
#
# Usage:  ./train.sh [extra train.py args...]
#   e.g.  ./train.sh --epochs 30 --lr 5e-5 --mlflow
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CACHE="$REPO/transformer_tracking/output/cache"
if [ ! -d "$CACHE/train" ]; then
    echo "No embedding cache at $CACHE/train."
    echo "Build it once first:  ./precompute.sh"
    exit 1
fi

PYTHONPATH="$REPO" uv --directory "$REPO/detection_models" run \
    python "$REPO/transformer_tracking/train.py" \
    --epochs 10 \
    --clip-len 2 \
    --lr 1e-4 \
    --grad-accum 8 \
    --cache-dir "$CACHE" \
    --mlflow \
    "$@"
