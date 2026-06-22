#!/usr/bin/env bash
# One-time: run frozen RF-DETR over every frame and cache its detections so
# training the matcher no longer recomputes the detector each epoch.
# Caches train + val by default (add 'test' as an arg to include it).
#
# Usage:  ./precompute.sh [splits...]    (default: train val)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPLITS=("${@:-train val}")

for SPLIT in ${SPLITS[@]}; do
    echo "=== precomputing embeddings: $SPLIT ==="
    PYTHONPATH="$REPO" uv --directory "$REPO/detection_models" run \
        python "$REPO/transformer_tracking/precompute_embeddings.py" --split "$SPLIT"
done
