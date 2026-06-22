#!/usr/bin/env bash
# Run online tracking with a trained associator, then score it (MOTA/IDF1/IDSW).
# Scoring needs motmetrics:  uv --directory ../detection_models add motmetrics
#
# Usage:  ./track_eval.sh [split] [weights]
#   e.g.  ./track_eval.sh val output/associator_last.pth
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPLIT="${1:-val}"
WEIGHTS="${2:-$REPO/transformer_tracking/output/associator_last.pth}"

run() { PYTHONPATH="$REPO" uv --directory "$REPO/detection_models" run python "$@"; }

run "$REPO/transformer_tracking/infer.py" --split "$SPLIT" --weights "$WEIGHTS"
run "$REPO/transformer_tracking/eval.py"  --split "$SPLIT"
