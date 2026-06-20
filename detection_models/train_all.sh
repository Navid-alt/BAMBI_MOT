#!/usr/bin/env bash
# Train all four detectors sequentially (a simple queue) on the 12 GB RTX 4070,
# logging to the MLflow "bambi-detection" experiment. Per-step batch sizes are
# the ones the speed benchmark confirmed fit the card; for RF-DETR, --grad-accum
# keeps the effective batch ~16 (train.py's default intent). Each model uses its
# own train.py defaults (50 epochs, imgsz/resolution, lr, ...) unless overridden.
#
#   ./train_all.sh
#
# A failure in one model is reported but does NOT stop the queue (no `set -e`).
set -uo pipefail
cd "$(dirname "$0")"

# Pin GPU to the 4070 (device 0) and silence albumentations warnings
export CUDA_VISIBLE_DEVICES=0
export NO_ALBUMENTATIONS_UPDATE=1

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-bambi-detection}"

echo "==> Building RF-DETR dataset view (val -> valid symlinks)"
uv run python tools/make_rfdetr_view.py

run_one () {
  local name="$1"; shift
  echo ""
  echo "==> [$(date '+%H:%M:%S')] training ${name}  ($*)"
  if uv run python "${name}/train.py" "$@"; then
    echo "==> ${name} done"
  else
    echo "==> ${name} FAILED (continuing queue)"
  fi
}

# All four train at the native 1024x1024 frame size (no downscaling). Batches are
# lowered from the 736 benchmark values to fit the 12 GB card at ~1.9x the pixels,
# and RF-DETR's --grad-accum is raised to keep the effective batch unchanged.
# Queue order as requested: yolo26-s, yolo26-l, rf-detr-l, rf-detr-s.
#run_one yolo26-s  --imgsz 1024 --batch 12 --device 0
#run_one yolo26-l  --imgsz 1024 --batch 5 --device 0
run_one rf-detr-l --imgsz 1024 --batch 3 --grad-accum 6 --device cuda --epochs 30  # effective batch 18
#run_one rf-detr-s --imgsz 1024 --batch 4 --grad-accum 4 --device cuda   # effective batch 16

echo ""
echo "All training jobs finished. MLflow: ${MLFLOW_TRACKING_URI} (experiment: ${MLFLOW_EXPERIMENT_NAME})"
f