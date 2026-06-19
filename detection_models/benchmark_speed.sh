#!/usr/bin/env bash
# Speed smoke-test: train yolo26-s/l and rf-detr-s/l for 1 epoch on a small
# subset (all at the native resolution 1024), measure throughput + VRAM, and write
# reports/speed_report.md with figures extrapolated to the full 17,684-image
# training set. Each run is also logged to the MLflow "bambi-bench" experiment.
#
#   ./benchmark_speed.sh                  # defaults: 500 train / 50 val imgs
#   N_TRAIN=1000 N_VAL=100 ./benchmark_speed.sh
#
# Each model runs in its own process so peak VRAM is measured cleanly and an OOM
# in one model does not stop the others.
set -euo pipefail
cd "$(dirname "$0")"

N_TRAIN="${N_TRAIN:-500}"
N_VAL="${N_VAL:-50}"

# Pin GPU to the 4070 (device 0) and silence albumentations warnings
export CUDA_VISIBLE_DEVICES=0
export NO_ALBUMENTATIONS_UPDATE=1

# Route MLflow logging to the Docker tracking server, into a dedicated benchmark
# experiment kept separate from the "bambi-detection" training experiment.
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-bambi-bench}"

echo "==> Building speed-test subset (n_train=${N_TRAIN}, n_val=${N_VAL})"
uv run python tools/make_subset.py --n-train "${N_TRAIN}" --n-val "${N_VAL}"

rm -rf reports/_bench
mkdir -p reports/_bench

MODELS=(yolo26-s yolo26-l rf-detr-s rf-detr-l yolo26-s-640 yolo26-l-640 rf-detr-s-640 rf-detr-l-640)
i=0
for m in "${MODELS[@]}"; do
  i=$((i + 1))
  echo "==> [${i}/${#MODELS[@]}] ${m}  (1 epoch on subset)"
  uv run python tools/benchmark.py --model "${m}" || true
done

echo "==> Writing report"
uv run python tools/make_report.py

echo "==> Rendering PDF"
uv run python tools/report_to_pdf.py || echo "(PDF skipped — see error above)"

echo ""
echo "Done -> reports/speed_report.md (+ .pdf)"
