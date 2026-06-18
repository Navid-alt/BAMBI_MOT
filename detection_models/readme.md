# Detection models — YOLO26 & RF-DETR on the BAMBI dataset

Training/benchmark setup for wildlife **detection** (3 classes: Wild boar, Red
deer, Roe deer) on `data/yolo_data`. All models share **one uv virtual
environment**, log to a **Dockerised MLflow** server, and read the dataset
**in-place** (no copies).

Models: `yolo26-s`, `yolo26-l`, `rf-detr-s`, `rf-detr-l`.

---

## 1. Prerequisites

- [`uv`](https://docs.astral.sh/uv/) and Docker + Docker Compose
- An NVIDIA GPU with the CUDA 12 driver (torch is installed from the cu124 wheels)

The dataset already lives at `/home/stas/my_git/BAMBI_MOT/data/yolo_data`
(`train` 17,684 · `val` 2,868 · `test` 3,277 images).

---

## 2. One-time setup

```bash
cd detection_models

# 2a. Create / sync the single joined venv (torch + ultralytics + rfdetr + mlflow)
uv sync

# 2b. Start the persistent MLflow server (UI at http://localhost:5000)
docker compose up -d
```

MLflow data (experiments, metrics, artifacts) is stored under
`detection_models/mlflow_data/` so it survives restarts. Stop with
`docker compose down` (data is kept).



## 3. Speed benchmark (quick)

Trains all four models (`yolo26-s/l`, `rf-detr-s/l`) for **1 epoch on a
~500-image subset** at **resolution 736** (small models use a larger batch, the
`-l` models a smaller one), then writes `reports/speed_report.md` with
throughput, peak VRAM and timings **extrapolated to the full 17,684-image epoch
(and ×50 epochs)**:

```bash
./benchmark_speed.sh
# or:  N_TRAIN=1000 N_VAL=100 ./benchmark_speed.sh
```

The subset is built with symlinks (no image copies). Each run is logged to the
MLflow **`bambi-bench`** experiment (separate from training). 

---

## 4. Full training

Each model has its own `train.py`. 
raise `--batch` / `--imgsz` 

```bash
# YOLO26 — reads data.yaml directly
uv run python yolo26-s/train.py           
uv run python yolo26-l/train.py            #

# RF-DETR 
uv run python tools/make_rfdetr_view.py
uv run python rf-detr-s/train.py           
uv run python rf-detr-l/train.py           
```

Common overrides: `--epochs`, `--batch`/`--grad-accum`, `--imgsz`, `--device`.

---

## 5. What lands in MLflow

Open http://localhost:5000. Two experiments: **`bambi-detection`** (real
`train.py` runs) and **`bambi-bench`** (the speed benchmark).

| | YOLO26 | RF-DETR |
|---|---|---|
| Metrics (mAP50, mAP50-95, losses, P/R) | ✅ per epoch | ✅ per epoch (COCO mAP/AP/AR) |
| Params / hyperparameters | ✅ | ✅ |
| Best/last weights as artifacts | ✅ | ✅ (checkpoints in `output/`) |
| **Confusion matrix** | ✅ `confusion_matrix.png` artifact | ❌ not produced (detection eval is COCO mAP) |
| PR / results curves | ✅ PNG artifacts | partial |






