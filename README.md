# BAMBI Wildlife MOT

Benchmarking and improving **detection** and **multi-object tracking (MOT)** for
aerial thermal wildlife monitoring on the **BAMBI** dataset.

**Course:** CVI4
**Group:** Stanislav Buinitski, Yahaya Danjuma, Navid Ghaderian, Afzaal Yasin
**Dataset:** https://www.bambi.eco/ · https://github.com/bambi-eco/Dataset

---

## 1. Project overview

We work with the **BAMBI** dataset (Bounding boxes for Animals in Multi-species
from Infrared cameras): ~21,509 thermal frames across 240 nadir UAV video
sequences, three species — **Wild boar**, **Red deer**, **Roe deer** — annotated
in COCO format.

The project has three connected parts:

1. **Data pipeline** (`notebooks/`, `src/`) — download BAMBI, convert
   annotations, extract frames, and produce both a YOLO dataset and per-sequence
   tracking splits.
2. **Detection** (`detection_models/`) — train and benchmark **YOLO26** and
   **RF-DETR** (small + large) detectors on the thermal frames.
3. **Tracking** — two paradigms compared on the same data:
   - **Tracking-by-detection baselines** (`notebooks/06`, `07`,
     [`README(Track).md`](README(Track).md)) — SORT, ByteTrack, BoT-SORT,
     BoT-SORT + ReID via the `BoxMOT` framework.
   - **Learned end-to-end tracking** (`transformer_tracking/`) — a MOTRv2-style
     `TrackAssociator` trained on top of the frozen RF-DETR detector.

---

## 2. Repository layout

| Path | What's in it |
|---|---|
| `notebooks/` | End-to-end data + baseline-tracking pipeline (numbered `01`–`07`) |
| `src/` | Shared helpers: `config.py` (config loader), `annotation_process.py` (BAMBI → YOLO labels) |
| `annotations/` | COCO annotation splits (`instances_{train,val,test,default}.json`) |
| `data/` | Generated frames, labels, and the YOLO/tracking splits — **not tracked** (see §4) |
| `detection_models/` | YOLO26 + RF-DETR training, benchmarking, reports — see [`detection_models/readme.md`](detection_models/readme.md) |
| `transformer_tracking/` | Learned RF-DETR tracker — see [`transformer_tracking/README.md`](transformer_tracking/README.md) and [`docs/report.pdf`](transformer_tracking/docs/report.pdf) |
| `plots/` | Tracking-benchmark figures used in `README(Track).md` |
| `presentation/` | Project slides (`CVI_presentation.pptx`, `CVI_project.pdf`) and the written report |
| `experiments/`, `tests/` | Placeholders for experiment outputs and tests |

Three documents cover the work in depth:
- [`README(Track).md`](README(Track).md) — tracking-by-detection benchmark + results.
- [`detection_models/readme.md`](detection_models/readme.md) — detector training/benchmarking.
- [`transformer_tracking/README.md`](transformer_tracking/README.md) + [report](transformer_tracking/docs/report.pdf) — learned tracker.

---

## 3. Setup

### Base environment (data pipeline + tracking baselines)

```bash
git clone https://github.com/Navid-alt/BAMBI_MOT.git
cd BAMBI_MOT

# Local development (Jupyter, linting, tests included)
pip install -e ".[dev]"

# On Google Colab
pip install -e ".[colab]"
```

The tracking-baseline notebooks additionally need `torchreid` / `filterpy`
(see [`README(Track).md`](README(Track).md) §8).

### Detection + learned tracking environment

`detection_models/` and `transformer_tracking/` share a **separate, GPU-focused
[`uv`](https://docs.astral.sh/uv/) environment** (PyTorch cu124 + Ultralytics +
RF-DETR + MLflow). It is independent of the base install above:

```bash
cd detection_models
uv sync                 # create the joined venv
docker compose up -d    # MLflow UI at http://localhost:5000
```

Full details in [`detection_models/readme.md`](detection_models/readme.md).

---

## 4. Data (not tracked in the repo)

The `data/` directory and all model weights are **git-ignored** because of their
size, the data was not changed either. they are not part of this repository. To reproduce them:

1. **Download BAMBI** from https://www.bambi.eco/ (see `notebooks/02_Download.ipynb`).
2. **Configure paths:** copy [`config.example.yaml`](config.example.yaml) to
   `config.local.yaml` and set `paths.bambi_data` to your download location.
   `config.local.yaml` is git-ignored so each person keeps their own paths.
3. **Run the notebooks in order** (§5) to regenerate `data/frames`,
   `data/labels`, and `data/yolo_data`.

> **Note on paths:** some scripts (e.g. `detection_models/data.yaml`,
> `transformer_tracking`) contain absolute paths from the original author's
> machine (e.g. `/home/stas/...`). Update these to your local checkout before
> running.

---

## 5. Pipeline (notebooks)

Run the numbered notebooks in `notebooks/` in order:

| Step | Notebook | Description |
|---|---|---|
| 1 | `01_Exploration.ipynb` | Inspect the BAMBI dataset and annotations |
| 2 | `02_Download.ipynb` | Download the raw BAMBI data |
| 3 | `03_Format_Conversion.ipynb` | Convert BAMBI annotations → COCO / YOLO labels |
| 4 | `04_Frame_Extraction.ipynb` | Extract thermal frames from the sequences |
| 5 | `05_Data_Split.ipynb` | Build train/val/test splits (YOLO + per-sequence tracking) |
| 6 | `06_Run_Tracker.ipynb` | Run SORT / ByteTrack / BoT-SORT (+ ReID) and visualize trajectories |
| 7 | `07_Benchmark_Tracker.ipynb` | Score trackers (MOTA / IDF1 / IDSW) and produce comparison plots |

Detection training (after step 5) and learned tracking are driven from their own
sub-folders — see their READMEs.

---

## 6. Model weights

Trained weights (YOLO26, RF-DETR, the `TrackAssociator`) are published on the
**Hugging Face Hub** (kept out of git so the repo stays lightweight):

### 📦 [`NavidGh/BambiMot`](https://huggingface.co/NavidGh/BambiMot)

Download a file and place it at the path its script expects (the layout on the
Hub mirrors this repo), then detection/tracking picks it up automatically.

| Model | File on the Hub | Place at |
|---|---|---|
| YOLO26-s (640) | `yolo26-s/runs/train_yolo26-s/weights/best.pt` | `detection_models/yolo26-s/runs/train_yolo26-s/weights/best.pt` |
| YOLO26-l (1024) | `yolo26-l/runs/train_yolo26-l_1024/weights/best.pt` | `detection_models/yolo26-l/runs/train_yolo26-l_1024/weights/best.pt` |
| RF-DETR-l (EMA, recommended) | `rf-detr-l/output/checkpoint_best_ema.pth` | `detection_models/rf-detr-l/output/checkpoint_best_ema.pth` |
| TrackAssociator | `transformer_tracking/output/associator_last.pth` | `transformer_tracking/output/associator_last.pth` |

The Hub repo also holds the `last.pt` / `best_regular.pth` variants, the full
training-state `.ckpt` files, and all per-epoch associator checkpoints. Download
one file with:

```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download("NavidGh/BambiMot", "rf-detr-l/output/checkpoint_best_ema.pth")
```

Everything needed to regenerate the weights bit-for-bit is also committed —
reproduce them by running the training scripts:

- **Detectors:** `detection_models/{yolo26,rf-detr}-{s,l}/train.py` — best/last
  weights are logged as **MLflow artifacts** and written under each model's
  `output/` (e.g. `detection_models/rf-detr-l/output/checkpoint_best_ema.pth`).
- **Learned tracker:** `transformer_tracking/train.sh` — the associator loads the
  frozen RF-DETR checkpoint above and trains the matcher only.

See [`detection_models/readme.md`](detection_models/readme.md) §4–5 and
[`transformer_tracking/README.md`](transformer_tracking/README.md) for exact
commands.

---

## 7. Results (summary)

**Tracking-by-detection** (ground-truth boxes, full benchmark in
[`README(Track).md`](README(Track).md)):

| Tracker | MOTA | IDF1 | TP | FN |
|---|---|---|---|---|
| SORT | 0.139 | 0.200 | 11,470 | 49,783 |
| ByteTrack | 0.230 | 0.301 | 21,427 | 39,826 |
| BoT-SORT | 0.352 | 0.386 | 28,907 | 32,346 |
| **BoT-SORT + ReID (fixed)** | **0.356** | **0.388** | **29,216** | 32,037 |

**Detection** and **learned tracking** results, plots, and qualitative overlays
are in [`detection_models/reports/`](detection_models/reports/) and
[`transformer_tracking/docs/report.pdf`](transformer_tracking/docs/report.pdf).
The combined write-up and slides are in [`presentation/`](presentation/).

---

## 8. References

- **BAMBI** dataset: https://www.bambi.eco/ · https://github.com/bambi-eco/Dataset
- **ByteTrack** — Zhang et al., ECCV 2022
- **BoT-SORT** — Aharon et al., arXiv 2022
- **MOTRv2** — Zhang et al., CVPR 2023
- **RF-DETR** — Roboflow
- **HOTA** — Luiten et al., IJCV 2020
