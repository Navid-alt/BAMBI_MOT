# BAMBI Wildlife MOT — Thermal Detection & Multi-Object Tracking

> Source material for the slide deck. Each `##` section maps roughly to one slide
> or a small group of slides. Images live in [images/](images/) and are
> referenced inline. Numbers are pulled directly from the code, reports and
> result files in the repo.

**Course:** CIV4 · **Team:** Stanislav Buinitski, Yahaya Danjuma, Navid Ghaderian, Afzaal Yasin
**Dataset:** BAMBI — *Bounding boxes for Animals in Multi-species from Infrared cameras* (https://www.bambi.eco/)

---

## 1. The one-line pitch

We benchmark and improve **Multi-Object Tracking (MOT)** for **aerial (nadir UAV)
thermal wildlife monitoring**, then build a **reproducible detection-training
stack** (YOLO26 + RF-DETR) on top of the same data.

Two parallel problems, one dataset:

1. **Tracking** — given boxes, how well can we keep a stable identity on each
   animal across a thermal video? (Notebooks 06–07)
2. **Detection** — can we produce those boxes automatically with a trained
   thermal detector? (the `detection_models/` MLOps stack)

---

## 2. Why thermal wildlife tracking is hard

Thermal/infrared aerial footage breaks most assumptions that standard RGB
trackers rely on:

- **No color, no texture.** Classic Re-ID uses color histograms — useless on a
  grayscale heat blob. Association must fall back on motion, shape and learned
  deep features.
- **Thermal flickering & camouflage.** Animals' heat signatures fluctuate and
  blend into sun-warmed soil and rocks, so detections "flicker" in and out and
  tracks die.
- **Severe class imbalance.** The data is dominated by Wild Boar, with far fewer
  Roe/Red Deer — hard to track consistently across different body shapes.
- **Tiny targets, top-down view.** Nadir UAV perspective means small, low-detail
  objects.

![Ground-truth thermal annotation example](images/annotation_example_1.png)
*A sample BAMBI thermal frame with ground-truth boxes — note the low contrast and small targets.*

---

## 3. The dataset in numbers

| Quantity | Value |
|---|---|
| Tracking subset | **21,509 frames** across **240 video sequences** |
| Detection split — train | **17,684 images** |
| Detection split — val | **2,868 images** |
| Detection split — test | **3,277 images** |
| Classes (detection) | **3** — Wild boar (0), Red deer (1), Roe deer (2) |
| Tracking species | Wild Boar (dominant), Roe Deer |
| Annotation format | COCO JSON / BAMBI MOT → converted to YOLO |
| Image resolution | 1024 × 1024 |
| Raw image set on disk | ~16 GB |

Roughly **~91%** of sampled training frames contain at least one animal box; the
rest are intentional **background frames** (kept so the detector learns to say
"nothing here").

---

## 4. Repository structure

```
BAMBI_MOT/
├── notebooks/                  # The tracking research pipeline (ordered 01→07)
│   ├── 01_Exploration.ipynb       # dataset EDA
│   ├── 02_Download.ipynb          # fetch BAMBI
│   ├── 03_Format_Conversion.ipynb # COCO/MOT → YOLO labels
│   ├── 04_Frame_Extraction.ipynb  # video → frames
│   ├── 05_Data_Split.ipynb        # train/val/test split
│   ├── 06_Run_Tracker.ipynb       # run trackers + trajectory viz
│   └── 07_Benchmark_Tracker.ipynb # HOTA/MOTA/IDF1 + analysis
│
├── src/                        # Reusable library code
│   ├── annotation_process.py      # BAMBI MOT → YOLO label writer
│   └── config.py                  # YAML config loader
│
├── detection_models/           # Self-contained detector training stack (MLOps)
│   ├── yolo26-s/ yolo26-l/         # Ultralytics YOLO26 train scripts
│   ├── rf-detr-s/ rf-detr-l/       # RF-DETR (DINOv2 backbone) train scripts
│   ├── tools/                      # benchmark, dataset views, reporting
│   ├── docker-compose.yml          # persistent MLflow server
│   ├── benchmark_speed.sh          # 4-model speed benchmark
│   ├── train_all.sh                # sequential training queue
│   └── reports/                    # speed_report.md/.pdf, annotation examples
│
├── data/yolo_data/{train,val,test}/{images,labels}   # (gitignored)
├── plots/                      # tracking result figures
└── pyproject.toml              # top-level package (bambi-mot)
```

**Design intent:** the repo is deliberately **two layers**:

- A **research layer** (`notebooks/` + `src/`) — exploratory, narrative, for the
  tracking experiments.
- A **production-style layer** (`detection_models/`) — scripted, reproducible,
  experiment-tracked, isolated in its own virtual environment and Docker
  service. It treats detector training like an engineering pipeline, not a
  notebook.

---

## 5. Methodology decision: Ground-Truth Isolation

To measure **pure tracking quality**, the tracking benchmark deliberately
**bypasses the detector** and feeds **human-verified ground-truth boxes** straight
into the trackers.

**Why this matters:** it cleanly separates the two error sources. A bad MOTA
number now reflects the *tracker's* association logic (Kalman prediction, IoU
matching, Re-ID) and **not** a weak upstream detector. This isolates the variable
we're actually studying.

---

## 6. The trackers we compared

Four configurations, all run through the **BoxMOT** framework:

| Tracker | Idea | Re-ID |
|---|---|---|
| **SORT** | Kalman + IoU only (the baseline) | none |
| **ByteTrack** | also recovers low-confidence detections | none |
| **BoT-SORT** | camera-motion compensation + better matching | none |
| **BoT-SORT + ReID** | adds deep appearance features | **OSNet (custom wrapper)** |

---

## 7. Implementation highlight A — Custom thermal Re-ID wrapper

Deep Re-ID libraries assume 3-channel RGB and **crash** on single-channel thermal
input. We wrote an `OSNetReIDWrapper` to bridge **Torchreid's OSNet** into BoxMOT:

1. Intercepts raw thermal image patches (1-channel grayscale numpy).
2. Stacks them into **pseudo-RGB** tensors.
3. Normalizes every crop to a fixed **256 × 128** layout.
4. Runs inference with **FP16 CUDA** acceleration.
5. Flattens embeddings back to CPU **Float32** arrays the tracker can consume.

This is the piece that makes "appearance-based" tracking even *possible* on
colorless thermal data.

---

## 8. Implementation highlight B — The sequence-boundary bug (and the fix)

**The bug:** all **21,509 frames** were initially fed as one continuous video.
Kalman states and track IDs **bled across completely different captures**,
corrupting metrics and instantly killing tracks at every scene cut.

**The fix:** parse the filename prefix (`get_sequence_id`) to group frames into
their **240 real sequences**, and **reset the trackers at each boundary**.

**The payoff — mean track lifetime:**

| | Mean track lifetime (BoT-SORT + ReID) |
|---|---|
| Before fix (frames bleed across videos) | **4.0 frames** |
| After fix (per-sequence reset) | **36.3 frames** |

That's a **~9× improvement** in how long a track survives — the single most
impactful engineering change in the tracking pipeline.

![Track lifetime before the fix](images/tracklifetime.png)
*Before: tracks die almost immediately (mean 4.0 frames).*

![Track lifetime after per-sequence reset](images/tracklifetime_PerSequence.png)
*After: tracks persist within their own sequence (mean 36.3 frames).*

---

## 9. Implementation highlight C — Short-track noise filtering & trajectory mapping

Thermal noise produces "blips" — micro-tracks from transient hot spots. A
**diagnostic temporal filter discards any track shorter than 60 consecutive
frames**, leaving **29 true long-term wildlife trajectories** out of hundreds of
noisy IDs.

These surviving paths are rendered onto a master thermal canvas (built with
OpenCV) by connecting box centers frame-to-frame, with the persistent ID stamped
at the trajectory's end.

![Long-term wildlife trajectories](images/long_term_tracks.png)
*29 filtered long-term trajectories drawn over the thermal canvas.*

![Predicted boxes vs ground truth](images/Predicted_Vs_GT.png)
*Predicted BoT-SORT+ReID boxes (colored by track ID) overlaid on white ground-truth boxes — tight spatial convergence.*

---

## 10. Tracking results

| Tracker | MOTA | IDF1 | IDSW | True Pos (TP) | False Neg (FN) |
|---|:---:|:---:|:---:|:---:|:---:|
| SORT (baseline) | 0.1389 | 0.2001 | 2,875 | 11,470 | 49,783 |
| ByteTrack | 0.2300 | 0.3010 | 6,630 | 21,427 | 39,826 |
| BoT-SORT (default) | 0.3523 | 0.3856 | 7,324 | 28,907 | 32,346 |
| BoT-SORT + ReID (broken boundaries) | 0.3518 | 0.3855 | 7,343 | 28,898 | 32,355 |
| **BoT-SORT + ReID (fixed sequences)** | **0.3559** | **0.3883** | 7,409 | **29,216** | **32,037** |

**Headline:** BoT-SORT + ReID with the sequence fix wins on every quality metric.
Compared to the SORT baseline it **more than doubles** true positives
(11,470 → 29,216) and **lifts MOTA by ~2.6×** (0.139 → 0.356).

![Tracker comparison](images/tracker_comparison.png)
![Per-tracker performance breakdown](images/performance_per_tracker.png)

**Reading the numbers carefully (a key talking point):**
- SORT's *low* ID-switch count is **deceptive** — it has few switches only
  because it **dropped most tracks entirely**. BoT-SORT's higher IDSW is the
  price of *actually maintaining* tracks across thousands of textureless frames.
- BoT-SORT keeps **false positives near zero (~4 FP across 21k frames)** while
  maximizing TP.
- The stubbornly high **False Negative** count across *all* trackers (~32k even
  for the best) is the honest signature of how hard thermal flickering is — the
  ceiling here is set by detection, which motivates the next layer.

![TP vs FN ratio](images/TP_and_TN.png)
![Per-frame TP timeline](images/per_frame_tp_timeline.png)
*Per-frame TP shows exactly where SORT collapses and where BoT-SORT holds continuity.*

---

## 11. The detection stack — engineering decisions worth showing

`detection_models/` is a small but opinionated MLOps setup. Interesting choices:

- **One joined `uv` virtual environment** shared by all four models
  (YOLO26 + RF-DETR together), with **torch pinned to the CUDA 12.4 wheels** so
  the GPU build is reproducible.
- **Dockerized, persistent MLflow** (SQLite backend + artifact proxy) at
  `localhost:5000`. Two experiments kept separate: **`bambi-detection`** (real
  training) and **`bambi-bench`** (speed tests). Survives container restarts.
- **Zero-copy dataset views via symlinks.** The ~16 GB image set is **never
  copied**. RF-DETR needs the Roboflow layout (`val` renamed to `valid` + a root
  `data.yaml`); we satisfy it with whole-directory symlinks. Speed-test subsets
  use per-file symlinks.
- **Read the data in place** — both detector families point at the same
  `data/yolo_data`, no duplication.
- **Resolution 736, on purpose.** Both Ultralytics and RF-DETR require input
  divisible by 32; the dataset is 720-ish, and `720 % 32 = 16`, so we round **up
  to 736** (nearest valid size) for a fair, identical comparison.
- **OOM is recorded, not fatal.** The benchmark catches CUDA OOM per model and
  logs it as a status instead of crashing the whole run — each model runs in its
  own process so peak VRAM is measured cleanly.
- **`train_all.sh` is a fault-tolerant queue** (no `set -e`): a failure in one
  model is reported but the queue continues.

---

## 12. Detector speed benchmark (RTX 4070, 1 epoch on a 500-image subset @ 736)

| Model | Params | Batch | img/s | Peak VRAM | Est. 1 epoch (17,684) | Est. 50 epochs |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **yolo26-s** | 10.0 M | 24 | **44.95** | 9.8 GB | **6.6 min** | 5.46 h |
| yolo26-l | 26.3 M | 10 | 25.3 | 9.83 GB | 11.6 min | 9.71 h |
| rf-detr-s | 32.1 M | 8 | 15.35 | 12.08 GB | 19.2 min | 16.00 h |
| rf-detr-l | 33.9 M | 6 | 13.48 | 10.02 GB | 21.9 min | 18.22 h |

**Takeaways:**
- **YOLO26-s is ~3.3× faster** than RF-DETR-L at training throughput and uses the
  least memory — the obvious choice for fast iteration.
- RF-DETR's **DINOv2 backbone is memory-hungry** (rf-detr-s peaks at 12 GB even
  at batch 8), which is exactly why batch sizes are tuned per model.
- These figures let us **plan compute up front**: a full 50-epoch YOLO26-s run is
  ~5.5 h vs ~18 h for RF-DETR-L on a single 12 GB card.

> The benchmark **extrapolates** subset throughput to the full epoch, so we get a
> reliable training-cost estimate from a 1-epoch smoke test — no need to burn a
> full run just to size the job.

---

## 13. Data-engineering detail: the annotation converter

`src/annotation_process.py` does the BAMBI MOT → YOLO conversion in a **single
pass** while also collecting tracking metadata. Decisions baked in:

- Maps the three Latin species names → class IDs 0/1/2.
- **Visibility threshold (default 0.3):** drops boxes that are too occluded to
  learn from.
- **Skips propagated boxes** by default (interpolated annotations, not true
  observations).
- Converts absolute MOT boxes → **normalized YOLO** `cx cy w h`.
- Records each track's **first/last frame span**, feeding the lifetime analysis
  later.

This single function is the bridge between the raw tracking annotations and the
detector training labels — the same source data powers both halves of the
project.

![Second annotation example](images/annotation_example_2.png)

---

## 14. What was genuinely interesting / reusable

- **Isolating the tracker from the detector** to get an honest measure of MOT
  quality on thermal data.
- A **9× track-lifetime win (4.0 → 36.3 frames)** from one architectural insight:
  reset state at real sequence boundaries.
- Making deep Re-ID work on **1-channel thermal** via a pseudo-RGB FP16 wrapper.
- A reproducible **multi-detector MLOps harness** — one uv env, Docker MLflow,
  symlink zero-copy views, per-model OOM handling, and self-extrapolating speed
  benchmarks.
- Honest metric interpretation: low ID-switches can mean a *worse* tracker, and a
  high false-negative floor points the finger at detection, not tracking.

---

## 15. Future work

- Train **YOLO26 / YOLOv11**-class detectors specifically on these thermal
  sequences with **heavy contrast augmentation** and a **150+ epoch** schedule to
  map subtle thermal gradients (training infra is already in place — see §11–12).
- Feed the trained detector into BoT-SORT + ReID to close the loop and attack the
  ~32k false-negative floor that ground-truth isolation exposed.
- Push HOTA/IDF1 further with thermal-specific motion models.

---

### Appendix — image index (in [images/](images/))

| File | Used in |
|---|---|
| `annotation_example_1.png`, `annotation_example_2.png` | §2, §13 — GT thermal annotations |
| `tracklifetime.png` / `tracklifetime_PerSequence.png` | §8 — the boundary fix |
| `long_term_tracks.png` | §9 — 29 filtered trajectories |
| `Predicted_Vs_GT.png` | §9 — prediction vs ground truth |
| `tracker_comparison.png` / `performance_per_tracker.png` | §10 — tracker results |
| `TP_and_TN.png` | §10 — TP vs FN |
| `per_frame_tp_timeline.png` | §10 — per-frame stability |

> **Note on detector accuracy numbers:** the speed benchmark (§12) is complete and
> measured. Full-accuracy (mAP) training was still running at the time of writing
> — present §12 as throughput/cost, and add final mAP from MLflow (`bambi-detection`
> experiment) once the 50-epoch runs finish.
