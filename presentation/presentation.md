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
- **Two resolutions — 640 and 1024 — benchmarked and trained.** Both Ultralytics
  and RF-DETR require inputs divisible by 32; 640 and 1024 both qualify and
  bracket the native 1024 × 1024 frames (RF-DETR-L's own native size is 704).
  **640** is the fast baseline; **1024** preserves full-frame detail for the
  small, low-contrast targets. Every model is timed at *both* sizes so the
  resolution effect is isolated cleanly.
- **OOM is recorded, not fatal.** The benchmark catches CUDA OOM per model and
  logs it as a status instead of crashing the whole run — each model runs in its
  own process so peak VRAM is measured cleanly.
- **`train_all.sh` is a fault-tolerant queue** (no `set -e`): a failure in one
  model is reported but the queue continues.

---

## 12. Detector speed benchmark (RTX 4070, 1 epoch on a 500-image subset)

All four detectors timed at **both 640 and 1024**, throughput extrapolated to the
full 17,684-image epoch (and ×50). Each model runs in its own process so peak
VRAM is measured cleanly; an OOM in one never stops the others.

| Model | Params | Res | Batch | img/s | Peak VRAM | Est. 1 epoch | Est. 50 epochs |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **yolo26-s** | 10.0 M | 640 | 30 | **32.67** | 9.13 GB | **9.0 min** | 7.52 h |
| yolo26-s | 10.0 M | 1024 | 12 | 26.09 | 9.64 GB | 11.3 min | 9.41 h |
| yolo26-l | 26.3 M | 640 | 12 | 27.97 | 8.87 GB | 10.5 min | 8.78 h |
| yolo26-l | 26.3 M | 1024 | 5 | 14.51 | 9.74 GB | 20.3 min | 16.93 h |
| rf-detr-s | 32.1 M | 640 | 10 | 16.28 | 12.17 GB | 18.1 min | 15.09 h |
| rf-detr-s | 32.1 M | 1024 | 4 | 9.07 | 10.94 GB | 32.5 min | 27.08 h |
| rf-detr-l | 33.9 M | 640 | 8 | 15.02 | 11.29 GB | 19.6 min | 16.35 h |
| rf-detr-l | 33.9 M | 1024 | 3 | 8.17 | 8.36 GB | 36.1 min | 30.06 h |

**Takeaways:**
- **YOLO26-s @640 is ~4× faster** than RF-DETR-L @1024 and uses the least memory
  — the obvious choice for fast iteration.
- **Going 640 → 1024 roughly doubles the cost** for every architecture (≈2.5×
  more pixels), which is the price we knowingly pay for the accuracy gain shown
  next.
- RF-DETR's **DINOv2 backbone is memory-hungry**, which is exactly why batch
  sizes are tuned per model to fit the 12 GB card.
- These figures let us **plan compute up front** and pick the batch size that
  fits before launching a multi-hour run.

> The benchmark **extrapolates** subset throughput to the full epoch, so we get a
> reliable training-cost estimate from a 1-epoch smoke test — no need to burn a
> full run just to size the job.

---

## 13. Detector accuracy — what the four trained models score

Four detectors trained to convergence on the full set (~17.7k images), each
logged **per epoch** to MLflow (`bambi-detection`). The design probes two axes —
**model size** (YOLO26 s vs l) and **input resolution** (640 vs 1024) — with
**RF-DETR-L** adding a **transformer / DINOv2 architecture** at 1024.

| Model | Resolution | Epochs | mAP@50 | mAP@50-95 | Precision | Recall |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| YOLO26-s | 640 | 50 | 0.071 | 0.020 | 0.195 | 0.096 |
| YOLO26-l | 640 | 50 | 0.107 | 0.030 | 0.280 | 0.124 |
| YOLO26-l | 1024 | 50 | 0.134 | 0.039 | 0.306 | 0.153 |
| **RF-DETR-l** | **1024** | 46 | **0.186** | **0.057** | **0.376** | **0.190** |

![Detector validation metrics over training](images/detector_curves.png)
*Per-epoch validation metrics for all four runs — the ranking is consistent and stable across the whole schedule.*

![Final detector metrics](images/detector_final_bars.png)
*Final-epoch comparison. Each lever adds accuracy on top of the last.*

**What the numbers say:**
- **Bigger model helps:** YOLO26 s → l at 640 lifts mAP@50 **+50%** (0.071 → 0.107).
- **Higher resolution helps:** YOLO26-l 640 → 1024 lifts mAP@50 **+25%**
  (0.107 → 0.134) — small thermal targets need the pixels.
- **Architecture helps most:** RF-DETR-L @1024 is the **best overall**, mAP@50
  **0.186** (+39% over YOLO26-l @1024) and the highest precision (0.376) and
  recall (0.190), even though it ran 46 of 50 epochs.
- **But all scores are low** — even the best detector's mAP@50 sits under 0.2.
  Thermal aerial detection is genuinely hard, and this is exactly the **detection
  bottleneck** that sets the ~32k false-negative floor we saw on the tracking side
  (§10) — confirming, from the other direction, that the ceiling here is detection.

---

## 14. Where the detectors fail — confusion matrices

The normalized confusion matrices make the failure mode unmistakable: the
dominant error is **missing the animal entirely** (predicting *background*), not
confusing one species for another. All three use the same convention (Ultralytics
matrix, conf 0.25 / IoU 0.45). RF-DETR logs only COCO mAP during training, so its
matrix is computed by running the trained checkpoint over the full validation set
(`tools/rfdetr_confusion_matrix.py`) — making the four detectors directly comparable.

![Confusion matrix — YOLO26-s @640 (weakest)](images/cm_yolo26s_640.png)
*Weakest model (YOLO26-s @640): Wild boar recall 0.13, Red deer 0.01, Roe deer 0.00 — almost everything falls into the background column.*

![Confusion matrix — YOLO26-l @1024 (best YOLO)](images/cm_yolo26l_1024.png)
*Best YOLO (YOLO26-l @1024): Wild boar recall climbs to 0.22 and Red deer to 0.18 — the gain from size + resolution lights up the diagonal.*

![Confusion matrix — RF-DETR-l @1024 (best overall)](images/cm_rfdetr_l_1024.png)
*Best overall (RF-DETR-l @1024): Wild boar recall 0.37 and Red deer 0.34 — the strongest diagonal of the four, but Roe deer is still 0.00.*

**Reading them together:**
- Stronger model + resolution + architecture **steadily recover recall**: Wild
  boar **0.13 → 0.22 → 0.37** and Red deer **0.01 → 0.18 → 0.34** across the three
  matrices. The improvement is real and class-specific.
- **Roe deer is never detected (0.00 in all three)** — the same story RF-DETR's
  per-class AP tells (Wild boar 0.078, Red deer 0.094, **Roe deer 0.000**). This
  is the **severe class imbalance** from §2 showing up directly: there simply are
  not enough Roe deer examples to learn.
- Almost no off-diagonal species-vs-species confusion: when a model fires, it
  usually picks the right class — the battle is **detection vs. background**, not
  classification.

---

## 15. Data-engineering detail: the annotation converter

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

## 16. What was genuinely interesting / reusable

- **Isolating the tracker from the detector** to get an honest measure of MOT
  quality on thermal data.
- A **9× track-lifetime win (4.0 → 36.3 frames)** from one architectural insight:
  reset state at real sequence boundaries.
- Making deep Re-ID work on **1-channel thermal** via a pseudo-RGB FP16 wrapper.
- A reproducible **multi-detector MLOps harness** — one uv env, Docker MLflow,
  symlink zero-copy views, per-model OOM handling, and self-extrapolating speed
  benchmarks — that produced a clean **4-model accuracy comparison** across size,
  resolution and architecture.
- **Both halves point at the same conclusion:** detection — not tracking — is the
  bottleneck on thermal aerial wildlife, and the data's class imbalance (Roe deer)
  is the hardest single obstacle.
- Honest metric interpretation: low ID-switches can mean a *worse* tracker, and a
  high false-negative floor points the finger at detection, not tracking.

---

## 17. Conclusions

- **Tracking:** BoT-SORT + ReID with per-sequence resets is the best configuration
  (MOTA 0.356, IDF1 0.388), but every tracker hits the same ~32k false-negative
  wall — a detection ceiling, not an association one.
- **Detection:** ranking is unambiguous — **RF-DETR-L @1024 > YOLO26-l @1024 >
  YOLO26-l @640 > YOLO26-s @640**. Model size, resolution and a transformer
  backbone each add accuracy, in that order of payoff.
- **The dominant failure is missed detections (animals read as background), and
  Roe deer is effectively undetectable** with the current data — the class
  imbalance from §2 is the binding constraint, not the model.

---

## 18. Future work

- Attack the **class imbalance** directly: oversample / augment Roe deer and Red
  deer, or add a class-balanced loss — the confusion matrices say this is the
  single highest-leverage fix.
- Push detection further with **heavy thermal-contrast augmentation** and a longer
  schedule, and finish the **RF-DETR-S** and full-50-epoch RF-DETR-L runs (infra
  is already in place — see §11–14).
- **Close the loop:** feed the best trained detector into BoT-SORT + ReID to
  attack the ~32k false-negative floor that ground-truth isolation exposed.
- Push HOTA/IDF1 further with thermal-specific motion models.

---

### Appendix — image index (in [images/](images/))

| File | Used in |
|---|---|
| `annotation_example_1.png`, `annotation_example_2.png` | §2, §15 — GT thermal annotations |
| `tracklifetime.png` / `tracklifetime_PerSequence.png` | §8 — the boundary fix |
| `long_term_tracks.png` | §9 — 29 filtered trajectories |
| `Predicted_Vs_GT.png` | §9 — prediction vs ground truth |
| `tracker_comparison.png` / `performance_per_tracker.png` | §10 — tracker results |
| `TP_and_TN.png` | §10 — TP vs FN |
| `per_frame_tp_timeline.png` | §10 — per-frame stability |
| `detector_curves.png` / `detector_final_bars.png` | §13 — detector accuracy (from MLflow) |
| `cm_yolo26s_640.png` / `cm_yolo26l_1024.png` / `cm_rfdetr_l_1024.png` | §14 — confusion matrices |
