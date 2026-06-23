# transformer_tracking — RF-DETR end-to-end tracker (MOTRv2-style)

A learned, end-to-end multi-object tracker built on top of the project's
**already-trained RF-DETR-L detector**. Unlike the tracking-by-detection
baselines in `README(Track).md` (SORT / ByteTrack / BoT-SORT, hand-coded
association), here the cross-frame association is **learned** while the detector
stays **frozen** — the MOTRv2 idea (strong frozen detector + learned association),
adapted so it trains on a single RTX 4070.

```
frame t-1 ─► [frozen RF-DETR] ─► 300 query embeds + boxes ─┐
                                                           ├─► [TrackAssociator] ─► matched IDs
frame t   ─► [frozen RF-DETR] ─► 300 query embeds + boxes ─┘     (+ births/deaths)
```

Only `TrackAssociator` (a small SuperGlue-style attentional matcher with a
Sinkhorn dustbin) is trained. RF-DETR is loaded from
`detection_models/rf-detr-l/output/checkpoint_best_ema.pth` and frozen, so
training uses < 1 GB VRAM.

> **Trained weights** (the associator and the frozen RF-DETR-l it needs) are on
> the Hugging Face Hub: [`NavidGh/BambiMot`](https://huggingface.co/NavidGh/BambiMot)
> — see `transformer_tracking/output/associator_last.pth` and
> `rf-detr-l/output/checkpoint_best_ema.pth`.

## Why it fits the project

- **Reuses your detector & data in place** — same `checkpoint_best_ema.pth`,
  the same `annotations/instances_*.json`, and the frames under
  `data/yolo_data/*/images` (resolved by basename; 100 % coverage verified).
- **Same env** — runs inside the `detection_models` uv environment and logs to
  the same Dockerised MLflow (`bambi-tracking` experiment).
- **Same metrics** — `eval.py` emits MOTA / IDF1 / IDSW so this becomes one more
  row in the comparison table, representing the *learned-association* paradigm.

## Layout

| File | Role |
|------|------|
| `probe_rfdetr_embeddings.py` | one-off check that RF-DETR exposes per-query embeddings |
| `precompute_embeddings.py` | run frozen RF-DETR once, cache detections per frame |
| `data/clip_dataset.py` | clips + per-video iterator + cached-feature dataset |
| `model/rfdetr_backbone.py` | frozen RF-DETR; hook captures `hs[-1]` query embeddings |
| `model/track_module.py` | `TrackAssociator` + Sinkhorn assignment + NLL loss |
| `model/matching.py` | detection selection, detection↔GT IoU assignment, pair targets |
| `train.py` | clip training loop (frozen backbone, MLflow) |
| `infer.py` | online tracking → MOTChallenge txt per video |
| `eval.py` | CLEAR-MOT + IDF1 via motmetrics |

## Usage

```bash
# 0. (optional) start MLflow for logging
cd detection_models && docker compose up -d && cd ..
cd transformer_tracking

# 1. cache frozen RF-DETR detections ONCE (train + val). The detector is
#    frozen, so this is the only time it runs — training reads the cache.
./precompute.sh

# 2. train the associator (no RF-DETR in the loop) — you launch this
./train.sh                      # defaults: 10 epochs, clip-len 2, lr 1e-4, cached
./train.sh --epochs 15 --limit 4000   # shorter epochs while iterating

# 3. track + score a split (needs motmetrics)
uv --directory ../detection_models add motmetrics   # one-time
./track_eval.sh val
```

### Why precompute

RF-DETR is frozen, so its per-frame embeddings are identical every epoch.
`precompute.sh` runs it once and caches the kept detections (~tens of MB/split,
fp16). Training then runs only the tiny matcher from disk — no per-epoch
detector recompute, no 1024² image decoding in the hot loop. This is what fixes
the "GPU barely utilised" symptom: the redundant 95% of the compute is removed
rather than packed onto the GPU. (`infer.py`/`track_eval.sh` still run RF-DETR
live, since tracking visits each frame once.)

All commands run through the `detection_models` uv env via
`uv --directory ../detection_models run`, with `PYTHONPATH` set to the repo root
so `transformer_tracking` imports resolve.

## Key knobs

- `--clip-len` — frames per clip (≥3 trains on multiple consecutive pairs).
- `--score-thresh` — RF-DETR confidence to keep a query as a detection.
- `--iou-thresh` — detection↔GT IoU for recovering `track_id` targets.
- `--match-thresh` (infer) — min assignment probability to accept a link.
- `--max-age` (infer) — frames a lost track is kept before it dies.

## Report

A full write-up (architecture, training, results, qualitative overlays) is in
[`docs/report.pdf`](docs/report.pdf) / [`docs/report.md`](docs/report.md).
Regenerate it with:

```bash
PYTHONPATH=.. uv --directory ../detection_models run python docs/make_report_assets.py
PYTHONPATH=.. uv --directory ../detection_models run python docs/make_pdf.py
```

## Notes / next steps

- v1 associates each frame against the **previous frame** only. A re-ID memory
  buffer (keep lost tracks for `max_age` frames and match against them) is the
  natural next improvement and the hook is already in `infer.py`.
- Detector is frozen by design; if detection recall on thermal animals limits
  tracking, fine-tuning RF-DETR is a separate, heavier step.
