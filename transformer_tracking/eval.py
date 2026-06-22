"""Evaluate tracker output against COCO GT with CLEAR-MOT + IDF1 metrics.

Reads the MOTChallenge txt files written by infer.py and the ground-truth boxes
from the COCO annotations, then reports MOTA / IDF1 / IDSW / etc. per video and
overall — the same metric family reported in README(Track).md for the BoxMOT
baselines, so this row drops straight into that comparison table.

Requires motmetrics:  uv add motmetrics   (or pip install motmetrics)

Run:
    cd detection_models
    PYTHONPATH=.. uv run python ../transformer_tracking/eval.py --split val
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_iou

from transformer_tracking.data.clip_dataset import load_video_frames

HERE = Path(__file__).resolve().parent


def _iou_distance(gt_xywh: list, pred_xywh: list, max_iou: float) -> np.ndarray:
    """IoU-based distance for motmetrics (NumPy 2-safe). Returns [G, P] costs."""
    if not gt_xywh or not pred_xywh:
        return np.empty((len(gt_xywh), len(pred_xywh)))
    # xywh -> xyxy
    def to_xyxy(boxes):
        t = torch.tensor(boxes, dtype=torch.float32)
        x, y, w, h = t.unbind(-1)
        return torch.stack([x, y, x + w, y + h], dim=-1)
    iou = box_iou(to_xyxy(gt_xywh), to_xyxy(pred_xywh))  # [G, P]
    dist = (1 - iou).numpy()
    dist[dist > max_iou] = np.nan  # motmetrics convention: unmatched if > max_iou
    return dist


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--tracks-dir", default=str(HERE / "output" / "tracks"))
    p.add_argument("--iou-thresh", type=float, default=0.5)
    return p.parse_args()


def read_pred(path: Path) -> dict[int, tuple[list[int], list[list[float]]]]:
    """frame(1-based) -> (ids, [x,y,w,h])."""
    out: dict[int, tuple[list, list]] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        fr, tid, x, y, w, h, *_ = line.split(",")
        fr, tid = int(fr), int(tid)
        out.setdefault(fr, ([], []))
        out[fr][0].append(tid)
        out[fr][1].append([float(x), float(y), float(w), float(h)])
    return out


def main() -> None:
    args = parse_args()
    try:
        import motmetrics as mm
    except ImportError:
        raise SystemExit("motmetrics not installed. Run: uv add motmetrics")

    tracks_dir = Path(args.tracks_dir) / args.split
    accs, names = [], []

    for name, frames in load_video_frames(args.split):
        acc = mm.MOTAccumulator(auto_id=False)
        preds = read_pred(tracks_dir / f"{name}.txt")
        for t, fr in enumerate(frames, start=1):
            gt_ids = fr.track_ids.tolist()
            gt_boxes = fr.boxes.tolist()  # xywh abs
            p_ids, p_boxes = preds.get(t, ([], []))
            dist = _iou_distance(gt_boxes, p_boxes, 1 - args.iou_thresh)
            acc.update(gt_ids, p_ids, dist, frameid=t)
        accs.append(acc)
        names.append(name)

    mh = mm.metrics.create()
    metrics = ["mota", "idf1", "num_switches", "num_false_positives",
               "num_misses", "num_objects", "mostly_tracked", "mostly_lost",
               "num_fragmentations", "precision", "recall"]
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    print(mm.io.render_summary(
        summary,
        namemap={"mota": "MOTA", "idf1": "IDF1", "num_switches": "IDSW",
                 "num_false_positives": "FP", "num_misses": "FN",
                 "mostly_tracked": "MT", "mostly_lost": "ML",
                 "num_fragmentations": "Frag", "precision": "Prec", "recall": "Rec"},
        formatters=mh.formatters,
    ))


if __name__ == "__main__":
    main()
