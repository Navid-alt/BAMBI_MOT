"""Precompute frozen RF-DETR detections once and cache them to disk.

RF-DETR is frozen, so its per-frame query embeddings/boxes never change across
training epochs — recomputing them every epoch is what starves the GPU. This
runs the detector over every frame a single time (batched) and writes the kept
detections per image to ``<cache-dir>/<split>/<image_id>.pt``. Training then
reads tensors from disk and only runs the tiny matcher.

Stored per image (fp16 to keep the cache small): embeds [K,D], boxes [K,4]
(norm cxcywh), scores [K], labels [K], where K = top detections with
score >= --min-score, capped at --max-keep.

Run:
    cd detection_models
    PYTHONPATH=.. uv run python ../transformer_tracking/precompute_embeddings.py --split train
    PYTHONPATH=.. uv run python ../transformer_tracking/precompute_embeddings.py --split val
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from transformer_tracking.data.clip_dataset import load_video_frames
from transformer_tracking.model.matching import select_detections
from transformer_tracking.model.rfdetr_backbone import DEFAULT_CKPT, FrozenRFDETR

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--ckpt", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8, help="frames per RF-DETR forward")
    p.add_argument("--min-score", type=float, default=0.05,
                   help="keep detections above this (train --score-thresh can rise above it)")
    p.add_argument("--max-keep", type=int, default=100)
    p.add_argument("--cache-dir", default=str(HERE / "output" / "cache"))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.cache_dir) / args.split
    out.mkdir(parents=True, exist_ok=True)
    backbone = FrozenRFDETR(ckpt=args.ckpt or DEFAULT_CKPT, device=args.device)

    def flush(frames):
        res = backbone.extract([f.load_image() for f in frames])
        for fr, r in zip(frames, res):
            idx = select_detections(r["scores"], args.min_score, args.max_keep)
            torch.save({
                "embeds": r["embeds"][idx].half().cpu(),
                "boxes": r["boxes"][idx].half().cpu(),
                "scores": r["scores"][idx].half().cpu(),
                "labels": r["labels"][idx].to(torch.int16).cpu(),
            }, out / f"{fr.image_id}.pt")

    batch, done, skipped = [], 0, 0
    for _, frames in load_video_frames(args.split):
        for fr in frames:
            if not args.overwrite and (out / f"{fr.image_id}.pt").exists():
                skipped += 1
                continue
            batch.append(fr)
            if len(batch) >= args.batch_size:
                flush(batch)
                done += len(batch)
                batch = []
                if done % 800 == 0:
                    print(f"  cached {done} frames...")
    if batch:
        flush(batch)
        done += len(batch)

    print(f"[precompute] {args.split}: cached {done}, skipped {skipped} -> {out}")


if __name__ == "__main__":
    main()
