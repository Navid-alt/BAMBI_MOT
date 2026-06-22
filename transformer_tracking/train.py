"""Train the RF-DETR track associator (frozen detector, MOTRv2-style).

The RF-DETR backbone is frozen; only the small attentional matcher trains, so a
single 4070 is plenty. Each clip contributes consecutive frame pairs; for every
pair we extract detections, assign them to GT to recover track_ids, and train
the soft-assignment to reproduce the GT correspondences.

Run (you launch this yourself):
    cd detection_models
    PYTHONPATH=.. uv run python ../transformer_tracking/train.py --epochs 20
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transformer_tracking.data.clip_dataset import (
    CachedClipDataset, ClipDataset, clip_collate,
)
from transformer_tracking.model.matching import (
    assign_tracks, build_pair_targets, select_detections,
)
from transformer_tracking.model.rfdetr_backbone import DEFAULT_CKPT, FrozenRFDETR
from transformer_tracking.model.track_module import TrackAssociator, assignment_nll_loss

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--clip-len", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--score-thresh", type=float, default=0.3)
    p.add_argument("--iou-thresh", type=float, default=0.5)
    p.add_argument("--max-dets", type=int, default=100)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cache-dir", default=None,
                   help="use precomputed embeddings from this dir (skips RF-DETR); "
                        "e.g. transformer_tracking/output/cache")
    p.add_argument("--ckpt", default=None, help="RF-DETR checkpoint (default: rf-detr-l best_ema)")
    p.add_argument("--output-dir", default=str(HERE / "output"))
    p.add_argument("--limit", type=int, default=0, help="cap clips/epoch (0 = all)")
    p.add_argument("--mlflow", action="store_true")
    p.add_argument("--run-name", default="rfdetr-track")
    return p.parse_args()


def _fmt(seconds: float) -> str:
    """Human-readable h/m/s for ETA printouts."""
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def frame_detections(frame, backbone, device, score_thresh, iou_thresh, max_dets):
    """Return (embeds, boxes, track_ids) for a frame, from cache or live backbone.

    Works for both a live FrameSample (backbone is not None) and a cached
    FrameFeat (backbone is None, detections already on the frame object).
    """
    if backbone is not None:  # live FrameSample
        out = backbone.extract([frame.load_image()])[0]
        scores, embeds_all, boxes_all = out["scores"], out["embeds"], out["boxes"]
    else:                     # cached FrameFeat
        scores, embeds_all, boxes_all = frame.scores, frame.embeds, frame.boxes
        gt_boxes, gt_tids, orig = frame.gt_boxes, frame.gt_track_ids, frame.orig_size
    if backbone is not None:
        gt_boxes, gt_tids, orig = frame.boxes, frame.track_ids, frame.orig_size

    idx = select_detections(scores.to(device), score_thresh, max_dets)
    embeds = embeds_all.to(device)[idx]
    boxes = boxes_all.to(device)[idx]
    tids = assign_tracks(boxes, gt_boxes.to(device), gt_tids.to(device), orig, iou_thresh)
    return embeds, boxes, tids


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cache_dir:
        backbone = None
        embed_dim = 256
        ds = CachedClipDataset("train", args.cache_dir, clip_len=args.clip_len)
        num_workers = 8
        print(f"[train] using cached embeddings from {args.cache_dir} (RF-DETR not loaded)")
    else:
        backbone = FrozenRFDETR(ckpt=args.ckpt or DEFAULT_CKPT, device=args.device)
        embed_dim = backbone.embed_dim
        ds = ClipDataset("train", clip_len=args.clip_len)
        num_workers = 4

    model = TrackAssociator(in_dim=embed_dim, layers=args.layers).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=num_workers,
                        collate_fn=clip_collate)

    run = _maybe_mlflow(args)
    n_clips = args.limit or len(ds)
    total_iters = args.epochs * n_clips
    t_start = time.time()
    print(f"[train] {n_clips} clips/epoch x {args.epochs} epochs, lr={args.lr}, "
          f"frozen RF-DETR, device={args.device}")

    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        running, steps = 0.0, 0
        for it, batch in enumerate(loader):
            if args.limit and it >= args.limit:
                break
            clip = batch[0]
            clip_loss = torch.zeros((), device=args.device)
            n_pairs = 0
            for k in range(len(clip) - 1):
                ea, ba, ta = frame_detections(clip[k], backbone, args.device,
                                              args.score_thresh, args.iou_thresh, args.max_dets)
                eb, bb, tb = frame_detections(clip[k + 1], backbone, args.device,
                                              args.score_thresh, args.iou_thresh, args.max_dets)
                if ea.shape[0] == 0 or eb.shape[0] == 0:
                    continue
                log_assign = model(ea, ba, eb, bb)
                m, ua, ub = build_pair_targets(ta, tb)
                clip_loss = clip_loss + assignment_nll_loss(log_assign, m, ua, ub)
                n_pairs += 1
            if n_pairs == 0:
                continue
            loss = clip_loss / n_pairs / args.grad_accum
            loss.backward()
            if (it + 1) % args.grad_accum == 0:
                opt.step()
                opt.zero_grad()
            running += float(clip_loss / n_pairs)
            steps += 1
            if steps % 200 == 0:
                done_iters = epoch * n_clips + (it + 1)
                elapsed = time.time() - t_start
                eta = elapsed / done_iters * (total_iters - done_iters)
                print(f"  epoch {epoch} it {steps}/{n_clips} loss {running / steps:.4f} "
                      f"| elapsed {_fmt(elapsed)} ETA {_fmt(eta)}")

        avg = running / max(steps, 1)
        done_iters = (epoch + 1) * n_clips
        elapsed = time.time() - t_start
        eta = elapsed / done_iters * (total_iters - done_iters)
        print(f"[epoch {epoch}] mean loss {avg:.4f} | elapsed {_fmt(elapsed)} "
              f"ETA {_fmt(eta)} ({epoch + 1}/{args.epochs} epochs)")
        ckpt = out_dir / f"associator_epoch{epoch}.pth"
        torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, ckpt)
        torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch},
                   out_dir / "associator_last.pth")
        if run is not None:
            import mlflow
            mlflow.log_metric("train/loss", avg, step=epoch)

    if run is not None:
        import mlflow
        mlflow.end_run()
    print("[train] done.")


def _maybe_mlflow(args):
    if not args.mlflow:
        return None
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
    import mlflow
    mlflow.set_experiment("bambi-tracking")
    run = mlflow.start_run(run_name=args.run_name)
    mlflow.log_params({k: v for k, v in vars(args).items()})
    return run


if __name__ == "__main__":
    main()
