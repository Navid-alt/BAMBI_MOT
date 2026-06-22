"""Online tracking with the trained RF-DETR associator.

For each video the frames are processed in order. Each frame's confident
detections are linked to the previous frame's detections via the learned
soft-assignment (mutual-nearest + dustbin threshold). Matched detections inherit
the track id; unmatched current detections start new tracks; tracks unseen for
``max_age`` frames die. Results are written as MOTChallenge txt per video:

    frame,id,x,y,w,h,score,-1,-1,-1   (frame & id are 1-indexed)

Run:
    cd detection_models
    PYTHONPATH=.. uv run python ../transformer_tracking/infer.py \
        --split val --weights ../transformer_tracking/output/associator_last.pth
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from transformer_tracking.data.clip_dataset import load_video_frames
from transformer_tracking.model.matching import select_detections
from transformer_tracking.model.rfdetr_backbone import DEFAULT_CKPT, FrozenRFDETR
from transformer_tracking.model.track_module import TrackAssociator, box_cxcywh_to_xyxy

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--weights", default=str(HERE / "output" / "associator_last.pth"))
    p.add_argument("--ckpt", default=None, help="RF-DETR checkpoint")
    p.add_argument("--score-thresh", type=float, default=0.3)
    p.add_argument("--match-thresh", type=float, default=0.2,
                   help="min assignment prob to accept a link")
    p.add_argument("--max-dets", type=int, default=100)
    p.add_argument("--max-age", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default=str(HERE / "output" / "tracks"))
    return p.parse_args()


@torch.no_grad()
def track_video(backbone, model, frames, args):
    """Return list of (frame_idx_1based, track_id, x, y, w, h, score) rows."""
    rows = []
    next_id = 1
    # previous-frame state: embeds, boxes (norm cxcywh), track ids
    prev = None
    for t, fr in enumerate(frames, start=1):
        out = backbone.extract([fr.load_image()])[0]
        idx = select_detections(out["scores"], args.score_thresh, args.max_dets)
        embeds, boxes, scores = out["embeds"][idx], out["boxes"][idx], out["scores"][idx]
        n = embeds.shape[0]
        cur_ids = [-1] * n

        if prev is not None and prev["embeds"].shape[0] and n:
            log_assign = model(prev["embeds"], prev["boxes"], embeds, boxes)
            assign = log_assign.exp()
            np_ = prev["embeds"].shape[0]
            scores_pc = assign[:np_, :n]            # prev x cur
            # mutual nearest neighbour with threshold
            cur_best = scores_pc.argmax(dim=0)       # [n] best prev for each cur
            prev_best = scores_pc.argmax(dim=1)      # [np] best cur for each prev
            for j in range(n):
                i = int(cur_best[j])
                if int(prev_best[i]) == j and float(scores_pc[i, j]) >= args.match_thresh:
                    cur_ids[j] = prev["ids"][i]

        for j in range(n):
            if cur_ids[j] == -1:
                cur_ids[j] = next_id
                next_id += 1

        # write rows (convert norm cxcywh -> abs xywh)
        w_img, h_img = fr.orig_size
        xyxy = box_cxcywh_to_xyxy(boxes)
        for j in range(n):
            x1, y1, x2, y2 = xyxy[j].tolist()
            rows.append((t, cur_ids[j], x1 * w_img, y1 * h_img,
                         (x2 - x1) * w_img, (y2 - y1) * h_img, float(scores[j])))

        prev = {"embeds": embeds, "boxes": boxes, "ids": cur_ids}
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone = FrozenRFDETR(ckpt=args.ckpt or DEFAULT_CKPT, device=args.device)
    model = TrackAssociator(in_dim=backbone.embed_dim).to(args.device)
    state = torch.load(args.weights, map_location=args.device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[infer] loaded associator from {args.weights} (epoch {state.get('epoch')})")

    for name, frames in load_video_frames(args.split):
        rows = track_video(backbone, model, frames, args)
        path = out_dir / f"{name}.txt"
        with open(path, "w") as f:
            for (fr, tid, x, y, w, h, s) in rows:
                f.write(f"{fr},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{s:.4f},-1,-1,-1\n")
        print(f"[infer] {name}: {len(frames)} frames, {len(rows)} dets -> {path.name}")
    print(f"[infer] done -> {out_dir}")


if __name__ == "__main__":
    main()
