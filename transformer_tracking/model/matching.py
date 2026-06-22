"""Detection selection and detection<->GT / GT-pair target construction.

RF-DETR emits 300 queries per frame; we keep the confident ones, assign each to
a ground-truth object (by IoU) to recover its ``track_id``, and from two frames'
assigned tracks build the correspondence targets the associator is trained on.
"""
from __future__ import annotations

import torch
from torchvision.ops import box_iou

from .track_module import box_cxcywh_to_xyxy


def select_detections(scores: torch.Tensor, score_thresh: float = 0.3,
                      max_dets: int = 100) -> torch.Tensor:
    """Indices of kept queries: score >= thresh, capped at top ``max_dets``."""
    keep = (scores >= score_thresh).nonzero(as_tuple=False).flatten()
    if keep.numel() > max_dets:
        top = scores[keep].topk(max_dets).indices
        keep = keep[top]
    return keep


def gt_boxes_xyxy_norm(boxes_xywh: torch.Tensor, orig_size: tuple[int, int]) -> torch.Tensor:
    w, h = orig_size
    if boxes_xywh.numel() == 0:
        return boxes_xywh.new_zeros((0, 4))
    x, y, bw, bh = boxes_xywh.unbind(-1)
    scale = boxes_xywh.new_tensor([w, h, w, h])
    return torch.stack([x, y, x + bw, y + bh], dim=-1) / scale


def assign_tracks(det_boxes_cxcywh: torch.Tensor, gt_boxes_xywh: torch.Tensor,
                  gt_track_ids: torch.Tensor, orig_size: tuple[int, int],
                  iou_thresh: float = 0.5) -> torch.Tensor:
    """Greedy IoU assignment of detections to GT; returns track_id per det (-1 if none)."""
    n = det_boxes_cxcywh.shape[0]
    out = det_boxes_cxcywh.new_full((n,), -1, dtype=torch.int64)
    if n == 0 or gt_boxes_xywh.numel() == 0:
        return out
    det_xyxy = box_cxcywh_to_xyxy(det_boxes_cxcywh)
    gt_xyxy = gt_boxes_xyxy_norm(gt_boxes_xywh, orig_size).to(det_xyxy)
    iou = box_iou(det_xyxy, gt_xyxy)  # [n, g]
    gt_taken = torch.zeros(gt_xyxy.shape[0], dtype=torch.bool)
    # greedy: highest IoU pairs first
    flat = iou.flatten()
    order = flat.argsort(descending=True)
    g = gt_xyxy.shape[0]
    det_taken = torch.zeros(n, dtype=torch.bool)
    for idx in order.tolist():
        if flat[idx] < iou_thresh:
            break
        di, gi = divmod(idx, g)
        if det_taken[di] or gt_taken[gi]:
            continue
        out[di] = gt_track_ids[gi]
        det_taken[di] = True
        gt_taken[gi] = True
    return out


def build_pair_targets(tids_a: torch.Tensor, tids_b: torch.Tensor):
    """From per-detection track_ids of two frames build matches + dustbin sets.

    Returns (matches [K,2], unmatched_a [.], unmatched_b [.]) as index tensors.
    Detections with track_id == -1 (background) always go to the dustbin.
    """
    na, nb = tids_a.shape[0], tids_b.shape[0]
    matches = []
    matched_a, matched_b = set(), set()
    # map track_id -> index in b (valid tracks only)
    b_lookup = {int(t): j for j, t in enumerate(tids_b.tolist()) if t >= 0}
    for i, t in enumerate(tids_a.tolist()):
        if t >= 0 and t in b_lookup:
            j = b_lookup[t]
            matches.append((i, j))
            matched_a.add(i)
            matched_b.add(j)
    dev = tids_a.device
    matches_t = torch.tensor(matches, dtype=torch.int64, device=dev).reshape(-1, 2)
    unmatched_a = torch.tensor([i for i in range(na) if i not in matched_a],
                               dtype=torch.int64, device=dev)
    unmatched_b = torch.tensor([j for j in range(nb) if j not in matched_b],
                               dtype=torch.int64, device=dev)
    return matches_t, unmatched_a, unmatched_b
