"""Build a confusion matrix for the trained RF-DETR-L on the BAMBI val split.

RF-DETR's training only logs COCO mAP/AP/AR — no confusion matrix. This script
runs the trained checkpoint over the validation images, matches predictions to
the YOLO ground-truth labels with Ultralytics' own ``ConfusionMatrix`` (same
conf=0.25 / IoU=0.45 convention as the YOLO26 matrices), and saves a normalized
matrix in the identical style so the four detectors are directly comparable.

    uv run python tools/rfdetr_confusion_matrix.py [--limit N]

Output: presentation/images/cm_rfdetr_l_1024.png
"""
from __future__ import annotations

import argparse
import shutil
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image
from ultralytics.utils.metrics import ConfusionMatrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRES_IMG = ROOT.parents[0] / "presentation" / "images"
NAMES = {0: "Wild boar", 1: "Red deer", 2: "Roe deer"}
CKPT = ROOT / "rf-detr-l" / "output" / "checkpoint_best_regular.pth"
VAL_IMAGES = ROOT / "_rfdetr_view" / "valid" / "images"


def load_gt(label_path: Path, w: int, h: int):
    """Read a YOLO label file -> (cls[M], xyxy[M,4]) in pixel coords."""
    cls, boxes = [], []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
            cls.append(c)
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h])
    return (np.array(cls, dtype=np.int64),
            np.array(boxes, dtype=np.float32).reshape(-1, 4))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="cap #images (0 = all)")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="prediction confidence floor (CM re-filters at 0.25)")
    args = ap.parse_args()

    from rfdetr import RFDETRLarge
    model = RFDETRLarge(pretrain_weights=str(CKPT), resolution=1024)

    cm = ConfusionMatrix(names=NAMES, task="detect")
    imgs = sorted(VAL_IMAGES.glob("*.png"))
    if args.limit:
        imgs = imgs[:args.limit]

    for i, ip in enumerate(imgs, 1):
        img = Image.open(ip).convert("RGB")
        w, h = img.size
        gt_cls, gt_xyxy = load_gt(VAL_IMAGES.parent / "labels" / f"{ip.stem}.txt", w, h)
        det = model.predict(img, threshold=args.threshold)
        detections = {
            "cls": torch.as_tensor(det.class_id, dtype=torch.int64)
            if len(det.xyxy) else torch.zeros(0, dtype=torch.int64),
            "conf": torch.as_tensor(det.confidence, dtype=torch.float32)
            if len(det.xyxy) else torch.zeros(0),
            "bboxes": torch.as_tensor(det.xyxy, dtype=torch.float32)
            if len(det.xyxy) else torch.zeros(0, 4),
        }
        batch = {
            "cls": torch.as_tensor(gt_cls, dtype=torch.int64),
            "bboxes": torch.as_tensor(gt_xyxy, dtype=torch.float32),
        }
        cm.process_batch(detections, batch, conf=0.25, iou_thres=0.45)
        if i % 200 == 0:
            print(f"  {i}/{len(imgs)} images")

    print("matrix (rows=Pred, cols=True; last=background):")
    print(cm.matrix.astype(int))

    tmp = ROOT / "_cm_tmp"
    tmp.mkdir(exist_ok=True)
    cm.plot(normalize=True, save_dir=str(tmp))
    out = PRES_IMG / "cm_rfdetr_l_1024.png"
    shutil.copy(tmp / "confusion_matrix_normalized.png", out)
    shutil.rmtree(tmp, ignore_errors=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
