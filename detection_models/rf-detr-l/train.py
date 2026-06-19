"""Train RF-DETR-L on the BAMBI wildlife detection set.

Largest detector here — needs the 12 GB GPU; the 4 GB laptop card will OOM.
Build the RF-DETR view once (shared with rf-detr-s), then train:

    uv run python tools/make_rfdetr_view.py
    uv run python rf-detr-l/train.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default=str(ROOT / "_rfdetr_view"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)   # effective batch 16
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--name", default="rf-detr-l")
    # Named --imgsz to match the YOLO scripts; maps to RF-DETR's resolution arg.
    ap.add_argument("--imgsz", type=int, default=None,
                    help="square input size (multiple of 32); default: the "
                         "model's native resolution (RFDETRLarge = 704)")
    args = ap.parse_args()

    if args.imgsz is not None and args.imgsz % 32:
        raise SystemExit(
            "--imgsz must be a multiple of 32 for RFDETRSmall/Large "
            f"(patch_size 16 * num_windows 2); got {args.imgsz}."
        )

    ds = Path(args.dataset_dir)
    if not (ds / "data.yaml").exists():
        raise SystemExit(
            f"RF-DETR view not found at {ds}.\n"
            "Build it first:  uv run python tools/make_rfdetr_view.py"
        )

    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")

    from rfdetr import RFDETRLarge

    model = RFDETRLarge()
    resolution = args.imgsz if args.imgsz is not None else 704  # RFDETRLarge default
    run_name = f"train_{args.name}_{resolution}"
    train_kwargs = dict(
        dataset_dir=str(ds), epochs=args.epochs, batch_size=args.batch,
        grad_accum_steps=args.grad_accum, lr=args.lr, device=args.device,
        output_dir=str(HERE / "output"),
        mlflow=True, tensorboard=False,
        project="bambi-detection", run=run_name,
    )
    if args.imgsz is not None:
        train_kwargs["resolution"] = args.imgsz
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
