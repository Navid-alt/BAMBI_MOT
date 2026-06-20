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

    log_yolo_metric_aliases(run_name)


def log_yolo_metric_aliases(run_name: str) -> None:
    """Additively copy the 4 core (non-EMA) val metrics into the just-finished
    MLflow run under YOLO names on an epoch axis, so they overlay the YOLO runs.
    Native RF-DETR metrics are untouched; failures here never fail training."""
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        from log_yolo_metric_aliases import find_latest_run_id, relog_yolo_aliases

        uri = os.environ.get("MLFLOW_TRACKING_URI")
        run_id = find_latest_run_id("bambi-detection", run_name, tracking_uri=uri)
        if run_id is None:
            print(f"[yolo-aliases] no MLflow run named {run_name!r}; skipped.")
            return
        written = relog_yolo_aliases(run_id, tracking_uri=uri)
        print(f"[yolo-aliases] run {run_id}: "
              + ", ".join(f"{k} +{v}" for k, v in written.items()))
    except Exception as exc:  # logging must never break the training queue
        print(f"[yolo-aliases] skipped ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
