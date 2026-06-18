"""Train + validate YOLO26-L on the BAMBI wildlife detection set.

Same as yolo26-s but the Large variant. Likely needs the 12 GB GPU — the 4 GB
laptop card will probably OOM at imgsz=640. Logs to MLflow at localhost:5000.

    uv run python yolo26-l/train.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data.yaml"))
    ap.add_argument("--weights", default="yolo26l.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=2)       # raise on the 12 GB card
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="yolo26-l")
    args = ap.parse_args()

    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "bambi-detection")
    os.environ.setdefault("MLFLOW_RUN", f"train_{args.name}")

    from ultralytics import YOLO, settings

    settings.update({"mlflow": True})

    model = YOLO(args.weights)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=str(HERE / "runs"), name=f"train_{args.name}",
        exist_ok=True,
    )
    metrics = model.val(
        data=args.data, imgsz=args.imgsz, batch=args.batch, device=args.device,
        project=str(HERE / "runs"), name=f"val_{args.name}", exist_ok=True,
    )
    print(f"mAP50-95: {metrics.box.map:.4f}  mAP50: {metrics.box.map50:.4f}")


if __name__ == "__main__":
    main()
