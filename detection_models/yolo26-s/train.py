"""Train + validate YOLO26-S on the BAMBI wildlife detection set.

Logs to the Docker MLflow server (http://localhost:5000) via Ultralytics' native
MLflow integration. Defaults are tuned for the 4 GB laptop GPU; raise --batch /
--imgsz on the 12 GB card.

    uv run python yolo26-s/train.py                 # full run (50 epochs)
    uv run python yolo26-s/train.py --epochs 1      # quick smoke test
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
    ap.add_argument("--weights", default="yolo26s.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)       # 4 GB-friendly
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="yolo26-s")
    args = ap.parse_args()

    # Route Ultralytics' MLflow callback to the Docker tracking server.
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "bambi-detection")
    run_name = f"train_{args.name}_{args.imgsz}"
    os.environ.setdefault("MLFLOW_RUN", run_name)

    from ultralytics import YOLO, settings

    settings.update({"mlflow": True})

    model = YOLO(args.weights)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=str(HERE / "runs"), name=run_name,
        exist_ok=True,
    )
    metrics = model.val(
        data=args.data, imgsz=args.imgsz, batch=args.batch, device=args.device,
        project=str(HERE / "runs"), name=f"val_{args.name}_{args.imgsz}", exist_ok=True,
    )
    print(f"mAP50-95: {metrics.box.map:.4f}  mAP50: {metrics.box.map50:.4f}")


if __name__ == "__main__":
    main()
