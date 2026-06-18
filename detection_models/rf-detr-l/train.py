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
    args = ap.parse_args()

    ds = Path(args.dataset_dir)
    if not (ds / "data.yaml").exists():
        raise SystemExit(
            f"RF-DETR view not found at {ds}.\n"
            "Build it first:  uv run python tools/make_rfdetr_view.py"
        )

    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")

    from rfdetr import RFDETRLarge

    model = RFDETRLarge()
    model.train(
        dataset_dir=str(ds), epochs=args.epochs, batch_size=args.batch,
        grad_accum_steps=args.grad_accum, lr=args.lr, device=args.device,
        output_dir=str(HERE / "output"),
        mlflow=True, tensorboard=False,
        project="bambi-detection", run=f"train_{args.name}",
    )


if __name__ == "__main__":
    main()
