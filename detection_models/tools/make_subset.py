"""Build a small symlinked subset for the speed benchmark.

Produces two layouts under ``detection_models/_subset`` from the same selected
images (prefers frames that actually contain boxes):

    _subset/yolo/    -> train/ + val/   (Ultralytics / YOLO26)
    _subset/rfdetr/  -> train/ + valid/ (RF-DETR native YOLO loader)

    uv run python tools/make_subset.py --n-train 500 --n-val 50
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _dataset_utils import DEFAULT_SRC, build_subset

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="YOLO dataset root")
    ap.add_argument("--out", type=Path, default=ROOT / "_subset", help="subset output dir")
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-val", type=int, default=50)
    args = ap.parse_args()

    yolo = build_subset(args.src, args.out / "yolo", args.n_train, args.n_val, val_dirname="val")
    rfdetr = build_subset(args.src, args.out / "rfdetr", args.n_train, args.n_val, val_dirname="valid")
    print(f"YOLO subset   -> {args.out / 'yolo'}   {yolo}")
    print(f"RF-DETR subset -> {args.out / 'rfdetr'}  {rfdetr}")


if __name__ == "__main__":
    main()
