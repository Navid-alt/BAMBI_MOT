"""Build the full-dataset RF-DETR view (symlinks, zero-copy).

RF-DETR reads YOLO format natively but needs the val split named ``valid`` plus a
``data.yaml`` at the root. This creates ``detection_models/_rfdetr_view`` with
directory symlinks back into ``data/yolo_data``. Run once before full RF-DETR
training:

    uv run python tools/make_rfdetr_view.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _dataset_utils import DEFAULT_SRC, build_full_view

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="YOLO dataset root")
    ap.add_argument("--out", type=Path, default=ROOT / "_rfdetr_view", help="view output dir")
    args = ap.parse_args()

    out = build_full_view(args.src, args.out, {"train": "train", "val": "valid", "test": "test"})
    print(f"RF-DETR view ready at: {out}")
    for split in ("train", "valid", "test"):
        d = out / split / "images"
        n = sum(1 for _ in d.iterdir()) if d.exists() else 0
        print(f"  {split:5s}: {n} images ({d})")


if __name__ == "__main__":
    main()
