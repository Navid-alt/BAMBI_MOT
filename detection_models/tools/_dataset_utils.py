"""Shared helpers for arranging the BAMBI YOLO dataset for YOLO26 and RF-DETR.

The source dataset (``data/yolo_data``) is in YOLO format with splits named
``train`` / ``val`` / ``test``. Two consumers need slightly different layouts:

* **YOLO26 (Ultralytics)** reads ``data.yaml`` directly — no view needed for the
  full dataset.
* **RF-DETR 1.8.0** reads YOLO format natively *but* (Roboflow convention)
  requires the val split to be named ``valid`` and a ``data.yaml`` at the
  dataset root.

To avoid copying the 16 GB image set we build lightweight **symlink views**:
whole-directory symlinks for the full dataset, per-file symlinks for subsets.
"""
from __future__ import annotations

import shutil
from pathlib import Path

DEFAULT_SRC = Path("/home/stas/my_git/BAMBI_MOT/data/yolo_data")
CLASS_NAMES: dict[int, str] = {0: "Wild boar", 1: "Red deer", 2: "Roe deer"}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Number of training images in the full dataset (data/yolo_data/train) — used by
# the benchmark to extrapolate subset throughput to a full epoch.
FULL_TRAIN_IMAGES = 17684


def _relink(target: Path, link: Path) -> None:
    """Create/replace ``link`` as a symlink pointing at absolute ``target``."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link)
        else:
            link.unlink()
    link.symlink_to(target.resolve())


def write_data_yaml(out_dir: Path, val_dirname: str, names: dict[int, str] = CLASS_NAMES) -> None:
    """Write a YOLO ``data.yaml`` understood by both Ultralytics and RF-DETR."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Auto-generated — do not edit by hand (see tools/_dataset_utils.py).",
        f"path: {out_dir.resolve()}",
        "train: train/images",
        f"val: {val_dirname}/images",
        f"nc: {len(names)}",
        "names:",
    ]
    lines += [f"  {idx}: {names[idx]}" for idx in sorted(names)]
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n")


def build_full_view(src: Path, out: Path, split_map: dict[str, str]) -> Path:
    """Symlink whole split directories (zero-copy view).

    ``split_map`` maps source split name -> destination split name, e.g.
    ``{"train": "train", "val": "valid", "test": "test"}`` for RF-DETR.
    """
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for src_split, dst_split in split_map.items():
        s = src / src_split
        if s.exists():
            _relink(s, out / dst_split)
    write_data_yaml(out, val_dirname=split_map.get("val", "val"))
    return out


def _select_stems(split_dir: Path, n: int, require_boxes: bool = True) -> list[tuple[Path, Path]]:
    """Pick up to ``n`` (image, label) pairs; prefer images that contain boxes."""
    img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
    with_boxes: list[tuple[Path, Path]] = []
    without: list[tuple[Path, Path]] = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        has_boxes = lbl.exists() and lbl.stat().st_size > 0
        (with_boxes if has_boxes else without).append((img, lbl))
        if require_boxes and len(with_boxes) >= n:
            break
    chosen = with_boxes[:n]
    if len(chosen) < n:  # top up with background frames if needed
        chosen += without[: n - len(chosen)]
    return chosen


def build_subset(src: Path, out: Path, n_train: int, n_val: int, val_dirname: str) -> dict[str, int]:
    """Build a per-file symlink subset (train + val) for a speed test.

    ``val_dirname`` is ``"val"`` for the Ultralytics layout or ``"valid"`` for
    the RF-DETR layout. Returns the actual image counts per split.
    """
    src, out = Path(src), Path(out)
    counts: dict[str, int] = {}
    for src_split, dst_split, n in (("train", "train", n_train), ("val", val_dirname, n_val)):
        pairs = _select_stems(src / src_split, n)
        for img, lbl in pairs:
            _relink(img, out / dst_split / "images" / img.name)
            if lbl.exists():
                _relink(lbl, out / dst_split / "labels" / lbl.name)
            else:  # background frame: emit an empty label so loaders see it
                dst_lbl = out / dst_split / "labels" / (img.stem + ".txt")
                dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                dst_lbl.write_text("")
        counts[dst_split] = len(pairs)
    write_data_yaml(out, val_dirname=val_dirname)
    return counts
