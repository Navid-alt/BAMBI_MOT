"""Sample images from the YOLO dataset and render their ground-truth bboxes.

Outputs 20 annotated images into reports/annotation_examples/.
"""
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "yolo_data"
OUT = ROOT / "reports" / "annotation_examples"
OUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "Wild boar", 1: "Red deer", 2: "Roe deer"}
# distinct, high-contrast colors per class (RGB)
CLASS_COLORS = {0: (255, 56, 56), 1: (56, 168, 255), 2: (80, 220, 100)}

N_SAMPLES = 20
SEED = 42


def load_font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def yolo_to_xyxy(cx, cy, bw, bh, W, H):
    x1 = (cx - bw / 2) * W
    y1 = (cy - bh / 2) * H
    x2 = (cx + bw / 2) * W
    y2 = (cy + bh / 2) * H
    return x1, y1, x2, y2


def collect_labeled_pairs():
    """Return list of (image_path, label_path) where the label is non-empty."""
    pairs = []
    for split in ("train", "val", "test"):
        img_dir = DATA / split / "images"
        lbl_dir = DATA / split / "labels"
        if not lbl_dir.is_dir():
            continue
        for lbl in lbl_dir.glob("*.txt"):
            if lbl.stat().st_size == 0:
                continue
            img = img_dir / (lbl.stem + ".png")
            if img.exists():
                pairs.append((img, lbl))
    return pairs


def draw_annotations(img_path: Path, lbl_path: Path, out_path: Path):
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)
    font = load_font(max(14, W // 45))
    line_w = max(2, W // 320)

    for line in lbl_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])
        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, W, H)
        color = CLASS_COLORS.get(cls, (255, 255, 0))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)

        label = CLASS_NAMES.get(cls, str(cls))
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, y1 - th - 4)
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=color)
        draw.text((x1 + 3, ty + 2), label, fill=(0, 0, 0), font=font)

    im.save(out_path)


def main():
    random.seed(SEED)
    pairs = collect_labeled_pairs()
    if not pairs:
        raise SystemExit("No labeled images found.")
    print(f"Found {len(pairs)} labeled images; sampling {N_SAMPLES}.")
    sample = random.sample(pairs, min(N_SAMPLES, len(pairs)))
    for i, (img, lbl) in enumerate(sorted(sample), 1):
        out = OUT / f"annotated_{i:02d}_{img.stem}.png"
        draw_annotations(img, lbl, out)
        print(f"[{i:02d}] {img.name} -> {out.name}")
    print(f"\nSaved {len(sample)} annotated images to {OUT}")


if __name__ == "__main__":
    main()
