"""Generate all figures/tables for the transformer_tracking report.

Writes into docs/images/:
  - architecture.png        schematic of frozen RF-DETR + trainable associator
  - train_loss.png          associator training loss per epoch (from MLflow)
  - per_video_metrics.png   MOTA / IDF1 bar chart per val video + overall
  - metrics_table.csv       full CLEAR-MOT summary
  - track_<vid>.png         tracking overlays (boxes coloured by track id)
  - gt_vs_pred_<vid>.png    GT (green) vs predicted (coloured) comparison

Run:
    cd detection_models
    PYTHONPATH=.. uv run python ../transformer_tracking/docs/make_report_assets.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from transformer_tracking.data.clip_dataset import _build_basename_index, load_video_frames
from transformer_tracking.eval import _iou_distance, read_pred

HERE = Path(__file__).resolve().parent
IMG = HERE / "images"
IMG.mkdir(parents=True, exist_ok=True)
TRACKS = HERE.parent / "output" / "tracks" / "val"
SPLIT = "val"

# RF-DETR loss history (MLflow run rfdetr-track); fallback if server is down.
LOSS_FALLBACK = [0.7346, 0.5288, 0.4604, 0.4321, 0.4072, 0.3897, 0.3658, 0.3499, 0.3333, 0.3114]

PALETTE = (plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors)
PALETTE = [tuple(int(255 * c) for c in rgb) for rgb in PALETTE]


def color_for(tid: int) -> tuple[int, int, int]:
    return PALETTE[tid % len(PALETTE)]


# --------------------------------------------------------------------------- #
def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#333"):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                     fc=fc, ec=ec, lw=1.5))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    box(0.02, 0.62, 0.20, 0.22, "Frame t-1", "#eef")
    box(0.02, 0.16, 0.20, 0.22, "Frame t", "#eef")
    box(0.28, 0.40, 0.24, 0.30, "Frozen\nRF-DETR-L\n(detector)", "#bcd6ff")
    box(0.60, 0.40, 0.30, 0.30, "TrackAssociator\n(trainable:\nattention + Sinkhorn)", "#ffe2a8")
    box(0.60, 0.04, 0.30, 0.18, "Matched IDs\n+ births / deaths", "#cdeccd")

    ax.annotate("", (0.28, 0.55), (0.22, 0.73), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", (0.28, 0.55), (0.22, 0.27), arrowprops=dict(arrowstyle="->"))
    ax.annotate("300 query embeds\n+ boxes", (0.55, 0.60), (0.52, 0.78),
                fontsize=7.5, ha="center", arrowprops=dict(arrowstyle="->"))
    ax.annotate("", (0.60, 0.50), (0.52, 0.50), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", (0.75, 0.22), (0.75, 0.40), arrowprops=dict(arrowstyle="->"))
    ax.text(0.40, 0.34, "FROZEN  ❄", color="#1d4ed8", fontsize=8, ha="center")
    ax.text(0.75, 0.74, "TRAINED  ●", color="#b45309", fontsize=8, ha="center")

    ax.set_xlim(0, 0.93)
    ax.set_ylim(0, 0.88)
    ax.set_title("End-to-end RF-DETR tracker — only the associator is trained", fontsize=11)
    fig.tight_layout()
    fig.savefig(IMG / "architecture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote architecture.png")


def fig_loss() -> None:
    losses = LOSS_FALLBACK
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        c = mlflow.tracking.MlflowClient()
        exp = c.get_experiment_by_name("bambi-tracking")
        runs = c.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
        hist = sorted(c.get_metric_history(runs[0].info.run_id, "train/loss"), key=lambda m: m.step)
        if hist:
            losses = [m.value for m in hist]
    except Exception as e:
        print(f"  (mlflow unavailable, using fallback losses: {e})")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(range(len(losses)), losses, "-o", color="#b45309", lw=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("assignment NLL loss")
    ax.set_title("TrackAssociator training loss (frozen RF-DETR)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMG / "train_loss.png", dpi=150)
    plt.close(fig)
    print("wrote train_loss.png")


# --------------------------------------------------------------------------- #
def compute_metrics():
    import motmetrics as mm
    accs, names = [], []
    for name, frames in load_video_frames(SPLIT):
        acc = mm.MOTAccumulator(auto_id=False)
        preds = read_pred(TRACKS / f"{name}.txt")
        for t, fr in enumerate(frames, start=1):
            p_ids, p_boxes = preds.get(t, ([], []))
            dist = _iou_distance(fr.boxes.tolist(), p_boxes, 0.5)
            acc.update(fr.track_ids.tolist(), p_ids, dist, frameid=t)
        accs.append(acc)
        names.append(name)
    mh = mm.metrics.create()
    metrics = ["mota", "idf1", "num_switches", "num_false_positives", "num_misses",
               "num_objects", "mostly_tracked", "mostly_lost", "num_fragmentations",
               "precision", "recall"]
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    summary.to_csv(IMG / "metrics_table.csv")
    print("wrote metrics_table.csv")
    return summary


def fig_per_video(summary) -> None:
    df = summary.drop(index="OVERALL")
    df = df.sort_values("idf1", ascending=False)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - 0.2, df["mota"] * 100, 0.4, label="MOTA", color="#2563eb")
    ax.bar(x + 0.2, df["idf1"] * 100, 0.4, label="IDF1", color="#f59e0b")
    ax.axhline(0, color="#333", lw=0.8)
    ov = summary.loc["OVERALL"]
    ax.axhline(ov["idf1"] * 100, color="#f59e0b", ls="--", lw=1,
               label=f"overall IDF1 {ov['idf1']*100:.1f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=90, fontsize=7)
    ax.set_ylabel("%")
    ax.set_title("Per-video MOTA / IDF1 on the val split (sorted by IDF1)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG / "per_video_metrics.png", dpi=150)
    plt.close(fig)
    print("wrote per_video_metrics.png")


# --------------------------------------------------------------------------- #
def _load_frame(idx, name):
    for n, frames in load_video_frames(SPLIT):
        if n == name:
            return frames
    return None


def overlay_strip(name: str, n_panels: int = 4, gt: bool = False) -> None:
    """Save a horizontal strip of frames with predicted (and optionally GT) boxes."""
    bn = _build_basename_index()
    frames = None
    for n, fr in load_video_frames(SPLIT):
        if n == name:
            frames = fr
            break
    if not frames:
        return
    preds = read_pred(TRACKS / f"{name}.txt")
    # choose frames that actually have predictions, evenly spaced
    have = [t for t in range(1, len(frames) + 1) if preds.get(t, ([], []))[0]]
    if not have:
        have = list(range(1, len(frames) + 1))
    pick = [have[int(i)] for i in np.linspace(0, len(have) - 1, min(n_panels, len(have)))]

    W, H = frames[0].orig_size

    def panel_crop(boxes):
        """Square crop centred on this frame's boxes so small animals are visible."""
        if not boxes:
            return (0, 0, W, H)
        xs = [b[0] for b in boxes] + [b[0] + b[2] for b in boxes]
        ys = [b[1] for b in boxes] + [b[1] + b[3] for b in boxes]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        half = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.7 + 90
        half = min(max(half, 110), max(W, H) / 2)
        return (max(0, cx - half), max(0, cy - half), min(W, cx + half), min(H, cy + half))

    panels = []
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    for t in pick:
        fr = frames[t - 1]
        p_ids, p_boxes = preds.get(t, ([], []))
        cboxes = list(p_boxes) + (fr.boxes.tolist() if gt else [])
        crop = panel_crop(cboxes)
        im = Image.open(fr.image_path).convert("RGB")
        im = ImageEnhance.Contrast(im).enhance(1.25)
        im = im.crop(tuple(map(int, crop)))
        ox, oy = crop[0], crop[1]
        d = ImageDraw.Draw(im)
        if gt:
            for b in fr.boxes.tolist():
                x, y, w, h = b
                d.rectangle([x - ox, y - oy, x + w - ox, y + h - oy], outline=(0, 230, 0), width=2)
        p_ids, p_boxes = preds.get(t, ([], []))
        for tid, b in zip(p_ids, p_boxes):
            x, y, w, h = b
            c = color_for(tid)
            d.rectangle([x - ox, y - oy, x + w - ox, y + h - oy], outline=c, width=3)
            d.text((x - ox, max(0, y - oy - 16)), f"#{tid}", fill=c, font=font)
        d.text((6, 6), f"frame {t}", fill=(255, 255, 0), font=font)
        panels.append(im.resize((360, 360)))

    strip = Image.new("RGB", (360 * len(panels), 360), "white")
    for i, p in enumerate(panels):
        strip.paste(p, (360 * i, 0))
    out = IMG / (f"gt_vs_pred_{name}.png" if gt else f"track_{name}.png")
    strip.save(out)
    print(f"wrote {out.name}")


def main() -> None:
    fig_architecture()
    fig_loss()
    summary = compute_metrics()
    fig_per_video(summary)
    for vid in ("372", "189", "193", "142"):
        overlay_strip(vid)
    overlay_strip("189", gt=True)
    print("done.")


if __name__ == "__main__":
    main()
