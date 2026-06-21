"""Build detector comparison plots for the presentation from MLflow history.

Pulls the per-epoch metric history for the four trained `bambi-detection` runs
and renders two figures into presentation/images/:
  - detector_curves.png      (2x2 grid: mAP50, mAP50-95, precision, recall vs epoch)
  - detector_final_bars.png  (grouped bars of the final-epoch metrics)

    uv run python tools/make_presentation_plots.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

OUT = Path(__file__).resolve().parents[2] / "presentation" / "images"
OUT.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri("http://localhost:5000")
C = MlflowClient()

# name -> (run_id, colour)  ordered worst→best so the legend reads naturally.
RUNS = [
    ("YOLO26-s · 640",   "b933bca7418c41fe994509942289dbab", "#111111"),
    ("YOLO26-l · 640",   "ccb49d353bed4cd48963ab1c9b9bd0db", "#4C9BE8"),
    ("YOLO26-l · 1024",  "c98b9e34544f4d6599f5987649fa934e", "#B5328A"),
    ("RF-DETR-l · 1024", "57dfb1f2fd704210a658e32ecea487f9", "#8C9BAA"),
]

METRICS = [
    ("metrics/mAP50B",    "mAP@50"),
    ("metrics/mAP50-95B", "mAP@50-95"),
    ("metrics/precisionB", "Precision"),
    ("metrics/recallB",    "Recall"),
]


def history(run_id: str, key: str):
    pts = sorted(C.get_metric_history(run_id, key), key=lambda m: m.step)
    return [p.step for p in pts], [p.value for p in pts]


def curves() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (key, title) in zip(axes.flat, METRICS):
        for name, rid, col in RUNS:
            xs, ys = history(rid, key)
            ax.plot(xs, ys, color=col, lw=1.8, marker="o", ms=2.5, label=name)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=9, loc="lower right", framealpha=0.9)
    fig.suptitle("Detector validation metrics over training (BAMBI, 50 epochs)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "detector_curves.png", dpi=150)
    plt.close(fig)
    print("wrote", OUT / "detector_curves.png")


def final_bars() -> None:
    finals = {}
    for name, rid, col in RUNS:
        finals[name] = {key: history(rid, key)[1][-1] for key, _ in METRICS}
    labels = [t for _, t in METRICS]
    x = range(len(labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, (name, rid, col) in enumerate(RUNS):
        vals = [finals[name][key] for key, _ in METRICS]
        bars = ax.bar([xi + (i - 1.5) * w for xi in x], vals, w, label=name, color=col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (validation)")
    ax.set_title("Final detector metrics — best epoch per model",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "detector_final_bars.png", dpi=150)
    plt.close(fig)
    print("wrote", OUT / "detector_final_bars.png")


if __name__ == "__main__":
    curves()
    final_bars()
