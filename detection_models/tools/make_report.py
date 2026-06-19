"""Aggregate benchmark JSON files into reports/speed_report.md.

Reads every ``reports/_bench/*.json`` produced by benchmark.py and extrapolates
the measured subset throughput to the full training set (data/yolo_data/train).
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from _dataset_utils import FULL_TRAIN_IMAGES

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "reports" / "_bench"
REPORT = ROOT / "reports" / "speed_report.md"


def _fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def _fmt_params(record: dict) -> str:
    p = record.get("params")
    return f"{p:,} ({p / 1e6:.1f} M)" if isinstance(p, int) else "-"


def main() -> None:
    records = [json.loads(p.read_text()) for p in sorted(BENCH_DIR.glob("*.json"))]
    if not records:
        raise SystemExit(f"No benchmark results in {BENCH_DIR}. Run benchmark_speed.sh first.")

    gpu = next((r.get("gpu") for r in records if r.get("gpu")), "unknown")
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    header = (
        "| Model | Params | Config | Subset imgs | img/s | Peak VRAM | Est. 1 epoch (17,684) | "
        "Est. 50 epochs |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    rows = []
    for r in records:
        if r.get("status") == "ok":
            ips = r["img_per_s"]
            epoch_s = FULL_TRAIN_IMAGES / ips
            rows.append(
                f"| {r['model']} | {_fmt_params(r)} | {r.get('config','')} | {r['n_train']} | {ips} | "
                f"{r['peak_vram_gb']} GB | {_fmt_duration(epoch_s)} | "
                f"{_fmt_duration(epoch_s * 50)} |"
            )
        else:
            note = r.get("error", r.get("status", ""))
            rows.append(
                f"| {r['model']} | {_fmt_params(r)} | {r.get('config','')} | {r.get('n_train','-')} | - | - | - | - |"
            )

    body = "\n".join(
        [
            "# Detector speed benchmark",
            "",
            f"- **Generated:** {now}",
            f"- **GPU:** {gpu}",
            f"- **Method:** 1 epoch on a small symlinked subset (resolution 736); throughput "
            f"extrapolated to the full {FULL_TRAIN_IMAGES:,}-image training set.",
            "- **MLflow:** each run is logged to the `bambi-bench` experiment at "
            "http://localhost:5000 (separate from the `bambi-detection` training experiment).",
            "",
            header,
            *rows,
            "",
        ]
    )
    REPORT.write_text(body)
    print(f"wrote {REPORT}")
    print(body)


if __name__ == "__main__":
    main()
