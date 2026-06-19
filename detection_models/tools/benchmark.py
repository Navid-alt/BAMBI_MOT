"""Run ONE detector for 1 epoch on the speed-test subset and record metrics.

Measures wall-clock, throughput (img/s) and peak VRAM, then writes a JSON file
to ``reports/_bench/<model>.json`` AND logs a run to the MLflow ``bambi-bench``
experiment (separate from the ``bambi-detection`` experiment used by the real
``train.py`` scripts). OOM (possible for the heavier ``-l`` models at this
resolution) is caught and recorded rather than crashing the run.

    uv run python tools/benchmark.py --model yolo26-s
    uv run python tools/benchmark.py --model rf-detr-l --batch 2   # override batch

All four models train at the native frame size 1024 (the BAMBI frames are
1024x1024, so there is no downscaling). 1024 is valid for both detectors:
Ultralytics needs a multiple of 32, and RFDETRSmall/Large need a multiple of
patch_size(16) * num_windows(2) = 32 — 1024 satisfies both. Small models use a
larger batch, ``-l`` models a smaller one, sized for a 12 GB GPU.

Aggregate the JSON files into a Markdown report with tools/make_report.py.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from _dataset_utils import FULL_TRAIN_IMAGES, IMG_EXTS

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "reports" / "_bench"
RUNS_DIR = ROOT / "_bench_runs"

# Shared input resolution. The BAMBI frames are natively 1024x1024, so we train
# at full resolution (no downscaling). 1024 is valid for both detectors: YOLO
# needs a multiple of 32, and RFDETRSmall/Large need a multiple of
# patch_size(16) * num_windows(2) = 32 — 1024 = 32 * 32 satisfies both.
RESOLUTION = 1024

# MLflow experiment for benchmark runs, kept separate from training.
BENCH_EXPERIMENT = "bambi-bench"


@dataclass(frozen=True)
class Spec:
    family: str            # "yolo" | "rfdetr"
    batch: int             # small models -> larger batch, -l models -> smaller
    resolution: int = 1024 # image size (1024 native, 640 for speed)
    weights: str = ""      # YOLO weights file (auto-downloaded by Ultralytics)
    rf_class: str = ""     # RF-DETR class name exported by the rfdetr package
    grad_accum: int = 1


# Batches sized for a 12 GB GPU at resolution 1024 (lowered from the 736 values
# since 1024 is ~1.9x the pixels). The benchmark records OOM gracefully, so tune
# here if a model still overflows. 640 resolution uses higher batches (~2.5x)
# since it's only 0.39x the pixels of 1024.
SPECS: dict[str, Spec] = {
    "yolo26-s":      Spec("yolo",   batch=12, resolution=1024, weights="yolo26s.pt"),
    "yolo26-l":      Spec("yolo",   batch=5,  resolution=1024, weights="yolo26l.pt"),
    "rf-detr-s":     Spec("rfdetr", batch=4,  resolution=1024, rf_class="RFDETRSmall"),
    "rf-detr-l":     Spec("rfdetr", batch=3,  resolution=1024, rf_class="RFDETRLarge"),
    "yolo26-s-640":  Spec("yolo",   batch=30, resolution=640,  weights="yolo26s.pt"),
    "yolo26-l-640":  Spec("yolo",   batch=12, resolution=640,  weights="yolo26l.pt"),
    "rf-detr-s-640": Spec("rfdetr", batch=10, resolution=640,  rf_class="RFDETRSmall"),
    "rf-detr-l-640": Spec("rfdetr", batch=8,  resolution=640,  rf_class="RFDETRLarge"),
}


def _count_images(d: Path) -> int:
    return sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXTS) if d.exists() else 0


def _gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"


def _peak_vram() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_reserved() / 1e9, 2)


def _param_count(obj) -> int:
    """Exact number of parameters of the underlying torch model.

    Accepts an ``nn.Module`` or a wrapper exposing one through nested ``.model``
    attributes (Ultralytics: ``YOLO.model``; RF-DETR: ``RFDETR.model.model``).
    """
    cur = obj
    for _ in range(4):  # descend through a few .model wrappers
        if isinstance(cur, torch.nn.Module):
            return sum(p.numel() for p in cur.parameters())
        cur = getattr(cur, "model", None)
        if cur is None:
            break
    raise AttributeError(f"no nn.Module found under {type(obj).__name__}")


def run_yolo(spec: Spec, name: str, record: dict) -> dict:
    from ultralytics import YOLO

    data_yaml = ROOT / "_subset" / "yolo" / "data.yaml"
    n_train = _count_images(ROOT / "_subset" / "yolo" / "train" / "images")
    model = YOLO(spec.weights)
    record["params"] = _param_count(model)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model.train(
        data=str(data_yaml), epochs=1, imgsz=spec.resolution, batch=spec.batch, device=0,
        workers=4, project=str(RUNS_DIR), name=name, exist_ok=True,
        val=False, plots=False, verbose=False,
    )
    elapsed = time.perf_counter() - t0
    return {"n_train": n_train, "elapsed_s": elapsed, "peak_vram_gb": _peak_vram(),
            "batch": spec.batch, "resolution": spec.resolution,
            "config": f"imgsz={spec.resolution}, batch={spec.batch}"}


def run_rfdetr(spec: Spec, name: str, record: dict) -> dict:
    import rfdetr

    dataset_dir = ROOT / "_subset" / "rfdetr"
    n_train = _count_images(dataset_dir / "train" / "images")
    model = getattr(rfdetr, spec.rf_class)()
    record["params"] = _param_count(model)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model.train(
        dataset_dir=str(dataset_dir), epochs=1, batch_size=spec.batch,
        grad_accum_steps=spec.grad_accum, resolution=spec.resolution,
        output_dir=str(RUNS_DIR / name), device="cuda",
        tensorboard=False, mlflow=False, progress_bar="tqdm",
    )
    elapsed = time.perf_counter() - t0
    return {"n_train": n_train, "elapsed_s": elapsed, "peak_vram_gb": _peak_vram(),
            "batch": spec.batch, "resolution": spec.resolution,
            "config": f"resolution={spec.resolution}, batch_size={spec.batch}, grad_accum={spec.grad_accum}"}


def run_model(spec: Spec, name: str, record: dict) -> dict:
    runner = run_yolo if spec.family == "yolo" else run_rfdetr
    return runner(spec, name, record)


def _mlflow_run(model_name: str):
    """Open an MLflow run in the benchmark experiment, or a no-op context.

    Started *before* training so Ultralytics' MLflow callback logs into this run
    instead of opening its own (it only creates a run when none is active). The
    benchmark must never depend on the tracking server, so any failure (URI
    unset, server down) falls back to a null context and logging is skipped.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        return contextlib.nullcontext(), False
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", BENCH_EXPERIMENT))
        return mlflow.start_run(run_name=model_name), True
    except Exception as exc:  # noqa: BLE001 - logging is best-effort
        print(f"[mlflow] logging disabled: {exc}")
        return contextlib.nullcontext(), False


def _log_to_mlflow(record: dict) -> None:
    """Log benchmark params/metrics to the active run (prefixed to avoid clashing
    with the params Ultralytics' own callback logs into the same YOLO run)."""
    try:
        import mlflow

        params = {f"bench_{k}": record[k] for k in
                  ("model", "family", "config", "gpu", "status", "n_train", "batch", "resolution", "params")
                  if k in record}
        mlflow.log_params(params)
        mlflow.set_tag("status", record.get("status", "?"))

        metrics = {f"bench_{k}": record[k] for k in
                   ("img_per_s", "peak_vram_gb", "elapsed_s") if k in record}
        if record.get("img_per_s"):
            epoch_s = FULL_TRAIN_IMAGES / record["img_per_s"]
            metrics["bench_est_epoch_min"] = round(epoch_s / 60, 2)
            metrics["bench_est_50ep_h"] = round(epoch_s * 50 / 3600, 2)
        if metrics:
            mlflow.log_metrics(metrics)
    except Exception as exc:  # noqa: BLE001 - logging is best-effort
        print(f"[mlflow] failed to log metrics: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(SPECS))
    ap.add_argument("--batch", type=int, default=None,
                    help="override the per-model default batch size")
    args = ap.parse_args()

    spec = SPECS[args.model]
    if args.batch is not None:
        spec = replace(spec, batch=args.batch)

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    record = {"model": args.model, "family": spec.family, "gpu": _gpu_name(), "status": "ok"}

    run_ctx, mlflow_on = _mlflow_run(args.model)
    with run_ctx:
        try:
            record.update(run_model(spec, args.model, record))
            record["img_per_s"] = round(record["n_train"] / record["elapsed_s"], 2)
            print(f"[{args.model}] {record['img_per_s']} img/s, "
                  f"peak VRAM {record['peak_vram_gb']} GB over {record['n_train']} imgs")
        except torch.cuda.OutOfMemoryError as exc:
            record["status"] = "OOM"
            record["error"] = str(exc).splitlines()[0][:200]
            print(f"[{args.model}] OUT OF MEMORY on {record['gpu']} — recorded as OOM")
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 - record any failure for the report
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}".splitlines()[0][:200]
            print(f"[{args.model}] ERROR: {record['error']}")

        if mlflow_on:
            _log_to_mlflow(record)

    out = BENCH_DIR / f"{args.model}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
