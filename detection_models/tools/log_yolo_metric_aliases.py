"""Re-log RF-DETR validation metrics under YOLO/Ultralytics names on an epoch axis.

RF-DETR (PyTorch-Lightning) logs validation metrics on a *global-step* x-axis
(e.g. step 982, 1965, ... — one epoch's worth of optimizer steps apart), while
Ultralytics logs on an *epoch* x-axis (step 0, 1, 2, ...). To make the two
directly comparable in the same MLflow chart panel, this helper copies the four
core RF-DETR (non-EMA) metrics into the *same run* under the exact YOLO metric
names, re-keyed to step = epoch index.

Everything RF-DETR already logs is left untouched; this only *adds* four series.

Mapping (RF-DETR non-EMA  ->  YOLO name):

    val/mAP_50_95  ->  metrics/mAP50-95B
    val/mAP_50     ->  metrics/mAP50B
    val/precision  ->  metrics/precisionB
    val/recall     ->  metrics/recallB

Run from train.py after `model.train()` returns, or standalone for testing:

    uv run python tools/log_yolo_metric_aliases.py --run-id <mlflow_run_id>
    uv run python tools/log_yolo_metric_aliases.py --run-name train_rf-detr-l_1024
"""
from __future__ import annotations

import argparse
import os

from mlflow.tracking import MlflowClient

# RF-DETR non-EMA source metric  ->  Ultralytics/YOLO target name.
SOURCE_TO_YOLO: dict[str, str] = {
    "val/mAP_50_95": "metrics/mAP50-95B",
    "val/mAP_50": "metrics/mAP50B",
    "val/precision": "metrics/precisionB",
    "val/recall": "metrics/recallB",
}


def find_latest_run_id(
    experiment_name: str, run_name: str, *, tracking_uri: str | None = None
) -> str | None:
    """Return the most recent run id matching (experiment_name, run_name), or None."""
    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


def relog_yolo_aliases(
    run_id: str, *, tracking_uri: str | None = None, force: bool = False
) -> dict[str, int]:
    """Copy the four source metrics into `run_id` under YOLO names, keyed by epoch.

    The i-th logged point of each source metric (sorted by step) is re-logged at
    step=i — RF-DETR validates once per epoch, so i is the epoch index.

    Idempotent: a target epoch already present is skipped, so re-running (or
    resuming training) never duplicates points. Pass ``force=True`` to re-write
    every point regardless.

    Returns {yolo_name: n_points_written}.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    existing = client.get_run(run_id).data.metrics
    written: dict[str, int] = {}

    for src, dst in SOURCE_TO_YOLO.items():
        history = sorted(client.get_metric_history(run_id, src), key=lambda h: h.step)
        if not history:
            continue
        done_epochs = (
            {h.step for h in client.get_metric_history(run_id, dst)}
            if (dst in existing and not force)
            else set()
        )
        n = 0
        for epoch, point in enumerate(history):
            if epoch in done_epochs:
                continue
            client.log_metric(run_id, dst, point.value, step=epoch)
            n += 1
        written[dst] = n
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", help="MLflow run id to update (overrides --run-name)")
    ap.add_argument("--experiment", default="bambi-detection")
    ap.add_argument("--run-name", help="resolve the latest run with this name")
    ap.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    ap.add_argument(
        "--force", action="store_true", help="re-write every point, even existing ones"
    )
    args = ap.parse_args()

    run_id = args.run_id
    if run_id is None:
        if not args.run_name:
            raise SystemExit("Provide --run-id or --run-name.")
        run_id = find_latest_run_id(
            args.experiment, args.run_name, tracking_uri=args.tracking_uri
        )
        if run_id is None:
            raise SystemExit(
                f"No run named {args.run_name!r} in experiment {args.experiment!r}."
            )

    written = relog_yolo_aliases(run_id, tracking_uri=args.tracking_uri, force=args.force)
    if not written:
        print(f"No source metrics found on run {run_id}; nothing logged.")
        return
    print(f"Re-logged YOLO-named aliases on run {run_id} (step = epoch):")
    for name, n in written.items():
        print(f"  {name:22s} +{n} point(s)")


if __name__ == "__main__":
    main()
