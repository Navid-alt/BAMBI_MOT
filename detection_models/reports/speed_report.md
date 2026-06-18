# Detector speed benchmark

- **Generated:** 2026-06-18 18:44
- **GPU:** NVIDIA GeForce RTX 4070
- **Method:** 1 epoch on a small symlinked subset (resolution 736); throughput extrapolated to the full 17,684-image training set.
- **MLflow:** each run is logged to the `bambi-bench` experiment at http://localhost:5000 (separate from the `bambi-detection` training experiment).

| Model | Params | Config | Subset imgs | img/s | Peak VRAM | Est. 1 epoch (17,684) | Est. 50 epochs | Status |
|---|---|---|---|---|---|---|---|---|
| rf-detr-l | 33,933,218 (33.9 M) | resolution=736, batch_size=6, grad_accum=1 | 500 | 13.48 | 10.02 GB | 21.9 min | 18.22 h | ok |
| rf-detr-s | 32,111,170 (32.1 M) | resolution=736, batch_size=8, grad_accum=1 | 500 | 15.35 | 12.08 GB | 19.2 min | 16.00 h | ok |
| yolo26-l | 26,299,704 (26.3 M) | imgsz=736, batch=10 | 500 | 25.3 | 9.83 GB | 11.6 min | 9.71 h | ok |
| yolo26-s | 10,009,784 (10.0 M) | imgsz=736, batch=24 | 500 | 44.95 | 9.8 GB | 6.6 min | 5.46 h | ok |
