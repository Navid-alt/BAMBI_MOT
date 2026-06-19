# Detector speed benchmark

- **Generated:** 2026-06-19 07:46
- **GPU:** NVIDIA GeForce RTX 4070
- **Method:** 1 epoch on a small symlinked subset (resolution 736); throughput extrapolated to the full 17,684-image training set.
- **MLflow:** each run is logged to the `bambi-bench` experiment at http://localhost:5000 (separate from the `bambi-detection` training experiment).

| Model | Params | Config | Subset imgs | img/s | Peak VRAM | Est. 1 epoch (17,684) | Est. 50 epochs |
|---|---|---|---|---|---|---|---|
| rf-detr-l-640 | 33,933,218 (33.9 M) | resolution=640, batch_size=8, grad_accum=1 | 500 | 15.02 | 11.29 GB | 19.6 min | 16.35 h |
| rf-detr-l | 33,933,218 (33.9 M) | resolution=1024, batch_size=3, grad_accum=1 | 500 | 8.17 | 8.36 GB | 36.1 min | 30.06 h |
| rf-detr-s-640 | 32,111,170 (32.1 M) | resolution=640, batch_size=10, grad_accum=1 | 500 | 16.28 | 12.17 GB | 18.1 min | 15.09 h |
| rf-detr-s | 32,111,170 (32.1 M) | resolution=1024, batch_size=4, grad_accum=1 | 500 | 9.07 | 10.94 GB | 32.5 min | 27.08 h |
| yolo26-l-640 | 26,299,704 (26.3 M) | imgsz=640, batch=12 | 500 | 27.97 | 8.87 GB | 10.5 min | 8.78 h |
| yolo26-l | 26,299,704 (26.3 M) | imgsz=1024, batch=5 | 500 | 14.51 | 9.74 GB | 20.3 min | 16.93 h |
| yolo26-s-640 | 10,009,784 (10.0 M) | imgsz=640, batch=30 | 500 | 32.67 | 9.13 GB | 9.0 min | 7.52 h |
| yolo26-s | 10,009,784 (10.0 M) | imgsz=1024, batch=12 | 500 | 26.09 | 9.64 GB | 11.3 min | 9.41 h |
