# Random Seeds

All reported values use paired seeds: the baseline and the variant are trained
with the same seed within each pair, so paired tests remove shared
run-to-run variation.

| Experiment | Seeds | n |
|---|---|---|
| HRSID main line (baseline & DIA-Conv) | 42, 123, 2024, 7, 2025, 11, 88, 314 | 8 |
| ISO control (HRSID) | 42, 123, 2024, 7, 2025, 11, 88, 314 | 8 |
| HRSC2016 (optical, baseline & DIA-Conv) | 42, 123, 2024, 7, 2025 | 5 |
| StripCGLU external audit (HRSID) | 42, 123, 2024, 7, 2025 | 5 |
| Channel-ratio 3:1 (single, HRSID) | 42 | 1 |
| Bootstrap resampling (RNG) | 20261006 | 500 resamples |

The seed is passed to Ultralytics `model.train(seed=...)`, which seeds Python,
NumPy, and PyTorch (including CUDA determinism settings).
