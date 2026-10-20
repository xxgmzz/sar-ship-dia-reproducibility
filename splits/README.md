# Dataset Splits

These lists record the exact image file names used in our train/validation splits.
All splits are **random** (not the official fixed splits); the random-split protocol
is itself part of the study and is reported in the paper. Re-using these lists
reproduces our exact partitions.

| Dataset | Split ratio | Train | Val | Lists |
|---|---|---|---|---|
| HRSID | 8:2 | 4484 | 1120 | HRSID_train.txt / HRSID_val.txt |
| SSDD  | 7:3 | 812  | 348  | SSDD_train.txt / SSDD_val.txt |
| HRSC2016 (optical) | ~64:36 | 1070 | 610 | HRSC_train.txt / HRSC_val.txt |

Notes:
- SSDD uses a 7:3 random split (rather than the common 8:2) to keep a larger
  validation set so that size-specific AP metrics retain enough large targets.
- Labels are YOLO/Ultralytics normalized format (one .txt per image, same stem).
- The original images/labels are not redistributed here; obtain them from the
  official dataset releases and apply these file lists.
