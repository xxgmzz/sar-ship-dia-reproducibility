# Scripts (`src/`)

Each script is **self-contained**: the custom modules (DIA-Conv / StripCGLU /
ISO control) are defined inside the script, and models are built from the
official `yolov8n.pt` with a custom trainer that swaps modules dynamically — no
hand-written model architecture YAML is required.

Every script has a clearly marked **configuration region near the top**
(paths, seeds, output root, hyperparameters). Before running, edit those paths
to match your local layout. Scripts are resumable: completed runs are skipped.

| Script | Purpose | Runs | Entry |
|---|---|---|---|
| `train_hrsid_n8.py` | HRSID main line: baseline vs DIA-Conv (shallow injection at layer 2, channel 1:1, k=3) | 8 paired seeds | `python train_hrsid_n8.py` |
| `eval_bootstrap_mde.py` | **Zero training.** Per-image bootstrap (test-set sampling noise), run-to-run noise floor, paired differences, MDE table, required-runs table | 500 resamples | `python eval_bootstrap_mde.py` (or `smoke`) |
| `audit_stripcglu_n5.py` | External component-level audit: faithful reproduction of the StripCGLU backbone core, compared to a shared-seed baseline | 5 paired seeds | `python audit_stripcglu_n5.py` |
| `iso_control_n8.py` | Matched-parameter isotropic control (depthwise 3x3 second branch) to probe attribution | 8 paired seeds | `python iso_control_n8.py` |
| `train_hrsc_n5.py` | Optical HRSC2016: baseline vs DIA-Conv (cross-modality test) | 5 paired seeds | `python train_hrsc_n5.py` |
| `train_ssdd_n5.py` | SSDD (saturated small dataset): baseline vs the final DIA-Conv (L2-only, k=3) under the unified protocol | 5 paired seeds | `python train_ssdd_n5.py` |
| `run_ssdd_parallel.py` | Parallel dispatcher (4 workers) for the SSDD n=5 jobs; auto-locates `train_ssdd_n5.py` and summarizes at the end | - | `python run_ssdd_parallel.py` |
| `analyze_ssdd_n5.py` | Paired statistics for SSDD n=5 (paired t, Wilcoxon, sign test, Cohen's d_z, Bonferroni); **requires SciPy** | - | `python analyze_ssdd_n5.py` |
| `eval_crossdomain_n8.py` | Cross-domain evaluation (HRSID-trained models tested on SSDD), inference only | 8 paired seeds | `python eval_crossdomain_n8.py` |
| `channel_ratio_3to1.py` | Channel-ratio 3:1 variant (single run, seed 42) | 1 | `python channel_ratio_3to1.py` |
| `gradient_erf.py` | Effective receptive field from real trained-model gradients (64 images x 9 positions) | - | `python gradient_erf.py` |

## Important notes
- `audit_stripcglu_n5.py` must run in **full precision** (`amp=False`): the
  StripCGLU core overflows in fp16. This is reported in the paper.
- `eval_bootstrap_mde.py` reads the per-run CSVs and representative weights;
  point its `DATASETS` entries at the outputs of the training scripts.
- Paired tests and the t-distribution helpers are implemented inside the
  training scripts; `analyze_ssdd_n5.py` additionally uses SciPy (listed in
  `requirements.txt`).
- The SSDD n=5 results (Section 4.4.1, Table 2) show no metric reaching
  significance; the AP-large trend is negative in all five pairs (p = 0.099)
  and is reported as an underpowered, non-resolved trend rather than a
  confirmed decline.
- Suggested reproduction order: `train_hrsid_n8.py` -> `eval_bootstrap_mde.py`
  -> `audit_stripcglu_n5.py` / `iso_control_n8.py`; cross-domain and
  cross-modality use the same trained models.
