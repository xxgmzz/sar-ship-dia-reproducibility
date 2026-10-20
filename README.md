# Repeated-Run Evaluation of an Anisotropic Convolutional Prior for SAR Ship Detection

Companion code and data for the manuscript *"Repeated-Run Evaluation of an
Anisotropic Convolutional Prior for SAR Ship Detection"*, submitted to
*Remote Sensing* (MDPI).

The paper is first and foremost a **reproducibility / evaluation-methodology
study**. Rather than reporting a single-run accuracy gain, we use repeated
paired runs to separate genuine effects from noise, quantify the noise floor,
define minimum detectable effects (MDEs), conduct an external component-level
audit of a published method, and derive a reporting checklist. A lightweight
dual-branch anisotropic convolution (DIA-Conv) serves as a transparent case
study for the framework.

## Key findings

- **Small but confirmed positive effect on HRSID (8 paired seeds).**
  mAP@0.5:0.95 +0.47 points (p = 0.008) and AP-small +0.67 (p = 0.0134);
  the paired difference is positive in 8/8 pairs (sign test, p = 0.008).
- **Cautionary single-run artifact.** A single run (seed 42) appeared to give
  AP-large +10.33; over eight runs the effect shrank to a non-significant
  +1.18 — illustrating why single-run comparisons mislead.
- **External component-level audit (5 paired seeds).** A faithful reproduction
  of the StripCGLU backbone core is statistically indistinguishable from the
  baseline on every metric (p = 0.15–0.99). We do not claim the original work
  is wrong; the result bounds what the core alone establishes.
- **Matched-parameter isotropic control.** Total effect confirmed, attribution
  left open: an isotropic second branch alone gives +0.16 (p = 0.25), and the
  additional anisotropic increment is +0.31 (p = 0.0545).
- **Cross-modality negative result (HRSC2016, 5 paired seeds).**
  mAP@0.5:0.95 −2.06 (p = 0.0017), negative across all size-specific metrics
  and orientations (survives Bonferroni correction).
- **Cross-domain (HRSID -> SSDD).** No significant mAP@0.5 shift; AP-medium
  decreases by 1.87 (p = 0.02), reported explicitly in the conclusions.

## Repository layout

| Path | Contents |
|---|---|
| `configs/` | `custom_blocks.py` (module definitions) and a clean `hrsid.yaml` |
| `src/` | Self-contained training, evaluation, bootstrap/MDE, audit, and ERF scripts (see `src/README.md`) |
| `splits/` | Exact train/val image lists for HRSID, SSDD, and HRSC2016 (random splits) |
| `results/` | Per-run CSVs, mean/std tables, paired tests, and the bootstrap/MDE tables |
| `figures/` | Reproduced figures (two-layer noise, gradient ERF, etc.) |
| `seeds.md` | All seeds and the bootstrap RNG |
| `requirements.txt` | Python dependencies |

## Quick start

1. **Environment** (Python 3.10; CUDA 12.1 used in our experiments):
   ```bash
   pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```
2. **Datasets**: obtain HRSID, SSDD, and HRSC2016 from their official sources
   and arrange them in Ultralytics layout (`images/` + `labels/`, each with
   `train/` and `val/`). Apply the exact file lists in `splits/` to reproduce
   our partitions.
3. **Edit the configuration region** at the top of each script (paths, seeds,
   output root).
4. **Reproduce** (see `src/README.md` for details):
   ```bash
   python src/train_hrsid_n8.py        # HRSID baseline vs DIA-Conv, 8 seeds
   python src/eval_bootstrap_mde.py    # noise floor, bootstrap, MDE, required runs
   python src/audit_stripcglu_n5.py    # StripCGLU external audit (amp=False)
   python src/iso_control_n8.py        # isotropic control, 8 seeds
   python src/train_hrsc_n5.py         # optical HRSC2016, 5 seeds
   python src/eval_crossdomain_n8.py   # cross-domain inference on SSDD
   ```

## Model weights

Trained weights (all `best.pt` files, ~263 MB) are packaged separately and
published on the **[Releases page](../../releases)** of this repository.
Download `weights_release.zip` and extract it next to the scripts to reproduce
the evaluation without retraining.

## Citation

Citation details will be completed upon publication. Until then, please cite
the manuscript as *Repeated-Run Evaluation of an Anisotropic Convolutional
Prior for SAR Ship Detection*, Remote Sensing (MDPI), under review.

## License

Released under the [MIT License](LICENSE).
