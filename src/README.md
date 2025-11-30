# ds000117 MNE pipeline

`src/pipeline.py` implements an end-to-end preprocessing and decoding pipeline for the ds000117 BIDS dataset using MNE-Python and scikit-learn. The module can be executed directly from the command line or imported into notebooks/scripts.

## Quick start

```bash
python -m src.pipeline \
  --bids-root /workspace/ECE5206P2 \
  --subject 01 \
  --task facerecognition \
  --classifier svm \
  --morlet --include-time-series --pca 50
```

### Key stages

- Load ds000117 via `mne_bids`, auto-discover and concatenate runs per subject, and apply 0.1–40 Hz band-pass plus site-specific line-noise notch filtering (prefers `line_freq` in metadata, otherwise 50/60 Hz).
- Automatic bad-channel detection (peak-to-peak z-score) and interpolation.
- ICA fitting with automatic EOG/ECG component removal.
- Epoching (-200 ms to 800 ms) with baseline correction and trial rejection.
- Feature extraction options: ERP window means, flattened time-series, optional Morlet band power, and PCA.
- Linear SVM, logistic regression, or LDA classifiers evaluated with within-fold standardization, stratified 5-fold CV, and permutation testing.
- Label modes: binary face (famous+unfamiliar) vs scrambled, or triple (famous vs unfamiliar vs scrambled).

### Useful flags

- `--erp start:end` – override default ERP windows (seconds) e.g., `--erp 0.05:0.15 0.15:0.3`.
- `--morlet` – include Morlet band-power features (theta/alpha/beta).
- `--include-time-series` – append flattened epoch time-series.
- `--pca N` – keep the first `N` PCA components after scaling.
- `--no-notch` – disable 50/60 Hz notch filtering.
- `--label-mode {binary,triple}` – choose primary face-vs-scrambled labels or the three-way famous/unfamiliar/scrambled labels.

Results are printed to stdout with cross-validation scores and permutation-test p-values.
