"""
MNE-Python pipeline for ds000117.

This module wires together preprocessing, feature extraction, and
classifier evaluation helpers. The command-line interface matches the
original monolithic pipeline while delegating implementation to the
`preprocess`, `features`, and `evaluate` modules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from .evaluate import EvaluationConfig, evaluate_model
from .features import DEFAULT_ERP_WINDOWS, DEFAULT_MORLET_BANDS, FeatureConfig, extract_features
from .preprocess import (
    DEFAULT_FILTER,
    DEFAULT_NOTCH,
    LABEL_MODES,
    PipelineError,
    PreprocessingConfig,
    load_and_preprocess,
)


def run_pipeline(preproc_cfg: PreprocessingConfig, feature_cfg: FeatureConfig, eval_cfg: EvaluationConfig):
    epochs = load_and_preprocess(preproc_cfg)
    X, y = extract_features(epochs, feature_cfg)
    results = evaluate_model(X, y, feature_cfg, eval_cfg)
    return results


def _parse_arguments() -> tuple[PreprocessingConfig, FeatureConfig, EvaluationConfig]:
    import argparse

    parser = argparse.ArgumentParser(description="Run MNE preprocessing and decoding pipeline per subject.")
    parser.add_argument("--bids-root", type=Path, required=True, help="Path to BIDS root (ds000117).")
    parser.add_argument("--subject", required=True, help="Subject label (e.g., 01).")
    parser.add_argument("--task", default="facerecognition", help="Task label, default: facerecognition.")
    parser.add_argument("--session", default=None, help="Optional session label.")
    parser.add_argument("--run", default=None, help="Optional run label.")
    parser.add_argument("--datatype", default="meg", choices=["meg", "eeg"], help="Datatype to load.")
    parser.add_argument("--l-freq", type=float, default=DEFAULT_FILTER[0], help="High-pass cutoff (Hz).")
    parser.add_argument("--h-freq", type=float, default=DEFAULT_FILTER[1], help="Low-pass cutoff (Hz).")
    parser.add_argument("--no-notch", action="store_true", help="Disable notch filtering.")
    parser.add_argument("--z-threshold", type=float, default=5.0, help="Z-score threshold for bad channels.")
    parser.add_argument(
        "--label-mode",
        choices=list(LABEL_MODES),
        default="binary",
        help="Label scheme: binary (face vs scrambled) or triple (famous vs unfamiliar vs scrambled).",
    )
    parser.add_argument("--erp", nargs="*", default=None, help="ERP windows as start:end (seconds).")
    parser.add_argument("--include-time-series", action="store_true", help="Include flattened time-series features.")
    parser.add_argument("--morlet", action="store_true", help="Include Morlet band-power features.")
    parser.add_argument(
        "--pca",
        type=int,
        default=None,
        help="Number of PCA components to keep (applied after scaling).",
    )
    parser.add_argument(
        "--classifier",
        choices=["svm", "logreg", "lda"],
        default="svm",
        help="Classifier to train/evaluate.",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of stratified CV folds.")
    parser.add_argument("--n-permutations", type=int, default=1000, help="Number of permutations.")

    args = parser.parse_args()

    notch_freqs = [] if args.no_notch else list(DEFAULT_NOTCH)
    erp_windows: Tuple[Tuple[float, float], ...] = tuple(DEFAULT_ERP_WINDOWS)
    if args.erp:
        parsed_erp = []
        for window in args.erp:
            try:
                start, end = map(float, window.split(":"))
            except ValueError as exc:
                raise PipelineError(f"Invalid ERP window specification: {window}") from exc
            parsed_erp.append((start, end))
        erp_windows = tuple(parsed_erp)

    morlet_bands = DEFAULT_MORLET_BANDS if args.morlet else None

    preproc_cfg = PreprocessingConfig(
        bids_root=args.bids_root,
        subject=args.subject,
        task=args.task,
        session=args.session,
        run=args.run,
        datatype=args.datatype,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        notch_freqs=notch_freqs,
        bad_z_threshold=args.z_threshold,
        label_mode=args.label_mode,
    )

    feature_cfg = FeatureConfig(
        erp_windows=erp_windows,
        include_time_series=args.include_time_series,
        morlet_bands=morlet_bands,
        n_pca_components=args.pca,
    )

    eval_cfg = EvaluationConfig(
        classifier=args.classifier,
        n_splits=args.n_splits,
        n_permutations=args.n_permutations,
    )
    return preproc_cfg, feature_cfg, eval_cfg


def main():
    preproc_cfg, feature_cfg, eval_cfg = _parse_arguments()
    results = run_pipeline(preproc_cfg, feature_cfg, eval_cfg)
    print("=== Cross-validation ===")
    print(f"Scores: {results['cv_scores']}")
    print(f"Mean: {results['cv_mean']:.3f} +/- {results['cv_std']:.3f}")
    print("=== Permutation test ===")
    print(f"Score: {results['perm_score']:.3f}, p-value: {results['perm_pvalue']:.4f}")


if __name__ == "__main__":
    main()
