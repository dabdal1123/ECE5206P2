"""
MNE-Python pipeline for ds000117.

This module provides utilities to load MEG/EEG data from the ds000117
BIDS dataset via ``mne_bids``. It performs standard preprocessing
(filtering, bad-channel detection/interpolation, ICA-based artifact
removal), epochs the data, and extracts features that can be fed to
scikit-learn classifiers with cross-validation and permutation testing.

Typical usage from the command line::

    python -m src.pipeline --bids-root /workspace/ECE5206P2 --subject 01 \
        --task facerecognition --classifier svm --morlet --include-time-series

The module exposes composable functions so it can also be imported from
notebooks or other scripts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mne
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


DEFAULT_NOTCH = (50.0, 60.0)
DEFAULT_FILTER = (0.1, 40.0)
DEFAULT_BASELINE = (None, 0.0)
DEFAULT_EPOCH_WINDOW = (-0.2, 0.8)
#DEFAULT_REJECT = dict(grad=4000e-13, mag=4e-12, eeg=150e-6, eog=250e-6)
DEFAULT_REJECT = dict(grad=4000e-13, mag=4e-12, eeg=150e-6)
DEFAULT_ERP_WINDOWS = [(0.05, 0.15), (0.15, 0.3), (0.3, 0.6)]
DEFAULT_MORLET_BANDS = {
    "theta": np.arange(4.0, 8.1, 1.0),
    "alpha": np.arange(8.0, 13.0, 1.0),
    "beta": np.arange(13.0, 30.0, 2.0),
}
DEFAULT_TRIGGER_SETS = {
    "Famous": {5, 6, 7},
    "Unfamiliar": {13, 14, 15},
    "Scrambled": {17, 18, 19},
}
LABEL_MODES = ("binary", "triple")


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing and epoching."""

    bids_root: Path
    subject: str
    task: str = "facerecognition"
    session: Optional[str] = "meg"
    run: Optional[str] = None
    datatype: str = "meg"
    l_freq: float = DEFAULT_FILTER[0]
    h_freq: float = DEFAULT_FILTER[1]
    notch_freqs: Sequence[float] = DEFAULT_NOTCH
    tmin: float = DEFAULT_EPOCH_WINDOW[0]
    tmax: float = DEFAULT_EPOCH_WINDOW[1]
    baseline: Tuple[Optional[float], Optional[float]] = DEFAULT_BASELINE
    reject: Optional[Dict[str, float]] = field(default_factory=lambda: DEFAULT_REJECT.copy())
    event_id: Optional[Dict[str, int]] = None
    bad_z_threshold: float = 5.0
    random_state: int = 97
    label_mode: str = "binary"  # binary (face vs scrambled) or triple (famous/unfamiliar/scrambled)


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    erp_windows: Sequence[Tuple[float, float]] = tuple(DEFAULT_ERP_WINDOWS)
    include_time_series: bool = False
    morlet_bands: Optional[Dict[str, np.ndarray]] = None
    n_pca_components: Optional[int] = None


@dataclass
class EvaluationConfig:
    """Configuration for classifier evaluation."""

    classifier: str = "svm"  # options: svm, logreg, lda
    n_splits: int = 5
    n_permutations: int = 1000


class PipelineError(RuntimeError):
    """Raised when the pipeline encounters a fatal error."""


def _mad(arr: np.ndarray) -> float:
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    if mad <= 1e-12:
        mad = np.std(arr)
    return max(mad, 1e-12)


def detect_bad_channels(raw: mne.io.BaseRaw, z_threshold: float = 5.0) -> List[str]:
    """Detect bad channels using peak-to-peak z-scores.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object to analyze.
    z_threshold : float
        Z-score threshold for marking a channel as bad.

    Returns
    -------
    list of str
        Names of channels flagged as bad.
    """

    picks = mne.pick_types(raw.info, meg=True, eeg=True, exclude=())
    data = raw.get_data(picks=picks)
    ptp = np.ptp(data, axis=1)
    mad = _mad(ptp)
    zscores = (ptp - np.median(ptp)) / (mad * 1.4826)
    bads = [raw.ch_names[p] for p, z in zip(picks, zscores) if z > z_threshold]
    raw.info["bads"] = sorted(set(raw.info.get("bads", []) + bads))
    return bads


def run_ica(raw: mne.io.BaseRaw, random_state: int = 97) -> mne.preprocessing.ICA:
    """Run ICA to identify and exclude EOG/ECG components."""

    ica = mne.preprocessing.ICA(n_components=0.99, random_state=random_state, max_iter="auto")
    ica.fit(raw.copy())

    eog_indices, _ = ica.find_bads_eog(raw)
    ecg_indices, _ = ica.find_bads_ecg(raw)
    ica.exclude = list(set(eog_indices + ecg_indices))
    return ica


def _resolve_runs(cfg: PreprocessingConfig) -> List[str]:
    if cfg.run:
        return [cfg.run]
    runs = get_entity_vals(
        cfg.bids_root,
        entity_key="run",
        task=cfg.task,
        subject=cfg.subject,
        session=cfg.session,
        datatype=cfg.datatype,
    )
    if not runs:
        raise PipelineError("No runs found for the given subject/task/session.")
    return sorted(runs)


def _resolve_line_noise(raw: mne.io.BaseRaw, notch_freqs: Iterable[float]) -> List[float]:
    line_freq = raw.info.get("line_freq")
    if line_freq:
        return [float(line_freq)]
    return list(notch_freqs)


def _load_concat_raw(cfg: PreprocessingConfig) -> mne.io.BaseRaw:
    raws: List[mne.io.BaseRaw] = []
    runs = _resolve_runs(cfg)
    for run in runs:
        bids_path = BIDSPath(
            root=cfg.bids_root,
            subject=cfg.subject,
            session=cfg.session,
            task=cfg.task,
            run=run,
            datatype=cfg.datatype,
        )
        raw = read_raw_bids(bids_path=bids_path, verbose="error")
        raw.load_data()

        notch = _resolve_line_noise(raw, cfg.notch_freqs)
        if notch:
            raw.notch_filter(notch)
        raw.filter(cfg.l_freq, cfg.h_freq, fir_design="firwin")
        raws.append(raw)

    if len(raws) == 1:
        return raws[0]
    mne.channels.equalize_channels(raws)
    return mne.concatenate_raws(raws)


def _relabel_events(events: np.ndarray, event_id: Dict[str, int], label_mode: str) -> Tuple[np.ndarray, Dict[str, int]]:
    trigger_to_category: Dict[int, str] = {}
    for category, triggers in DEFAULT_TRIGGER_SETS.items():
        for trig in triggers:
            trigger_to_category[trig] = category

    code_to_desc = {v: k for k, v in event_id.items()}

    def _get_category(trig_code: int) -> Optional[str]:
        desc = code_to_desc.get(trig_code)
        if desc:
            base_desc = desc.split("/")[0]
            if base_desc in DEFAULT_TRIGGER_SETS:
                return base_desc
        return trigger_to_category.get(trig_code)

    relabeled: List[List[int]] = []
    for onset, dur, trig in events:
        category = _get_category(int(trig))
        if not category:
            continue
        if label_mode == "binary":
            label = "face" if category in {"Famous", "Unfamiliar"} else "scrambled"
        elif label_mode == "triple":
            label = category.lower()
        else:
            raise PipelineError(f"Unsupported label_mode '{label_mode}'. Choose from {LABEL_MODES}.")
        relabeled.append([onset, int(dur), label])

    if not relabeled:
        raise PipelineError("No usable events found after relabeling. Check trigger mapping.")

    labels = [row[2] for row in relabeled]
    label_to_code = {label: idx + 1 for idx, label in enumerate(sorted(set(labels)))}
    new_events = np.array([[row[0], row[1], label_to_code[row[2]]] for row in relabeled], dtype=int)
    return new_events, label_to_code


def load_and_preprocess(cfg: PreprocessingConfig) -> mne.Epochs:
    """Load ds000117 data via BIDS and return cleaned epochs."""

    raw = _load_concat_raw(cfg)

    bads = detect_bad_channels(raw, cfg.bad_z_threshold)
    if bads:
        mne.utils.logger.info("Detected bad channels: %s", ", ".join(bads))
    raw.interpolate_bads(reset_bads=True)

    #ica = run_ica(raw, cfg.random_state)
    #raw = ica.apply(raw.copy())

    events, event_id = mne.events_from_annotations(raw, event_id=cfg.event_id)
    events, label_map = _relabel_events(events, event_id, cfg.label_mode)
    if len(label_map) < 2:
        raise PipelineError("At least two classes are required for classification.")

    epochs = mne.Epochs(
        raw,
        events,
        event_id=label_map,
        tmin=cfg.tmin,
        tmax=cfg.tmax,
        baseline=cfg.baseline,
        reject=cfg.reject,
        preload=True,
    )
    epochs.drop_bad()
    return epochs


def _time_window_indices(times: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
    return np.where((times >= window[0]) & (times <= window[1]))[0]


def extract_features(epochs: mne.Epochs, cfg: FeatureConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and labels from epochs.

    Returns
    -------
    X : np.ndarray
        Feature matrix (n_epochs, n_features).
    y : np.ndarray
        Labels for each epoch.
    """

    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    times = epochs.times
    features: List[np.ndarray] = []

    # ERP window means
    if cfg.erp_windows:
        erp_feats = []
        for window in cfg.erp_windows:
            idx = _time_window_indices(times, window)
            if idx.size == 0:
                raise PipelineError(f"No samples found for ERP window {window}.")
            erp_feats.append(data[:, :, idx].mean(axis=2))
        features.append(np.stack(erp_feats, axis=2).reshape(len(data), -1))

    # Flattened time series
    if cfg.include_time_series:
        features.append(data.reshape(len(data), -1))

    # Morlet power
    if cfg.morlet_bands:
        power_features = []
        for band_name, freqs in cfg.morlet_bands.items():
            tfr = mne.time_frequency.tfr_morlet(
                epochs,
                freqs=freqs,
                n_cycles=freqs / 2.0,
                return_itc=False,
                average=False,
                decim=1,
            )
            power_features.append(np.log1p(tfr.data.mean(axis=3)).reshape(len(data), -1))
        features.append(np.concatenate(power_features, axis=1))

    if not features:
        raise PipelineError("No features were extracted. Enable at least one feature type.")

    X = np.concatenate(features, axis=1)
    inv_event_id = {v: k for k, v in epochs.event_id.items()}
    y = np.array([inv_event_id[e] for e in epochs.events[:, 2]])
    return X, y


def build_classifier(name: str) -> Pipeline:
    """Create a sklearn pipeline with scaling, optional PCA, and classifier."""

    name = name.lower()
    if name == "svm":
        clf = LinearSVC(class_weight="balanced")
    elif name == "logreg":
        clf = LogisticRegression(max_iter=1000, n_jobs=1, class_weight="balanced")
    elif name == "lda":
        clf = LinearDiscriminantAnalysis()
    else:
        raise ValueError(f"Unknown classifier '{name}'")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def evaluate_model(X: np.ndarray, y: np.ndarray, feature_cfg: FeatureConfig, eval_cfg: EvaluationConfig):
    """Run cross-validation and permutation testing."""

    steps = [("scaler", StandardScaler())]
    if feature_cfg.n_pca_components:
        steps.append(("pca", PCA(n_components=feature_cfg.n_pca_components, random_state=0)))
    steps.append(("clf", build_classifier(eval_cfg.classifier).steps[-1][1]))
    model = Pipeline(steps)

    cv = StratifiedKFold(n_splits=eval_cfg.n_splits, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, cv=cv, n_jobs=1)
    perm_score, perm_scores, pvalue = permutation_test_score(
        model, X, y, cv=cv, n_permutations=eval_cfg.n_permutations, n_jobs=1, random_state=0
    )
    return {
        "cv_scores": scores,
        "cv_mean": float(np.mean(scores)),
        "cv_std": float(np.std(scores)),
        "perm_score": float(perm_score),
        "perm_pvalue": float(pvalue),
        "perm_scores": perm_scores,
    }


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
    erp_windows = DEFAULT_ERP_WINDOWS
    if args.erp:
        erp_windows = []
        for window in args.erp:
            try:
                start, end = map(float, window.split(":"))
            except ValueError as exc:
                raise PipelineError(f"Invalid ERP window specification: {window}") from exc
            erp_windows.append((start, end))

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
        erp_windows=tuple(erp_windows),
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
