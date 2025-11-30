"""
Feature extraction helpers for ds000117 epochs.

Provides ERP window means, flattened time-series, and Morlet band-power
features used by the decoding pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import mne
import numpy as np

from .preprocess import PipelineError

DEFAULT_ERP_WINDOWS = [(0.05, 0.15), (0.15, 0.3), (0.3, 0.6)]
DEFAULT_MORLET_BANDS = {
    "theta": np.arange(4.0, 8.1, 1.0),
    "alpha": np.arange(8.0, 13.0, 1.0),
    "beta": np.arange(13.0, 30.0, 2.0),
}


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    erp_windows: Sequence[Tuple[float, float]] = tuple(DEFAULT_ERP_WINDOWS)
    include_time_series: bool = False
    morlet_bands: Optional[Dict[str, np.ndarray]] = None
    n_pca_components: Optional[int] = None


def _time_window_indices(times: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
    return np.where((times >= window[0]) & (times <= window[1]))[0]


def extract_features(epochs: mne.Epochs, cfg: FeatureConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and labels from epochs."""

    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    times = epochs.times
    features = []

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
        for _, freqs in cfg.morlet_bands.items():
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
