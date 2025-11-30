"""
Preprocessing and epoching helpers for the ds000117 pipeline.

These utilities load ds000117 data from BIDS via ``mne_bids``, apply
filtering and bad-channel handling, run ICA to suppress EOG/ECG
components, and return cleaned epochs ready for feature extraction.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mne
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids
import numpy as np

DEFAULT_NOTCH = (50.0, 60.0)
DEFAULT_FILTER = (0.1, 40.0)
DEFAULT_BASELINE = (None, 0.0)
DEFAULT_EPOCH_WINDOW = (-0.2, 0.8)
DEFAULT_REJECT = dict(grad=4000e-13, mag=4e-12, eeg=150e-6, eog=250e-6)
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
    reject: Optional[Dict[str, float]] = DEFAULT_REJECT
    event_id: Optional[Dict[str, int]] = None
    bad_z_threshold: float = 5.0
    random_state: int = 97
    label_mode: str = "binary"  # binary (face vs scrambled) or triple (famous/unfamiliar/scrambled)


class PipelineError(RuntimeError):
    """Raised when the pipeline encounters a fatal error."""


def _mad(arr: np.ndarray) -> float:
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    if mad <= 1e-12:
        mad = np.std(arr)
    return max(mad, 1e-12)


def detect_bad_channels(raw: mne.io.BaseRaw, z_threshold: float = 5.0) -> List[str]:
    """Detect bad channels using peak-to-peak z-scores."""

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

    ica = run_ica(raw, cfg.random_state)
    raw = ica.apply(raw.copy())

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
