"""
Classifier construction and evaluation helpers for the ds000117 pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


@dataclass
class EvaluationConfig:
    """Configuration for classifier evaluation."""

    classifier: str = "svm"  # options: svm, logreg, lda
    n_splits: int = 5
    n_permutations: int = 1000


def build_classifier(name: str) -> Pipeline:
    """Create a sklearn pipeline with scaling and classifier."""

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


def evaluate_model(X: np.ndarray, y: np.ndarray, feature_cfg, eval_cfg: EvaluationConfig):
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
