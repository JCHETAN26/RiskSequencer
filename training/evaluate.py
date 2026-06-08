"""Evaluation metrics and threshold tuning for RiskSequencer.

Centralises the numbers the spec gates on: AUC-ROC, precision at a fixed
FPR, and the operating threshold that hits FPR <= 5% while maximising
precision. Pure-numpy/sklearn so it works for both the LSTM and LightGBM.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvalResult:
    auc_roc: float
    avg_precision: float
    threshold: float
    precision_at_threshold: float
    recall_at_threshold: float
    fpr_at_threshold: float

    def as_dict(self) -> dict:
        return asdict(self)


def tune_threshold(
    y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = 0.05
) -> float:
    """Pick the threshold with FPR <= max_fpr that maximises precision.

    Falls back to the threshold with the smallest FPR if none satisfy the
    constraint.
    """
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    feasible = np.where(fpr <= max_fpr)[0]
    if len(feasible) == 0:
        return float(roc_thresholds[np.argmin(fpr)])

    # Among feasible ROC points, choose the one giving best precision.
    best_thr, best_prec = roc_thresholds[feasible[0]], -1.0
    for idx in feasible:
        thr = roc_thresholds[idx]
        pred = (y_score >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        if prec > best_prec:
            best_prec, best_thr = prec, thr
    return float(best_thr)


def evaluate(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float | None = None
) -> EvalResult:
    """Compute the full evaluation bundle at a given (or tuned) threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    auc = float(roc_auc_score(y_true, y_score))
    ap = float(average_precision_score(y_true, y_score))
    thr = tune_threshold(y_true, y_score) if threshold is None else threshold

    pred = (y_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return EvalResult(
        auc_roc=auc,
        avg_precision=ap,
        threshold=float(thr),
        precision_at_threshold=float(precision),
        recall_at_threshold=float(recall),
        fpr_at_threshold=float(fpr),
    )


def pr_curve(y_true: np.ndarray, y_score: np.ndarray):
    """Convenience wrapper returning (precision, recall, thresholds)."""
    return precision_recall_curve(y_true, y_score)
