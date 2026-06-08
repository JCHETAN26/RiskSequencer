"""Translate model predictions into dollars (build-plan Phase 3.3).

Fraud detection trades two costs against each other: missed fraud (false
negatives) loses the fraud amount; a false positive blocks a legitimate
customer and risks churn. These functions turn a threshold into a net-savings
number so the operating point can be chosen on business value, not just AUC.

All functions are pure and array-based so they're unit-testable and reusable
from the error-analysis notebook, the eval step, and MLflow logging.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Default estimated cost of one false positive (blocked legit customer):
# a blended figure for customer churn + manual-review handling. Tune per org.
DEFAULT_FP_UNIT_COST = 50.0


@dataclass
class BusinessMetrics:
    dollars_fraud_caught: float      # TP: fraud $ we flagged
    dollars_fraud_missed: float      # FN: fraud $ we let through
    dollars_fp_cost: float           # FP: churn/handling cost of false alarms
    net_savings: float               # caught - fp_cost
    n_true_positive: int
    n_false_negative: int
    n_false_positive: int
    n_true_negative: int
    fp_unit_cost: float

    def as_dict(self) -> dict:
        return asdict(self)


def business_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fraud_amount: np.ndarray,
    fp_unit_cost: float = DEFAULT_FP_UNIT_COST,
) -> BusinessMetrics:
    """Compute dollar outcomes for a set of predictions.

    Parameters
    ----------
    y_true : (N,) 0/1 ground-truth sequence labels.
    y_pred : (N,) 0/1 predictions at the chosen threshold.
    fraud_amount : (N,) dollars at risk per sequence (sum of fraudulent
        transaction amounts in that user's window; 0 for legit sequences).
    fp_unit_cost : estimated cost of a single false positive.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    fraud_amount = np.asarray(fraud_amount, dtype=float)

    tp = (y_true == 1) & (y_pred == 1)
    fn = (y_true == 1) & (y_pred == 0)
    fp = (y_true == 0) & (y_pred == 1)
    tn = (y_true == 0) & (y_pred == 0)

    caught = float(fraud_amount[tp].sum())
    missed = float(fraud_amount[fn].sum())
    fp_cost = float(int(fp.sum()) * fp_unit_cost)

    return BusinessMetrics(
        dollars_fraud_caught=caught,
        dollars_fraud_missed=missed,
        dollars_fp_cost=fp_cost,
        net_savings=caught - fp_cost,
        n_true_positive=int(tp.sum()),
        n_false_negative=int(fn.sum()),
        n_false_positive=int(fp.sum()),
        n_true_negative=int(tn.sum()),
        fp_unit_cost=float(fp_unit_cost),
    )


def net_savings_by_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fraud_amount: np.ndarray,
    thresholds: np.ndarray | None = None,
    fp_unit_cost: float = DEFAULT_FP_UNIT_COST,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep thresholds and return (thresholds, net_savings) for a profit curve.

    Useful for picking the *profit-maximising* operating point, which may differ
    from the FPR≤5% point used for the deployment threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    savings = np.array(
        [
            business_metrics(
                y_true, (np.asarray(y_score) >= t).astype(int), fraud_amount, fp_unit_cost
            ).net_savings
            for t in thresholds
        ]
    )
    return np.asarray(thresholds), savings


def save_business_metrics_json(metrics: BusinessMetrics, path: str | Path) -> Path:
    """Write metrics to JSON (the artifact logged to MLflow per the plan)."""
    path = Path(path)
    path.write_text(json.dumps(metrics.as_dict(), indent=2))
    return path
