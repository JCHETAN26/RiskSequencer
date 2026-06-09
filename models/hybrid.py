"""Hybrid LightGBM + LSTM stacking ensemble (dataset-agnostic core).

Combines two complementary fraud signals into one user-level probability:

* **LightGBM** scores transactions on rich tabular features; aggregated to the
  user with `max` (a user is fraud if *any* of their transactions is).
* **LSTM + attention** scores the user's behavioral sequence directly.

A small logistic-regression **meta-learner** (stacking) is fit on the two
signals over a held-out split, so the combination weight is learned rather than
guessed. This module deliberately imports only numpy/sklearn — the heavy torch
and lightgbm training lives in `training/train_hybrid.py` (which must apply the
OpenMP guard before importing both).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def aggregate_to_users(
    user_ids: np.ndarray,
    txn_scores: np.ndarray,
    how: str = "max",
) -> pd.Series:
    """Reduce per-transaction scores to one score per user.

    Returns a Series indexed by user_id. `max` matches the user-level label
    ("fraud anywhere in the window"); `mean` is available as an alternative.
    """
    s = pd.Series(np.asarray(txn_scores, dtype=float), index=np.asarray(user_ids))
    grouped = s.groupby(level=0)
    return grouped.max() if how == "max" else grouped.mean()


def align_signals(
    lgbm_user_scores: pd.Series,
    lstm_user_ids: np.ndarray,
    lstm_scores: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align the two models' user-level scores onto the LSTM's user ordering.

    The LSTM defines the row set (one sequence per user). The LightGBM signal is
    looked up per user; users with no LightGBM score (shouldn't happen, but be
    safe) get the global median. Returns (lgbm, lstm, y) as parallel arrays.
    """
    lstm_user_ids = np.asarray(lstm_user_ids)
    fill = float(lgbm_user_scores.median())
    lgbm_aligned = lgbm_user_scores.reindex(lstm_user_ids).fillna(fill).to_numpy()
    return lgbm_aligned, np.asarray(lstm_scores, dtype=float), np.asarray(y, dtype=float)


@dataclass
class StackedEnsemble:
    """A fitted stacking meta-learner over [lgbm_user_score, lstm_score]."""

    meta: LogisticRegression
    coef_: np.ndarray
    intercept_: float

    def predict_proba(self, lgbm_scores: np.ndarray, lstm_scores: np.ndarray) -> np.ndarray:
        X = np.column_stack([np.asarray(lgbm_scores, float), np.asarray(lstm_scores, float)])
        return self.meta.predict_proba(X)[:, 1]


def fit_stacker(
    lgbm_scores: np.ndarray,
    lstm_scores: np.ndarray,
    y: np.ndarray,
) -> StackedEnsemble:
    """Fit the logistic-regression meta-learner on held-out (validation) signals.

    Fitting on a split the base models did *not* train on is what makes this
    proper stacking rather than leak-prone in-sample blending.
    """
    X = np.column_stack([np.asarray(lgbm_scores, float), np.asarray(lstm_scores, float)])
    meta = LogisticRegression(max_iter=1000)
    meta.fit(X, np.asarray(y, dtype=int))
    return StackedEnsemble(meta=meta, coef_=meta.coef_[0], intercept_=float(meta.intercept_[0]))
