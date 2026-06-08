"""LightGBM baseline + second-layer confirmation model.

Establishes the "strong baseline" the LSTM must beat by >=4 AUC points.
Operates on the *last* transaction's features per user (no sequence), so it
is a fair, non-sequential reference point. SHAP explainability is wired in
for the explainability phase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS, LABEL_COL

try:
    import lightgbm as lgb
except ImportError:  # keep module importable without the dep installed
    lgb = None


def last_txn_table(features_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Collapse each user's history to their final transaction's features.

    Returns (X, y) where y is the sequence-level fraud label (fraud anywhere
    in the user's history), matching the LSTM's labelling so the comparison
    is apples-to-apples.
    """
    df = features_df.sort_values(["user_id", "timestamp"])
    last = df.groupby("user_id", sort=False).tail(1)
    y_seq = df.groupby("user_id", sort=False)[LABEL_COL].max().to_numpy(dtype=np.float32)
    X = last[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    return X, y_seq


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict | None = None,
):
    """Train a LightGBM classifier with early stopping on validation AUC."""
    if lgb is None:
        raise ImportError("lightgbm is not installed; `pip install lightgbm`")

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    default = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "scale_pos_weight": neg / max(pos, 1.0),
        "verbose": -1,
    }
    params = {**default, **(params or {})}

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLUMNS)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    return model


def shap_summary(model, X: np.ndarray):
    """Return SHAP values for `X` (top-feature attribution per prediction)."""
    import shap

    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X)
