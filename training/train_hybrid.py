"""Train and evaluate the hybrid LightGBM + LSTM stacking ensemble.

Pipeline (one user-level prediction per user, label = fraud anywhere in window):

  1. LightGBM on native tabular features (per transaction)  -> aggregate to user (max)
  2. LSTM + attention on behavioral sequences (per user)
  3. Stacking meta-learner over [lgbm_user, lstm_user], fit on validation, eval on test

IMPORTANT — this module trains torch *and* LightGBM in one process, which
segfaults on macOS because both bundle OpenMP. The guard below MUST run before
either library is imported, so these two lines stay at the very top of the file.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from config import LABEL_COL, TrainConfig  # noqa: E402
from data.sequence_builder import build_sequences, fit_scaler, time_based_split  # noqa: E402
from features.feature_pipeline import build_features  # noqa: E402
from models.hybrid import aggregate_to_users, align_signals, fit_stacker  # noqa: E402
from training.evaluate import evaluate  # noqa: E402
from training.train import _scores, run_training  # noqa: E402


@dataclass
class HybridResult:
    lgbm_auc: float       # LightGBM alone (user-level)
    lstm_auc: float       # LSTM alone (user-level)
    hybrid_auc: float     # stacked ensemble
    stacker_coef: tuple[float, float]


def _split_native_by_time(native: pd.DataFrame, t_train, t_val):
    ts = native["timestamp"]
    return (
        native[ts <= t_train],
        native[(ts > t_train) & (ts <= t_val)],
        native[ts > t_val],
    )


def _train_lgbm_native(
    tr: pd.DataFrame, va: pd.DataFrame, feat_cols: list[str], cat_cols: list[str]
):
    cat = [c for c in cat_cols if c in feat_cols]
    dtrain = lgb.Dataset(tr[feat_cols], tr["is_fraud"], categorical_feature=cat or "auto")
    dval = lgb.Dataset(va[feat_cols], va["is_fraud"], reference=dtrain)
    pos = float(tr["is_fraud"].sum())
    neg = float(len(tr) - pos)
    params = {
        "objective": "binary", "metric": "auc", "learning_rate": 0.03,
        "num_leaves": 256, "min_child_samples": 50,
        "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l1": 0.1, "lambda_l2": 1.0,
        "scale_pos_weight": neg / max(pos, 1.0),
        "max_cat_to_onehot": 8, "num_threads": 1, "verbose": -1,
    }
    return lgb.train(
        params, dtrain, num_boost_round=1500, valid_sets=[dval],
        callbacks=[lgb.early_stopping(60), lgb.log_evaluation(0)],
    )


def run_hybrid(
    canonical: pd.DataFrame,
    native: pd.DataFrame,
    cfg: TrainConfig | None = None,
    max_epochs: int = 20,
) -> HybridResult:
    """Train both layers + the stacker and report user-level AUCs."""
    cfg = cfg or TrainConfig(max_epochs=max_epochs)

    # --- shared time-based split (same cut points for both views) ----------
    ts = canonical["timestamp"]
    t_train, t_val = ts.quantile(0.7), ts.quantile(0.85)

    feats = build_features(canonical)
    tr_f, va_f, te_f = time_based_split(feats)
    scaler = fit_scaler(tr_f)
    train_ds = build_sequences(tr_f, scaler)
    val_ds = build_sequences(va_f, scaler)
    test_ds = build_sequences(te_f, scaler)

    from data.ieee_cis import categorical_columns, feature_columns

    feat_cols = feature_columns(native)
    cat_cols = categorical_columns(native)
    tr_n, va_n, te_n = _split_native_by_time(native, t_train, t_val)

    # --- LightGBM layer ----------------------------------------------------
    lgbm = _train_lgbm_native(tr_n, va_n, feat_cols, cat_cols)

    def lgbm_user(split_native):
        scores = lgbm.predict(split_native[feat_cols])
        return aggregate_to_users(split_native["user_id"].to_numpy(), scores, how="max")

    lgbm_val_u = lgbm_user(va_n)
    lgbm_test_u = lgbm_user(te_n)

    # --- LSTM layer --------------------------------------------------------
    out = run_training(train_ds, val_ds, cfg)
    lstm_val = _scores(out.model, val_ds, device="cpu")
    lstm_test = _scores(out.model, test_ds, device="cpu")

    # --- align + stack (fit on val, eval on test) --------------------------
    lg_val, ls_val, y_val = align_signals(lgbm_val_u, val_ds.user_ids, lstm_val, val_ds.y)
    lg_test, ls_test, y_test = align_signals(lgbm_test_u, test_ds.user_ids, lstm_test, test_ds.y)

    stacker = fit_stacker(lg_val, ls_val, y_val)
    hybrid_test = stacker.predict_proba(lg_test, ls_test)

    return HybridResult(
        lgbm_auc=evaluate(y_test, lg_test).auc_roc,
        lstm_auc=evaluate(y_test, ls_test).auc_roc,
        hybrid_auc=evaluate(y_test, hybrid_test).auc_roc,
        stacker_coef=(float(stacker.coef_[0]), float(stacker.coef_[1])),
    )


def main() -> None:
    import argparse

    from data.ieee_cis import load_hybrid

    parser = argparse.ArgumentParser(description="Train the hybrid LightGBM+LSTM ensemble.")
    parser.add_argument("--transaction-csv", type=Path, required=True)
    parser.add_argument("--identity-csv", type=Path, default=None)
    parser.add_argument("--max-epochs", type=int, default=20)
    args = parser.parse_args()

    canonical, native = load_hybrid(args.transaction_csv, args.identity_csv)
    res = run_hybrid(canonical, native, max_epochs=args.max_epochs)
    print("\n==== HYBRID RESULTS (held-out test, user-level AUC) ====")
    print(f"  LightGBM alone : {res.lgbm_auc:.4f}")
    print(f"  LSTM alone     : {res.lstm_auc:.4f}")
    print(f"  HYBRID (stack) : {res.hybrid_auc:.4f}")
    print(f"  stacker coef [lgbm, lstm] = {res.stacker_coef}")
    print(f"  TARGET >= 0.94 : {'PASS' if res.hybrid_auc >= 0.94 else 'MISS'}")


if __name__ == "__main__":
    main()
