"""Fast smoke test for the hyperparameter search (random-search path)."""

from __future__ import annotations

from data.sequence_builder import build_sequences, fit_scaler, time_based_split
from data.synthetic import generate_transactions
from features.feature_pipeline import build_features
from training.hyperparameter_search import SEARCH_SPACE, search


def _tiny_datasets():
    raw = generate_transactions(n_users=120, seed=11)
    feats = build_features(raw)
    tr, va, _ = time_based_split(feats)
    scaler = fit_scaler(tr)
    return build_sequences(tr, scaler), build_sequences(va, scaler)


def test_search_returns_valid_result():
    train_ds, val_ds = _tiny_datasets()
    result = search(
        train_ds,
        val_ds,
        n_trials=2,
        max_epochs=1,
        use_optuna=False,   # exercise the deterministic fallback
        log_mlflow=False,
        seed=0,
    )
    assert set(result.best_params) == set(SEARCH_SPACE)
    assert 0.0 <= result.best_val_auc <= 1.0
    assert len(result.trials) == 2
    # best recorded trial AUC matches the reported best
    assert result.best_val_auc == max(t["val_auc"] for t in result.trials)
