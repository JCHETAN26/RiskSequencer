"""Tests for the hybrid stacking core (numpy/sklearn only — no torch/lightgbm)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from models.hybrid import aggregate_to_users, align_signals, fit_stacker


def test_aggregate_to_users_max():
    uids = np.array([1, 1, 2, 2, 2])
    scores = np.array([0.1, 0.9, 0.3, 0.2, 0.25])
    agg = aggregate_to_users(uids, scores, how="max")
    assert agg.loc[1] == 0.9
    assert agg.loc[2] == 0.3


def test_align_signals_reorders_to_lstm_users():
    lgbm_u = aggregate_to_users(np.array([10, 20, 30]), np.array([0.2, 0.8, 0.5]))
    # LSTM presents users in a different order
    lstm_ids = np.array([30, 10, 20])
    lstm_scores = np.array([0.4, 0.1, 0.9])
    y = np.array([1, 0, 1])
    lg, ls, yy = align_signals(lgbm_u, lstm_ids, lstm_scores, y)
    assert list(lg) == [0.5, 0.2, 0.8]   # reordered to match lstm_ids
    assert list(ls) == [0.4, 0.1, 0.9]
    assert list(yy) == [1.0, 0.0, 1.0]


def test_align_fills_missing_user_with_median():
    lgbm_u = aggregate_to_users(np.array([1, 2]), np.array([0.2, 0.6]))
    lg, _, _ = align_signals(lgbm_u, np.array([1, 2, 99]), np.array([0.1, 0.1, 0.1]), np.array([0, 0, 1]))
    assert lg[2] == np.median([0.2, 0.6])  # user 99 absent -> median fill


def test_stacker_beats_each_weak_signal():
    # Two complementary, individually-weak signals; stacking should beat both.
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 2, n)
    # signal A informative on even-indexed, noisy elsewhere; B the opposite
    a = np.where(np.arange(n) % 2 == 0, y + rng.normal(0, 1.0, n), rng.normal(0, 1, n))
    b = np.where(np.arange(n) % 2 == 1, y + rng.normal(0, 1.0, n), rng.normal(0, 1, n))
    half = n // 2
    stacker = fit_stacker(a[:half], b[:half], y[:half])
    pred = stacker.predict_proba(a[half:], b[half:])
    auc_hybrid = roc_auc_score(y[half:], pred)
    auc_a = roc_auc_score(y[half:], a[half:])
    auc_b = roc_auc_score(y[half:], b[half:])
    assert auc_hybrid >= max(auc_a, auc_b) - 1e-6
