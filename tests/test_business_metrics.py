"""Tests for the dollar-outcome business metrics."""

from __future__ import annotations

import json

import numpy as np

from training.business_metrics import (
    business_metrics,
    net_savings_by_threshold,
    save_business_metrics_json,
)


def test_perfect_predictions_capture_all_fraud_no_fp_cost():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    fraud_amount = np.array([100.0, 0.0, 250.0, 0.0])
    m = business_metrics(y_true, y_pred, fraud_amount, fp_unit_cost=50.0)
    assert m.dollars_fraud_caught == 350.0
    assert m.dollars_fraud_missed == 0.0
    assert m.dollars_fp_cost == 0.0
    assert m.net_savings == 350.0


def test_false_negative_counts_as_missed():
    y_true = np.array([1, 1])
    y_pred = np.array([1, 0])      # second fraud missed
    fraud_amount = np.array([100.0, 400.0])
    m = business_metrics(y_true, y_pred, fraud_amount)
    assert m.dollars_fraud_caught == 100.0
    assert m.dollars_fraud_missed == 400.0
    assert m.n_false_negative == 1


def test_false_positive_cost_applied():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 0])   # two false positives
    fraud_amount = np.zeros(3)
    m = business_metrics(y_true, y_pred, fraud_amount, fp_unit_cost=30.0)
    assert m.n_false_positive == 2
    assert m.dollars_fp_cost == 60.0
    assert m.net_savings == -60.0


def test_net_savings_curve_shape():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_score = rng.random(200)
    fraud_amount = np.where(y_true == 1, rng.uniform(50, 500, 200), 0.0)
    thr, savings = net_savings_by_threshold(y_true, y_score, fraud_amount)
    assert thr.shape == savings.shape
    assert len(thr) == 19


def test_save_json_roundtrip(tmp_path):
    m = business_metrics(np.array([1, 0]), np.array([1, 0]), np.array([100.0, 0.0]))
    p = save_business_metrics_json(m, tmp_path / "business_metrics.json")
    loaded = json.loads(p.read_text())
    assert loaded["net_savings"] == 100.0
    assert loaded["fp_unit_cost"] == m.fp_unit_cost
