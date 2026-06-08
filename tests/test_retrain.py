"""Tests for the retrain orchestration: promotion gate + champion/challenger."""

from __future__ import annotations

import pandas as pd

from config import MLFLOW_AUC_GATE, TrainConfig
from data.synthetic import generate_transactions
from pipelines.retrain import (
    decide_promotion,
    run_drift_triggered_retrain,
    run_local_retrain,
)


# --- decide_promotion: the absolute gate -----------------------------------
def test_below_gate_is_blocked():
    d = decide_promotion(0.85, champion_auc=None, gate=0.90)
    assert not d.promote
    assert "below gate" in d.reason


def test_above_gate_no_champion_promotes():
    d = decide_promotion(0.95, champion_auc=None, gate=0.90)
    assert d.promote


# --- decide_promotion: champion/challenger ---------------------------------
def test_challenger_must_beat_champion():
    # clears gate but does not beat the incumbent
    d = decide_promotion(0.95, champion_auc=0.96, gate=0.90)
    assert not d.promote
    assert "does not beat champion" in d.reason


def test_challenger_beats_champion_promotes():
    d = decide_promotion(0.97, champion_auc=0.95, gate=0.90)
    assert d.promote


def test_min_improvement_margin_enforced():
    # 0.955 beats 0.95 but not by the required 0.01 margin
    assert not decide_promotion(0.955, champion_auc=0.95, gate=0.90, min_improvement=0.01).promote
    assert decide_promotion(0.965, champion_auc=0.95, gate=0.90, min_improvement=0.01).promote


def test_default_gate_constant_used():
    d = decide_promotion(MLFLOW_AUC_GATE - 0.01)
    assert not d.promote


# --- end-to-end local retrain (fast) ---------------------------------------
def test_local_retrain_runs_and_decides():
    txns = generate_transactions(n_users=400, seed=5)
    result = run_local_retrain(
        txns, champion_auc=None, cfg=TrainConfig(max_epochs=3), package=False
    )
    assert 0.0 <= result.test_auc <= 1.0
    # decision is internally consistent with the gate
    expected = result.test_auc >= MLFLOW_AUC_GATE
    assert result.decision.promote == expected
    assert result.artifact_path is None  # package=False


def test_drift_triggered_skips_when_no_drift():
    # identical reference/current -> no drift -> no retrain
    ref = pd.DataFrame({f: [0.0, 1.0, 2.0, 3.0] * 50 for f in
                        ["txn_velocity_1h", "amount_zscore", "new_device_flag"]})
    out = run_drift_triggered_retrain(ref, ref.copy())
    assert out is None
