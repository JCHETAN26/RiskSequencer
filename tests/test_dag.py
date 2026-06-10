"""Tests for the retrain DAG's task logic, runnable without Airflow.

These prove the automated loop (Evidently drift -> retrain -> promotion decision)
executes end-to-end with no scheduler and no manual intervention.
"""

from __future__ import annotations

from data.synthetic import generate_transactions
from pipelines.retrain_dag import check_drift_task, run_pipeline_locally


def test_check_drift_task_detects_injected_drift():
    # the default reference/current providers inject drift, so the gate opens
    assert check_drift_task() is True


def test_pipeline_runs_end_to_end_and_decides():
    out = run_pipeline_locally(
        transactions=generate_transactions(n_users=500, seed=3), max_epochs=2
    )
    assert out["retrained"] is True
    assert out["engine"] in ("evidently", "fallback")
    assert out["breached"]                     # at least one feature breached PSI
    assert 0.0 <= out["test_auc"] <= 1.0
    assert isinstance(out["promote"], bool)
    # promotion decision is consistent with the 0.90 gate
    from config import MLFLOW_AUC_GATE
    assert out["promote"] == (out["test_auc"] >= MLFLOW_AUC_GATE)
