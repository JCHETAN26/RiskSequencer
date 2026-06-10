"""Airflow DAG: weekly (and drift-triggered) retraining for RiskSequencer.

Flow: check_drift (Evidently PSI) -> [short-circuit if no drift] -> retrain ->
promotion decision -> notify. Runs @weekly; retraining fires only when PSI
exceeds 0.20 on a monitored feature. Each task has 3 retries / 10-min delay and
a Slack notification on failure.

The task callables are REAL and runnable — they default to synthetic data so the
whole DAG executes end-to-end with `airflow dags test` (no S3/AWS needed). The
two `_default_*` providers are the documented swap points for production: replace
them with S3 loaders for the reference window, the last-24h inference logs, and
the fresh labelled training data. Everything else stays the same.

The module is import-guarded: the DAG object is only built when Airflow is
installed, so the pure task functions remain importable/testable without it.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from config import MLFLOW_AUC_GATE, MONITORED_FEATURES, REGISTERED_MODEL_NAME
from monitoring.slack_alerts import (
    alert_drift,
    alert_gate_failed,
    alert_retrain_complete,
    alert_task_failure,
)

# --- Data providers (swap these for S3 loaders in production) ---------------


def _default_reference_current() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reference window vs the recent window, over the monitored features.

    Default: synthetic reference + a drifted current window so the DAG runs a
    full cycle out of the box. PRODUCTION: load the saved reference dataset and
    the last-24h inference logs from s3://risksequencer-logs/.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    ref = pd.DataFrame({f: rng.normal(0, 1, 4000) for f in MONITORED_FEATURES})
    cur = pd.DataFrame({f: rng.normal(0, 1, 4000) for f in MONITORED_FEATURES})
    cur[MONITORED_FEATURES[0]] = rng.normal(4, 1.2, 4000)  # inject drift
    return ref, cur


def _default_training_transactions() -> pd.DataFrame:
    """Fresh labelled data to retrain on.

    Default: synthetic transactions. PRODUCTION: pull the latest labelled
    window from s3://risksequencer-data/.
    """
    from data.synthetic import generate_transactions

    return generate_transactions(n_users=2000, seed=7)


# --- Task callables (pure functions, runnable without Airflow) --------------


def check_drift_task(**context) -> bool:
    """Run Evidently drift; alert + return True iff retraining should proceed."""
    from monitoring.evidently_report import check_drift

    reference_df, current_df = _default_reference_current()
    result = check_drift(reference_df, current_df)
    for feat in result.breached:
        alert_drift(feat, result.psi_by_feature[feat])
    print(f"[check_drift] engine={result.engine} psi={result.psi_by_feature} "
          f"breached={result.breached}")
    return result.should_retrain  # ShortCircuitOperator stops here if False


def retrain_task(**context) -> dict:
    """Retrain on fresh data, apply the promotion gate, return the decision."""
    from pipelines.retrain import run_local_retrain

    txns = _default_training_transactions()
    result = run_local_retrain(txns, champion_auc=None, package=True)
    decision = result.decision
    if decision.promote:
        alert_retrain_complete(result.test_auc, MLFLOW_AUC_GATE)
    else:
        alert_gate_failed(result.test_auc, MLFLOW_AUC_GATE)
    payload = {
        "test_auc": result.test_auc,
        "promote": decision.promote,
        "reason": decision.reason,
        "artifact": str(result.artifact_path) if result.artifact_path else None,
    }
    print(f"[retrain] {payload}")
    return payload


def promote_model(run_id: str, auc_threshold: float = MLFLOW_AUC_GATE,
                  champion_auc: float | None = None):
    """MLflow promotion gate for the PRODUCTION path (registers the model).

    Shares the decision rule with the local pipeline so both agree. Kept for the
    SageMaker/MLflow deployment; the local DAG run uses `retrain_task`'s gate.
    """
    import mlflow

    from pipelines.retrain import decide_promotion

    run = mlflow.get_run(run_id)
    auc = float(run.data.metrics["test_auc_roc"])
    decision = decide_promotion(auc, champion_auc, gate=auc_threshold)
    if not decision.promote:
        alert_gate_failed(auc, auc_threshold)
        raise ValueError(decision.reason)
    mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL_NAME)
    alert_retrain_complete(auc, auc_threshold)
    return auc


def notify_task(**context) -> None:
    """Post a run summary to Slack."""
    from monitoring.slack_alerts import _post

    ti = context.get("ti")
    payload = ti.xcom_pull(task_ids="retrain") if ti else None
    msg = "📊 RiskSequencer retrain DAG completed."
    if payload:
        msg += (f" test_auc={payload['test_auc']:.4f} promote={payload['promote']} "
                f"({payload['reason']})")
    _post(msg)


def run_pipeline_locally(transactions: pd.DataFrame | None = None, max_epochs: int = 12) -> dict:
    """Execute the full loop end-to-end WITHOUT Airflow (drift -> retrain -> decide).

    This is the same logic the DAG schedules, runnable directly so the automated
    pipeline can be demonstrated and tested with no scheduler. Returns a summary.
    """
    from config import TrainConfig
    from monitoring.evidently_report import check_drift
    from pipelines.retrain import run_local_retrain

    ref, cur = _default_reference_current()
    drift = check_drift(ref, cur)
    for feat in drift.breached:
        alert_drift(feat, drift.psi_by_feature[feat])
    if not drift.should_retrain:
        return {"retrained": False, "engine": drift.engine, "psi": drift.psi_by_feature}

    txns = transactions if transactions is not None else _default_training_transactions()
    result = run_local_retrain(txns, cfg=TrainConfig(max_epochs=max_epochs), package=False)
    (alert_retrain_complete if result.decision.promote else alert_gate_failed)(
        result.test_auc, MLFLOW_AUC_GATE
    )
    return {
        "retrained": True,
        "engine": drift.engine,
        "breached": drift.breached,
        "test_auc": result.test_auc,
        "promote": result.decision.promote,
        "reason": result.decision.reason,
    }


def _on_failure(context) -> None:
    task_id = context["task_instance"].task_id
    alert_task_failure(task_id, str(context.get("exception", "unknown")))


# --- DAG definition (only when Airflow is available) ------------------------
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator, ShortCircuitOperator
    from airflow.utils.dates import days_ago

    default_args = {
        "owner": "risksequencer",
        "retries": 3,
        "retry_delay": timedelta(minutes=10),
        "on_failure_callback": _on_failure,
    }

    with DAG(
        dag_id="risksequencer_weekly_retrain",
        schedule_interval="@weekly",
        start_date=days_ago(1),
        catchup=False,
        default_args=default_args,
        tags=["risksequencer", "mlops"],
    ) as dag:
        check_drift = ShortCircuitOperator(
            task_id="check_drift", python_callable=check_drift_task
        )
        retrain = PythonOperator(task_id="retrain", python_callable=retrain_task)
        notify = PythonOperator(task_id="notify", python_callable=notify_task)

        check_drift >> retrain >> notify

except ImportError:  # Airflow not installed — keep module importable for tests
    dag = None
