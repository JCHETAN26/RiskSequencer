"""Airflow DAG: weekly (and drift-triggered) retraining for RiskSequencer.

Pipeline: check_drift -> fetch_data -> retrain -> evaluate -> auc_gate ->
deploy -> notify. Each task has 3 retries with a 10-minute delay and a Slack
notification on failure (HARD constraints from the build plan).

The DAG file is import-guarded: it only builds the DAG object when Airflow is
installed, so the repo's unit tests can import sibling modules without pulling
Airflow into the test environment. The task callables live at module level so
they are independently testable.
"""

from __future__ import annotations

from datetime import timedelta

from config import MLFLOW_AUC_GATE, REGISTERED_MODEL_NAME
from monitoring.slack_alerts import (
    alert_drift,
    alert_gate_failed,
    alert_retrain_complete,
    alert_task_failure,
)

# --- Task callables (pure functions, testable without Airflow) -------------


def run_evidently_report(**context) -> bool:
    """Daily/weekly drift check; returns True if retraining should proceed."""
    from monitoring.evidently_report import check_drift

    reference_df, current_df = _load_reference_and_current()
    result = check_drift(reference_df, current_df)
    for feat in result.breached:
        alert_drift(feat, result.psi_by_feature[feat])
    return result.should_retrain


def pull_latest_data(**context):
    """Pull the latest labelled window from S3 (stub for the data adapter)."""
    raise NotImplementedError("wire to s3://risksequencer-data in deployment")


def run_training_job(**context):
    """Kick off training (local loop or a SageMaker Training Job)."""
    raise NotImplementedError("invoke training.train.run_training on fresh data")


def evaluate_model(**context):
    """Evaluate the candidate on the held-out window; push test_auc to XCom."""
    raise NotImplementedError("evaluate candidate -> return test_auc_roc")


def promote_model(
    run_id: str,
    auc_threshold: float = MLFLOW_AUC_GATE,
    champion_auc: float | None = None,
):
    """MLflow promotion gate — no manual override path.

    Reads the run's test AUC and registers the model only if `decide_promotion`
    approves it (absolute gate + champion/challenger comparison); otherwise
    raises so the DAG fails loudly. The decision logic is shared with the local
    pipeline (`pipelines.retrain.decide_promotion`) so both agree.
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


def update_endpoint(**context):
    """Deploy challenger to 10% traffic (champion/challenger A/B)."""
    raise NotImplementedError("update SageMaker endpoint variant weights to 90/10")


def send_slack_summary(**context):
    """Post a run summary to Slack."""
    from monitoring.slack_alerts import _post

    _post("📊 RiskSequencer retrain DAG completed.")


def _on_failure(context) -> None:
    task_id = context["task_instance"].task_id
    alert_task_failure(task_id, str(context.get("exception", "unknown")))


def _load_reference_and_current():  # pragma: no cover - deployment wiring
    raise NotImplementedError("load reference window + last-24h inference logs from S3")


# --- DAG definition (only when Airflow is available) -----------------------
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
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
        check_drift = PythonOperator(task_id="check_drift", python_callable=run_evidently_report)
        fetch_data = PythonOperator(task_id="fetch_data", python_callable=pull_latest_data)
        retrain = PythonOperator(task_id="retrain", python_callable=run_training_job)
        evaluate_t = PythonOperator(task_id="evaluate", python_callable=evaluate_model)
        gate = PythonOperator(
            task_id="auc_gate",
            python_callable=lambda **c: promote_model(c["ti"].xcom_pull(task_ids="retrain")),
        )
        deploy = PythonOperator(task_id="deploy", python_callable=update_endpoint)
        notify = PythonOperator(task_id="notify", python_callable=send_slack_summary)

        check_drift >> fetch_data >> retrain >> evaluate_t >> gate >> deploy >> notify

except ImportError:  # Airflow not installed — keep module importable for tests
    dag = None
