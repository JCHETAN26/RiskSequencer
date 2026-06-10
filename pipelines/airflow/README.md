# RiskSequencer — Airflow weekly retraining

The DAG `risksequencer_weekly_retrain` runs the retraining loop on a **weekly**
schedule with **no manual intervention**:

```
check_drift (Evidently AI, PSI > 0.20)  ──short-circuit if no drift──►  retrain  ──►  notify
```

- **check_drift** runs Evidently AI's `DataDriftPreset` with the **PSI** stattest
  (threshold 0.20) over the 3 monitored features
  (`txn_velocity_1h`, `amount_zscore`, `new_device_flag`). If no feature breaches,
  the run short-circuits and nothing retrains.
- **retrain** pulls fresh data, retrains, and applies the promotion gate
  (≥ 0.90 AUC, no manual override) plus champion/challenger — see
  `pipelines/retrain.py`.
- **notify** posts a Slack summary. Drift, promotion, and task failures all alert.

Retries: 3 per task, 10-minute delay. Failures call `alert_task_failure`.

## Run it locally (Docker)

From the repo root:

```bash
docker compose -f pipelines/airflow/docker-compose.yaml up --build
```

Open http://localhost:8080 (the admin password is printed in the logs). The DAG
appears automatically and is scheduled `@weekly`.

### Run one full cycle immediately (no waiting for the schedule)

```bash
docker compose -f pipelines/airflow/docker-compose.yaml exec airflow \
  airflow dags test risksequencer_weekly_retrain 2024-01-01
```

This executes check_drift → retrain → notify end-to-end in the container.

## Run the loop without Airflow

The exact same logic is runnable directly (handy for a quick demo or CI):

```bash
python -c "from pipelines.retrain_dag import run_pipeline_locally; print(run_pipeline_locally())"
```

Covered by `tests/test_dag.py` (drift gate + full end-to-end decision).

## Productionizing the data providers

`pipelines/retrain_dag.py` ships with synthetic defaults so the DAG runs out of
the box. Swap these two functions for S3 loaders:

- `_default_reference_current()` → reference window + last-24h inference logs
  from `s3://risksequencer-logs/`.
- `_default_training_transactions()` → latest labelled window from
  `s3://risksequencer-data/`.

Everything downstream (drift, retrain, gate, notify) stays the same.
