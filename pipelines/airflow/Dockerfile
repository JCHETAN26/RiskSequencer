# Airflow image for RiskSequencer's weekly retrain DAG.
# Extends the official Airflow image with the project's runtime deps so the
# DAG's tasks (Evidently drift check + LSTM retrain) run inside the scheduler.
FROM apache/airflow:2.9.3-python3.11

COPY pipelines/airflow/requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt

# Project code is mounted at /opt/project (see docker-compose.yaml) and added to
# PYTHONPATH there, so `from pipelines.retrain_dag import ...` resolves.
