# RiskSequencer — How it actually works

Two loops: the **inference loop** (score a user, log the prediction) and the
**monitoring loop** (watch the logged data for drift, retrain automatically).
PostgreSQL sits in the middle — predictions are written to it, and the drift
monitor reads from it.

---

## Big picture

```mermaid
flowchart LR
    subgraph SERVE["Serving (per request)"]
      TX["Transaction"] --> FE["Feature engineering<br/>(causal, no leakage)"]
      FE --> SEQ["Sequence builder<br/>(50 x F + mask)"]
      FE --> TAB["Tabular features<br/>(C/D/V + identity)"]
      SEQ --> LSTM["2-layer LSTM<br/>+ attention"]
      TAB --> LGB["LightGBM"]
      LSTM --> STK["Stacking<br/>meta-learner"]
      LGB --> STK
      STK --> PROB["fraud probability<br/>+ attention explanation"]
      PROB --> EP["SageMaker endpoint<br/>p99 ~45ms"]
    end

    PROB --> DB[("PostgreSQL<br/>inference_log")]

    subgraph LOOP["MLOps loop (scheduled)"]
      AF["Airflow @weekly"] --> EV["Evidently AI<br/>drift (PSI)"]
      DB --> EV
      EV -->|"PSI > 0.20"| RT["Retrain"]
      EV -->|"no drift"| STOP["stop"]
      RT --> GATE{"AUC ≥ 0.90<br/>AND beats champion?"}
      GATE -->|yes| DEP["Package / deploy"]
      GATE -->|no| KEEP["keep old model"]
      RT --> MLF["MLflow logs run"]
      EV & DEP & GATE --> SLACK["Slack alerts"]
    end
```

---

## Loop 1 — Inference (scoring one user)

```mermaid
sequenceDiagram
    participant U as User / caller
    participant EP as SageMaker endpoint
    participant M as Model (LSTM + LightGBM + stacker)
    participant DB as PostgreSQL
    U->>EP: transaction sequence (JSON)
    EP->>M: features -> score
    M-->>EP: fraud probability + attention
    EP-->>U: {probability, is_fraud}
    EP->>DB: log prediction + monitored features
```

**In plain words:** a transaction arrives → we build features from the user's
recent history → the LSTM reads the *sequence* and LightGBM reads the *tabular
details* → a small stacker merges them into one probability → above the threshold
we flag it (and return *which* past events drove it via attention). Every
prediction, plus its 3 monitored feature values, is written to the
`inference_log` table in Postgres.

---

## Loop 2 — Monitoring & retraining (the automated part)

```mermaid
flowchart TD
    A["Airflow wakes up @weekly"] --> B["read reference window<br/>+ current window FROM Postgres"]
    B --> C["Evidently AI computes PSI<br/>on 3 monitored features"]
    C --> D{"any PSI > 0.20?"}
    D -->|no| E["stop — model still healthy"]
    D -->|yes| F["retrain on fresh data"]
    F --> G{"new model: AUC ≥ 0.90<br/>AND beats champion?"}
    G -->|no| H["keep champion<br/>(challenger discarded)"]
    G -->|yes| I["package model.tar.gz<br/>+ promote"]
    F --> J["MLflow logs the run"]
    C --> K["Slack: drift alert"]
    I --> L["Slack: promoted"]
    H --> M["Slack: gate failed"]
```

**In plain words:** every week Airflow reads recent behavior from Postgres and
asks Evidently "has anything drifted?" using PSI. If any of the 3 watched
features crosses 0.20, it retrains. The new model only ships if it clears the
0.90 AUC gate *and* beats the current model. Slack is notified at each step.
Nobody has to touch it.

---

## See it run yourself

```bash
# 1. start Postgres (host port 5440)
docker compose -f storage/docker-compose.yaml up -d

# 2. point the app at it and run the demo
export DATABASE_URL=postgresql+psycopg2://risk:risk@localhost:5440/risksequencer
python -m storage.seed_demo
```

What the demo does (and what you'll see):

1. trains a quick model and scores users,
2. **logs ~73k predictions** (+ monitored features) to the Postgres `inference_log`,
3. reads a reference window and a (drift-injected) current window **back from Postgres**,
4. runs **Evidently** on them and prints the PSI per feature and the retrain decision:

```
[db] logged predictions: 60017 reference + 12861 current rows
[drift] engine=evidently
   PSI txn_velocity_1h   = 15.394
   PSI amount_zscore     = 0.013
   PSI new_device_flag   = 0.065
[drift] breached: ['txn_velocity_1h']
[decision] retrain? True  (triggered by PSI > 0.20 read from Postgres)
```

Inspect the data directly:

```bash
docker exec -it storage-postgres-1 \
  psql -U risk -d risksequencer -c "SELECT count(*) FROM inference_log;"
```

Tear down when done:

```bash
docker compose -f storage/docker-compose.yaml down -v
```

---

## Where Postgres fits

| Used for | Table / role |
|---|---|
| Raw transactions the model scored | `transactions` |
| Every prediction + monitored features | `inference_log` (the drift source) |
| Airflow metadata (dag runs, task logs) | Airflow tables in the same DB (LocalExecutor) |

The scheduled DAG (`pipelines/retrain_dag.py`) reads its drift windows from
Postgres when `DATABASE_URL` is set, and falls back to synthetic data otherwise,
so it runs anywhere. The Dockerized Airflow (`pipelines/airflow/`) points both
its metadata DB **and** the inference log at one Postgres instance.
