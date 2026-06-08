# RiskSequencer — Build Plan

## Project Overview
A 2-layer LSTM-based behavioral anomaly detection system for transaction sequences,
with real-time SageMaker inference, automated retraining via Airflow, and Evidently AI
drift monitoring. Target: AUC-ROC ≥ 0.94, p99 inference < 50ms, fully automated MLOps loop.

---

## Tech Stack
| Layer | Tools |
|---|---|
| Modeling | Python, PyTorch, LightGBM |
| Experiment Tracking | MLflow |
| Deployment | AWS SageMaker |
| Orchestration | Apache Airflow |
| Monitoring | Evidently AI |
| Data | IEEE-CIS Fraud Detection (dev) → real transaction data (prod) |
| Explainability | SHAP (LightGBM), Attention weights (LSTM) |
| Alerting | Slack webhook on PSI breach |

---

## Phases

---

### Phase 0 — Environment Setup (Days 1–3)

**Goal:** Get all tools installed, connected, and talking to each other before writing model code.

#### Tasks
- [ ] Create a Python virtual environment (`Python 3.10+`)
- [ ] Install core dependencies:
  ```
  torch torchvision lightgbm mlflow evidently apache-airflow boto3
  scikit-learn pandas numpy shap matplotlib seaborn
  ```
- [ ] Set up AWS account and configure SageMaker IAM roles + S3 buckets:
  - `s3://risksequencer-data/` — raw + processed data
  - `s3://risksequencer-models/` — model artifacts
  - `s3://risksequencer-logs/` — inference logs
- [ ] Initialize MLflow tracking server (local or remote URI)
- [ ] Set up Airflow locally via Docker Compose
- [ ] Create project repo structure:
  ```
  risksequencer/
  ├── data/               # raw, processed, sequences
  ├── features/           # feature engineering scripts
  ├── models/             # LSTM + LightGBM code
  ├── training/           # train + eval scripts
  ├── serving/            # SageMaker inference code
  ├── monitoring/         # Evidently AI reports + PSI logic
  ├── pipelines/          # Airflow DAGs
  ├── notebooks/          # EDA + experimentation
  └── tests/              # unit + integration tests
  ```
- [ ] Set up Git branch protection on `main`:
  - No direct pushes to `main` — ever
  - All changes must go through a pull request
  - Require at least 1 approving review before merge
  - Branch naming convention: `feat/`, `fix/`, `exp/` (for experiments), `chore/`
  - Example: `git checkout -b feat/lstm-attention-layer`

#### Success Criteria
- `mlflow ui` launches without error
- Airflow webserver accessible at `localhost:8080`
- `boto3` can list S3 buckets in your AWS account
- All imports resolve cleanly

---

### Phase 1 — Data & Feature Engineering (Days 4–12)

**Goal:** Go from raw transactions to labeled, padded sequence tensors ready for LSTM training.

#### 1.1 — Data Acquisition
- [ ] Download IEEE-CIS Fraud Detection dataset from Kaggle (~590K transactions)
- [ ] Alternatively, generate synthetic data using PaySim or a custom simulator
- [ ] Store raw CSVs in `data/raw/`

#### 1.2 — Exploratory Data Analysis
- [ ] Profile class imbalance (fraud is typically <1–3% of transactions)
- [ ] Analyze transaction velocity, time gaps, merchant categories
- [ ] Identify top behavioral features with mutual information or correlation analysis
- [ ] Document findings in `notebooks/01_eda.ipynb`

#### 1.3 — Feature Engineering
Build the following feature groups per transaction event:

| Feature Group | Examples |
|---|---|
| Velocity | txn count in last 1h / 6h / 24h |
| Time | hour of day, day of week, time since last txn |
| Amount | amount, amount z-score vs user history, amount delta |
| Location | new country flag, new device flag, distance from last txn |
| Merchant | new merchant flag, merchant risk score, category |
| Account | account age, days since password change, login attempts |

- [ ] Write `features/feature_pipeline.py` with modular, testable functions
- [ ] Normalize continuous features (StandardScaler or RobustScaler)
- [ ] One-hot or ordinal encode categorical features

#### 1.4 — Sequence Construction
- [ ] Group transactions by `user_id`, sorted by timestamp
- [ ] Define sequence window: last **50 transactions** per user (pad shorter histories)
- [ ] Padding strategy: left-pad with zeros, add padding mask
- [ ] Label: `1` if any transaction in the sequence is fraudulent, `0` otherwise
- [ ] Write `data/sequence_builder.py` → outputs `(N, 50, F)` tensors where F = feature count

#### 1.5 — Train/Validation/Test Split
- [ ] Split **by time**, not randomly (prevents data leakage):
  - Train: months 1–8
  - Validation: months 9–10
  - Test (held-out): months 11–12
- [ ] Save splits to `s3://risksequencer-data/splits/`

#### Success Criteria
- Feature matrix shape: `(~500K, 50, F)` — no NaNs, no leakage
- Class imbalance ratio documented; decide on SMOTE, class weighting, or oversampling strategy
- Sequence builder runs end-to-end in under 10 minutes

---

### Phase 2 — Modeling (Days 13–28)

**Goal:** Train a 2-layer LSTM with attention that hits AUC-ROC ≥ 0.94 on validation set.

#### 2.1 — LightGBM Baseline
Before the LSTM, establish a strong baseline:
- [ ] Flatten the last transaction's features (no sequence)
- [ ] Train LightGBM with default params
- [ ] Add SHAP explainability: `shap.TreeExplainer(lgbm_model)`
- [ ] Target: AUC-ROC > 0.88 baseline
- [ ] Log to MLflow: params, AUC, feature importance plot

#### 2.2 — LSTM Architecture
```python
class RiskSequencer(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        self.attention = nn.Linear(hidden_size, 1)   # additive attention
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        lstm_out, _ = self.lstm(x)                   # (B, T, H)
        attn_weights = self.attention(lstm_out)       # (B, T, 1)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1), -1e9)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (attn_weights * lstm_out).sum(dim=1)  # (B, H)
        return self.classifier(context), attn_weights
```

#### 2.3 — Training Setup
- [ ] Loss: `BCELoss` with `pos_weight` to handle class imbalance
- [ ] Optimizer: `AdamW`, lr=`1e-3`, weight decay=`1e-4`
- [ ] Scheduler: `ReduceLROnPlateau` on validation AUC
- [ ] Early stopping: patience=5 epochs on validation AUC
- [ ] Batch size: 512 (tune based on GPU memory)
- [ ] Train on GPU (local or SageMaker Training Job)

#### 2.4 — Hyperparameter Search
Use MLflow + manual grid or Optuna:
- hidden_size: [64, 128, 256]
- num_layers: [1, 2, 3]
- dropout: [0.1, 0.2, 0.3, 0.4]
- learning_rate: [1e-4, 5e-4, 1e-3]
- sequence_length: [30, 50, 100]

#### 2.5 — Threshold Tuning
- [ ] Plot precision-recall curve across thresholds
- [ ] Find threshold where FPR ≤ 5% and maximize precision
- [ ] Target: **88% precision at 5% FPR**
- [ ] Store optimal threshold as a model artifact

#### 2.6 — Evaluation on Held-Out Test Set (run once, at the end)
- [ ] AUC-ROC ≥ 0.94
- [ ] Precision ≥ 88% at 5% FPR
- [ ] Log confusion matrix, ROC curve, PR curve to MLflow

#### Success Criteria
- At least 3 MLflow experiment runs logged
- LSTM outperforms LightGBM baseline by ≥ 4 AUC points
- Attention weights visualized for 5 example sequences (shows interpretability)

---

### Phase 3 — Explainability & Evaluation (Days 29–34)

**Goal:** Make the model auditable and understandable to non-ML stakeholders.

#### 3.1 — LSTM Attention Visualization
- [ ] For flagged transactions, extract attention weights → show which past events drove the flag
- [ ] Build `notebooks/03_attention_viz.ipynb` with heatmap plots per sequence

#### 3.2 — SHAP for LightGBM Layer
- [ ] Run SHAP on LightGBM validation predictions
- [ ] Plot top 10 feature contributors per prediction
- [ ] Save SHAP summary plot as MLflow artifact

#### 3.3 — Business Metrics
Calculate and log:
- [ ] **$ fraud caught** = sum of flagged fraud transaction amounts
- [ ] **$ false positive cost** = estimated churn value × false positive count
- [ ] **Net savings** = $ fraud caught − $ false positive cost
- [ ] Add these to a `business_metrics.json` artifact in MLflow

#### 3.4 — Error Analysis
- [ ] Analyze false negatives: which fraud patterns does the model miss?
- [ ] Analyze false positives: which legitimate patterns look fraudulent?
- [ ] Document findings in `notebooks/04_error_analysis.ipynb`

---

### Phase 4 — Deployment (Days 35–45)

**Goal:** Serve the model as a SageMaker real-time endpoint with p99 < 50ms and an MLflow AUC gate.

#### 4.1 — Model Packaging
- [ ] Write `serving/inference.py` with `model_fn`, `input_fn`, `predict_fn`, `output_fn`
- [ ] Package model + threshold + scaler into a single `model.tar.gz`
- [ ] Upload to `s3://risksequencer-models/`

#### 4.2 — SageMaker Endpoint
```python
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    model_data="s3://risksequencer-models/model.tar.gz",
    role=sagemaker_role,
    framework_version="2.0",
    py_version="py310",
    entry_point="inference.py"
)

predictor = model.deploy(
    instance_type="ml.m5.xlarge",   # start here, benchmark p99 latency
    initial_instance_count=1,
)
```

- [ ] Run latency benchmark: 1000 requests, measure p50/p95/p99
- [ ] If p99 > 50ms: try `ml.c5.xlarge`, reduce sequence length, or apply TorchScript

#### 4.3 — MLflow Promotion Gate
```python
def promote_model(run_id, auc_threshold=0.90):
    run = mlflow.get_run(run_id)
    auc = float(run.data.metrics["test_auc_roc"])
    if auc < auc_threshold:
        raise ValueError(f"Model AUC {auc:.3f} below gate {auc_threshold}. Blocked.")
    mlflow.register_model(f"runs:/{run_id}/model", "risksequencer-prod")
    print(f"Model promoted. AUC: {auc:.3f}")
```

- [ ] Integrate gate into deployment script — no manual override path

#### 4.4 — Champion/Challenger Setup
- [ ] Keep current production model as **champion**
- [ ] Route 10% of live traffic to **challenger** (new model) via SageMaker A/B variant
- [ ] Promote challenger only if it beats champion AUC on live traffic over 48hrs

#### 4.5 — Integration Test
- [ ] Write `tests/test_endpoint.py` — sends 100 sample payloads, checks response shape + latency
- [ ] Assert p99 < 50ms across test calls
- [ ] Assert output is a valid probability in [0, 1]

#### Success Criteria
- Endpoint live and returning predictions
- p99 latency < 50ms confirmed via benchmark
- MLflow gate rejects a dummy model with AUC=0.85

---

### Phase 5 — Monitoring & Automated Retraining (Days 46–56)

**Goal:** Weekly automated retraining with Evidently AI drift detection and Slack alerting.

#### 5.1 — Evidently AI Drift Monitoring
Define 3 key behavioral features to monitor (e.g., `txn_velocity_1h`, `amount_zscore`, `new_device_flag`):

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_df, current_data=current_df)

for feature in KEY_FEATURES:
    psi = report.as_dict()["metrics"][0]["result"]["drift_by_columns"][feature]["stattest_threshold"]
    if psi > 0.20:
        trigger_retraining(feature, psi)
```

- [ ] Store reference dataset (last training window) in S3
- [ ] Run Evidently report daily on last 24h of inference logs
- [ ] Trigger retraining if PSI > 0.20 on any of the 3 key features

#### 5.2 — Slack Alerting
- [ ] Set up Slack webhook
- [ ] Send alert when PSI threshold is breached:
  ```
  🚨 RiskSequencer Drift Alert
  Feature: txn_velocity_1h
  PSI: 0.27 (threshold: 0.20)
  Retraining triggered automatically.
  ```
- [ ] Send alert when retraining completes (with new AUC)
- [ ] Send alert if retraining model fails AUC gate

#### 5.3 — Airflow DAG
```python
# pipelines/retrain_dag.py
with DAG("risksequencer_weekly_retrain", schedule_interval="@weekly") as dag:

    check_drift = PythonOperator(task_id="check_drift", python_callable=run_evidently_report)
    fetch_data  = PythonOperator(task_id="fetch_data",  python_callable=pull_latest_data)
    retrain     = PythonOperator(task_id="retrain",     python_callable=run_training_job)
    evaluate    = PythonOperator(task_id="evaluate",    python_callable=evaluate_model)
    gate        = PythonOperator(task_id="auc_gate",    python_callable=promote_model)
    deploy      = PythonOperator(task_id="deploy",      python_callable=update_endpoint)
    notify      = PythonOperator(task_id="notify",      python_callable=send_slack_summary)

    check_drift >> fetch_data >> retrain >> evaluate >> gate >> deploy >> notify
```

- [ ] Also trigger DAG on-demand when PSI breach detected (not just weekly)
- [ ] Add retry logic: 3 retries with 10-minute delay on any task failure
- [ ] Store DAG run history and logs in Airflow metadata DB

#### Success Criteria
- Airflow DAG runs end-to-end without errors on a test run
- Evidently report correctly identifies a manually injected drift scenario
- Slack message received within 2 minutes of PSI breach
- Full retraining pipeline (data → train → gate → deploy) completes in < 2 hours

---

## Overall Timeline Summary

| Phase | Duration | Cumulative |
|---|---|---|
| 0 — Environment Setup | 3 days | Day 3 |
| 1 — Data & Features | 9 days | Day 12 |
| 2 — Modeling | 16 days | Day 28 |
| 3 — Explainability | 6 days | Day 34 |
| 4 — Deployment | 11 days | Day 45 |
| 5 — Monitoring & Retraining | 11 days | Day 56 |

**Total: ~8 weeks solo at mid-level pace**

---

## Key Metrics to Track Throughout

| Metric | Target |
|---|---|
| AUC-ROC (test) | ≥ 0.94 |
| Precision at 5% FPR | ≥ 88% |
| p99 inference latency | < 50ms |
| MLflow AUC gate | 0.90 |
| PSI retraining trigger | > 0.20 on 3 features |
| Retraining pipeline runtime | < 2 hours |

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| LSTM doesn't hit 0.94 AUC | Add attention, tune sequence length, try bidirectional LSTM |
| p99 latency > 50ms | Try TorchScript, batch norm removal, smaller hidden size, or async inference |
| Severe class imbalance | Use focal loss or `pos_weight` in BCELoss; try oversampling with SMOTE |
| SageMaker costs spiral | Use Spot Instances for training jobs; monitor with AWS Cost Explorer |
| Airflow DAG silent failures | Add Slack notification on task failure; use SLAs in Airflow |
