# RiskSequencer

[![CI](https://github.com/JCHETAN26/RiskSequencer/actions/workflows/ci.yml/badge.svg)](https://github.com/JCHETAN26/RiskSequencer/actions/workflows/ci.yml)

A 2-layer LSTM-with-attention behavioral anomaly detection system for transaction
sequences. Detects fraud by reasoning over **sequences of user behavior over time**
(device change → address update → burst of transactions) rather than scoring
transactions in isolation.

Targets: AUC-ROC ≥ 0.94, precision ≥ 88% @ 5% FPR, p99 inference < 50ms, fully
automated retraining loop with drift-triggered + weekly retraining.

## Status

All of Phases 0–5 are implemented, tested, and **validated on the real IEEE-CIS
dataset** (not just the built-in synthetic generator). The full local pipeline —
features → sequences → models → ensemble → eval → packaging → retrain
orchestration — runs end-to-end. The remaining work is the parts that genuinely
need AWS: deploying the SageMaker endpoint and wiring the Phase 5 loop to live
data. Deployment/monitoring code is import-guarded so AWS/Airflow aren't needed
to run the test suite.

### Results on real IEEE-CIS (held-out test, user-level AUC)

| Model | AUC | Notes |
|---|---|---|
| LSTM + attention alone | 0.812 | behavioral sequences only |
| LightGBM alone | 0.926 | native `C/D/V` + identity + categorical features |
| **Hybrid (stacked)** | **0.929** | LightGBM + LSTM via a logistic meta-learner |

The build plan's 0.94 target assumed a sequence-friendly dataset; IEEE-CIS is
fundamentally per-transaction (the `card1+addr1` user proxy yields a median of 2
txns/user), so most signal is tabular. The **hybrid architecture** (from the
system-prompt diagram) captures it: LightGBM on rich native features + the LSTM's
sequence signal, combined by a meta-learner. On the **synthetic** generator the
LSTM alone reaches ≈0.99 (fraud is deliberately separable there). See
`training/train_hybrid.py`.

| Phase | Area | State |
|---|---|---|
| 0 | Environment & structure | ✅ `config.py`, `requirements.txt`, package layout |
| 1 | Data & features | ✅ synthetic generator, **IEEE-CIS adapter**, causal feature pipeline, sequence builder, time-based split, **EDA notebook** |
| 2 | Modeling | ✅ LSTM+attention, LightGBM baseline, **hybrid stacking ensemble** (`train_hybrid.py`), HPO — validated on real IEEE-CIS (hybrid 0.929) |
| 3 | Explainability | ✅ attention viz + SHAP + **error analysis** (`notebooks/04`) + **business metrics** ($ caught / FP cost / net savings) |
| 4 | Deployment | ✅ handlers + MLflow gate + `package_model.py` + **`deploy_sagemaker.py`** — deployed a real SageMaker endpoint and **verified p99 inference 45 ms < 50 ms** (CloudWatch ModelLatency, ml.c5.large), then torn down |
| 5 | Monitoring & retrain | ✅ **Evidently AI** drift (PSI stattest, threshold 0.20), Slack alerts, retrain orchestration + champion/challenger gate, **runnable Airflow DAG** (`@weekly`, drift-triggered) with Docker compose + runbook |

## Quickstart

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # torch, lightgbm, sklearn, mlflow, ...

# 1a. Generate synthetic transactions (no Kaggle needed)
python -m data.synthetic --n-users 2000

# 1b. ...OR adapt the real IEEE-CIS dataset to the same schema
#     (download via: kaggle competitions download -c ieee-fraud-detection,
#      unzip train_transaction.csv / train_identity.csv into data/raw/)
python -m data.ieee_cis            # writes data/raw/transactions.parquet

# 2. Train the LSTM end-to-end and print validation metrics
python -m training.train

# 3. Run the test suite
pytest -q
```

`python -m training.train` builds features, splits by time, scales (fit on train
only), builds `(N, 50, F)` sequences, and trains the LSTM with
`BCEWithLogitsLoss(pos_weight=...)`, early stopping on validation AUC. On the
synthetic data it reaches val AUC ≈ 0.99.

> **macOS gotcha — torch + LightGBM in one process.** Both ship their own OpenMP
> runtime and segfault when loaded together (e.g. `02_modeling.ipynb`, which
> trains both). Guard it by setting these **before** importing either library:
> `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE` (plus `torch.set_num_threads(1)`
> and LightGBM `num_threads=1`). The notebook's first cell does this; the rest of
> the codebase never imports both in one process. Locked in by
> `tests/test_torch_lgbm_coexist.py` (runs in a subprocess so a crash can't fail
> the whole suite).

## Architecture

```
raw transactions
  ├─ features/feature_pipeline.py → data/sequence_builder.py   (behavioral, per-user)
  │    └─ models/lstm_model.py     2-layer LSTM + additive attention ─┐
  │                                                                    │
  └─ native tabular features (C/D/V + identity + categorical) ─────────┤
       └─ LightGBM (per-txn → user-max aggregation) ───────────────────┤
                                                                        ▼
                                       models/hybrid.py  stacking meta-learner
                                       training/train_hybrid.py  (real IEEE-CIS)
                                                 │
                                training/evaluate.py   AUC, threshold @ FPR≤5%
                                serving/{inference,package_model}.py   model.tar.gz
                                monitoring/   PSI drift + Slack alerts
                                pipelines/retrain.py + retrain_dag.py   gate + champion/challenger
```

## Design constraints (enforced in code)

- **No leakage** — features are causal (`amount_zscore` uses `expanding().shift(1)`,
  flags use `duplicated()`); splits are time-based (`time_based_split`). Tests assert
  causality (`tests/test_features.py::test_amount_zscore_is_causal`).
- **Attention is mandatory** — `RiskSequencer.forward` returns `(logits, attn_weights)`
  with a padding mask; tests assert padded positions get ~0 attention.
- **Sequence length = 50**, left-padded with a mask. Centralised in `config.py`.
- **Class imbalance** via `pos_weight` in BCE — no blind oversampling.
- **MLflow AUC gate (0.90)** with no manual override path (`pipelines.retrain_dag.promote_model`).
- **PSI threshold 0.20** monitored on exactly `txn_velocity_1h`, `amount_zscore`,
  `new_device_flag` (`config.MONITORED_FEATURES`).
- **No direct pushes to `main`** — branch, PR, merge.

## Layout

```
config.py                 single source of truth for constants/thresholds
data/synthetic.py         synthetic transaction generator (fraud = behavioral burst)
data/ieee_cis.py          IEEE-CIS Kaggle dataset -> canonical schema adapter
data/sequence_builder.py  (N, 50, F) tensors + mask, time-based split, scaler
features/feature_pipeline.py   modular causal feature functions
models/lstm_model.py      RiskSequencer (LSTM + additive attention)
models/lgbm_baseline.py   LightGBM baseline + SHAP
models/hybrid.py          LightGBM+LSTM stacking ensemble core (meta-learner)
training/train.py         training loop (MLflow-aware)
training/evaluate.py      AUC, threshold tuning @ FPR≤5%
training/business_metrics.py   $ fraud caught / FP cost / net savings + profit curve
training/hyperparameter_search.py   Optuna/random search over the plan's grid
training/train_hybrid.py  hybrid LightGBM+LSTM training/eval (real IEEE-CIS)
notebooks/01_eda.ipynb    EDA (imbalance, velocity, amount, mutual information)
notebooks/02_modeling.ipynb   LightGBM baseline vs LSTM comparison (+ SHAP, ROC/PR)
notebooks/03_attention_viz.ipynb   attention heatmaps — why a sequence was flagged
notebooks/04_error_analysis.ipynb  FN/FP analysis, business metrics, profit curve
serving/inference.py      SageMaker inference handlers
serving/package_model.py  build a deployable model.tar.gz (artifacts + code/)
serving/deploy_sagemaker.py   deploy real-time endpoint, benchmark p99, teardown
monitoring/evidently_report.py   Evidently AI drift (PSI stattest) + fallback PSI
pipelines/airflow/        Dockerized Airflow (compose + Dockerfile + runbook)
monitoring/slack_alerts.py       webhook alerts
pipelines/retrain.py      local end-to-end retrain + promotion (gate + champion/challenger)
pipelines/retrain_dag.py  Airflow retrain DAG (thin wrapper over pipelines/retrain.py)
tests/                    pytest suite (features, sequences, model, endpoint, drift)
```

## Moving to real data / AWS (next steps)

- Swap `data/synthetic.py` for an IEEE-CIS adapter that normalises Kaggle CSVs to the
  canonical schema in `config.py` (same downstream code).
- Set real bucket names / `SLACK_WEBHOOK_URL` via env; point MLflow at a tracking server.
- Package `model.pt + scaler.pkl + threshold.json` into `model.tar.gz` and deploy via
  `sagemaker.pytorch.PyTorchModel` with `entry_point="serving/inference.py"`.
- Install Airflow in its own env (see `requirements.txt` note) and register the DAG.
```
