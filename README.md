# RiskSequencer

[![CI](https://github.com/JCHETAN26/RiskSequencer/actions/workflows/ci.yml/badge.svg)](https://github.com/JCHETAN26/RiskSequencer/actions/workflows/ci.yml)

A 2-layer LSTM-with-attention behavioral anomaly detection system for transaction
sequences. Detects fraud by reasoning over **sequences of user behavior over time**
(device change → address update → burst of transactions) rather than scoring
transactions in isolation.

Targets: AUC-ROC ≥ 0.94, precision ≥ 88% @ 5% FPR, p99 inference < 50ms, fully
automated retraining loop with drift-triggered + weekly retraining.

## Status

This repository contains the **runnable foundation** — Phases 0–2 are implemented
and verified end-to-end on a built-in synthetic data generator, so the whole
pipeline (features → sequences → model → eval) trains locally with **no AWS or
Kaggle download required**. The deployment/monitoring code (Phases 3–5) is written
to spec and import-guarded so it doesn't require AWS/Airflow to be installed for the
test suite to pass.

| Phase | Area | State |
|---|---|---|
| 0 | Environment & structure | ✅ `config.py`, `requirements.txt`, package layout |
| 1 | Data & features | ✅ synthetic generator, **IEEE-CIS adapter**, causal feature pipeline, sequence builder, time-based split, **EDA notebook** |
| 2 | Modeling | ✅ LSTM+attention, LightGBM baseline, training loop, eval/threshold tuning, **hyperparameter search** |
| 3 | Explainability | ✅ attention viz + SHAP + **error analysis** (`notebooks/04`) + **business metrics** ($ caught / FP cost / net savings) |
| 4 | Deployment | 🧩 `serving/inference.py` handlers + MLflow gate + **`package_model.py`** (builds deployable `model.tar.gz`); endpoint not deployed |
| 5 | Monitoring & retrain | ✅ PSI drift, Slack alerts, **local end-to-end retrain orchestration + champion/challenger gate** (`pipelines/retrain.py`), Airflow DAG wrapper; live scheduling needs AWS |

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
  └─ features/feature_pipeline.py   causal velocity / time / amount / flags
       └─ data/sequence_builder.py  group by user, last 50 txns, left-pad + mask
            ├─ models/lgbm_baseline.py   LightGBM on last-txn features (+ SHAP)
            └─ models/lstm_model.py      2-layer LSTM + additive attention
                 └─ training/train.py    pos_weight BCE, ReduceLROnPlateau, early stop
                      └─ training/evaluate.py   AUC, threshold @ FPR≤5%
                           └─ serving/inference.py   SageMaker model/input/predict/output_fn
                                └─ monitoring/   PSI drift + Slack alerts
                                     └─ pipelines/retrain_dag.py   Airflow weekly + on-drift
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
monitoring/evidently_report.py   PSI + drift trigger
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
