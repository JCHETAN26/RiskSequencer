# RiskSequencer

[![CI](https://github.com/JCHETAN26/RiskSequencer/actions/workflows/ci.yml/badge.svg)](https://github.com/JCHETAN26/RiskSequencer/actions/workflows/ci.yml)

Behavioral fraud detection that scores a user's risk from **sequences of behavior
over time**, not single transactions in isolation. A device change, a new
country, or a large purchase can each look fine alone — but the *pattern*
(device change → address update → burst of fast transactions) is the signature of
an account takeover. RiskSequencer models that pattern with a **2-layer LSTM +
attention**, combines it with a **LightGBM** model on rich tabular features via a
**stacking ensemble**, and wraps it all in a production MLOps loop: MLflow
tracking, a SageMaker real-time endpoint, Evidently AI drift monitoring, Slack
alerting, and automated Airflow retraining with a promotion gate.

## Results (real IEEE-CIS, held-out test, user-level AUC)

| Model | AUC | Notes |
|---|---|---|
| LSTM + attention alone | 0.812 | behavioral sequences only |
| LightGBM alone | 0.926 | native `C/D/V` + identity + categorical features |
| **Hybrid (stacked)** | **0.929** | LightGBM + LSTM via a logistic meta-learner |

**Serving:** deployed to a real SageMaker endpoint (`ml.c5.large`) and verified
**p99 inference = 45 ms (< 50 ms)** via CloudWatch `ModelLatency` (the true
inference time, excluding client/network), then torn down.

> **Why not 0.94?** The original target assumed a sequence-friendly dataset.
> IEEE-CIS is fundamentally *per-transaction* — its `card1+addr1` user proxy gives
> a median of **2 transactions per user**, so the LSTM alone has almost no sequence
> to learn from and tops out at 0.81. Most of the signal is tabular, which is why
> the **hybrid** recovers it (0.93). On the synthetic generator, where fraud is a
> clean behavioral burst, the LSTM alone reaches ≈0.99.

## Quickstart

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Get data — synthetic (zero setup) ...
python -m data.synthetic --n-users 2000
#    ... or the real IEEE-CIS dataset (Kaggle), mapped to the same schema:
#    kaggle competitions download -c ieee-fraud-detection  (unzip into data/raw/)
python -m data.ieee_cis                 # -> data/raw/transactions.parquet

# 2. Train the LSTM end-to-end (val metrics printed)
python -m training.train

# 3. Train + evaluate the hybrid on real IEEE-CIS
python -m training.train_hybrid \
  --transaction-csv data/raw/train_transaction.csv \
  --identity-csv    data/raw/train_identity.csv

# 4. Run the automated retrain loop end-to-end (drift -> retrain -> gate), no Airflow
python -c "from pipelines.retrain_dag import run_pipeline_locally; print(run_pipeline_locally())"

# 5. Tests
pytest -q
```

## Architecture

```
INFERENCE  (scoring a user)
  raw transactions
        │
        ▼
  feature engineering ── causal: velocity, time gaps, amount z-score,
   (no leakage)          new-device / new-country / new-merchant flags
        │
        ├──────────────► sequence builder (N,50,F)+mask ──► LSTM + attention ──┐
        │                                                                       │ score
        └──────────────► native tabular features ─────────► LightGBM ──────────┤ (user-max)
                         (C/D/V + identity + categorical)                       │
                                                                                ▼
                                       stacking meta-learner (fit on validation)
                                                  │
                                       fraud probability → threshold → FLAG + attention
                                                  │
                                       SageMaker real-time endpoint  (p99 ~45 ms)

MLOps LOOP  (keeping the model fresh)
  Airflow @weekly → check_drift (Evidently AI, PSI > 0.20 on 3 features)
        │                                   │
        │ drift                             └─ no drift → stop
        ▼
  retrain → promotion gate (AUC ≥ 0.90 AND beats champion) → package/deploy
        │                                                          │
        └────────── Slack alerts · MLflow logs every run ──────────┘
```

## How it works (plain English)

- **Scoring:** pull a user's last 50 transactions → turn them into features → the
  LSTM reads the *sequence* and LightGBM reads the *tabular details* → a small
  meta-learner merges both into one fraud probability → above threshold = flag,
  with attention weights showing *which* past events drove it.
- **Staying fresh:** every week Airflow checks whether behavior has drifted
  (Evidently PSI > 0.20). If yes, it retrains, and the new model ships only if it
  clears the 0.90 AUC gate *and* beats the current model — fully automatic.

## Design constraints (enforced in code)

- **No leakage** — causal features (`amount_zscore` = `expanding().mean().shift(1)`,
  first-seen flags via `duplicated()`); **time-based** split; the stacking
  meta-learner is fit on the *validation* split, never train. Tests assert
  causality (`tests/test_features.py::test_amount_zscore_is_causal`).
- **Attention is mandatory** — `RiskSequencer.forward` returns
  `(logits, attn_weights)` with a padding mask; tests assert padded positions get
  ~0 attention.
- **Sequence length = 50**, left-padded + masked (most-recent txn always last).
- **Class imbalance** via `pos_weight` in `BCEWithLogitsLoss` — no blind
  oversampling. Model emits **logits** (sigmoid applied only at inference).
- **MLflow AUC gate = 0.90** with no manual override, plus champion/challenger
  (`pipelines/retrain.py::decide_promotion`, shared by the DAG).
- **PSI = 0.20** monitored on exactly `txn_velocity_1h`, `amount_zscore`,
  `new_device_flag` (`config.MONITORED_FEATURES`).
- **No direct pushes to `main`** — branch, PR, CI-gated, merge.

## Drift monitoring (Evidently AI)

`monitoring/evidently_report.py` runs Evidently's `DataDriftPreset` with the
**PSI** stattest at threshold 0.20 over the 3 monitored features; if any feature
breaches, it triggers retraining + a Slack alert. A dependency-free hand-rolled
PSI is kept as a **fallback** so the trigger logic stays testable (and CI runnable)
without Evidently installed. The Airflow DAG runs `@weekly`, short-circuiting when
there's no drift — see `pipelines/airflow/README.md` for the Docker runbook.

## Project layout

```
config.py                    single source of truth (seq_len=50, PSI=0.20, AUC gate=0.90, ...)

data/synthetic.py            synthetic generator (fraud = behavioral burst)
data/ieee_cis.py             IEEE-CIS → canonical schema adapter (+ hybrid native-feature view)
data/sequence_builder.py     (N,50,F) tensors + mask, time-based split, scaler

features/feature_pipeline.py modular causal feature functions

models/lstm_model.py         RiskSequencer (2-layer LSTM + additive attention)
models/lgbm_baseline.py      LightGBM baseline + SHAP
models/hybrid.py             stacking meta-learner (aggregate, align, fit_stacker)

training/train.py            LSTM training loop (pos_weight BCE, ReduceLROnPlateau, MLflow-aware)
training/train_hybrid.py     hybrid LightGBM + LSTM training/eval on real IEEE-CIS
training/evaluate.py         AUC + threshold tuning @ FPR≤5%
training/business_metrics.py $ caught / FP cost / net savings + profit curve
training/hyperparameter_search.py   Optuna / random search

serving/inference.py         SageMaker handlers (model/input/predict/output_fn)
serving/package_model.py      build a deployable model.tar.gz
serving/deploy_sagemaker.py   deploy endpoint, benchmark p99 (CloudWatch), auto-teardown

monitoring/evidently_report.py  Evidently AI drift (PSI) + dependency-free fallback
monitoring/slack_alerts.py      webhook alerts
pipelines/retrain.py            end-to-end retrain + promotion (gate + champion/challenger)
pipelines/retrain_dag.py        Airflow DAG (runs the loop end-to-end)
pipelines/airflow/              Dockerized Airflow (compose + Dockerfile + runbook)

notebooks/01_eda.ipynb          imbalance, velocity, amount, mutual information
notebooks/02_modeling.ipynb     LightGBM vs LSTM comparison (+ SHAP, ROC/PR)
notebooks/03_attention_viz.ipynb  attention heatmaps — why a sequence was flagged
notebooks/04_error_analysis.ipynb FN/FP analysis, business metrics, profit curve

tests/                          50 tests — features, sequences, model, packaging, drift, hybrid, DAG
```

## Gotcha: torch + LightGBM in one process (macOS)

Both bundle their own OpenMP runtime and **segfault** when loaded together (e.g.
`02_modeling.ipynb`, `training/train_hybrid.py`). Guard it by setting these
**before** importing either library: `OMP_NUM_THREADS=1` and
`KMP_DUPLICATE_LIB_OK=TRUE` (plus `torch.set_num_threads(1)` and LightGBM
`num_threads=1`). Locked in by `tests/test_torch_lgbm_coexist.py`, which runs in a
subprocess so a crash can't take down the whole suite.

## Status

| Phase | State |
|---|---|
| 0 — Environment & structure | ✅ config, CI, branch protection |
| 1 — Data & features | ✅ synthetic + IEEE-CIS adapter, causal features, sequences, time split, EDA |
| 2 — Modeling | ✅ LSTM+attention, LightGBM, **hybrid** (0.93 on real data), HPO |
| 3 — Explainability | ✅ attention viz, SHAP, error analysis, business metrics |
| 4 — Deployment | ✅ handlers + packaging + MLflow gate; **endpoint deployed & p99 45 ms verified** |
| 5 — Monitoring & retrain | ✅ Evidently AI (PSI), Slack, champion/challenger, **Airflow DAG runs end-to-end** |

**Honest scope note:** the SageMaker endpoint was deployed and benchmarked *once*
to verify p99, then torn down (it is not a standing production service). The
Airflow loop is built, scheduled, and demonstrated end-to-end, but has not been
operated in production for an extended period. Everything is reproducible from the
commands above.
