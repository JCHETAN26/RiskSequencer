"""End-to-end demo of the Postgres-backed monitoring loop (real DB).

Shows the production data flow concretely:
  1. train a quick model + score users
  2. LOG predictions (+ monitored features) to Postgres  (the inference log)
  3. read a reference window and a current window back FROM Postgres
  4. run Evidently drift on what was logged -> decide whether to retrain

Run against a real Postgres (see storage/docker-compose.yaml):
  export DATABASE_URL=postgresql+psycopg2://risk:risk@localhost:5432/risksequencer
  python -m storage.seed_demo
"""

from __future__ import annotations

import numpy as np

from config import MONITORED_FEATURES, TrainConfig
from data.sequence_builder import build_sequences, fit_scaler, time_based_split
from data.synthetic import generate_transactions
from features.feature_pipeline import build_features
from storage.db import (
    drift_from_db,
    get_engine,
    init_schema,
    insert_transactions,
    record_predictions,
)
from training.evaluate import tune_threshold
from training.train import _scores, run_training


def main() -> None:
    engine = get_engine()
    init_schema(engine)
    print(f"[db] connected: {engine.url}")

    # --- 1. data + a quick model ------------------------------------------
    raw = generate_transactions(n_users=2000, seed=42)
    feats = build_features(raw)
    tr, va, te = time_based_split(feats)
    scaler = fit_scaler(tr)
    val_ds = build_sequences(va, scaler)
    out = run_training(build_sequences(tr, scaler), val_ds, TrainConfig(max_epochs=5))
    model = out.model
    threshold = tune_threshold(val_ds.y, _scores(model, val_ds, "cpu"))

    # score every user once (per-user probability)
    full_ds = build_sequences(feats, scaler)
    import torch
    probs = model.predict_proba(torch.from_numpy(full_ds.X),
                                torch.from_numpy(full_ds.mask)).numpy()
    prob_by_user = dict(zip(full_ds.user_ids, probs))

    # store raw transactions too (the data the model scored)
    tx = raw.rename(columns={"timestamp": "event_time"})
    insert_transactions(engine, tx)

    # --- 2. log predictions to Postgres -----------------------------------
    # reference window = training period (normal behavior)
    ref = tr.copy()
    ref_probs = [prob_by_user.get(u, 0.0) for u in ref["user_id"]]
    n_ref = record_predictions(engine, ref, ref_probs, threshold, "lstm-v1")

    # current window = test period, with INJECTED drift on txn_velocity_1h
    cur = te.copy()
    cur["txn_velocity_1h"] = cur["txn_velocity_1h"] * 5.0   # simulate a behavior shift
    cur_probs = [prob_by_user.get(u, 0.0) for u in cur["user_id"]]
    n_cur = record_predictions(engine, cur, cur_probs, threshold, "lstm-v1")
    print(f"[db] logged predictions: {n_ref} reference + {n_cur} current rows")

    # --- 3 + 4. read windows back from Postgres and run drift -------------
    import pandas as pd

    ref_window = (ref["timestamp"].min().to_pydatetime(), ref["timestamp"].max().to_pydatetime())
    cur_window = (cur["timestamp"].min().to_pydatetime(),
                  (cur["timestamp"].max() + pd.Timedelta(seconds=1)).to_pydatetime())
    result = drift_from_db(engine, ref_window, cur_window)

    print(f"\n[drift] engine={result.engine}")
    for f in MONITORED_FEATURES:
        print(f"   PSI {f:18s}= {result.psi_by_feature.get(f, float('nan')):.3f}")
    print(f"[drift] breached: {result.breached}")
    print(f"[decision] retrain? {result.should_retrain}  "
          f"(triggered by PSI > 0.20 read from Postgres)")


if __name__ == "__main__":
    main()
