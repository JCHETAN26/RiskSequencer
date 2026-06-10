"""Tests for the Postgres store — run against SQLite so they need no DB server."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from config import MONITORED_FEATURES
from storage.db import (
    drift_from_db,
    fetch_monitored,
    get_engine,
    init_schema,
    insert_transactions,
    log_inference,
)


def _engine(tmp_path):
    eng = get_engine(f"sqlite:///{tmp_path/'t.db'}")
    init_schema(eng)
    return eng


def test_schema_and_transactions_roundtrip(tmp_path):
    eng = _engine(tmp_path)
    df = pd.DataFrame({
        "user_id": ["u1", "u2"],
        "event_time": [datetime(2024, 1, 1), datetime(2024, 1, 1, 1)],
        "amount": [10.0, 20.0],
        "merchant_category": ["grocery", "travel"],
        "is_fraud": [0, 1],
    })
    n = insert_transactions(eng, df)
    assert n == 2


def test_inference_log_and_fetch_monitored(tmp_path):
    eng = _engine(tmp_path)
    recs = [{
        "user_id": "u1", "event_time": datetime(2024, 1, 1),
        "fraud_probability": 0.8, "is_flagged": True, "model_version": "v1",
        "txn_velocity_1h": 3.0, "amount_zscore": 1.2, "new_device_flag": 1.0,
    }]
    assert log_inference(eng, recs) == 1
    out = fetch_monitored(eng)
    assert list(out.columns) == MONITORED_FEATURES
    assert len(out) == 1


def test_drift_from_db_detects_drift(tmp_path):
    eng = _engine(tmp_path)
    rng = np.random.default_rng(0)

    def rows(n, t0, vel_mu):
        return [{
            "user_id": f"u{i}", "event_time": t0,
            "fraud_probability": 0.1, "is_flagged": False, "model_version": "v1",
            "txn_velocity_1h": float(rng.normal(vel_mu, 1)),
            "amount_zscore": float(rng.normal(0, 1)),
            "new_device_flag": float(rng.integers(0, 2)),
        } for i in range(n)]

    # reference window: normal velocity; current window: drifted velocity
    log_inference(eng, rows(600, datetime(2024, 1, 1, 12), vel_mu=1.0))
    log_inference(eng, rows(600, datetime(2024, 1, 2, 12), vel_mu=6.0))

    result = drift_from_db(
        eng,
        reference_window=(datetime(2024, 1, 1), datetime(2024, 1, 2)),
        current_window=(datetime(2024, 1, 2), datetime(2024, 1, 3)),
    )
    assert result.should_retrain
    assert "txn_velocity_1h" in result.breached
