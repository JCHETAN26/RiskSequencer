"""Tests for the feature pipeline — focus on causality (no leakage)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS
from features.feature_pipeline import (
    amount_zscore,
    build_features,
    first_seen_flag,
    time_since_last,
)


def test_all_feature_columns_present_no_nans(features_df):
    for col in FEATURE_COLUMNS:
        assert col in features_df.columns, f"missing feature {col}"
    assert features_df[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_first_seen_flag_first_row_is_one():
    g = pd.DataFrame({"device_id": ["a", "a", "b", "a"]})
    flags = first_seen_flag(g, "device_id").tolist()
    assert flags == [1.0, 0.0, 1.0, 0.0]


def test_time_since_last_first_is_zero():
    g = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 01:30"])}
    )
    secs = time_since_last(g).tolist()
    assert secs[0] == 0.0
    assert secs[1] == 3600.0
    assert secs[2] == 1800.0


def test_amount_zscore_is_causal():
    # The z-score of row i must not depend on rows > i. Appending a huge
    # future amount must leave earlier z-scores unchanged.
    base = pd.DataFrame({"amount": [100.0, 110.0, 90.0, 105.0]})
    extended = pd.DataFrame({"amount": [100.0, 110.0, 90.0, 105.0, 99999.0]})
    z_base = amount_zscore(base).to_numpy()
    z_ext = amount_zscore(extended).to_numpy()[:4]
    assert np.allclose(z_base, z_ext), "amount_zscore leaks future information"


def test_first_row_zscore_is_zero():
    g = pd.DataFrame({"amount": [100.0, 200.0]})
    z = amount_zscore(g).tolist()
    assert z[0] == 0.0  # no history -> 0


def test_velocity_monotonic_within_window():
    # Two txns one minute apart -> the second has velocity_1h == 2.
    df = pd.DataFrame(
        {
            "user_id": [1, 1],
            "timestamp": pd.to_datetime(["2023-01-01 00:00", "2023-01-01 00:01"]),
            "amount": [10.0, 20.0],
            "merchant_id": [1, 2],
            "merchant_category": ["grocery", "fuel"],
            "country": ["US", "US"],
            "device_id": ["d1", "d1"],
            "is_fraud": [0, 0],
        }
    )
    feats = build_features(df)
    assert feats["txn_velocity_1h"].iloc[1] == 2.0
