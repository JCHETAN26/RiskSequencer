"""Tests for the IEEE-CIS -> canonical adapter (no real download needed)."""

from __future__ import annotations

import pandas as pd

from data.ieee_cis import REFERENCE_DATE, to_canonical
from features.feature_pipeline import build_features
from config import FEATURE_COLUMNS

CANONICAL_COLS = [
    "TransactionID", "user_id", "timestamp", "amount",
    "merchant_id", "merchant_category", "country", "device_id", "is_fraud",
]


def _fake_txn() -> pd.DataFrame:
    # Two cards (-> two users), several transactions each.
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5],
            "isFraud": [0, 0, 1, 0, 1],
            "TransactionDT": [86400, 90000, 95000, 86500, 99999],
            "TransactionAmt": [50.0, 75.5, 800.0, 20.0, 1200.0],
            "ProductCD": ["W", "C", "C", "W", "R"],
            "card1": [1001, 1001, 1001, 2002, 2002],
            "addr1": [100.0, 100.0, 100.0, 200.0, 200.0],
            "addr2": [87.0, 87.0, 87.0, 60.0, 60.0],
            "P_emaildomain": ["gmail.com", "gmail.com", "yahoo.com", "aol.com", None],
        }
    )


def _fake_identity() -> pd.DataFrame:
    return pd.DataFrame(
        {"TransactionID": [1, 3, 4], "DeviceInfo": ["iOS", "Windows", "Android"]}
    )


def test_canonical_schema_columns():
    out = to_canonical(_fake_txn(), _fake_identity())
    assert set(out.columns) == set(CANONICAL_COLS)


def test_uid_proxy_groups_by_card_and_addr():
    out = to_canonical(_fake_txn())
    # card1+addr1 yields two distinct users.
    assert out["user_id"].nunique() == 2
    assert (out["user_id"] == "1001_100.0").sum() == 3


def test_timestamp_anchored_and_ordered():
    out = to_canonical(_fake_txn())
    assert out["timestamp"].min() == REFERENCE_DATE + pd.Timedelta(seconds=86400)
    for _, g in out.groupby("user_id"):
        assert g["timestamp"].is_monotonic_increasing


def test_device_defaults_to_unknown_when_no_identity_row():
    out = to_canonical(_fake_txn(), _fake_identity())
    # TransactionID 2 and 5 have no identity row -> unknown
    devs = dict(zip(out["TransactionID"], out["device_id"]))
    assert devs[2] == "unknown"
    assert devs[5] == "unknown"
    assert devs[1] == "iOS"


def test_flows_through_feature_pipeline_no_nans():
    out = to_canonical(_fake_txn(), _fake_identity())
    feats = build_features(out)
    for col in FEATURE_COLUMNS:
        assert col in feats.columns
    assert feats[FEATURE_COLUMNS].isna().sum().sum() == 0
    # ProductCD codes must encode to non-zero (i.e. be in the vocab).
    assert (feats["merchant_category_enc"] > 0).all()
