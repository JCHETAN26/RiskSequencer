"""Shared pytest fixtures + path setup so `config`, `data`, etc. import."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the project root importable when running `pytest` from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.synthetic import generate_transactions  # noqa: E402
from features.feature_pipeline import build_features  # noqa: E402


@pytest.fixture(scope="session")
def raw_df():
    return generate_transactions(n_users=200, seed=7)


@pytest.fixture(scope="session")
def features_df(raw_df):
    return build_features(raw_df)
