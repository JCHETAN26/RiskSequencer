"""Tests for sequence construction: shape, padding, masking, no leakage."""

from __future__ import annotations

import numpy as np

from config import FEATURE_COLUMNS, SEQUENCE_LENGTH
from data.sequence_builder import (
    build_sequences,
    fit_scaler,
    time_based_split,
)


def test_output_shape_is_N_50_F(features_df):
    scaler = fit_scaler(features_df)
    ds = build_sequences(features_df, scaler)
    n_users = features_df["user_id"].nunique()
    assert ds.X.shape == (n_users, SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
    assert ds.mask.shape == (n_users, SEQUENCE_LENGTH)
    assert ds.y.shape == (n_users,)


def test_left_padding_and_mask_alignment(features_df):
    scaler = fit_scaler(features_df)
    ds = build_sequences(features_df, scaler)
    # For any padded row, padding must be on the LEFT and exactly match mask.
    for i in range(min(20, ds.X.shape[0])):
        mask = ds.mask[i]
        if mask.any():
            assert mask[0], "padding must be left-aligned"
            # mask is a contiguous leading block
            n_pad = int(mask.sum())
            assert mask[:n_pad].all() and not mask[n_pad:].any()
            # padded positions are all zeros
            assert np.allclose(ds.X[i, :n_pad], 0.0)


def test_time_split_is_ordered(features_df):
    tr, va, te = time_based_split(features_df)
    assert tr["timestamp"].max() <= va["timestamp"].min()
    assert va["timestamp"].max() <= te["timestamp"].min()
    assert len(tr) and len(va) and len(te)


def test_scaler_fit_on_train_only(features_df):
    # Sanity: scaler transforms produce finite values on val data.
    tr, va, _ = time_based_split(features_df)
    scaler = fit_scaler(tr)
    ds = build_sequences(va, scaler)
    assert np.isfinite(ds.X).all()
