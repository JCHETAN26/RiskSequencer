"""Feature engineering for RiskSequencer.

Every feature is computed **causally**: for a given transaction, only that
transaction and the user's *earlier* transactions are used. This is the
single most important property here — any feature that peeks at future
rows leaks the label and inflates AUC.

Each public function takes a per-user, time-sorted DataFrame and returns a
Series (or small frame) aligned to its index, so functions are independently
unit-testable. `build_features` wires them together across all users via a
groupby and emits exactly `config.FEATURE_COLUMNS`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS

# Reference point used to anchor the merchant-category ordinal encoding so
# train/serve encodings match regardless of which categories appear.
_MERCHANT_CATEGORIES = [
    # synthetic generator vocabulary
    "grocery", "restaurant", "travel", "electronics",
    "fuel", "entertainment", "utilities", "online_retail",
    # IEEE-CIS ProductCD codes (so the adapter encodes without "unknown")
    "W", "C", "R", "H", "S",
]
_MERCHANT_ENC = {c: i + 1 for i, c in enumerate(_MERCHANT_CATEGORIES)}  # 0 = unknown


# --- Velocity --------------------------------------------------------------
def velocity_counts(g: pd.DataFrame, window: str) -> pd.Series:
    """Count of the user's transactions within `window` ending at each row.

    Includes the current row (it is known at scoring time). Implemented with
    a time-indexed rolling count, which is causal by construction.
    """
    s = pd.Series(1, index=pd.DatetimeIndex(g["timestamp"]))
    counts = s.rolling(window).count()
    return pd.Series(counts.to_numpy(), index=g.index)


# --- Time ------------------------------------------------------------------
def time_since_last(g: pd.DataFrame) -> pd.Series:
    """Seconds since the user's previous transaction (0 for the first)."""
    dt = g["timestamp"].diff().dt.total_seconds()
    return dt.fillna(0.0)


# --- Amount ----------------------------------------------------------------
def amount_zscore(g: pd.DataFrame) -> pd.Series:
    """Z-score of amount vs the user's history *excluding* the current row.

    Uses expanding mean/std shifted by one so the current amount never
    contributes to its own normalisation (no leakage). Early rows with no
    history get 0.
    """
    amt = g["amount"]
    mean = amt.expanding().mean().shift(1)
    std = amt.expanding().std().shift(1)
    z = (amt - mean) / std.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def amount_delta(g: pd.DataFrame) -> pd.Series:
    """Signed change in amount vs the previous transaction (0 for first)."""
    return g["amount"].diff().fillna(0.0)


# --- First-seen flags ------------------------------------------------------
def first_seen_flag(g: pd.DataFrame, col: str) -> pd.Series:
    """1 the first time a value of `col` appears for this user, else 0.

    `duplicated()` marks every occurrence after the first as a duplicate, so
    the negation gives a causal "new entity" flag.
    """
    return (~g[col].duplicated()).astype(float)


# --- Location distance -----------------------------------------------------
def distance_from_last(g: pd.DataFrame) -> pd.Series:
    """Proxy distance from the previous transaction.

    Without lat/long we use a country-change indicator scaled to a coarse
    "distance" so the model sees a continuous geo-movement signal. Real prod
    data should swap this for haversine on resolved coordinates.
    """
    changed = (g["country"] != g["country"].shift(1)).astype(float)
    changed.iloc[0] = 0.0
    return changed * 1000.0


def _per_user(df: pd.DataFrame, func) -> np.ndarray:
    """Apply a per-user function and return one array aligned to `df`'s rows.

    `df` must already be sorted so each user's rows are contiguous; groups are
    iterated in that order and concatenated, sidestepping the shape ambiguity
    of `GroupBy.apply` across pandas versions.
    """
    parts = [np.asarray(func(g), dtype=float) for _, g in df.groupby("user_id", sort=False)]
    return np.concatenate(parts)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all per-transaction features for a raw transaction log.

    Parameters
    ----------
    df : raw transactions in the canonical schema. May span many users.

    Returns
    -------
    The input rows (sorted by user, time) with every column in
    `config.FEATURE_COLUMNS` added. Index is reset.
    """
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    out = df.copy()

    # Velocity (time-windowed counts).
    out["txn_velocity_1h"] = _per_user(out, lambda g: velocity_counts(g, "1h"))
    out["txn_velocity_6h"] = _per_user(out, lambda g: velocity_counts(g, "6h"))
    out["txn_velocity_24h"] = _per_user(out, lambda g: velocity_counts(g, "24h"))

    # Time.
    out["hour_of_day"] = out["timestamp"].dt.hour.astype(float)
    out["day_of_week"] = out["timestamp"].dt.dayofweek.astype(float)
    out["time_since_last_txn"] = _per_user(out, time_since_last)

    # Amount.
    out["amount_zscore"] = _per_user(out, amount_zscore)
    out["amount_delta"] = _per_user(out, amount_delta)

    # Location / device.
    out["new_country_flag"] = _per_user(out, lambda g: first_seen_flag(g, "country"))
    out["new_device_flag"] = _per_user(out, lambda g: first_seen_flag(g, "device_id"))
    out["distance_from_last_txn"] = _per_user(out, distance_from_last)

    # Merchant.
    out["new_merchant_flag"] = _per_user(out, lambda g: first_seen_flag(g, "merchant_id"))
    out["merchant_category_enc"] = (
        out["merchant_category"].map(_MERCHANT_ENC).fillna(0).astype(float)
    )

    # Account: age in days since the user's first observed transaction.
    first_ts = out.groupby("user_id", sort=False)["timestamp"].transform("min")
    out["account_age_days"] = (
        (out["timestamp"] - first_ts).dt.total_seconds() / 86400.0
    )

    missing = [c for c in FEATURE_COLUMNS if c not in out.columns]
    if missing:
        raise RuntimeError(f"feature pipeline did not produce: {missing}")

    return out


if __name__ == "__main__":
    from data.synthetic import generate_transactions

    sample = generate_transactions(n_users=50, seed=0)
    feats = build_features(sample)
    print(feats[FEATURE_COLUMNS].describe().T)
    assert feats[FEATURE_COLUMNS].isna().sum().sum() == 0, "NaNs in features!"
    print("\nOK: no NaNs, all feature columns present.")
