"""IEEE-CIS Fraud Detection -> canonical schema adapter.

Normalises the Kaggle IEEE-CIS dataset (``train_transaction.csv`` +
optional ``train_identity.csv``) into the same canonical transaction schema
the synthetic generator emits, so every downstream module — feature
pipeline, sequence builder, training, serving — runs unchanged.

The challenge with IEEE-CIS is that it has **no explicit user_id** and no
absolute timestamp. We reconstruct both with well-established proxies:

* ``timestamp``  — ``TransactionDT`` is a seconds offset from an (undocumented)
  reference; the community-standard anchor is 2017-12-01. We add the offset to
  that anchor to get an absolute, *correctly ordered* datetime (only ordering
  and inter-arrival gaps matter for our features, not the absolute date).
* ``user_id``    — there is no card/account id, so we build a uid proxy from
  ``card1`` + ``addr1`` (a common, if imperfect, identity proxy used across
  IEEE-CIS solutions). Configurable via ``uid_cols``.
* ``merchant_*`` — IEEE-CIS has no merchant; ``ProductCD`` is the closest
  category signal and ``P_emaildomain`` serves as a merchant/recipient proxy.
* ``country``    — ``addr2`` is a coarse country code.
* ``device_id``  — ``DeviceInfo`` from the identity table (``unknown`` if no
  identity row joins).

Canonical output columns (see ``config``):
    TransactionID, user_id, timestamp, amount, merchant_id,
    merchant_category, country, device_id, is_fraud
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Community-standard anchor for TransactionDT (seconds offset).
REFERENCE_DATE = pd.Timestamp("2017-12-01")

# Only the columns we actually map — IEEE-CIS has 393, reading all is wasteful.
_TXN_USECOLS = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
    "ProductCD", "card1", "addr1", "addr2", "P_emaildomain",
]
_ID_USECOLS = ["TransactionID", "DeviceInfo"]


def _build_uid(df: pd.DataFrame, uid_cols: list[str]) -> pd.Series:
    """Construct a stable string user-id proxy from the given columns."""
    parts = [df[c].astype("string").fillna("NA") for c in uid_cols]
    uid = parts[0]
    for p in parts[1:]:
        uid = uid.str.cat(p, sep="_")
    return uid


def to_canonical(
    txn: pd.DataFrame,
    identity: pd.DataFrame | None = None,
    uid_cols: tuple[str, ...] = ("card1", "addr1"),
) -> pd.DataFrame:
    """Map raw IEEE-CIS frames to the canonical transaction schema.

    Parameters
    ----------
    txn : a ``train_transaction``-shaped frame (must contain ``_TXN_USECOLS``).
    identity : optional ``train_identity``-shaped frame for ``DeviceInfo``.
    uid_cols : columns combined into the user-id proxy.

    Returns
    -------
    DataFrame in the canonical schema, sorted by (user_id, timestamp).
    """
    missing = [c for c in _TXN_USECOLS if c not in txn.columns]
    if missing:
        raise ValueError(f"transaction frame missing required columns: {missing}")

    out = pd.DataFrame()
    out["TransactionID"] = txn["TransactionID"].astype("int64")
    out["user_id"] = _build_uid(txn, list(uid_cols))
    out["timestamp"] = REFERENCE_DATE + pd.to_timedelta(txn["TransactionDT"], unit="s")
    out["amount"] = txn["TransactionAmt"].astype(float)
    out["merchant_category"] = txn["ProductCD"].astype("string").fillna("unknown")
    out["merchant_id"] = txn["P_emaildomain"].astype("string").fillna("unknown")
    out["country"] = txn["addr2"].astype("Int64").astype("string").fillna("unknown")
    out["is_fraud"] = txn["isFraud"].astype(int)

    if identity is not None and "DeviceInfo" in identity.columns:
        dev = identity[["TransactionID", "DeviceInfo"]].copy()
        dev["TransactionID"] = dev["TransactionID"].astype("int64")
        out = out.merge(dev, on="TransactionID", how="left")
        out["device_id"] = out.pop("DeviceInfo").astype("string").fillna("unknown")
    else:
        out["device_id"] = "unknown"

    out = out.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return out


def load_ieee_cis(
    transaction_csv: str | Path,
    identity_csv: str | Path | None = None,
    nrows: int | None = None,
    uid_cols: tuple[str, ...] = ("card1", "addr1"),
) -> pd.DataFrame:
    """Read the IEEE-CIS CSV(s) from disk and return the canonical schema.

    Reads only the columns we map (memory-friendly on the full ~590K x 393 file).
    Pass ``nrows`` to sample for a quick smoke test.
    """
    txn = pd.read_csv(transaction_csv, usecols=_TXN_USECOLS, nrows=nrows)
    identity = None
    if identity_csv is not None and Path(identity_csv).exists():
        identity = pd.read_csv(identity_csv, usecols=_ID_USECOLS)
    return to_canonical(txn, identity, uid_cols=uid_cols)


def native_feature_columns(all_columns: list[str]) -> list[str]:
    """Select IEEE-CIS's native numeric signal columns for the LightGBM layer.

    These are the engineered features where most of IEEE-CIS's predictive power
    lives — the Vesta ``V*`` features, the ``C*`` counts, ``D*`` timedeltas, plus
    a few raw numerics. LightGBM handles their NaNs natively, so no imputation.
    """
    keep = [c for c in all_columns if c.startswith(("C", "D", "V"))]
    for c in ("TransactionAmt", "card1", "card2", "card3", "card5",
              "addr1", "addr2", "dist1", "dist2"):
        if c in all_columns:
            keep.append(c)
    # exclude the time/id/label columns even if they match a prefix
    return [c for c in keep if c not in ("TransactionID", "TransactionDT")]


def load_hybrid(
    transaction_csv: str | Path,
    identity_csv: str | Path | None = None,
    nrows: int | None = None,
    uid_cols: tuple[str, ...] = ("card1", "addr1"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both views in one pass, sharing user_id/timestamp/label for the hybrid.

    Returns ``(canonical_df, native_df)``:
      * ``canonical_df`` — the schema the behavioral feature pipeline + LSTM consume.
      * ``native_df`` — ``[TransactionID, user_id, timestamp, is_fraud]`` plus the
        native numeric feature columns for the LightGBM layer.

    Both frames carry the same ``user_id`` (uid proxy) and ``timestamp`` so a
    time-based split lines the two models up on identical rows.
    """
    all_cols = pd.read_csv(transaction_csv, nrows=0).columns.tolist()
    native_cols = native_feature_columns(all_cols)
    read_cols = sorted(set(_TXN_USECOLS) | set(native_cols))
    txn = pd.read_csv(transaction_csv, usecols=read_cols, nrows=nrows)

    identity = None
    if identity_csv is not None and Path(identity_csv).exists():
        identity = pd.read_csv(identity_csv, usecols=_ID_USECOLS)

    canonical = to_canonical(txn, identity, uid_cols=uid_cols)

    native = pd.DataFrame()
    native["TransactionID"] = txn["TransactionID"].astype("int64")
    native["user_id"] = _build_uid(txn, list(uid_cols))
    native["timestamp"] = REFERENCE_DATE + pd.to_timedelta(txn["TransactionDT"], unit="s")
    native["is_fraud"] = txn["isFraud"].astype(int)
    for c in native_cols:
        native[c] = txn[c]
    native = native.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return canonical, native


def main() -> None:
    import argparse

    from config import RAW_DIR, ensure_dirs

    parser = argparse.ArgumentParser(
        description="Convert IEEE-CIS CSVs to canonical transactions.parquet."
    )
    parser.add_argument(
        "--transaction-csv", type=Path,
        default=RAW_DIR / "train_transaction.csv",
        help="path to train_transaction.csv",
    )
    parser.add_argument(
        "--identity-csv", type=Path,
        default=RAW_DIR / "train_identity.csv",
        help="path to train_identity.csv (optional)",
    )
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.transaction_csv.exists():
        raise SystemExit(
            f"{args.transaction_csv} not found.\n"
            "Download the IEEE-CIS Fraud Detection dataset from Kaggle:\n"
            "  kaggle competitions download -c ieee-fraud-detection\n"
            f"and unzip train_transaction.csv / train_identity.csv into {RAW_DIR}/"
        )

    ensure_dirs()
    df = load_ieee_cis(
        args.transaction_csv,
        args.identity_csv if args.identity_csv.exists() else None,
        nrows=args.nrows,
    )
    out = args.out or (RAW_DIR / "transactions.parquet")
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df):,} transactions for {df['user_id'].nunique():,} users -> {out}")
    print(f"Row-level fraud rate: {df['is_fraud'].mean():.3%}")
    print(f"Median txns/user: {df.groupby('user_id').size().median():.0f}")


if __name__ == "__main__":
    main()
