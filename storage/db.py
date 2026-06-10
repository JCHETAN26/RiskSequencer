"""PostgreSQL store for transactions + model predictions (the monitoring source).

This is the data backbone of the production loop: instead of reading drift from
files, every prediction is logged to Postgres along with the monitored feature
values, and the drift monitor queries reference-vs-current windows straight from
the table. That closes the loop: inference -> log -> drift -> retrain.

Written with SQLAlchemy Core (no Postgres-only SQL), so the exact same code runs
against Postgres in production and SQLite in tests/CI.

Connection:
  * DATABASE_URL env var, or pass `url=` (e.g. sqlite:///... in tests).
  * Default: postgresql+psycopg2://risk:risk@localhost:5432/risksequencer
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    select,
)
from sqlalchemy.engine import Engine

from config import MONITORED_FEATURES

DEFAULT_URL = "postgresql+psycopg2://risk:risk@localhost:5432/risksequencer"

metadata = MetaData()

# Raw transactions (the data the model scores).
transactions = Table(
    "transactions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", String, index=True),
    Column("event_time", DateTime, index=True),
    Column("amount", Float),
    Column("merchant_category", String),
    Column("is_fraud", Integer),
)

# Inference log: one row per prediction, with the monitored features snapshotted
# so drift can be computed directly from what the model actually saw in prod.
inference_log = Table(
    "inference_log", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", String, index=True),
    Column("event_time", DateTime, index=True),
    Column("fraud_probability", Float),
    Column("is_flagged", Boolean),
    Column("model_version", String),
    # monitored behavioral features (the drift signals)
    Column("txn_velocity_1h", Float),
    Column("amount_zscore", Float),
    Column("new_device_flag", Float),
)


def get_engine(url: str | None = None) -> Engine:
    """Create an Engine from `url`, the DATABASE_URL env var, or the default."""
    return create_engine(url or os.environ.get("DATABASE_URL", DEFAULT_URL), future=True)


def init_schema(engine: Engine) -> None:
    """Create the tables if they don't exist."""
    metadata.create_all(engine)


def insert_transactions(engine: Engine, df: pd.DataFrame) -> int:
    """Bulk-insert raw transactions. Expects user_id/event_time/amount/... columns."""
    cols = ["user_id", "event_time", "amount", "merchant_category", "is_fraud"]
    rows = df[[c for c in cols if c in df.columns]].to_dict("records")
    with engine.begin() as conn:
        conn.execute(insert(transactions), rows)
    return len(rows)


def log_inference(engine: Engine, records: list[dict]) -> int:
    """Append prediction rows to the inference log.

    Each record: user_id, event_time, fraud_probability, is_flagged,
    model_version, and the 3 monitored feature values.
    """
    with engine.begin() as conn:
        conn.execute(insert(inference_log), records)
    return len(records)


def record_predictions(
    engine: Engine,
    features_df: pd.DataFrame,
    probabilities,
    threshold: float,
    model_version: str,
) -> int:
    """Log a batch of predictions + their monitored-feature snapshot.

    `features_df` must carry user_id, timestamp, and the 3 monitored features
    (the output of the feature pipeline). One inference_log row per input row.
    """
    records = []
    for (_, row), prob in zip(features_df.iterrows(), probabilities):
        records.append({
            "user_id": str(row["user_id"]),
            "event_time": pd.to_datetime(row["timestamp"]).to_pydatetime(),
            "fraud_probability": float(prob),
            "is_flagged": bool(prob >= threshold),
            "model_version": model_version,
            **{f: float(row[f]) for f in MONITORED_FEATURES},
        })
    return log_inference(engine, records)


def fetch_monitored(
    engine: Engine,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Return the monitored feature columns from the inference log for a window."""
    cols = [inference_log.c[f] for f in MONITORED_FEATURES]
    stmt = select(*cols)
    if start is not None:
        stmt = stmt.where(inference_log.c.event_time >= start)
    if end is not None:
        stmt = stmt.where(inference_log.c.event_time < end)
    with engine.connect() as conn:
        rows = conn.execute(stmt).fetchall()
    return pd.DataFrame(rows, columns=MONITORED_FEATURES)


def reference_current_split(engine: Engine, frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the logged monitored features by time into (reference, current).

    Convenience for the scheduled DAG: the older `frac` of logged inference rows
    is the reference window, the rest is the current window.
    """
    cols = [inference_log.c[f] for f in MONITORED_FEATURES]
    stmt = select(*cols).order_by(inference_log.c.event_time)
    with engine.connect() as conn:
        rows = conn.execute(stmt).fetchall()
    df = pd.DataFrame(rows, columns=MONITORED_FEATURES)
    k = int(len(df) * frac)
    return df.iloc[:k].reset_index(drop=True), df.iloc[k:].reset_index(drop=True)


def drift_from_db(
    engine: Engine,
    reference_window: tuple[datetime, datetime],
    current_window: tuple[datetime, datetime],
):
    """Pull reference + current windows from Postgres and run drift detection."""
    from monitoring.evidently_report import check_drift

    ref = fetch_monitored(engine, *reference_window)
    cur = fetch_monitored(engine, *current_window)
    return check_drift(ref, cur)
