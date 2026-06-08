"""Synthetic transaction generator for RiskSequencer.

Produces a raw transaction log in the canonical schema so the entire
pipeline (features -> sequences -> model -> eval) runs end-to-end without
the Kaggle IEEE-CIS download or any AWS access.

Fraud is injected as a *behavioral sequence*, not a per-row property:
a victim experiences a device/country change followed by a burst of
rapid, larger-than-usual transactions. This is exactly the pattern the
sequence model + attention is meant to surface, so a well-built model
should clear the AUC bar on this data.

Usage
-----
>>> from data.synthetic import generate_transactions
>>> df = generate_transactions(n_users=2000, seed=42)
>>> df.columns.tolist()
['TransactionID', 'user_id', 'timestamp', 'amount', 'merchant_id',
 'merchant_category', 'country', 'device_id', 'is_fraud']
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "travel", "electronics",
    "fuel", "entertainment", "utilities", "online_retail",
]
COUNTRIES = ["US", "CA", "GB", "DE", "FR", "IN", "BR", "NG", "RU"]


def generate_transactions(
    n_users: int = 2000,
    fraud_user_rate: float = 0.06,
    min_txns: int = 5,
    max_txns: int = 80,
    start: str = "2023-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a raw transaction log with embedded fraud sequences.

    Parameters
    ----------
    n_users : number of distinct users.
    fraud_user_rate : fraction of users who experience an account takeover.
    min_txns, max_txns : per-user transaction-count bounds.
    start : ISO date for the earliest possible transaction.
    seed : RNG seed for reproducibility.

    Returns
    -------
    DataFrame in the canonical raw schema, sorted by (user_id, timestamp).
    """
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    rows: list[dict] = []
    txn_id = 0

    fraud_users = set(
        rng.choice(n_users, size=int(n_users * fraud_user_rate), replace=False).tolist()
    )

    for user in range(n_users):
        n_txns = int(rng.integers(min_txns, max_txns + 1))
        # Stable behavioral baseline per user.
        home_country = rng.choice(COUNTRIES[:5])  # legit users skew to low-risk
        home_device = f"dev_{user}_{int(rng.integers(0, 3))}"
        amount_mu = rng.uniform(20, 200)
        amount_sigma = amount_mu * rng.uniform(0.2, 0.5)
        # Inter-arrival times in hours (legit cadence).
        t = start_ts + pd.Timedelta(hours=float(rng.uniform(0, 24 * 30)))

        is_fraud_user = user in fraud_users
        # Pick the index where takeover begins (latter half of history).
        takeover_at = int(rng.integers(n_txns // 2, n_txns)) if is_fraud_user else n_txns

        for i in range(n_txns):
            in_fraud_burst = is_fraud_user and i >= takeover_at

            if in_fraud_burst:
                gap_h = float(rng.uniform(0.02, 0.5))      # rapid burst
                country = rng.choice(COUNTRIES[5:])        # high-risk geo
                device = f"dev_{user}_attacker"
                amount = float(abs(rng.normal(amount_mu * 4, amount_sigma * 2)))
                category = rng.choice(["electronics", "online_retail", "travel"])
                label = 1
            else:
                gap_h = float(rng.exponential(36))         # ~1.5 day cadence
                country = home_country if rng.random() > 0.05 else rng.choice(COUNTRIES)
                device = home_device if rng.random() > 0.08 else f"dev_{user}_{int(rng.integers(0, 5))}"
                amount = float(abs(rng.normal(amount_mu, amount_sigma)))
                category = rng.choice(MERCHANT_CATEGORIES)
                label = 0

            t = t + pd.Timedelta(hours=gap_h)
            rows.append(
                {
                    "TransactionID": txn_id,
                    "user_id": user,
                    "timestamp": t,
                    "amount": round(amount, 2),
                    "merchant_id": int(rng.integers(0, 500)),
                    "merchant_category": category,
                    "country": country,
                    "device_id": device,
                    "is_fraud": label,
                }
            )
            txn_id += 1

    df = pd.DataFrame(rows)
    return df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)


def main() -> None:
    import argparse

    from config import RAW_DIR, ensure_dirs

    parser = argparse.ArgumentParser(description="Generate synthetic transactions.")
    parser.add_argument("--n-users", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    ensure_dirs()
    out = args.out or (RAW_DIR / "transactions.parquet")
    df = generate_transactions(n_users=args.n_users, seed=args.seed)
    df.to_parquet(out, index=False)
    fraud_rate = df["is_fraud"].mean()
    print(f"Wrote {len(df):,} transactions for {df['user_id'].nunique():,} users -> {out}")
    print(f"Row-level fraud rate: {fraud_rate:.3%}")


if __name__ == "__main__":
    main()
