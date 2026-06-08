"""Turn per-transaction features into padded sequence tensors.

Output contract (HARD constraint): for each user we take their last
`config.SEQUENCE_LENGTH` (=50) transactions, **left-pad** shorter histories
with zeros, and emit a padding mask so the attention layer can ignore the
pad positions.

Shapes
------
X    : (N, 50, F)  float32   — feature sequences
mask : (N, 50)     bool      — True where the position is PADDING
y    : (N,)        float32   — 1 if any txn in the window is fraud, else 0

Scaling is fit on training data only (`fit_scaler`) and reused at val/test
and serve time to avoid leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from config import FEATURE_COLUMNS, LABEL_COL, SEQUENCE_LENGTH


@dataclass
class SequenceDataset:
    X: np.ndarray      # (N, T, F) float32
    mask: np.ndarray   # (N, T) bool — True = padding
    y: np.ndarray      # (N,) float32
    user_ids: np.ndarray


def fit_scaler(features_df: pd.DataFrame) -> RobustScaler:
    """Fit a RobustScaler on the flat feature matrix (training rows only)."""
    scaler = RobustScaler()
    scaler.fit(features_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64))
    return scaler


def _pad_left(arr: np.ndarray, length: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Left-pad a (t, F) array to (length, F); return (padded, mask)."""
    t = arr.shape[0]
    if t >= length:
        return arr[-length:], np.zeros(length, dtype=bool)
    pad = np.zeros((length - t, width), dtype=arr.dtype)
    padded = np.vstack([pad, arr])
    mask = np.zeros(length, dtype=bool)
    mask[: length - t] = True  # leading positions are padding
    return padded, mask


def build_sequences(
    features_df: pd.DataFrame,
    scaler: RobustScaler,
    seq_len: int = SEQUENCE_LENGTH,
) -> SequenceDataset:
    """Build left-padded, scaled sequence tensors, one row per user.

    Parameters
    ----------
    features_df : output of `features.feature_pipeline.build_features`.
    scaler : a RobustScaler already fit on training features.
    seq_len : window length (defaults to the mandated 50).
    """
    df = features_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    scaled = scaler.transform(df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)).astype(np.float32)
    df_scaled = pd.DataFrame(scaled, columns=FEATURE_COLUMNS)
    df_scaled["user_id"] = df["user_id"].to_numpy()
    df_scaled[LABEL_COL] = df[LABEL_COL].to_numpy()

    F = len(FEATURE_COLUMNS)
    X_list, mask_list, y_list, uid_list = [], [], [], []

    for uid, g in df_scaled.groupby("user_id", sort=False):
        arr = g[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        padded, mask = _pad_left(arr, seq_len, F)
        label = float(g[LABEL_COL].iloc[-seq_len:].max())  # fraud anywhere in window
        X_list.append(padded)
        mask_list.append(mask)
        y_list.append(label)
        uid_list.append(uid)

    return SequenceDataset(
        X=np.stack(X_list).astype(np.float32),
        mask=np.stack(mask_list),
        y=np.asarray(y_list, dtype=np.float32),
        user_ids=np.asarray(uid_list),
    )


def time_based_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows by global timestamp quantiles (no random, no leakage).

    Returns (train, val, test). Earliest `train_frac` of time goes to train,
    the next `val_frac` to validation, the remainder to test.
    """
    ts = df["timestamp"]
    t_train = ts.quantile(train_frac)
    t_val = ts.quantile(train_frac + val_frac)
    train = df[ts <= t_train]
    val = df[(ts > t_train) & (ts <= t_val)]
    test = df[ts > t_val]
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


if __name__ == "__main__":
    from data.synthetic import generate_transactions
    from features.feature_pipeline import build_features

    raw = generate_transactions(n_users=300, seed=1)
    feats = build_features(raw)
    tr, va, te = time_based_split(feats)
    sc = fit_scaler(tr)
    ds = build_sequences(tr, sc)
    print(f"X={ds.X.shape}  mask={ds.mask.shape}  y={ds.y.shape}  pos_rate={ds.y.mean():.3%}")
    assert ds.X.shape[1] == SEQUENCE_LENGTH
    assert ds.X.shape[2] == len(FEATURE_COLUMNS)
    print("OK: sequence builder produced (N, 50, F) tensors.")
