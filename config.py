"""Central configuration for RiskSequencer.

Single source of truth for the constants that the spec marks as
non-negotiable (sequence length, PSI threshold, monitored features,
AUC gate). Importing these from one place keeps the feature pipeline,
sequence builder, training, serving, and monitoring code in agreement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- Filesystem layout -----------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SEQUENCES_DIR = DATA_DIR / "sequences"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# --- S3 layout (override with real bucket names via env in prod) -----------
S3_DATA = "s3://risksequencer-data"
S3_MODELS = "s3://risksequencer-models"
S3_LOGS = "s3://risksequencer-logs"

# --- Canonical raw transaction schema --------------------------------------
# The synthetic generator and the IEEE-CIS adapter both normalise to this.
ID_COL = "user_id"
TIME_COL = "timestamp"
LABEL_COL = "is_fraud"
AMOUNT_COL = "amount"

# --- Sequence construction (HARD constraint: seq_len = 50) -----------------
SEQUENCE_LENGTH = 50
PAD_VALUE = 0.0  # left-pad shorter histories with zeros, mask in attention

# --- Feature columns produced by features/feature_pipeline.py --------------
# Order matters: this defines the F axis of the (N, 50, F) tensor.
FEATURE_COLUMNS: list[str] = [
    # velocity
    "txn_velocity_1h",
    "txn_velocity_6h",
    "txn_velocity_24h",
    # time
    "hour_of_day",
    "day_of_week",
    "time_since_last_txn",
    # amount
    "amount",
    "amount_zscore",
    "amount_delta",
    # location / device
    "new_country_flag",
    "new_device_flag",
    "distance_from_last_txn",
    # merchant
    "new_merchant_flag",
    "merchant_category_enc",
    # account
    "account_age_days",
]

# Features watched by Evidently for drift (HARD constraint: exactly these 3).
MONITORED_FEATURES: list[str] = [
    "txn_velocity_1h",
    "amount_zscore",
    "new_device_flag",
]

# --- Thresholds & gates ----------------------------------------------------
PSI_THRESHOLD = 0.20          # retraining trigger
MLFLOW_AUC_GATE = 0.90        # promotion gate — no manual override
TARGET_TEST_AUC = 0.94
TARGET_PRECISION_AT_5PCT_FPR = 0.88
P99_LATENCY_MS = 50.0

# --- MLflow ----------------------------------------------------------------
MLFLOW_EXPERIMENTS = {
    "lgbm": "risksequencer/lgbm-baseline",
    "lstm": "risksequencer/lstm-experiments",
    "prod": "risksequencer/production-candidates",
}
REGISTERED_MODEL_NAME = "risksequencer-prod"


@dataclass
class ModelConfig:
    """LSTM hyperparameters (defaults match the reference architecture)."""

    input_size: int = len(FEATURE_COLUMNS)
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3


@dataclass
class TrainConfig:
    """Training-loop hyperparameters."""

    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 30
    early_stopping_patience: int = 5
    lr_scheduler_patience: int = 2
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)


def ensure_dirs() -> None:
    """Create the local data/artifact directories if missing."""
    for d in (RAW_DIR, PROCESSED_DIR, SEQUENCES_DIR, ARTIFACTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
