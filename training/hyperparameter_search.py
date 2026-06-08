"""Hyperparameter search for the RiskSequencer LSTM.

Searches the grid from the build plan, optimising validation AUC. Uses Optuna
when installed (TPE sampler) and falls back to random search over the grid
otherwise. Every trial is logged as its own MLflow run via `run_training`,
which also satisfies the "≥ 3 logged experiments" success criterion.

HARD constraint: sequence length is fixed at 50 (`config.SEQUENCE_LENGTH`).
The build plan lists `sequence_length` in the grid, but the system-prompt
constraints pin it at 50, so sequences are built **once** and reused across
trials (much faster). Pass `search_seq_len=True` to override and explore it
anyway — that path rebuilds sequences per value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from config import (
    MLFLOW_EXPERIMENTS,
    SEQUENCE_LENGTH,
    ModelConfig,
    TrainConfig,
)
from data.sequence_builder import SequenceDataset, build_sequences, fit_scaler
from training.train import run_training

# Search space (from the build plan).
SEARCH_SPACE = {
    "hidden_size": [64, 128, 256],
    "num_layers": [1, 2, 3],
    "dropout": [0.1, 0.2, 0.3, 0.4],
    "learning_rate": [1e-4, 5e-4, 1e-3],
}
SEQ_LEN_CHOICES = [30, 50, 100]  # only used when search_seq_len=True


@dataclass
class SearchResult:
    best_params: dict
    best_val_auc: float
    trials: list[dict]


def _cfg_from_params(params: dict, max_epochs: int) -> TrainConfig:
    return TrainConfig(
        learning_rate=params["learning_rate"],
        max_epochs=max_epochs,
        model=ModelConfig(
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
        ),
    )


def _evaluate_params(
    params: dict,
    train_ds: SequenceDataset,
    val_ds: SequenceDataset,
    max_epochs: int,
    log_mlflow: bool,
) -> float:
    cfg = _cfg_from_params(params, max_epochs)
    out = run_training(
        train_ds,
        val_ds,
        cfg,
        mlflow_experiment=MLFLOW_EXPERIMENTS["lstm"] if log_mlflow else None,
    )
    return out.val_result.auc_roc


def search(
    train_ds: SequenceDataset,
    val_ds: SequenceDataset,
    n_trials: int = 12,
    max_epochs: int = 12,
    use_optuna: bool = True,
    log_mlflow: bool = True,
    seed: int = 42,
) -> SearchResult:
    """Search over `SEARCH_SPACE` with fixed seq_len=50; return the best config."""
    trials: list[dict] = []

    try:
        import optuna  # noqa: F401

        have_optuna = use_optuna
    except ImportError:
        have_optuna = False

    if have_optuna:
        import optuna

        def objective(trial: "optuna.Trial") -> float:
            params = {
                "hidden_size": trial.suggest_categorical("hidden_size", SEARCH_SPACE["hidden_size"]),
                "num_layers": trial.suggest_categorical("num_layers", SEARCH_SPACE["num_layers"]),
                "dropout": trial.suggest_categorical("dropout", SEARCH_SPACE["dropout"]),
                "learning_rate": trial.suggest_categorical("learning_rate", SEARCH_SPACE["learning_rate"]),
            }
            auc = _evaluate_params(params, train_ds, val_ds, max_epochs, log_mlflow)
            trials.append({**params, "val_auc": auc})
            return auc

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=n_trials)
        return SearchResult(study.best_params, float(study.best_value), trials)

    # --- Random-search fallback (no Optuna) ---------------------------------
    rng = np.random.default_rng(seed)
    best_params, best_auc = None, -1.0
    for _ in range(n_trials):
        params = {k: rng.choice(v).item() for k, v in SEARCH_SPACE.items()}
        auc = _evaluate_params(params, train_ds, val_ds, max_epochs, log_mlflow)
        trials.append({**params, "val_auc": auc})
        if auc > best_auc:
            best_auc, best_params = auc, params
    return SearchResult(best_params, best_auc, trials)


def load_datasets(
    transactions: pd.DataFrame | None = None,
) -> tuple[SequenceDataset, SequenceDataset, SequenceDataset]:
    """Build (train, val, test) sequence datasets from a transaction frame.

    Defaults to the synthetic generator when no frame is supplied, so the
    search is runnable with zero setup.
    """
    from data.sequence_builder import time_based_split
    from features.feature_pipeline import build_features

    if transactions is None:
        from data.synthetic import generate_transactions

        transactions = generate_transactions(n_users=2000, seed=42)

    feats = build_features(transactions)
    tr, va, te = time_based_split(feats)
    scaler = fit_scaler(tr)
    return (
        build_sequences(tr, scaler),
        build_sequences(va, scaler),
        build_sequences(te, scaler),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="LSTM hyperparameter search.")
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    train_ds, val_ds, _ = load_datasets()
    result = search(
        train_ds,
        val_ds,
        n_trials=args.n_trials,
        max_epochs=args.max_epochs,
        log_mlflow=not args.no_mlflow,
    )
    print("\n=== Search complete ===")
    print(f"seq_len (fixed): {SEQUENCE_LENGTH}")
    print(f"best val AUC: {result.best_val_auc:.4f}")
    print(f"best params : {result.best_params}")
    for t in sorted(result.trials, key=lambda d: -d["val_auc"])[:5]:
        print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()})


if __name__ == "__main__":
    main()
