"""Training loop for the RiskSequencer LSTM.

Handles class imbalance with `BCEWithLogitsLoss(pos_weight=...)` (HARD
constraint: no blind oversampling), tunes the LR on validation AUC with
`ReduceLROnPlateau`, and early-stops on validation AUC. Logs to MLflow when
available; otherwise runs fine and just prints.

Entry point `run_training` returns the best model + its validation
EvalResult so the same function is callable from a notebook, a SageMaker
training job, or the Airflow retrain DAG.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import TrainConfig
from data.sequence_builder import SequenceDataset
from models.lstm_model import RiskSequencer
from training.evaluate import EvalResult, evaluate

try:
    import mlflow
except ImportError:
    mlflow = None


@dataclass
class TrainOutput:
    model: RiskSequencer
    val_result: EvalResult
    best_epoch: int
    history: list[dict]


def _loader(ds: SequenceDataset, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = TensorDataset(
        torch.from_numpy(ds.X),
        torch.from_numpy(ds.mask),
        torch.from_numpy(ds.y),
    )
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)


def _pos_weight(y: np.ndarray) -> torch.Tensor:
    pos = float(y.sum())
    neg = float(len(y) - pos)
    return torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)


@torch.no_grad()
def _scores(model: RiskSequencer, ds: SequenceDataset, device: str) -> np.ndarray:
    model.eval()
    out = []
    for xb, mb, _ in _loader(ds, 1024, shuffle=False):
        p = model.predict_proba(xb.to(device), mb.to(device))
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def run_training(
    train_ds: SequenceDataset,
    val_ds: SequenceDataset,
    cfg: TrainConfig | None = None,
    device: str | None = None,
    mlflow_experiment: str | None = None,
) -> TrainOutput:
    cfg = cfg or TrainConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = RiskSequencer(
        input_size=cfg.model.input_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=_pos_weight(train_ds.y).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=cfg.lr_scheduler_patience
    )

    train_loader = _loader(train_ds, cfg.batch_size, shuffle=True)

    best_auc, best_state, best_epoch, no_improve = -1.0, None, -1, 0
    history: list[dict] = []

    use_mlflow = mlflow is not None and mlflow_experiment is not None
    if use_mlflow:
        mlflow.set_experiment(mlflow_experiment)
        mlflow.start_run()
        mlflow.log_params(
            {
                "hidden_size": cfg.model.hidden_size,
                "num_layers": cfg.model.num_layers,
                "dropout": cfg.model.dropout,
                "lr": cfg.learning_rate,
                "batch_size": cfg.batch_size,
                "seq_len": train_ds.X.shape[1],
            }
        )

    try:
        for epoch in range(cfg.max_epochs):
            model.train()
            epoch_loss = 0.0
            for xb, mb, yb in train_loader:
                xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits, _ = model(xb, mb)
                loss = criterion(logits.squeeze(-1), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(train_ds.y)

            val_scores = _scores(model, val_ds, device)
            val_res = evaluate(val_ds.y, val_scores)
            scheduler.step(val_res.auc_roc)
            history.append({"epoch": epoch, "loss": epoch_loss, "val_auc": val_res.auc_roc})

            if use_mlflow:
                mlflow.log_metrics(
                    {"train_loss": epoch_loss, "val_auc_roc": val_res.auc_roc}, step=epoch
                )

            print(
                f"epoch {epoch:02d}  loss={epoch_loss:.4f}  val_auc={val_res.auc_roc:.4f}"
            )

            if val_res.auc_roc > best_auc:
                best_auc, best_epoch = val_res.auc_roc, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.early_stopping_patience:
                    print(f"early stopping at epoch {epoch} (best={best_epoch})")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        final_val = evaluate(val_ds.y, _scores(model, val_ds, device))
        if use_mlflow:
            mlflow.log_metrics({f"best_{k}": v for k, v in final_val.as_dict().items()
                                if isinstance(v, float)})
    finally:
        if use_mlflow:
            mlflow.end_run()

    return TrainOutput(model=model, val_result=final_val, best_epoch=best_epoch, history=history)


if __name__ == "__main__":
    from config import MLFLOW_EXPERIMENTS
    from data.sequence_builder import build_sequences, fit_scaler, time_based_split
    from data.synthetic import generate_transactions
    from features.feature_pipeline import build_features

    raw = generate_transactions(n_users=2000, seed=42)
    feats = build_features(raw)
    tr, va, te = time_based_split(feats)
    scaler = fit_scaler(tr)
    train_ds = build_sequences(tr, scaler)
    val_ds = build_sequences(va, scaler)

    cfg = TrainConfig(max_epochs=15)
    out = run_training(train_ds, val_ds, cfg, mlflow_experiment=MLFLOW_EXPERIMENTS["lstm"])
    print("\nBest validation metrics:")
    for k, v in out.val_result.as_dict().items():
        print(f"  {k}: {v}")
