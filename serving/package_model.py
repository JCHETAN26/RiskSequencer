"""Package a trained RiskSequencer into a SageMaker-ready ``model.tar.gz``.

Two responsibilities:

1. ``save_artifacts`` — persist the three things the inference handler loads:
   ``model.pt`` (state_dict + config), ``scaler.pkl``, ``threshold.json``.
2. ``build_model_tar`` — assemble the standard SageMaker layout and tar it::

       model.tar.gz
       ├── model.pt
       ├── scaler.pkl
       ├── threshold.json
       └── code/
           ├── inference.py        # entry point (model_fn/input_fn/...)
           ├── requirements.txt
           ├── config.py
           └── models/
               ├── __init__.py
               └── lstm_model.py

   SageMaker unpacks the archive and puts ``code/`` on ``sys.path``, so
   ``inference.py``'s ``from config import ...`` / ``from models.lstm_model
   import ...`` resolve. Bundling the code makes the artifact self-contained
   and reproducible.

The ``main`` CLI trains end-to-end (synthetic by default, or a real
transactions parquet) and emits a deployable archive — no AWS needed to build
it; the same file is what you'd upload to ``s3://risksequencer-models/``.
"""

from __future__ import annotations

import json
import pickle
import shutil
import tarfile
from pathlib import Path

import torch
from sklearn.preprocessing import RobustScaler

from config import ARTIFACTS_DIR, PROJECT_ROOT, ModelConfig, ensure_dirs
from models.lstm_model import RiskSequencer

# Inference-code files copied into the archive's code/ directory.
_CODE_FILES = [
    ("serving/inference.py", "inference.py"),
    ("config.py", "config.py"),
    ("models/__init__.py", "models/__init__.py"),
    ("models/lstm_model.py", "models/lstm_model.py"),
]
_INFERENCE_REQUIREMENTS = "torch>=2.0\nscikit-learn>=1.3\nnumpy>=1.24\n"


def save_artifacts(
    model: RiskSequencer,
    scaler: RobustScaler,
    threshold: float,
    model_config: ModelConfig,
    out_dir: str | Path,
) -> Path:
    """Write model.pt / scaler.pkl / threshold.json into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {
                "input_size": model_config.input_size,
                "hidden_size": model_config.hidden_size,
                "num_layers": model_config.num_layers,
                "dropout": model_config.dropout,
            },
        },
        out / "model.pt",
    )
    with open(out / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(out / "threshold.json", "w") as f:
        json.dump({"threshold": float(threshold)}, f)
    return out


def build_model_tar(
    artifact_dir: str | Path,
    out_tar: str | Path,
    include_code: bool = True,
) -> Path:
    """Assemble the SageMaker layout from ``artifact_dir`` and tar it.

    Expects ``artifact_dir`` to already contain model.pt / scaler.pkl /
    threshold.json (e.g. from ``save_artifacts``).
    """
    artifact_dir = Path(artifact_dir)
    for required in ("model.pt", "scaler.pkl", "threshold.json"):
        if not (artifact_dir / required).exists():
            raise FileNotFoundError(f"missing artifact: {artifact_dir / required}")

    staging = artifact_dir / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    for name in ("model.pt", "scaler.pkl", "threshold.json"):
        shutil.copy2(artifact_dir / name, staging / name)

    if include_code:
        code = staging / "code"
        code.mkdir()
        for src_rel, dst_rel in _CODE_FILES:
            src = PROJECT_ROOT / src_rel
            dst = code / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        (code / "requirements.txt").write_text(_INFERENCE_REQUIREMENTS)

    out_tar = Path(out_tar)
    with tarfile.open(out_tar, "w:gz") as tar:
        for item in sorted(staging.iterdir()):
            tar.add(item, arcname=item.name)

    shutil.rmtree(staging)
    return out_tar


def package_from_training(
    transactions=None,
    out_tar: str | Path | None = None,
    max_epochs: int = 15,
) -> Path:
    """Train end-to-end, tune the threshold, and emit ``model.tar.gz``."""
    from config import TrainConfig
    from data.sequence_builder import build_sequences, fit_scaler, time_based_split
    from features.feature_pipeline import build_features
    from training.evaluate import tune_threshold
    from training.train import _scores, run_training

    if transactions is None:
        from data.synthetic import generate_transactions

        transactions = generate_transactions(n_users=2000, seed=42)

    feats = build_features(transactions)
    tr, va, te = time_based_split(feats)
    scaler = fit_scaler(tr)
    train_ds = build_sequences(tr, scaler)
    val_ds = build_sequences(va, scaler)

    cfg = TrainConfig(max_epochs=max_epochs)
    result = run_training(train_ds, val_ds, cfg)

    # Operating threshold tuned on validation (FPR <= 5%, max precision).
    val_scores = _scores(result.model, val_ds, device="cpu")
    threshold = tune_threshold(val_ds.y, val_scores)

    ensure_dirs()
    save_artifacts(result.model, scaler, threshold, cfg.model, ARTIFACTS_DIR)
    out_tar = Path(out_tar) if out_tar else (ARTIFACTS_DIR / "model.tar.gz")
    build_model_tar(ARTIFACTS_DIR, out_tar)
    print(f"val AUC: {result.val_result.auc_roc:.4f}  threshold: {threshold:.4f}")
    print(f"wrote {out_tar} ({out_tar.stat().st_size / 1e6:.1f} MB)")
    return out_tar


def main() -> None:
    import argparse

    import pandas as pd

    parser = argparse.ArgumentParser(description="Package a trained model for SageMaker.")
    parser.add_argument(
        "--transactions", type=Path, default=None,
        help="canonical transactions parquet (default: synthetic)",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-epochs", type=int, default=15)
    args = parser.parse_args()

    txns = pd.read_parquet(args.transactions) if args.transactions else None
    package_from_training(txns, out_tar=args.out, max_epochs=args.max_epochs)


if __name__ == "__main__":
    main()
