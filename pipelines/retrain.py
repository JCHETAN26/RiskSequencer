"""Local, AWS-free orchestration of the RiskSequencer retraining loop.

This is the testable core that the Airflow DAG wraps. It chains the pieces
that already exist — drift check, training, evaluation, the promotion gate,
champion/challenger comparison, and packaging — into one function you can run
end-to-end on a laptop with synthetic data, no S3/SageMaker/Airflow required.

The two promotion rules from the spec are combined in `decide_promotion`:

* **MLflow AUC gate (absolute):** a model below `MLFLOW_AUC_GATE` is blocked
  with no manual override.
* **Champion/challenger (relative):** even above the gate, a challenger is only
  promoted if it beats the current champion (by `min_improvement`). The live
  48-hour A/B is a deployment concern; this is the offline decision core.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import MLFLOW_AUC_GATE, TrainConfig


@dataclass
class PromotionDecision:
    promote: bool
    reason: str
    challenger_auc: float
    champion_auc: float | None
    gate: float
    min_improvement: float


def decide_promotion(
    challenger_auc: float,
    champion_auc: float | None = None,
    gate: float = MLFLOW_AUC_GATE,
    min_improvement: float = 0.0,
) -> PromotionDecision:
    """Apply the absolute gate, then the champion/challenger comparison.

    `champion_auc=None` means there is no incumbent (first model), so clearing
    the gate is sufficient.
    """
    base = dict(
        challenger_auc=float(challenger_auc),
        champion_auc=None if champion_auc is None else float(champion_auc),
        gate=float(gate),
        min_improvement=float(min_improvement),
    )

    if challenger_auc < gate:
        return PromotionDecision(
            promote=False,
            reason=f"blocked: AUC {challenger_auc:.4f} below gate {gate:.2f}",
            **base,
        )
    if champion_auc is not None and challenger_auc <= champion_auc + min_improvement:
        return PromotionDecision(
            promote=False,
            reason=(
                f"not promoted: challenger {challenger_auc:.4f} does not beat "
                f"champion {champion_auc:.4f} (+{min_improvement:.4f})"
            ),
            **base,
        )
    incumbent = "no champion" if champion_auc is None else f"beats champion {champion_auc:.4f}"
    return PromotionDecision(
        promote=True,
        reason=f"promote: AUC {challenger_auc:.4f} clears gate and {incumbent}",
        **base,
    )


@dataclass
class RetrainResult:
    test_auc: float
    decision: PromotionDecision
    artifact_path: Path | None
    val_auc: float


def run_local_retrain(
    transactions: pd.DataFrame | None = None,
    champion_auc: float | None = None,
    cfg: TrainConfig | None = None,
    min_improvement: float = 0.0,
    package: bool = True,
    mlflow_experiment: str | None = None,
) -> RetrainResult:
    """Train -> evaluate on held-out test -> decide promotion -> (maybe) package.

    Mirrors the DAG's retrain->evaluate->gate->deploy path, runnable locally.
    Defaults to synthetic data so it executes with zero setup.
    """
    from data.sequence_builder import build_sequences, fit_scaler, time_based_split
    from features.feature_pipeline import build_features
    from training.evaluate import evaluate
    from training.train import _scores, run_training

    if transactions is None:
        from data.synthetic import generate_transactions

        transactions = generate_transactions(n_users=2000, seed=42)

    cfg = cfg or TrainConfig()
    feats = build_features(transactions)
    tr, va, te = time_based_split(feats)
    scaler = fit_scaler(tr)
    train_ds = build_sequences(tr, scaler)
    val_ds = build_sequences(va, scaler)
    test_ds = build_sequences(te, scaler)

    out = run_training(train_ds, val_ds, cfg, mlflow_experiment=mlflow_experiment)
    test_auc = evaluate(test_ds.y, _scores(out.model, test_ds, device="cpu")).auc_roc

    decision = decide_promotion(
        test_auc, champion_auc, gate=MLFLOW_AUC_GATE, min_improvement=min_improvement
    )

    artifact_path: Path | None = None
    if package and decision.promote:
        from config import ARTIFACTS_DIR, ensure_dirs
        from serving.package_model import build_model_tar, save_artifacts
        from training.evaluate import tune_threshold

        ensure_dirs()
        threshold = tune_threshold(val_ds.y, _scores(out.model, val_ds, device="cpu"))
        save_artifacts(out.model, scaler, threshold, cfg.model, ARTIFACTS_DIR)
        artifact_path = build_model_tar(ARTIFACTS_DIR, ARTIFACTS_DIR / "model.tar.gz")

    return RetrainResult(
        test_auc=test_auc,
        decision=decision,
        artifact_path=artifact_path,
        val_auc=out.val_result.auc_roc,
    )


def run_drift_triggered_retrain(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    transactions: pd.DataFrame | None = None,
    champion_auc: float | None = None,
    **kwargs,
) -> RetrainResult | None:
    """Check drift first; only retrain if a monitored feature breaches PSI.

    Returns the RetrainResult when retraining ran, or None when no drift was
    detected (the weekly schedule would still call `run_local_retrain`).
    """
    from monitoring.evidently_report import check_drift
    from monitoring.slack_alerts import alert_drift

    drift = check_drift(reference_df, current_df)
    if not drift.should_retrain:
        return None
    for feat in drift.breached:
        alert_drift(feat, drift.psi_by_feature[feat])
    return run_local_retrain(transactions, champion_auc=champion_auc, **kwargs)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run a local retrain + promotion decision.")
    parser.add_argument("--champion-auc", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--no-package", action="store_true")
    args = parser.parse_args()

    result = run_local_retrain(
        champion_auc=args.champion_auc,
        cfg=TrainConfig(max_epochs=args.max_epochs),
        min_improvement=args.min_improvement,
        package=not args.no_package,
    )
    print(f"\nval AUC : {result.val_auc:.4f}")
    print(f"test AUC: {result.test_auc:.4f}")
    print(f"decision: {result.decision.reason}")
    if result.artifact_path:
        print(f"artifact: {result.artifact_path}")


if __name__ == "__main__":
    main()
