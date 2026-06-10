"""Drift monitoring for RiskSequencer.

Primary engine is **Evidently AI**: we run its DataDriftPreset with the **PSI**
stattest and a 0.20 threshold over the three monitored behavioral features, and
trigger retraining when any feature's PSI exceeds the threshold.

A hand-rolled PSI implementation is kept as a dependency-free FALLBACK so the
trigger logic stays testable (and CI runnable) in environments where Evidently
isn't installed. `check_drift` uses Evidently when available and transparently
falls back otherwise; both return the same `DriftResult`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import MONITORED_FEATURES, PSI_THRESHOLD


@dataclass
class DriftResult:
    psi_by_feature: dict[str, float]
    breached: list[str]
    engine: str = "evidently"   # "evidently" or "fallback"

    @property
    def should_retrain(self) -> bool:
        return len(self.breached) > 0


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """PSI between a reference and current distribution (dependency-free).

    Bins are fixed on the reference quantiles. A small epsilon avoids
    division-by-zero / log(0) when a bin is empty in one distribution.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    eps = 1e-6

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:  # degenerate (constant) reference
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_pct = np.histogram(current, bins=edges)[0] / len(current)
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def evidently_psi(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
    threshold: float = PSI_THRESHOLD,
) -> dict[str, float]:
    """Per-feature PSI via Evidently AI's DataDriftPreset (PSI stattest).

    Raises ImportError if Evidently isn't installed (callers fall back).
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    dd = DataDefinition(numerical_columns=list(features))
    report = Report([DataDriftPreset(method="psi", threshold=threshold)])
    snapshot = report.run(
        Dataset.from_pandas(reference_df[features], data_definition=dd),
        Dataset.from_pandas(current_df[features], data_definition=dd),
    )
    psi_by_feature: dict[str, float] = {}
    for metric in snapshot.dict()["metrics"]:
        cfg = metric.get("config", {})
        if cfg.get("type", "").endswith("ValueDrift") and cfg.get("column") in features:
            psi_by_feature[cfg["column"]] = float(metric["value"])
    return psi_by_feature


def check_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str] | None = None,
    threshold: float = PSI_THRESHOLD,
    use_evidently: bool = True,
) -> DriftResult:
    """Compute PSI per monitored feature and flag those above `threshold`.

    Uses Evidently AI when available; falls back to the built-in PSI otherwise.
    """
    features = features or MONITORED_FEATURES
    engine = "evidently"
    psi_by_feature: dict[str, float] = {}

    if use_evidently:
        try:
            psi_by_feature = evidently_psi(reference_df, current_df, features, threshold)
        except ImportError:
            engine = "fallback"

    if not psi_by_feature:  # fallback path (no Evidently, or it returned nothing)
        engine = "fallback"
        psi_by_feature = {
            feat: population_stability_index(
                reference_df[feat].to_numpy(), current_df[feat].to_numpy()
            )
            for feat in features
        }

    breached = [f for f, psi in psi_by_feature.items() if psi > threshold]
    return DriftResult(psi_by_feature=psi_by_feature, breached=breached, engine=engine)


def evidently_html_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    out_html: str,
    features: list[str] | None = None,
    threshold: float = PSI_THRESHOLD,
):
    """Save a full Evidently HTML drift report (for the monitoring dashboard)."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    features = features or MONITORED_FEATURES
    dd = DataDefinition(numerical_columns=list(features))
    report = Report([DataDriftPreset(method="psi", threshold=threshold)])
    snapshot = report.run(
        Dataset.from_pandas(reference_df[features], data_definition=dd),
        Dataset.from_pandas(current_df[features], data_definition=dd),
    )
    snapshot.save_html(out_html)
    return out_html


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({f: rng.normal(0, 1, 5000) for f in MONITORED_FEATURES})
    cur = pd.DataFrame({f: rng.normal(0, 1, 5000) for f in MONITORED_FEATURES})
    cur[MONITORED_FEATURES[0]] = rng.normal(3, 1.5, 5000)  # inject drift
    res = check_drift(ref, cur)
    print(f"engine: {res.engine}")
    print("PSI:", {k: round(v, 3) for k, v in res.psi_by_feature.items()})
    print("breached:", res.breached, "-> retrain:", res.should_retrain)
