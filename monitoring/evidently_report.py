"""Drift monitoring for RiskSequencer.

Computes Population Stability Index (PSI) on the three mandated behavioral
features and decides whether to trigger retraining (PSI > 0.20). PSI is
implemented directly so the trigger logic is testable without Evidently
installed; `evidently_report` adds the rich HTML report when the dep is
available.
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

    @property
    def should_retrain(self) -> bool:
        return len(self.breached) > 0


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """PSI between a reference and current distribution.

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


def check_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str] | None = None,
    threshold: float = PSI_THRESHOLD,
) -> DriftResult:
    """Compute PSI per monitored feature and flag those above `threshold`."""
    features = features or MONITORED_FEATURES
    psi_by_feature, breached = {}, []
    for feat in features:
        psi = population_stability_index(reference_df[feat].to_numpy(), current_df[feat].to_numpy())
        psi_by_feature[feat] = psi
        if psi > threshold:
            breached.append(feat)
    return DriftResult(psi_by_feature=psi_by_feature, breached=breached)


def evidently_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, out_html: str | None = None):
    """Build an Evidently DataDriftPreset report (requires `evidently`)."""
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df[MONITORED_FEATURES],
               current_data=current_df[MONITORED_FEATURES])
    if out_html:
        report.save_html(out_html)
    return report


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({f: rng.normal(0, 1, 5000) for f in MONITORED_FEATURES})
    # shift one feature hard to simulate drift
    cur = pd.DataFrame({f: rng.normal(0, 1, 5000) for f in MONITORED_FEATURES})
    cur[MONITORED_FEATURES[0]] = rng.normal(3, 1.5, 5000)
    res = check_drift(ref, cur)
    print("PSI:", {k: round(v, 3) for k, v in res.psi_by_feature.items()})
    print("breached:", res.breached, "-> retrain:", res.should_retrain)
