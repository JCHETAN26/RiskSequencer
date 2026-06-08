"""Tests for PSI drift detection and the retraining trigger."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import MONITORED_FEATURES, PSI_THRESHOLD
from monitoring.evidently_report import check_drift, population_stability_index


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    assert population_stability_index(x, x) < 1e-6


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(4, 1, 5000)
    assert population_stability_index(ref, cur) > PSI_THRESHOLD


def test_check_drift_flags_only_breached_feature():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({f: rng.normal(0, 1, 4000) for f in MONITORED_FEATURES})
    cur = pd.DataFrame({f: rng.normal(0, 1, 4000) for f in MONITORED_FEATURES})
    drifted = MONITORED_FEATURES[0]
    cur[drifted] = rng.normal(5, 1, 4000)
    res = check_drift(ref, cur)
    assert res.should_retrain
    assert drifted in res.breached
    assert set(res.psi_by_feature) == set(MONITORED_FEATURES)
