"""Tests for PSI drift detection and the retraining trigger."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def test_evidently_engine_detects_drift():
    """The Evidently AI path (PSI stattest) flags the drifted feature.

    Skipped when Evidently isn't installed (e.g. CI runs the fallback path).
    """
    pytest.importorskip("evidently")
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({f: rng.normal(0, 1, 4000) for f in MONITORED_FEATURES})
    cur = pd.DataFrame({f: rng.normal(0, 1, 4000) for f in MONITORED_FEATURES})
    drifted = MONITORED_FEATURES[0]
    cur[drifted] = rng.normal(5, 1, 4000)
    res = check_drift(ref, cur, use_evidently=True)
    assert res.engine == "evidently"
    assert drifted in res.breached
    # an undrifted feature stays well under the 0.20 PSI threshold
    assert res.psi_by_feature[MONITORED_FEATURES[1]] < 0.20


def test_fallback_engine_when_evidently_disabled():
    rng = np.random.default_rng(3)
    ref = pd.DataFrame({f: rng.normal(0, 1, 3000) for f in MONITORED_FEATURES})
    cur = pd.DataFrame({f: rng.normal(0, 1, 3000) for f in MONITORED_FEATURES})
    cur[MONITORED_FEATURES[0]] = rng.normal(5, 1, 3000)
    res = check_drift(ref, cur, use_evidently=False)
    assert res.engine == "fallback"
    assert MONITORED_FEATURES[0] in res.breached
