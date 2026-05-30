"""Tests for PSI drift computation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitoring.alerts import PSI_THRESHOLDS
from src.monitoring.drift import (
    compute_feature_drift,
    compute_prediction_drift,
    compute_psi,
)


def test_psi_is_zero_for_identical_distributions():
    """Identical reference and target distributions produce PSI ~ 0."""
    rng = np.random.default_rng(42)
    reference = rng.normal(size=10_000)
    target = rng.normal(size=10_000)
    psi = compute_psi(reference, target)
    # Sampling variance gives non-zero PSI even on identical distributions;
    # the bound below covers that variance without admitting real drift.
    assert psi < PSI_THRESHOLDS.monitor_max


def test_psi_grows_under_clear_shift():
    """A meaningful mean shift produces PSI above the warning threshold."""
    rng = np.random.default_rng(42)
    reference = rng.normal(loc=0.0, size=10_000)
    target = rng.normal(loc=1.5, size=10_000)
    psi = compute_psi(reference, target)
    assert psi >= PSI_THRESHOLDS.monitor_max


def test_psi_handles_empty_distribution():
    """An empty reference or target returns 0 without raising."""
    assert compute_psi(np.array([]), np.array([1.0, 2.0])) == 0.0
    assert compute_psi(np.array([1.0]), np.array([])) == 0.0


def test_feature_drift_returns_sorted_results():
    """The result list is sorted by descending PSI."""
    rng = np.random.default_rng(42)
    ref = pd.DataFrame(
        {
            "a": rng.normal(size=1000),
            "b": rng.normal(size=1000),
        }
    )
    tgt = pd.DataFrame(
        {
            "a": rng.normal(loc=2.0, size=1000),  # large shift
            "b": rng.normal(size=1000),  # no shift
        }
    )
    results = compute_feature_drift(reference=ref, target=tgt, feature_columns=["a", "b"])
    assert results[0].feature == "a"
    assert results[0].psi >= results[1].psi


def test_prediction_drift_returns_severity():
    """Prediction drift result includes the severity classification."""
    rng = np.random.default_rng(42)
    ref_scores = rng.uniform(size=5000)
    tgt_scores = rng.uniform(size=5000)
    result = compute_prediction_drift(reference_scores=ref_scores, target_scores=tgt_scores)
    assert result.severity in ("monitor", "warning", "regulator-relevant")
