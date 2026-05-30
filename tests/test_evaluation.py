"""Tests for cost-weighted Precision@k and calibration utilities."""

from __future__ import annotations

import numpy as np

from src.evaluation.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_curve,
)
from src.evaluation.metrics import (
    CostMatrix,
    cost_weighted_precision_at_k,
    precision_at_k,
)


def _make_cost_matrix() -> CostMatrix:
    return CostMatrix(
        false_negative_cost_usd=11_500.0,
        false_positive_cost_usd=22.17,
        investigator_hourly_rate_usd=95.0,
        daily_alert_capacity_per_analyst=48,
        analyst_count=8,
    )


def test_precision_at_k_perfect_ranking():
    """A perfect ranking yields P@k = 1.0 for k <= n_positives."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    assert precision_at_k(y_true, scores, k=3) == 1.0


def test_cost_weighted_precision_at_k_returns_components():
    """The auditable record carries TP/FP/FN counts and the cost matrix."""
    y_true = np.zeros(1000, dtype=int)
    y_true[[10, 20, 30]] = 1
    scores = np.zeros(1000)
    scores[[10, 20, 30]] = 0.99  # perfect top-3 ranking
    result = cost_weighted_precision_at_k(
        y_true,
        scores,
        cost_matrix=_make_cost_matrix(),
        k=10,
    )
    assert result["true_positives"] == 3
    assert result["k"] == 10
    assert "cost_matrix" in result


def test_brier_score_perfect_predictions_is_zero():
    """Brier score is 0 when predicted == true (perfect calibration)."""
    y = np.array([0, 1, 0, 1])
    probs = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score(y, probs) == 0.0


def test_reliability_curve_sparse_bins_flagged():
    """Bins below min_samples_per_bin are flagged in sparse_bins."""
    y = np.array([1, 0, 1, 0])  # only 4 samples
    probs = np.array([0.1, 0.3, 0.7, 0.9])
    curve = reliability_curve(y, probs, n_bins=10, min_samples_per_bin=10)
    assert curve.sparse_bins.all()


def test_expected_calibration_error_perfect_is_zero():
    """ECE is 0 for perfectly calibrated probabilities."""
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    probs = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    assert expected_calibration_error(y, probs) == 0.0
