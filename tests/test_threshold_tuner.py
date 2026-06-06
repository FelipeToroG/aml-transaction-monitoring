"""Regression tests for the fixed-capacity decision-threshold tuner.

Coverage for the "threshold tuner misaligned with operational k" finding
in docs/INCIDENT_REPORT.md. The v0 tuner swept thresholds at a *variable*
k (the count of predictions above each candidate) and could settle on a
near-zero threshold that flagged most of the fold, which made val and test
metrics non-comparable. The fixed tuner sets the threshold at the
operational alert budget, so it must produce exactly k alerts.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import CostMatrix
from src.models.train import _tune_threshold


def _make_cost_matrix() -> CostMatrix:
    # k_per_day = 48 * 8 = 384, the fixed operational alert capacity the
    # tuner sizes the threshold to.
    return CostMatrix(
        false_negative_cost_usd=19_000.0,
        false_positive_cost_usd=22.17,
        investigator_hourly_rate_usd=95.0,
        daily_alert_capacity_per_analyst=48,
        analyst_count=8,
    )


def test_threshold_produces_exactly_operational_k_alerts():
    """On distinct scores the threshold admits exactly k alerts."""
    rng = np.random.default_rng(0)
    n = 2000
    # A permutation guarantees distinct scores, so there are no boundary
    # ties that could push the alert count above k.
    scores = rng.permutation(n).astype(float)
    y_true = np.zeros(n, dtype=int)
    y_true[rng.choice(n, size=50, replace=False)] = 1

    operational_k = 384
    threshold, result = _tune_threshold(
        scores=scores,
        y_true=y_true,
        cost_matrix=_make_cost_matrix(),
        operational_k=operational_k,
    )

    assert int((scores >= threshold).sum()) == operational_k
    # The threshold is precisely the k-th largest score (descending rank k).
    kth_largest = np.sort(scores)[::-1][operational_k - 1]
    assert threshold == kth_largest
    # The returned evaluation is computed at the same fixed k, so val tuning
    # and the downstream test eval share an alert budget.
    assert result["k"] == operational_k


def test_threshold_clamps_when_k_exceeds_fold_size():
    """k larger than the fold clamps to the fold size without error."""
    n = 50
    scores = np.linspace(0.0, 1.0, n)  # distinct, ascending
    y_true = np.zeros(n, dtype=int)
    y_true[:3] = 1

    # operational_k (384) exceeds the 50-row fold; every row becomes an
    # alert and the threshold drops to the minimum score.
    threshold, result = _tune_threshold(
        scores=scores,
        y_true=y_true,
        cost_matrix=_make_cost_matrix(),
        operational_k=384,
    )

    assert int((scores >= threshold).sum()) == n
    assert threshold == scores.min()
    assert result["k"] == n
