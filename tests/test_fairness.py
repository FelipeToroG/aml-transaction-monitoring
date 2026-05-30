"""Tests for fairness audit computations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitoring.fairness import compute_parity_gaps, compute_segment_metrics


def _make_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_per_segment = 500
    rows = []
    for segment in ("A", "B"):
        # Segment A: 5% positive rate, 10% alert rate
        # Segment B: 5% positive rate, 20% alert rate (bias against B)
        for _ in range(n_per_segment):
            label = int(rng.random() < 0.05)
            base_alert_rate = 0.10 if segment == "A" else 0.20
            prediction = int(rng.random() < (0.8 if label == 1 else base_alert_rate))
            rows.append({"segment": segment, "label": label, "prediction": prediction})
    return pd.DataFrame(rows)


def test_segment_metrics_returns_per_segment_rates():
    """Each qualifying segment yields a SegmentMetrics record."""
    frame = _make_frame()
    metrics = compute_segment_metrics(
        frame=frame,
        segment_column="segment",
        label_column="label",
        prediction_column="prediction",
    )
    assert {m.segment for m in metrics} == {"A", "B"}
    for m in metrics:
        assert 0.0 <= m.alert_rate <= 1.0
        assert 0.0 <= m.true_positive_rate <= 1.0
        assert 0.0 <= m.false_positive_rate <= 1.0


def test_parity_gaps_identifies_extremes():
    """Parity gaps surface the contributing segments by name."""
    frame = _make_frame()
    metrics = compute_segment_metrics(
        frame=frame,
        segment_column="segment",
        label_column="label",
        prediction_column="prediction",
    )
    gaps = compute_parity_gaps(metrics)
    # Alert rate is intentionally biased against segment B.
    alert_gap = next(g for g in gaps if g.metric == "demographic_parity_gap")
    assert alert_gap.max_segment == "B"
    assert alert_gap.gap > 0.0


def test_sparse_segments_are_excluded():
    """Segments below min_segment_size are filtered out."""
    frame = pd.DataFrame(
        {
            "segment": ["A"] * 200 + ["tiny"] * 10,
            "label": [0] * 210,
            "prediction": [0] * 210,
        }
    )
    metrics = compute_segment_metrics(
        frame=frame,
        segment_column="segment",
        label_column="label",
        prediction_column="prediction",
        min_segment_size=50,
    )
    segments = {m.segment for m in metrics}
    assert "tiny" not in segments
    assert "A" in segments
