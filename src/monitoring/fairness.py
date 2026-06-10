"""Segment-level fairness audit.

Bank model-risk-management practice (US SR 11-7) requires that
classification models be evaluated for disparate impact across
customer or transaction segments. The three canonical fairness
metrics for an alert-classification model are:

* **Demographic parity** - alert rate should not differ
  disproportionately across segments unless empirically justified
  by underlying risk differences.
* **Equal opportunity (TPR parity)** - true-positive rate (the share
  of actual laundering caught) should not differ across segments.
  Under-detection in any segment is a compliance failure.
* **FPR parity** - false-positive rate should not differ across
  segments. Disparate false-positive rates concentrate investigator
  workload on specific customer cohorts in a way that has produced
  enforcement actions.

The function :func:`generate_fairness_snapshot` produces a JSON file
consumed by the Streamlit fairness-audit page. The snapshot contains
per-segment metrics plus the global parity gap (max segment minus
min segment) for each metric, with severity classification from
:mod:`src.monitoring.alerts`.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.monitoring.alerts import (
    DEMOGRAPHIC_PARITY_THRESHOLDS,
    FPR_PARITY_THRESHOLDS,
    TPR_PARITY_THRESHOLDS,
    Severity,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SegmentMetrics:
    """Per-segment metric breakdown."""

    segment: str
    n: int
    n_positives: int
    n_predicted_positive: int
    alert_rate: float
    true_positive_rate: float
    false_positive_rate: float


@dataclass(frozen=True, slots=True)
class ParityGap:
    """Maximum-minus-minimum gap for one metric across segments."""

    metric: str
    gap: float
    severity: Severity
    min_segment: str
    max_segment: str


# ---------------------------------------------------------------------
# Core per-segment computation
# ---------------------------------------------------------------------


def compute_segment_metrics(
    *,
    frame: pd.DataFrame,
    segment_column: str,
    label_column: str,
    prediction_column: str,
    min_segment_size: int = 50,
) -> list[SegmentMetrics]:
    """Compute alert_rate, TPR, FPR per segment.

    Parameters
    ----------
    frame : pd.DataFrame
        Frame containing the segment, label, and prediction columns.
    segment_column : str
        Column whose distinct values define the segments (e.g.,
        ``payment_currency``, ``payment_format``).
    label_column : str
        Binary ground-truth label column.
    prediction_column : str
        Binary prediction column (the model's alert decision after
        threshold application).
    min_segment_size : int
        Segments with fewer rows than this are excluded from the
        result. Sparse segments produce unstable rate estimates and
        would dominate the parity gap with noise.

    Returns
    -------
    list[SegmentMetrics]
        One result per qualifying segment, sorted by segment label.
    """
    results: list[SegmentMetrics] = []
    for segment_value, group in frame.groupby(segment_column):
        if len(group) < min_segment_size:
            logger.debug(
                "Skipping segment %r - %d rows below min_segment_size=%d",
                segment_value,
                len(group),
                min_segment_size,
            )
            continue

        n = int(len(group))
        labels = group[label_column].to_numpy()
        predictions = group[prediction_column].to_numpy()

        n_positives = int(labels.sum())
        n_negatives = n - n_positives
        n_predicted_positive = int(predictions.sum())

        alert_rate = n_predicted_positive / n if n > 0 else 0.0

        # Avoid divide-by-zero on segments with no positives or no
        # negatives. The fallback values (NaN-like 0) are filtered
        # downstream by the parity-gap computation.
        true_positive_rate = (
            float(np.sum((labels == 1) & (predictions == 1)) / n_positives)
            if n_positives > 0
            else 0.0
        )
        false_positive_rate = (
            float(np.sum((labels == 0) & (predictions == 1)) / n_negatives)
            if n_negatives > 0
            else 0.0
        )

        results.append(
            SegmentMetrics(
                segment=str(segment_value),
                n=n,
                n_positives=n_positives,
                n_predicted_positive=n_predicted_positive,
                alert_rate=alert_rate,
                true_positive_rate=true_positive_rate,
                false_positive_rate=false_positive_rate,
            )
        )

    return sorted(results, key=lambda r: r.segment)


def compute_parity_gaps(segment_metrics: list[SegmentMetrics]) -> list[ParityGap]:
    """Compute max-minus-min parity gap for each fairness metric.

    Returns one :class:`ParityGap` per metric (alert rate, TPR, FPR).
    Each carries the gap magnitude, the severity classification from
    the configured thresholds, and the segment labels that produced
    the extremes - the gap is the maximally-actionable summary.
    """
    if not segment_metrics:
        return []

    gaps: list[ParityGap] = []
    for metric_attr, thresholds in (
        ("alert_rate", DEMOGRAPHIC_PARITY_THRESHOLDS),
        ("true_positive_rate", TPR_PARITY_THRESHOLDS),
        ("false_positive_rate", FPR_PARITY_THRESHOLDS),
    ):
        values = [(s.segment, getattr(s, metric_attr)) for s in segment_metrics]
        max_segment, max_value = max(values, key=lambda kv: kv[1])
        min_segment, min_value = min(values, key=lambda kv: kv[1])
        gap = max_value - min_value
        gaps.append(
            ParityGap(
                metric=thresholds.metric_name,
                gap=gap,
                severity=thresholds.classify(gap),
                min_segment=min_segment,
                max_segment=max_segment,
            )
        )
    return gaps


# ---------------------------------------------------------------------
# Snapshot generation
# ---------------------------------------------------------------------


def generate_fairness_snapshot(
    *,
    frame: pd.DataFrame,
    segment_column: str,
    label_column: str,
    prediction_column: str,
    output_path: Path | str = "mlruns/fairness_snapshot.json",
    min_segment_size: int = 50,
) -> dict[str, Any]:
    """Write the fairness snapshot consumed by the UI's fairness page.

    Returns the snapshot dict so notebook and test callers can inspect
    the result without re-reading from disk.
    """
    segment_metrics = compute_segment_metrics(
        frame=frame,
        segment_column=segment_column,
        label_column=label_column,
        prediction_column=prediction_column,
        min_segment_size=min_segment_size,
    )
    parity_gaps = compute_parity_gaps(segment_metrics)

    snapshot: dict[str, Any] = {
        "snapshot_schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "segment_column": segment_column,
        "label_column": label_column,
        "prediction_column": prediction_column,
        "min_segment_size": min_segment_size,
        "segments": [asdict(s) for s in segment_metrics],
        "parity_gaps": [asdict(g) for g in parity_gaps],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2, default=str))
    logger.info("Wrote fairness snapshot to %s", output_path)

    return snapshot
