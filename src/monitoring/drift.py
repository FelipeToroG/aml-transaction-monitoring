"""Population Stability Index (PSI) drift detection.

PSI is the bank model-risk-management default for distribution drift.
It compares a reference distribution against a target distribution by
binning both and summing the bin-weighted log-ratio of proportions:

.. math::
    PSI = \\sum_{i=1}^{K} (p_{target,i} - p_{ref,i})
          \\cdot \\ln(p_{target,i} / p_{ref,i})

The resulting scalar is interpreted via the severity bands defined in
:mod:`src.monitoring.alerts`:

* ``PSI < 0.10`` — monitor (normal variation, no action)
* ``0.10 \\le PSI < 0.25`` — warning (model-team review recommended)
* ``PSI \\ge 0.25`` — regulator-relevant (escalate; consider rollback)

Output contract
---------------
:func:`generate_drift_snapshot` writes a JSON file consumed by the
Streamlit drift-monitor page. The shape is stable across releases and
documented inline below; changes require an explicit version bump in
the snapshot payload.
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

from src.monitoring.alerts import PSI_THRESHOLDS, Severity
from src.observability.metrics import record_drift_event

logger = logging.getLogger(__name__)

# Small epsilon added to per-bin proportions before the log-ratio so a
# bin that is empty in either distribution does not yield log(0) or
# log(inf). 1e-6 is small enough that it does not perturb a normal
# computation and large enough to dominate floating-point round-off
# in the underflow case.
_PROPORTION_EPSILON: float = 1e-6


@dataclass(frozen=True, slots=True)
class FeatureDriftResult:
    """PSI result for a single feature."""

    feature: str
    psi: float
    severity: Severity
    bin_count: int
    reference_size: int
    target_size: int


@dataclass(frozen=True, slots=True)
class PredictionDriftResult:
    """PSI result for the prediction-score distribution."""

    psi: float
    severity: Severity
    bin_count: int
    reference_size: int
    target_size: int


# ---------------------------------------------------------------------
# Core PSI computation
# ---------------------------------------------------------------------


def compute_psi(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Compute the Population Stability Index of target against reference.

    Parameters
    ----------
    reference : np.ndarray
        Reference-window values. Shape ``(n_ref,)``.
    target : np.ndarray
        Target-window values. Shape ``(n_target,)``.
    n_bins : int
        Number of quantile bins. Defaults to 10 (deciles), the
        published convention in bank-monitoring literature.

    Returns
    -------
    float
        Non-negative PSI scalar. Values around 0 indicate no shift;
        larger values indicate larger shifts.
    """
    if len(reference) == 0 or len(target) == 0:
        # Cannot compute PSI on an empty distribution. Returning 0 with
        # a logged warning lets the caller proceed; the snapshot will
        # show the metric as 0 ("monitor") which is the safe default.
        logger.warning(
            "PSI computation called on empty distribution (ref=%d target=%d); "
            "returning 0.",
            len(reference),
            len(target),
        )
        return 0.0

    # Drop NaNs before binning. Quantile binning on a series containing
    # NaN puts every NaN into one bin and skews the comparison.
    reference_clean = reference[~np.isnan(reference)]
    target_clean = target[~np.isnan(target)]
    if len(reference_clean) == 0 or len(target_clean) == 0:
        return 0.0

    # Bin edges are derived from the reference distribution so a stable
    # baseline defines the partition. Quantile edges produce balanced
    # reference bins, which keeps PSI numerically stable.
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference_clean, quantiles))
    if len(edges) < 2:
        # Degenerate reference (e.g., constant feature). PSI is
        # mathematically undefined; return 0 and let the operator
        # decide whether to drop the feature.
        return 0.0

    # ``np.digitize`` assigns each value to a bin index. The
    # ``right=False`` argument makes intervals left-closed,
    # right-open. We clip the resulting indices into the valid bin
    # range so values outside the reference's empirical support
    # fall into the boundary bins rather than producing out-of-range
    # indices.
    ref_bins = np.clip(np.digitize(reference_clean, edges, right=False) - 1, 0, len(edges) - 2)
    tgt_bins = np.clip(np.digitize(target_clean, edges, right=False) - 1, 0, len(edges) - 2)

    ref_proportions = np.bincount(ref_bins, minlength=len(edges) - 1) / len(reference_clean)
    tgt_proportions = np.bincount(tgt_bins, minlength=len(edges) - 1) / len(target_clean)

    # Add epsilon to avoid log(0) on empty bins.
    ref_proportions = np.where(ref_proportions == 0, _PROPORTION_EPSILON, ref_proportions)
    tgt_proportions = np.where(tgt_proportions == 0, _PROPORTION_EPSILON, tgt_proportions)

    psi_terms = (tgt_proportions - ref_proportions) * np.log(
        tgt_proportions / ref_proportions
    )
    return float(np.sum(psi_terms))


# ---------------------------------------------------------------------
# Feature- and prediction-level drift
# ---------------------------------------------------------------------


def compute_feature_drift(
    *,
    reference: pd.DataFrame,
    target: pd.DataFrame,
    feature_columns: list[str],
    n_bins: int = 10,
    emit_metric: bool = True,
) -> list[FeatureDriftResult]:
    """Compute per-feature PSI for the supplied feature list.

    Parameters
    ----------
    reference : pd.DataFrame
        Reference-window frame. Must contain ``feature_columns``.
    target : pd.DataFrame
        Target-window frame. Must contain ``feature_columns``.
    feature_columns : list[str]
        Names of the features to evaluate.
    n_bins : int
        Number of quantile bins. Defaults to 10.
    emit_metric : bool
        When True, increment ``aml_drift_events_total`` for each
        feature whose severity is above ``monitor``.

    Returns
    -------
    list[FeatureDriftResult]
        Per-feature results, sorted by descending PSI so the most
        drifted features appear first in the snapshot.
    """
    results: list[FeatureDriftResult] = []
    for feature in feature_columns:
        if feature not in reference.columns or feature not in target.columns:
            logger.debug("Skipping feature %r — missing from one of the frames.", feature)
            continue
        psi = compute_psi(
            reference[feature].to_numpy(),
            target[feature].to_numpy(),
            n_bins=n_bins,
        )
        severity = PSI_THRESHOLDS.classify(psi)
        results.append(
            FeatureDriftResult(
                feature=feature,
                psi=psi,
                severity=severity,
                bin_count=n_bins,
                reference_size=int(len(reference)),
                target_size=int(len(target)),
            )
        )
        if emit_metric and severity != "monitor":
            record_drift_event(feature=feature, severity=severity)

    return sorted(results, key=lambda r: r.psi, reverse=True)


def compute_prediction_drift(
    *,
    reference_scores: np.ndarray,
    target_scores: np.ndarray,
    n_bins: int = 10,
    emit_metric: bool = True,
) -> PredictionDriftResult:
    """Compute PSI on the prediction-score distribution.

    Score-distribution drift is the headline drift signal because it
    directly indicates the model's behaviour has shifted, regardless of
    whether any one input feature has drifted. A shift in the score
    distribution at constant input distribution typically indicates a
    bug or a data-pipeline change that altered preprocessing semantics.
    """
    psi = compute_psi(reference_scores, target_scores, n_bins=n_bins)
    severity = PSI_THRESHOLDS.classify(psi)

    if emit_metric and severity != "monitor":
        record_drift_event(feature="__prediction_distribution__", severity=severity)

    return PredictionDriftResult(
        psi=psi,
        severity=severity,
        bin_count=n_bins,
        reference_size=int(len(reference_scores)),
        target_size=int(len(target_scores)),
    )


# ---------------------------------------------------------------------
# Snapshot generation
# ---------------------------------------------------------------------


def generate_drift_snapshot(
    *,
    reference: pd.DataFrame,
    target: pd.DataFrame,
    feature_columns: list[str],
    reference_window_label: str,
    target_window_label: str,
    reference_scores: np.ndarray | None = None,
    target_scores: np.ndarray | None = None,
    output_path: Path | str = "mlruns/drift_snapshot.json",
) -> dict[str, Any]:
    """Write a drift snapshot consumed by the UI's drift-monitor page.

    Returns the snapshot dict in addition to writing it so notebook
    and test callers can inspect the result without re-reading from
    disk. The output path's parent directory is created if missing.
    """
    feature_results = compute_feature_drift(
        reference=reference,
        target=target,
        feature_columns=feature_columns,
    )

    prediction_result: PredictionDriftResult | None = None
    if reference_scores is not None and target_scores is not None:
        prediction_result = compute_prediction_drift(
            reference_scores=reference_scores,
            target_scores=target_scores,
        )

    snapshot: dict[str, Any] = {
        "snapshot_schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reference_window": reference_window_label,
        "target_window": target_window_label,
        "feature_drift": [asdict(r) for r in feature_results],
        "prediction_drift": asdict(prediction_result) if prediction_result else None,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2, default=str))
    logger.info("Wrote drift snapshot to %s", output_path)

    return snapshot
