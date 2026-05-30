"""Probability calibration evaluation utilities.

A model's discrimination (AUC-PR, Precision@k) describes whether it
ranks positives above negatives. A model's calibration describes
whether the predicted probabilities match empirical positive rates.
For cost-sensitive AML scoring the two are independent properties that
both matter:

* The ensemble layer combines anomaly scores with supervised
  probabilities at the score level. If the supervised probabilities are
  miscalibrated, the score-level combination is also miscalibrated, and
  the threshold tuned during training will not correspond to the same
  alert-volume operating point at runtime.
* The Streamlit investigator dashboard surfaces the predicted
  probability alongside every alert. Investigators learn over time
  whether a "0.9" alert is genuinely a 90%-laundering case; if not,
  they discount the model's outputs and the value of the system
  collapses.

This module provides three calibration utilities:

* :func:`reliability_curve` — Binned predicted-versus-actual data for
  plotting reliability diagrams.
* :func:`brier_score` — Mean squared error between predicted
  probability and binary label. A scalar discriminator-and-calibration
  combined score.
* :func:`expected_calibration_error` — Weighted mean absolute
  bin-by-bin calibration gap. The headline calibration metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ReliabilityCurve:
    """Result of a reliability-curve binning.

    Attributes
    ----------
    bin_centers : np.ndarray
        Centre of each bin in predicted-probability space.
    predicted_mean : np.ndarray
        Mean predicted probability of the items in each bin.
    actual_rate : np.ndarray
        Empirical positive rate of the items in each bin.
    bin_counts : np.ndarray
        Number of items in each bin. Bins below a minimum count are
        flagged in :attr:`sparse_bins` and should be discounted in
        downstream plotting.
    sparse_bins : np.ndarray
        Boolean mask indicating bins with sample size below the
        ``min_samples_per_bin`` argument used at construction.
    """

    bin_centers: np.ndarray
    predicted_mean: np.ndarray
    actual_rate: np.ndarray
    bin_counts: np.ndarray
    sparse_bins: np.ndarray


def reliability_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 10,
    min_samples_per_bin: int = 30,
) -> ReliabilityCurve:
    """Compute a reliability-curve binning of predicted probabilities.

    Splits ``[0, 1]`` into ``n_bins`` equal-width bins and reports the
    mean predicted probability and empirical positive rate within each
    bin. A perfectly calibrated model lies on the diagonal:
    ``predicted_mean == actual_rate`` for every bin.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels (1 = positive). Shape ``(n,)``.
    probabilities : np.ndarray
        Predicted probabilities of the positive class. Shape ``(n,)``.
    n_bins : int
        Number of bins for the partition. Default 10 matches the
        convention in the calibration literature.
    min_samples_per_bin : int
        Minimum samples a bin must contain to be considered statistically
        reliable. Bins below this threshold are flagged in the result
        so plotting code can de-emphasise them.

    Returns
    -------
    ReliabilityCurve
        Per-bin statistics plus the sparse-bin mask.
    """
    if len(y_true) != len(probabilities):
        raise ValueError(
            f"y_true and probabilities must be the same length; got "
            f"{len(y_true)} and {len(probabilities)}."
        )

    # Equal-width binning in [0, 1]. The edges array has n_bins + 1
    # entries; we use np.digitize to assign each probability to its bin.
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(probabilities, edges, right=False) - 1, 0, n_bins - 1)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0

    predicted_mean = np.zeros(n_bins, dtype=np.float64)
    actual_rate = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        in_bin = bin_indices == b
        bin_counts[b] = int(in_bin.sum())
        if bin_counts[b] == 0:
            # Empty bin: leave the bin entries at zero. The sparse mask
            # below flags them so the consumer does not plot them.
            continue
        predicted_mean[b] = float(np.mean(probabilities[in_bin]))
        actual_rate[b] = float(np.mean(y_true[in_bin]))

    sparse_bins = bin_counts < min_samples_per_bin
    return ReliabilityCurve(
        bin_centers=bin_centers,
        predicted_mean=predicted_mean,
        actual_rate=actual_rate,
        bin_counts=bin_counts,
        sparse_bins=sparse_bins,
    )


def brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute the Brier score.

    Definition: mean squared error between predicted probability and
    the binary outcome label. Ranges from 0 (perfect) to 1 (worst).
    Lower is better. Captures both discrimination and calibration in a
    single scalar.

    The Brier score is properly scored — a model that misrepresents its
    own confidence to game the metric will score worse than one that
    reports true probabilities — which makes it the right primary
    calibration scalar.
    """
    if len(y_true) != len(probabilities):
        raise ValueError(
            f"y_true and probabilities must be the same length; got "
            f"{len(y_true)} and {len(probabilities)}."
        )
    return float(np.mean((probabilities - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Compute the Expected Calibration Error (ECE).

    Definition: the bin-weighted absolute gap between predicted
    probability and empirical positive rate. Lower is better; an ECE of
    0 means the model is perfectly calibrated. Standard binning is 10
    equal-width bins.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels.
    probabilities : np.ndarray
        Predicted probabilities.
    n_bins : int
        Number of bins for the partition.

    Returns
    -------
    float
        ECE in the ``[0, 1]`` range.
    """
    curve = reliability_curve(y_true, probabilities, n_bins=n_bins, min_samples_per_bin=1)
    total = curve.bin_counts.sum()
    if total == 0:
        return 0.0

    weights = curve.bin_counts.astype(np.float64) / float(total)
    bin_errors = np.abs(curve.predicted_mean - curve.actual_rate)
    return float(np.sum(weights * bin_errors))
