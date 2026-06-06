"""Cost-aware evaluation metrics for AML model selection.

The training driver imports from this module. The cost matrix that
parameterises every metric here is loaded from
``configs/cost_matrix.yaml`` at runtime; see ``docs/EVALUATION.md`` for
the documentation of the methodology and the rationale behind each
metric.

The headline metric is :func:`cost_weighted_precision_at_k`. It captures
what a deployed AML model is actually optimising for: the dollar value
delivered per unit of investigator time, evaluated at the alert rate
the team can realistically review. The function returns a positive
number expressed in USD per investigator-hour — interpretable directly
by anyone with operational AML context, and unambiguous in the way
AUC-PR is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True, slots=True)
class CostMatrix:
    """Cost configuration for the cost-weighted Precision@k objective.

    Frozen because a CostMatrix is an immutable input to every metric
    calculation; mutating it mid-evaluation would invalidate any
    cross-trial comparison. Construct one per evaluation run from the
    loaded YAML and pass it through.

    Attributes
    ----------
    false_negative_cost_usd : float
        Cost of missing a laundering transaction: the illicit dollars
        flowing through plus the discounted regulatory exposure. Pulled
        from the ``derived`` section of ``cost_matrix.yaml``.
    false_positive_cost_usd : float
        Cost of investigating a benign transaction: investigator time
        valued at the fully loaded hourly rate.
    investigator_hourly_rate_usd : float
        Used to convert from investigator-minute units to dollars.
    daily_alert_capacity_per_analyst : int
        Maximum alerts a single analyst can review per day.
    analyst_count : int
        Number of analysts on the AML monitoring queue.

    Derived
    -------
    k_per_day : int
        ``daily_alert_capacity_per_analyst * analyst_count``. The
        natural choice of ``k`` for Precision@k — the model is
        evaluated on the top ``k`` alerts the team can actually review.
    """

    false_negative_cost_usd: float
    false_positive_cost_usd: float
    investigator_hourly_rate_usd: float
    daily_alert_capacity_per_analyst: int
    analyst_count: int

    @property
    def k_per_day(self) -> int:
        return self.daily_alert_capacity_per_analyst * self.analyst_count

    @classmethod
    def from_yaml(cls, path: str) -> "CostMatrix":
        """Load a CostMatrix from the canonical YAML configuration.

        The function reads the same file the API reads at runtime
        (``configs/cost_matrix.yaml``), so the cost assumptions
        underpinning training metrics are guaranteed to match the cost
        assumptions underpinning runtime alert prioritisation.

        Before constructing the matrix it recomputes the two derived
        costs from their documented source inputs and hard-fails on any
        mismatch. The YAML stores the derived costs as explicit numbers
        for audit-report readability, which means they can silently drift
        from the assumptions they claim to summarise. In a compliance
        system an internally inconsistent cost config must not load
        quietly; surfacing the drift at load time is the whole point.

        Raises
        ------
        ValueError
            If a stored ``derived`` cost differs from the value
            recomputed from its source fields by more than one cent.
        """
        with open(path) as fh:
            cfg = yaml.safe_load(fh)

        econ = cfg["economic_assumptions"]
        reg = cfg["regulatory_cost"]
        derived = cfg["derived"]

        stored_fn = float(derived["false_negative_cost_usd"])
        stored_fp = float(derived["false_positive_cost_usd"])
        hourly_rate = float(econ["investigator_hourly_rate_usd"])

        # Mirror the arithmetic documented in the YAML's `derived` block.
        expected_fn = (
            float(econ["average_illicit_dollars_per_missed_alert"])
            + float(reg["expected_penalty_per_undetected_case"])
            * float(reg["detection_probability_per_missed_case"])
        )
        expected_fp = hourly_rate * (
            float(econ["average_review_minutes_per_alert"]) / 60.0
        )

        _assert_derived_consistent(
            "derived.false_negative_cost_usd", stored_fn, expected_fn
        )
        _assert_derived_consistent(
            "derived.false_positive_cost_usd", stored_fp, expected_fp
        )

        return cls(
            false_negative_cost_usd=stored_fn,
            false_positive_cost_usd=stored_fp,
            investigator_hourly_rate_usd=hourly_rate,
            daily_alert_capacity_per_analyst=int(
                cfg["operational"]["daily_alert_capacity_per_analyst"]
            ),
            analyst_count=int(cfg["operational"]["analyst_count"]),
        )


def _assert_derived_consistent(field: str, stored: float, expected: float) -> None:
    """Raise if a stored derived cost has drifted from its source inputs.

    Uses a one-cent absolute tolerance because the YAML rounds the stored
    values to cents (FP is held as 22.17 against a computed 22.1667), so
    exact equality would false-fail on rounding alone. Anything beyond a
    cent is a genuine inconsistency between a derived cost and the
    assumptions it is documented to summarise.
    """
    if not math.isclose(stored, expected, abs_tol=0.01):
        raise ValueError(
            f"{field} is inconsistent with its source inputs: stored "
            f"{stored:.4f}, recomputed {expected:.4f} from the "
            "economic_assumptions / regulatory_cost fields. Reconcile "
            "configs/cost_matrix.yaml so the derived value matches the "
            "documented arithmetic before loading."
        )


def precision_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    k: int,
) -> float:
    """Compute Precision at the top-``k`` scored items.

    Standard rank-cutoff precision. Used by the cost-weighted variant
    below; exposed separately because it is the most-cited bare metric
    in AML monitoring system documentation.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels (1 = laundering, 0 = legitimate).
        Shape ``(n,)``.
    scores : np.ndarray
        Predicted risk scores. Shape ``(n,)``.
    k : int
        The rank cutoff. Must satisfy ``1 <= k <= len(y_true)``.

    Returns
    -------
    float
        The fraction of the top-``k`` items that are true positives.
    """
    if len(y_true) != len(scores):
        raise ValueError(
            f"y_true and scores must be the same length; got "
            f"{len(y_true)} and {len(scores)}."
        )
    if not 1 <= k <= len(y_true):
        raise ValueError(f"k={k} out of range [1, {len(y_true)}].")

    # argpartition is O(n) average and avoids the full sort that
    # argsort would do. We pick the indices of the largest k scores,
    # then check the corresponding labels.
    top_k_indices = np.argpartition(scores, -k)[-k:]
    return float(np.mean(y_true[top_k_indices]))


def cost_weighted_precision_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    cost_matrix: CostMatrix,
    k: int | None = None,
) -> dict[str, float]:
    """Compute the cost-weighted Precision@k objective.

    This is the headline metric the training driver optimises. The
    function returns a dict with the cost-weighted objective itself
    plus the underlying counts so the result is auditable — anyone
    reviewing a training-run summary can verify the arithmetic without
    re-running the evaluation.

    Methodology
    -----------
    Given the top ``k`` predictions ranked by ``scores``:

    * ``true_positives`` = ground-truth illicit transactions in the top ``k``.
    * ``false_positives`` = benign transactions in the top ``k``.
    * ``false_negatives`` = ground-truth illicit transactions ranked
      below position ``k`` (which means they were not surfaced for
      investigator review).

    The cost-weighted objective is the negative total cost expressed
    per investigator-hour. Negating turns it into a quantity to
    *maximise*, which matches Optuna's convention without requiring
    direction overrides.

    .. math::
        \\text{objective} = - \\frac{\\text{TP} \\cdot 0 + \\text{FP}
        \\cdot c_{FP} + \\text{FN} \\cdot c_{FN}}{T_{\\text{investigator-hours}}}

    True positives contribute zero cost (the analyst is paid the same
    whether the alert is true or false; the *value* of a true positive
    is the avoided ``c_{FN}``, which is captured in the savings vs.
    not-alerting baseline).

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels. Shape ``(n,)``.
    scores : np.ndarray
        Predicted risk scores. Shape ``(n,)``.
    cost_matrix : CostMatrix
        Cost configuration. Determines ``c_{FN}``, ``c_{FP}``, and the
        default ``k``.
    k : int | None
        Optional override of the rank cutoff. Defaults to the matrix's
        ``k_per_day``, the team's actual review capacity per day.

    Returns
    -------
    dict[str, float]
        Auditable evaluation record with the objective plus its
        underlying components.
    """
    effective_k = k if k is not None else cost_matrix.k_per_day
    if len(y_true) != len(scores):
        raise ValueError(
            f"y_true and scores must be the same length; got "
            f"{len(y_true)} and {len(scores)}."
        )
    if not 1 <= effective_k <= len(y_true):
        raise ValueError(
            f"k={effective_k} out of range [1, {len(y_true)}]; the eval set "
            "may be smaller than the daily review capacity."
        )

    # Identify top-k indices. The top-k set is what the investigators
    # actually see in production at the configured capacity.
    top_k_indices = np.argpartition(scores, -effective_k)[-effective_k:]
    top_k_mask = np.zeros(len(y_true), dtype=bool)
    top_k_mask[top_k_indices] = True

    tp = int(np.sum(y_true[top_k_mask] == 1))
    fp = int(np.sum(y_true[top_k_mask] == 0))
    fn = int(np.sum((y_true == 1) & ~top_k_mask))

    total_dollar_cost = (
        fp * cost_matrix.false_positive_cost_usd
        + fn * cost_matrix.false_negative_cost_usd
    )

    # Total investigator-hours spent reviewing the top-k. Computed from
    # the FP-cost denominator to keep the units consistent.
    investigator_hours = (
        effective_k * (cost_matrix.false_positive_cost_usd / cost_matrix.investigator_hourly_rate_usd)
    )

    # Cost per investigator-hour, negated so larger is better (matches
    # Optuna's maximise direction without per-trial direction handling).
    cost_per_hour = total_dollar_cost / investigator_hours
    objective = -cost_per_hour

    # Also report the raw precision at the cutoff for diagnostic
    # purposes. The Optuna sweep selects on `objective`; the precision
    # is for human readability in the trial logs.
    precision = tp / effective_k if effective_k > 0 else 0.0

    return {
        "objective_cost_per_investigator_hour_usd": objective,
        "precision_at_k": precision,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "k": effective_k,
        "total_dollar_cost_usd": total_dollar_cost,
        "investigator_hours": investigator_hours,
        "cost_matrix": _serialise_cost_matrix(cost_matrix),
    }


def _serialise_cost_matrix(cost_matrix: CostMatrix) -> dict[str, Any]:
    """Internal: serialise a CostMatrix for inclusion in eval records."""
    return {
        "false_negative_cost_usd": cost_matrix.false_negative_cost_usd,
        "false_positive_cost_usd": cost_matrix.false_positive_cost_usd,
        "investigator_hourly_rate_usd": cost_matrix.investigator_hourly_rate_usd,
        "daily_alert_capacity_per_analyst": cost_matrix.daily_alert_capacity_per_analyst,
        "analyst_count": cost_matrix.analyst_count,
        "k_per_day": cost_matrix.k_per_day,
    }
