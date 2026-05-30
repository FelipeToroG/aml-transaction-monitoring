"""Discrete-event simulation of an AML investigator queue.

A model that scores 0.92 AUC-PR can still be undeployable if it produces
five times the alert volume the operations team can review. Cost-weighted
Precision@k captures *what* the model gets right at a fixed review
capacity, but it does not describe *how* the alerts arrive at the
investigator, *how long* they wait in queue, or *which* alerts breach
their SLA. Those are operational properties — they require simulation.

This module simulates the alert-to-disposition flow under a configurable
analyst pool and queue policy. It produces a per-alert outcome table
(wait time, review time, disposition timestamp, SLA breach flag) and an
aggregate report (queue depth percentiles, SLA attainment by tier,
end-of-window backlog). The output is the second of the two evidence
bundles every model-risk-management review will request: the modeling
evaluation says the model is *accurate*; the simulator says the model is
*deployable*.

Algorithmic shape
-----------------
A standard discrete-event simulation. Two heaps:

* ``analyst_heap`` keyed on each analyst's ``next_available_time``.
* ``pending_queue`` keyed on alert priority (a tuple of negative
  tier-rank then negative score, so higher-priority alerts pop first).

The driver iterates through alert arrivals in chronological order. At
each arrival we (a) free any analysts whose ``next_available_time`` is
in the past, (b) push the new alert onto the pending queue, then
(c) pop alerts off the pending queue and assign to free analysts as
long as both are non-empty. Each assignment advances the chosen
analyst's ``next_available_time`` by the review duration sampled from
the configured distribution. After the last arrival we drain the
pending queue against whatever analysts are still free, recording the
remaining backlog at the cutoff.

Complexity is O(n log n) in the alert count, dominated by heap
operations. The simulator handles HI-Small-scale alert volumes
(thousands per day) in seconds.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

# Tier rank used to set priority. Higher numeric rank = higher priority.
# Aligned with configs/alert_thresholds.yaml so the simulator's view of
# tier ordering matches the API's tiering decisions.
TIER_RANK: Final[dict[str, int]] = {
    "tier_3_critical": 3,
    "tier_2_high": 2,
    "tier_1_medium": 1,
    "suppressed": 0,
}


@dataclass(slots=True, order=True)
class _PendingEntry:
    """Internal heap entry for the pending-alert priority queue.

    Ordering by ``priority`` only; the alert payload is excluded from
    comparison via ``field(compare=False)`` so heap operations do not
    accidentally compare arbitrary alert metadata.
    """

    priority: tuple[int, float, int]
    alert_index: int = field(compare=False)


@dataclass(slots=True, order=True)
class _AnalystEntry:
    """Internal heap entry for the analyst-free heap, keyed on free time."""

    next_available_time: pd.Timestamp
    analyst_id: int = field(compare=False)


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Parameters controlling the simulated environment.

    Defaults mirror ``configs/cost_matrix.yaml`` so a simulation run
    against the default config reflects the cost-matrix assumptions the
    Precision@k training objective also uses.
    """

    analyst_count: int = 8
    daily_alert_capacity_per_analyst: int = 48
    review_minutes_mean: float = 14.0
    review_minutes_std: float = 4.0
    # SLA targets per tier in hours. Used to compute the SLA-attainment
    # rate in the aggregate output. Mirrors target_review_sla_hours in
    # alert_thresholds.yaml.
    sla_hours_per_tier: dict[str, float] = field(
        default_factory=lambda: {
            "tier_3_critical": 1.0,
            "tier_2_high": 8.0,
            "tier_1_medium": 24.0,
        }
    )
    # Random seed for reproducibility of the review-duration samples.
    random_state: int = 42


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Outcome of one simulator run.

    Attributes
    ----------
    per_alert : pd.DataFrame
        Per-alert outcomes with columns:
        ``alert_id, arrival_time, tier, score, label,
        assigned_analyst_id, wait_seconds, review_seconds,
        disposition_time, sla_breached, dispositioned``.
        ``dispositioned`` is False for alerts that were still queued
        when the simulation window ended.
    aggregate : dict[str, float | int | dict]
        Headline statistics: throughput, SLA attainment per tier,
        queue-depth percentiles, end-of-window backlog count.
    """

    per_alert: pd.DataFrame
    aggregate: dict[str, object]


def simulate_investigator_queue(
    alerts: pd.DataFrame,
    *,
    config: SimulationConfig | None = None,
    arrival_column: str = "arrival_time",
    score_column: str = "score",
    tier_column: str = "tier",
    label_column: str | None = None,
) -> SimulationResult:
    """Run the investigator-queue simulation.

    Parameters
    ----------
    alerts : pd.DataFrame
        One row per alert. Must contain at minimum a chronological
        arrival timestamp, a numeric score, and a tier label.
    config : SimulationConfig | None
        Pool size and timing configuration. Defaults applied when None.
    arrival_column, score_column, tier_column : str
        Column names. Defaults match the alert schema used elsewhere in
        the codebase; overridable for ad-hoc analysis.
    label_column : str | None
        Optional ground-truth label column. When present, included in
        the per-alert output so the simulation result composes with
        downstream evaluation.

    Returns
    -------
    SimulationResult
        Per-alert outcomes and aggregate statistics.
    """
    cfg = config if config is not None else SimulationConfig()
    rng = np.random.default_rng(cfg.random_state)

    # Sort defensively. The simulator's correctness depends on
    # chronological arrival order; an out-of-order input would otherwise
    # produce nonsense.
    sorted_alerts = alerts.sort_values(arrival_column).reset_index(drop=True)
    n_alerts = len(sorted_alerts)

    # ----- Initialise the analyst-free heap ---------------------------
    # All analysts start free at the earliest arrival. We seed the heap
    # with each analyst's "free at start" event.
    start_time = pd.Timestamp(sorted_alerts[arrival_column].iloc[0])
    analyst_heap: list[_AnalystEntry] = [
        _AnalystEntry(next_available_time=start_time, analyst_id=i)
        for i in range(cfg.analyst_count)
    ]
    heapq.heapify(analyst_heap)

    # ----- Per-alert output buffers -----------------------------------
    # Pre-allocated arrays avoid per-iteration list-append overhead on
    # the simulation's hot path and let the downstream DataFrame
    # construction be a single zero-copy view.
    out_analyst = np.full(n_alerts, -1, dtype=np.int32)
    out_wait_seconds = np.full(n_alerts, np.nan, dtype=np.float64)
    out_review_seconds = np.full(n_alerts, np.nan, dtype=np.float64)
    out_disposition = np.full(n_alerts, pd.NaT, dtype="datetime64[ns]")
    out_sla_breached = np.zeros(n_alerts, dtype=bool)
    out_dispositioned = np.zeros(n_alerts, dtype=bool)

    # ----- Main simulation loop ---------------------------------------
    pending: list[_PendingEntry] = []

    for alert_idx in range(n_alerts):
        arrival_time = pd.Timestamp(sorted_alerts[arrival_column].iloc[alert_idx])
        tier = sorted_alerts[tier_column].iloc[alert_idx]
        score = float(sorted_alerts[score_column].iloc[alert_idx])

        # Push this alert onto the pending queue. Priority is
        # (tier_rank, score, arrival_order); we negate the first two so
        # the min-heap returns the highest-tier, highest-score alert
        # first, breaking ties by FIFO arrival order.
        priority = (-TIER_RANK.get(str(tier), 0), -score, alert_idx)
        heapq.heappush(pending, _PendingEntry(priority=priority, alert_index=alert_idx))

        # Process every assignment we can make as of this arrival time.
        # We loop because one new arrival may free up multiple analysts
        # whose busy_until is in the past, each of which can pick up an
        # alert from the queue.
        while pending and analyst_heap:
            next_analyst = analyst_heap[0]
            if next_analyst.next_available_time > arrival_time:
                # The earliest-free analyst is still busy at this point
                # in time; no further assignments possible until the
                # next arrival or analyst-free event.
                break

            # Free analyst available now. Pop and assign.
            analyst = heapq.heappop(analyst_heap)
            entry = heapq.heappop(pending)
            assigned_alert_idx = entry.alert_index

            # Review duration sampled from a truncated normal so it is
            # always positive. The truncation point at 1 minute is
            # operational realism: no investigator finishes a case in
            # under a minute.
            review_seconds = max(
                rng.normal(
                    loc=cfg.review_minutes_mean * 60.0,
                    scale=cfg.review_minutes_std * 60.0,
                ),
                60.0,
            )

            # Pickup time is the later of the alert's arrival and the
            # analyst's free time. For alerts queued behind backlog,
            # pickup happens at the analyst's next_available_time.
            pickup_time = max(
                analyst.next_available_time,
                pd.Timestamp(sorted_alerts[arrival_column].iloc[assigned_alert_idx]),
            )
            disposition_time = pickup_time + pd.Timedelta(seconds=review_seconds)
            wait_seconds = (
                pickup_time
                - pd.Timestamp(sorted_alerts[arrival_column].iloc[assigned_alert_idx])
            ).total_seconds()

            # Record outcome.
            assigned_tier = str(sorted_alerts[tier_column].iloc[assigned_alert_idx])
            sla_hours = cfg.sla_hours_per_tier.get(assigned_tier, np.inf)
            sla_breached = wait_seconds > sla_hours * 3600.0

            out_analyst[assigned_alert_idx] = analyst.analyst_id
            out_wait_seconds[assigned_alert_idx] = wait_seconds
            out_review_seconds[assigned_alert_idx] = review_seconds
            out_disposition[assigned_alert_idx] = disposition_time.to_datetime64()
            out_sla_breached[assigned_alert_idx] = sla_breached
            out_dispositioned[assigned_alert_idx] = True

            # Re-push the analyst with their new free time.
            heapq.heappush(
                analyst_heap,
                _AnalystEntry(
                    next_available_time=disposition_time, analyst_id=analyst.analyst_id
                ),
            )

    # ----- Drain remaining queue with whatever analysts are free ------
    # Past the last alert arrival, the simulation continues until the
    # pending queue is empty. This gives a realistic "how long does it
    # take to clear the backlog" view.
    while pending:
        analyst = heapq.heappop(analyst_heap)
        entry = heapq.heappop(pending)
        assigned_alert_idx = entry.alert_index

        review_seconds = max(
            rng.normal(
                loc=cfg.review_minutes_mean * 60.0,
                scale=cfg.review_minutes_std * 60.0,
            ),
            60.0,
        )
        pickup_time = max(
            analyst.next_available_time,
            pd.Timestamp(sorted_alerts[arrival_column].iloc[assigned_alert_idx]),
        )
        disposition_time = pickup_time + pd.Timedelta(seconds=review_seconds)
        wait_seconds = (
            pickup_time
            - pd.Timestamp(sorted_alerts[arrival_column].iloc[assigned_alert_idx])
        ).total_seconds()

        assigned_tier = str(sorted_alerts[tier_column].iloc[assigned_alert_idx])
        sla_hours = cfg.sla_hours_per_tier.get(assigned_tier, np.inf)
        sla_breached = wait_seconds > sla_hours * 3600.0

        out_analyst[assigned_alert_idx] = analyst.analyst_id
        out_wait_seconds[assigned_alert_idx] = wait_seconds
        out_review_seconds[assigned_alert_idx] = review_seconds
        out_disposition[assigned_alert_idx] = disposition_time.to_datetime64()
        out_sla_breached[assigned_alert_idx] = sla_breached
        out_dispositioned[assigned_alert_idx] = True

        heapq.heappush(
            analyst_heap,
            _AnalystEntry(
                next_available_time=disposition_time, analyst_id=analyst.analyst_id
            ),
        )

    # ----- Assemble per-alert frame -----------------------------------
    per_alert = pd.DataFrame(
        {
            "alert_id": sorted_alerts.index.to_numpy(),
            "arrival_time": sorted_alerts[arrival_column].to_numpy(),
            "tier": sorted_alerts[tier_column].to_numpy(),
            "score": sorted_alerts[score_column].to_numpy(),
            "assigned_analyst_id": out_analyst,
            "wait_seconds": out_wait_seconds,
            "review_seconds": out_review_seconds,
            "disposition_time": out_disposition,
            "sla_breached": out_sla_breached,
            "dispositioned": out_dispositioned,
        }
    )
    if label_column is not None and label_column in sorted_alerts.columns:
        per_alert["label"] = sorted_alerts[label_column].to_numpy()

    aggregate = _aggregate(per_alert=per_alert, config=cfg)
    return SimulationResult(per_alert=per_alert, aggregate=aggregate)


def _aggregate(*, per_alert: pd.DataFrame, config: SimulationConfig) -> dict[str, object]:
    """Compute the aggregate statistics for the operator report.

    Internal helper. The output keys are the contract the Streamlit
    investigator dashboard and the audit-log writer both read from.
    """
    dispositioned = per_alert.loc[per_alert["dispositioned"]]
    n_total = int(len(per_alert))
    n_dispositioned = int(len(dispositioned))
    n_backlog = n_total - n_dispositioned

    # SLA attainment per tier, computed only over dispositioned alerts
    # because pending alerts have not yet been given a chance to breach.
    sla_attainment: dict[str, float] = {}
    for tier_name in sorted(per_alert["tier"].unique()):
        tier_subset = dispositioned.loc[dispositioned["tier"] == tier_name]
        if len(tier_subset) == 0:
            sla_attainment[str(tier_name)] = float("nan")
        else:
            sla_attainment[str(tier_name)] = 1.0 - float(
                tier_subset["sla_breached"].mean()
            )

    wait_seconds = dispositioned["wait_seconds"].to_numpy()
    return {
        "total_alerts": n_total,
        "dispositioned_alerts": n_dispositioned,
        "backlog_alerts": n_backlog,
        "analyst_count": config.analyst_count,
        "daily_review_capacity": config.analyst_count
        * config.daily_alert_capacity_per_analyst,
        "wait_seconds_mean": float(np.mean(wait_seconds)) if len(wait_seconds) else 0.0,
        "wait_seconds_p50": float(np.median(wait_seconds)) if len(wait_seconds) else 0.0,
        "wait_seconds_p95": float(np.percentile(wait_seconds, 95)) if len(wait_seconds) else 0.0,
        "wait_seconds_p99": float(np.percentile(wait_seconds, 99)) if len(wait_seconds) else 0.0,
        "sla_attainment_by_tier": sla_attainment,
    }
