"""Prometheus metrics for the AML monitoring service.

Metric definitions are centralised here so the operator dashboard
(Grafana, Datadog, or any Prometheus-compatible scraper) has a stable
contract. Adding a new metric requires a one-line addition here plus
the matching ``record_*`` call from the producing code path.

Naming convention
-----------------
Every metric is prefixed ``aml_`` to namespace the service in shared
Prometheus environments. Counters are suffixed ``_total`` per
Prometheus convention. Histograms surfaced in seconds use the
``_seconds`` suffix; counts of items use the ``_count`` or ``_total``
suffix depending on monotonicity.

Cost estimation
---------------
``record_llm_call`` computes an estimated USD cost from token counts
using the published model-tier rates in :data:`LLM_PRICE_TABLE`.
The estimate is approximate (Anthropic occasionally adjusts rates),
but it is exact enough for budget alerting and cross-model cost
comparisons in production dashboards.
"""

from __future__ import annotations

import logging
from typing import Final

from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------

ALERTS_CREATED_TOTAL: Final[Counter] = Counter(
    "aml_alerts_created_total",
    "Total alerts created since process start, partitioned by tier.",
    labelnames=["tier"],
)

SCORE_DISTRIBUTION: Final[Histogram] = Histogram(
    "aml_transaction_score",
    "Distribution of combined risk scores across all scored transactions.",
    labelnames=["tier"],
    # Buckets concentrate where decisions flip - between the suppressed
    # threshold (0.30) and the tier-3 threshold (0.85). Outside that band
    # the score distribution is dense enough that a single bucket suffices.
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0),
)

LLM_LATENCY_SECONDS: Final[Histogram] = Histogram(
    "aml_llm_latency_seconds",
    "End-to-end LLM call latency including retries.",
    labelnames=["model", "outcome"],
    # Anthropic Sonnet typically returns in 2-8 s for the project's
    # prompt size; Haiku in 0.5-2 s. Buckets cover both with headroom
    # for tail latencies that would page the on-call operator.
    buckets=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0),
)

LLM_TOKENS_TOTAL: Final[Counter] = Counter(
    "aml_llm_tokens_total",
    "Total LLM tokens consumed since process start.",
    labelnames=["model", "direction"],
)

LLM_COST_USD_TOTAL: Final[Counter] = Counter(
    "aml_llm_cost_usd_total",
    "Estimated LLM cost in USD, computed from token usage and published rates.",
    labelnames=["model"],
)

LLM_REFUSALS_TOTAL: Final[Counter] = Counter(
    "aml_llm_refusals_total",
    "Total narrator refusals, partitioned by refusal code.",
    labelnames=["code"],
)

WEBHOOK_DELIVERIES_TOTAL: Final[Counter] = Counter(
    "aml_webhook_deliveries_total",
    "Total webhook delivery attempts, partitioned by outcome.",
    labelnames=["outcome"],
)

FEEDBACK_TOTAL: Final[Counter] = Counter(
    "aml_feedback_total",
    "Total investigator feedback entries, partitioned by disposition and tier.",
    labelnames=["disposition", "tier"],
)

DRIFT_EVENTS_TOTAL: Final[Counter] = Counter(
    "aml_drift_events_total",
    "Total drift-detection events, partitioned by feature and severity.",
    labelnames=["feature", "severity"],
)


# ---------------------------------------------------------------------
# LLM cost estimation
# ---------------------------------------------------------------------

# Per-million-token USD rates by model. These match the published
# Anthropic pricing at time of writing; operators override via
# Prometheus relabel rules if the published rates change without a
# code release.
LLM_PRICE_TABLE: Final[dict[str, dict[str, float]]] = {
    "claude-sonnet-4-5": {"input_per_million": 3.00, "output_per_million": 15.00},
    "claude-haiku-4-5": {"input_per_million": 1.00, "output_per_million": 5.00},
}


def _estimate_llm_cost_usd(
    *, model: str, input_tokens: int, output_tokens: int
) -> float:
    """Return the estimated USD cost of one LLM call.

    Internal helper. Returns 0 for unknown models (logged at WARNING)
    rather than raising - operability over strict-typing matters for
    the metrics layer because a misconfigured model name should not
    take down the scoring path.
    """
    rates = LLM_PRICE_TABLE.get(model)
    if rates is None:
        logger.debug(
            "No price table entry for model %r; cost metric will report 0.",
            model,
        )
        return 0.0
    input_cost = (input_tokens / 1_000_000.0) * rates["input_per_million"]
    output_cost = (output_tokens / 1_000_000.0) * rates["output_per_million"]
    return input_cost + output_cost


# ---------------------------------------------------------------------
# Public producer helpers
# ---------------------------------------------------------------------
# Producer functions wrap the underlying ``.labels(...).inc()`` /
# ``.observe(...)`` calls so the call site reads as a single intent
# statement. Wrapping also gives us one place to add Prometheus-side
# failure handling if (for example) the registry ever becomes
# unavailable in a future runtime configuration.


def record_score(*, score: float, tier: str) -> None:
    """Observe a single transaction's risk score in the histogram."""
    try:
        SCORE_DISTRIBUTION.labels(tier=tier).observe(score)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Metrics: record_score failed silently: %s", exc)


def record_alert_created(*, tier: str) -> None:
    """Increment the alert-volume counter for the given tier."""
    try:
        ALERTS_CREATED_TOTAL.labels(tier=tier).inc()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Metrics: record_alert_created failed silently: %s", exc)


def record_llm_call(
    *,
    model: str,
    latency_ms: float,
    input_tokens: int | None,
    output_tokens: int | None,
    success: bool,
    refusal_code: str | None = None,
) -> None:
    """Record one LLM call's outcome across latency, tokens, cost, and refusals.

    Single helper instead of four because every narrator invocation
    produces all four observations in lockstep; a single helper
    eliminates the risk of recording some but not all on a refactor.
    """
    outcome = "success" if success else "refusal"
    try:
        LLM_LATENCY_SECONDS.labels(model=model, outcome=outcome).observe(
            latency_ms / 1000.0
        )
        if input_tokens is not None:
            LLM_TOKENS_TOTAL.labels(model=model, direction="input").inc(input_tokens)
        if output_tokens is not None:
            LLM_TOKENS_TOTAL.labels(model=model, direction="output").inc(output_tokens)
        if input_tokens is not None and output_tokens is not None:
            cost = _estimate_llm_cost_usd(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            if cost > 0:
                LLM_COST_USD_TOTAL.labels(model=model).inc(cost)
        if not success and refusal_code is not None:
            LLM_REFUSALS_TOTAL.labels(code=refusal_code).inc()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Metrics: record_llm_call failed silently: %s", exc)


def record_webhook_delivery(*, delivered: bool) -> None:
    """Increment the webhook delivery counter."""
    try:
        outcome = "delivered" if delivered else "failed"
        WEBHOOK_DELIVERIES_TOTAL.labels(outcome=outcome).inc()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Metrics: record_webhook_delivery failed silently: %s", exc)


def record_feedback(*, disposition: str, tier: str) -> None:
    """Increment the feedback counter by disposition and source tier."""
    try:
        FEEDBACK_TOTAL.labels(disposition=disposition, tier=tier).inc()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Metrics: record_feedback failed silently: %s", exc)


def record_drift_event(*, feature: str, severity: str) -> None:
    """Increment the drift-event counter for a single feature."""
    try:
        DRIFT_EVENTS_TOTAL.labels(feature=feature, severity=severity).inc()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Metrics: record_drift_event failed silently: %s", exc)
