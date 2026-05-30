"""Observability: LLM tracing and Prometheus metrics.

Two production telemetry concerns:

* ``tracing`` — Langfuse client construction. Every Claude call passes
  through a context-managed trace that records the prompt, the
  validated response, token usage, latency, and the alert ID. Traces
  are searchable in Langfuse for quality review and cost analysis.
* ``metrics`` — Prometheus metric definitions and producer helpers.
  Exposes inference latency histograms, alert volume counters by tier,
  feedback distribution, LLM cost and latency, webhook delivery
  outcomes, and drift events.

Both subsystems are best-effort: a Langfuse outage or Prometheus
scraping failure must not affect the scoring path. The producer
helpers wrap their calls in try/except guards that log and continue.
"""

from src.observability.metrics import (
    ALERTS_CREATED_TOTAL,
    DRIFT_EVENTS_TOTAL,
    FEEDBACK_TOTAL,
    LLM_COST_USD_TOTAL,
    LLM_LATENCY_SECONDS,
    LLM_PRICE_TABLE,
    LLM_REFUSALS_TOTAL,
    LLM_TOKENS_TOTAL,
    SCORE_DISTRIBUTION,
    WEBHOOK_DELIVERIES_TOTAL,
    record_alert_created,
    record_drift_event,
    record_feedback,
    record_llm_call,
    record_score,
    record_webhook_delivery,
)
from src.observability.tracing import build_langfuse_client

__all__ = [
    "ALERTS_CREATED_TOTAL",
    "DRIFT_EVENTS_TOTAL",
    "FEEDBACK_TOTAL",
    "LLM_COST_USD_TOTAL",
    "LLM_LATENCY_SECONDS",
    "LLM_PRICE_TABLE",
    "LLM_REFUSALS_TOTAL",
    "LLM_TOKENS_TOTAL",
    "SCORE_DISTRIBUTION",
    "WEBHOOK_DELIVERIES_TOTAL",
    "build_langfuse_client",
    "record_alert_created",
    "record_drift_event",
    "record_feedback",
    "record_llm_call",
    "record_score",
    "record_webhook_delivery",
]
