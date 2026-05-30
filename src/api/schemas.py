"""Pydantic v2 request and response schemas for the AML API.

Every input model sets ``extra='forbid'`` so unknown fields produce
an HTTP 422 rather than silent acceptance. Silent acceptance of
unknown fields is the most common path to client-server contract
drift; the API surfaces it loudly instead.

The schemas import from the canonical domain types where possible
(e.g., ``CaseNarrative`` from ``src.triage``) so the HTTP surface and
the internal types stay in lock-step. A change to the internal
narrative shape propagates to the API contract automatically and
breaks the OpenAPI doc test in CI — the right place to catch it.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.persistence.models import AlertStatus, AlertTier, FeedbackDisposition
from src.triage.schemas import CaseNarrative, RefusalReason


# ---------------------------------------------------------------------
# /score
# ---------------------------------------------------------------------


class TransactionScoreRequest(BaseModel):
    """A single transaction to score.

    Mirrors the IBM AML HI-Small schema. Field names use the same
    underscore-cased Python identifiers exported by
    :mod:`src.data.loader`; the underlying CSV's mixed-case column
    names with spaces are mapped at the API boundary.
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [
        {
            "transaction_id": "txn_2026_05_01_000123",
            "timestamp": "2026-05-01T14:23:11Z",
            "source_bank": 11,
            "source_account": "8000A1B2",
            "dest_bank": 22,
            "dest_account": "8000C3D4",
            "amount_received": 9850.00,
            "receiving_currency": "USD",
            "amount_paid": 9850.00,
            "payment_currency": "USD",
            "payment_format": "Wire",
        }
    ]})

    transaction_id: str = Field(..., min_length=1, max_length=128)
    timestamp: dt.datetime
    source_bank: int = Field(..., ge=0)
    source_account: str = Field(..., min_length=1, max_length=64)
    dest_bank: int = Field(..., ge=0)
    dest_account: str = Field(..., min_length=1, max_length=64)
    amount_received: float = Field(..., ge=0)
    receiving_currency: str = Field(..., min_length=3, max_length=8)
    amount_paid: float = Field(..., ge=0)
    payment_currency: str = Field(..., min_length=3, max_length=8)
    payment_format: str = Field(..., min_length=1, max_length=32)


class TopFeatureContribution(BaseModel):
    """A single feature's contribution surfaced on the score response."""

    model_config = ConfigDict(extra="forbid")

    feature_name: str
    observed_value: float
    contribution_rank: int = Field(
        ..., ge=1, description="1 = strongest contributor; ascending"
    )


class TransactionScoreResponse(BaseModel):
    """Score endpoint response.

    ``alert_id`` is populated when the score crossed the alert
    threshold and an alert was persisted. Below the threshold,
    ``alert_id`` is null and ``alert_created`` is false — the caller
    has scored the transaction but no operational alert exists.
    """

    model_config = ConfigDict(extra="forbid")

    transaction_id: str
    risk_score: float = Field(..., ge=0, le=1)
    anomaly_score: float = Field(..., ge=0, le=1)
    supervised_score: float = Field(..., ge=0, le=1)
    tier: AlertTier
    alert_created: bool
    alert_id: str | None = None
    threshold_applied: float = Field(..., ge=0, le=1)
    model_version: str
    model_schema_version: int
    inference_latency_ms: float = Field(..., ge=0)
    top_features: list[TopFeatureContribution] = Field(default_factory=list)


# ---------------------------------------------------------------------
# /triage
# ---------------------------------------------------------------------


class TriageRequest(BaseModel):
    """Request a case narrative for an existing alert."""

    model_config = ConfigDict(extra="forbid")

    alert_id: str = Field(..., min_length=1, max_length=64)
    use_eval_model: bool = Field(
        default=False,
        description=(
            "When true, route to the cheaper eval-tier LLM. Used by offline "
            "replays and the test suite to avoid burning production budget."
        ),
    )


class TriageResponse(BaseModel):
    """Triage endpoint response.

    Exactly one of ``narrative`` or ``refusal`` is populated, mirroring
    the discriminated union in :class:`~src.triage.NarratorResult`.
    """

    model_config = ConfigDict(extra="forbid")

    alert_id: str
    success: bool
    narrative: CaseNarrative | None = None
    refusal: RefusalReason | None = None
    model_name: str
    prompt_version: str
    latency_ms: float = Field(..., ge=0)
    input_tokens: int | None = None
    output_tokens: int | None = None


# ---------------------------------------------------------------------
# /feedback
# ---------------------------------------------------------------------


class FeedbackRequest(BaseModel):
    """Investigator disposition for an alert."""

    model_config = ConfigDict(extra="forbid")

    alert_id: str = Field(..., min_length=1, max_length=64)
    investigator_id: str = Field(..., min_length=1, max_length=64)
    disposition: FeedbackDisposition
    justification: str | None = Field(
        default=None,
        max_length=4000,
        description=(
            "Free-text reasoning. Required by API validation for SAR_FILED "
            "and recommended for ESCALATED; optional for CLEARED."
        ),
    )


class FeedbackResponse(BaseModel):
    """Feedback endpoint response."""

    model_config = ConfigDict(extra="forbid")

    feedback_id: str
    alert_id: str
    new_alert_status: AlertStatus
    recorded_at: dt.datetime


# ---------------------------------------------------------------------
# /alerts
# ---------------------------------------------------------------------


class AlertSummary(BaseModel):
    """Compact alert representation for the queue listing."""

    model_config = ConfigDict(extra="forbid")

    alert_id: str
    transaction_id: str
    risk_score: float
    tier: AlertTier
    status: AlertStatus
    suspected_typology: str | None
    has_narrative: bool
    created_at: dt.datetime


class AlertsListResponse(BaseModel):
    """Paginated queue response."""

    model_config = ConfigDict(extra="forbid")

    alerts: list[AlertSummary]
    total_matching: int = Field(..., ge=0)
    limit: int = Field(..., ge=1)
    offset: int = Field(..., ge=0)


# ---------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Liveness and readiness response.

    Includes the configured threshold so a deployer can verify a fresh
    rollout is actually serving the version they expect — the embedded
    threshold differs between releases and is the single most common
    "did the deployment actually happen?" signal.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded"]
    service_version: str
    model_loaded: bool
    model_version: str | None
    model_schema_version: int | None
    decision_threshold: float | None
    database_healthy: bool
    llm_configured: bool
    langfuse_configured: bool
    webhook_enabled: bool


# ---------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------


class APIError(BaseModel):
    """Structured error envelope.

    FastAPI's default error envelope is a bare ``{"detail": "..."}``.
    Wrapping it in a typed schema gives clients something to ``isinstance``
    against and gives us a place to add fields (request_id, retry_hint)
    without a breaking change.
    """

    model_config = ConfigDict(extra="forbid")

    detail: str
    code: str = "internal_error"
    context: dict[str, Any] = Field(default_factory=dict)
