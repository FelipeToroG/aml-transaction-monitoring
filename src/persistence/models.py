"""SQLAlchemy 2.x models for the alert persistence layer.

Three tables capture the full alert lifecycle:

* :class:`Alert` — one row per scored transaction that crossed the
  alert threshold. Carries the alert's score breakdown, the assigned
  tier, the assembled evidence bundle, the generated narrative (or
  refusal), and the current operational status.
* :class:`Feedback` — investigator dispositions recorded against
  alerts. One alert may have multiple feedback entries over time (a
  tier-1 alert escalated to tier-2 review accumulates a chain of
  feedbacks).
* :class:`AuditLog` — append-only event stream. Every state transition
  on an alert, every model load, every drift-detection event lands
  here. Regulators reviewing a historical incident query this table.

Why three tables and not one
----------------------------
A single ``alerts`` table with embedded feedback and audit columns
would degrade into a wide row that nobody can query efficiently. Three
normalised tables make the access patterns explicit: the API mostly
reads ``alerts`` and writes ``feedback``; the audit log is append-only
and queried only for compliance review. The trade-off is two extra
joins on the rare cross-table query, which both databases optimise
trivially.

JSON columns for structured payloads
------------------------------------
``evidence_snapshot`` (on Alert) and ``event_data`` (on AuditLog) are
SQLAlchemy ``JSON`` columns. SQLite stores them as TEXT; Postgres
stores them as ``jsonb`` and exposes structured querying. The model
definitions are portable across both; the only behavioural difference
is that Postgres deployments gain operator-side JSON query support
without code changes.
"""

from __future__ import annotations

import datetime as dt
import enum
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.persistence.db import Base


# ---------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------
# Enums are persisted as strings (not integers) so the database is
# readable from a SQL prompt without having to remember the mapping.
# A schema migration that adds a value remains backwards-compatible
# with existing rows.


class AlertStatus(str, enum.Enum):
    """Lifecycle status of an alert.

    The state machine is:

        open -> in_review -> { cleared | escalated | sar_filed }

    Open alerts await investigator pickup. In-review alerts have been
    claimed but not dispositioned. Cleared, escalated, and sar_filed
    are terminal dispositions.
    """

    OPEN = "open"
    IN_REVIEW = "in_review"
    CLEARED = "cleared"
    ESCALATED = "escalated"
    SAR_FILED = "sar_filed"


class AlertTier(str, enum.Enum):
    """Tier classification for an alert. Mirrors alert_thresholds.yaml."""

    TIER_3_CRITICAL = "tier_3_critical"
    TIER_2_HIGH = "tier_2_high"
    TIER_1_MEDIUM = "tier_1_medium"
    SUPPRESSED = "suppressed"


class FeedbackDisposition(str, enum.Enum):
    """Investigator disposition values captured via the feedback endpoint.

    Distinct from :class:`AlertStatus` because an alert can accumulate
    multiple feedback entries (e.g., tier-1 cleared by a reviewer is
    later escalated by a tier-2 reviewer). The latest feedback
    determines the canonical alert status.
    """

    CLEARED = "cleared"
    ESCALATED = "escalated"
    SAR_FILED = "sar_filed"


class AuditEventType(str, enum.Enum):
    """Types of events written to the audit log.

    Kept as an open-ended enum so adding a new event class (e.g., a
    new monitoring signal) is a one-line schema change rather than a
    table migration.
    """

    ALERT_CREATED = "alert_created"
    ALERT_TRIAGED = "alert_triaged"
    ALERT_STATUS_CHANGED = "alert_status_changed"
    FEEDBACK_RECORDED = "feedback_recorded"
    WEBHOOK_DISPATCHED = "webhook_dispatched"
    WEBHOOK_FAILED = "webhook_failed"
    MODEL_LOADED = "model_loaded"
    DRIFT_DETECTED = "drift_detected"


# ---------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------


class Alert(Base):
    """One row per scored transaction that crossed the alert threshold.

    Indexed on ``status``, ``tier``, and ``created_at`` because those
    are the columns the investigator UI's queue page filters and sorts
    on. ``transaction_id`` carries a single-column index because the
    audit-trace endpoint looks alerts up by transaction.
    """

    __tablename__ = "alerts"

    alert_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        comment="Stable alert identifier (UUID4). Generated at alert creation.",
    )
    transaction_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
        comment="Identifier of the underlying scored transaction.",
    )

    # ----- Scoring breakdown -----
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    supervised_score: Mapped[float] = mapped_column(Float, nullable=False)
    threshold_applied: Mapped[float] = mapped_column(Float, nullable=False)

    tier: Mapped[AlertTier] = mapped_column(
        Enum(AlertTier, native_enum=False),
        nullable=False,
        index=True,
    )
    status: Mapped[AlertStatus] = mapped_column(
        Enum(AlertStatus, native_enum=False),
        nullable=False,
        default=AlertStatus.OPEN,
        index=True,
    )

    # ----- Evidence and narrative -----
    suspected_typology: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evidence_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        comment="Full evidence bundle assembled at scoring time, JSON-encoded.",
    )
    narrative_payload: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        comment=(
            "Narrator output. Either the full CaseNarrative dict or a "
            "RefusalReason dict. Null when triage has not yet run."
        ),
    )
    narrative_confidence: Mapped[str | None] = mapped_column(
        String(16),
        nullable=True,
        comment="low / medium / high. Mirrors CaseNarrative.confidence.",
    )
    narrator_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    narrator_prompt_version: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # ----- Model provenance -----
    model_version: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="Service version that produced the alert. Mirrors src.__version__.",
    )
    model_schema_version: Mapped[int] = mapped_column(
        nullable=False,
        comment="Ensemble schema version embedded in models/ensemble.pkl.",
    )

    # ----- Timestamps -----
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: dt.datetime.now(dt.timezone.utc),
        index=True,
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: dt.datetime.now(dt.timezone.utc),
        onupdate=lambda: dt.datetime.now(dt.timezone.utc),
    )

    # ----- Relationships -----
    feedback_entries: Mapped[list["Feedback"]] = relationship(
        back_populates="alert",
        cascade="all, delete-orphan",
        order_by="Feedback.created_at",
    )

    # Composite indexes for the investigator UI's most common access
    # patterns. The (status, tier, created_at) index supports "show me
    # the open tier-2 alerts in chronological order"; the
    # (tier, created_at) index supports "show me all today's tier-3
    # alerts regardless of disposition".
    __table_args__ = (
        Index("ix_alerts_status_tier_created", "status", "tier", "created_at"),
        Index("ix_alerts_tier_created", "tier", "created_at"),
    )


class Feedback(Base):
    """Investigator disposition recorded against an alert.

    An alert may have multiple feedback entries; the canonical alert
    status is derived from the latest one. Capturing the history rather
    than overwriting matters for audit traceability (the reviewer
    sequence is what regulators ask for) and for active-learning
    pipelines (every feedback is a labelled example).
    """

    __tablename__ = "feedback"

    feedback_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    alert_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("alerts.alert_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    investigator_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Identifier for the investigator submitting the feedback.",
    )
    disposition: Mapped[FeedbackDisposition] = mapped_column(
        Enum(FeedbackDisposition, native_enum=False),
        nullable=False,
    )
    justification: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment=(
            "Free-text reasoning. Optional for cleared, recommended for "
            "escalated, mandatory for sar_filed in the API validation layer."
        ),
    )

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: dt.datetime.now(dt.timezone.utc),
        index=True,
    )

    alert: Mapped[Alert] = relationship(back_populates="feedback_entries")


class AuditLog(Base):
    """Append-only event log for compliance and operational review.

    Every state transition writes one row here. The table is queried
    rarely (compliance reviews, post-incident investigations) but
    written constantly, so the column set is intentionally narrow and
    the indexing favours write throughput over query flexibility.
    """

    __tablename__ = "audit_log"

    audit_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    alert_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("alerts.alert_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Optional: not all events are alert-scoped (e.g., model_loaded).",
    )
    event_type: Mapped[AuditEventType] = mapped_column(
        Enum(AuditEventType, native_enum=False),
        nullable=False,
        index=True,
    )
    event_data: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        comment="Free-form event payload; structure depends on event_type.",
    )
    service_version: Mapped[str] = mapped_column(String(32), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(32), nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: dt.datetime.now(dt.timezone.utc),
        index=True,
    )

    # Compound index supports the most common audit query:
    # "show me all events of type X for alert Y in date range Z".
    __table_args__ = (
        Index("ix_audit_alert_type_created", "alert_id", "event_type", "created_at"),
    )
