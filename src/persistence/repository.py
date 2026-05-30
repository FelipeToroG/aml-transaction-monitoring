"""Repository pattern over the persistence models.

Repository classes are the only place SQL-like query construction
happens. Every API route and every service-layer caller obtains a
repository via dependency injection, never a bare ``Session``. This
buys two things:

1. **Testability.** The test suite swaps in an in-memory SQLite
   session at fixture time; nothing else in the application changes.
2. **Query auditability.** Every persistence query is a typed method
   with explicit parameters. There is no string-substituted SQL
   anywhere, and there is one place to add row-level access controls
   when the deployment grows multi-tenant.

Three repositories — one per table — keep the responsibilities clean.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src import __version__
from src.persistence.models import (
    Alert,
    AlertStatus,
    AlertTier,
    AuditEventType,
    AuditLog,
    Feedback,
    FeedbackDisposition,
)


def _generate_id() -> str:
    """Generate a UUID4 primary key as a string.

    Centralised so every table gets the same ID shape. UUID4 is
    appropriate because alert IDs are globally unique across deployment
    environments (we frequently replay alerts from production into
    staging for incident review), and a UUID4 collision rate of one in
    five undecillion is operationally indistinguishable from never.
    """
    return uuid.uuid4().hex


# ---------------------------------------------------------------------
# Alert repository
# ---------------------------------------------------------------------


class AlertRepository:
    """CRUD and query interface for the alerts table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create_alert(
        self,
        *,
        transaction_id: str,
        risk_score: float,
        anomaly_score: float,
        supervised_score: float,
        threshold_applied: float,
        tier: AlertTier,
        evidence_snapshot: dict[str, Any],
        model_version: str,
        model_schema_version: int,
        suspected_typology: str | None = None,
    ) -> Alert:
        """Persist a freshly created alert.

        The narrator output (``narrative_payload``, etc.) is null at
        creation time and populated later via :meth:`set_narrative`.
        This split keeps the scoring path's transactional boundary
        minimal — the LLM call happens outside the alert's creation
        commit so a triage failure does not roll back the alert
        record itself.
        """
        alert = Alert(
            alert_id=_generate_id(),
            transaction_id=transaction_id,
            risk_score=risk_score,
            anomaly_score=anomaly_score,
            supervised_score=supervised_score,
            threshold_applied=threshold_applied,
            tier=tier,
            status=AlertStatus.OPEN,
            suspected_typology=suspected_typology,
            evidence_snapshot=evidence_snapshot,
            model_version=model_version,
            model_schema_version=model_schema_version,
        )
        self._session.add(alert)
        self._session.flush()  # populate alert_id without committing the surrounding transaction
        return alert

    def set_narrative(
        self,
        *,
        alert_id: str,
        narrative_payload: dict[str, Any],
        narrative_confidence: str | None,
        narrator_model: str,
        narrator_prompt_version: str,
    ) -> Alert:
        """Attach the narrator's output to an existing alert.

        Raises ``LookupError`` if no alert exists with the given ID.
        """
        alert = self.get_alert(alert_id=alert_id)
        if alert is None:
            raise LookupError(f"No alert with id {alert_id!r}")
        alert.narrative_payload = narrative_payload
        alert.narrative_confidence = narrative_confidence
        alert.narrator_model = narrator_model
        alert.narrator_prompt_version = narrator_prompt_version
        self._session.flush()
        return alert

    def get_alert(self, *, alert_id: str) -> Alert | None:
        """Retrieve a single alert by ID; returns None if absent."""
        return self._session.get(Alert, alert_id)

    def list_alerts(
        self,
        *,
        status: AlertStatus | None = None,
        tier: AlertTier | None = None,
        created_after: dt.datetime | None = None,
        created_before: dt.datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Alert]:
        """Filtered, paginated alert query for the investigator queue.

        Filters are AND-combined. Ordering is newest-first by
        ``created_at``. Default ``limit=100`` matches a reasonable
        single-page batch for the investigator UI; the offset
        parameter supports straightforward pagination.
        """
        stmt = select(Alert)
        if status is not None:
            stmt = stmt.where(Alert.status == status)
        if tier is not None:
            stmt = stmt.where(Alert.tier == tier)
        if created_after is not None:
            stmt = stmt.where(Alert.created_at >= created_after)
        if created_before is not None:
            stmt = stmt.where(Alert.created_at < created_before)
        stmt = stmt.order_by(Alert.created_at.desc()).limit(limit).offset(offset)
        return self._session.scalars(stmt).all()

    def update_status(
        self,
        *,
        alert_id: str,
        new_status: AlertStatus,
    ) -> Alert:
        """Transition an alert to a new status.

        The repository does not enforce the state-machine transitions
        (open → in_review → terminal); that policy lives in the API
        route handler so a privileged-override path can bypass it. The
        repository simply updates the field and records the change in
        the caller's transaction.
        """
        alert = self.get_alert(alert_id=alert_id)
        if alert is None:
            raise LookupError(f"No alert with id {alert_id!r}")
        alert.status = new_status
        self._session.flush()
        return alert

    def count_alerts(
        self,
        *,
        status: AlertStatus | None = None,
        tier: AlertTier | None = None,
    ) -> int:
        """Return the count of alerts matching the filter combination.

        Used by the investigator-queue header and by the operator
        dashboard's "alerts open right now" gauge. The implementation
        materialises a count rather than ``len(list(...))`` so it
        round-trips a single integer rather than every matching row.
        """
        from sqlalchemy import func

        stmt = select(func.count()).select_from(Alert)
        if status is not None:
            stmt = stmt.where(Alert.status == status)
        if tier is not None:
            stmt = stmt.where(Alert.tier == tier)
        return int(self._session.scalar(stmt) or 0)


# ---------------------------------------------------------------------
# Feedback repository
# ---------------------------------------------------------------------


class FeedbackRepository:
    """CRUD and query interface for the feedback table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create_feedback(
        self,
        *,
        alert_id: str,
        investigator_id: str,
        disposition: FeedbackDisposition,
        justification: str | None = None,
    ) -> Feedback:
        """Record an investigator disposition.

        Persists the feedback row and returns it. The caller is
        responsible for updating the alert's canonical status if the
        feedback should drive the status transition (typically yes;
        the API route does this in the same transaction).
        """
        feedback = Feedback(
            feedback_id=_generate_id(),
            alert_id=alert_id,
            investigator_id=investigator_id,
            disposition=disposition,
            justification=justification,
        )
        self._session.add(feedback)
        self._session.flush()
        return feedback

    def list_for_alert(self, *, alert_id: str) -> Sequence[Feedback]:
        """Return feedback entries for a single alert in chronological order."""
        stmt = (
            select(Feedback)
            .where(Feedback.alert_id == alert_id)
            .order_by(Feedback.created_at.asc())
        )
        return self._session.scalars(stmt).all()


# ---------------------------------------------------------------------
# Audit log repository
# ---------------------------------------------------------------------


class AuditLogRepository:
    """Append-only writer and bounded reader for the audit log.

    The audit log has different access patterns from the alert table:
    writes happen on every state transition, reads happen rarely (during
    compliance review). The repository surface reflects this — write
    methods are first-class, read methods are deliberately narrow.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def write_event(
        self,
        *,
        event_type: AuditEventType,
        event_data: dict[str, Any],
        alert_id: str | None = None,
        service_version: str | None = None,
        model_version: str | None = None,
    ) -> AuditLog:
        """Append one event to the audit log.

        ``service_version`` defaults to the running service version if
        not supplied; explicit override is supported because batch
        replays and tests need to write events stamped with historical
        versions.
        """
        event = AuditLog(
            audit_id=_generate_id(),
            alert_id=alert_id,
            event_type=event_type,
            event_data=event_data,
            service_version=service_version or __version__,
            model_version=model_version,
        )
        self._session.add(event)
        self._session.flush()
        return event

    def list_events_for_alert(
        self,
        *,
        alert_id: str,
        limit: int = 1000,
    ) -> Sequence[AuditLog]:
        """Return all audit events for an alert in chronological order."""
        stmt = (
            select(AuditLog)
            .where(AuditLog.alert_id == alert_id)
            .order_by(AuditLog.created_at.asc())
            .limit(limit)
        )
        return self._session.scalars(stmt).all()

    def list_recent_events(
        self,
        *,
        event_type: AuditEventType | None = None,
        since: dt.datetime | None = None,
        limit: int = 500,
    ) -> Sequence[AuditLog]:
        """Return recent events for the operator dashboard.

        Filters are AND-combined. Ordering is newest-first.
        """
        stmt = select(AuditLog)
        if event_type is not None:
            stmt = stmt.where(AuditLog.event_type == event_type)
        if since is not None:
            stmt = stmt.where(AuditLog.created_at >= since)
        stmt = stmt.order_by(AuditLog.created_at.desc()).limit(limit)
        return self._session.scalars(stmt).all()
