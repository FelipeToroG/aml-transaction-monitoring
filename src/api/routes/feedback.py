"""Investigator feedback endpoint.

``POST /feedback`` accepts an investigator's disposition for an alert.
The endpoint persists the feedback row, transitions the alert's
canonical status to match the disposition, and writes an audit-log
event. The investigator UI is the primary client; an HTTP client is
also used by the offline-replay tooling to bulk-import historical
dispositions for retraining.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.schemas import FeedbackRequest, FeedbackResponse
from src.observability.metrics import record_feedback
from src.persistence.models import (
    AlertStatus,
    AuditEventType,
    FeedbackDisposition,
)
from src.persistence.repository import AlertRepository, AuditLogRepository, FeedbackRepository

logger = logging.getLogger(__name__)
router = APIRouter(tags=["feedback"])


# Mapping from investigator disposition to the resulting canonical
# alert status. Centralised here so the state-transition policy is
# inspectable in one place rather than scattered across the routing
# layer. A new disposition value requires updating this map.
_DISPOSITION_TO_STATUS: dict[FeedbackDisposition, AlertStatus] = {
    FeedbackDisposition.CLEARED: AlertStatus.CLEARED,
    FeedbackDisposition.ESCALATED: AlertStatus.ESCALATED,
    FeedbackDisposition.SAR_FILED: AlertStatus.SAR_FILED,
}


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record an investigator's disposition for an alert.",
)
def record_feedback(
    request: FeedbackRequest,
    session: Annotated[Session, Depends(get_db)],
) -> FeedbackResponse:
    """Persist investigator feedback and transition alert status.

    Business-rule validation:

    * Filing a SAR requires a justification. SARs are regulator-bound
      filings; an empty justification on a SAR record is an audit
      finding waiting to happen, so we reject it at the API layer
      rather than at investigator-document-review time.
    """
    if request.disposition == FeedbackDisposition.SAR_FILED and not request.justification:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Justification is required when filing a SAR. "
                "Provide investigator-authored reasoning in the justification field."
            ),
        )

    alert_repo = AlertRepository(session)
    feedback_repo = FeedbackRepository(session)
    audit_repo = AuditLogRepository(session)

    alert = alert_repo.get_alert(alert_id=request.alert_id)
    if alert is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {request.alert_id} not found.",
        )

    # Persist the feedback row first so the audit-log event below can
    # reference it. Order matters: the feedback is the system of record
    # for the disposition; the alert status change is derived.
    feedback = feedback_repo.create_feedback(
        alert_id=alert.alert_id,
        investigator_id=request.investigator_id,
        disposition=request.disposition,
        justification=request.justification,
    )

    # Apply the status transition. The repository does not enforce
    # state-machine constraints; the policy lives here so a future
    # privileged-reopen route can bypass it cleanly.
    new_status = _DISPOSITION_TO_STATUS[request.disposition]
    alert_repo.update_status(alert_id=alert.alert_id, new_status=new_status)

    audit_repo.write_event(
        event_type=AuditEventType.FEEDBACK_RECORDED,
        alert_id=alert.alert_id,
        event_data={
            "feedback_id": feedback.feedback_id,
            "investigator_id": request.investigator_id,
            "disposition": request.disposition.value,
            "previous_status": alert.status.value,
            "new_status": new_status.value,
            "justification_present": bool(request.justification),
        },
    )

    # Increment the feedback distribution counter. The disposition-by-tier
    # breakdown is the headline signal that the model is producing the
    # right alerts at the right rates — a sustained shift in
    # cleared-rate-per-tier indicates either model drift or the
    # investigator team's evolving disposition policy.
    record_feedback(disposition=request.disposition.value, tier=alert.tier.value)

    return FeedbackResponse(
        feedback_id=feedback.feedback_id,
        alert_id=alert.alert_id,
        new_alert_status=new_status,
        recorded_at=feedback.created_at,
    )
