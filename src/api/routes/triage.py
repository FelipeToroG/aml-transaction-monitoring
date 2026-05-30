"""On-demand triage endpoint.

``POST /triage`` re-runs the Claude narrator against an existing
alert's evidence bundle. Used by the investigator UI when a
human-in-the-loop reviewer wants a narrative for a tier-1 alert
whose narrative was deferred at scoring time, and by the test suite
to exercise the narrator path without going through the full scoring
flow.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.dependencies import get_db, get_narrator
from src.api.schemas import TriageRequest, TriageResponse
from src.persistence.models import AuditEventType
from src.persistence.repository import AlertRepository, AuditLogRepository
from src.triage.narrator import EvidenceBundle, Narrator

logger = logging.getLogger(__name__)
router = APIRouter(tags=["triage"])


@router.post(
    "/triage",
    response_model=TriageResponse,
    summary="Generate a case narrative for an existing alert.",
)
def triage_alert(
    request: TriageRequest,
    narrator: Annotated[Narrator, Depends(get_narrator)],
    session: Annotated[Session, Depends(get_db)],
) -> TriageResponse:
    """Generate (or regenerate) a case narrative.

    The endpoint always re-invokes the narrator even if the alert
    already has a narrative on file. Re-running is a normal operation
    when the prompt version has been bumped or when an investigator
    wants a fresh take on an ambiguous alert. The audit log captures
    every invocation, so a regulator can reconstruct the history of
    narrative revisions for any alert.
    """
    alert_repo = AlertRepository(session)
    audit_repo = AuditLogRepository(session)

    alert = alert_repo.get_alert(alert_id=request.alert_id)
    if alert is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {request.alert_id} not found.",
        )

    snapshot = alert.evidence_snapshot or {}
    bundle = _evidence_bundle_from_snapshot(alert_id=alert.alert_id, snapshot=snapshot)

    result = narrator.generate(bundle, use_eval_model=request.use_eval_model)

    # Persist the result the same way the scoring path's background
    # task does. The conditional below mirrors the persistence logic
    # in ``src.api.routes.score._persist_narrative_result`` and is
    # duplicated rather than extracted because the audit-event types
    # differ and a thin helper would be more confusing than this
    # explicit branch.
    if result.success and result.narrative is not None:
        narrative_payload = result.narrative.model_dump()
        confidence = result.narrative.confidence
    elif result.refusal is not None:
        narrative_payload = {"refusal": result.refusal.model_dump()}
        confidence = None
    else:
        narrative_payload = {"error": "Empty narrator result"}
        confidence = None

    alert_repo.set_narrative(
        alert_id=alert.alert_id,
        narrative_payload=narrative_payload,
        narrative_confidence=confidence,
        narrator_model=result.model_name,
        narrator_prompt_version=result.prompt_version,
    )
    audit_repo.write_event(
        event_type=AuditEventType.ALERT_TRIAGED,
        alert_id=alert.alert_id,
        event_data={
            "success": result.success,
            "model": result.model_name,
            "prompt_version": result.prompt_version,
            "latency_ms": result.latency_ms,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "invocation": "on_demand",
        },
    )

    return TriageResponse(
        alert_id=alert.alert_id,
        success=result.success,
        narrative=result.narrative,
        refusal=result.refusal,
        model_name=result.model_name,
        prompt_version=result.prompt_version,
        latency_ms=result.latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


def _evidence_bundle_from_snapshot(
    *, alert_id: str, snapshot: dict
) -> EvidenceBundle:
    """Convert a stored evidence snapshot dict into an EvidenceBundle.

    Local helper to avoid a circular import between the score and
    triage modules. The conversion is straightforward: keys map
    one-to-one with the EvidenceBundle dataclass fields.
    """
    import datetime as dt

    scoring = snapshot.get("scoring", {})
    return EvidenceBundle(
        alert_id=alert_id,
        risk_score=float(scoring.get("risk_score", 0.0)),
        anomaly_score=float(scoring.get("anomaly_score", 0.0)),
        supervised_score=float(scoring.get("supervised_score", 0.0)),
        tier=str(scoring.get("tier", "unknown")),
        generated_at_iso=dt.datetime.now(dt.timezone.utc).isoformat(),
        transaction=snapshot.get("transaction", {}),
        source_activity=snapshot.get("source_activity", {}),
        destination_activity=snapshot.get("destination_activity", {}),
        triggered_features=snapshot.get("triggered_features", []),
    )
