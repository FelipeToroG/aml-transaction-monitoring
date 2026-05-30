"""Alerts listing endpoint.

``GET /alerts`` returns a paginated, filterable view of the alert
queue. The Streamlit investigator UI's queue page calls this
endpoint; downstream operational reporting and the per-alert detail
view also depend on it.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.schemas import AlertsListResponse, AlertSummary
from src.persistence.models import AlertStatus, AlertTier
from src.persistence.repository import AlertRepository

logger = logging.getLogger(__name__)
router = APIRouter(tags=["alerts"])


@router.get(
    "/alerts",
    response_model=AlertsListResponse,
    summary="List alerts with optional filters and pagination.",
)
def list_alerts(
    session: Annotated[Session, Depends(get_db)],
    status_filter: Annotated[
        AlertStatus | None,
        Query(
            alias="status",
            description="Optional status filter (open, in_review, cleared, escalated, sar_filed).",
        ),
    ] = None,
    tier: Annotated[
        AlertTier | None,
        Query(description="Optional tier filter."),
    ] = None,
    created_after: Annotated[
        dt.datetime | None,
        Query(description="Return alerts created at or after this ISO-8601 timestamp."),
    ] = None,
    created_before: Annotated[
        dt.datetime | None,
        Query(description="Return alerts created strictly before this ISO-8601 timestamp."),
    ] = None,
    limit: Annotated[
        int,
        Query(ge=1, le=500, description="Page size; capped at 500."),
    ] = 100,
    offset: Annotated[
        int,
        Query(ge=0, description="Page offset for pagination."),
    ] = 0,
) -> AlertsListResponse:
    """Return a page of alerts matching the supplied filters.

    Filters AND-combine. Ordering is newest-first by ``created_at``.
    The ``total_matching`` field reflects the full count across all
    pages so the UI can render a "page n of m" indicator.
    """
    alert_repo = AlertRepository(session)

    alerts = alert_repo.list_alerts(
        status=status_filter,
        tier=tier,
        created_after=created_after,
        created_before=created_before,
        limit=limit,
        offset=offset,
    )
    total_matching = alert_repo.count_alerts(
        status=status_filter,
        tier=tier,
    )

    summaries = [
        AlertSummary(
            alert_id=alert.alert_id,
            transaction_id=alert.transaction_id,
            risk_score=alert.risk_score,
            tier=alert.tier,
            status=alert.status,
            suspected_typology=alert.suspected_typology,
            has_narrative=bool(alert.narrative_payload),
            created_at=alert.created_at,
        )
        for alert in alerts
    ]

    return AlertsListResponse(
        alerts=summaries,
        total_matching=total_matching,
        limit=limit,
        offset=offset,
    )
