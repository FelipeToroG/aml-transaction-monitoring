"""Health and metrics endpoints.

``GET /health`` returns a structured liveness/readiness response
including model version and decision threshold - the latter is the
most reliable "is this actually the deployment I expect" signal,
because the threshold changes between training runs while everything
else (version string) may not.

``GET /metrics`` is exposed by ``prometheus-fastapi-instrumentator``
which is wired in ``src.api.main``. This module does not need to
declare a separate route for it.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from src import __version__
from src.api.dependencies import get_db, get_ensemble
from src.api.schemas import HealthResponse
from src.models.ensemble import AMLEnsemble
from src.utils.config import Settings, get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    settings: Annotated[Settings, Depends(get_settings)],
    session: Annotated[Session, Depends(get_db)],
) -> HealthResponse:
    """Liveness and readiness probe.

    The endpoint deliberately does not depend on the LLM provider
    being reachable - Claude availability is not a readiness condition
    for the service because the scoring path runs without LLM
    involvement. Triage-only failure modes degrade gracefully via the
    narrator's refusal path.
    """
    # ----- Model load check -----
    # Attempt to access the cached ensemble. We swallow load errors
    # here because health probes must return promptly even when the
    # model artifact is missing - the operator needs that signal.
    model_loaded = False
    model_version: str | None = None
    model_schema_version: int | None = None
    decision_threshold: float | None = None
    try:
        ensemble: AMLEnsemble = get_ensemble()
        model_loaded = True
        if ensemble.metadata is not None:
            model_version = ensemble.metadata.service_version
            model_schema_version = ensemble.metadata.schema_version
        decision_threshold = ensemble.decision_threshold
    except Exception as exc:  # noqa: BLE001
        logger.warning("Health probe: ensemble load failed: %s", exc)

    # ----- Database reachability check -----
    try:
        session.execute(text("SELECT 1"))
        database_healthy = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Health probe: database check failed: %s", exc)
        database_healthy = False

    # ----- Configuration presence checks -----
    llm_configured = bool(settings.anthropic_api_key)
    langfuse_configured = bool(
        settings.langfuse_public_key and settings.langfuse_secret_key
    )

    # ----- Overall status -----
    # ``degraded`` indicates the service is up but a component is not
    # available. ``ok`` means scoring path, triage path, and database
    # are all available. Webhook is not part of the status because it
    # is a notification side channel.
    status = "ok" if (model_loaded and database_healthy and llm_configured) else "degraded"

    return HealthResponse(
        status=status,
        service_version=__version__,
        model_loaded=model_loaded,
        model_version=model_version,
        model_schema_version=model_schema_version,
        decision_threshold=decision_threshold,
        database_healthy=database_healthy,
        llm_configured=llm_configured,
        langfuse_configured=langfuse_configured,
        webhook_enabled=settings.webhook_enabled,
    )
