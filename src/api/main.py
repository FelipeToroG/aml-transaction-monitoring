"""FastAPI application factory.

This module assembles the API surface. The ``create_app`` factory:

1. Constructs the FastAPI app with versioning and structured metadata.
2. Initialises the database schema (idempotent ``create_all`` for SQLite;
   production deployments replace this with Alembic migrations).
3. Wires the Prometheus instrumentation onto every route.
4. Registers the route modules in order.
5. Sets up the structured-logging middleware.

The default entry point is ``app = create_app()``, which is what
``uvicorn src.api.main:app`` references.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from src import __version__
from src.api.routes import alerts, feedback, health, score, triage
from src.persistence.db import init_db
from src.utils.config import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown lifecycle for the app.

    Runs once at process start (before the first request) and once at
    process stop. We use the lifecycle for one-time setup that cannot
    live inside route code: database schema materialisation,
    structured-logging configuration, and a startup-banner log line so
    the operator can confirm the configured environment.
    """
    settings = get_settings()
    _configure_logging(settings.log_level)
    logger.info(
        "AML monitoring API starting (service_version=%s environment=%s)",
        __version__,
        settings.environment,
    )

    # Materialise schema on startup. The call is idempotent against an
    # existing schema; a missing table is created, an existing table is
    # left untouched. For production with Alembic this becomes a no-op
    # because the migration step ran before the app started.
    try:
        init_db()
        logger.info("Database schema verified / created.")
    except Exception as exc:  # noqa: BLE001
        # A failure here is operator-actionable (likely bad
        # DATABASE_URL) but should not prevent the app from starting - 
        # the health endpoint surfaces the database_healthy=False
        # signal, which is the right place for the operator to see it.
        logger.error("Database initialisation failed: %s", exc)

    yield

    logger.info("AML monitoring API shutting down.")


def create_app() -> FastAPI:
    """Factory that constructs and configures the FastAPI application."""
    app = FastAPI(
        title="AML Transaction Monitoring API",
        description=(
            "Production AML monitoring service. Hybrid anomaly + supervised "
            "scoring, Claude-powered case triage, full investigator feedback "
            "loop, and live drift/fairness monitoring."
        ),
        version=__version__,
        lifespan=_lifespan,
        # Exposing the docs UI at /docs is the FastAPI default; we
        # explicitly opt in because some operators require the
        # OpenAPI surface to be documented in their service catalog.
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ----- Routes -----
    # Order is documentation order rather than execution order: health
    # is the first thing operators look at, scoring is the primary
    # workload, triage and feedback close the lifecycle, alerts
    # supports the UI.
    app.include_router(health.router)
    app.include_router(score.router)
    app.include_router(triage.router)
    app.include_router(feedback.router)
    app.include_router(alerts.router)

    # ----- Prometheus instrumentation -----
    # Wired after routes so the instrumentator can introspect the
    # registered routes and emit per-route latency histograms.
    Instrumentator(
        excluded_handlers=["/health", "/metrics"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    return app


def _configure_logging(level: str) -> None:
    """Configure root-logger format and level."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


# Module-level app instance for `uvicorn src.api.main:app`.
app: FastAPI = create_app()
