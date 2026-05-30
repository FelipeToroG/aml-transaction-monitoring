"""FastAPI dependency-injected resources.

Every shared resource the API uses — the loaded model, the database
session factory, the Anthropic narrator, the webhook client — is
obtained through these dependency functions. Route handlers declare
what they need via ``Depends(...)``, FastAPI resolves and injects, and
the test suite can override any of them with one ``app.dependency_overrides``
assignment.

Why centralise here
-------------------
1. **Testability.** Replacing the live narrator with a stub for tests
   is a single line. Replacing the database with an in-memory SQLite
   is the same.
2. **Single construction point.** The model artifact is loaded once at
   startup, not per request. The narrator client maintains its
   connection pool. Centralising the lifecycle here keeps the
   resource lifetimes honest.
3. **No hidden globals in route code.** Route handlers do not import
   the model or the narrator directly. Every dependency is explicit
   in the signature, so the dependency graph is grep-able.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from src.models.ensemble import AMLEnsemble
from src.observability.tracing import build_langfuse_client
from src.persistence.db import get_session_factory
from src.persistence.repository import (
    AlertRepository,
    AuditLogRepository,
    FeedbackRepository,
)
from src.triage.narrator import Narrator
from src.utils.config import Settings, get_settings, load_yaml_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Database session
# ---------------------------------------------------------------------


def get_db() -> Iterator[Session]:
    """Yield a database session scoped to the request.

    Commits on successful return; rolls back on exception; always
    closes. FastAPI's dependency machinery invokes the generator and
    provides the yielded value as the route's parameter.
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------
# Repositories
# ---------------------------------------------------------------------


def get_alert_repository(session: Session) -> AlertRepository:
    """Construct an AlertRepository for the request session.

    Repository construction is trivial; we wrap it in a function so the
    test suite can override the dependency to inject a mock repository
    when needed.
    """
    return AlertRepository(session)


def get_feedback_repository(session: Session) -> FeedbackRepository:
    return FeedbackRepository(session)


def get_audit_log_repository(session: Session) -> AuditLogRepository:
    return AuditLogRepository(session)


# ---------------------------------------------------------------------
# Model ensemble — singleton loaded at startup
# ---------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_ensemble_cached() -> AMLEnsemble:
    """Load the model artifact from disk exactly once.

    The cache is process-wide because the AMLEnsemble is immutable
    after load. Test code calls ``_load_ensemble_cached.cache_clear()``
    before swapping in a test artifact.
    """
    settings = get_settings()
    model_path = Path(settings.model_path)
    logger.info("Loading model ensemble from %s", model_path)
    return AMLEnsemble.load(model_path)


def get_ensemble() -> AMLEnsemble:
    """Return the loaded ensemble. Loaded on first call and cached."""
    return _load_ensemble_cached()


# ---------------------------------------------------------------------
# Narrator — singleton constructed at startup
# ---------------------------------------------------------------------


@lru_cache(maxsize=1)
def _build_narrator_cached() -> Narrator:
    """Construct the Narrator exactly once.

    The configuration comes from ``configs/api_config.yaml`` (model
    names, max_tokens, temperature) plus the Anthropic key from
    settings. The Langfuse client is constructed here so it shares the
    narrator's lifecycle.
    """
    settings = get_settings()
    api_cfg = load_yaml_config("configs/api_config.yaml")
    triage_cfg = api_cfg["triage"]

    langfuse_client = _build_langfuse(settings)

    return Narrator(
        api_key=settings.anthropic_api_key,
        primary_model=triage_cfg["primary_model"],
        eval_model=triage_cfg["eval_model"],
        max_tokens=int(triage_cfg["max_tokens"]),
        temperature=float(triage_cfg["temperature"]),
        max_validation_retries=int(triage_cfg["max_validation_retries"]),
        langfuse=langfuse_client,
    )


def get_narrator() -> Narrator:
    """Return the Narrator. Constructed on first call and cached."""
    return _build_narrator_cached()


def _build_langfuse(settings: Settings) -> Any | None:
    """Construct the Langfuse client via the shared observability helper.

    Thin wrapper that adapts the env-var-based ``settings`` interface
    to the ``build_langfuse_client`` signature in
    :mod:`src.observability.tracing`. Keeping the construction logic
    in one place (the observability package) means future provider
    additions or migrations require changing one file, not two.
    """
    return build_langfuse_client(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


# ---------------------------------------------------------------------
# Webhook client
# ---------------------------------------------------------------------


@lru_cache(maxsize=1)
def _build_webhook_client_cached() -> "WebhookClient":
    """Construct the webhook client once per process."""
    # Local import to avoid a circular import: webhook imports from
    # persistence which imports from db which is unrelated to this
    # module, but the cycle exists in practice via shared utilities.
    from src.api.webhook import WebhookClient

    settings = get_settings()
    api_cfg = load_yaml_config("configs/api_config.yaml")
    webhook_cfg = api_cfg["webhook"]

    return WebhookClient(
        url=settings.webhook_url,
        enabled=settings.webhook_enabled,
        max_retries=int(webhook_cfg["max_retries"]),
        backoff_base_seconds=float(webhook_cfg["backoff_base_seconds"]),
        backoff_max_seconds=float(webhook_cfg["backoff_max_seconds"]),
        request_timeout_seconds=float(webhook_cfg["request_timeout_seconds"]),
    )


def get_webhook_client() -> "WebhookClient":
    """Return the webhook client. Constructed on first call and cached."""
    return _build_webhook_client_cached()


# ---------------------------------------------------------------------
# Alert-threshold configuration
# ---------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_alert_thresholds() -> dict[str, Any]:
    """Return the parsed alert-thresholds YAML.

    Loaded once and cached. Used by the scoring route to map a raw
    score to a tier and to decide whether triage runs synchronously.
    """
    return load_yaml_config("configs/alert_thresholds.yaml")
