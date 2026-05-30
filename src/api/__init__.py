"""FastAPI application: the scoring, triage, feedback, and alert API.

The API is the only externally visible surface of the service. It owns
the HTTP contract, request validation, response serialisation, outbound
webhook dispatch, and the dependency wiring for the model, the LLM
client, and the database session.

Public surface
--------------
Most callers want the FastAPI ``app`` instance:

    from src.api.main import app

Other modules import the schemas, the dependencies, and the webhook
client directly.
"""

from src.api.dependencies import (
    get_alert_repository,
    get_alert_thresholds,
    get_audit_log_repository,
    get_db,
    get_ensemble,
    get_feedback_repository,
    get_narrator,
    get_webhook_client,
)
from src.api.main import app, create_app
from src.api.schemas import (
    AlertsListResponse,
    AlertSummary,
    APIError,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    TopFeatureContribution,
    TransactionScoreRequest,
    TransactionScoreResponse,
    TriageRequest,
    TriageResponse,
)
from src.api.webhook import WebhookClient, WebhookDeliveryResult

__all__ = [
    "APIError",
    "AlertSummary",
    "AlertsListResponse",
    "FeedbackRequest",
    "FeedbackResponse",
    "HealthResponse",
    "TopFeatureContribution",
    "TransactionScoreRequest",
    "TransactionScoreResponse",
    "TriageRequest",
    "TriageResponse",
    "WebhookClient",
    "WebhookDeliveryResult",
    "app",
    "create_app",
    "get_alert_repository",
    "get_alert_thresholds",
    "get_audit_log_repository",
    "get_db",
    "get_ensemble",
    "get_feedback_repository",
    "get_narrator",
    "get_webhook_client",
]
