"""SQLite persistence for alerts, investigator feedback, and audit log.

SQLAlchemy 2.x is used so the same code targets SQLite locally and
Postgres in production with no code changes - only the connection URL
differs. The repository pattern keeps queries out of API handlers, which
makes the persistence layer testable in isolation against an in-memory
SQLite database.

Public surface
--------------
The API and the test suite import from here. The exported symbols are
the model classes, the repository classes, the enums for status / tier
/ disposition / event types, and the session-management helpers.
"""

from src.persistence.db import (
    Base,
    build_engine,
    get_engine,
    get_session_factory,
    init_db,
    reset_engine_for_testing,
    session_scope,
)
from src.persistence.models import (
    Alert,
    AlertStatus,
    AlertTier,
    AuditEventType,
    AuditLog,
    Feedback,
    FeedbackDisposition,
)
from src.persistence.repository import (
    AlertRepository,
    AuditLogRepository,
    FeedbackRepository,
)

__all__ = [
    "Alert",
    "AlertRepository",
    "AlertStatus",
    "AlertTier",
    "AuditEventType",
    "AuditLog",
    "AuditLogRepository",
    "Base",
    "Feedback",
    "FeedbackDisposition",
    "FeedbackRepository",
    "build_engine",
    "get_engine",
    "get_session_factory",
    "init_db",
    "reset_engine_for_testing",
    "session_scope",
]
