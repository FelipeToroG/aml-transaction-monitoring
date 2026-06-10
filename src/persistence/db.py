"""Database engine and session management.

This module owns the SQLAlchemy engine, session factory, and the
declarative ``Base`` that every model class inherits from. It is the
only place in the codebase that knows the database URL: every other
module obtains a session via the dependency-injected ``get_db``
generator (see ``src.api.dependencies``).

URL portability
---------------
The engine is constructed from the ``DATABASE_URL`` environment
variable with a SQLite fallback. The same model definitions and
repository code work against SQLite (local development, tests,
investigator-laptop deployments) and PostgreSQL (production fintech
deployments) - the URL is the only thing that changes.

Schema migration
----------------
For local development and the demo deployment, :func:`init_db` calls
``Base.metadata.create_all`` to materialise the schema from the model
classes. Production deployments should use Alembic migrations instead
so schema evolution is reviewable and rollback-safe; the model
definitions in :mod:`src.persistence.models` are already compatible
with Alembic autogeneration.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# Default SQLite URL is relative to the project root. The ``check_same_thread``
# argument is required for SQLite when sessions cross threads (which
# happens under FastAPI's threadpool); the broader cost is that SQLite
# enforces only optimistic concurrency, which is acceptable at the
# alert-volume scale this service handles.
_DEFAULT_DATABASE_URL: Final[str] = "sqlite:///./aml_alerts.db"


class Base(DeclarativeBase):
    """Declarative base class for every persistence model.

    Inheriting from ``DeclarativeBase`` (SQLAlchemy 2.x style) rather
    than the older ``declarative_base()`` factory gives proper type
    annotations on :class:`~sqlalchemy.orm.Mapped` columns and clean
    type-checker support for the repository pattern.
    """


def build_engine(database_url: str | None = None) -> Engine:
    """Construct a SQLAlchemy engine from the resolved database URL.

    Parameters
    ----------
    database_url : str | None
        Explicit URL override. When ``None`` (the default), the URL is
        read from the ``DATABASE_URL`` environment variable, falling
        back to the in-project SQLite default.

    Returns
    -------
    Engine
        A configured engine with sensible defaults for both SQLite and
        Postgres. Connection pre-ping is enabled so stale connections
        in a long-running container are detected and recycled rather
        than producing confusing errors on the first reuse after an
        idle period.
    """
    url = database_url or os.getenv("DATABASE_URL", _DEFAULT_DATABASE_URL)

    # SQLite-specific connect args. Postgres ignores the ``connect_args``
    # dict when its keys are not Postgres-relevant, so the same code
    # path serves both backends.
    connect_args: dict[str, object] = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    return create_engine(
        url,
        connect_args=connect_args,
        pool_pre_ping=True,
        # ``echo=False`` in production. Operators flip this to True
        # temporarily when debugging a query; never wire it to a
        # config flag, because echoed SQL inevitably leaks
        # transaction identifiers into logs.
        echo=False,
        future=True,
    )


# Module-level engine and session factory. Constructed lazily at first
# access via the helpers below so test code can swap the engine before
# any code path materialises a session.
_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Return the process-wide engine, building it on first access."""
    global _engine
    if _engine is None:
        _engine = build_engine()
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide session factory, building it on first access."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
    return _SessionFactory


def init_db() -> None:
    """Materialise the schema from the declared model classes.

    Idempotent: SQLAlchemy's ``create_all`` is a no-op against tables
    that already exist with matching shape. Called once at API startup
    so a fresh deployment against a new database has its schema before
    the first request.

    For production deployments that need versioned migrations, replace
    this call with an Alembic upgrade in the deployment workflow.
    """
    # Import models here so the metadata is fully populated before
    # create_all runs. The import is local rather than top-level to
    # break what would otherwise be a circular import between
    # db.py → models.py → db.py (models import Base from here).
    from src.persistence import models  # noqa: F401 - import for side effect

    Base.metadata.create_all(bind=get_engine())


@contextmanager
def session_scope() -> Iterator[Session]:
    """Yield a transactional session for synchronous code paths.

    Use in scripts, notebooks, and any non-FastAPI code that needs a
    transactional context. FastAPI routes should depend on
    :func:`src.api.dependencies.get_db` instead, which integrates with
    the request lifecycle.

    Examples
    --------
    >>> with session_scope() as session:
    ...     alert_repo = AlertRepository(session)
    ...     alert_repo.create_alert(payload)
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


def reset_engine_for_testing() -> None:
    """Tear down the module-level engine; intended for the test suite.

    The test suite constructs a fresh in-memory SQLite engine for each
    test module and calls this function in a fixture teardown so the
    next module starts from a clean slate. Production code never calls
    this.
    """
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None
