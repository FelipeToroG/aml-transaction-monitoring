"""Shared pytest fixtures.

Three classes of fixtures live here:

1. **Data fixtures** — small synthetic transaction frames that
   exercise the loader, splitter, and feature pipelines without
   requiring the 5M-row IBM AML download.
2. **Database fixtures** — an in-memory SQLite database with the
   schema materialised, scoped to a single test for isolation.
3. **API fixtures** — a FastAPI TestClient with the model, narrator,
   and database dependencies overridden with stubs.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from src.data.loader import (
    AMOUNT_PAID_COLUMN,
    AMOUNT_RECEIVED_COLUMN,
    DEST_ACCOUNT_COLUMN,
    DEST_BANK_COLUMN,
    LABEL_COLUMN,
    PAYMENT_CURRENCY_COLUMN,
    PAYMENT_FORMAT_COLUMN,
    RECEIVING_CURRENCY_COLUMN,
    SOURCE_ACCOUNT_COLUMN,
    SOURCE_BANK_COLUMN,
    TIMESTAMP_COLUMN,
)
from src.persistence.db import Base, build_engine


@pytest.fixture
def synthetic_transactions() -> pd.DataFrame:
    """Return a small synthetic transaction frame matching the canonical schema."""
    base_time = pd.Timestamp("2026-01-01T00:00:00Z")
    rows: list[dict[str, Any]] = []
    for i in range(200):
        rows.append(
            {
                TIMESTAMP_COLUMN: base_time + pd.Timedelta(minutes=i),
                SOURCE_BANK_COLUMN: 10 + (i % 5),
                SOURCE_ACCOUNT_COLUMN: f"SRC{i % 30:04d}",
                DEST_BANK_COLUMN: 20 + (i % 7),
                DEST_ACCOUNT_COLUMN: f"DST{i % 40:04d}",
                AMOUNT_RECEIVED_COLUMN: 100.0 + i * 12.5,
                RECEIVING_CURRENCY_COLUMN: "USD",
                AMOUNT_PAID_COLUMN: 100.0 + i * 12.5,
                PAYMENT_CURRENCY_COLUMN: "USD",
                PAYMENT_FORMAT_COLUMN: "Wire" if i % 3 == 0 else "ACH",
                LABEL_COLUMN: 1 if i % 50 == 0 else 0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def in_memory_db_session() -> Iterator[Session]:
    """Yield a session bound to a fresh in-memory SQLite database."""
    engine = build_engine(database_url="sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)

    from sqlalchemy.orm import sessionmaker

    Session_ = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    session = Session_()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def sample_evidence_bundle() -> dict[str, Any]:
    """Return a stub evidence-bundle dict matching what the narrator expects."""
    return {
        "alert_id": "test_alert_001",
        "risk_score": 0.92,
        "anomaly_score": 0.78,
        "supervised_score": 0.95,
        "tier": "tier_3_critical",
        "generated_at_iso": dt.datetime.now(dt.timezone.utc).isoformat(),
        "transaction": {
            "transaction_id": "txn_001",
            "amount_paid": 9850.00,
            "payment_format": "Wire",
        },
        "source_activity": {"recent_transactions": []},
        "destination_activity": {"recent_transactions": []},
        "triggered_features": [
            {
                "feature_name": "src_24h_sub_threshold_share",
                "observed_value": 0.78,
                "contribution_rank": 1,
            },
            {
                "feature_name": "src_24h_round_amount_share",
                "observed_value": 0.55,
                "contribution_rank": 2,
            },
        ],
    }
