"""Tests for the batch evidence-bundle assembly (src/triage/evidence.py).

The batch path's reason for existing is that it populates recent_transactions
from in-batch entity history, where the single-transaction API path leaves it
empty. These tests pin that behavior: the window bound, the no-look-ahead
bound, the recency ordering, the role filter, the cold-start empty case, and
the snapshot shape parity with the online path.
"""

from __future__ import annotations

import pandas as pd

from src.data.loader import (
    AMOUNT_PAID_COLUMN,
    AMOUNT_RECEIVED_COLUMN,
    DEST_ACCOUNT_COLUMN,
    DEST_BANK_COLUMN,
    PAYMENT_CURRENCY_COLUMN,
    PAYMENT_FORMAT_COLUMN,
    RECEIVING_CURRENCY_COLUMN,
    SOURCE_ACCOUNT_COLUMN,
    SOURCE_BANK_COLUMN,
    TIMESTAMP_COLUMN,
)
from src.triage.evidence import build_batch_evidence_snapshot, build_recent_activity


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal canonical-schema frame from partial row dicts."""
    defaults = {
        SOURCE_BANK_COLUMN: "10",
        DEST_BANK_COLUMN: "20",
        AMOUNT_PAID_COLUMN: 100.0,
        AMOUNT_RECEIVED_COLUMN: 100.0,
        PAYMENT_CURRENCY_COLUMN: "US Dollar",
        RECEIVING_CURRENCY_COLUMN: "US Dollar",
        PAYMENT_FORMAT_COLUMN: "ACH",
    }
    df = pd.DataFrame([{**defaults, **r} for r in rows])
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
    return df


def test_recent_activity_window_ordering_and_lookahead():
    # Entity ACC1 sends at -1h, -5h, -50h (outside 24h), and +1h (future).
    alert_ts = pd.Timestamp("2022-09-10 12:00:00")
    history = _frame(
        [
            {SOURCE_ACCOUNT_COLUMN: "ACC1", DEST_ACCOUNT_COLUMN: "X", TIMESTAMP_COLUMN: "2022-09-10 11:00:00", AMOUNT_PAID_COLUMN: 11.0},
            {SOURCE_ACCOUNT_COLUMN: "ACC1", DEST_ACCOUNT_COLUMN: "Y", TIMESTAMP_COLUMN: "2022-09-10 07:00:00", AMOUNT_PAID_COLUMN: 7.0},
            {SOURCE_ACCOUNT_COLUMN: "ACC1", DEST_ACCOUNT_COLUMN: "Z", TIMESTAMP_COLUMN: "2022-09-08 10:00:00", AMOUNT_PAID_COLUMN: 99.0},
            {SOURCE_ACCOUNT_COLUMN: "ACC1", DEST_ACCOUNT_COLUMN: "F", TIMESTAMP_COLUMN: "2022-09-10 13:00:00", AMOUNT_PAID_COLUMN: 13.0},
        ]
    )
    recent = build_recent_activity(
        history, account="ACC1", before_timestamp=alert_ts, role="outbound"
    )
    # The -50h row (outside window) and the future row (look-ahead) are excluded.
    assert [r["amount_paid"] for r in recent] == [11.0, 7.0]  # most-recent first
    assert all(r["direction"] == "outbound" for r in recent)
    assert recent[0]["counterparty"] == "X"


def test_recent_activity_role_filter_and_limit():
    alert_ts = pd.Timestamp("2022-09-10 12:00:00")
    rows = [
        {SOURCE_ACCOUNT_COLUMN: "OTHER", DEST_ACCOUNT_COLUMN: "ACC2", TIMESTAMP_COLUMN: f"2022-09-10 {h:02d}:00:00", AMOUNT_PAID_COLUMN: float(h)}
        for h in range(1, 12)
    ]
    history = _frame(rows)
    inbound = build_recent_activity(
        history, account="ACC2", before_timestamp=alert_ts, role="inbound", limit=8
    )
    assert len(inbound) == 8  # capped
    assert inbound[0]["amount_paid"] == 11.0  # most recent first
    assert all(r["direction"] == "inbound" and r["counterparty"] == "OTHER" for r in inbound)
    # An outbound query for the same account finds nothing (it only received).
    assert build_recent_activity(history, account="ACC2", before_timestamp=alert_ts, role="outbound") == []


def test_recent_activity_cold_start_returns_empty():
    assert build_recent_activity(None, account="A", before_timestamp=pd.Timestamp("2022-09-10"), role="outbound") == []
    empty = _frame([{SOURCE_ACCOUNT_COLUMN: "A", DEST_ACCOUNT_COLUMN: "B", TIMESTAMP_COLUMN: "2022-09-10 10:00:00"}]).iloc[0:0]
    assert build_recent_activity(empty, account="A", before_timestamp=pd.Timestamp("2022-09-10"), role="outbound") == []


def test_batch_snapshot_shape_matches_online_path_and_populates_history():
    history = _frame(
        [
            {SOURCE_ACCOUNT_COLUMN: "SRC", DEST_ACCOUNT_COLUMN: "CP", TIMESTAMP_COLUMN: "2022-09-10 11:30:00", AMOUNT_PAID_COLUMN: 500.0},
            {SOURCE_ACCOUNT_COLUMN: "PREV", DEST_ACCOUNT_COLUMN: "DST", TIMESTAMP_COLUMN: "2022-09-10 11:45:00", AMOUNT_PAID_COLUMN: 250.0},
        ]
    )
    alert = _frame(
        [{SOURCE_ACCOUNT_COLUMN: "SRC", DEST_ACCOUNT_COLUMN: "DST", TIMESTAMP_COLUMN: "2022-09-10 12:00:00", AMOUNT_PAID_COLUMN: 9000.0}]
    ).iloc[0]

    snap = build_batch_evidence_snapshot(
        row=alert,
        history=history,
        risk_score=0.91,
        anomaly_score=0.42,
        supervised_score=0.91,
        tier="tier_3_critical",
        triggered_features=[{"feature_name": "src_24h_amount_sum", "observed_value": 9500.0, "contribution_rank": 1}],
    )

    # Same top-level keys as the single-transaction snapshot.
    assert set(snap) == {
        "alert_id_placeholder",
        "transaction",
        "scoring",
        "triggered_features",
        "source_activity",
        "destination_activity",
    }
    # History is populated (the whole point of the batch path).
    assert len(snap["source_activity"]["recent_transactions"]) == 1
    assert snap["source_activity"]["recent_transactions"][0]["counterparty"] == "CP"
    assert len(snap["destination_activity"]["recent_transactions"]) == 1
    assert snap["destination_activity"]["recent_transactions"][0]["direction"] == "inbound"
    assert snap["scoring"]["tier"] == "tier_3_critical"
