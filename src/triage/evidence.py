"""Batch evidence-bundle assembly with entity recent-activity lookup.

Two scoring paths assemble the narrator's evidence differently because they
have different information available:

* Single-transaction API (``src/api/routes/score.py``). Scores one
  transaction per request. At request time the service has no view of the
  entity's prior activity, so it emits empty ``recent_transactions`` and the
  narrator may refuse with ``no_baseline``. That cold-start behavior is
  deliberate and documented there; the feature-store cache that would supply
  per-entity history to the online path is on the roadmap.

* Batch monitoring (this module). The common AML pattern scores a batch of
  transactions together, so each alert's entity history is already present in
  the batch frame. Loading it into ``recent_transactions`` lets the narrator
  reason over the entity's real recent activity instead of refusing for
  ``no_baseline`` on evidence that exists but was never assembled into the
  bundle.

The snapshot this module produces is shape-identical to the single-transaction
path's snapshot (same keys), so it flows through the same persistence, audit,
and ``EvidenceBundle`` conversion unchanged. The only difference is that the
``recent_transactions`` lists are populated.
"""

from __future__ import annotations

from typing import Any

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

# Default trailing window for "recent" activity. The narrator's prompt frames
# source/destination activity as a trailing-24h view, so the batch builder
# defaults to the same window; callers can widen it for sparse-history
# entities where a 24h slice would be uninformative.
DEFAULT_WINDOW_HOURS = 24
DEFAULT_RECENT_LIMIT = 8


def _row_id(row: pd.Series, id_column: str | None) -> str:
    """Resolve a citeable transaction id for a frame row.

    Production callers pass the source system's transaction-id column. When
    none is supplied the index label is used, which keeps offline/batch runs
    (where the IBM corpus has no native transaction id) citeable without
    forcing a synthetic id column upstream.
    """
    return str(row[id_column]) if id_column is not None else str(row.name)


def build_recent_activity(
    history: pd.DataFrame | None,
    *,
    account: str,
    before_timestamp: pd.Timestamp,
    role: str,
    id_column: str | None = None,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    limit: int = DEFAULT_RECENT_LIMIT,
) -> list[dict[str, Any]]:
    """Return an entity's most-recent prior transactions as citeable records.

    Parameters
    ----------
    history : DataFrame | None
        Prior transactions available to the batch (canonical schema). ``None``
        or empty yields ``[]`` so the caller degrades to cold-start behavior.
    account : str
        The entity whose activity to pull.
    before_timestamp : Timestamp
        Strict upper bound; only transactions earlier than the alert are
        included (no look-ahead, mirroring the zero-leakage feature rule).
    role : {"outbound", "inbound"}
        ``outbound`` matches the account on the source leg, ``inbound`` on the
        destination leg. Determines which column is filtered and which leg is
        reported as the counterparty.
    id_column : str | None
        Column holding the transaction id. ``None`` uses the row index label.
    window_hours, limit : int
        Trailing window and max records (most-recent first).

    Returns
    -------
    list[dict]
        Each record carries ``transaction_id`` (so the narrator can cite it),
        plus timestamp, amount, payment format, direction, and counterparty.
    """
    if history is None or len(history) == 0:
        return []
    if role not in ("outbound", "inbound"):
        raise ValueError(f"role must be 'outbound' or 'inbound', got {role!r}")

    match_col = SOURCE_ACCOUNT_COLUMN if role == "outbound" else DEST_ACCOUNT_COLUMN
    counterparty_col = DEST_ACCOUNT_COLUMN if role == "outbound" else SOURCE_ACCOUNT_COLUMN
    window_start = before_timestamp - pd.Timedelta(hours=window_hours)

    mask = (
        (history[match_col] == account)
        & (history[TIMESTAMP_COLUMN] < before_timestamp)
        & (history[TIMESTAMP_COLUMN] >= window_start)
    )
    recent = history.loc[mask].sort_values(TIMESTAMP_COLUMN, ascending=False).head(limit)

    records: list[dict[str, Any]] = []
    for _, h in recent.iterrows():
        ts = h[TIMESTAMP_COLUMN]
        records.append(
            {
                "transaction_id": _row_id(h, id_column),
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "amount_paid": float(h[AMOUNT_PAID_COLUMN]),
                "payment_format": str(h[PAYMENT_FORMAT_COLUMN]),
                "direction": role,
                "counterparty": str(h[counterparty_col]),
            }
        )
    return records


def build_batch_evidence_snapshot(
    *,
    row: pd.Series,
    history: pd.DataFrame | None,
    risk_score: float,
    anomaly_score: float,
    supervised_score: float,
    tier: str,
    triggered_features: list[dict[str, Any]],
    id_column: str | None = None,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    limit: int = DEFAULT_RECENT_LIMIT,
) -> dict[str, Any]:
    """Assemble a narrator evidence snapshot for one batch alert.

    Shape-identical to ``src/api/routes/score.py::_build_evidence_snapshot``
    so it flows through the same persistence/audit path and
    ``_evidence_bundle_from_snapshot`` conversion. The difference is that the
    ``source_activity`` / ``destination_activity`` ``recent_transactions``
    lists are populated from ``history`` rather than left empty.

    ``triggered_features`` is supplied by the caller (the batch scoring loop
    computes top contributors the same way the online path does) so this
    module stays decoupled from the feature-engineering layer.
    """
    src_account = str(row[SOURCE_ACCOUNT_COLUMN])
    dst_account = str(row[DEST_ACCOUNT_COLUMN])
    ts = row[TIMESTAMP_COLUMN]

    return {
        "alert_id_placeholder": "set_on_persist",
        "transaction": {
            "transaction_id": _row_id(row, id_column),
            "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "source_bank": str(row[SOURCE_BANK_COLUMN]),
            "source_account": src_account,
            "dest_bank": str(row[DEST_BANK_COLUMN]),
            "dest_account": dst_account,
            "amount_paid": float(row[AMOUNT_PAID_COLUMN]),
            "amount_received": float(row[AMOUNT_RECEIVED_COLUMN]),
            "payment_currency": str(row[PAYMENT_CURRENCY_COLUMN]),
            "receiving_currency": str(row[RECEIVING_CURRENCY_COLUMN]),
            "payment_format": str(row[PAYMENT_FORMAT_COLUMN]),
        },
        "scoring": {
            "risk_score": risk_score,
            "anomaly_score": anomaly_score,
            "supervised_score": supervised_score,
            "tier": tier,
        },
        "triggered_features": triggered_features,
        "source_activity": {
            "recent_transactions": build_recent_activity(
                history,
                account=src_account,
                before_timestamp=ts,
                role="outbound",
                id_column=id_column,
                window_hours=window_hours,
                limit=limit,
            )
        },
        "destination_activity": {
            "recent_transactions": build_recent_activity(
                history,
                account=dst_account,
                before_timestamp=ts,
                role="inbound",
                id_column=id_column,
                window_hours=window_hours,
                limit=limit,
            )
        },
    }
