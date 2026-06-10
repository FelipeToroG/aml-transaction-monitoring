"""Transaction scoring endpoint.

``POST /score`` ingests a single transaction, runs it through the
hybrid ensemble, and either creates and persists an alert (if the
score crosses the threshold) or returns the score with no
alert-creation side effect (if it does not). Triage is dispatched
synchronously for tier-3 critical alerts and as a background task for
tier-2 high alerts.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.dependencies import (
    get_alert_thresholds,
    get_db,
    get_ensemble,
    get_narrator,
    get_webhook_client,
)
from src.api.schemas import (
    TopFeatureContribution,
    TransactionScoreRequest,
    TransactionScoreResponse,
)
from src.api.webhook import WebhookClient
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
from src.models.ensemble import AMLEnsemble
from src.observability.metrics import record_alert_created, record_score
from src.persistence.models import AlertTier, AuditEventType
from src.persistence.repository import AlertRepository, AuditLogRepository
from src.triage.narrator import EvidenceBundle, Narrator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["scoring"])


@router.post(
    "/score",
    response_model=TransactionScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Score a single transaction and optionally create an alert.",
)
def score_transaction(
    request: TransactionScoreRequest,
    background_tasks: BackgroundTasks,
    ensemble: Annotated[AMLEnsemble, Depends(get_ensemble)],
    narrator: Annotated[Narrator, Depends(get_narrator)],
    webhook: Annotated[WebhookClient, Depends(get_webhook_client)],
    alert_thresholds: Annotated[dict[str, Any], Depends(get_alert_thresholds)],
    session: Annotated[Session, Depends(get_db)],
) -> TransactionScoreResponse:
    """Score one transaction.

    Hot path properties:

    * **Synchronous scoring**: the ensemble's ``score()`` call runs
      inline. Designed for sub-150 ms p99.
    * **Conditional triage**: tier-3 critical alerts run the LLM
      synchronously so the response includes the narrative. Tier-2
      alerts dispatch the LLM as a FastAPI background task so the
      response returns immediately. Tier-1 alerts defer triage to
      investigator pickup.
    * **Webhook fire-and-forget**: tier-3 alerts schedule a webhook
      dispatch on the background-task system; the response does not
      block on webhook delivery.
    """
    start = time.perf_counter()

    # Build a single-row DataFrame matching the loader's canonical
    # schema. The feature engineering layer expects column names as
    # emitted by the source CSV, which differ from the API field
    # names; the mapping is explicit here so any schema rename
    # surfaces at this boundary, not deep inside the model code.
    frame = _build_scoring_frame(request)

    # ----- Ensemble scoring -----
    try:
        risk_scores, anomaly_scores = ensemble.score_components(frame)
    except Exception as exc:
        logger.exception("Ensemble scoring failed for transaction %s", request.transaction_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring engine error: {exc}",
        ) from exc

    risk_score = float(risk_scores[0])
    # The anomaly score comes from the same transform that produced the risk
    # score (the trailing stacking-feature column), so the component shown to
    # investigators is exactly the value the model consumed.
    anomaly_score = float(anomaly_scores[0])
    # Under stacking the supervised model IS the final model: it consumes the
    # anomaly score as a feature and emits the risk score directly, so the
    # supervised component equals the risk score. Retained for response / DB
    # compatibility.
    # TODO: could be renamed to stacked_model_score or dropped once the
    # response/persistence schema is revised.
    supervised_score = risk_score

    # The engineered subset is still needed for the top-feature breakdown
    # below (raw feature magnitudes), not for re-scoring.
    feature_columns = list(ensemble.feature_columns)
    feature_subset_engineered = _engineered_feature_subset(
        frame=frame, ensemble=ensemble, feature_columns=feature_columns
    )

    # ----- Tier assignment -----
    tier_name = _assign_tier(risk_score, alert_thresholds)
    tier = AlertTier(tier_name)

    # Observe the score distribution by tier. This metric is the
    # foundation of the operator dashboard's "is the model drifting?"
    # panel - a shift in the per-tier score distribution is the first
    # visible sign of model or upstream-data drift.
    record_score(score=risk_score, tier=tier_name)

    # Default threshold used as the alert cutoff. The ensemble's
    # configured threshold is the boundary between "alert" and
    # "no alert"; the tier breakdown is finer-grained.
    threshold_applied = ensemble.decision_threshold
    alert_should_be_created = risk_score >= threshold_applied and tier != AlertTier.SUPPRESSED

    inference_latency_ms = (time.perf_counter() - start) * 1000.0

    # ----- Top contributing features -----
    top_features = _build_top_features(
        engineered_frame=feature_subset_engineered,
        ensemble=ensemble,
    )

    # ----- Alert persistence -----
    alert_id: str | None = None
    if alert_should_be_created:
        alert_repo = AlertRepository(session)
        audit_repo = AuditLogRepository(session)

        evidence_snapshot = _build_evidence_snapshot(
            request=request,
            risk_score=risk_score,
            anomaly_score=anomaly_score,
            supervised_score=supervised_score,
            tier=tier_name,
            top_features=top_features,
        )

        alert = alert_repo.create_alert(
            transaction_id=request.transaction_id,
            risk_score=risk_score,
            anomaly_score=anomaly_score,
            supervised_score=supervised_score,
            threshold_applied=threshold_applied,
            tier=tier,
            evidence_snapshot=evidence_snapshot,
            model_version=ensemble.metadata.service_version if ensemble.metadata else "unknown",
            model_schema_version=ensemble.metadata.schema_version if ensemble.metadata else 0,
        )
        alert_id = alert.alert_id

        # Increment the per-tier alert-volume counter. This drives the
        # operator dashboard's headline "alerts per hour by tier" chart.
        record_alert_created(tier=tier_name)

        audit_repo.write_event(
            event_type=AuditEventType.ALERT_CREATED,
            alert_id=alert.alert_id,
            event_data={
                "transaction_id": request.transaction_id,
                "tier": tier_name,
                "risk_score": risk_score,
                "anomaly_score": anomaly_score,
                "supervised_score": supervised_score,
            },
            model_version=alert.model_version,
        )

        # ----- Tier-specific downstream actions -----
        tier_config = _find_tier_config(tier_name, alert_thresholds)

        if tier_config and tier_config.get("triage_synchronously", False):
            _run_triage_inline(
                alert_id=alert.alert_id,
                evidence_snapshot=evidence_snapshot,
                narrator=narrator,
                session=session,
            )
        elif tier_config:
            background_tasks.add_task(
                _run_triage_background,
                alert_id=alert.alert_id,
                evidence_snapshot=evidence_snapshot,
            )

        if tier == AlertTier.TIER_3_CRITICAL:
            background_tasks.add_task(
                _dispatch_webhook,
                webhook=webhook,
                alert_id=alert.alert_id,
                transaction_id=alert.transaction_id,
                tier=tier_name,
                risk_score=risk_score,
                narrative_summary=None,  # narrative may be populated by inline triage above
                suspected_typology=None,
            )

    return TransactionScoreResponse(
        transaction_id=request.transaction_id,
        risk_score=risk_score,
        anomaly_score=anomaly_score,
        supervised_score=supervised_score,
        tier=tier,
        alert_created=alert_should_be_created,
        alert_id=alert_id,
        threshold_applied=threshold_applied,
        model_version=ensemble.metadata.service_version if ensemble.metadata else "unknown",
        model_schema_version=ensemble.metadata.schema_version if ensemble.metadata else 0,
        inference_latency_ms=inference_latency_ms,
        top_features=top_features,
    )


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _build_scoring_frame(request: TransactionScoreRequest) -> pd.DataFrame:
    """Construct the single-row DataFrame the ensemble expects.

    Maps the API's underscore-cased field names to the canonical
    CSV column names defined in :mod:`src.data.loader`. The label
    column is included with a sentinel so the downstream feature
    engineering does not complain about its absence.
    """
    return pd.DataFrame(
        [
            {
                TIMESTAMP_COLUMN: pd.Timestamp(request.timestamp),
                SOURCE_BANK_COLUMN: request.source_bank,
                SOURCE_ACCOUNT_COLUMN: request.source_account,
                DEST_BANK_COLUMN: request.dest_bank,
                DEST_ACCOUNT_COLUMN: request.dest_account,
                AMOUNT_RECEIVED_COLUMN: request.amount_received,
                RECEIVING_CURRENCY_COLUMN: request.receiving_currency,
                AMOUNT_PAID_COLUMN: request.amount_paid,
                PAYMENT_CURRENCY_COLUMN: request.payment_currency,
                PAYMENT_FORMAT_COLUMN: request.payment_format,
                LABEL_COLUMN: 0,  # sentinel; not used by inference
            }
        ]
    )


def _engineered_feature_subset(
    *,
    frame: pd.DataFrame,
    ensemble: AMLEnsemble,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Engineer features and return the column subset the pipeline consumes."""
    from src.features.pipelines import build_engineered_frame

    bundle = build_engineered_frame(frame)
    return bundle.frame[feature_columns]


def _assign_tier(score: float, thresholds_config: dict[str, Any]) -> str:
    """Return the tier name corresponding to a raw risk score."""
    for tier in thresholds_config["tiers"]:
        if tier["min_score"] <= score < tier["max_score"]:
            return str(tier["name"])
    # Score exactly at 1.0 falls through the half-open ranges; the
    # top tier owns it.
    return str(thresholds_config["tiers"][0]["name"])


def _find_tier_config(
    tier_name: str, thresholds_config: dict[str, Any]
) -> dict[str, Any] | None:
    for tier in thresholds_config["tiers"]:
        if tier["name"] == tier_name:
            return tier
    return None


def _build_top_features(
    *,
    engineered_frame: pd.DataFrame,
    ensemble: AMLEnsemble,
    top_n: int = 5,
) -> list[TopFeatureContribution]:
    """Return the top contributing numerical features by magnitude.

    For models with native feature_importances_ we would use those;
    here we surface the largest-magnitude raw feature values from the
    engineered frame as a transparent proxy for "what's unusual about
    this transaction". A future enhancement plugs in SHAP values per
    request, which the README's roadmap calls out.
    """
    numeric_only = engineered_frame.select_dtypes(include="number")
    if numeric_only.empty:
        return []

    row = numeric_only.iloc[0].abs().sort_values(ascending=False)
    top = row.head(top_n)
    return [
        TopFeatureContribution(
            feature_name=str(name),
            observed_value=float(engineered_frame[name].iloc[0]),
            contribution_rank=rank,
        )
        for rank, (name, _) in enumerate(top.items(), start=1)
    ]


def _build_evidence_snapshot(
    *,
    request: TransactionScoreRequest,
    risk_score: float,
    anomaly_score: float,
    supervised_score: float,
    tier: str,
    top_features: list[TopFeatureContribution],
) -> dict[str, Any]:
    """Assemble the JSON evidence snapshot persisted with the alert.

    The snapshot is what the narrator reads at triage time and what
    the audit-trace endpoint returns when a regulator queries an
    historical alert. Structure must therefore be stable across
    deployments - schema changes require a model_schema_version bump.
    """
    return {
        "alert_id_placeholder": "set_on_persist",
        "transaction": {
            "transaction_id": request.transaction_id,
            "timestamp": request.timestamp.isoformat(),
            "source_bank": request.source_bank,
            "source_account": request.source_account,
            "dest_bank": request.dest_bank,
            "dest_account": request.dest_account,
            "amount_paid": request.amount_paid,
            "amount_received": request.amount_received,
            "payment_currency": request.payment_currency,
            "receiving_currency": request.receiving_currency,
            "payment_format": request.payment_format,
        },
        "scoring": {
            "risk_score": risk_score,
            "anomaly_score": anomaly_score,
            "supervised_score": supervised_score,
            "tier": tier,
        },
        "triggered_features": [
            {
                "feature_name": f.feature_name,
                "observed_value": f.observed_value,
                "contribution_rank": f.contribution_rank,
            }
            for f in top_features
        ],
        "source_activity": {"recent_transactions": []},
        "destination_activity": {"recent_transactions": []},
    }


def _run_triage_inline(
    *,
    alert_id: str,
    evidence_snapshot: dict[str, Any],
    narrator: Narrator,
    session: Session,
) -> None:
    """Run triage synchronously inside the scoring request."""
    bundle = _evidence_bundle_from_snapshot(alert_id=alert_id, snapshot=evidence_snapshot)
    result = narrator.generate(bundle)
    _persist_narrative_result(
        alert_id=alert_id, result=result, session=session
    )


def _run_triage_background(*, alert_id: str, evidence_snapshot: dict[str, Any]) -> None:
    """Background task wrapper around inline triage.

    The background task obtains its own session and narrator
    rather than capturing the request-scoped ones, because the
    request-scoped resources are torn down when the response returns.
    """
    from src.api.dependencies import get_narrator
    from src.persistence.db import session_scope

    bundle = _evidence_bundle_from_snapshot(alert_id=alert_id, snapshot=evidence_snapshot)
    narrator = get_narrator()
    result = narrator.generate(bundle)
    with session_scope() as bg_session:
        _persist_narrative_result(alert_id=alert_id, result=result, session=bg_session)


async def _dispatch_webhook(
    *,
    webhook: WebhookClient,
    alert_id: str,
    transaction_id: str,
    tier: str,
    risk_score: float,
    narrative_summary: str | None,
    suspected_typology: str | None,
) -> None:
    """Background task that dispatches the alert notification webhook."""
    await webhook.dispatch_alert_notification(
        alert_id=alert_id,
        transaction_id=transaction_id,
        tier=tier,
        risk_score=risk_score,
        narrative_summary=narrative_summary,
        suspected_typology=suspected_typology,
    )


def _evidence_bundle_from_snapshot(
    *, alert_id: str, snapshot: dict[str, Any]
) -> EvidenceBundle:
    """Convert a persisted evidence snapshot into a Narrator EvidenceBundle."""
    import datetime as dt

    scoring = snapshot.get("scoring", {})
    return EvidenceBundle(
        alert_id=alert_id,
        risk_score=float(scoring.get("risk_score", 0.0)),
        anomaly_score=float(scoring.get("anomaly_score", 0.0)),
        supervised_score=float(scoring.get("supervised_score", 0.0)),
        tier=str(scoring.get("tier", "unknown")),
        generated_at_iso=dt.datetime.now(dt.timezone.utc).isoformat(),
        transaction=snapshot.get("transaction", {}),
        source_activity=snapshot.get("source_activity", {}),
        destination_activity=snapshot.get("destination_activity", {}),
        triggered_features=snapshot.get("triggered_features", []),
    )


def _persist_narrative_result(
    *,
    alert_id: str,
    result: Any,  # NarratorResult
    session: Session,
) -> None:
    """Persist a NarratorResult to the alert and audit log."""
    alert_repo = AlertRepository(session)
    audit_repo = AuditLogRepository(session)

    if result.success and result.narrative is not None:
        narrative_payload = result.narrative.model_dump()
        confidence = result.narrative.confidence
    elif result.refusal is not None:
        narrative_payload = {"refusal": result.refusal.model_dump()}
        confidence = None
    else:
        narrative_payload = {"error": "Empty narrator result"}
        confidence = None

    alert_repo.set_narrative(
        alert_id=alert_id,
        narrative_payload=narrative_payload,
        narrative_confidence=confidence,
        narrator_model=result.model_name,
        narrator_prompt_version=result.prompt_version,
    )
    audit_repo.write_event(
        event_type=AuditEventType.ALERT_TRIAGED,
        alert_id=alert_id,
        event_data={
            "success": result.success,
            "model": result.model_name,
            "prompt_version": result.prompt_version,
            "latency_ms": result.latency_ms,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        },
    )
