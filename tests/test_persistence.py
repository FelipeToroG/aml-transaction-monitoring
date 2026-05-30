"""Tests for the persistence repositories."""

from __future__ import annotations

from src.persistence.models import (
    AlertStatus,
    AlertTier,
    AuditEventType,
    FeedbackDisposition,
)
from src.persistence.repository import (
    AlertRepository,
    AuditLogRepository,
    FeedbackRepository,
)


def _make_alert(repo: AlertRepository, **overrides):
    payload = dict(
        transaction_id="txn_001",
        risk_score=0.95,
        anomaly_score=0.80,
        supervised_score=0.97,
        threshold_applied=0.50,
        tier=AlertTier.TIER_3_CRITICAL,
        evidence_snapshot={"foo": "bar"},
        model_version="0.1.0",
        model_schema_version=1,
    )
    payload.update(overrides)
    return repo.create_alert(**payload)


def test_create_and_retrieve_alert(in_memory_db_session):
    repo = AlertRepository(in_memory_db_session)
    created = _make_alert(repo)
    retrieved = repo.get_alert(alert_id=created.alert_id)
    assert retrieved is not None
    assert retrieved.transaction_id == "txn_001"
    assert retrieved.status == AlertStatus.OPEN


def test_list_alerts_filters_by_status_and_tier(in_memory_db_session):
    repo = AlertRepository(in_memory_db_session)
    _make_alert(repo, tier=AlertTier.TIER_3_CRITICAL)
    _make_alert(repo, tier=AlertTier.TIER_2_HIGH)
    _make_alert(repo, tier=AlertTier.TIER_2_HIGH)

    tier2 = repo.list_alerts(tier=AlertTier.TIER_2_HIGH)
    assert len(tier2) == 2
    assert all(a.tier == AlertTier.TIER_2_HIGH for a in tier2)


def test_feedback_links_to_alert(in_memory_db_session):
    alert_repo = AlertRepository(in_memory_db_session)
    fb_repo = FeedbackRepository(in_memory_db_session)

    alert = _make_alert(alert_repo)
    fb = fb_repo.create_feedback(
        alert_id=alert.alert_id,
        investigator_id="inv_001",
        disposition=FeedbackDisposition.CLEARED,
    )
    fetched = fb_repo.list_for_alert(alert_id=alert.alert_id)
    assert len(fetched) == 1
    assert fetched[0].feedback_id == fb.feedback_id


def test_audit_log_write_and_query(in_memory_db_session):
    alert_repo = AlertRepository(in_memory_db_session)
    audit_repo = AuditLogRepository(in_memory_db_session)

    alert = _make_alert(alert_repo)
    audit_repo.write_event(
        event_type=AuditEventType.ALERT_CREATED,
        alert_id=alert.alert_id,
        event_data={"risk_score": 0.95},
    )
    events = audit_repo.list_events_for_alert(alert_id=alert.alert_id)
    assert len(events) == 1
    assert events[0].event_type == AuditEventType.ALERT_CREATED


def test_update_status_persists(in_memory_db_session):
    repo = AlertRepository(in_memory_db_session)
    alert = _make_alert(repo)
    repo.update_status(alert_id=alert.alert_id, new_status=AlertStatus.CLEARED)
    refetched = repo.get_alert(alert_id=alert.alert_id)
    assert refetched.status == AlertStatus.CLEARED
