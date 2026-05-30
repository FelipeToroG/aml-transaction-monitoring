"""Tests for the narrator's output schemas.

The schema-level enforcement is the hallucination guard. These tests
ensure the guard fires on the specific malformed shapes the LLM is
known to occasionally produce.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.triage.schemas import (
    CaseNarrative,
    FeatureCitation,
    NarratorResult,
    RecommendedAction,
    RefusalReason,
    RiskIndicator,
    TransactionCitation,
)


def _valid_narrative_payload() -> dict:
    return {
        "alert_id": "test_001",
        "summary": "Entity executed multiple sub-threshold transactions in a short window.",
        "suspected_typology": "structuring",
        "risk_indicators": [
            {
                "description": "Eight transactions of $9,500 within 30 minutes — classic structuring signature.",
                "citations": [
                    {
                        "citation_type": "feature",
                        "feature_name": "src_24h_sub_threshold_share",
                        "observed_value": 0.85,
                        "interpretation": "Above the warning threshold of 0.50.",
                    }
                ],
                "severity": "high",
            }
        ],
        "recommended_action": {
            "priority": "immediate",
            "suggested_steps": ["Review customer profile", "Check counterparty history"],
            "sar_consideration": True,
        },
        "confidence": "high",
        "regulatory_references": [],
    }


def test_valid_narrative_round_trips():
    narrative = CaseNarrative.model_validate(_valid_narrative_payload())
    assert narrative.alert_id == "test_001"
    assert isinstance(narrative.risk_indicators[0].citations[0], FeatureCitation)


def test_risk_indicator_rejects_zero_citations():
    """RiskIndicator without citations fails validation."""
    payload = _valid_narrative_payload()
    payload["risk_indicators"][0]["citations"] = []
    with pytest.raises(ValidationError):
        CaseNarrative.model_validate(payload)


def test_unknown_field_rejected_by_extra_forbid():
    """A field not in the schema fails validation."""
    payload = _valid_narrative_payload()
    payload["extra_field"] = "should_not_be_here"
    with pytest.raises(ValidationError):
        CaseNarrative.model_validate(payload)


def test_narrator_result_xor_invariant_success():
    """success=True requires narrative present and refusal absent."""
    narrative = CaseNarrative.model_validate(_valid_narrative_payload())
    with pytest.raises(ValidationError):
        NarratorResult(
            alert_id="test_001",
            success=True,
            narrative=None,
            refusal=None,
            model_name="claude-sonnet-4-5",
            prompt_version="v1.0",
            latency_ms=1234.5,
        )


def test_narrator_result_xor_invariant_failure():
    """success=False requires refusal present and narrative absent."""
    refusal = RefusalReason(
        alert_id="test_001",
        code="insufficient_evidence",
        reason="No features above tier-1 threshold.",
        recommended_action="Investigator review without LLM assistance.",
    )
    result = NarratorResult(
        alert_id="test_001",
        success=False,
        narrative=None,
        refusal=refusal,
        model_name="claude-sonnet-4-5",
        prompt_version="v1.0",
        latency_ms=987.6,
    )
    assert result.success is False
    assert result.refusal is not None
