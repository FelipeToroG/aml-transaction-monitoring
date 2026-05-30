"""Pydantic schemas for evidence-bound case narratives.

The triage layer's central design constraint is: the LLM never invents
facts. Every claim in a generated case narrative must trace to either a
specific transaction in the evidence bundle or a specific named feature
that fired against the entity. This module enforces that contract at
the schema level, so a narrative that violates it fails validation and
is rejected before reaching the persistence layer or the investigator.

Two output types
----------------
The narrator returns one of two structured outputs:

* :class:`CaseNarrative` — a complete, citation-bearing narrative
  suitable for investigator review and possible inclusion in a SAR
  filing.
* :class:`RefusalReason` — a structured refusal indicating the evidence
  is insufficient for a defensible narrative. Refusals are first-class
  outputs, not errors: a model that refuses on weak evidence is
  preferable to a model that hallucinates a plausible story.

Why discriminate at the schema level
------------------------------------
A free-text narrative can sound confident regardless of evidentiary
support. By forcing the model into a typed schema where every claim
must point at a specific transaction_id or feature_name, citation
becomes a *parsing requirement*, not a stylistic suggestion. Outputs
that fail to cite fail to deserialise — which is the strongest
hallucination check available short of human review.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------
# Citation building blocks
# ---------------------------------------------------------------------


class FeatureCitation(BaseModel):
    """Reference to a feature value present in the evidence bundle.

    The narrator emits one of these whenever it attributes a risk
    indicator to a feature value (e.g., "the entity's 24-hour
    sub-threshold share was 0.78"). Validation against the evidence
    bundle is performed in the narrator after parsing — the schema
    here only constrains the *shape* of the citation.
    """

    model_config = ConfigDict(extra="forbid")

    citation_type: Literal["feature"] = "feature"
    feature_name: str = Field(
        ..., min_length=1, description="Feature name as emitted by src.features"
    )
    observed_value: float = Field(
        ..., description="Value of the feature for the current alert"
    )
    interpretation: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description="One-sentence interpretation grounded in the feature value",
    )


class TransactionCitation(BaseModel):
    """Reference to a specific transaction in the evidence bundle.

    Emitted when the narrator attributes a risk indicator to one or more
    specific transactions (e.g., "the entity executed three transfers
    of $9,800 within four minutes; see transaction ids ...").
    """

    model_config = ConfigDict(extra="forbid")

    citation_type: Literal["transaction"] = "transaction"
    transaction_id: str = Field(
        ..., min_length=1, description="Transaction identifier from the evidence bundle"
    )
    interpretation: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description="One-sentence interpretation grounded in the transaction",
    )


# Union of citation types. Pydantic v2 dispatches the discriminator on
# the ``citation_type`` field at parse time, so a JSON payload with
# ``citation_type: "feature"`` becomes a FeatureCitation and one with
# ``citation_type: "transaction"`` becomes a TransactionCitation —
# no manual disambiguation in the narrator.
Citation = FeatureCitation | TransactionCitation


# ---------------------------------------------------------------------
# Risk indicators
# ---------------------------------------------------------------------


class RiskIndicator(BaseModel):
    """A single risk observation the narrator surfaces in the case write-up.

    Each indicator is bound to one or more citations. The
    ``model_validator`` below enforces the bind: a RiskIndicator with
    zero citations is the worst-case hallucination signal — an
    unsupported assertion — and is rejected.
    """

    model_config = ConfigDict(extra="forbid")

    description: str = Field(
        ...,
        min_length=20,
        max_length=600,
        description="Description of the risk observation in regulator-friendly language",
    )
    citations: list[Citation] = Field(
        ...,
        description="One or more citations grounding the description in the evidence bundle",
    )
    severity: Literal["low", "medium", "high"] = Field(
        ..., description="Severity assessment for this individual indicator"
    )

    @model_validator(mode="after")
    def _require_at_least_one_citation(self) -> "RiskIndicator":
        # An empty citation list is the hallucination footprint we are
        # explicitly defending against. The validator runs after the
        # default list constraint so we can produce a clearer error
        # message than ``min_length=1`` would yield on the parent field.
        if not self.citations:
            raise ValueError(
                "RiskIndicator must include at least one citation; "
                "uncited risk claims are rejected to prevent hallucination."
            )
        return self


# ---------------------------------------------------------------------
# Recommended action
# ---------------------------------------------------------------------


class RecommendedAction(BaseModel):
    """Action recommendation the narrator surfaces alongside the narrative.

    Investigators retain the final disposition decision; this section
    captures what the model thinks the next step should be, framed in
    operational terms (priority, concrete steps).
    """

    model_config = ConfigDict(extra="forbid")

    priority: Literal["immediate", "same_day", "next_cycle"] = Field(
        ...,
        description="Suggested review priority. Maps to the alert tier SLA targets.",
    )
    suggested_steps: list[str] = Field(
        ...,
        min_length=1,
        max_length=8,
        description="Concrete next steps for the investigator",
    )
    sar_consideration: bool = Field(
        ...,
        description="Whether the narrator believes a SAR filing should be considered",
    )


# ---------------------------------------------------------------------
# Top-level outputs
# ---------------------------------------------------------------------


class CaseNarrative(BaseModel):
    """Full evidence-bound case narrative for a single alert.

    The primary output of the triage layer when the evidence supports a
    defensible narrative. The structure is engineered to be both
    human-readable for compliance officers and machine-queryable for
    the audit log.
    """

    model_config = ConfigDict(extra="forbid")

    alert_id: str = Field(..., min_length=1)
    summary: str = Field(
        ...,
        min_length=40,
        max_length=800,
        description="Two- to four-sentence executive summary of the alert",
    )
    suspected_typology: str | None = Field(
        default=None,
        description=(
            "Typology code from src.data.typologies (e.g., 'structuring', "
            "'layering'). Optional because not every alert matches a "
            "single named typology."
        ),
    )
    risk_indicators: list[RiskIndicator] = Field(
        ...,
        min_length=1,
        max_length=8,
        description="Specific risk observations, each citation-bound",
    )
    recommended_action: RecommendedAction
    confidence: Literal["low", "medium", "high"] = Field(
        ...,
        description="Narrator's confidence in the overall assessment",
    )
    regulatory_references: list[str] = Field(
        default_factory=list,
        max_length=6,
        description=(
            "Optional citations to regulatory guidance (FATF, FinCEN, OFAC). "
            "Sourced from the typology catalog's regulatory_reference field "
            "when applicable."
        ),
    )


class RefusalReason(BaseModel):
    """Structured refusal when evidence is insufficient.

    A refusal is itself a valid narrator output. It signals that the
    alert should be reviewed by an investigator without LLM assistance.
    Surfaced to the operator metrics dashboard as the refusal rate; a
    spike in refusals indicates the upstream scoring model is producing
    weak-evidence alerts and warrants model-team attention.
    """

    model_config = ConfigDict(extra="forbid")

    alert_id: str = Field(..., min_length=1)
    code: Literal[
        "insufficient_evidence",
        "no_baseline",
        "ambiguous_pattern",
        "schema_failure",
    ] = Field(
        ...,
        description=(
            "Machine-readable refusal code. 'schema_failure' is reserved for "
            "the narrator's internal use when validation retries are exhausted."
        ),
    )
    reason: str = Field(
        ...,
        min_length=20,
        max_length=500,
        description="Investigator-facing explanation of why automated triage was declined",
    )
    recommended_action: str = Field(
        ...,
        min_length=20,
        max_length=400,
        description="What the investigator should do given the refusal",
    )


class NarratorResult(BaseModel):
    """Discriminated container for whichever output the narrator produced.

    Exactly one of ``narrative`` or ``refusal`` is populated. The
    container exists because the persistence layer stores both
    successful narratives and refusals in the same audit table, and
    callers should not have to handle either type via exception flow.
    """

    # Disable the "model_" protected namespace so the descriptive
    # ``model_name`` field (which records the upstream Claude model that
    # produced the output) does not trip pydantic's UserWarning on every
    # construction. The field is data, not pydantic machinery.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    alert_id: str
    success: bool
    narrative: CaseNarrative | None = None
    refusal: RefusalReason | None = None
    model_name: str = Field(..., description="LLM model that produced the output")
    prompt_version: str = Field(..., description="Prompt version that produced the output")
    latency_ms: float = Field(..., ge=0, description="End-to-end narrator latency in milliseconds")
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _require_exactly_one_payload(self) -> "NarratorResult":
        # XOR check: success implies narrative present and refusal
        # absent; failure implies the opposite. Catches construction
        # bugs at the boundary rather than letting them propagate.
        if self.success and self.narrative is None:
            raise ValueError("success=True requires narrative to be present.")
        if self.success and self.refusal is not None:
            raise ValueError("success=True is incompatible with a refusal payload.")
        if not self.success and self.refusal is None:
            raise ValueError("success=False requires refusal to be present.")
        if not self.success and self.narrative is not None:
            raise ValueError("success=False is incompatible with a narrative payload.")
        return self
