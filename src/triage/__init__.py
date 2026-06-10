"""Claude-powered evidence-bound case narrative generation.

Public surface for the triage layer. The FastAPI route and the test
suite both import from here:

* ``Narrator`` - the high-level facade for case-narrative generation.
* ``EvidenceBundle`` - the structured input shape the narrator consumes.
* ``CaseNarrative`` / ``RefusalReason`` / ``NarratorResult`` - the
  Pydantic models constraining the narrator output.
* ``CASE_NARRATIVE_V1`` / ``get_prompt`` / ``PROMPT_REGISTRY`` - the
  versioned prompt library.
"""

from src.triage.narrator import EvidenceBundle, Narrator
from src.triage.prompts import (
    CASE_NARRATIVE_V1,
    PROMPT_REGISTRY,
    PromptTemplate,
    get_prompt,
    render_output_schema_doc,
)
from src.triage.schemas import (
    CaseNarrative,
    Citation,
    FeatureCitation,
    NarratorResult,
    RecommendedAction,
    RefusalReason,
    RiskIndicator,
    TransactionCitation,
)

__all__ = [
    "CASE_NARRATIVE_V1",
    "CaseNarrative",
    "Citation",
    "EvidenceBundle",
    "FeatureCitation",
    "Narrator",
    "NarratorResult",
    "PROMPT_REGISTRY",
    "PromptTemplate",
    "RecommendedAction",
    "RefusalReason",
    "RiskIndicator",
    "TransactionCitation",
    "get_prompt",
    "render_output_schema_doc",
]
