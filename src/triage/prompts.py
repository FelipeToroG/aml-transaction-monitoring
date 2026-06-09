"""Versioned prompt templates for the Claude case narrator.

Prompt templates are kept in code (not YAML) for three reasons:

1. **Review parity with code.** A prompt change is a behavioural
   change. Putting prompts in the codebase means they go through pull
   request review like any other behavioural change.
2. **Versioning.** Each prompt revision is assigned an explicit version
   string. The narrator records the version on every result, so an
   alert's narrative can always be traced back to the exact prompt
   that produced it.
3. **A/B testing.** Multiple versions can coexist in the file; a
   feature flag in the runtime config selects which one is active.
   The narrator stamps the active version into the audit record.

The current production prompt is ``CASE_NARRATIVE_V1``. New versions
are added by appending — never by mutating an existing constant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final

from src.data.typologies import TYPOLOGIES


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A named, versioned prompt.

    Attributes
    ----------
    version : str
        Stable identifier emitted in every narrator result for audit
        traceability. Use semver-style strings (``"v1.0"``) so future
        revisions are sortable.
    system_prompt : str
        System message setting the persona, constraints, and refusal
        criteria.
    user_template : str
        Format string for the user message. Format keys correspond to
        the fields the narrator's ``_render_user_prompt`` populates.
    """

    version: str
    system_prompt: str
    user_template: str


# ---------------------------------------------------------------------
# v1: production prompt
# ---------------------------------------------------------------------

_TYPOLOGY_DESCRIPTIONS_BLOCK = "\n".join(
    f"- **{t.code}** ({t.name}): {t.description} [Reference: {t.regulatory_reference}]"
    for t in TYPOLOGIES.values()
)


_SYSTEM_PROMPT_V1 = f"""You are a tier-2 AML compliance analyst at a US-regulated payments platform. You draft case narratives for investigator review and may produce text that is incorporated into Suspicious Activity Reports filed with FinCEN. Your output is evidence in a regulated process.

# Core constraints (non-negotiable)

1. You produce narratives *only* from the evidence bundle provided in each user message. You never assert facts not present in the bundle. You never speculate about the entity's intent, business model, customer base, or activities outside the bundle's scope.

2. Every risk indicator you surface must include at least one citation. A citation references either:
   - A specific `transaction_id` present in the bundle, or
   - A specific `feature_name` present in the bundle's triggered_features section
   You attach citations using the structured schema below — citations are not free text.

3. If the evidence is insufficient to draft a defensible narrative, you produce a structured refusal. Insufficient means any of the following:
   - The triggered_features list is empty or all features are below their tier-1 thresholds.
   - The entity has no baseline of prior activity to compare against.
   - The pattern is ambiguous enough that two compliance analysts could reasonably reach different conclusions from the same evidence.
   Refusals are first-class outputs. They are not failures — they correctly signal alerts that require investigator review without LLM assistance.

# AML typology catalog (cite by code when applicable)

{_TYPOLOGY_DESCRIPTIONS_BLOCK}

# Output format

Your entire response is a single JSON object. No preamble. No commentary outside the JSON. No code fences. The JSON must conform to the schema specified in each user message.

When the evidence supports a narrative, return a `CaseNarrative` object. When it does not, return a `RefusalReason` object. Wrap whichever you produce in a top-level discriminator object of the form:

    {{
        "outcome": "narrative" | "refusal",
        "payload": {{ ... the corresponding CaseNarrative or RefusalReason object ... }}
    }}

# Style notes

- Write in the professional, evidence-cautious register of a SAR narrative. Prefer "the entity transferred" over "the criminal moved".
- Cite typologies by their codes (e.g., "structuring") so the downstream system can map your output to the typology catalog.
- Keep the summary to two to four sentences. Risk indicators carry the detail.
"""


_USER_TEMPLATE_V1 = """## Alert under triage

**Alert ID:** {alert_id}
**Score:** {risk_score:.4f} (anomaly: {anomaly_score:.4f}, supervised: {supervised_score:.4f})
**Tier:** {tier}
**Generated at:** {generated_at_iso}

## Current transaction

```json
{transaction_json}
```

## Source entity recent activity (trailing 24h, top features)

```json
{source_activity_json}
```

## Destination entity recent activity (trailing 24h, top features)

```json
{destination_activity_json}
```

## Triggered features (top contributors to the score)

```json
{triggered_features_json}
```

## Required output schema

```json
{output_schema_json}
```

Produce a single JSON object exactly matching the discriminator format described in the system prompt. Cite specific `transaction_id` or `feature_name` values from this bundle for every risk indicator. If the evidence is insufficient, return a refusal. Output JSON only.
"""


CASE_NARRATIVE_V1: Final[PromptTemplate] = PromptTemplate(
    version="v1.0",
    system_prompt=_SYSTEM_PROMPT_V1,
    user_template=_USER_TEMPLATE_V1,
)


# ---------------------------------------------------------------------
# Retry-strengthening overlay
# ---------------------------------------------------------------------
# When the narrator's first attempt fails schema validation, the second
# attempt prepends this strengthening preamble to the user message.
# The preamble names the specific validation error so the model can
# correct rather than guess.

VALIDATION_RETRY_PREAMBLE: Final[
    str
] = """Your previous response failed schema validation with the following error:

```
{validation_error}
```

Re-read the schema carefully. Produce a corrected response. Common causes:
- Missing required fields.
- Risk indicators without at least one citation.
- A citation_type field that does not match its sibling fields.
- Extra fields not allowed by the schema.

Output the corrected JSON object only. No commentary.
"""


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------
# Centralised lookup so the narrator's runtime version selection has a
# single source of truth. Add new versions here; never delete old
# versions (audit records reference them by string).

PROMPT_REGISTRY: Final[dict[str, PromptTemplate]] = {
    CASE_NARRATIVE_V1.version: CASE_NARRATIVE_V1,
}


def get_prompt(version: str) -> PromptTemplate:
    """Look up a prompt by version with a helpful error on miss."""
    try:
        return PROMPT_REGISTRY[version]
    except KeyError as exc:
        raise KeyError(
            f"Unknown prompt version {version!r}. "
            f"Available: {sorted(PROMPT_REGISTRY)}"
        ) from exc


def render_output_schema_doc() -> str:
    """Render a JSON-string description of the expected output schema.

    Used by the user template to give the model a verbatim view of the
    required structure. We render manually rather than dumping the
    Pydantic schema because the model produces dramatically better
    structured output when it sees a concrete example with annotations
    than when it sees a JSON Schema draft.

    Both outcomes are documented. The refusal shape is included because
    omitting it caused the model to invent a refusal payload that mirrored
    the narrative's nested ``recommended_action`` object and overshot the
    ``reason`` length cap, failing validation and degrading every refusal
    to a ``schema_failure``. The refusal's ``recommended_action`` is a
    plain string, NOT the structured object the narrative uses.
    """
    narrative_shape = {
        "outcome": "narrative",
        "payload": {
            "alert_id": "<alert_id from this message>",
            "summary": "<2-4 sentence executive summary>",
            "suspected_typology": "<typology code or null>",
            "risk_indicators": [
                {
                    "description": "<risk observation in regulator-friendly language>",
                    "citations": [
                        {
                            "citation_type": "feature",
                            "feature_name": "<feature_name from triggered_features>",
                            "observed_value": 0.0,
                            "interpretation": "<one-sentence interpretation>",
                        },
                        {
                            "citation_type": "transaction",
                            "transaction_id": "<transaction_id from bundle>",
                            "interpretation": "<one-sentence interpretation>",
                        },
                    ],
                    "severity": "low | medium | high",
                }
            ],
            "recommended_action": {
                "priority": "immediate | same_day | next_cycle",
                "suggested_steps": ["<concrete step>", "<concrete step>"],
                "sar_consideration": True,
            },
            "confidence": "low | medium | high",
            "regulatory_references": ["<optional regulatory citation>"],
        },
    }
    refusal_shape = {
        "outcome": "refusal",
        "payload": {
            "alert_id": "<alert_id from this message>",
            "code": "insufficient_evidence | no_baseline | ambiguous_pattern",
            "reason": (
                "<plain string, 20-500 characters, investigator-facing. "
                "MUST be a string, not an object or list. Keep it under 500 "
                "characters.>"
            ),
            "recommended_action": (
                "<plain string, 20-400 characters describing what the "
                "investigator should do. MUST be a single string, NOT a "
                "nested object and NOT a list. This differs from the "
                "narrative's structured recommended_action.>"
            ),
        },
    }
    doc = {
        "instructions": (
            "Return ONE JSON object matching the {outcome, payload} "
            "discriminator. Use the narrative_shape when the evidence "
            "supports a defensible write-up; use the refusal_shape when it "
            "does not. Emit exactly one shape, not both."
        ),
        "narrative_shape": narrative_shape,
        "refusal_shape": refusal_shape,
    }
    return json.dumps(doc, indent=2)
