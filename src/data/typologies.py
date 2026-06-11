"""AML typology catalog.

A typology is a named laundering pattern that compliance teams recognise
and investigators are trained to look for. Centralising the vocabulary
here serves three downstream consumers:

1. **Feature engineering**: each typology pins one or more features that
   exist specifically to detect it. The link from typology to feature is
   visible in code review rather than buried in commit history.
2. **The Claude narrator**: when assembling a case narrative the
   narrator references typologies by name so investigators read familiar
   regulatory language ("structuring" rather than "cluster-7-pattern").
3. **The Streamlit UI**: alert cards display typology badges, sourced
   from this catalog, so the colour-coded badges are consistent across
   the dashboard and the case write-ups.

The typology definitions are drawn from FinCEN's *SAR Narrative Guidance*
and the FATF *Typologies* series, both of which are the regulatory
references investigators cite in SAR filings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class Typology:
    """Named AML laundering pattern with detection metadata.

    Attributes
    ----------
    code : str
        Short stable identifier used as a primary key by the rest of the
        codebase. Lowercase snake_case. Never renamed after release
        because persisted alert records reference this value.
    name : str
        Human-readable name as it appears in narratives and on the UI.
    description : str
        One-paragraph description of what the typology looks like in
        practice. Surfaces in tooltips and in the narrator's evidence
        bundle so the LLM is grounded in the same definition the
        investigator sees.
    detection_signals : tuple[str, ...]
        The feature names (as emitted by ``src.features``) that
        contribute the most signal for this typology. Used by the
        narrator's evidence assembler to prioritise which feature values
        to cite when explaining an alert that matches this pattern.
    regulatory_reference : str
        Authoritative source - typically FinCEN guidance or a FATF
        typology report. Surfaces in the narrator output so SAR drafters
        can cite the reference directly when filing.
    """

    code: str
    name: str
    description: str
    detection_signals: tuple[str, ...]
    regulatory_reference: str


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
# The TYPOLOGIES dict is the authoritative catalog. Add typologies here
# rather than scattering string constants across the codebase. Order is
# insignificant; the dict is keyed by code for O(1) lookup from the
# narrator and the API.

TYPOLOGIES: Final[dict[str, Typology]] = {
    "structuring": Typology(
        code="structuring",
        name="Structuring",
        description=(
            "Splitting a single large transaction into multiple smaller "
            "transactions to keep each individual amount below the regulatory "
            "Currency Transaction Report threshold (USD 10,000 in the United "
            "States). Detected when one entity transacts repeatedly at "
            "amounts just under the threshold within a short window."
        ),
        detection_signals=(
            "entity_24h_txn_count",
            "entity_24h_sub_threshold_share",
            "entity_24h_max_amount",
            "amount_proximity_to_threshold",
        ),
        regulatory_reference=(
            "31 CFR §1010.311 (Currency Transaction Report threshold); "
            "FinCEN SAR Narrative Guidance §III.B."
        ),
    ),
    "smurfing": Typology(
        code="smurfing",
        name="Smurfing",
        description=(
            "Distributing a large laundering flow across many lower-profile "
            "accounts ('smurfs') so that no single account exhibits the "
            "volume that would normally trigger scrutiny. Detected by the "
            "convergent-source pattern where a destination entity receives "
            "many small inbound transfers from unrelated sources in a short "
            "window."
        ),
        detection_signals=(
            "dest_24h_unique_sources",
            "dest_24h_inbound_count",
            "dest_24h_inbound_amount_variance",
            "source_dest_edge_novelty",
        ),
        regulatory_reference=(
            "FATF Typologies Report 2018: Professional Money Laundering, §3.2."
        ),
    ),
    "layering": Typology(
        code="layering",
        name="Layering",
        description=(
            "Chaining transfers through multiple intermediate accounts to "
            "obscure the origin of funds. Detected by entity-graph features "
            "that quantify how an account participates in long paths of "
            "rapidly successive transfers, often crossing institutional or "
            "jurisdictional boundaries."
        ),
        detection_signals=(
            "entity_pagerank_24h",
            "entity_in_out_ratio_1h",
            "entity_pass_through_velocity",
            "cross_bank_hop_count_24h",
        ),
        regulatory_reference=(
            "FATF: The Three Stages of Money Laundering - Layering; "
            "FinCEN SAR Narrative Guidance §III.D."
        ),
    ),
    "integration": Typology(
        code="integration",
        name="Integration",
        description=(
            "Reintroducing laundered funds into the legitimate economy "
            "through normal-looking commerce - investments, real-estate, or "
            "luxury purchases. Hardest typology to detect from transaction "
            "data alone; the system flags integration via inbound-flow "
            "anomalies on accounts whose historical activity does not "
            "justify the size of the deposit."
        ),
        detection_signals=(
            "entity_inbound_30d_to_baseline_ratio",
            "entity_baseline_inbound_variance",
            "entity_dormant_then_active_flag",
        ),
        regulatory_reference=(
            "FATF: The Three Stages of Money Laundering - Integration."
        ),
    ),
    "rapid_movement": Typology(
        code="rapid_movement",
        name="Rapid In-Out Movement (Money Mule)",
        description=(
            "Funds arrive in an account and leave again within minutes or "
            "hours, with no consumption or merchant activity in between. "
            "The classic money-mule pattern. Detected by per-entity "
            "in-out velocity in short time windows."
        ),
        detection_signals=(
            "entity_in_out_ratio_1h",
            "entity_settlement_latency_seconds",
            "entity_1h_throughput_to_baseline_ratio",
        ),
        regulatory_reference=(
            "FinCEN Advisory FIN-2020-A003 on Money Mule Schemes."
        ),
    ),
    "round_amounts": Typology(
        code="round_amounts",
        name="Round-Amount Anomaly",
        description=(
            "Statistically improbable concentration of round-dollar amounts "
            "(e.g., 1,000.00 / 5,000.00 / 10,000.00) in an entity's recent "
            "activity. Round amounts are the signature of human-orchestrated "
            "laundering as opposed to natural commerce, where transaction "
            "amounts cluster on non-round values driven by item pricing and "
            "tax."
        ),
        detection_signals=(
            "entity_24h_round_amount_share",
            "amount_modulo_signal",
        ),
        regulatory_reference=(
            "FATF Typologies Report 2021: Indicators of Suspicious Activity."
        ),
    ),
    "high_risk_corridor": Typology(
        code="high_risk_corridor",
        name="High-Risk Corridor Flow",
        description=(
            "Transactions flowing through banking corridors flagged as "
            "elevated risk by FATF, OFAC, or institutional risk policy. "
            "Detected when the source-destination bank pair maps to a "
            "corridor on the institution's elevated-risk list."
        ),
        detection_signals=(
            "corridor_risk_score",
            "bank_pair_historical_illicit_rate",
        ),
        regulatory_reference=(
            "FATF Public Statement on High-Risk Jurisdictions; OFAC SDN List."
        ),
    ),
}


def get_typology(code: str) -> Typology:
    """Look up a typology by code with an actionable error on miss.

    Raises a ``KeyError`` with the available codes listed in the message
    rather than a bare ``KeyError(code)``, so an upstream operator who
    typoes a code in a config sees the valid options immediately.
    """
    try:
        return TYPOLOGIES[code]
    except KeyError as exc:
        raise KeyError(
            f"Unknown typology '{code}'. Valid codes: {sorted(TYPOLOGIES)}"
        ) from exc
