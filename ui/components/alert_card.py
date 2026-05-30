"""Reusable alert-card component.

Used by the queue page and the alert-detail page to render an alert's
headline information in a single, consistent visual unit. Centralising
the rendering means a styling tweak applied here propagates everywhere
the card is used without copy-paste drift.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

# Tier-specific colour assignments. Sourced once here so the queue
# colour coding, the detail-page banner, and any future operator-
# dashboard visualisation stay in lockstep.
TIER_COLOURS: dict[str, str] = {
    "tier_3_critical": "#ef4444",
    "tier_2_high": "#f59e0b",
    "tier_1_medium": "#eab308",
    "suppressed": "#94a3b8",
}

TIER_DISPLAY_NAMES: dict[str, str] = {
    "tier_3_critical": "TIER 3 — CRITICAL",
    "tier_2_high": "TIER 2 — HIGH",
    "tier_1_medium": "TIER 1 — MEDIUM",
    "suppressed": "SUPPRESSED",
}

STATUS_DISPLAY: dict[str, str] = {
    "open": "Open",
    "in_review": "In Review",
    "cleared": "Cleared",
    "escalated": "Escalated",
    "sar_filed": "SAR Filed",
}


def render_alert_card(alert: dict[str, Any], *, link_to_detail: bool = True) -> None:
    """Render an alert summary as a card.

    The card displays the alert ID, transaction ID, score, tier badge,
    status, typology (if assigned), and a button linking to the detail
    page when ``link_to_detail`` is true.
    """
    tier = str(alert.get("tier", "suppressed"))
    tier_colour = TIER_COLOURS.get(tier, "#94a3b8")
    tier_display = TIER_DISPLAY_NAMES.get(tier, tier.upper())

    with st.container(border=True):
        # Header row: tier badge + status
        col_badge, col_score, col_status = st.columns([2, 1, 1])

        with col_badge:
            st.markdown(
                f"""
                <div style="display:inline-block; padding:4px 12px; border-radius:4px;
                background-color:{tier_colour}; color:white; font-weight:600;
                font-size:0.85rem; letter-spacing:0.5px;">
                {tier_display}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_score:
            st.metric(
                "Risk Score",
                f"{float(alert.get('risk_score', 0)):.4f}",
                label_visibility="visible",
            )

        with col_status:
            st.markdown(
                f"**Status**\n\n{STATUS_DISPLAY.get(str(alert.get('status', 'open')), 'Unknown')}"
            )

        # Body row: identifiers and typology
        st.markdown(
            f"**Alert ID:** `{alert.get('alert_id', '?')}`  \n"
            f"**Transaction ID:** `{alert.get('transaction_id', '?')}`  \n"
            f"**Created:** {alert.get('created_at', '—')}"
        )

        typology = alert.get("suspected_typology")
        if typology:
            st.markdown(f"**Suspected typology:** `{typology}`")

        has_narrative = alert.get("has_narrative", alert.get("narrative_payload") is not None)
        narrative_indicator = "✓ Narrative present" if has_narrative else "○ Narrative pending"
        st.caption(narrative_indicator)

        if link_to_detail:
            # Streamlit's link_button takes a relative URL; we pass the
            # alert_id as a query parameter so the detail page can read it.
            st.link_button(
                "View detail →",
                f"/alert_detail?alert_id={alert.get('alert_id')}",
                use_container_width=False,
            )
