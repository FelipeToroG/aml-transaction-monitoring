"""Feature contribution display component.

Used by the alert-detail page to render the top contributing features
for an alert in a sortable table with optional sparkline hints for
each feature's distribution.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def render_feature_table(features: list[dict[str, Any]]) -> None:
    """Render a table of feature contributions with rank, name, and value.

    Inputs are the ``top_features`` list from the scoring response or
    the ``triggered_features`` list from a persisted evidence snapshot;
    both have the same dict shape.
    """
    if not features:
        st.info("No top-feature contributions recorded for this alert.")
        return

    frame = pd.DataFrame(features)
    column_order = [
        c
        for c in ("contribution_rank", "feature_name", "observed_value")
        if c in frame.columns
    ]
    frame = frame[column_order].rename(
        columns={
            "contribution_rank": "Rank",
            "feature_name": "Feature",
            "observed_value": "Observed value",
        }
    )

    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Observed value": st.column_config.NumberColumn(format="%.4f"),
        },
    )


def render_risk_indicators(indicators: list[dict[str, Any]]) -> None:
    """Render the case-narrative's risk indicators with citations.

    Each indicator is rendered as a bordered container so investigators
    can scan severity at a glance and click through to the citations
    when they want to verify the underlying evidence.
    """
    if not indicators:
        st.info("No risk indicators on this narrative.")
        return

    severity_colours = {
        "high": "#ef4444",
        "medium": "#f59e0b",
        "low": "#eab308",
    }

    for idx, indicator in enumerate(indicators, start=1):
        severity = str(indicator.get("severity", "low"))
        colour = severity_colours.get(severity, "#94a3b8")

        with st.container(border=True):
            col_marker, col_body = st.columns([1, 12])
            with col_marker:
                st.markdown(
                    f"""<div style="background-color:{colour}; color:white;
                    border-radius:50%; width:32px; height:32px; display:flex;
                    align-items:center; justify-content:center; font-weight:600;">
                    {idx}</div>""",
                    unsafe_allow_html=True,
                )
            with col_body:
                st.markdown(f"**Severity:** `{severity.upper()}`")
                st.markdown(indicator.get("description", ""))

                citations = indicator.get("citations", [])
                if citations:
                    with st.expander(f"Citations ({len(citations)})"):
                        for c_idx, citation in enumerate(citations, start=1):
                            cite_type = citation.get("citation_type", "?")
                            interp = citation.get("interpretation", "")
                            if cite_type == "feature":
                                st.markdown(
                                    f"**{c_idx}. Feature** `{citation.get('feature_name')}` "
                                    f"= `{citation.get('observed_value')}` — {interp}"
                                )
                            elif cite_type == "transaction":
                                st.markdown(
                                    f"**{c_idx}. Transaction** "
                                    f"`{citation.get('transaction_id')}` — {interp}"
                                )
                            else:
                                st.markdown(f"**{c_idx}.** {interp}")
