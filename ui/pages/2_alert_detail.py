"""Alert detail page — full alert, narrative, citations, and feedback capture.

URL contract: this page expects ``?alert_id=<id>`` in the query string.
Linked from :mod:`ui.components.alert_card` and from the queue page.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from ui.components.alert_card import (
    STATUS_DISPLAY,
    TIER_COLOURS,
    TIER_DISPLAY_NAMES,
)
from ui.components.api_client import APIError, get_api_client
from ui.components.feature_explanation import (
    render_feature_table,
    render_risk_indicators,
)

st.set_page_config(
    page_title="Alert Detail — AML Monitoring",
    page_icon="🔍",
    layout="wide",
)

client = get_api_client()

# ----- Resolve alert_id from the URL -------------------------------------
query_params = st.query_params
alert_id = query_params.get("alert_id")

if not alert_id:
    st.title("Alert Detail")
    st.warning(
        "No alert selected. Open this page from the **Alert Queue** by "
        "clicking a card's _View detail_ button."
    )
    st.stop()

# ----- Fetch alert via the alerts listing (filtered to this id) ---------
# The current API does not expose a per-alert GET endpoint; the detail
# page filters the listing by tier/status and then matches in-memory.
# This is a portfolio-version constraint — production deployments add
# a /alerts/{id} endpoint and call it directly.
try:
    listing = client.list_alerts(limit=500)
except APIError as exc:
    st.error(f"Failed to fetch alerts — {exc}")
    st.stop()

alert: dict[str, Any] | None = next(
    (a for a in listing.get("alerts", []) if a.get("alert_id") == alert_id),
    None,
)

if alert is None:
    st.error(
        f"Alert `{alert_id}` not found in the most recent 500 listings. "
        "It may have been archived or the ID is malformed."
    )
    st.stop()

# ----- Header -----------------------------------------------------------
tier = str(alert.get("tier", "suppressed"))
tier_colour = TIER_COLOURS.get(tier, "#94a3b8")
tier_display = TIER_DISPLAY_NAMES.get(tier, tier.upper())

st.markdown(
    f"""
    <div style="display:inline-block; padding:6px 16px; border-radius:4px;
    background-color:{tier_colour}; color:white; font-weight:700;
    font-size:0.95rem; letter-spacing:0.5px;">{tier_display}</div>
    """,
    unsafe_allow_html=True,
)

st.title("Alert Detail")
st.markdown(
    f"**Alert ID:** `{alert['alert_id']}`  \n"
    f"**Transaction:** `{alert['transaction_id']}`  \n"
    f"**Created:** {alert.get('created_at', '—')}"
)

header_cols = st.columns(3)
header_cols[0].metric("Risk Score", f"{float(alert['risk_score']):.4f}")
header_cols[1].metric(
    "Status",
    STATUS_DISPLAY.get(str(alert.get("status", "open")), "Unknown"),
)
header_cols[2].metric(
    "Has Narrative",
    "Yes" if alert.get("has_narrative") else "No",
)

st.divider()

# ----- Tabs --------------------------------------------------------------
tab_narrative, tab_evidence, tab_feedback = st.tabs(
    ["📝 Narrative", "📊 Evidence", "✏️ Feedback"]
)

# ----- Narrative tab -----------------------------------------------------
with tab_narrative:
    if not alert.get("has_narrative"):
        st.info("Triage has not run for this alert.")
        if st.button("Run triage now", type="primary"):
            try:
                with st.spinner("Generating case narrative..."):
                    triage_response = client.post_triage(alert_id=alert_id)
                st.success(
                    "Triage complete. Refresh the page to view the narrative."
                )
                st.json(triage_response)
            except APIError as exc:
                st.error(f"Triage failed — {exc}")
    else:
        # In the portfolio version the narrative payload is not returned
        # by the listing endpoint; the on-demand /triage call returns
        # the freshly generated narrative which we render below. A
        # production /alerts/{id} endpoint would return the persisted
        # narrative directly.
        if st.button("Generate fresh narrative", type="secondary"):
            try:
                with st.spinner("Regenerating case narrative..."):
                    triage_response = client.post_triage(alert_id=alert_id)

                if triage_response.get("success") and triage_response.get("narrative"):
                    narrative = triage_response["narrative"]
                    st.markdown("### Summary")
                    st.markdown(narrative.get("summary", ""))

                    typology = narrative.get("suspected_typology")
                    if typology:
                        st.markdown(f"**Suspected typology:** `{typology}`")

                    st.markdown("### Risk indicators")
                    render_risk_indicators(narrative.get("risk_indicators", []))

                    recommended = narrative.get("recommended_action", {})
                    if recommended:
                        st.markdown("### Recommended action")
                        st.markdown(
                            f"**Priority:** {recommended.get('priority', '—')}  \n"
                            f"**SAR consideration:** "
                            f"{'Yes' if recommended.get('sar_consideration') else 'No'}"
                        )
                        steps = recommended.get("suggested_steps", [])
                        if steps:
                            st.markdown("**Suggested steps:**")
                            for step in steps:
                                st.markdown(f"- {step}")

                    refs = narrative.get("regulatory_references", [])
                    if refs:
                        st.markdown("### Regulatory references")
                        for ref in refs:
                            st.markdown(f"- {ref}")

                elif triage_response.get("refusal"):
                    refusal = triage_response["refusal"]
                    st.warning(
                        f"**Narrator refused** (code: `{refusal.get('code')}`)  \n\n"
                        f"{refusal.get('reason', '')}  \n\n"
                        f"_{refusal.get('recommended_action', '')}_"
                    )
            except APIError as exc:
                st.error(f"Triage failed — {exc}")
        else:
            st.info(
                "Click _Generate fresh narrative_ to view the case write-up. "
                "Re-triage is cheap and produces a current view."
            )

# ----- Evidence tab ------------------------------------------------------
with tab_evidence:
    st.markdown("### Top contributing features")
    # The listing endpoint does not include top features; surface a
    # message pointing to the on-demand triage flow that produces them.
    st.info(
        "The current API listing does not expose the per-alert feature "
        "breakdown. The /score endpoint returns the top features at "
        "alert-creation time; a production /alerts/{id} endpoint would "
        "include them on read."
    )
    # If the alert dict happens to carry top_features (some deployments
    # add this), render them.
    if alert.get("top_features"):
        render_feature_table(alert["top_features"])

# ----- Feedback tab ------------------------------------------------------
with tab_feedback:
    st.markdown(
        "Record your disposition for this alert. The feedback is persisted "
        "and feeds the model's continuous-learning loop."
    )

    investigator_id = st.text_input(
        "Investigator ID",
        value=st.session_state.get("investigator_id", "investigator-001"),
        help="Your operator identifier. Persisted across sessions.",
    )
    st.session_state["investigator_id"] = investigator_id

    disposition = st.selectbox(
        "Disposition",
        options=["cleared", "escalated", "sar_filed"],
        format_func=lambda v: v.replace("_", " ").title(),
    )

    justification = st.text_area(
        "Justification",
        placeholder=(
            "Optional for cleared. Recommended for escalated. "
            "Required for SAR filed."
        ),
        height=160,
    )

    if disposition == "sar_filed" and not justification.strip():
        st.warning(
            "A justification is required when filing a SAR. The API will reject "
            "the submission without one."
        )

    submit = st.button(
        "Submit feedback",
        type="primary",
        disabled=(disposition == "sar_filed" and not justification.strip()),
    )

    if submit:
        try:
            with st.spinner("Recording feedback..."):
                fb_response = client.post_feedback(
                    alert_id=alert_id,
                    investigator_id=investigator_id,
                    disposition=disposition,
                    justification=justification.strip() or None,
                )
            st.success(
                f"Feedback recorded. Alert is now **{fb_response.get('new_alert_status')}**."
            )
        except APIError as exc:
            st.error(f"Feedback submission failed — {exc}")
