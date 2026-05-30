"""Alert queue page — filtered, paginated alert listing for investigators."""

from __future__ import annotations

import streamlit as st

from ui.components.alert_card import render_alert_card
from ui.components.api_client import APIError, get_api_client

st.set_page_config(
    page_title="Alert Queue — AML Monitoring",
    page_icon="📋",
    layout="wide",
)

st.title("Alert Queue")
st.caption("Filter, scan, and pick up alerts in priority order.")

client = get_api_client()

# ----- Sidebar filters ---------------------------------------------------
st.sidebar.header("Filters")

status_filter = st.sidebar.selectbox(
    "Status",
    options=[None, "open", "in_review", "cleared", "escalated", "sar_filed"],
    format_func=lambda v: "All statuses" if v is None else v.replace("_", " ").title(),
)

tier_filter = st.sidebar.selectbox(
    "Tier",
    options=[None, "tier_3_critical", "tier_2_high", "tier_1_medium", "suppressed"],
    format_func=lambda v: "All tiers" if v is None else v.replace("_", " ").title(),
)

page_size = st.sidebar.slider("Page size", min_value=10, max_value=100, value=25, step=5)
page_number = st.sidebar.number_input("Page", min_value=1, value=1, step=1)
offset = (int(page_number) - 1) * int(page_size)

# ----- Fetch alerts ------------------------------------------------------
try:
    response = client.list_alerts(
        status=status_filter,
        tier=tier_filter,
        limit=page_size,
        offset=offset,
    )
except APIError as exc:
    st.error(f"Failed to load alerts — {exc}")
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Unexpected error while loading alerts: {exc}")
    st.stop()

alerts = response.get("alerts", [])
total_matching = int(response.get("total_matching", 0))

# ----- Header metrics ----------------------------------------------------
metric_cols = st.columns(3)
metric_cols[0].metric("Showing", f"{len(alerts)} of {total_matching}")
metric_cols[1].metric("Page", f"{page_number}")
metric_cols[2].metric("Page size", page_size)

st.divider()

# ----- Empty state -------------------------------------------------------
if not alerts:
    st.info(
        "No alerts match the current filters. Loosen the status or tier "
        "filter, or check the **Service status** on the landing page."
    )
    st.stop()

# ----- Render alerts -----------------------------------------------------
for alert in alerts:
    render_alert_card(alert, link_to_detail=True)
