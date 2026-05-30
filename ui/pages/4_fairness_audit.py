"""Fairness audit page — segment-level parity metrics.

Reads from ``mlruns/fairness_snapshot.json`` written by the model
monitoring loop. As with the drift page, a production deployment
plugs Prometheus queries in here instead of file IO.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Fairness Audit — AML Monitoring",
    page_icon="⚖️",
    layout="wide",
)

st.title("Fairness Audit")
st.caption(
    "Segment-level demographic parity, equal opportunity, and FPR parity. "
    "Required by US SR 11-7 model risk management."
)

snapshot_path = Path("mlruns/fairness_snapshot.json")

if not snapshot_path.exists():
    st.info(
        "No fairness snapshot found at `mlruns/fairness_snapshot.json`. The "
        "fairness audit runs in `src.monitoring.fairness` as part of the "
        "monitoring loop. After the first run completes, refresh this page."
    )

    # Synthetic demonstration view ----------------------------------------
    st.divider()
    st.subheader("Demonstration view (synthetic data)")
    st.caption(
        "The charts below are generated from synthetic data to illustrate the "
        "page's output. The real visualisation reads from the snapshot file."
    )

    demo_segments = pd.DataFrame(
        {
            "segment": ["USD", "EUR", "GBP", "CHF", "JPY", "Other"],
            "alert_rate": [0.012, 0.011, 0.014, 0.010, 0.018, 0.009],
            "true_positive_rate": [0.78, 0.81, 0.74, 0.77, 0.72, 0.79],
            "false_positive_rate": [0.015, 0.014, 0.019, 0.012, 0.024, 0.013],
        }
    )

    cols = st.columns(2)
    with cols[0]:
        fig = px.bar(
            demo_segments,
            x="segment",
            y="alert_rate",
            title="Alert rate by payment currency (demographic parity)",
            labels={"alert_rate": "Share of transactions alerted", "segment": ""},
        )
        st.plotly_chart(fig, use_container_width=True)
    with cols[1]:
        fig = px.bar(
            demo_segments,
            x="segment",
            y="false_positive_rate",
            title="False-positive rate by payment currency (FPR parity)",
            color="false_positive_rate",
            color_continuous_scale="RdYlGn_r",
            labels={"false_positive_rate": "FPR", "segment": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(demo_segments, use_container_width=True, hide_index=True)

    st.markdown(
        """
        **How to read this page:**

        - **Demographic parity** — alert rate should not differ
          disproportionately across segments unless the difference is
          empirically justified by the underlying risk profile.
        - **Equal opportunity (TPR parity)** — the model should catch
          true positives at similar rates across segments. A meaningful
          gap is a model-team escalation.
        - **FPR parity** — the model should not over-alert any segment.
          Gaps here are the most common source of disparate-impact
          findings in regulator review.
        """
    )

    st.stop()

with snapshot_path.open() as fh:
    snapshot = json.load(fh)

segments = pd.DataFrame(snapshot.get("segments", []))
if segments.empty:
    st.warning("Fairness snapshot is empty. Check the monitoring run logs.")
    st.stop()

# ----- Header ------------------------------------------------------------
st.markdown(
    f"**Snapshot generated:** {snapshot.get('generated_at_utc', 'unknown')}  \n"
    f"**Segment dimension:** `{snapshot.get('segment_column', 'unknown')}`"
)
st.divider()

# ----- Side-by-side bar charts ------------------------------------------
cols = st.columns(2)
with cols[0]:
    if "alert_rate" in segments.columns:
        fig = px.bar(
            segments,
            x="segment",
            y="alert_rate",
            title="Alert rate (demographic parity)",
            labels={"alert_rate": "Share alerted", "segment": ""},
        )
        st.plotly_chart(fig, use_container_width=True)
with cols[1]:
    if "false_positive_rate" in segments.columns:
        fig = px.bar(
            segments,
            x="segment",
            y="false_positive_rate",
            color="false_positive_rate",
            color_continuous_scale="RdYlGn_r",
            title="False-positive rate (FPR parity)",
            labels={"false_positive_rate": "FPR", "segment": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

# ----- Full table --------------------------------------------------------
st.divider()
st.subheader("Segment-level metrics")
st.dataframe(segments, use_container_width=True, hide_index=True)
