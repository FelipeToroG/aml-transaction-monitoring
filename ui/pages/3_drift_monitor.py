"""Drift monitor page — feature and prediction PSI visualisation.

The current portfolio version reads drift snapshots from a JSON file
at ``mlruns/drift_snapshot.json`` written by ``src.monitoring.drift``.
A production deployment integrates this page directly with Prometheus
via PromQL queries against the metrics scraped from the API; the
data-binding surface here is what would change in that integration.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Drift Monitor — AML Monitoring",
    page_icon="📈",
    layout="wide",
)

st.title("Drift Monitor")
st.caption("Population Stability Index for features and prediction distribution.")

# ----- Load drift snapshot -----------------------------------------------
snapshot_path = Path("mlruns/drift_snapshot.json")

if not snapshot_path.exists():
    st.info(
        "No drift snapshot found at `mlruns/drift_snapshot.json`. The drift "
        "computation runs as part of the monitoring loop in "
        "`src.monitoring.drift`. After the first run completes, refresh this page."
    )

    # Render a demonstration view with synthetic data so the page is not
    # blank when a recruiter or interviewer is browsing the project.
    st.divider()
    st.subheader("Demonstration view (synthetic data)")
    st.caption(
        "The chart below is generated from synthetic data to illustrate the "
        "page's output. The real visualisation reads from the snapshot file."
    )

    demo_psi = pd.DataFrame(
        {
            "feature": [
                "src_24h_sub_threshold_share",
                "src_24h_amount_sum",
                "dst_24h_pagerank",
                "entity_in_out_amount_ratio_24h",
                "src_24h_round_amount_share",
                "src_24h_txn_count",
                "edge_novelty_24h",
                "supervised_score",
            ],
            "psi": [0.42, 0.31, 0.18, 0.15, 0.09, 0.07, 0.05, 0.12],
        }
    ).sort_values("psi", ascending=True)

    severity_thresholds = (0.10, 0.25)
    demo_psi["severity"] = demo_psi["psi"].apply(
        lambda v: "regulator-relevant" if v >= severity_thresholds[1]
        else "warning" if v >= severity_thresholds[0]
        else "monitor"
    )

    severity_colour_map = {
        "monitor": "#22c55e",
        "warning": "#f59e0b",
        "regulator-relevant": "#ef4444",
    }
    fig = px.bar(
        demo_psi,
        x="psi",
        y="feature",
        color="severity",
        color_discrete_map=severity_colour_map,
        orientation="h",
        title="PSI by feature (synthetic demonstration)",
        labels={"psi": "Population Stability Index", "feature": ""},
    )
    fig.add_vline(x=severity_thresholds[0], line_dash="dash", line_color="#f59e0b")
    fig.add_vline(x=severity_thresholds[1], line_dash="dash", line_color="#ef4444")
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        **Severity interpretation:**

        - **PSI < 0.10 (monitor)** — distribution shift is within normal
          variation. No action required.
        - **0.10 ≤ PSI < 0.25 (warning)** — meaningful shift. Model-team
          review recommended; check upstream data source for breakage.
        - **PSI ≥ 0.25 (regulator-relevant)** — distribution has shifted
          enough that model decisions are likely affected. Escalate to
          model risk; consider retraining or rolling back.
        """
    )

    st.stop()

with snapshot_path.open() as fh:
    snapshot = json.load(fh)

# ----- Snapshot metadata -------------------------------------------------
generated_at = snapshot.get("generated_at_utc", "unknown")
reference_window = snapshot.get("reference_window", "unknown")
target_window = snapshot.get("target_window", "unknown")

st.markdown(
    f"**Snapshot generated:** {generated_at}  \n"
    f"**Reference window:** {reference_window}  \n"
    f"**Target window:** {target_window}"
)

st.divider()

# ----- Per-feature PSI ---------------------------------------------------
feature_drift = snapshot.get("feature_drift", [])
if feature_drift:
    drift_frame = pd.DataFrame(feature_drift)
    drift_frame = drift_frame.sort_values("psi", ascending=True)

    severity_colour_map = {
        "monitor": "#22c55e",
        "warning": "#f59e0b",
        "regulator-relevant": "#ef4444",
    }

    fig = px.bar(
        drift_frame,
        x="psi",
        y="feature",
        color="severity",
        color_discrete_map=severity_colour_map,
        orientation="h",
        title="PSI by feature",
        labels={"psi": "Population Stability Index", "feature": ""},
    )
    fig.update_layout(height=max(400, 28 * len(drift_frame)))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        drift_frame[["feature", "psi", "severity"]],
        use_container_width=True,
        hide_index=True,
    )

# ----- Prediction distribution drift -------------------------------------
prediction_drift = snapshot.get("prediction_drift")
if prediction_drift:
    st.subheader("Prediction-score drift")
    psi = float(prediction_drift.get("psi", 0))
    severity = prediction_drift.get("severity", "monitor")
    cols = st.columns(2)
    cols[0].metric("PSI", f"{psi:.3f}")
    cols[1].metric("Severity", severity)
