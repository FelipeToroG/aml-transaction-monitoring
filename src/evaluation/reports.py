"""Evaluation report generators.

Takes the training summary JSON produced by ``src.models.train`` and
emits two artifact types:

* A Markdown table block to be injected into the README's results
  section. Consumed by ``scripts/update_results.py``.
* A JSON-safe audit snapshot suitable for inclusion in the database's
  audit log alongside the model artifact reference.

Both views are constructed from the same underlying summary so the
README and the audit log can never disagree about what the model's
performance was at training time.
"""

from __future__ import annotations

import json
from typing import Any

# Map from internal family code to the display name used in the README.
# Centralised here so the README's per-family comparison table renders
# consistently across runs without per-call name mapping at the call
# site.
_FAMILY_DISPLAY_NAMES: dict[str, str] = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "isolation_forest": "Isolation Forest",
}


def build_results_markdown(summary: dict[str, Any]) -> str:
    """Render the README results table from a training summary.

    The output is the full Markdown block that replaces the content
    between the ``<!-- RESULTS:START -->`` and ``<!-- RESULTS:END -->``
    markers in the README. Three subsections:

    1. Headline results table.
    2. Per-family comparison table.
    3. Cost-matrix transparency block.
    """
    test_eval = summary["test"]
    val_eval = summary["val"]
    winning_family = summary["winning_family"]
    threshold = summary["decision_threshold"]
    cost_matrix = summary["cost_matrix"]
    families = summary.get("all_families", {})

    headline = _build_headline_table(
        winning_family=winning_family,
        test_eval=test_eval,
        val_eval=val_eval,
        threshold=threshold,
        n_families=len(families),
    )
    family_table = _build_family_comparison_table(families=families)
    cost_block = _build_cost_matrix_block(cost_matrix=cost_matrix)

    return (
        "### Headline results (test fold)\n\n"
        f"{headline}\n\n"
        "### Per-family comparison (validation fold)\n\n"
        f"{family_table}\n\n"
        "### Cost matrix used for selection\n\n"
        f"{cost_block}\n"
    )


def build_audit_snapshot(summary: dict[str, Any]) -> dict[str, Any]:
    """Build the JSON-safe audit snapshot for the persistence layer.

    The snapshot is the structured record stored alongside the model
    artifact reference in the audit log. Regulators reviewing a
    historical alert may request this snapshot for the model that
    produced the alert; the structure must therefore be both human
    readable and machine queryable.
    """
    return {
        "service_version": summary.get("service_version"),
        "run_timestamp_utc": summary.get("run_timestamp_utc"),
        "winning_family": summary.get("winning_family"),
        "winning_hyperparameters": summary.get("winning_hyperparameters"),
        "all_family_objectives": {
            name: float(info["objective"])
            for name, info in summary.get("all_families", {}).items()
        },
        "anomaly": summary.get("anomaly_metadata"),
        "decision_threshold": float(summary.get("decision_threshold", 0.0)),
        "test_metrics": {
            "objective": float(summary["test"]["objective_cost_per_investigator_hour_usd"]),
            "precision_at_k": float(summary["test"]["precision_at_k"]),
            "true_positives": int(summary["test"]["true_positives"]),
            "false_positives": int(summary["test"]["false_positives"]),
            "false_negatives": int(summary["test"]["false_negatives"]),
            "k": int(summary["test"]["k"]),
            "total_dollar_cost_usd": float(summary["test"]["total_dollar_cost_usd"]),
        },
        "cost_matrix": summary.get("cost_matrix"),
        "split": summary.get("split"),
    }


def build_audit_snapshot_json(summary: dict[str, Any], *, indent: int = 2) -> str:
    """Convenience wrapper: ``build_audit_snapshot`` rendered as JSON."""
    return json.dumps(build_audit_snapshot(summary), indent=indent, default=str)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _build_headline_table(
    *,
    winning_family: str,
    test_eval: dict[str, Any],
    val_eval: dict[str, Any],
    threshold: float,
    n_families: int,
) -> str:
    """Render the headline test-set table."""
    family_display = _FAMILY_DISPLAY_NAMES.get(winning_family, winning_family)
    tp = int(test_eval["true_positives"])
    fp = int(test_eval["false_positives"])
    fn = int(test_eval["false_negatives"])
    total_positives = tp + fn
    recall = (tp / total_positives) if total_positives > 0 else 0.0

    rows = [
        ("Winning model", f"{family_display} (selected on cost-weighted Precision@k)"),
        (
            "Test set objective",
            f"${-test_eval['objective_cost_per_investigator_hour_usd']:,.2f} cost per investigator-hour (negated)",
        ),
        (
            "Test Precision @ k",
            f"{test_eval['precision_at_k'] * 100:.1f}% (k = {int(test_eval['k']):,} alerts)",
        ),
        (
            "Test recall (fraud caught)",
            f"{recall * 100:.1f}% ({tp:,} of {total_positives:,})",
        ),
        ("Test true positives", f"{tp:,}"),
        ("Test false positives", f"{fp:,}"),
        ("Test false negatives", f"{fn:,}"),
        (
            "Test total dollar cost",
            f"${test_eval['total_dollar_cost_usd']:,.2f}",
        ),
        (
            "Validation objective (tuning fold)",
            f"${-val_eval['objective_cost_per_investigator_hour_usd']:,.2f} cost per investigator-hour (negated)",
        ),
        ("Decision threshold", f"{threshold:.4f}"),
        (
            "Models evaluated",
            f"{n_families} families across the Optuna sweep",
        ),
    ]
    return _render_two_column_table(rows, header=("What", "Value"))


def _build_family_comparison_table(families: dict[str, Any]) -> str:
    """Render the per-family comparison table.

    Sorted by objective descending so the winner sits at the top.
    """
    if not families:
        return "_No family comparison available - single-family run._"

    sorted_families = sorted(
        families.items(),
        key=lambda kv: kv[1]["objective"],
        reverse=True,
    )

    lines: list[str] = []
    lines.append("| Rank | Family | Validation objective (USD/hr) |")
    lines.append("|------|--------|-------------------------------|")
    for rank, (name, info) in enumerate(sorted_families, start=1):
        display_name = _FAMILY_DISPLAY_NAMES.get(name, name)
        objective_usd = -float(info["objective"])
        lines.append(f"| {rank} | {display_name} | ${objective_usd:,.2f} |")
    return "\n".join(lines)


def _build_cost_matrix_block(cost_matrix: dict[str, Any]) -> str:
    """Render the cost-matrix block showing what selection optimised against."""
    rows = [
        ("Daily review capacity (k)", f"{int(cost_matrix['k_per_day']):,} alerts"),
        (
            "False-negative cost",
            f"${float(cost_matrix['false_negative_cost_usd']):,.2f} per missed alert",
        ),
        (
            "False-positive cost",
            f"${float(cost_matrix['false_positive_cost_usd']):,.2f} per investigator-cleared alert",
        ),
    ]
    return _render_two_column_table(rows, header=("Parameter", "Value"))


def _render_two_column_table(
    rows: list[tuple[str, str]], *, header: tuple[str, str]
) -> str:
    """Render a two-column Markdown table.

    Internal helper. Centralising table rendering ensures every section
    of the README results block has consistent column widths and
    alignment.
    """
    left_label, right_label = header
    lines = [
        f"| {left_label} | {right_label} |",
        "|---|---|",
    ]
    for left, right in rows:
        lines.append(f"| {left} | {right} |")
    return "\n".join(lines)
