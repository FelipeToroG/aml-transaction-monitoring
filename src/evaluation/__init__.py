"""Evaluation utilities for cost-aware AML model selection.

Banking ML evaluation differs from academic evaluation in one decisive
way: the cost matrix is asymmetric and known. False negatives cost
laundered dollars plus regulatory exposure; false positives cost
investigator time. This package implements metrics that respect that
asymmetry plus the operational simulation that proves the model is
deployable, not just accurate.

Public surface
--------------
The training driver and the API both import from here. The module
boundaries are deliberate:

* ``metrics`` — cost-weighted Precision@k objective, the ``CostMatrix``
  configuration dataclass, basic Precision@k.
* ``investigator_simulator`` — discrete-event simulation of an
  investigator queue with configurable analyst pool and SLA targets.
* ``calibration`` — reliability curves, Brier score, expected
  calibration error.
* ``reports`` — Markdown and JSON eval-report generators consumed by
  ``scripts/update_results.py`` and the audit log.
"""

from src.evaluation.calibration import (
    ReliabilityCurve,
    brier_score,
    expected_calibration_error,
    reliability_curve,
)
from src.evaluation.investigator_simulator import (
    SimulationConfig,
    SimulationResult,
    simulate_investigator_queue,
)
from src.evaluation.metrics import (
    CostMatrix,
    cost_weighted_precision_at_k,
    precision_at_k,
)
from src.evaluation.reports import (
    build_audit_snapshot,
    build_audit_snapshot_json,
    build_results_markdown,
)

__all__ = [
    "CostMatrix",
    "ReliabilityCurve",
    "SimulationConfig",
    "SimulationResult",
    "brier_score",
    "build_audit_snapshot",
    "build_audit_snapshot_json",
    "build_results_markdown",
    "cost_weighted_precision_at_k",
    "expected_calibration_error",
    "precision_at_k",
    "reliability_curve",
    "simulate_investigator_queue",
]
