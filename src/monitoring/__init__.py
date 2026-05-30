"""Production monitoring: drift detection and fairness audit.

Three monitoring concerns live here. All are non-optional for any AML
system that needs to satisfy bank model-risk-management requirements
(US SR 11-7; equivalent provisions in EU and UK guidelines):

* ``drift`` — Population Stability Index on top-k features and on the
  prediction distribution, with severity classification and snapshot
  generation.
* ``fairness`` — segment-level demographic parity, equal opportunity,
  and FPR parity computations plus per-metric parity-gap calculation.
* ``alerts`` — threshold tables that classify drift and fairness
  metric values into ``monitor`` / ``warning`` / ``regulator-relevant``
  severity bands.

Public surface
--------------
The Streamlit pages consume the JSON snapshots produced by
``generate_drift_snapshot`` and ``generate_fairness_snapshot``. The
narrator and the API do not import from this package; monitoring is
strictly a periodic / offline concern in the current architecture.
"""

from src.monitoring.alerts import (
    DEMOGRAPHIC_PARITY_THRESHOLDS,
    FPR_PARITY_THRESHOLDS,
    PSI_THRESHOLDS,
    TPR_PARITY_THRESHOLDS,
    Severity,
    SeverityThresholds,
)
from src.monitoring.drift import (
    FeatureDriftResult,
    PredictionDriftResult,
    compute_feature_drift,
    compute_prediction_drift,
    compute_psi,
    generate_drift_snapshot,
)
from src.monitoring.fairness import (
    ParityGap,
    SegmentMetrics,
    compute_parity_gaps,
    compute_segment_metrics,
    generate_fairness_snapshot,
)

__all__ = [
    "DEMOGRAPHIC_PARITY_THRESHOLDS",
    "FPR_PARITY_THRESHOLDS",
    "FeatureDriftResult",
    "PSI_THRESHOLDS",
    "ParityGap",
    "PredictionDriftResult",
    "SegmentMetrics",
    "Severity",
    "SeverityThresholds",
    "TPR_PARITY_THRESHOLDS",
    "compute_feature_drift",
    "compute_parity_gaps",
    "compute_prediction_drift",
    "compute_psi",
    "compute_segment_metrics",
    "generate_drift_snapshot",
    "generate_fairness_snapshot",
]
