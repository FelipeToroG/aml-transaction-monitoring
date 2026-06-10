"""Severity-threshold configuration for drift and fairness monitoring.

Centralises the cut points that classify a numeric metric (PSI, FPR
parity gap, demographic parity gap) into one of three operational
severity bands: ``monitor`` (normal variation, no action),
``warning`` (model-team review recommended), or ``regulator-relevant``
(escalate to model risk management; consider rollback or retrain).

Why three bands
---------------
Bank model risk management practice (US SR 11-7; equivalent EU/UK
guidance) routinely groups model monitoring signals into three
severity tiers that map to distinct operational responses. Two tiers
is too coarse to distinguish "watch this" from "act now"; four tiers
becomes hard to keep distinct. Three is the convention every operator
already knows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

Severity = Literal["monitor", "warning", "regulator-relevant"]


@dataclass(frozen=True, slots=True)
class SeverityThresholds:
    """Threshold tiers for a single metric.

    Attributes
    ----------
    monitor_max : float
        Strict upper bound of the ``monitor`` band. Values strictly
        below this are normal variation.
    warning_max : float
        Strict upper bound of the ``warning`` band. Values at or above
        ``monitor_max`` and below ``warning_max`` are warnings; values
        at or above ``warning_max`` are regulator-relevant.
    metric_name : str
        Display name embedded in audit-log events and snapshot rows.
    """

    monitor_max: float
    warning_max: float
    metric_name: str

    def classify(self, value: float) -> Severity:
        """Return the severity band a metric value falls into."""
        if value < self.monitor_max:
            return "monitor"
        if value < self.warning_max:
            return "warning"
        return "regulator-relevant"


# ---------------------------------------------------------------------
# Calibrated default thresholds
# ---------------------------------------------------------------------
# These thresholds match the conventions in the academic monitoring
# literature and the published guidance from major bank
# model-risk-management functions. Operators override per institution
# if their risk appetite differs.

PSI_THRESHOLDS: Final[SeverityThresholds] = SeverityThresholds(
    monitor_max=0.10,
    warning_max=0.25,
    metric_name="population_stability_index",
)

# Maximum absolute difference in alert rate across segments before a
# disparate-impact concern is raised. 5 percentage points is the
# convention in US fair-lending guidance (4/5 rule context); we use
# the same calibration for AML alert-rate parity because the auditor
# perspective is similar.
DEMOGRAPHIC_PARITY_THRESHOLDS: Final[SeverityThresholds] = SeverityThresholds(
    monitor_max=0.02,
    warning_max=0.05,
    metric_name="demographic_parity_gap",
)

# False-positive-rate parity gap. The threshold is tighter than the
# demographic-parity gap because false positives directly consume
# investigator time - a 5pp FPR gap is operationally significant in
# a way a 5pp alert-rate gap may not be (the latter could reflect
# legitimate underlying risk differences).
FPR_PARITY_THRESHOLDS: Final[SeverityThresholds] = SeverityThresholds(
    monitor_max=0.015,
    warning_max=0.030,
    metric_name="fpr_parity_gap",
)

# True-positive-rate parity gap. Measures equal opportunity: does the
# model catch laundering at similar rates across segments? Bounded
# tighter than FPR parity because under-detection in any segment is a
# direct compliance failure.
TPR_PARITY_THRESHOLDS: Final[SeverityThresholds] = SeverityThresholds(
    monitor_max=0.05,
    warning_max=0.10,
    metric_name="tpr_parity_gap",
)
