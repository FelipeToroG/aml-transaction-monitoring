"""AML Transaction Monitoring service.

This package implements the runtime and training-time machinery for a
production AML monitoring system: hybrid anomaly-plus-supervised scoring,
Claude-powered evidence-bound case narratives, an investigator feedback
loop, and live drift and fairness monitoring.

The version exposed here is the authoritative service version. It is read
at API startup and embedded in every persisted alert so production records
can be reconciled to the exact code revision that produced them.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
