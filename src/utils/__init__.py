"""Cross-cutting utilities shared by the training and runtime layers.

This package owns concerns that do not belong to any one domain module
but are needed by several: typed configuration loading, structured
logging setup, and small helpers for things like request-ID generation
and JSON serialisation of domain dataclasses.

Modules here are deliberately small and dependency-light. They should
import from the standard library and well-established third-party
packages (Pydantic, structlog) but never from `src.api`, `src.models`,
`src.triage`, or other domain layers - keeping the dependency arrow
pointing inward avoids cyclic imports as the codebase grows.
"""
