"""Langfuse tracing for LLM calls.

The narrator's :class:`~src.triage.Narrator` consumes a Langfuse-shaped
client and emits traces on every LLM invocation. This module wraps
Langfuse construction behind a defensive boundary so callers obtain a
client when credentials are present and ``None`` otherwise - the
narrator's trace code already handles both cases.

Why behind a defensive boundary
-------------------------------
Tracing is non-load-bearing: a Langfuse outage, misconfiguration, or
SDK version mismatch must never affect the scoring path. Constructing
the client in one place lets every failure mode degrade to ``None``
exactly once, rather than ten times across the codebase.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_langfuse_client(
    *,
    public_key: str,
    secret_key: str,
    host: str = "https://cloud.langfuse.com",
) -> Any | None:
    """Construct a Langfuse client or return None on any failure.

    Parameters
    ----------
    public_key : str
        Langfuse public key. Empty string means tracing is disabled.
    secret_key : str
        Langfuse secret key. Empty string means tracing is disabled.
    host : str
        Langfuse host URL. Defaults to the managed cloud endpoint.

    Returns
    -------
    object | None
        Initialised Langfuse client, or ``None`` if credentials are
        missing or initialisation failed.
    """
    if not (public_key and secret_key):
        return None
    try:
        # Imported lazily because the langfuse package's import-time
        # initialisation does I/O (SDK version probe) that we do not
        # want to pay if tracing is disabled.
        from langfuse import Langfuse

        return Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    except Exception as exc:  # noqa: BLE001 - tracing is best-effort
        logger.warning(
            "Langfuse initialisation failed; continuing without tracing. %s", exc
        )
        return None
