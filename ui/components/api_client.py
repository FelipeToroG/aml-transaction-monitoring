"""HTTP client for the AML monitoring API.

The Streamlit UI is a thin client over the FastAPI service. Every
page resolves its data through the methods on this class, never
directly through the database or filesystem. That separation keeps
the UI deployable in environments where it can reach the API but not
the underlying data store (the standard production deployment shape).

Caching strategy
----------------
``@st.cache_resource`` is used for the client itself (a single
connection pool per Streamlit session). ``@st.cache_data`` is used at
the call-site level on individual fetches when the UI page wants the
result for the same input to be reused on a re-render. Cache TTLs are
deliberately short (15–30 s) because alert data is operationally hot.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
import streamlit as st


@dataclass(slots=True)
class APIError(Exception):
    """Structured error raised on a non-2xx response.

    Carries the status code and decoded body so calling pages can
    render the operator-relevant detail rather than a generic banner.
    """

    status_code: int
    detail: str

    def __str__(self) -> str:
        return f"API error {self.status_code}: {self.detail}"


class AMLAPIClient:
    """Typed wrapper around the AML FastAPI service.

    Every method either returns a typed dict mirroring the response
    schema or raises :class:`APIError`. The dict shape is defined by
    ``src.api.schemas`` and stable across releases via the OpenAPI
    contract.
    """

    def __init__(self, base_url: str, *, timeout_seconds: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    # ----- Health / metrics --------------------------------------------

    def get_health(self) -> dict[str, Any]:
        """Return the API's health envelope."""
        return self._get("/health")

    # ----- Alerts -------------------------------------------------------

    def list_alerts(
        self,
        *,
        status: str | None = None,
        tier: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Return a paginated alert listing."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status is not None:
            params["status"] = status
        if tier is not None:
            params["tier"] = tier
        return self._get("/alerts", params=params)

    # ----- Triage / feedback --------------------------------------------

    def post_triage(self, alert_id: str, *, use_eval_model: bool = False) -> dict[str, Any]:
        """Request a fresh narrative for an alert."""
        return self._post(
            "/triage",
            json={"alert_id": alert_id, "use_eval_model": use_eval_model},
        )

    def post_feedback(
        self,
        *,
        alert_id: str,
        investigator_id: str,
        disposition: str,
        justification: str | None = None,
    ) -> dict[str, Any]:
        """Record an investigator disposition for an alert."""
        payload: dict[str, Any] = {
            "alert_id": alert_id,
            "investigator_id": investigator_id,
            "disposition": disposition,
        }
        if justification is not None:
            payload["justification"] = justification
        return self._post("/feedback", json=payload)

    # ----- Internal HTTP plumbing ---------------------------------------

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = httpx.get(
            f"{self.base_url}{path}",
            params=params,
            timeout=self._timeout,
        )
        self._raise_for_status(response)
        return response.json()

    def _post(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
        response = httpx.post(
            f"{self.base_url}{path}",
            json=json,
            timeout=self._timeout,
        )
        self._raise_for_status(response)
        return response.json()

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code >= 400:
            try:
                body = response.json()
                detail = str(body.get("detail", body))
            except ValueError:
                detail = response.text
            raise APIError(status_code=response.status_code, detail=detail)


@st.cache_resource(show_spinner=False)
def get_api_client() -> AMLAPIClient:
    """Return the process-wide API client.

    The base URL is read from the ``API_BASE_URL`` environment variable
    so the UI container can target either ``http://localhost:8000``
    (local development) or the service mesh hostname (production)
    without code changes.
    """
    base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    return AMLAPIClient(base_url)
