"""Outbound webhook delivery with bounded retry.

Alerts above the tier-3 threshold dispatch a Slack-compatible webhook
so on-call investigators are paged immediately. Delivery is
fire-and-forget with respect to the scoring path: a webhook failure
must not fail the alert. The retry envelope, the timeout, and the
exponential backoff are bounded so a misbehaving webhook target does
not consume unbounded resources.

Design properties
-----------------
1. **Fire-and-forget**: the scoring path's HTTP response does not
   block on webhook delivery. The webhook coroutine is scheduled
   onto FastAPI's background-task system or via ``asyncio.create_task``.
2. **Bounded retry**: tenacity is configured for at most ``max_retries``
   attempts with exponential backoff capped at ``backoff_max_seconds``.
3. **Operator visibility**: every delivery (success or final failure)
   writes an audit-log event so the operator dashboard can show
   webhook delivery health without parsing application logs.
4. **Slack-compatible payload**: the payload structure matches Slack's
   incoming-webhook spec (``blocks`` for rich formatting). Microsoft
   Teams, Mattermost, and most other webhook receivers accept the
   same shape via their compatibility modes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.observability.metrics import record_webhook_delivery

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WebhookDeliveryResult:
    """Outcome of a single webhook dispatch attempt.

    Persisted to the audit log so the operator dashboard can render
    delivery health over time without re-scraping application logs.
    """

    delivered: bool
    attempts: int
    last_status_code: int | None
    error_message: str | None


class WebhookClient:
    """Async client for outbound Slack-compatible webhook delivery."""

    def __init__(
        self,
        *,
        url: str,
        enabled: bool,
        max_retries: int = 3,
        backoff_base_seconds: float = 1.0,
        backoff_max_seconds: float = 8.0,
        request_timeout_seconds: float = 5.0,
    ) -> None:
        self.url = url
        self.enabled = enabled and bool(url)
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self.request_timeout_seconds = request_timeout_seconds

    async def dispatch_alert_notification(
        self,
        *,
        alert_id: str,
        transaction_id: str,
        tier: str,
        risk_score: float,
        narrative_summary: str | None,
        suspected_typology: str | None,
    ) -> WebhookDeliveryResult:
        """Dispatch a new-alert notification.

        Returns a structured result so the caller can log the outcome
        to the audit log. No exception is raised on final failure - 
        delivery failures are an expected operational condition.
        """
        if not self.enabled:
            # Disabled-by-config is a normal state, not an error. Return
            # a delivered=False result with no attempts so the audit log
            # records the no-op decision. Metric is not incremented for
            # disabled deliveries - they are not failures.
            return WebhookDeliveryResult(
                delivered=False,
                attempts=0,
                last_status_code=None,
                error_message="Webhook delivery disabled by configuration.",
            )

        payload = self._build_payload(
            alert_id=alert_id,
            transaction_id=transaction_id,
            tier=tier,
            risk_score=risk_score,
            narrative_summary=narrative_summary,
            suspected_typology=suspected_typology,
        )

        attempts = 0
        last_status: int | None = None
        last_error: str | None = None

        try:
            # AsyncRetrying gives us the tenacity policy in async form.
            # ``retry_if_exception_type`` covers both network errors and
            # the explicit raise in the loop body for non-2xx responses.
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(
                    multiplier=self.backoff_base_seconds,
                    max=self.backoff_max_seconds,
                ),
                retry=retry_if_exception_type((httpx.HTTPError, _RetryableWebhookError)),
                reraise=True,
            ):
                with attempt:
                    attempts += 1
                    async with httpx.AsyncClient(
                        timeout=self.request_timeout_seconds
                    ) as client:
                        response = await client.post(self.url, json=payload)
                    last_status = response.status_code

                    if response.status_code >= 500:
                        # 5xx is retryable; raise so tenacity reschedules.
                        raise _RetryableWebhookError(
                            f"Webhook returned {response.status_code}; will retry."
                        )
                    if response.status_code >= 400:
                        # 4xx is a client error - likely a misconfigured
                        # URL. Retrying will not help; surface and stop.
                        last_error = (
                            f"Webhook target returned {response.status_code}: "
                            f"{response.text[:200]}"
                        )
                        record_webhook_delivery(delivered=False)
                        return WebhookDeliveryResult(
                            delivered=False,
                            attempts=attempts,
                            last_status_code=last_status,
                            error_message=last_error,
                        )

            record_webhook_delivery(delivered=True)
            return WebhookDeliveryResult(
                delivered=True,
                attempts=attempts,
                last_status_code=last_status,
                error_message=None,
            )

        except RetryError as exc:
            last_error = f"All {attempts} attempts failed: {exc}"
        except _RetryableWebhookError as exc:
            last_error = str(exc)
        except httpx.HTTPError as exc:
            last_error = f"HTTP error during webhook delivery: {exc}"

        logger.warning(
            "Webhook delivery failed after %d attempts for alert %s: %s",
            attempts,
            alert_id,
            last_error,
        )
        record_webhook_delivery(delivered=False)
        return WebhookDeliveryResult(
            delivered=False,
            attempts=attempts,
            last_status_code=last_status,
            error_message=last_error,
        )

    @staticmethod
    def _build_payload(
        *,
        alert_id: str,
        transaction_id: str,
        tier: str,
        risk_score: float,
        narrative_summary: str | None,
        suspected_typology: str | None,
    ) -> dict[str, Any]:
        """Construct the Slack-compatible block payload."""
        summary_text = narrative_summary or "Triage pending - review in dashboard."
        typology_text = (
            f"\n*Suspected typology:* {suspected_typology}"
            if suspected_typology
            else ""
        )
        header_text = f"New AML Alert ({tier.replace('_', ' ').title()})"

        return {
            "text": header_text,
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": header_text},
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Alert ID:*\n`{alert_id}`"},
                        {"type": "mrkdwn", "text": f"*Transaction ID:*\n`{transaction_id}`"},
                        {"type": "mrkdwn", "text": f"*Risk Score:*\n{risk_score:.4f}"},
                        {"type": "mrkdwn", "text": f"*Tier:*\n{tier}"},
                    ],
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Summary:*\n{summary_text}{typology_text}",
                    },
                },
            ],
        }


class _RetryableWebhookError(Exception):
    """Internal marker exception for retryable HTTP responses (5xx)."""
