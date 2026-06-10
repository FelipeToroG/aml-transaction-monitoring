"""Claude case-narrative narrator with evidence-bound output validation.

The narrator is the LLM-facing component of the AML system. Given an
alert and its assembled evidence bundle, it:

1. Renders the active prompt template against the evidence.
2. Calls Anthropic's Claude API with deterministic settings
   (``temperature=0``, fixed ``max_tokens``).
3. Parses the response as JSON.
4. Validates the parsed structure against the discriminated
   :class:`NarratorResult` schema.
5. If validation fails, retries with a strengthening preamble that
   names the specific validation error. After
   ``max_validation_retries`` exhausted retries, returns a
   ``schema_failure`` refusal - never a silent dropout.
6. Verifies that every citation in the parsed narrative refers to an
   identifier present in the evidence bundle. A narrative that cites
   features or transactions that do not exist in the bundle is itself
   a hallucination footprint and is downgraded to a refusal.

Cost and latency telemetry is emitted on every call. When Langfuse
credentials are configured the call is traced end-to-end with prompt,
response, latency, and token usage attached to the alert ID.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import anthropic
from pydantic import ValidationError

from src.observability.metrics import record_llm_call
from src.triage.prompts import (
    PROMPT_REGISTRY,
    VALIDATION_RETRY_PREAMBLE,
    PromptTemplate,
    get_prompt,
    render_output_schema_doc,
)
from src.triage.schemas import (
    CaseNarrative,
    Citation,
    FeatureCitation,
    NarratorResult,
    RefusalReason,
    RiskIndicator,
    TransactionCitation,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """Structured evidence passed to the narrator for a single alert.

    The bundle is assembled by the API's triage route from the alert's
    feature vector, the entity's recent activity, and the top
    contributing features (the ones that pushed the score above
    threshold). Every identifier the narrator may cite must be present
    in this bundle; the narrator validates citations against it.

    Attributes
    ----------
    alert_id : str
        Stable identifier for the alert.
    risk_score : float
        Combined ensemble score.
    anomaly_score : float
        Anomaly component score.
    supervised_score : float
        Supervised component probability.
    tier : str
        Alert tier from ``configs/alert_thresholds.yaml``.
    generated_at_iso : str
        ISO-8601 timestamp at which the alert was generated.
    transaction : dict
        The scored transaction's structured representation.
    source_activity : dict
        Recent-history snapshot for the source entity.
    destination_activity : dict
        Recent-history snapshot for the destination entity.
    triggered_features : list[dict]
        Top-contributing features. Each dict has at minimum
        ``feature_name`` and ``observed_value`` keys.
    """

    alert_id: str
    risk_score: float
    anomaly_score: float
    supervised_score: float
    tier: str
    generated_at_iso: str
    transaction: dict[str, Any]
    source_activity: dict[str, Any]
    destination_activity: dict[str, Any]
    triggered_features: list[dict[str, Any]]


class Narrator:
    """High-level facade over the Anthropic client for AML case narration.

    Construct once at API startup and inject as a FastAPI dependency.
    Thread-safe for concurrent ``generate`` calls because the Anthropic
    SDK client handles its own connection pooling.

    Parameters
    ----------
    api_key : str
        Anthropic API key. Sourced from ``ANTHROPIC_API_KEY`` at the
        configuration layer; passed in explicitly here so the class
        does not reach into environment state itself (testability).
    primary_model : str
        Model used for production triage calls.
    eval_model : str
        Model used for offline evaluation runs. Cheaper and faster but
        less robust than the primary; appropriate for cost-bounded
        replay.
    max_tokens : int
        Cap on the model's output. Sized to the worst-case
        full-narrative JSON plus some slack.
    temperature : float
        Sampling temperature. 0.0 for deterministic, reproducible
        outputs - a regulatory requirement for any LLM whose output
        may end up in a SAR filing.
    max_validation_retries : int
        Number of times to re-prompt with a strengthening preamble
        before giving up and emitting a schema_failure refusal.
    prompt_version : str
        Which prompt template version to use. The narrator stamps this
        value into every result for audit traceability.
    langfuse : LangfuseClient | None
        Optional tracing client. When provided, every call is traced.
    """

    def __init__(
        self,
        *,
        api_key: str,
        primary_model: str = "claude-sonnet-4-5",
        eval_model: str = "claude-haiku-4-5",
        max_tokens: int = 1500,
        temperature: float = 0.0,
        max_validation_retries: int = 2,
        prompt_version: str = "v1.0",
        langfuse: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError(
                "Anthropic API key is required. Configure ANTHROPIC_API_KEY "
                "in your environment."
            )
        if prompt_version not in PROMPT_REGISTRY:
            raise ValueError(
                f"Unknown prompt_version {prompt_version!r}. "
                f"Available: {sorted(PROMPT_REGISTRY)}"
            )

        self._client = anthropic.Anthropic(api_key=api_key)
        self.primary_model = primary_model
        self.eval_model = eval_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_validation_retries = max_validation_retries
        self.prompt_template: PromptTemplate = get_prompt(prompt_version)
        self.langfuse = langfuse

    def generate(
        self,
        evidence: EvidenceBundle,
        *,
        use_eval_model: bool = False,
    ) -> NarratorResult:
        """Produce a case narrative or structured refusal for one alert.

        Parameters
        ----------
        evidence : EvidenceBundle
            The structured evidence to ground the narrative in.
        use_eval_model : bool
            When True, route to the cheaper eval model. Used by offline
            replay runs and the test suite to avoid burning production
            budget on non-production work.

        Returns
        -------
        NarratorResult
            Validated discriminator carrying either a CaseNarrative or
            a RefusalReason, plus telemetry metadata.
        """
        model = self.eval_model if use_eval_model else self.primary_model
        user_prompt = self._render_user_prompt(evidence)

        trace = self._maybe_start_trace(evidence=evidence, model=model)
        start = time.perf_counter()

        # Multi-attempt loop: first pass uses the canonical user prompt;
        # subsequent passes prepend the validation-retry preamble with
        # the specific validation error.
        attempt_user_prompt = user_prompt
        last_validation_error: str | None = None
        last_response_text: str | None = None
        input_tokens_total = 0
        output_tokens_total = 0

        for attempt in range(self.max_validation_retries + 1):
            try:
                response = self._client.messages.create(
                    model=model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=self.prompt_template.system_prompt,
                    messages=[{"role": "user", "content": attempt_user_prompt}],
                )
            except anthropic.APIError as exc:
                # API errors are infrastructure failures, not modeling
                # failures. Emit a refusal so the alert still flows to
                # investigators with a clear annotation rather than
                # silently disappearing into an exception path.
                latency_ms = (time.perf_counter() - start) * 1000.0
                logger.warning(
                    "Anthropic API error after %d attempts: %s",
                    attempt,
                    exc,
                )
                result = self._build_api_error_refusal(
                    evidence=evidence,
                    model=model,
                    latency_ms=latency_ms,
                    reason=str(exc),
                )
                self._record_metrics(result)
                self._maybe_finish_trace(trace=trace, result=result, raw_response=None)
                return result

            response_text = self._extract_text(response)
            input_tokens_total += getattr(response.usage, "input_tokens", 0) or 0
            output_tokens_total += getattr(response.usage, "output_tokens", 0) or 0
            last_response_text = response_text

            parsed, validation_error = self._try_parse_response(
                evidence=evidence,
                response_text=response_text,
                model=model,
                input_tokens=input_tokens_total,
                output_tokens=output_tokens_total,
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )

            if parsed is not None:
                self._record_metrics(parsed)
                self._maybe_finish_trace(trace=trace, result=parsed, raw_response=response_text)
                return parsed

            last_validation_error = validation_error
            attempt_user_prompt = (
                VALIDATION_RETRY_PREAMBLE.format(
                    validation_error=validation_error or "unspecified parse error"
                )
                + "\n\n---\n\n"
                + user_prompt
            )

        # Retries exhausted. Emit a structured schema-failure refusal.
        latency_ms = (time.perf_counter() - start) * 1000.0
        logger.warning(
            "Validation retries exhausted for alert %s; emitting schema_failure refusal. "
            "Last validation error: %s. Last response (truncated): %s",
            evidence.alert_id,
            last_validation_error,
            (last_response_text or "")[:200],
        )
        result = self._build_schema_failure_refusal(
            evidence=evidence,
            model=model,
            latency_ms=latency_ms,
            input_tokens=input_tokens_total,
            output_tokens=output_tokens_total,
            reason=last_validation_error or "Validation failure with no error captured.",
        )
        self._record_metrics(result)
        self._maybe_finish_trace(trace=trace, result=result, raw_response=last_response_text)
        return result

    # ----- Internal: prompt rendering ---------------------------------

    def _render_user_prompt(self, evidence: EvidenceBundle) -> str:
        """Format the user-template with the evidence bundle JSON.

        We render each section as ``json.dumps`` with indentation rather
        than the bundle's repr so the model receives a clean, parseable
        view that matches how it sees its own output.
        """
        return self.prompt_template.user_template.format(
            alert_id=evidence.alert_id,
            risk_score=evidence.risk_score,
            anomaly_score=evidence.anomaly_score,
            supervised_score=evidence.supervised_score,
            tier=evidence.tier,
            generated_at_iso=evidence.generated_at_iso,
            transaction_json=json.dumps(evidence.transaction, indent=2, default=str),
            source_activity_json=json.dumps(evidence.source_activity, indent=2, default=str),
            destination_activity_json=json.dumps(
                evidence.destination_activity, indent=2, default=str
            ),
            triggered_features_json=json.dumps(evidence.triggered_features, indent=2, default=str),
            output_schema_json=render_output_schema_doc(),
        )

    # ----- Internal: response handling --------------------------------

    @staticmethod
    def _extract_text(response: anthropic.types.Message) -> str:
        """Concatenate text blocks from an Anthropic message response.

        The Claude API returns content as a list of typed blocks. For
        text-only responses we expect a single text block; we
        concatenate defensively in case the model returns multiple.
        """
        parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "".join(parts).strip()

    def _try_parse_response(
        self,
        *,
        evidence: EvidenceBundle,
        response_text: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> tuple[NarratorResult | None, str | None]:
        """Parse, schema-validate, and citation-check the model output.

        Returns ``(NarratorResult, None)`` on success or
        ``(None, error_message)`` on any failure. The error message is
        used to construct the strengthening preamble on retry.
        """
        # Strip code-fence noise the model occasionally adds despite
        # explicit instructions. The defensive parse is one line; the
        # incremental complexity is well worth not failing every
        # response that has a stray "```json" wrapper.
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.lstrip("`").lstrip("json").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned.rstrip("`").strip()

        try:
            raw = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return None, f"Response is not valid JSON: {exc}"

        if not isinstance(raw, dict) or "outcome" not in raw or "payload" not in raw:
            return None, (
                "Response missing the required {outcome, payload} discriminator. "
                "Wrap the CaseNarrative or RefusalReason in the discriminator object."
            )

        outcome = raw["outcome"]
        payload = raw["payload"]

        if outcome == "narrative":
            try:
                narrative = CaseNarrative.model_validate(payload)
            except ValidationError as exc:
                return None, f"CaseNarrative validation failed: {exc.errors()}"

            citation_error = self._verify_citations(
                narrative=narrative, evidence=evidence
            )
            if citation_error is not None:
                return None, citation_error

            return (
                NarratorResult(
                    alert_id=evidence.alert_id,
                    success=True,
                    narrative=narrative,
                    refusal=None,
                    model_name=model,
                    prompt_version=self.prompt_template.version,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
                None,
            )

        if outcome == "refusal":
            try:
                refusal = RefusalReason.model_validate(payload)
            except ValidationError as exc:
                return None, f"RefusalReason validation failed: {exc.errors()}"

            return (
                NarratorResult(
                    alert_id=evidence.alert_id,
                    success=False,
                    narrative=None,
                    refusal=refusal,
                    model_name=model,
                    prompt_version=self.prompt_template.version,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
                None,
            )

        return None, f"Unknown outcome value {outcome!r}. Expected 'narrative' or 'refusal'."

    # ----- Internal: citation grounding -------------------------------

    @staticmethod
    def _verify_citations(
        *,
        narrative: CaseNarrative,
        evidence: EvidenceBundle,
    ) -> str | None:
        """Verify every citation references an identifier in the bundle.

        Returns None on success or an error string identifying the
        first invalid citation. This is the citation-grounding step
        that elevates "the model said it cited a feature" to "the
        model actually cited a feature that exists".

        Constructing the lookup sets from the bundle once and querying
        them per citation keeps the check O(n_citations) regardless of
        bundle size.
        """
        valid_feature_names: set[str] = {
            str(f.get("feature_name", ""))
            for f in evidence.triggered_features
            if "feature_name" in f
        }

        valid_transaction_ids: set[str] = set()
        if "transaction_id" in evidence.transaction:
            valid_transaction_ids.add(str(evidence.transaction["transaction_id"]))
        for activity_block in (evidence.source_activity, evidence.destination_activity):
            txn_list = activity_block.get("recent_transactions", [])
            for txn in txn_list:
                if isinstance(txn, dict) and "transaction_id" in txn:
                    valid_transaction_ids.add(str(txn["transaction_id"]))

        for indicator_idx, indicator in enumerate(narrative.risk_indicators):
            for cite_idx, citation in enumerate(indicator.citations):
                if isinstance(citation, FeatureCitation):
                    if citation.feature_name not in valid_feature_names:
                        return (
                            f"risk_indicators[{indicator_idx}].citations[{cite_idx}] "
                            f"cites feature {citation.feature_name!r} which is not in "
                            "the evidence bundle's triggered_features."
                        )
                elif isinstance(citation, TransactionCitation):
                    if citation.transaction_id not in valid_transaction_ids:
                        return (
                            f"risk_indicators[{indicator_idx}].citations[{cite_idx}] "
                            f"cites transaction_id {citation.transaction_id!r} which "
                            "is not in the evidence bundle."
                        )
                else:
                    # Defensive: pydantic should already have rejected an
                    # unknown citation_type, but keep the branch for type
                    # narrowing and as a safety net during schema changes.
                    return (
                        f"risk_indicators[{indicator_idx}].citations[{cite_idx}] "
                        f"has an unrecognised citation_type."
                    )
        return None

    # ----- Internal: refusal builders ---------------------------------

    def _record_metrics(self, result: NarratorResult) -> None:
        """Emit Prometheus metrics for one narrator call.

        Wraps the producer helper so the call sites read as a single
        intent line and the import surface for narrator.py stays
        narrow.
        """
        record_llm_call(
            model=result.model_name,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            success=result.success,
            refusal_code=result.refusal.code if result.refusal else None,
        )

    def _build_schema_failure_refusal(
        self,
        *,
        evidence: EvidenceBundle,
        model: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        reason: str,
    ) -> NarratorResult:
        """Construct a NarratorResult for the exhausted-retries case."""
        refusal = RefusalReason(
            alert_id=evidence.alert_id,
            code="schema_failure",
            reason=(
                "Automated narrative generation could not produce a schema-valid "
                "output after retries. This typically indicates an unusual "
                "evidence pattern the prompt was not designed for. "
                f"Underlying error: {reason[:300]}"
            ),
            recommended_action=(
                "Investigator review without LLM assistance. Capture the "
                "investigator's disposition reason in the feedback endpoint "
                "so the prompt-team can review whether the template needs an "
                "extension."
            ),
        )
        return NarratorResult(
            alert_id=evidence.alert_id,
            success=False,
            narrative=None,
            refusal=refusal,
            model_name=model,
            prompt_version=self.prompt_template.version,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _build_api_error_refusal(
        self,
        *,
        evidence: EvidenceBundle,
        model: str,
        latency_ms: float,
        reason: str,
    ) -> NarratorResult:
        """Construct a NarratorResult for an Anthropic API failure."""
        refusal = RefusalReason(
            alert_id=evidence.alert_id,
            code="schema_failure",
            reason=(
                "Upstream LLM provider error prevented narrative generation. "
                f"Error: {reason[:300]}"
            ),
            recommended_action=(
                "Investigator review without LLM assistance. Page the on-call "
                "platform engineer if the LLM provider error rate exceeds the "
                "configured threshold."
            ),
        )
        return NarratorResult(
            alert_id=evidence.alert_id,
            success=False,
            narrative=None,
            refusal=refusal,
            model_name=model,
            prompt_version=self.prompt_template.version,
            latency_ms=latency_ms,
            input_tokens=None,
            output_tokens=None,
        )

    # ----- Internal: Langfuse tracing ---------------------------------

    def _maybe_start_trace(
        self, *, evidence: EvidenceBundle, model: str
    ) -> Any | None:
        """Begin a Langfuse trace if the client is configured.

        Wrapped in a try-except because a tracing-layer failure must
        never bring down the scoring path. The same defensive guard
        applies in :meth:`_maybe_finish_trace`.
        """
        if self.langfuse is None:
            return None
        try:
            return self.langfuse.trace(
                name="case_narrative",
                input={"alert_id": evidence.alert_id, "model": model},
                metadata={
                    "tier": evidence.tier,
                    "risk_score": evidence.risk_score,
                    "prompt_version": self.prompt_template.version,
                },
            )
        except Exception as exc:  # noqa: BLE001 - tracing is best-effort
            logger.debug("Langfuse trace start failed; continuing without trace: %s", exc)
            return None

    def _maybe_finish_trace(
        self,
        *,
        trace: Any | None,
        result: NarratorResult,
        raw_response: str | None,
    ) -> None:
        """Attach output to the Langfuse trace if it was started."""
        if trace is None:
            return
        try:
            trace.update(
                output={
                    "success": result.success,
                    "alert_id": result.alert_id,
                    "narrative_summary": (
                        result.narrative.summary if result.narrative else None
                    ),
                    "refusal_code": (
                        result.refusal.code if result.refusal else None
                    ),
                },
                metadata={
                    "latency_ms": result.latency_ms,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "raw_response_truncated": (raw_response or "")[:1000],
                },
            )
        except Exception as exc:  # noqa: BLE001 - tracing is best-effort
            logger.debug("Langfuse trace finish failed silently: %s", exc)
