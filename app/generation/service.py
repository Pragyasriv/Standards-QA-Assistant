"""Generate a grounded answer from an approved guardrail result."""

from __future__ import annotations

import logging
import time

from app.core.config import get_settings
from app.generation.citations import validate_draft
from app.generation.client import GenerationClient, GroqGenerationClient
from app.generation.errors import (
    GenerationError,
    GenerationInvocationError,
    GenerationRejected,
    GenerationValidationError,
    public_error_text,
)
from app.generation.models import GenerationMetadata, GenerationResponse, ResolvedCitation
from app.generation.parse import assistant_text_candidates, message_content_shape, parse_model_output
from app.generation.prompt import build_messages
from app.generation.retry import is_complete_list_fidelity_retry_eligible
from app.generation.settings import GenerationSettings
from app.guardrails.models import GuardrailResponse

logger = logging.getLogger(__name__)

_RESPONSE_PREVIEW_LIMIT = 1500


def _log_model_response_diagnostic(
    raw: object,
    settings: GenerationSettings,
    *,
    attempt: int = 1,
) -> None:
    """TEMPORARY: log response shape in non-production for local debugging. Remove before release."""
    if get_settings().app_env == "production":
        return
    content_type, block_types = message_content_shape(raw)
    candidates = assistant_text_candidates(raw)
    candidate_lengths = [len(item) for item in candidates]
    preview = candidates[0] if candidates else ""
    if len(preview) > _RESPONSE_PREVIEW_LIMIT:
        preview = preview[:_RESPONSE_PREVIEW_LIMIT] + "…"
    metadata = getattr(raw, "response_metadata", None)
    finish_reason: str | None = None
    token_usage: object = None
    if isinstance(metadata, dict):
        finish_reason = metadata.get("finish_reason") or metadata.get("stop_reason")
        if not isinstance(finish_reason, str):
            finish_reason = None
        token_usage = metadata.get("token_usage") or metadata.get("usage")
    logger.info(
        "TEMPORARY DIAGNOSTIC model_response type=%s attempt=%s content_type=%s "
        "block_types=%s candidate_lengths=%s finish_reason=%s token_usage=%s preview=%s",
        type(raw).__name__,
        attempt,
        content_type,
        block_types,
        candidate_lengths,
        finish_reason,
        token_usage,
        public_error_text(RuntimeError(preview), settings.configured_key()),
    )


def _log_validation_draft_diagnostic(draft: object, safe_reason: str) -> None:
    """TEMPORARY: log rejected drafts in non-production for local debugging. Remove before release."""
    if get_settings().app_env == "production":
        return
    from app.generation.models import ModelDraft

    if not isinstance(draft, ModelDraft):
        return
    logger.info(
        "TEMPORARY DIAGNOSTIC validation_draft abstained=%s cited_chunk_ids=%s "
        "validation_reason=%s answer=%s",
        draft.abstained,
        draft.cited_chunk_ids,
        safe_reason,
        draft.answer,
    )


class GenerationService:
    """Answer from approved evidence. This service does not retrieve chunks."""

    def __init__(
        self,
        settings: GenerationSettings | None = None,
        client: GenerationClient | None = None,
    ) -> None:
        self.settings = settings or GenerationSettings()
        self._client = client

    def generate(self, question: str, guardrail: GuardrailResponse) -> GenerationResponse:
        """Generate from the question and the evidence the guardrail already approved."""
        self._require_approved(guardrail)
        chunks = list(guardrail.retrieved_chunks)
        client = self._resolve_client()
        messages = build_messages(question, chunks)
        started = time.perf_counter()
        attempt = 1
        retry_triggered = False
        draft = None
        status: str | None = None
        reason: str | None = None
        citations: list[ResolvedCitation] = []

        while attempt <= 2:
            draft = None
            try:
                raw = client.invoke(messages)
            except GenerationError:
                raise
            except Exception as exc:
                self._log_failure(len(chunks), exc, attempt=attempt)
                raise GenerationInvocationError(
                    f"The configured model {self.settings.configured_model() or '(unset)'} "
                    "could not generate an answer. "
                    + public_error_text(exc, self.settings.configured_key())
                ) from exc
            self._reject_model_switch(raw)
            _log_model_response_diagnostic(raw, self.settings, attempt=attempt)
            try:
                draft = parse_model_output(raw)
                status, reason, citations = validate_draft(draft, chunks, question)
                logger.info(
                    "Generation attempt outcome model=%s evidence_chunks=%s attempt=%s "
                    "retry_triggered=%s outcome=succeeded",
                    self.settings.configured_model() or "(unset)",
                    len(chunks),
                    attempt,
                    retry_triggered,
                )
                break
            except GenerationValidationError as exc:
                safe_reason = public_error_text(exc, self.settings.configured_key())
                phase = "validation" if draft is not None else "parse"
                logger.info(
                    "Generation output rejected model=%s evidence_chunks=%s attempt=%s "
                    "retry_triggered=%s phase=%s validation_reason=%s",
                    self.settings.configured_model() or "(unset)",
                    len(chunks),
                    attempt,
                    retry_triggered,
                    phase,
                    safe_reason,
                )
                if draft is not None:
                    _log_validation_draft_diagnostic(draft, safe_reason)
                if (
                    attempt == 1
                    and draft is not None
                    and is_complete_list_fidelity_retry_eligible(exc, question)
                ):
                    retry_triggered = True
                    logger.info(
                        "Generation retry triggered model=%s evidence_chunks=%s attempt=1",
                        self.settings.configured_model() or "(unset)",
                        len(chunks),
                    )
                    messages = build_messages(
                        question, chunks, validation_feedback=safe_reason
                    )
                    attempt = 2
                    continue
                logger.info(
                    "Generation attempt outcome model=%s evidence_chunks=%s attempt=%s "
                    "retry_triggered=%s outcome=rejected",
                    self.settings.configured_model() or "(unset)",
                    len(chunks),
                    attempt,
                    retry_triggered,
                )
                raise

        if draft is None or status is None:
            raise GenerationValidationError(
                "The model did not return valid structured output."
            )

        latency_ms = (time.perf_counter() - started) * 1000
        response = GenerationResponse(
            question=question,
            answer=draft.answer.strip() or (reason or ""),
            status=status,  # type: ignore[arg-type]
            citations=citations,
            abstention_reason=reason,
            model=self.settings.configured_model(),
            generation_metadata=GenerationMetadata(
                evidence_chunk_count=len(chunks),
                validated_citation_count=len(citations),
                abstained=status == "abstained",
                latency_ms=latency_ms,
                temperature=self.settings.groq_temperature,
            ),
        )
        logger.info(
            "Generation succeeded model=%s evidence_chunks=%s citations=%s abstained=%s latency_ms=%.1f",
            response.model,
            response.generation_metadata.evidence_chunk_count,
            response.generation_metadata.validated_citation_count,
            response.generation_metadata.abstained,
            response.generation_metadata.latency_ms,
        )
        return response

    def _require_approved(self, guardrail: GuardrailResponse) -> None:
        if guardrail.status != "approved":
            raise GenerationRejected(
                "Generation requires an approved guardrail result. "
                f"The result was {guardrail.status}."
            )
        if not guardrail.retrieved_chunks:
            raise GenerationRejected(
                "The approved guardrail result did not include evidence chunks."
            )

    def _resolve_client(self) -> GenerationClient:
        if self._client is None:
            self._client = GroqGenerationClient(self.settings)
        return self._client

    def _reject_model_switch(self, raw: object) -> None:
        configured = self.settings.configured_model()
        metadata = getattr(raw, "response_metadata", None)
        if not isinstance(metadata, dict):
            return
        returned = metadata.get("model_name") or metadata.get("model")
        if isinstance(returned, str) and returned.strip() and returned.strip() != configured:
            raise GenerationInvocationError(
                f"The provider returned model {returned.strip()}, "
                f"which is not the configured model {configured}."
            )

    def _log_failure(self, evidence_chunks: int, exc: Exception, *, attempt: int = 1) -> None:
        logger.info(
            "Generation failed model=%s evidence_chunks=%s attempt=%s error=%s",
            self.settings.configured_model() or "(unset)",
            evidence_chunks,
            attempt,
            public_error_text(exc, self.settings.configured_key()),
        )
