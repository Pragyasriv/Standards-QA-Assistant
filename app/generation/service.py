"""Generate a grounded answer from an approved guardrail result."""

from __future__ import annotations

import logging
import time

from app.generation.citations import validate_draft
from app.generation.client import GenerationClient, GroqGenerationClient
from app.generation.errors import (
    GenerationError,
    GenerationInvocationError,
    GenerationRejected,
    GenerationValidationError,
    public_error_text,
)
from app.generation.models import GenerationMetadata, GenerationResponse
from app.generation.parse import parse_model_output
from app.generation.prompt import build_messages
from app.generation.settings import GenerationSettings
from app.guardrails.models import GuardrailResponse

logger = logging.getLogger(__name__)


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
        try:
            raw = client.invoke(messages)
        except GenerationError:
            raise
        except Exception as exc:
            self._log_failure(len(chunks), exc)
            raise GenerationInvocationError(
                f"The configured model {self.settings.configured_model() or '(unset)'} "
                "could not generate an answer. "
                + public_error_text(exc, self.settings.configured_key())
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        self._reject_model_switch(raw)
        draft = parse_model_output(raw)
        try:
            status, reason, citations = validate_draft(draft, chunks, question)
        except GenerationValidationError:
            logger.info(
                "Generation output rejected model=%s evidence_chunks=%s",
                self.settings.configured_model() or "(unset)",
                len(chunks),
            )
            raise
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

    def _log_failure(self, evidence_chunks: int, exc: Exception) -> None:
        logger.info(
            "Generation failed model=%s evidence_chunks=%s error=%s",
            self.settings.configured_model() or "(unset)",
            evidence_chunks,
            public_error_text(exc, self.settings.configured_key()),
        )
