"""Run one question through the same services POST /ask uses.

Retrieved chunks are kept when generation abstains. The public API drops them.
Validation failures are not retried and the rejected answer is not stored.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from app.ask.service import _VALIDATION_REASON
from app.evaluation.dataset import RagasQuestion
from app.evaluation.results import PipelineRecord, utc_now
from app.generation.errors import (
    GenerationConfigError,
    GenerationInvocationError,
    GenerationValidationError,
)
from app.generation.models import GenerationResponse
from app.guardrails.models import GuardrailResponse

_TRANSIENT = ("429", "rate limit", "timeout", "timed out", "503", "502", "overloaded", "temporarily")


class SupportsReview(Protocol):
    """Guardrail service surface used by the evaluation runner."""

    def review(self, question: object) -> GuardrailResponse:
        """Classify and, when in scope, retrieve."""


class SupportsGenerate(Protocol):
    """Generation service surface used by the evaluation runner."""

    def generate(self, question: str, guardrail: GuardrailResponse) -> GenerationResponse:
        """Generate from an approved guardrail result."""


def is_transient(exc: BaseException) -> bool:
    """True for rate limits and timeouts. Validation errors are never transient."""
    if isinstance(exc, (GenerationValidationError, GenerationConfigError)):
        return False
    text = str(exc).lower()
    return any(token in text for token in _TRANSIENT)


def run_question(
    item: RagasQuestion,
    guardrails: SupportsReview,
    generation: SupportsGenerate,
    *,
    max_retries: int = 3,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], str] = utc_now,
) -> PipelineRecord:
    """Review, then generate when the guardrail approves the question."""
    try:
        decision = guardrails.review(item.question)
    except Exception as exc:
        return _record(
            item,
            guardrail_status="pipeline_error",
            final_status="pipeline_error",
            failure_reason=_safe(exc),
            clock=clock,
        )
    contexts = [chunk.text for chunk in decision.retrieved_chunks]
    chunk_ids = [chunk.chunk_id for chunk in decision.retrieved_chunks]
    if decision.status != "approved":
        return _record(
            item,
            guardrail_status=decision.status,
            final_status=decision.status,  # type: ignore[arg-type]
            retrieved_contexts=contexts,
            retrieved_chunk_ids=chunk_ids,
            failure_reason=decision.reason,
            abstention_reason=decision.clarification_question,
            clock=clock,
        )
    attempt = 0
    while True:
        try:
            generated = generation.generate(item.question, decision)
            break
        except GenerationValidationError as exc:
            return _record(
                item,
                guardrail_status=decision.status,
                final_status="abstained",
                retrieved_contexts=contexts,
                retrieved_chunk_ids=chunk_ids,
                failure_reason=_safe(exc),
                abstention_reason=_VALIDATION_REASON,
                clock=clock,
            )
        except GenerationConfigError:
            raise
        except GenerationInvocationError as exc:
            if attempt >= max_retries or not is_transient(exc):
                return _record(
                    item,
                    guardrail_status=decision.status,
                    final_status="provider_failure",
                    retrieved_contexts=contexts,
                    retrieved_chunk_ids=chunk_ids,
                    failure_reason=_safe(exc),
                    clock=clock,
                )
            sleeper(2.0 * (2**attempt))
            attempt += 1
        except Exception as exc:
            return _record(
                item,
                guardrail_status=decision.status,
                final_status="pipeline_error",
                retrieved_contexts=contexts,
                retrieved_chunk_ids=chunk_ids,
                failure_reason=_safe(exc),
                clock=clock,
            )
    citations = [citation.model_dump() for citation in generated.citations]
    temperature = None
    if generated.generation_metadata is not None:
        temperature = generated.generation_metadata.temperature
    failure = generated.abstention_reason if generated.status == "abstained" else None
    return _record(
        item,
        guardrail_status=decision.status,
        final_status=generated.status,
        actual_answer=generated.answer,
        retrieved_contexts=contexts,
        retrieved_chunk_ids=chunk_ids,
        actual_citations=citations,
        generation_model=generated.model,
        generation_temperature=temperature,
        failure_reason=failure,
        abstention_reason=generated.abstention_reason,
        clock=clock,
    )


def _record(item: RagasQuestion, *, clock: Callable[[], str], **fields: object) -> PipelineRecord:
    payload = {
        "id": item.id,
        "question": item.question,
        "regulation": item.regulation,
        "reference_answer": item.reference_answer,
        "expected_chunk_ids": list(item.expected_chunk_ids),
        "evaluated_at": clock(),
    }
    payload.update(fields)
    return PipelineRecord.model_validate(payload)


def _safe(exc: BaseException) -> str:
    text = getattr(exc, "message", None) or str(exc)
    return text.replace("\n", " ")[:500]
