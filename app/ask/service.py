"""Run guardrails first, then generation only for an approved result."""

from __future__ import annotations

import logging

from app.ask.readiness import ReadinessReport, assess_readiness
from app.core.exceptions import AppError
from app.generation.errors import (
    GenerationConfigError,
    GenerationInvocationError,
    GenerationValidationError,
    public_error_text,
)
from app.generation.models import GenerationResponse
from app.generation.service import GenerationService
from app.guardrails.models import GuardrailResponse
from app.guardrails.service import GuardrailService
from app.schemas.ask import AskResponse

logger = logging.getLogger(__name__)

_VALIDATION_REASON = (
    "The generated answer could not be validated against the supplied evidence."
)


class AskService:
    """Coordinate the existing guardrail and generation services.

    This service does not search Qdrant itself and does not call the model
    again after a validation failure.
    """

    def __init__(self, guardrails: GuardrailService, generation: GenerationService) -> None:
        self.guardrails = guardrails
        self.generation = generation

    def ask(self, question: str) -> AskResponse:
        """Return a guardrail result, or a validated answer when one is allowed."""
        decision = self.guardrails.review(question)
        if decision.status != "approved":
            logger.info("Ask status=%s", decision.status)
            return _from_guardrail(question, decision)
        try:
            generated = self.generation.generate(question, decision)
        except GenerationValidationError:
            logger.info("Ask generation rejected code=generation_validation")
            return AskResponse(
                question=question,
                status="abstained",
                answer=None,
                reason=_VALIDATION_REASON,
                abstention_reason=_VALIDATION_REASON,
                citations=[],
                generation_metadata=None,
            )
        except GenerationConfigError as exc:
            raise AppError(exc.message, status_code=503, error_code=exc.error_code) from exc
        except GenerationInvocationError as exc:
            safe = public_error_text(exc, self.generation.settings.configured_key())
            raise AppError(safe, status_code=502, error_code=exc.error_code) from exc
        logger.info(
            "Ask status=%s citations=%s",
            generated.status,
            len(generated.citations),
        )
        return _from_generation(generated)

    def readiness(self) -> ReadinessReport:
        """Report local retrieval resources. Groq is not called."""
        retrieval = getattr(self.guardrails, "retrieval", None)
        return assess_readiness(retrieval)


def _from_guardrail(question: str, decision: GuardrailResponse) -> AskResponse:
    return AskResponse(
        question=question,
        status=decision.status,  # type: ignore[arg-type]
        answer=None,
        reason=decision.reason,
        clarification_question=decision.clarification_question,
        citations=[],
        abstention_reason=None,
        generation_metadata=None,
    )


def _from_generation(generated: GenerationResponse) -> AskResponse:
    return AskResponse(
        question=generated.question,
        status=generated.status,
        answer=generated.answer,
        reason=None,
        clarification_question=None,
        citations=generated.citations,
        abstention_reason=generated.abstention_reason,
        generation_metadata=generated.generation_metadata,
    )
