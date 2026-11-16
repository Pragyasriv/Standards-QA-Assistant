"""Request and response models for POST /ask."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.generation.models import GenerationMetadata, ResolvedCitation

AskStatus = Literal[
    "invalid",
    "out_of_scope",
    "clarification_required",
    "insufficient_evidence",
    "answered",
    "abstained",
]


class AskRequest(BaseModel):
    """One user question. Empty text is judged by the guardrails, not rejected here."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        description=(
            "The question to check and, when the guardrails approve it, answer from "
            "the retrieved RoHS or REACH text. Whitespace-only text, text with no "
            "readable characters, and text longer than the configured guardrail limit "
            "come back with status invalid."
        ),
        examples=["What does RoHS Article 4 require?"],
    )


class AskResponse(BaseModel):
    """The guardrail decision, or a validated answer when generation was allowed.

    Status values:
    - invalid: the question text cannot be used.
    - out_of_scope: the question names another instrument, or it is not a regulatory question.
    - clarification_required: the topic is shared by RoHS and REACH and neither was named.
    - insufficient_evidence: a named RoHS or REACH question did not retrieve sufficient text.
    - answered: the guardrail approved the question and the answer passed citation checks.
    - abstained: the model declined, or its output failed validation. A failed validation
      does not include the rejected answer or any citation.
    """

    model_config = ConfigDict(extra="forbid")

    question: str
    status: AskStatus
    answer: str | None = None
    reason: str | None = None
    clarification_question: str | None = None
    citations: list[ResolvedCitation] = Field(default_factory=list)
    abstention_reason: str | None = None
    generation_metadata: GenerationMetadata | None = None


class ReadinessResponse(BaseModel):
    """Local resource check. This endpoint does not call Groq."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "not_ready"]
    qdrant_available: bool
    bm25_index_available: bool
