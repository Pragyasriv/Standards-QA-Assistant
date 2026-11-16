"""Score the guardrail dataset. Retrieval-ranking questions are a separate file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.guardrails.models import GuardrailResponse
from app.guardrails.service import GuardrailService

ExpectedLabel = Literal["supported", "ambiguous", "unsupported"]
_LABELS = ("supported", "ambiguous", "unsupported", "invalid")


class GuardrailExample(BaseModel):
    """One manually labeled guardrail question."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    expected: ExpectedLabel
    expected_regulation: str | None = None


class GuardrailFailure(BaseModel):
    """A question whose decision does not match its label."""

    model_config = ConfigDict(extra="forbid")

    id: str
    expected: str
    status: str
    predicted: str
    detected_regulation: str | None = None
    reason: str


class GuardrailEvaluation(BaseModel):
    """Measured guardrail rates for one run."""

    model_config = ConfigDict(extra="forbid")

    questions: int
    supported_questions: int
    ambiguous_questions: int
    unsupported_questions: int
    supported_acceptance_rate: float
    supported_classification_rate: float
    ambiguous_clarification_rate: float
    unsupported_rejection_rate: float
    false_acceptance_rate: float
    false_rejection_rate: float
    insufficient_evidence_on_supported: int
    evidence_failures: list[str] = Field(default_factory=list)
    confusion_matrix: dict[str, dict[str, int]]
    failures: list[GuardrailFailure] = Field(default_factory=list)


def load_guardrail_questions(path: Path) -> list[GuardrailExample]:
    """Read the guardrail question file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [GuardrailExample.model_validate(item) for item in raw]


def evaluate_guardrails(service: GuardrailService, questions: list[GuardrailExample]) -> GuardrailEvaluation:
    """Run every question and aggregate acceptance, rejection, and confusion counts."""
    responses = [(item, service.review(item.question)) for item in questions]
    matrix = {expected: {predicted: 0 for predicted in _LABELS} for expected in _LABELS}
    failures: list[GuardrailFailure] = []
    supported = [pair for pair in responses if pair[0].expected == "supported"]
    ambiguous = [pair for pair in responses if pair[0].expected == "ambiguous"]
    unsupported = [pair for pair in responses if pair[0].expected == "unsupported"]
    for item, response in responses:
        predicted = _predicted_label(response)
        matrix[item.expected][predicted] = matrix[item.expected].get(predicted, 0) + 1
        regulation_mismatch = (
            item.expected == "supported"
            and item.expected_regulation is not None
            and response.detected_regulation != item.expected_regulation
            and response.status in {"approved", "insufficient_evidence"}
        )
        if predicted != item.expected or regulation_mismatch:
            failures.append(
                GuardrailFailure(
                    id=item.id,
                    expected=item.expected,
                    status=response.status,
                    predicted=predicted,
                    detected_regulation=response.detected_regulation,
                    reason=response.reason,
                )
            )
    approved = sum(1 for _, response in supported if response.status == "approved")
    classified_supported = sum(1 for _, response in supported if _predicted_label(response) == "supported")
    clarified = sum(1 for _, response in ambiguous if response.status == "clarification_required")
    rejected = sum(1 for _, response in unsupported if response.status == "out_of_scope")
    false_accept = sum(1 for _, response in unsupported if response.status == "approved")
    false_reject = sum(1 for _, response in supported if response.status in {"out_of_scope", "invalid"})
    insufficient = sum(1 for _, response in supported if response.status == "insufficient_evidence")
    return GuardrailEvaluation(
        questions=len(questions),
        supported_questions=len(supported),
        ambiguous_questions=len(ambiguous),
        unsupported_questions=len(unsupported),
        supported_acceptance_rate=_rate(approved, len(supported)),
        supported_classification_rate=_rate(classified_supported, len(supported)),
        ambiguous_clarification_rate=_rate(clarified, len(ambiguous)),
        unsupported_rejection_rate=_rate(rejected, len(unsupported)),
        false_acceptance_rate=_rate(false_accept, len(unsupported)),
        false_rejection_rate=_rate(false_reject, len(supported)),
        insufficient_evidence_on_supported=insufficient,
        evidence_failures=[item.id for item, response in supported if response.status == "insufficient_evidence"],
        confusion_matrix=matrix,
        failures=failures,
    )


def _predicted_label(response: GuardrailResponse) -> str:
    if response.status in {"approved", "insufficient_evidence"}:
        return "supported"
    if response.status == "clarification_required":
        return "ambiguous"
    if response.status == "out_of_scope":
        return "unsupported"
    return "invalid"


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(count / total, 4)
