"""Validation-aware retry for complete-list fidelity failures."""

from __future__ import annotations

import json
import logging

import pytest
from langchain_core.messages import HumanMessage

from app.generation.errors import GenerationValidationError
from app.generation.retry import is_complete_list_fidelity_retry_eligible
from app.generation.service import GenerationService
from app.generation.settings import GenerationSettings
from app.guardrails.models import GuardrailResponse
from app.retrieval.models import RetrievedChunk
from tests.test_answer_fidelity import ANNEX_II, CONDITIONS, ENTRIES, _chunk as annex_chunk

LIST_QUESTION = "What substances are restricted under the RoHS Directive?"


class _SequentialClient:
    def __init__(self, results: list[object]) -> None:
        self.results = list(results)
        self.calls = 0
        self.messages: list[list[object]] = []

    def invoke(self, messages: list[object]) -> object:
        self.calls += 1
        self.messages.append(messages)
        if self.calls > len(self.results):
            raise AssertionError("unexpected extra model call")
        result = self.results[self.calls - 1]
        if isinstance(result, Exception):
            raise result
        return result


def _guardrail() -> GuardrailResponse:
    return GuardrailResponse(
        status="approved",
        scope="supported",
        reason="Annex II evidence.",
        detected_regulation="RoHS",
        evidence_available=True,
        retrieved_chunks=[annex_chunk()],
        annex="II",
    )


def _service(*results: object, temperature: float = 0.0) -> tuple[GenerationService, _SequentialClient]:
    client = _SequentialClient(list(results))
    settings = GenerationSettings(
        groq_api_key="gsk_testsecretvalue",
        groq_model="configured-model",
        groq_temperature=temperature,
    )
    return GenerationService(settings=settings, client=client), client


def _payload(answer: str) -> str:
    return json.dumps(
        {
            "answer": answer,
            "cited_chunk_ids": ["02011L0065-20260701:00141"],
            "abstained": False,
            "abstention_reason": None,
        }
    )


def _evidence_blob(messages: list[object]) -> str:
    human = next(message for message in messages if isinstance(message, HumanMessage))
    content = human.content
    assert isinstance(content, str)
    return content


def test_retry_eligible_only_for_complete_list_qualification_gap() -> None:
    exc = GenerationValidationError(
        "The answer omits an application condition or exception present in the cited evidence: shall apply."
    )
    assert is_complete_list_fidelity_retry_eligible(exc, LIST_QUESTION) is True
    assert is_complete_list_fidelity_retry_eligible(exc, "What is the concentration limit for lead?") is False
    citation_exc = GenerationValidationError(
        "The model cited chunk IDs that were not in the supplied evidence: x."
    )
    assert is_complete_list_fidelity_retry_eligible(citation_exc, LIST_QUESTION) is False
    partial = GenerationValidationError(
        "The answer presents a partial list as exhaustive. The cited evidence also contains: Cadmium (0,01 %)."
    )
    assert is_complete_list_fidelity_retry_eligible(partial, LIST_QUESTION) is True


def test_second_attempt_with_qualifications_passes() -> None:
    partial = _payload("\n".join(ENTRIES))
    complete = _payload(
        "The restricted substances listed in the supplied text are:\n"
        + "\n".join(ENTRIES)
        + "\n"
        + CONDITIONS
    )
    service, client = _service(partial, complete)
    response = service.generate(LIST_QUESTION, _guardrail())
    assert response.status == "answered"
    assert client.calls == 2
    assert "shall not apply" in response.answer
    assert response.generation_metadata.temperature == 0.0


def test_second_attempt_still_fails_validation() -> None:
    partial = _payload("\n".join(ENTRIES))
    service, client = _service(partial, partial)
    with pytest.raises(GenerationValidationError, match="omits an application condition"):
        service.generate(LIST_QUESTION, _guardrail())
    assert client.calls == 2


def test_success_on_first_attempt_invokes_once() -> None:
    complete = _payload("\n".join(ENTRIES) + "\n" + CONDITIONS)
    service, client = _service(complete)
    response = service.generate(LIST_QUESTION, _guardrail())
    assert response.status == "answered"
    assert client.calls == 1


def test_invalid_citation_does_not_retry() -> None:
    payload = json.dumps(
        {
            "answer": "\n".join(ENTRIES) + "\n" + CONDITIONS,
            "cited_chunk_ids": ["02011L0065-20260701:00141", "not-retrieved"],
            "abstained": False,
            "abstention_reason": None,
        }
    )
    service, client = _service(payload)
    with pytest.raises(GenerationValidationError, match="not-retrieved"):
        service.generate(LIST_QUESTION, _guardrail())
    assert client.calls == 1


def test_malformed_output_does_not_retry() -> None:
    service, client = _service("not json at all")
    with pytest.raises(GenerationValidationError):
        service.generate(LIST_QUESTION, _guardrail())
    assert client.calls == 1


def test_same_evidence_in_both_attempts() -> None:
    partial = _payload("\n".join(ENTRIES))
    complete = _payload("\n".join(ENTRIES) + "\n" + CONDITIONS)
    service, client = _service(partial, complete)
    service.generate(LIST_QUESTION, _guardrail())
    assert len(client.messages) == 2
    first = _evidence_blob(client.messages[0])
    second = _evidence_blob(client.messages[1])
    assert ANNEX_II in first and ANNEX_II in second
    assert first.startswith("Question:")
    assert second.startswith(first)
    assert "did not pass validation" in second
    assert "[chunk_id] 02011L0065-20260701:00141" in first
    assert first in second


def test_attempt_two_parse_failure_does_not_reuse_prior_draft(
    caplog: pytest.LogCaptureFixture,
) -> None:
    partial = _payload("\n".join(ENTRIES))
    service, client = _service(partial, "not-json")
    with caplog.at_level(logging.INFO, logger="app.generation.service"):
        with pytest.raises(GenerationValidationError):
            service.generate(LIST_QUESTION, _guardrail())
    assert client.calls == 2
    assert caplog.text.count("TEMPORARY DIAGNOSTIC validation_draft") == 1
    assert "attempt=2" in caplog.text
    assert "phase=parse" in caplog.text


def test_logging_shows_attempt_and_retry(caplog: pytest.LogCaptureFixture) -> None:
    partial = _payload("\n".join(ENTRIES))
    complete = _payload("\n".join(ENTRIES) + "\n" + CONDITIONS)
    service, _client = _service(partial, complete)
    with caplog.at_level(logging.INFO, logger="app.generation.service"):
        service.generate(LIST_QUESTION, _guardrail())
    joined = caplog.text
    assert "Generation retry triggered" in joined
    assert "attempt=1" in joined
    assert "retry_triggered=True" in joined
    assert "outcome=succeeded" in joined