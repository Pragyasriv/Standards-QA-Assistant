"""Grounded generation with a mocked chat model. No Groq call is made."""

import json
import logging

import pytest

from app.generation.client import build_chat_model
from app.generation.errors import (
    GenerationConfigError,
    GenerationInvocationError,
    GenerationRejected,
    GenerationValidationError,
)
from app.generation.models import ModelDraft
from app.generation.service import GenerationService
from app.generation.settings import GenerationSettings
from app.guardrails.models import GuardrailResponse
from app.retrieval.models import RetrievedChunk


class _FakeClient:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls = 0
        self.messages: list[object] = []

    def invoke(self, messages: list[object]) -> object:
        self.calls += 1
        self.messages = messages
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _chunk(**overrides: object) -> RetrievedChunk:
    values: dict[str, object] = {
        "chunk_id": "02011L0065-20260701:00141",
        "document_id": "02011L0065-20260701",
        "regulation": "RoHS",
        "regulation_number": "2011/65/EU",
        "text": "ANNEX II\nRestricted substances\nLead (0,1 %)\nMercury (0,1 %)",
        "article": None,
        "paragraph": None,
        "subparagraph": None,
        "annex": "II",
        "page_start": 22,
        "page_end": 22,
        "source_filename": "CL2011L0065EN0230010.0001.pdf",
    }
    values.update(overrides)
    return RetrievedChunk.model_validate(values)


def _approved(chunks: list[RetrievedChunk], status: str = "approved") -> GuardrailResponse:
    return GuardrailResponse(
        status=status,  # type: ignore[arg-type]
        scope="supported" if status in {"approved", "insufficient_evidence"} else status,  # type: ignore[arg-type]
        reason="test",
        detected_regulation="RoHS",
        evidence_available=status == "approved",
        retrieved_chunks=chunks if status == "approved" else [],
    )


def _draft(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "answer": "The supplied Annex II text lists Lead (0,1 %) and Mercury (0,1 %).",
        "cited_chunk_ids": ["02011L0065-20260701:00141"],
        "abstained": False,
        "abstention_reason": None,
    }
    payload.update(overrides)
    return payload


def _service(result: object, **settings: object) -> tuple[GenerationService, _FakeClient]:
    client = _FakeClient(result)
    config = GenerationSettings(
        groq_api_key="gsk_testsecretvalue",
        groq_model="configured-model",
        groq_temperature=0,
    )
    for name, value in settings.items():
        setattr(config, name, value)
    return GenerationService(settings=config, client=client), client


def test_one_citation_is_resolved_from_the_chunk() -> None:
    service, client = _service(json.dumps(_draft(source_filename="forged.pdf", page_start=999)))
    response = service.generate("What are the restricted substances under RoHS?", _approved([_chunk()]))
    assert response.status == "answered"
    assert response.model == "configured-model"
    assert len(response.citations) == 1
    citation = response.citations[0]
    assert citation.chunk_id == "02011L0065-20260701:00141"
    assert citation.regulation == "RoHS"
    assert citation.regulation_number == "2011/65/EU"
    assert citation.annex == "II"
    assert citation.page_start == 22
    assert citation.page_end == 22
    assert citation.source_filename == "CL2011L0065EN0230010.0001.pdf"
    assert citation.document_id == "02011L0065-20260701"
    assert "forged.pdf" not in response.model_dump_json()
    assert client.calls == 1
    assert response.generation_metadata.evidence_chunk_count == 1
    assert response.generation_metadata.validated_citation_count == 1
    assert response.generation_metadata.temperature == 0


def test_multiple_citations_preserve_order() -> None:
    first = _chunk()
    second = _chunk(
        chunk_id="02011L0065-20260701:00142",
        text="Annex II continued\nCadmium (0,01 %)",
        paragraph="2",
    )
    payload = _draft(
        answer="Lead (0,1 %)\nMercury (0,1 %)\nCadmium (0,01 %)",
        cited_chunk_ids=[second.chunk_id, first.chunk_id],
    )
    service, _client = _service(payload)
    response = service.generate("What are the restricted substances under RoHS?", _approved([first, second]))
    assert [item.chunk_id for item in response.citations] == [second.chunk_id, first.chunk_id]
    assert response.citations[0].paragraph == "2"


def test_duplicate_citation_ids_keep_the_first_occurrence() -> None:
    chunk = _chunk()
    payload = _draft(cited_chunk_ids=[chunk.chunk_id, chunk.chunk_id, chunk.chunk_id])
    service, _client = _service(ModelDraft.model_validate(payload))
    response = service.generate("What are the restricted substances under RoHS?", _approved([chunk]))
    assert [item.chunk_id for item in response.citations] == [chunk.chunk_id]


def test_unknown_citation_rejects_the_whole_answer() -> None:
    chunk = _chunk()
    payload = _draft(cited_chunk_ids=[chunk.chunk_id, "not-retrieved"])
    service, _client = _service(payload)
    with pytest.raises(GenerationValidationError, match="not-retrieved"):
        service.generate("What are the restricted substances under RoHS?", _approved([chunk]))


def test_citation_from_another_question_is_rejected() -> None:
    supplied = _chunk()
    other = _chunk(chunk_id="02006R1907-20260622:00104", regulation="REACH", text="Article 6")
    payload = _draft(
        answer="Article 6 requires registration.",
        cited_chunk_ids=[other.chunk_id],
    )
    service, _client = _service(payload)
    with pytest.raises(GenerationValidationError, match=other.chunk_id):
        service.generate("What does REACH Article 6 require?", _approved([supplied]))


def test_malformed_output_is_rejected() -> None:
    service, _client = _service("this is not json")
    with pytest.raises(GenerationValidationError):
        service.generate("What are the restricted substances under RoHS?", _approved([_chunk()]))


def test_empty_answer_without_abstention_is_rejected() -> None:
    service, _client = _service(_draft(answer="   ", abstained=False))
    with pytest.raises(GenerationValidationError, match="empty answer"):
        service.generate("What are the restricted substances under RoHS?", _approved([_chunk()]))


def test_valid_abstention_returns_no_citations() -> None:
    payload = _draft(
        answer="The supplied documents do not establish the answer.",
        cited_chunk_ids=[],
        abstained=True,
        abstention_reason="The supplied text does not contain the requested list.",
    )
    service, _client = _service(payload)
    response = service.generate("What are the restricted substances under RoHS?", _approved([_chunk()]))
    assert response.status == "abstained"
    assert response.citations == []
    assert response.abstention_reason == "The supplied text does not contain the requested list."
    assert response.generation_metadata.abstained is True


def test_abstention_that_states_a_requirement_is_rejected() -> None:
    payload = _draft(
        answer="Lead shall not exceed 0,1 %.",
        cited_chunk_ids=[],
        abstained=True,
        abstention_reason="Lead shall not exceed 0,1 %.",
    )
    service, _client = _service(payload)
    with pytest.raises(GenerationValidationError, match="abstained"):
        service.generate("What are the restricted substances under RoHS?", _approved([_chunk()]))


def test_validation_rejection_logs_safe_reason(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="app.generation.service")
    payload = _draft(cited_chunk_ids=["not-in-evidence"])
    service, _client = _service(payload)
    with pytest.raises(GenerationValidationError, match="not-in-evidence"):
        service.generate(
            "What substances are restricted under the RoHS Directive?",
            _approved([_chunk()]),
        )
    joined = caplog.text
    assert "Generation output rejected" in joined
    assert "phase=validation" in joined
    assert "validation_reason=" in joined
    assert "not-in-evidence" in joined
    assert "gsk_" not in joined
    assert "TEMPORARY DIAGNOSTIC validation_draft" in joined
    assert f"abstained={payload['abstained']}" in joined
    assert "cited_chunk_ids=" in joined
    assert str(payload["answer"]) in joined


def test_parse_rejection_logs_phase_parse_without_draft_diagnostic(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="app.generation.service")
    service, _client = _service("this is not json")
    with pytest.raises(GenerationValidationError):
        service.generate("What are the restricted substances under RoHS?", _approved([_chunk()]))
    joined = caplog.text
    assert "Generation output rejected" in joined
    assert "phase=parse" in joined
    assert "validation_reason=" in joined
    assert "TEMPORARY DIAGNOSTIC validation_draft" not in joined
    assert "gsk_" not in joined


def test_validation_draft_diagnostic_skipped_in_production(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    caplog.set_level(logging.INFO, logger="app.generation.service")
    monkeypatch.setenv("APP_ENV", "production")
    from app.core.config import get_settings

    get_settings.cache_clear()
    service, _client = _service(_draft(cited_chunk_ids=["not-in-evidence"]))
    try:
        with pytest.raises(GenerationValidationError):
            service.generate(
                "What substances are restricted under the RoHS Directive?",
                _approved([_chunk()]),
            )
    finally:
        get_settings.cache_clear()
    assert "TEMPORARY DIAGNOSTIC validation_draft" not in caplog.text


@pytest.mark.parametrize(
    "status",
    ["invalid", "out_of_scope", "clarification_required", "insufficient_evidence"],
)
def test_non_approved_guardrail_result_does_not_call_the_model(status: str) -> None:
    service, client = _service(_draft())
    scope = {
        "invalid": "invalid",
        "out_of_scope": "out_of_scope",
        "clarification_required": "clarification_required",
        "insufficient_evidence": "supported",
    }[status]
    guardrail = GuardrailResponse(
        status=status,  # type: ignore[arg-type]
        scope=scope,  # type: ignore[arg-type]
        reason="blocked",
        retrieved_chunks=[_chunk()] if status == "insufficient_evidence" else [],
    )
    with pytest.raises(GenerationRejected, match=status):
        service.generate("What are the restricted substances under RoHS?", guardrail)
    assert client.calls == 0


def test_missing_api_key_does_not_call_groq() -> None:
    settings = GenerationSettings(groq_api_key="", groq_model="configured-model")
    service = GenerationService(settings=settings)
    with pytest.raises(GenerationConfigError, match="GROQ_API_KEY"):
        service.generate("What are the restricted substances under RoHS?", _approved([_chunk()]))


def test_placeholder_model_is_rejected() -> None:
    settings = GenerationSettings(groq_api_key="gsk_testsecretvalue", groq_model="your_available_groq_model")
    with pytest.raises(GenerationConfigError, match="GROQ_MODEL"):
        build_chat_model(settings)


def test_model_invocation_failure_is_controlled(caplog: pytest.LogCaptureFixture) -> None:
    service, _client = _service(RuntimeError("model configured-model is unavailable gsk_testsecretvalue"))
    caplog.set_level(logging.INFO)
    with pytest.raises(GenerationInvocationError, match="configured-model") as caught:
        service.generate("What are the restricted substances under RoHS?", _approved([_chunk()]))
    assert "gsk_" not in str(caught.value)
    assert "gsk_" not in caplog.text
    assert "ANNEX II" not in caplog.text


def test_provider_model_switch_is_rejected() -> None:
    class _Switched:
        response_metadata = {"model_name": "some-other-model"}
        content = json.dumps(_draft())

    service, _client = _service(_Switched())
    with pytest.raises(GenerationInvocationError, match="some-other-model"):
        service.generate("What are the restricted substances under RoHS?", _approved([_chunk()]))


def test_missing_optional_metadata_stays_null() -> None:
    chunk = _chunk(
        regulation_number=None,
        article=None,
        paragraph=None,
        subparagraph=None,
        source_filename=None,
        page_start=None,
        page_end=None,
    )
    payload = _draft(
        answer="The supplied text lists Lead (0,1 %).",
        source_filename="invented.pdf",
        regulation_number="9999/99/EU",
        page_start=1,
    )
    service, _client = _service(payload)
    response = service.generate("What is the concentration limit for lead?", _approved([chunk]))
    citation = response.citations[0]
    assert citation.regulation_number is None
    assert citation.article is None
    assert citation.paragraph is None
    assert citation.subparagraph is None
    assert citation.source_filename is None
    assert citation.page_start is None
    assert citation.page_end is None
    assert citation.regulation == "RoHS"
    assert citation.annex == "II"
