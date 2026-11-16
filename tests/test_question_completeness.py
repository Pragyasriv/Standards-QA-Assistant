"""Question intent decides whether a partial list is allowed."""

import pytest

from app.generation.errors import GenerationValidationError
from app.generation.service import GenerationService
from app.generation.settings import GenerationSettings
from app.guardrails.models import GuardrailResponse
from app.retrieval.models import RetrievedChunk
from tests.test_answer_fidelity import CONDITIONS, ENTRIES, _review

QUESTION = "What are the restricted substances under RoHS?"


def test_complete_list_with_every_entry_and_condition_is_accepted() -> None:
    answer = "\n".join(ENTRIES) + "\n" + CONDITIONS
    service, _client, guardrail = _review(answer)
    response = service.generate("List all RoHS restricted substances.", guardrail)
    assert response.status == "answered"
    assert "shall not apply to toys" in response.answer
    assert "22 July 2019" in response.answer


@pytest.mark.parametrize(
    "disclaimer",
    [
        "This is not a complete list.",
        "Further entries and conditions are not restated here.",
        "Only a subset is reproduced below.",
    ],
)
def test_complete_list_disclaimer_does_not_replace_missing_entries(disclaimer: str) -> None:
    answer = "Lead (0,1 %)\nMercury (0,1 %)\n" + disclaimer
    service, _client, guardrail = _review(answer)
    with pytest.raises(GenerationValidationError, match="partial list"):
        service.generate("Give me the complete list of substances.", guardrail)


def test_complete_list_missing_a_substance_is_rejected() -> None:
    answer = "\n".join(line for line in ENTRIES if not line.startswith("Cadmium")) + "\n" + CONDITIONS
    service, _client, guardrail = _review(answer)
    with pytest.raises(GenerationValidationError, match="Cadmium"):
        service.generate(QUESTION, guardrail)


def test_complete_list_missing_a_qualification_is_rejected() -> None:
    answer = "\n".join(ENTRIES) + (
        "\nDEHP, BBP, DBP and DIBP shall apply to medical devices from 22 July 2021. "
        "The restriction shall not apply to cables or spare parts before 22 July 2019 "
        "and before 22 July 2021."
    )
    service, _client, guardrail = _review(answer)
    with pytest.raises(GenerationValidationError, match="toys"):
        service.generate("What are the exemptions under RoHS Annex III?", guardrail)


def test_lead_question_does_not_require_the_other_substances() -> None:
    service, _client, guardrail = _review("The concentration limit for lead is Lead (0,1 %).")
    response = service.generate("What is the concentration limit for lead?", guardrail)
    assert response.status == "answered"
    assert "Mercury" not in response.answer
    assert "toys" not in response.answer


def test_dehp_question_keeps_the_application_conditions() -> None:
    answer = (
        "Bis(2-ethylhexyl) phthalate (DEHP) (0,1 %)\n"
        + CONDITIONS
    )
    service, _client, guardrail = _review(answer)
    response = service.generate("When does the DEHP restriction apply?", guardrail)
    assert response.status == "answered"
    assert "22 July 2021" in response.answer
    assert "shall not apply to toys" in response.answer
    assert "Lead (0,1 %)" not in response.answer


def test_dehp_question_without_dehp_evidence_must_abstain() -> None:
    chunk = RetrievedChunk(
        chunk_id="02011L0065-20260701:00141",
        document_id="02011L0065-20260701",
        regulation="RoHS",
        text="Lead (0,1 %)",
        annex="II",
        page_start=22,
        page_end=22,
    )
    client_payload = {
        "answer": "DEHP applies to medical devices.",
        "cited_chunk_ids": [chunk.chunk_id],
        "abstained": False,
        "abstention_reason": None,
    }

    class _Client:
        def invoke(self, messages: list[object]) -> dict[str, object]:
            return client_payload

    service = GenerationService(
        settings=GenerationSettings(groq_api_key="gsk_testsecretvalue", groq_model="configured-model"),
        client=_Client(),
    )
    guardrail = GuardrailResponse(
        status="approved",
        scope="supported",
        reason="Only lead is present.",
        detected_regulation="RoHS",
        evidence_available=True,
        retrieved_chunks=[chunk],
        annex="II",
    )
    with pytest.raises(GenerationValidationError, match="must abstain"):
        service.generate("When does the DEHP restriction apply?", guardrail)


def test_specific_question_can_abstain_when_the_item_is_absent() -> None:
    service, _client, guardrail = _review(
        "The supplied documents do not establish the answer.",
        cited=[],
        abstained=True,
        reason="The supplied text does not mention toys.",
    )
    response = service.generate("What is the exception for toys?", guardrail)
    assert response.status == "abstained"
    assert response.citations == []


def test_complete_list_without_list_evidence_must_abstain() -> None:
    chunk = RetrievedChunk(
        chunk_id="02011L0065-20260701:00056",
        document_id="02011L0065-20260701",
        regulation="RoHS",
        text=(
            "Article 6\nReview and amendment of list of restricted substances in Annex II\n"
            "1. A review and amendment of the list shall be considered by the Commission."
        ),
        article="6",
        page_start=11,
        page_end=11,
    )

    class _Client:
        def invoke(self, messages: list[object]) -> dict[str, object]:
            return {
                "answer": "The restricted substances are reviewed by the Commission. This is not a complete list.",
                "cited_chunk_ids": [chunk.chunk_id],
                "abstained": False,
                "abstention_reason": None,
            }

    service = GenerationService(
        settings=GenerationSettings(groq_api_key="gsk_testsecretvalue", groq_model="configured-model"),
        client=_Client(),
    )
    guardrail = GuardrailResponse(
        status="approved",
        scope="supported",
        reason="Article 6 only.",
        detected_regulation="RoHS",
        evidence_available=True,
        retrieved_chunks=[chunk],
        annex=None,
    )
    with pytest.raises(GenerationValidationError, match="must abstain"):
        service.generate(QUESTION, guardrail)
