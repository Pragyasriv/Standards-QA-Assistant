"""Fidelity checks against the RoHS Annex II wording. The model is mocked."""

import pytest

from app.generation.errors import GenerationRejected, GenerationValidationError
from app.generation.service import GenerationService
from app.generation.settings import GenerationSettings
from app.guardrails.models import GuardrailResponse
from app.retrieval.models import RetrievedChunk

ANNEX_II = """ANNEX II
Restricted substances referred to in Article 4(1) and maximum concentration
values tolerated by weight in homogeneous materials
Lead (0,1 %)
Mercury (0,1 %)
Cadmium (0,01 %)
Hexavalent chromium (0,1 %)
Polybrominated biphenyls (PBB) (0,1 %)
Polybrominated diphenyl ethers (PBDE) (0,1 %)
Bis(2-ethylhexyl) phthalate (DEHP) (0,1 %)
Butyl benzyl phthalate (BBP) (0,1 %)
Dibutyl phthalate (DBP) (0,1 %)
Diisobutyl phthalate (DIBP) (0,1 %)
The restriction of DEHP, BBP, DBP and DIBP shall apply to medical devices, including in vitro medical devices, and monitoring and control instruments, including industrial monitoring and control instruments, from 22 July 2021.
The restriction of DEHP, BBP, DBP and DIBP shall not apply to cables or spare parts for the repair, the reuse, the updating of functionalities or upgrading of capacity of EEE placed on the market before 22 July 2019, and of medical devices, including in vitro medical devices, and monitoring and control instruments, including industrial monitoring and control instruments, placed on the market before 22 July 2021.
The restriction of DEHP, BBP and DBP shall not apply to toys which are already subject to the restriction of DEHP, BBP and DBP through entry 51 of Annex XVII to Regulation (EC) No 1907/2006."""

ENTRIES = [
    "Lead (0,1 %)",
    "Mercury (0,1 %)",
    "Cadmium (0,01 %)",
    "Hexavalent chromium (0,1 %)",
    "Polybrominated biphenyls (PBB) (0,1 %)",
    "Polybrominated diphenyl ethers (PBDE) (0,1 %)",
    "Bis(2-ethylhexyl) phthalate (DEHP) (0,1 %)",
    "Butyl benzyl phthalate (BBP) (0,1 %)",
    "Dibutyl phthalate (DBP) (0,1 %)",
    "Diisobutyl phthalate (DIBP) (0,1 %)",
]

CONDITIONS = (
    "The restriction of DEHP, BBP, DBP and DIBP shall apply to medical devices from 22 July 2021. "
    "It shall not apply to cables or spare parts for EEE placed on the market before 22 July 2019, "
    "and it shall not apply to those medical devices placed on the market before 22 July 2021. "
    "It shall not apply to toys already subject to entry 51 of Annex XVII."
)


class _FakeClient:
    def __init__(self, answer: str, *, cited: list[str] | None = None, abstained: bool = False, reason: str | None = None) -> None:
        self.payload = {
            "answer": answer,
            "cited_chunk_ids": cited if cited is not None else ["02011L0065-20260701:00141"],
            "abstained": abstained,
            "abstention_reason": reason,
        }
        self.calls = 0

    def invoke(self, messages: list[object]) -> object:
        self.calls += 1
        return self.payload


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="02011L0065-20260701:00141",
        document_id="02011L0065-20260701",
        regulation="RoHS",
        regulation_number="2011/65/EU",
        text=ANNEX_II,
        annex="II",
        page_start=22,
        page_end=22,
        source_filename="CL2011L0065EN0230010.0001.pdf",
    )


def _review(answer: str, **kwargs: object):
    client = _FakeClient(answer, **kwargs)  # type: ignore[arg-type]
    service = GenerationService(
        settings=GenerationSettings(groq_api_key="gsk_testsecretvalue", groq_model="configured-model"),
        client=client,
    )
    guardrail = GuardrailResponse(
        status="approved",
        scope="supported",
        reason="Annex II evidence.",
        detected_regulation="RoHS",
        evidence_available=True,
        retrieved_chunks=[_chunk()],
        annex="II",
    )
    return service, client, guardrail


def test_complete_list_keeps_every_entry_and_phthalate_condition() -> None:
    answer = "The restricted substances listed in the supplied text are:\n" + "\n".join(ENTRIES) + "\n" + CONDITIONS
    service, _client, guardrail = _review(answer)
    response = service.generate("What are the restricted substances under RoHS?", guardrail)
    assert response.status == "answered"
    assert response.citations[0].chunk_id == "02011L0065-20260701:00141"
    assert response.citations[0].annex == "II"
    assert "22 July 2019" in response.answer
    assert "22 July 2021" in response.answer
    assert "shall not apply" in response.answer


def test_phthalate_answer_keeps_application_dates_and_exceptions() -> None:
    answer = (
        "DEHP, BBP, DBP and DIBP are listed at 0,1 %. "
        + CONDITIONS
    )
    service, _client, guardrail = _review(answer)
    response = service.generate("What limits apply to the phthalates under RoHS?", guardrail)
    assert response.status == "answered"
    assert "toys" in response.answer
    assert "cables or spare parts" in response.answer
    assert "22 July 2019" in response.answer


def test_changed_decimal_notation_is_rejected() -> None:
    service, _client, guardrail = _review("Lead is limited to 0.1 % and mercury to 0.1 %.")
    with pytest.raises(GenerationValidationError, match="numerical value"):
        service.generate("What are the restricted substances under RoHS?", guardrail)


def test_changed_concentration_is_rejected() -> None:
    service, _client, guardrail = _review("Lead (0,2 %) is the value in the supplied text.")
    with pytest.raises(GenerationValidationError, match="0,2 %"):
        service.generate("What are the restricted substances under RoHS?", guardrail)


def test_conditional_requirement_stated_as_universal_is_rejected() -> None:
    service, _client, guardrail = _review(
        "DEHP, BBP, DBP and DIBP at 0,1 % apply to all EEE without exception."
    )
    with pytest.raises(GenerationValidationError, match="unconditional"):
        service.generate("What limits apply to the phthalates under RoHS?", guardrail)


def test_omitted_toys_exception_is_rejected() -> None:
    answer = "\n".join(ENTRIES) + (
        "\nDEHP, BBP, DBP and DIBP shall apply to medical devices from 22 July 2021. "
        "The restriction shall not apply to cables or spare parts before 22 July 2019 "
        "and before 22 July 2021."
    )
    service, _client, guardrail = _review(answer)
    with pytest.raises(GenerationValidationError, match="toys"):
        service.generate("What are the restricted substances under RoHS?", guardrail)


def test_partial_list_presented_as_exhaustive_is_rejected() -> None:
    service, _client, guardrail = _review(
        "The restricted substances are Lead (0,1 %) and Mercury (0,1 %)."
    )
    with pytest.raises(GenerationValidationError, match="partial list"):
        service.generate("What are the restricted substances under RoHS?", guardrail)


def test_disclaimer_does_not_excuse_a_partial_list() -> None:
    answer = (
        "The supplied text includes Lead (0,1 %) and Mercury (0,1 %). "
        "This is not a complete list: further entries and application conditions are not restated here."
    )
    service, _client, guardrail = _review(answer)
    with pytest.raises(GenerationValidationError, match="partial list"):
        service.generate("What are the restricted substances under RoHS?", guardrail)


def test_unknown_citation_is_still_rejected_without_a_partial_result() -> None:
    service, client, guardrail = _review(
        "\n".join(ENTRIES) + "\n" + CONDITIONS,
        cited=["02011L0065-20260701:00141", "not-retrieved"],
    )
    with pytest.raises(GenerationValidationError, match="not-retrieved"):
        service.generate("What are the restricted substances under RoHS?", guardrail)
    assert client.calls == 1


def test_abstention_still_returns_no_citations() -> None:
    service, _client, guardrail = _review(
        "The supplied documents do not establish the answer.",
        cited=[],
        abstained=True,
        reason="The question asks for a comparison the supplied text does not contain.",
    )
    response = service.generate("Does Annex II regulate tyres?", guardrail)
    assert response.status == "abstained"
    assert response.citations == []


def test_non_approved_result_still_skips_generation() -> None:
    service, client, _guardrail = _review("Lead (0,1 %).")
    blocked = GuardrailResponse(status="insufficient_evidence", scope="supported", reason="No list.")
    with pytest.raises(GenerationRejected):
        service.generate("What are the restricted substances under RoHS?", blocked)
    assert client.calls == 0
