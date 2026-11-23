"""HTTP tests for POST /ask. The model and Qdrant client are fakes."""

import logging

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_ask_service
from app.ask.service import AskService
from app.generation.service import GenerationService
from app.generation.settings import GenerationSettings
from app.guardrails.service import GuardrailService
from app.retrieval.filters import SearchFilters
from app.retrieval.models import RetrievalResponse, RetrievedChunk


class _Retriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls = 0

    def _selected(self, filters: SearchFilters | None) -> list[RetrievedChunk]:
        selected: list[RetrievedChunk] = []
        for chunk in self.chunks:
            if filters is None:
                selected.append(chunk)
                continue
            if filters.regulation is not None and chunk.regulation != filters.regulation:
                continue
            if filters.article is not None and chunk.article != filters.article:
                continue
            if filters.annex is not None and chunk.annex != filters.annex:
                continue
            selected.append(chunk)
        return selected

    def _response(self, question: str, filters: SearchFilters | None, method: str) -> RetrievalResponse:
        self.calls += 1
        chunks = self._selected(filters)
        return RetrievalResponse(
            status="ok" if chunks else "insufficient_evidence",
            question=question,
            method=method,  # type: ignore[arg-type]
            top_k=5,
            minimum_similarity=None,
            best_score=None,
            results=chunks,
            filters=filters or SearchFilters(),
        )

    def search(self, question: str, filters: SearchFilters | None = None, top_k: int | None = None, minimum_similarity: float | None = None, *, apply_threshold: bool = True) -> RetrievalResponse:
        return self._response(question, filters, "dense")

    def search_bm25(self, question: str, filters: SearchFilters | None = None, top_k: int | None = None) -> RetrievalResponse:
        return self._response(question, filters, "bm25")

    def search_hybrid(self, question: str, filters: SearchFilters | None = None, top_k: int | None = None, dense_depth: int | None = None, sparse_depth: int | None = None, rrf_k: int | None = None) -> RetrievalResponse:
        return self._response(question, filters, "hybrid")


class _Model:
    def __init__(self, mode: str = "answer") -> None:
        self.mode = mode
        self.calls = 0

    def invoke(self, messages: list[object]) -> object:
        self.calls += 1
        if self.mode == "invalid-citation":
            return {
                "answer": "THIS_REJECTED_ANSWER",
                "cited_chunk_ids": ["not-retrieved"],
                "abstained": False,
                "abstention_reason": None,
            }
        if self.mode == "provider-down":
            raise RuntimeError("provider unavailable gsk_testsecretvalue")
        text = str(getattr(messages[-1], "content", ""))
        if "reach-6" in text:
            chunk_id = "reach-6"
            answer = "Article 6 states the registration duty described in the supplied text."
        else:
            chunk_id = "rohs-4"
            answer = "Article 4 states the prevention duty described in the supplied text."
        return {
            "answer": answer,
            "cited_chunk_ids": [chunk_id],
            "abstained": False,
            "abstention_reason": None,
        }


def _chunk(chunk_id: str, regulation: str, article: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"{regulation.lower()}-doc",
        regulation=regulation,
        regulation_number="2011/65/EU" if regulation == "RoHS" else "(EC) No 1907/2006",
        text=f"Article {article} states the duty in the supplied text.",
        bm25_score=12.0,
        article=article,
        page_start=8,
        page_end=8,
        source_filename=f"{regulation.lower()}.pdf",
    )


def _install(client: TestClient, ask: AskService) -> None:
    client.app.dependency_overrides[get_ask_service] = lambda: ask


def _ask_service(model: _Model | None = None, *, chunks: list[RetrievedChunk] | None = None, generation: GenerationService | None = None) -> tuple[AskService, _Model, _Retriever]:
    model = model or _Model()
    retriever = _Retriever(chunks if chunks is not None else [_chunk("rohs-4", "RoHS", "4"), _chunk("reach-6", "REACH", "6")])
    if generation is None:
        generation = GenerationService(
            settings=GenerationSettings(groq_api_key="gsk_testsecretvalue", groq_model="configured-model"),
            client=model,
        )
    return AskService(GuardrailService(retrieval=retriever), generation), model, retriever


def test_health_does_not_require_the_question_service(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_does_not_call_groq_or_expose_a_key(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ready", "not_ready"}
    assert "qdrant_available" in body
    assert "bm25_index_available" in body
    assert "gsk_" not in response.text
    assert "GROQ_API_KEY" not in response.text


def test_ask_without_started_resources_is_unavailable(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "What does RoHS Article 4 require?"})
    assert response.status_code == 503
    assert response.json()["error"] == "not_ready"


def test_openapi_describes_ask() -> None:
    from app.main import create_app

    schema = create_app().openapi()
    operation = schema["paths"]["/ask"]["post"]
    assert "approved" in operation["description"]
    assert "abstained" in operation["description"]
    properties = schema["components"]["schemas"]["AskResponse"]["properties"]
    assert "citations" in properties
    assert "clarification_question" in properties
    assert "abstention_reason" in properties


def test_approved_rohs_question_returns_a_validated_citation(client: TestClient) -> None:
    ask, model, _retriever = _ask_service()
    _install(client, ask)
    response = client.post("/ask", json={"question": "What does RoHS Article 4 require?"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert "prevention duty" in body["answer"]
    citation = body["citations"][0]
    assert citation["chunk_id"] == "rohs-4"
    assert citation["regulation"] == "RoHS"
    assert citation["article"] == "4"
    assert citation["source_filename"] == "rohs.pdf"
    assert citation["page_start"] == 8
    assert model.calls == 1


def test_approved_reach_question_returns_a_validated_citation(client: TestClient) -> None:
    ask, model, _retriever = _ask_service()
    _install(client, ask)
    response = client.post("/ask", json={"question": "What does REACH Article 6 require?"})
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "answered"
    assert body["citations"][0]["chunk_id"] == "reach-6"
    assert body["citations"][0]["regulation"] == "REACH"
    assert body["citations"][0]["regulation_number"] == "(EC) No 1907/2006"
    assert model.calls == 1


@pytest.mark.parametrize(
    ("question", "status"),
    [
        ("What are the restrictions on lead?", "clarification_required"),
        ("What are GDPR data retention requirements?", "out_of_scope"),
        ("What is the capital of France?", "out_of_scope"),
        ("   ", "invalid"),
        ("???", "invalid"),
    ],
)
def test_non_approved_questions_do_not_call_generation(client: TestClient, question: str, status: str) -> None:
    ask, model, retriever = _ask_service()
    _install(client, ask)
    response = client.post("/ask", json={"question": question})
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == status
    assert body["citations"] == []
    assert body["answer"] is None
    assert model.calls == 0
    assert retriever.calls == 0
    if status == "clarification_required":
        assert body["clarification_question"]


def test_missing_evidence_returns_insufficient_evidence(client: TestClient) -> None:
    ask, model, retriever = _ask_service(chunks=[])
    _install(client, ask)
    response = client.post("/ask", json={"question": "What does RoHS Article 4 require?"})
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "insufficient_evidence"
    assert body["citations"] == []
    assert model.calls == 0
    assert retriever.calls > 0


def test_question_over_the_guardrail_limit_is_invalid(client: TestClient) -> None:
    ask, model, _retriever = _ask_service()
    ask.guardrails.settings.guardrail_max_question_length = 20
    _install(client, ask)
    response = client.post("/ask", json={"question": "What does RoHS Article 4 require in full?"})
    assert response.status_code == 200
    assert response.json()["status"] == "invalid"
    assert model.calls == 0


def test_missing_question_field_is_a_validation_error(client: TestClient) -> None:
    ask, _model, _retriever = _ask_service()
    _install(client, ask)
    response = client.post("/ask", json={})
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_invalid_model_citation_abstains_without_the_rejected_answer(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ask, model, _retriever = _ask_service(_Model("invalid-citation"))
    _install(client, ask)
    caplog.set_level(logging.INFO)
    response = client.post("/ask", json={"question": "What does RoHS Article 4 require?"})
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "abstained"
    assert body["answer"] is None
    assert body["citations"] == []
    assert body["generation_metadata"] is None
    assert "THIS_REJECTED_ANSWER" not in response.text
    assert "not-retrieved" not in response.text
    non_diagnostic = "\n".join(
        line for line in caplog.text.splitlines() if "TEMPORARY DIAGNOSTIC" not in line
    )
    assert "THIS_REJECTED_ANSWER" not in non_diagnostic
    assert model.calls == 1


def test_missing_api_key_returns_503_without_a_secret(client: TestClient) -> None:
    generation = GenerationService(settings=GenerationSettings(groq_api_key="", groq_model="configured-model"))
    ask, _model, _retriever = _ask_service(generation=generation)
    _install(client, ask)
    response = client.post("/ask", json={"question": "What does RoHS Article 4 require?"})
    assert response.status_code == 503
    assert response.json()["error"] == "generation_config"
    assert "gsk_" not in response.text


def test_provider_failure_returns_502_without_a_secret(client: TestClient) -> None:
    ask, _model, _retriever = _ask_service(_Model("provider-down"))
    _install(client, ask)
    response = client.post("/ask", json={"question": "What does RoHS Article 4 require?"})
    assert response.status_code == 502
    assert response.json()["error"] == "generation_invocation"
    assert "gsk_" not in response.text


def test_validation_error_is_not_rerun(client: TestClient) -> None:
    ask, model, _retriever = _ask_service(_Model("invalid-citation"))
    _install(client, ask)
    client.post("/ask", json={"question": "What does RoHS Article 4 require?"})
    assert model.calls == 1
