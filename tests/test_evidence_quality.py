"""Evidence sufficiency cases that the original guardrail set does not cover.

The scripted retriever supplies the chunks. These tests do not read Qdrant and
do not change the original evaluation files.
"""

import pytest

from app.guardrails.models import ScopeDecision
from app.guardrails.service import GuardrailService
from app.retrieval.filters import SearchFilters
from app.retrieval.models import RetrievalResponse, RetrievedChunk
from app.guardrails.sufficiency import has_list_entries, is_procedural_pointer


def _chunk(**overrides: object) -> RetrievedChunk:
    values: dict[str, object] = {
        "chunk_id": "rohs:1",
        "document_id": "rohs-doc",
        "regulation": "RoHS",
        "text": "Member States shall apply this provision.",
        "bm25_score": 8.0,
        "similarity_score": 0.4,
        "dense_score": 0.4,
        "rrf_score": 0.016,
    }
    values.update(overrides)
    return RetrievedChunk.model_validate(values)


ARTICLE_6 = (
    "Article 6\nReview and amendment of list of restricted substances in Annex II\n"
    "1. With a view to achieving the objectives set out in Article 1, a review "
    "and amendment of the list of restricted substances in Annex II shall be "
    "considered by the Commission."
)
ANNEX_II = (
    "ANNEX II\nRestricted substances referred to in Article 4(1) and maximum "
    "concentration values\nLead (0,1 %)\nMercury (0,1 %)\nCadmium (0,01 %)"
)


class _Selective:
    """Return one pool, or a second pool when the caller asks for an annex."""

    def __init__(self, chunks: list[RetrievedChunk], annex_chunks: list[RetrievedChunk] | None = None) -> None:
        self.chunks = chunks
        self.annex_chunks = annex_chunks or []
        self.filters: list[SearchFilters | None] = []

    def _pick(self, filters: SearchFilters | None) -> list[RetrievedChunk]:
        self.filters.append(filters)
        if filters is not None and filters.annex is not None:
            return [chunk for chunk in self.annex_chunks if chunk.annex == filters.annex]
        return list(self.chunks)

    def _response(self, question: str, filters: SearchFilters | None, method: str) -> RetrievalResponse:
        chunks = self._pick(filters)
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
        assert apply_threshold is False
        return self._response(question, filters, "dense")

    def search_bm25(self, question: str, filters: SearchFilters | None = None, top_k: int | None = None) -> RetrievalResponse:
        return self._response(question, filters, "bm25")

    def search_hybrid(self, question: str, filters: SearchFilters | None = None, top_k: int | None = None, dense_depth: int | None = None, sparse_depth: int | None = None, rrf_k: int | None = None) -> RetrievalResponse:
        return self._response(question, filters, "hybrid")


def _review(question: str, chunks: list[RetrievedChunk], annex_chunks: list[RetrievedChunk] | None = None):
    service = GuardrailService(retrieval=_Selective(chunks, annex_chunks))
    return service.review(question), service.retrieval


def test_a_numbered_paragraph_is_not_a_list() -> None:
    pointer = _chunk(text=ARTICLE_6, article="6", annex=None, chunk_id="article-6")
    listing = _chunk(text=ANNEX_II, annex="II", chunk_id="annex-ii")
    assert is_procedural_pointer(pointer) is True
    assert has_list_entries(pointer) is False
    assert has_list_entries(listing) is True


def test_related_article_is_not_enough_for_a_list_question() -> None:
    response, _retriever = _review(
        "What are the restricted substances under RoHS?",
        [_chunk(chunk_id="article-6", article="6", text=ARTICLE_6, bm25_score=9.6, rrf_score=0.032)],
    )
    assert response.status == "insufficient_evidence"
    assert response.retrieved_chunks == []
    assert "do not contain the requested list" in response.reason


@pytest.mark.parametrize(
    "question",
    [
        "What are the restricted substances under RoHS?",
        "What substances are restricted under the RoHS Directive?",
        "Which substances are restricted by RoHS?",
        "List all restricted substances under RoHS.",
    ],
)
def test_follow_up_reads_the_annex_named_by_the_article(question: str) -> None:
    article = _chunk(chunk_id="article-6", article="6", text=ARTICLE_6, bm25_score=9.6, rrf_score=0.032)
    listing = _chunk(
        chunk_id="annex-ii",
        annex="II",
        text=ANNEX_II,
        bm25_score=4.0,
        similarity_score=0.51,
        dense_score=0.51,
        rrf_score=0.01,
    )
    response, retriever = _review(question, [article], [listing])
    assert response.status == "approved"
    assert [chunk.chunk_id for chunk in response.retrieved_chunks] == ["annex-ii"]
    assert response.retrieved_chunks[0].bm25_score == 4.0
    assert response.retrieved_chunks[0].rrf_score == 0.01
    assert response.retrieved_chunks[0].dense_score == 0.51
    assert any(item is not None and item.annex == "II" for item in retriever.filters)


def test_list_chunk_is_kept_when_it_is_not_ranked_first() -> None:
    article = _chunk(chunk_id="article-6", article="6", text=ARTICLE_6, bm25_score=20.0)
    listing = _chunk(chunk_id="annex-ii", annex="II", text=ANNEX_II, bm25_score=3.0)
    response, _retriever = _review("What are the restricted substances under RoHS?", [article, listing])
    assert [chunk.chunk_id for chunk in response.retrieved_chunks] == ["annex-ii"]


def test_multiple_list_chunks_are_all_returned() -> None:
    first = _chunk(chunk_id="annex-a", annex="II", text=ANNEX_II, bm25_score=5.0)
    second = _chunk(
        chunk_id="annex-b",
        annex="II",
        text="Restricted substances continued\nLead (0,1 %)\nMercury (0,1 %)",
        bm25_score=4.0,
    )
    response, _retriever = _review("What are the restricted substances under RoHS?", [first, second])
    assert [chunk.chunk_id for chunk in response.retrieved_chunks] == ["annex-a", "annex-b"]


def test_positive_bm25_without_the_list_is_rejected() -> None:
    noise = _chunk(
        chunk_id="noise",
        article="9",
        text="The substances mentioned here shall be reviewed by the Commission in a later report.",
        bm25_score=6.0,
    )
    response, _retriever = _review("What are the restricted substances under RoHS?", [noise])
    assert response.status == "insufficient_evidence"


def test_another_regulation_cannot_supply_the_list() -> None:
    reach = _chunk(
        chunk_id="reach-list",
        document_id="reach-doc",
        regulation="REACH",
        annex="XVII",
        text="Restricted substances\nLead (0,1 %)",
        bm25_score=12.0,
    )
    response, _retriever = _review("What are the restricted substances under RoHS?", [reach])
    assert response.status == "insufficient_evidence"
    assert response.detected_regulation == "RoHS"


def test_a_named_missing_article_stays_insufficient() -> None:
    response, _retriever = _review("What does RoHS Article 112 require?", [])
    assert response.status == "insufficient_evidence"
    assert response.article == "112"


def test_naming_article_6_accepts_the_review_provision() -> None:
    article = _chunk(chunk_id="article-6", article="6", text=ARTICLE_6, bm25_score=12.0)
    response, _retriever = _review("What does RoHS Article 6 require?", [article])
    assert response.status == "approved"
    assert response.retrieved_chunks[0].chunk_id == "article-6"


def test_two_annex_pointers_are_not_collapsed_into_one_follow_up() -> None:
    text = (
        "Review and amendment of the list of restricted substances in Annex II "
        "and the exemptions in Annex III shall be considered by the Commission."
    )
    response, retriever = _review(
        "What are the restricted substances under RoHS?",
        [_chunk(text=text, article="6", bm25_score=5.0)],
        [_chunk(chunk_id="annex-ii", annex="II", text=ANNEX_II)],
    )
    assert response.status == "insufficient_evidence"
    assert all(item is None or item.annex is None for item in retriever.filters)


def test_scope_decision_copy_does_not_change_the_callers_annex() -> None:
    decision = ScopeDecision(scope="supported", reason="named", detected_regulation="RoHS")
    copied = decision.model_copy(update={"annex": "II"})
    assert decision.annex is None
    assert copied.annex == "II"
