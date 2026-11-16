"""Tests for deterministic RoHS and REACH guardrails."""

from pathlib import Path

import pytest

from app.guardrails.classify import classify_question
from app.guardrails.evaluate import GuardrailExample, evaluate_guardrails, load_guardrail_questions
from app.guardrails.service import GuardrailService
from app.guardrails.settings import GuardrailSettings
from app.retrieval.evaluate import load_questions
from app.retrieval.filters import SearchFilters
from app.retrieval.models import RetrievalResponse, RetrievedChunk


def test_identifies_rohs_and_reach_names_and_abbreviations() -> None:
    rohs = classify_question("What are the restricted substances under RoHS?")
    reach = classify_question("What does REACH Article 6 require?")
    full_rohs = classify_question("What does Article 4 of Directive 2011/65/EU require?")
    full_reach = classify_question("What is the aim of Regulation (EC) No 1907/2006?")

    assert rohs.scope == "supported"
    assert rohs.detected_regulation == "RoHS"
    assert reach.scope == "supported"
    assert reach.detected_regulation == "REACH"
    assert reach.article == "6"
    assert full_rohs.detected_regulation == "RoHS"
    assert full_rohs.article == "4"
    assert full_reach.detected_regulation == "REACH"


def test_rejects_unsupported_instruments_and_unrelated_questions() -> None:
    gdpr = classify_question("What are GDPR data retention requirements?")
    weee = classify_question("Explain the WEEE collection targets.")
    lvd = classify_question("What are the requirements under the Low Voltage Directive?")
    lvd_ce = classify_question("How does the LVD require CE marking to be applied?")
    france = classify_question("What is the capital of France?")
    poem = classify_question("Write a poem about flowers.")

    assert gdpr.scope == "out_of_scope"
    assert gdpr.detected_unsupported_instrument == "General Data Protection Regulation"
    assert weee.detected_unsupported_instrument == "WEEE Directive"
    assert lvd.detected_unsupported_instrument == "Low Voltage Directive"
    assert lvd_ce.scope == "out_of_scope"
    assert france.scope == "out_of_scope"
    assert france.detected_unsupported_instrument is None
    assert poem.scope == "out_of_scope"


def test_asks_for_clarification_on_overlap_topics() -> None:
    lead = classify_question("What are the restrictions on lead?")
    marking = classify_question("What are the requirements for CE marking?")
    chemicals = classify_question("What are the rules for chemical substances?")

    assert lead.scope == "clarification_required"
    assert lead.detected_regulation is None
    assert "Lead is restricted under both RoHS and REACH" in (lead.clarification_question or "")
    assert marking.scope == "clarification_required"
    assert "CE marking" in (marking.clarification_question or "")
    assert chemicals.scope == "clarification_required"


def test_keeps_an_explicit_rohs_question_that_mentions_ce_marking() -> None:
    decision = classify_question("What are the CE marking rules in the RoHS Directive?")

    assert decision.scope == "supported"
    assert decision.detected_regulation == "RoHS"
    assert decision.detected_unsupported_instrument is None


def test_invalid_inputs_return_structured_errors() -> None:
    service = GuardrailService(retrieval=_EmptyRetriever(), settings=GuardrailSettings(guardrail_max_question_length=20))

    empty = service.review("   ")
    missing = service.review(None)
    symbols = service.review("???")
    too_long = service.review("What does RoHS require in this very long question?")
    malformed = service.review("RoHS\x00question")

    assert empty.status == "invalid"
    assert empty.reason == "Question is empty."
    assert missing.status == "invalid"
    assert missing.reason == "Question must be text."
    assert symbols.status == "invalid"
    assert "readable" in (symbols.reason or "")
    assert too_long.status == "invalid"
    assert "maximum length" in (too_long.reason or "")
    assert malformed.status == "invalid"
    assert service.retrieval.calls == []  # type: ignore[attr-defined]


def test_empty_and_insufficient_evidence() -> None:
    empty = GuardrailService(retrieval=_EmptyRetriever()).review("What does REACH Article 6 require?")
    neighbor = _chunk("reach:other", "REACH", article="6", bm25=None, dense=0.62, rrf=0.02)
    weak = GuardrailService(retrieval=_Scripted([neighbor])).review("What does REACH Article 6 require?")

    assert empty.status == "insufficient_evidence"
    assert empty.evidence_available is False
    assert empty.retrieved_chunks == []
    assert weak.status == "insufficient_evidence"
    assert "shares the terms" in weak.reason


def test_wrong_regulation_is_not_used_as_evidence() -> None:
    chunk = _chunk("reach:gunshot", "REACH", annex="XVII", bm25=8.0, dense=0.65, rrf=0.03)
    response = GuardrailService(retrieval=_Scripted([chunk], honor_filters=False)).review(
        "What are the restricted substances under RoHS?"
    )

    assert response.status == "insufficient_evidence"
    assert response.detected_regulation == "RoHS"
    assert response.retrieved_chunks == []
    assert "REACH" in response.reason


def test_article_and_annex_evidence_and_preserved_citations() -> None:
    article = _chunk(
        "02006R1907-20260622:00104",
        "REACH",
        article="6",
        text="Article 6 General obligation to register substances on their own.",
        bm25=12.6,
        dense=0.6,
        rrf=0.016,
        page_start=15,
        source_filename="CL2006R1907EN0690010.0001.pdf",
        consolidation_date="2026-06-22",
    )
    annex = _chunk(
        "02011L0065-20260701:00141",
        "RoHS",
        annex="II",
        text="Restricted substances referred to in Article 4(1).",
        bm25=9.0,
        dense=0.41,
        rrf=0.02,
        page_start=22,
        source_filename="CL2011L0065EN0230010.0001.pdf",
        consolidation_date="2026-07-01",
    )
    wrong_article = _chunk("reach:other", "REACH", article="9", bm25=4.0, dense=0.7, rrf=0.03)
    service = GuardrailService(retrieval=_Scripted([article, annex, wrong_article]))

    reach = service.review("What does REACH Article 6 require?")
    rohs = service.review("What concentration limits does RoHS Annex II set?")
    missed = GuardrailService(retrieval=_Scripted([wrong_article], honor_filters=False)).review(
        "What does REACH Article 6 require?"
    )

    assert reach.status == "approved"
    assert reach.retrieved_chunks[0].chunk_id == article.chunk_id
    assert reach.retrieved_chunks[0].similarity_score == 0.6
    assert reach.retrieved_chunks[0].dense_score == 0.6
    assert reach.retrieved_chunks[0].bm25_score == 12.6
    assert reach.retrieved_chunks[0].rrf_score == 0.016
    assert reach.retrieved_chunks[0].rrf_score != reach.retrieved_chunks[0].similarity_score
    citation = reach.citation_metadata[0]
    assert citation.page_start == 15
    assert citation.source_filename == "CL2006R1907EN0690010.0001.pdf"
    assert citation.consolidation_date == "2026-06-22"
    assert citation.article == "6"
    assert rohs.status == "approved"
    assert rohs.retrieved_chunks[0].annex == "II"
    assert rohs.citation_metadata[0].page_start == 22
    assert missed.status == "insufficient_evidence"
    assert "Article 6" in missed.reason


def test_low_rrf_does_not_block_lexical_evidence() -> None:
    chunk = _chunk("rohs:ce", "RoHS", article="15", bm25=19.5, dense=0.2, rrf=0.01)
    retriever = _Scripted([chunk])
    response = GuardrailService(retrieval=retriever).review("What are the CE marking rules in the RoHS Directive?")

    assert response.status == "approved"
    assert response.retrieved_chunks[0].rrf_score == 0.01
    assert retriever.dense_thresholds == [False]


def test_supported_and_unsupported_decisions_do_not_cross() -> None:
    retriever = _Scripted([_chunk("rohs:1", "RoHS", article="4", bm25=5.0, dense=0.4, rrf=0.02)])
    service = GuardrailService(retrieval=retriever)

    supported = service.review("What does Article 4 of Directive 2011/65/EU require?")
    rejected = service.review("What is the capital of France?")
    ambiguous = service.review("What are the restrictions on lead?")

    assert supported.status == "approved"
    assert rejected.status == "out_of_scope"
    assert ambiguous.status == "clarification_required"
    assert "What is the capital of France?" not in retriever.questions
    assert "What are the restrictions on lead?" not in retriever.questions


def test_guardrail_dataset_is_separate_from_retrieval_questions() -> None:
    guardrails = load_guardrail_questions(Path("evaluation/guardrail_questions.json"))
    retrieval = load_questions(Path("evaluation/retrieval_questions.json"))
    counts = {label: sum(1 for item in guardrails if item.expected == label) for label in ("supported", "ambiguous", "unsupported")}

    assert counts["supported"] >= 20
    assert counts["ambiguous"] >= 15
    assert counts["unsupported"] >= 15
    assert retrieval[0].id == "rohs-concentration-list"
    assert retrieval[0].expected[0].annex == "II"
    assert {item.question for item in guardrails}.isdisjoint({item.question for item in retrieval})


def test_metric_definitions_on_a_fixed_set() -> None:
    class _Fixed:
        def review(self, question: str):
            if question == "supported":
                return GuardrailService(retrieval=_EmptyRetriever()).review("What does REACH Article 6 require?")
            if question == "ambiguous":
                return GuardrailService(retrieval=_EmptyRetriever()).review("What are the restrictions on lead?")
            return GuardrailService(retrieval=_EmptyRetriever()).review("What is the capital of France?")

    report = evaluate_guardrails(
        _Fixed(),  # type: ignore[arg-type]
        [
            GuardrailExample(id="s", question="supported", expected="supported", expected_regulation="REACH"),
            GuardrailExample(id="a", question="ambiguous", expected="ambiguous"),
            GuardrailExample(id="u", question="unsupported", expected="unsupported"),
        ],
    )

    assert report.supported_classification_rate == 1.0
    assert report.supported_acceptance_rate == 0.0
    assert report.insufficient_evidence_on_supported == 1
    assert report.ambiguous_clarification_rate == 1.0
    assert report.unsupported_rejection_rate == 1.0
    assert report.false_acceptance_rate == 0.0
    assert report.false_rejection_rate == 0.0
    assert report.confusion_matrix["supported"]["supported"] == 1
    assert report.confusion_matrix["ambiguous"]["ambiguous"] == 1
    assert report.confusion_matrix["unsupported"]["unsupported"] == 1
    assert report.failures == []


@pytest.mark.skipif(
    not Path("storage/qdrant").exists() or not Path("storage/bm25/index.json").exists(),
    reason="The local indexes are created by earlier phases and are not in Git.",
)
def test_real_guardrail_does_not_change_the_collection() -> None:
    from app.indexing.store import QdrantStore
    from app.retrieval.settings import RetrievalSettings

    settings = RetrievalSettings()
    before = QdrantStore(settings)
    before.open_existing()
    try:
        count_before = before.count()
    finally:
        before.close()

    service = GuardrailService()
    try:
        response = service.review("What are the restricted substances under RoHS?")
    finally:
        service.close()

    after = QdrantStore(settings)
    after.open_existing()
    try:
        count_after = after.count()
    finally:
        after.close()

    assert count_before == 2018
    assert count_after == 2018
    assert response.detected_regulation == "RoHS"
    assert response.status in {"approved", "insufficient_evidence"}


class _Scripted:
    def __init__(self, chunks: list[RetrievedChunk], honor_filters: bool = True) -> None:
        self.chunks = chunks
        self.honor_filters = honor_filters
        self.dense_thresholds: list[bool] = []
        self.questions: list[str] = []

    def search(self, question: str, filters: SearchFilters | None = None, top_k: int | None = None, minimum_similarity: float | None = None, *, apply_threshold: bool = True) -> RetrievalResponse:
        self.dense_thresholds.append(apply_threshold)
        self.questions.append(question)
        return self._response(question, filters, "dense")

    def search_bm25(self, question: str, filters: SearchFilters | None = None, top_k: int | None = None) -> RetrievalResponse:
        self.questions.append(question)
        return self._response(question, filters, "bm25")

    def search_hybrid(self, question: str, filters: SearchFilters | None = None, top_k: int | None = None, dense_depth: int | None = None, sparse_depth: int | None = None, rrf_k: int | None = None) -> RetrievalResponse:
        self.questions.append(question)
        return self._response(question, filters, "hybrid")

    def _response(self, question: str, filters: SearchFilters | None, method: str) -> RetrievalResponse:
        chunks = self.chunks
        if self.honor_filters and filters is not None:
            chunks = [chunk for chunk in chunks if _accepts(chunk, filters)]
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


class _EmptyRetriever(_Scripted):
    def __init__(self) -> None:
        super().__init__([])

    @property
    def calls(self) -> list[str]:
        return self.questions


def _accepts(chunk: RetrievedChunk, filters: SearchFilters) -> bool:
    if filters.regulation is not None and chunk.regulation != filters.regulation:
        return False
    if filters.article is not None and chunk.article != filters.article:
        return False
    if filters.annex is not None and chunk.annex != filters.annex:
        return False
    return True


def _chunk(
    chunk_id: str,
    regulation: str,
    text: str | None = None,
    article: str | None = None,
    annex: str | None = None,
    bm25: float | None = None,
    dense: float | None = None,
    rrf: float | None = None,
    page_start: int | None = 8,
    source_filename: str | None = "source.pdf",
    consolidation_date: str | None = "2026-07-01",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=regulation.lower(),
        regulation=regulation,
        text=text or chunk_id,
        similarity_score=dense,
        dense_score=dense,
        bm25_score=bm25,
        rrf_score=rrf,
        article=article,
        annex=annex,
        page_start=page_start,
        page_end=page_start,
        consolidation_date=consolidation_date,
        source_filename=source_filename,
    )
