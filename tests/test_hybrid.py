"""Tests for BM25 indexing, reciprocal rank fusion, and evaluation consistency."""

from datetime import date
from pathlib import Path

import pytest

from app.indexing.store import QdrantStore
from app.processing.models import ChunkRecord
from app.retrieval.bm25 import BM25Index, document_from_chunk, document_from_mapping
from app.retrieval.errors import InvalidQuestionError, RetrievalError
from app.retrieval.evaluate import (
    compare_methods,
    evaluate_questions,
    load_baseline,
    load_questions,
)
from app.retrieval.filters import SearchFilters
from app.retrieval.models import RetrievalResponse, RetrievedChunk
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.service import RetrievalService
from app.retrieval.settings import RetrievalSettings


def test_bm25_indexes_original_text_and_retrieves_the_matching_chunk(tmp_path: Path) -> None:
    path = _write_chunks(tmp_path / "chunks.jsonl", [_sample("rohs:lead", "RoHS", "Lead shall not exceed the Annex II limit.", article="4")])
    index = BM25Index.build([path])
    results = index.search("Annex II lead limit", limit=5)

    assert len(index.documents) == 1
    assert results[0].chunk_id == "rohs:lead"
    assert results[0].text.startswith("Lead shall not exceed")
    assert results[0].article == "4"
    assert results[0].annex is None
    assert results[0].bm25_score is not None
    assert results[0].bm25_score > 0
    assert results[0].similarity_score is None
    assert results[0].rrf_score is None


def test_bm25_filters_regulation_article_and_annex(tmp_path: Path) -> None:
    path = _write_chunks(
        tmp_path / "chunks.jsonl",
        [
            _sample("rohs:lead", "RoHS", "Lead shall not exceed the Annex II limit.", article="4"),
            _sample("reach:gunshot", "REACH", "Restricting lead in gunshot.", annex="XVII", article=None),
        ],
    )
    index = BM25Index.build([path])

    rohs = index.search("lead", limit=5, filters=SearchFilters(regulation="RoHS"))
    article = index.search("lead", limit=5, filters=SearchFilters(article="4"))
    annex = index.search("lead", limit=5, filters=SearchFilters(annex="XVII"))
    empty = index.search("lead", limit=5, filters=SearchFilters(article="999"))

    assert [item.chunk_id for item in rohs] == ["rohs:lead"]
    assert [item.chunk_id for item in article] == ["rohs:lead"]
    assert [item.chunk_id for item in annex] == ["reach:gunshot"]
    assert empty == []


def test_bm25_index_is_persistent_and_skips_duplicate_ids(tmp_path: Path) -> None:
    first = _write_chunks(tmp_path / "a.jsonl", [_sample("rohs:lead", "RoHS", "Lead limit in Annex II.", article="4")])
    second = _write_chunks(tmp_path / "b.jsonl", [_sample("rohs:lead", "RoHS", "A different lead sentence.", article="9")])
    index = BM25Index.build([first, second])
    destination = tmp_path / "bm25" / "index.json"
    index.save(destination)
    loaded = BM25Index.load(destination)

    assert len(index.documents) == 1
    assert loaded.documents[0].text == "Lead limit in Annex II."
    assert loaded.search("Annex", limit=1)[0].chunk_id == "rohs:lead"
    assert not (tmp_path / "qdrant").exists()


def test_missing_identity_is_not_invented() -> None:
    assert document_from_mapping({"text": "Lead is restricted."}) is None
    kept = document_from_mapping(
        {
            "chunk_id": "rohs:lead",
            "document_id": "rohs",
            "regulation": "RoHS",
            "text": "Lead is restricted.",
        }
    )
    assert kept is not None
    assert kept.article is None
    assert kept.annex is None
    assert kept.page_start is None


def test_invalid_chunk_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(RetrievalError, match="not a valid chunk"):
        BM25Index.build([path])


def test_rrf_uses_ranks_and_removes_duplicates() -> None:
    lead = _hit("rohs:lead", dense=0.2, bm25=None)
    gunshot = _hit("reach:gunshot", dense=None, bm25=9.0)
    lead_sparse = _hit("rohs:lead", dense=None, bm25=1.0)
    fused = reciprocal_rank_fusion([[lead], [gunshot, lead_sparse]], k=60, limit=5)

    assert [item.chunk_id for item in fused] == ["rohs:lead", "reach:gunshot"]
    assert fused[0].rrf_score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[0].dense_score == 0.2
    assert fused[0].similarity_score == 0.2
    assert fused[0].bm25_score == 1.0
    assert fused[0].rrf_score != fused[0].bm25_score
    assert fused[1].similarity_score is None
    assert fused[1].bm25_score == 9.0
    assert fused[1].rrf_score == pytest.approx(1 / 61)


def test_service_bm25_hybrid_filters_and_empty_questions(tmp_path: Path) -> None:
    chunks = [
        _sample("rohs:lead", "RoHS", "Lead shall not exceed the Annex II limit.", article="4"),
        _sample("reach:gunshot", "REACH", "Restricting lead in gunshot.", annex="XVII", article=None),
    ]
    service, store = _service(tmp_path, chunks)
    store.open()
    _seed(store, chunks)
    try:
        with pytest.raises(InvalidQuestionError):
            service.search_bm25("   ")
        bm25 = service.search_bm25("lead gunshot", top_k=2)
        hybrid = service.search_hybrid("lead gunshot", top_k=2, dense_depth=1, sparse_depth=2, rrf_k=60)
        filtered = service.search_bm25("lead", filters=SearchFilters(regulation="RoHS"))
        empty = service.search_hybrid("lead", filters=SearchFilters(article="999"), dense_depth=5, sparse_depth=5)
    finally:
        service.close()

    assert bm25.method == "bm25"
    assert bm25.results[0].chunk_id == "reach:gunshot"
    assert bm25.results[0].similarity_score is None
    assert hybrid.method == "hybrid"
    assert [item.chunk_id for item in hybrid.results] == ["rohs:lead", "reach:gunshot"]
    assert hybrid.results[0].rrf_score == pytest.approx(1 / 61 + 1 / 62)
    assert hybrid.best_score == hybrid.best_rrf_score
    assert hybrid.results[0].similarity_score == pytest.approx(hybrid.results[0].dense_score)
    assert hybrid.results[1].dense_score is None
    assert [item.chunk_id for item in filtered.results] == ["rohs:lead"]
    assert empty.status == "insufficient_evidence"
    assert empty.results == []


def test_recorded_baseline_and_expected_references_stay_fixed() -> None:
    baseline = load_baseline(Path("evaluation/baseline_dense.json"))
    assert baseline == {
        "recall_at_k": 0.4,
        "precision_at_k": 0.18,
        "mean_reciprocal_rank": 0.2533,
        "off_topic_rejection_rate": 0.5,
    }
    questions = {item.id: item for item in load_questions(Path("evaluation/retrieval_questions.json"))}
    assert questions["rohs-concentration-list"].expected[0].annex == "II"
    assert questions["rohs-market-prevention"].expected[1].chunk_id == "02011L0065-20260701:00022"
    assert questions["reach-prior-approval"].expected[0].annex == "XIV"
    assert questions["rohs-article-4-wording"].expected[0].chunk_id == "02011L0065-20260701:00022"
    assert questions["outside-gdpr"].expected == []


def test_method_comparison_uses_the_same_metrics(tmp_path: Path) -> None:
    from app.retrieval.evaluate import EvaluationQuestion

    service = _Stub(tmp_path)
    questions = [
        EvaluationQuestion(
            id="lead",
            question="lead limit",
            category="rohs",
            expected=questions_expected_lead(),
        ),
        EvaluationQuestion(id="weather", question="unrelated gardening", category="unrelated", expected=[]),
    ]
    dense = evaluate_questions(service, questions, top_k=2)  # type: ignore[arg-type]
    comparison = compare_methods(
        service,  # type: ignore[arg-type]
        questions,
        top_k=2,
        baseline_recall=1.0,
        baseline_precision=0.5,
        baseline_mrr=1.0,
        baseline_off_topic_rejection=1.0,
    )
    dense_method = comparison.methods[0]
    assert dense.recall_at_k == dense_method.recall_at_k == 1.0
    assert dense.precision_at_k == dense_method.precision_at_k == 0.5
    assert dense.mean_reciprocal_rank == dense_method.mean_reciprocal_rank == 1.0
    assert dense.off_topic_rejected == dense_method.off_topic_rejected == 1
    assert comparison.changes[0].bm25_versus_dense == "unchanged"
    assert comparison.changes[0].hybrid_versus_dense == "unchanged"
    assert comparison.dense_matches_recorded_baseline


@pytest.mark.skipif(
    not Path("storage/qdrant").exists() or not Path("data/processed/rohs/chunks.jsonl").exists(),
    reason="The local Qdrant collection and chunk files are not in Git.",
)
def test_real_hybrid_search_leaves_qdrant_unchanged(tmp_path: Path) -> None:
    settings = RetrievalSettings(bm25_index_path=tmp_path / "bm25.json")
    before = QdrantStore(settings)
    before.open_existing()
    try:
        count_before = before.count()
    finally:
        before.close()

    index = BM25Index.build(
        [settings.rohs_chunks_path, settings.reach_chunks_path],
        k1=settings.bm25_k1,
        b=settings.bm25_b,
    )
    service = RetrievalService(settings, bm25=index)
    try:
        hybrid = service.search_hybrid(
            "Where are the time-limited exceptions for mercury in lamps written down?",
            top_k=5,
        )
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
    assert len(index.documents) == 2018
    assert hybrid.method == "hybrid"
    assert hybrid.results
    assert hybrid.results[0].rrf_score == hybrid.best_rrf_score
    assert hybrid.best_score == hybrid.best_rrf_score


class _Stub:
    """Return fixed rankings so metric code can be compared without Qdrant."""

    def __init__(self, tmp_path: Path) -> None:
        self.settings = RetrievalSettings(
            qdrant_path=tmp_path / "qdrant",
            bm25_index_path=tmp_path / "bm25.json",
            retrieval_min_similarity=0.35,
            retrieval_top_k=2,
        )

    def search(self, question: str, top_k: int = 5, apply_threshold: bool = True, **kwargs: object) -> RetrievalResponse:
        gardening = "gardening" in question
        score = 0.1 if gardening else 0.9
        hit = _hit("rohs:lead", dense=score, bm25=None)
        if apply_threshold and score < 0.35:
            return RetrievalResponse(
                status="insufficient_evidence",
                question=question,
                method="dense",
                top_k=top_k,
                minimum_similarity=0.35,
                best_score=score,
                best_dense_score=score,
                results=[],
                filters=SearchFilters(),
                message="below cutoff",
            )
        return RetrievalResponse(
            status="ok",
            question=question,
            method="dense",
            top_k=top_k,
            minimum_similarity=0.35 if apply_threshold else None,
            best_score=score,
            best_dense_score=score,
            results=[hit],
            filters=SearchFilters(),
        )

    def search_bm25(self, question: str, top_k: int = 5, **kwargs: object) -> RetrievalResponse:
        if "gardening" in question:
            return RetrievalResponse(
                status="insufficient_evidence",
                question=question,
                method="bm25",
                top_k=top_k,
                minimum_similarity=None,
                best_score=None,
                results=[],
                filters=SearchFilters(),
                message="no overlap",
            )
        hit = _hit("rohs:lead", dense=None, bm25=2.0)
        return RetrievalResponse(
            status="ok",
            question=question,
            method="bm25",
            top_k=top_k,
            minimum_similarity=None,
            best_score=2.0,
            best_bm25_score=2.0,
            results=[hit],
            filters=SearchFilters(),
        )

    def search_hybrid(self, question: str, top_k: int = 5, **kwargs: object) -> RetrievalResponse:
        hit = _hit("rohs:lead", dense=0.9, bm25=2.0)
        hit = hit.model_copy(update={"rrf_score": 1 / 61})
        return RetrievalResponse(
            status="ok",
            question=question,
            method="hybrid",
            top_k=top_k,
            minimum_similarity=None,
            best_score=hit.rrf_score,
            best_rrf_score=hit.rrf_score,
            results=[hit],
            filters=SearchFilters(),
        )


def questions_expected_lead():
    from app.retrieval.evaluate import ExpectedReference

    return [ExpectedReference(chunk_id="rohs:lead")]


def _service(tmp_path: Path, chunks: list[ChunkRecord]) -> tuple[RetrievalService, QdrantStore]:
    settings = RetrievalSettings(
        embedding_dimensions=384,
        qdrant_path=tmp_path / "qdrant",
        qdrant_collection="regulatory_documents",
        bm25_index_path=tmp_path / "bm25.json",
        retrieval_top_k=5,
        retrieval_min_similarity=0.35,
    )
    index = BM25Index([document for chunk in chunks if (document := document_from_chunk(chunk)) is not None])
    store = QdrantStore(settings)
    service = RetrievalService(settings, embeddings=_Embedder(), store=store, bm25=index)
    return service, store


class _Embedder:
    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * 384
        vector[0 if "lead" in text.lower() else 1] = 1.0
        return vector


def _seed(store: QdrantStore, chunks: list[ChunkRecord]) -> None:
    vectors = []
    for chunk in chunks:
        vector = [0.0] * 384
        vector[0 if chunk.chunk_id == "rohs:lead" else 3] = 1.0
        vectors.append(vector)
    store.upsert(chunks, vectors)


def _write_chunks(path: Path, chunks: list[ChunkRecord]) -> Path:
    path.write_text("".join(chunk.model_dump_json() + "\n" for chunk in chunks), encoding="utf-8")
    return path


def _sample(
    chunk_id: str,
    regulation: str,
    text: str,
    article: str | None = None,
    annex: str | None = None,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=regulation.lower(),
        regulation=regulation,
        document_title=f"{regulation} title",
        regulation_number="2011/65/EU",
        consolidation_date=date(2026, 7, 1),
        article=article,
        annex=annex,
        annex_section=None,
        page_start=8,
        page_end=8,
        source_filename=f"{regulation.lower()}.pdf",
        text=text,
        token_count=4,
    )


def _hit(chunk_id: str, dense: float | None, bm25: float | None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=chunk_id.split(":")[0],
        regulation="RoHS" if chunk_id.startswith("rohs") else "REACH",
        text=chunk_id,
        similarity_score=dense,
        dense_score=dense,
        bm25_score=bm25,
        article="4" if chunk_id == "rohs:lead" else None,
        page_start=8,
        page_end=8,
    )
