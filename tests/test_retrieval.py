"""Tests for semantic retrieval, filters, and evaluation metrics."""

from datetime import date
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from app.indexing.errors import CollectionCompatibilityError
from app.indexing.store import QdrantStore
from app.processing.models import ChunkRecord
from app.retrieval.errors import InvalidFilterError, InvalidQuestionError, RetrievalError
from app.retrieval.evaluate import (
    EvaluationQuestion,
    ExpectedReference,
    evaluate_questions,
    matches_reference,
)
from app.retrieval.filters import SearchFilters
from app.retrieval.models import RetrievalResponse, RetrievedChunk
from app.retrieval.service import RetrievalService, validate_question
from app.retrieval.settings import RetrievalSettings


class DirectionEmbedder:
    """Map a few words onto fixed orthogonal vectors."""

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        if "lead" in lowered:
            return _direction(0)
        if "registration" in lowered:
            return _direction(1)
        if "mercury" in lowered:
            return _direction(2)
        if len(text) < 3:
            return [0.0]
        return _direction(3)


def test_query_embedding_is_validated(tmp_path: Path) -> None:
    service, store = _service(tmp_path)
    store.open()
    try:
        _seed(store)
        response = service.search("lead limit", apply_threshold=False)
    finally:
        service.close()

    assert response.results
    assert len(DirectionEmbedder().embed_query("lead limit")) == 384


def test_short_embedding_is_rejected(tmp_path: Path) -> None:
    service, store = _service(tmp_path)
    store.open()
    _seed(store)
    try:
        with pytest.raises(RetrievalError, match="dimensions"):
            service.search("no")
    finally:
        service.close()


def test_empty_and_invalid_questions_are_rejected() -> None:
    with pytest.raises(InvalidQuestionError):
        validate_question("")
    with pytest.raises(InvalidQuestionError):
        validate_question("   ")
    with pytest.raises(InvalidQuestionError):
        validate_question(None)


def test_similarity_retrieval_top_k_and_metadata(tmp_path: Path) -> None:
    service, store = _service(tmp_path)
    store.open()
    _seed(store)
    try:
        response = service.search("lead limit", top_k=2, apply_threshold=False)
    finally:
        service.close()

    assert response.status == "ok"
    assert len(response.results) == 2
    assert response.results[0].chunk_id == "rohs:lead"
    assert response.results[0].document_id == "rohs"
    assert response.results[0].regulation == "RoHS"
    assert response.results[0].article == "4"
    assert response.results[0].annex is None
    assert response.results[0].page_start == 8
    assert response.results[0].similarity_score == pytest.approx(1.0, abs=1e-6)
    RetrievalResponse.model_validate(response.model_dump())


def test_regulation_article_and_annex_filters(tmp_path: Path) -> None:
    service, store = _service(tmp_path)
    store.open()
    _seed(store)
    try:
        rohs = service.search("lead limit", filters=SearchFilters(regulation="RoHS"), apply_threshold=False)
        reach = service.search(
            "lead limit",
            filters=SearchFilters(regulation="REACH"),
            apply_threshold=False,
        )
        article = service.search(
            "lead limit",
            filters=SearchFilters(article="4"),
            top_k=5,
            apply_threshold=False,
        )
        annex = service.search(
            "mercury lamps",
            filters=SearchFilters(annex="III"),
            apply_threshold=False,
        )
        empty = service.search(
            "lead limit",
            filters=SearchFilters(article="999"),
            apply_threshold=False,
        )
    finally:
        service.close()

    assert {item.regulation for item in rohs.results} == {"RoHS"}
    assert reach.results
    assert {item.regulation for item in reach.results} == {"REACH"}
    assert {item.article for item in article.results} == {"4"}
    assert annex.results[0].annex == "III"
    assert empty.status == "insufficient_evidence"
    assert empty.best_score is None


def test_filter_values_are_validated() -> None:
    assert SearchFilters(regulation="rohs").regulation == "RoHS"
    with pytest.raises(InvalidFilterError):
        SearchFilters(regulation="GDPR")
    with pytest.raises(InvalidFilterError):
        SearchFilters(article="  ")


def test_threshold_rejects_a_weak_match(tmp_path: Path) -> None:
    service, store = _service(tmp_path, minimum_similarity=0.9)
    store.open()
    _seed(store)
    try:
        weak = service.search("unrelated gardening")
        strong = service.search("lead limit")
    finally:
        service.close()

    assert weak.status == "insufficient_evidence"
    assert weak.results == []
    assert weak.best_score is not None
    assert weak.best_score < 0.9
    assert strong.status == "ok"
    assert strong.results[0].chunk_id == "rohs:lead"


def test_missing_storage_is_not_created(tmp_path: Path) -> None:
    missing = tmp_path / "absent" / "qdrant"
    settings = _settings(tmp_path, minimum_similarity=None).model_copy(update={"qdrant_path": missing})
    service = RetrievalService(settings, embeddings=DirectionEmbedder())

    with pytest.raises(RetrievalError, match="not found"):
        service.search("lead limit", apply_threshold=False)

    assert not missing.exists()


def test_incompatible_collection_is_left_unchanged(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.qdrant_path.mkdir(parents=True)
    client = QdrantClient(path=str(settings.qdrant_path))
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=8, distance=Distance.COSINE),
    )
    client.close()
    service = RetrievalService(settings, embeddings=DirectionEmbedder())

    with pytest.raises(RetrievalError, match="was not modified"):
        service.search("lead limit", apply_threshold=False)

    client = QdrantClient(path=str(settings.qdrant_path))
    try:
        info = client.get_collection(settings.qdrant_collection)
        assert info.config.params.vectors.size == 8
    finally:
        client.close()


def test_qdrant_search_errors_are_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service, store = _service(tmp_path)
    store.open()
    _seed(store)

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("storage locked")

    assert store.client is not None
    monkeypatch.setattr(store.client, "query_points", fail)
    try:
        with pytest.raises(RetrievalError, match="Qdrant search failed"):
            service.search("lead limit", apply_threshold=False)
    finally:
        service.close()


def test_metric_definitions_and_small_evaluation(tmp_path: Path) -> None:
    lead = RetrievedChunk(
        chunk_id="rohs:lead",
        document_id="rohs",
        regulation="RoHS",
        text="lead",
        similarity_score=0.8,
        article="4",
        page_start=8,
        page_end=8,
    )
    other = RetrievedChunk(
        chunk_id="other",
        document_id="rohs",
        regulation="RoHS",
        text="other",
        similarity_score=0.2,
        article="9",
        page_start=1,
        page_end=1,
    )
    reference = ExpectedReference(regulation="RoHS", article="4")
    assert matches_reference(lead, reference)
    assert not matches_reference(other, reference)

    service, store = _service(tmp_path, minimum_similarity=0.5)
    store.open()
    _seed(store)
    questions = [
        EvaluationQuestion(
            id="lead",
            question="lead limit",
            category="rohs",
            expected=[ExpectedReference(chunk_id="rohs:lead")],
        ),
        EvaluationQuestion(
            id="weather",
            question="unrelated gardening",
            category="unrelated",
            expected=[],
        ),
    ]
    try:
        report = evaluate_questions(service, questions, top_k=2)
    finally:
        service.close()

    assert report.recall_at_k == 1.0
    assert report.mean_reciprocal_rank == 1.0
    assert report.precision_at_k == 0.5
    assert report.off_topic_questions == 1
    assert report.off_topic_rejected == 1
    assert report.failures == []


@pytest.mark.skipif(
    not Path("storage/qdrant").exists(),
    reason="The local Qdrant collection is created by indexing and is not in Git.",
)
def test_real_collection_retrieval_does_not_change_the_index() -> None:
    from app.indexing.settings import IndexingSettings

    settings = RetrievalSettings()
    indexing = IndexingSettings()
    before = QdrantStore(indexing)
    before.open_existing()
    try:
        count_before = before.count()
    finally:
        before.close()

    service = RetrievalService(settings)
    try:
        lead = service.search("Which substances does RoHS restrict in electrical equipment?", top_k=3)
        filtered = service.search(
            "What are the restrictions on lead?",
            filters=SearchFilters(regulation="RoHS"),
            top_k=3,
            apply_threshold=False,
        )
        unrelated = service.search("What is the capital of France?", apply_threshold=False)
    finally:
        service.close()

    after = QdrantStore(indexing)
    after.open_existing()
    try:
        count_after = after.count()
    finally:
        after.close()

    assert count_before == 2018
    assert count_after == 2018
    assert lead.results
    assert lead.results[0].chunk_id
    assert lead.results[0].text
    assert lead.results[0].similarity_score > unrelated.best_score
    assert filtered.results
    assert {item.regulation for item in filtered.results} == {"RoHS"}
    assert unrelated.best_score is not None


def _service(tmp_path: Path, minimum_similarity: float | None = 0.35) -> tuple[RetrievalService, QdrantStore]:
    settings = _settings(tmp_path, minimum_similarity)
    store = QdrantStore(settings)
    service = RetrievalService(settings, embeddings=DirectionEmbedder(), store=store)
    return service, store


def _settings(tmp_path: Path, minimum_similarity: float | None = 0.35) -> RetrievalSettings:
    return RetrievalSettings(
        embedding_dimensions=384,
        qdrant_path=tmp_path / "qdrant",
        qdrant_collection="regulatory_documents",
        retrieval_top_k=5,
        retrieval_min_similarity=minimum_similarity,
    )


def _seed(store: QdrantStore) -> None:
    chunks = [
        _chunk("rohs:lead", "RoHS", "Lead shall not exceed the Annex II limit.", article="4"),
        _chunk("rohs:lead-2", "RoHS", "A second lead paragraph.", article="4", paragraph="2"),
        _chunk("rohs:lead-3", "RoHS", "A third lead paragraph.", article="4", paragraph="3"),
        _chunk("reach:register", "REACH", "Registration is required.", article="6"),
        _chunk("rohs:mercury", "RoHS", "Mercury lamp exemption.", annex="III", article=None),
    ]
    second = _direction(0)
    second[10] = 0.3
    third = _direction(0)
    third[11] = 0.5
    vectors = [_direction(0), second, third, _direction(1), _direction(2)]
    store.upsert(chunks, vectors)


def _direction(index: int) -> list[float]:
    vector = [0.0] * 384
    vector[index] = 1.0
    return vector


def _chunk(
    chunk_id: str,
    regulation: str,
    text: str,
    article: str | None = None,
    paragraph: str | None = None,
    annex: str | None = None,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=regulation.lower(),
        regulation=regulation,
        document_title=f"{regulation} title",
        regulation_number="2011/65/EU" if regulation == "RoHS" else "(EC) No 1907/2006",
        consolidation_date=date(2026, 7, 1),
        chapter=None,
        title=None,
        article=article,
        paragraph=paragraph,
        subparagraph=None,
        annex=annex,
        annex_section="Lamps" if annex else None,
        page_start=8,
        page_end=8,
        source_filename=f"{regulation.lower()}.pdf",
        text=text,
        token_count=8,
    )
