"""Tests for embeddings, Qdrant point ids, and indexing."""

from datetime import date
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from app.indexing.cache import EmbeddingCache
from app.indexing.embeddings import SentenceTransformerEmbeddings, validate_embedding
from app.indexing.errors import CollectionCompatibilityError, EmbeddingError
from app.indexing.ids import point_id_for_chunk
from app.indexing.indexer import index_regulatory_documents
from app.indexing.payload import chunk_to_payload
from app.indexing.settings import IndexingSettings
from app.indexing.store import QdrantStore
from app.processing.models import ChunkRecord


class FakeModel:
    """Stand-in for SentenceTransformer.encode."""

    def __init__(self, dimensions: int = 384, value: float = 0.25) -> None:
        self.dimensions = dimensions
        self.value = value
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append(list(texts))
        rows = []
        for _text in texts:
            row = np.full(self.dimensions, self.value, dtype=float)
            rows.append(row)
        return np.vstack(rows)


class FailingEmbedder:
    """Fail one batch and embed the others."""

    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if any(text == "bad" for text in texts):
            raise EmbeddingError("inference failed")
        return [[0.1] * 384 for _text in texts]


def test_embedding_dimensions_and_validation() -> None:
    embedder = _embedder()
    vectors = embedder.embed_documents(["Article 1 applies."])

    assert len(vectors) == 1
    assert len(vectors[0]) == 384
    assert all(np.isfinite(value) for value in vectors[0])
    validate_embedding(vectors[0], 384)


def test_empty_and_invalid_text_is_rejected() -> None:
    embedder = _embedder()

    with pytest.raises(EmbeddingError):
        embedder.embed_documents([""])
    with pytest.raises(EmbeddingError):
        embedder.embed_documents(["   "])
    with pytest.raises(EmbeddingError):
        embedder.embed_documents(["ok", ""])


def test_non_finite_and_wrong_size_embeddings_are_rejected() -> None:
    with pytest.raises(EmbeddingError):
        validate_embedding([float("nan")] * 384, 384)
    with pytest.raises(EmbeddingError):
        validate_embedding([0.0, 1.0], 384)
    with pytest.raises(EmbeddingError):
        validate_embedding([], 384)


def test_model_load_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    import sentence_transformers

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("model download failed")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fail)
    embedder = SentenceTransformerEmbeddings(_settings(Path("unused")))

    with pytest.raises(EmbeddingError, match="Failed to load embedding model"):
        embedder.embed_documents(["Article 1"])


def test_embedding_cache_avoids_a_second_encode(tmp_path: Path) -> None:
    model = FakeModel()
    embedder = _embedder(model)
    embedder.cache = EmbeddingCache(tmp_path / "vectors.jsonl")

    first = embedder.embed_documents(["same text"])
    second = embedder.embed_documents(["same text"])

    assert first == second
    assert len(model.calls) == 1
    assert embedder.cache_hits == 1


def test_point_ids_are_deterministic_uuids() -> None:
    first = point_id_for_chunk("02011L0065-20260701:00005")
    second = point_id_for_chunk("02011L0065-20260701:00005")
    other = point_id_for_chunk("02006R1907-20260622:00007")

    assert first == second
    assert first != other
    assert len(first) == 36


def test_payload_keeps_null_metadata_and_iso_date() -> None:
    payload = chunk_to_payload(_chunk("02011L0065-20260701:00005", "RoHS", article=None))

    assert payload["page_content"].startswith("Article 1")
    assert payload["metadata"]["chunk_id"] == "02011L0065-20260701:00005"
    assert payload["metadata"]["article"] is None
    assert payload["metadata"]["annex"] is None
    assert payload["metadata"]["consolidation_date"] == "2026-07-01"
    assert payload["metadata"]["page_start"] == 5


def test_batch_indexing_is_idempotent_and_preserves_metadata(tmp_path: Path) -> None:
    root = _write_chunks(tmp_path)
    settings = _settings(tmp_path)
    embedder = _embedder()

    first = index_regulatory_documents(root, settings, embeddings=embedder)
    calls_after_first = len(embedder._model.calls)  # type: ignore[attr-defined]
    second = index_regulatory_documents(root, settings, embeddings=embedder)

    assert first.verification_passed is True
    assert first.total_vectors_stored == 3
    assert first.rohs_chunks_indexed == 2
    assert first.reach_chunks_indexed == 1
    assert first.failed_chunks == []
    assert first.skipped_chunks == []
    assert second.total_vectors_stored == 3
    assert second.source_unchanged is True
    assert len(embedder._model.calls) == calls_after_first  # type: ignore[attr-defined]

    store = QdrantStore(settings)
    store.open()
    try:
        stored = store.retrieve_payload("02011L0065-20260701:00005")
    finally:
        store.close()
    assert stored is not None
    assert stored["metadata"]["article"] == "1"
    assert stored["metadata"]["paragraph"] is None
    assert stored["metadata"]["source_filename"].endswith(".pdf")
    assert stored["page_content"].startswith("Article 1")


def test_existing_collection_with_wrong_dimensions_is_left_unchanged(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.qdrant_path.mkdir(parents=True)
    client = QdrantClient(path=str(settings.qdrant_path))
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=8, distance=Distance.COSINE),
    )
    client.close()

    store = QdrantStore(settings)
    with pytest.raises(CollectionCompatibilityError, match="was not modified"):
        store.open()
    store.close()

    client = QdrantClient(path=str(settings.qdrant_path))
    try:
        info = client.get_collection(settings.qdrant_collection)
        assert info.config.params.vectors.size == 8
    finally:
        client.close()


def test_failed_embedding_is_recorded_and_not_stored(tmp_path: Path) -> None:
    root = _write_chunks(tmp_path, include_bad=True)
    settings = _settings(tmp_path).model_copy(update={"embedding_batch_size": 1})
    report = index_regulatory_documents(root, settings, embeddings=FailingEmbedder())

    assert report.verification_passed is False
    assert report.failed_chunks == ["bad:00001"]
    assert report.total_vectors_stored == 2
    store = QdrantStore(_settings(tmp_path))
    store.open()
    try:
        assert store.retrieve_payload("bad:00001") is None
        assert store.retrieve_payload("02011L0065-20260701:00005") is not None
    finally:
        store.close()


def test_empty_chunk_is_skipped(tmp_path: Path) -> None:
    root = tmp_path / "project"
    directory = root / "chunks"
    directory.mkdir(parents=True)
    _write_jsonl(
        directory / "rohs.jsonl",
        [_chunk("02011L0065-20260701:00001", "RoHS", text="   ")],
    )
    _write_jsonl(
        directory / "reach.jsonl",
        [_chunk("02006R1907-20260622:00001", "REACH", text="Article 1 applies.")],
    )
    settings = _settings(tmp_path).model_copy(
        update={
            "rohs_chunks_path": directory / "rohs.jsonl",
            "reach_chunks_path": directory / "reach.jsonl",
        }
    )

    report = index_regulatory_documents(root, settings, embeddings=_embedder())

    assert report.skipped_chunks == ["02011L0065-20260701:00001"]
    assert report.reach_chunks_indexed == 1
    assert report.rohs_chunks_indexed == 0
    assert report.total_vectors_stored == 1
    assert report.verification_passed is True


@pytest.mark.skipif(
    not Path("data/processed/rohs/chunks.jsonl").is_file()
    or not Path("data/processed/reach/chunks.jsonl").is_file(),
    reason="Official chunk files are stored locally and are not in Git.",
)
def test_real_rohs_and_reach_chunks_are_indexed(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    rohs = root / "data" / "processed" / "rohs" / "chunks.jsonl"
    reach = root / "data" / "processed" / "reach" / "chunks.jsonl"
    before = (rohs.read_bytes(), reach.read_bytes())
    settings = IndexingSettings(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimensions=384,
        embedding_batch_size=64,
        embedding_device="cpu",
        qdrant_path=tmp_path / "qdrant",
        qdrant_collection="regulatory_documents",
        rohs_chunks_path=rohs,
        reach_chunks_path=reach,
    )

    report = index_regulatory_documents(root, settings)

    assert (rohs.read_bytes(), reach.read_bytes()) == before
    assert report.verification_passed is True
    assert report.failed_chunks == []
    assert report.skipped_chunks == []
    assert report.rohs_chunks_indexed == 273
    assert report.reach_chunks_indexed == 1745
    assert report.total_vectors_stored == 2018
    assert report.embedding_dimensions == 384
    assert report.distance == "cosine"
    store = QdrantStore(settings)
    store.open()
    try:
        payload = store.retrieve_payload("02011L0065-20260701:00005")
    finally:
        store.close()
    assert payload is not None
    assert payload["metadata"]["regulation"] == "RoHS"
    assert payload["metadata"]["article"] == "1"
    assert payload["metadata"]["page_start"] == 5
    assert "Subject matter" in payload["page_content"]


def _embedder(model: FakeModel | None = None) -> SentenceTransformerEmbeddings:
    embedder = SentenceTransformerEmbeddings(_settings(Path("unused-settings")))
    embedder._model = model or FakeModel()
    return embedder


def _settings(tmp_path: Path) -> IndexingSettings:
    return IndexingSettings(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimensions=384,
        embedding_batch_size=2,
        embedding_device="cpu",
        qdrant_path=tmp_path / "qdrant",
        qdrant_collection="regulatory_documents",
        rohs_chunks_path=tmp_path / "project" / "chunks" / "rohs.jsonl",
        reach_chunks_path=tmp_path / "project" / "chunks" / "reach.jsonl",
    )


def _write_chunks(tmp_path: Path, include_bad: bool = False) -> Path:
    root = tmp_path / "project"
    directory = root / "chunks"
    directory.mkdir(parents=True)
    rohs = [
        _chunk("02011L0065-20260701:00005", "RoHS", article="1"),
        _chunk("02011L0065-20260701:00009", "RoHS", article="2", paragraph="4", subparagraph="(a)"),
    ]
    reach = [_chunk("02006R1907-20260622:00007", "REACH", article="1", title="I", chapter="1")]
    if include_bad:
        rohs = [_chunk("bad:00001", "RoHS", text="bad"), rohs[0]]
    _write_jsonl(directory / "rohs.jsonl", rohs)
    _write_jsonl(directory / "reach.jsonl", reach)
    return root


def _write_jsonl(path: Path, chunks: list[ChunkRecord]) -> None:
    path.write_text(
        "".join(chunk.model_dump_json() + "\n" for chunk in chunks),
        encoding="utf-8",
    )


def _chunk(
    chunk_id: str,
    regulation: str,
    text: str = "Article 1\nSubject matter\nThis Directive applies.",
    article: str | None = "1",
    paragraph: str | None = None,
    subparagraph: str | None = None,
    title: str | None = None,
    chapter: str | None = None,
) -> ChunkRecord:
    number = "2011/65/EU" if regulation == "RoHS" else "(EC) No 1907/2006"
    filename = "rohs.pdf" if regulation == "RoHS" else "reach.pdf"
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=chunk_id.split(":")[0],
        regulation=regulation,
        document_title=f"{regulation} title",
        regulation_number=number,
        consolidation_date=date(2026, 7, 1) if regulation == "RoHS" else date(2026, 6, 22),
        chapter=chapter,
        title=title,
        article=article,
        paragraph=paragraph,
        subparagraph=subparagraph,
        annex=None,
        annex_section=None,
        page_start=5,
        page_end=5,
        source_filename=filename,
        text=text,
        token_count=12,
    )
