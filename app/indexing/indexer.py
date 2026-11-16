"""Load Phase 3 chunks, embed them, and upsert them into Qdrant."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from pydantic import ValidationError

from app.indexing.cache import EmbeddingCache
from app.indexing.embeddings import SentenceTransformerEmbeddings
from app.indexing.errors import EmbeddingError, IndexingError
from app.indexing.models import IndexingReport
from app.indexing.payload import chunk_to_payload
from app.indexing.settings import IndexingSettings
from app.indexing.store import QdrantStore
from app.processing.models import ChunkRecord

logger = logging.getLogger(__name__)


def index_regulatory_documents(
    root: Path,
    settings: IndexingSettings | None = None,
    embeddings: SentenceTransformerEmbeddings | object | None = None,
) -> IndexingReport:
    """Index the configured RoHS and REACH chunk files.

    ``embeddings`` can be injected by tests. Production uses the sentence-transformer
    service and the on-disk embedding cache.
    """
    active = settings or IndexingSettings()
    started = time.perf_counter()
    rohs_path = _resolve(root, active.rohs_chunks_path)
    reach_path = _resolve(root, active.reach_chunks_path)
    rohs_chunks = load_chunk_file(rohs_path)
    reach_chunks = load_chunk_file(reach_path)
    _reject_duplicate_ids(rohs_chunks + reach_chunks)

    skipped = [chunk.chunk_id for chunk in rohs_chunks + reach_chunks if not chunk.text.strip()]
    indexable = [chunk for chunk in rohs_chunks + reach_chunks if chunk.text.strip()]
    cache_path = _resolve(root, active.storage_dir) / "embedding_cache" / "vectors.jsonl"
    stamp_path = _resolve(root, active.storage_dir) / "index_stamp.json"
    source_stamp = _source_stamp(active, rohs_path, reach_path, len(indexable))

    store = QdrantStore(active if _paths_are_absolute(active) else _absolute_settings(active, root))
    owns_embeddings = embeddings is None
    embedder = embeddings if embeddings is not None else SentenceTransformerEmbeddings(
        active,
        cache=EmbeddingCache(cache_path),
    )
    cache_hits_before = getattr(embedder, "cache_hits", 0)
    failed: list[str] = []
    source_unchanged = False
    try:
        store.open()
        if _stamp_matches(stamp_path, source_stamp) and store.count() == len(indexable):
            source_unchanged = True
            logger.info(
                "Chunk files and collection are unchanged. Embeddings were not regenerated."
            )
        else:
            failed = _embed_and_upsert(indexable, embedder, store, active.embedding_batch_size)
        by_regulation = store.count_by_regulation()
        stored = store.count()
        notes = _verify(indexable, skipped, failed, stored, by_regulation, store)
    finally:
        store.close()

    passed = not notes and not failed
    if passed:
        _write_json(stamp_path, source_stamp)
    report = IndexingReport(
        embedding_model=active.embedding_model,
        embedding_dimensions=active.embedding_dimensions,
        rohs_chunks_read=len(rohs_chunks),
        reach_chunks_read=len(reach_chunks),
        rohs_chunks_indexed=by_regulation.get("RoHS", 0),
        reach_chunks_indexed=by_regulation.get("REACH", 0),
        total_vectors_stored=stored,
        failed_chunks=failed,
        skipped_chunks=skipped,
        collection_name=active.qdrant_collection,
        distance="cosine",
        storage_path=str(_resolve(root, active.qdrant_path)),
        duration_seconds=round(time.perf_counter() - started, 3),
        embeddings_reused_from_cache=getattr(embedder, "cache_hits", 0) - cache_hits_before,
        source_unchanged=source_unchanged,
        verification_passed=passed,
        verification_notes=notes,
    )
    report_path = _resolve(root, active.storage_dir) / "indexing_report.json"
    _write_json(report_path, json.loads(report.model_dump_json()))
    logger.info(
        "Indexed collection %s vectors=%s rohs=%s reach=%s failed=%s skipped=%s",
        report.collection_name,
        report.total_vectors_stored,
        report.rohs_chunks_indexed,
        report.reach_chunks_indexed,
        len(report.failed_chunks),
        len(report.skipped_chunks),
    )
    if owns_embeddings:
        del embedder
    return report


def load_chunk_file(path: Path) -> list[ChunkRecord]:
    """Read and validate a Phase 3 chunks JSONL file."""
    if not path.is_file():
        raise IndexingError(f"Chunk file not found: {path}")
    records: list[ChunkRecord] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(ChunkRecord.model_validate_json(line))
        except ValidationError as exc:
            raise IndexingError(f"{path} line {number} is not a valid chunk: {exc}") from exc
    return records


def _embed_and_upsert(
    chunks: list[ChunkRecord],
    embedder: object,
    store: QdrantStore,
    batch_size: int,
) -> list[str]:
    failed: list[str] = []
    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        try:
            vectors = embedder.embed_documents([chunk.text for chunk in batch])  # type: ignore[attr-defined]
        except EmbeddingError:
            logger.exception(
                "Embedding failed for chunks %s to %s",
                start + 1,
                start + len(batch),
            )
            failed.extend(chunk.chunk_id for chunk in batch)
            continue
        store.upsert(batch, vectors)
        logger.info("Upserted %s/%s chunks", min(start + len(batch), total), total)
    return failed


def _verify(
    indexable: list[ChunkRecord],
    skipped: list[str],
    failed: list[str],
    stored: int,
    by_regulation: dict[str, int],
    store: QdrantStore,
) -> list[str]:
    notes: list[str] = []
    expected = len(indexable) - len(failed)
    if stored != expected:
        notes.append(f"Stored {stored} vectors; expected {expected} after failures.")
    expected_by_regulation: dict[str, int] = {}
    failed_ids = set(failed)
    for chunk in indexable:
        if chunk.chunk_id in failed_ids:
            continue
        expected_by_regulation[chunk.regulation] = expected_by_regulation.get(chunk.regulation, 0) + 1
    if by_regulation != expected_by_regulation:
        notes.append(
            f"Regulation counts {by_regulation} do not match the indexed chunks "
            f"{expected_by_regulation}."
        )
    sample = next((chunk for chunk in indexable if chunk.chunk_id not in failed_ids), None)
    if sample is not None:
        payload = store.retrieve_payload(sample.chunk_id)
        if not _payload_matches(payload, chunk_to_payload(sample)):
            notes.append(f"Stored payload for {sample.chunk_id} does not match the chunk.")
    return notes


def _payload_matches(stored: dict | None, expected: dict) -> bool:
    if not stored:
        return False
    if stored.get("page_content") != expected["page_content"]:
        return False
    metadata = stored.get("metadata")
    if not isinstance(metadata, dict):
        return False
    for key, value in expected["metadata"].items():
        if metadata.get(key) != value:
            return False
    return True


def _reject_duplicate_ids(chunks: list[ChunkRecord]) -> None:
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            raise IndexingError(f"Duplicate chunk_id {chunk.chunk_id}.")
        seen.add(chunk.chunk_id)


def _source_stamp(
    settings: IndexingSettings,
    rohs_path: Path,
    reach_path: Path,
    indexable_count: int,
) -> dict:
    return {
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "collection": settings.qdrant_collection,
        "rohs_sha256": _file_sha256(rohs_path),
        "reach_sha256": _file_sha256(reach_path),
        "indexable_count": indexable_count,
    }


def _stamp_matches(path: Path, stamp: dict) -> bool:
    if not path.is_file():
        return False
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return stored == stamp


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _resolve(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def _paths_are_absolute(settings: IndexingSettings) -> bool:
    return settings.qdrant_path.is_absolute()


def _absolute_settings(settings: IndexingSettings, root: Path) -> IndexingSettings:
    return settings.model_copy(
        update={
            "qdrant_path": _resolve(root, settings.qdrant_path),
            "rohs_chunks_path": _resolve(root, settings.rohs_chunks_path),
            "reach_chunks_path": _resolve(root, settings.reach_chunks_path),
        }
    )
