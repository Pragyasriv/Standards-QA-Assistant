"""Okapi BM25 over the existing regulatory chunks.

The index is built from the Phase 3 JSONL files and stored beside Qdrant.
Building or searching it does not read or write the Qdrant collection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.processing.models import ChunkRecord
from app.retrieval.errors import RetrievalError
from app.retrieval.filters import SearchFilters
from app.retrieval.models import RetrievedChunk
from app.retrieval.tokenize import tokenize

logger = logging.getLogger(__name__)

_INDEX_VERSION = 1


class BM25Document(BaseModel):
    """One chunk stored in the lexical index. Missing citation fields stay null."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    regulation: str
    regulation_number: str | None = None
    text: str
    article: str | None = None
    paragraph: str | None = None
    subparagraph: str | None = None
    annex: str | None = None
    annex_section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    consolidation_date: str | None = None
    source_filename: str | None = None
    tokens: list[str] = Field(default_factory=list)


class BM25Index:
    """In-memory Okapi BM25 index with a JSON snapshot on disk."""

    def __init__(
        self,
        documents: list[BM25Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        source_hashes: dict[str, str] | None = None,
    ) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.source_hashes = source_hashes or {}
        self._idf, self._avgdl = _idf_and_average_length(documents)

    @classmethod
    def build(
        cls,
        paths: list[Path],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> BM25Index:
        """Read chunk JSONL files and index their original text."""
        documents: list[BM25Document] = []
        seen: set[str] = set()
        hashes: dict[str, str] = {}
        for path in paths:
            if not path.is_file():
                raise RetrievalError(f"Chunk file not found: {path}")
            hashes[str(path)] = _file_sha256(path)
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    chunk = ChunkRecord.model_validate_json(line)
                except ValidationError as exc:
                    raise RetrievalError(f"{path} line {number} is not a valid chunk: {exc}") from exc
                document = document_from_chunk(chunk)
                if document is None:
                    logger.warning("Skipped %s line %s because citation identity is missing.", path, number)
                    continue
                if document.chunk_id in seen:
                    logger.warning("Skipped duplicate chunk id %s.", document.chunk_id)
                    continue
                seen.add(document.chunk_id)
                documents.append(document)
        logger.info("Built a BM25 index with %s chunks.", len(documents))
        return cls(documents, k1=k1, b=b, source_hashes=hashes)

    def save(self, path: Path) -> None:
        """Write the index. This does not touch Qdrant."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _INDEX_VERSION,
            "k1": self.k1,
            "b": self.b,
            "source_hashes": self.source_hashes,
            "documents": [document.model_dump() for document in self.documents],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        """Load a snapshot written by :meth:`save`."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RetrievalError(f"BM25 index at {path} could not be read: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("version") != _INDEX_VERSION:
            raise RetrievalError(f"BM25 index at {path} has an unsupported format.")
        try:
            documents = [BM25Document.model_validate(item) for item in payload.get("documents", [])]
        except ValidationError as exc:
            raise RetrievalError(f"BM25 index at {path} has an invalid document: {exc}") from exc
        hashes = payload.get("source_hashes") or {}
        if not isinstance(hashes, dict):
            hashes = {}
        return cls(
            documents,
            k1=float(payload.get("k1", 1.5)),
            b=float(payload.get("b", 0.75)),
            source_hashes={str(key): str(value) for key, value in hashes.items()},
        )

    def matches_sources(self, paths: list[Path]) -> bool:
        """Return whether this snapshot was built from the current chunk files."""
        if not self.source_hashes:
            return False
        try:
            current = {str(path): _file_sha256(path) for path in paths}
        except OSError:
            return False
        return current == self.source_hashes

    def search(self, question: str, limit: int, filters: SearchFilters | None = None) -> list[RetrievedChunk]:
        """Return chunks with a positive BM25 score, highest first.

        Filters select the candidate set before scoring. A document that
        contains none of the query terms scores zero and is left out, so a
        query with no lexical overlap does not receive an arbitrary ranking.
        """
        if limit < 1:
            raise RetrievalError("top_k must be at least 1.")
        query_tokens = tokenize(question)
        if not query_tokens or not self.documents or self._avgdl == 0:
            return []
        active = filters or SearchFilters()
        scored: list[tuple[float, BM25Document]] = []
        for document in self.documents:
            if not _matches_filter(document, active):
                continue
            score = self._score(query_tokens, document.tokens)
            if score > 0:
                scored.append((score, document))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [_to_result(document, score) for score, document in scored[:limit]]

    def _score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not doc_tokens:
            return 0.0
        frequencies = Counter(doc_tokens)
        length = len(doc_tokens)
        total = 0.0
        for term in query_tokens:
            idf = self._idf.get(term)
            frequency = frequencies.get(term, 0)
            if idf is None or frequency == 0:
                continue
            denominator = frequency + self.k1 * (1 - self.b + self.b * length / self._avgdl)
            total += idf * (frequency * (self.k1 + 1)) / denominator
        return total


def document_from_chunk(chunk: ChunkRecord) -> BM25Document | None:
    """Copy a chunk into the lexical index. Required identity fields must be present."""
    if not chunk.chunk_id or not chunk.document_id or not chunk.regulation or not isinstance(chunk.text, str):
        return None
    return BM25Document(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        regulation=chunk.regulation,
        regulation_number=chunk.regulation_number,
        text=chunk.text,
        article=chunk.article,
        paragraph=chunk.paragraph,
        subparagraph=chunk.subparagraph,
        annex=chunk.annex,
        annex_section=chunk.annex_section,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        consolidation_date=chunk.consolidation_date.isoformat(),
        source_filename=chunk.source_filename,
        tokens=tokenize(chunk.text),
    )


def document_from_mapping(payload: dict) -> BM25Document | None:
    """Build a document from a mapping. Missing identity is skipped, not invented."""
    try:
        chunk_id = payload["chunk_id"]
        document_id = payload["document_id"]
        regulation = payload["regulation"]
        text = payload["text"]
    except KeyError:
        return None
    if not isinstance(chunk_id, str) or not isinstance(document_id, str) or not isinstance(regulation, str):
        return None
    if not isinstance(text, str) or not chunk_id or not document_id or not regulation:
        return None
    return BM25Document(
        chunk_id=chunk_id,
        document_id=document_id,
        regulation=regulation,
        regulation_number=_optional_str(payload.get("regulation_number")),
        text=text,
        article=_optional_str(payload.get("article")),
        paragraph=_optional_str(payload.get("paragraph")),
        subparagraph=_optional_str(payload.get("subparagraph")),
        annex=_optional_str(payload.get("annex")),
        annex_section=_optional_str(payload.get("annex_section")),
        page_start=payload.get("page_start") if isinstance(payload.get("page_start"), int) else None,
        page_end=payload.get("page_end") if isinstance(payload.get("page_end"), int) else None,
        consolidation_date=_optional_str(payload.get("consolidation_date")),
        source_filename=_optional_str(payload.get("source_filename")),
        tokens=tokenize(text),
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return None


def _matches_filter(document: BM25Document, filters: SearchFilters) -> bool:
    if filters.regulation is not None and document.regulation != filters.regulation:
        return False
    if filters.document_id is not None and document.document_id != filters.document_id:
        return False
    if filters.article is not None and document.article != filters.article:
        return False
    if filters.annex is not None and document.annex != filters.annex:
        return False
    return True


def _to_result(document: BM25Document, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=document.chunk_id,
        document_id=document.document_id,
        regulation=document.regulation,
        regulation_number=document.regulation_number,
        text=document.text,
        similarity_score=None,
        dense_score=None,
        bm25_score=score,
        rrf_score=None,
        article=document.article,
        paragraph=document.paragraph,
        subparagraph=document.subparagraph,
        annex=document.annex,
        annex_section=document.annex_section,
        page_start=document.page_start,
        page_end=document.page_end,
        consolidation_date=document.consolidation_date,
        source_filename=document.source_filename,
    )


def _idf_and_average_length(documents: list[BM25Document]) -> tuple[dict[str, float], float]:
    """IDF matches the positive rank-BM25 form ``log((N - df + 0.5) / (df + 0.5) + 1)``."""
    total = len(documents)
    document_frequency: Counter[str] = Counter()
    lengths: list[int] = []
    for document in documents:
        lengths.append(len(document.tokens))
        document_frequency.update(set(document.tokens))
    idf = {
        term: math.log((total - frequency + 0.5) / (frequency + 0.5) + 1)
        for term, frequency in document_frequency.items()
    }
    average = sum(lengths) / total if total else 0.0
    return idf, average


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
