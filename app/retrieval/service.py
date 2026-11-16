"""Retrieve regulatory chunks for a natural-language question.

Query checking, embedding, Qdrant search, filtering, the similarity cutoff,
and response formatting stay in separate functions. This module does not
define HTTP routes.
"""

from __future__ import annotations

import logging
from typing import Protocol

from qdrant_client.http import models

from app.indexing.embeddings import SentenceTransformerEmbeddings, validate_embedding
from app.indexing.errors import EmbeddingError
from app.indexing.store import QdrantStore
from app.retrieval.bm25 import BM25Index
from app.retrieval.errors import InvalidQuestionError, RetrievalError
from app.retrieval.filters import SearchFilters, build_qdrant_filter
from app.retrieval.models import RetrievalResponse, RetrievedChunk
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.search import search_similar
from app.retrieval.settings import RetrievalSettings

logger = logging.getLogger(__name__)


class QueryEmbedder(Protocol):
    """Encode one question with the same model used for the stored chunks."""

    def embed_query(self, text: str) -> list[float]:
        """Return one embedding."""


class RetrievalService:
    """Search the existing regulatory collection."""

    def __init__(
        self,
        settings: RetrievalSettings | None = None,
        embeddings: QueryEmbedder | None = None,
        store: QdrantStore | None = None,
        bm25: BM25Index | None = None,
    ) -> None:
        self.settings = settings or RetrievalSettings()
        self.embeddings = embeddings or SentenceTransformerEmbeddings(self.settings)
        self.store = store or QdrantStore(self.settings)
        self._bm25 = bm25
        self._opened = False

    def close(self) -> None:
        """Release the local Qdrant lock."""
        self.store.close()
        self._opened = False

    def search(
        self,
        question: str,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        minimum_similarity: float | None = None,
        *,
        apply_threshold: bool = True,
    ) -> RetrievalResponse:
        """Embed a question and return the closest stored chunks.

        ``apply_threshold=False`` returns the raw top-K ranking. That is used
        to inspect scores. The default path drops scores below the configured
        minimum and returns insufficient evidence when nothing remains.
        """
        checked = validate_question(question)
        active_filters = filters or SearchFilters()
        limit = top_k if top_k is not None else self.settings.retrieval_top_k
        if limit < 1:
            raise RetrievalError("top_k must be at least 1.")
        if not apply_threshold:
            cutoff = None
        elif minimum_similarity is not None:
            cutoff = minimum_similarity
        else:
            cutoff = self.settings.retrieval_min_similarity
        vector = self._embed_question(checked)
        self._ensure_open()
        assert self.store.client is not None
        points = search_similar(
            self.store.client,
            self.settings.qdrant_collection,
            vector,
            limit,
            build_qdrant_filter(active_filters),
        )
        ranked = [_format_point(point) for point in points]
        ranked = [item for item in ranked if item is not None]
        return apply_similarity_threshold(
            question=checked,
            ranked=ranked,
            top_k=limit,
            minimum_similarity=cutoff,
            filters=active_filters,
        )

    def search_bm25(
        self,
        question: str,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
    ) -> RetrievalResponse:
        """Rank chunks with BM25. The cosine cutoff is not applied to these scores."""
        checked = validate_question(question)
        active_filters = filters or SearchFilters()
        limit = top_k if top_k is not None else self.settings.retrieval_top_k
        if limit < 1:
            raise RetrievalError("top_k must be at least 1.")
        ranked = self._ensure_bm25().search(checked, limit, active_filters)
        best = ranked[0].bm25_score if ranked else None
        if not ranked:
            return RetrievalResponse(
                status="insufficient_evidence",
                question=checked,
                method="bm25",
                top_k=limit,
                minimum_similarity=None,
                best_score=None,
                best_bm25_score=None,
                results=[],
                filters=active_filters,
                message="Insufficient evidence: no indexed chunk shares a query term.",
            )
        return RetrievalResponse(
            status="ok",
            question=checked,
            method="bm25",
            top_k=limit,
            minimum_similarity=None,
            best_score=best,
            best_bm25_score=best,
            results=ranked,
            filters=active_filters,
            message=None,
        )

    def search_hybrid(
        self,
        question: str,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        dense_depth: int | None = None,
        sparse_depth: int | None = None,
        rrf_k: int | None = None,
    ) -> RetrievalResponse:
        """Fuse dense and BM25 rankings with reciprocal rank fusion.

        The cosine cutoff is not applied to the fused list. ``best_score`` on
        this response is the RRF score.
        """
        checked = validate_question(question)
        active_filters = filters or SearchFilters()
        limit = top_k if top_k is not None else self.settings.retrieval_top_k
        if limit < 1:
            raise RetrievalError("top_k must be at least 1.")
        dense_limit = dense_depth if dense_depth is not None else self.settings.retrieval_dense_depth
        sparse_limit = sparse_depth if sparse_depth is not None else self.settings.retrieval_sparse_depth
        constant = rrf_k if rrf_k is not None else self.settings.retrieval_rrf_k
        dense = self.search(
            checked,
            filters=active_filters,
            top_k=dense_limit,
            apply_threshold=False,
        )
        sparse = self.search_bm25(checked, filters=active_filters, top_k=sparse_limit)
        fused = reciprocal_rank_fusion(
            [dense.results, sparse.results],
            k=constant,
            limit=limit,
        )
        best_rrf = fused[0].rrf_score if fused else None
        best_dense = dense.best_score
        best_bm25 = sparse.best_score
        if not fused:
            return RetrievalResponse(
                status="insufficient_evidence",
                question=checked,
                method="hybrid",
                top_k=limit,
                minimum_similarity=None,
                best_score=None,
                best_dense_score=best_dense,
                best_bm25_score=best_bm25,
                best_rrf_score=None,
                results=[],
                filters=active_filters,
                message="Insufficient evidence: dense and BM25 retrieval returned no chunks.",
            )
        return RetrievalResponse(
            status="ok",
            question=checked,
            method="hybrid",
            top_k=limit,
            minimum_similarity=None,
            best_score=best_rrf,
            best_dense_score=best_dense,
            best_bm25_score=best_bm25,
            best_rrf_score=best_rrf,
            results=fused,
            filters=active_filters,
            message=None,
        )

    def _embed_question(self, question: str) -> list[float]:
        try:
            vector = self.embeddings.embed_query(question)
            validate_embedding(vector, self.settings.embedding_dimensions)
        except EmbeddingError as exc:
            raise RetrievalError(str(exc)) from exc
        except Exception as exc:
            raise RetrievalError(f"Query embedding failed: {exc}") from exc
        return vector

    def _ensure_open(self) -> None:
        if self.store.client is not None:
            self._opened = True
            return
        try:
            self.store.open_existing()
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(str(exc)) from exc
        self._opened = True

    def _ensure_bm25(self) -> BM25Index:
        paths = [self.settings.rohs_chunks_path, self.settings.reach_chunks_path]
        if self._bm25 is not None and (not self._bm25.source_hashes or self._bm25.matches_sources(paths)):
            return self._bm25
        path = self.settings.bm25_index_path
        if path.is_file():
            loaded = BM25Index.load(path)
            if loaded.matches_sources(paths):
                self._bm25 = loaded
                return loaded
        built = BM25Index.build(paths, k1=self.settings.bm25_k1, b=self.settings.bm25_b)
        built.save(path)
        self._bm25 = built
        return built


def validate_question(question: object) -> str:
    """Return the trimmed question, or reject text that cannot be embedded."""
    if not isinstance(question, str):
        raise InvalidQuestionError("Question must be text.")
    stripped = question.strip()
    if not stripped:
        raise InvalidQuestionError("Question is empty.")
    return stripped


def apply_similarity_threshold(
    *,
    question: str,
    ranked: list[RetrievedChunk],
    top_k: int,
    minimum_similarity: float | None,
    filters: SearchFilters,
) -> RetrievalResponse:
    """Keep chunks at or above the cutoff. An empty kept list is insufficient evidence."""
    best = ranked[0].similarity_score if ranked else None
    if minimum_similarity is None:
        kept = ranked
    else:
        kept = [item for item in ranked if item.similarity_score >= minimum_similarity]
    if not kept:
        observed = "no chunks were retrieved" if best is None else f"the best score was {best:.4f}"
        requirement = (
            "No similarity minimum is configured."
            if minimum_similarity is None
            else f"The configured minimum is {minimum_similarity:.4f}."
        )
        return RetrievalResponse(
            status="insufficient_evidence",
            question=question,
            method="dense",
            top_k=top_k,
            minimum_similarity=minimum_similarity,
            best_score=best,
            best_dense_score=best,
            results=[],
            filters=filters,
            message=f"Insufficient evidence: {observed}. {requirement}",
        )
    return RetrievalResponse(
        status="ok",
        question=question,
        method="dense",
        top_k=top_k,
        minimum_similarity=minimum_similarity,
        best_score=best,
        best_dense_score=best,
        results=kept,
        filters=filters,
        message=None,
    )


def _format_point(point: models.ScoredPoint) -> RetrievedChunk | None:
    """Map a Qdrant payload to a result. Missing citation fields stay null."""
    payload = point.payload or {}
    metadata = payload.get("metadata")
    text = payload.get("page_content")
    if not isinstance(metadata, dict) or not isinstance(text, str):
        logger.warning("Skipped a Qdrant point with an unreadable payload.")
        return None
    chunk_id = metadata.get("chunk_id")
    document_id = metadata.get("document_id")
    regulation = metadata.get("regulation")
    if not isinstance(chunk_id, str) or not isinstance(document_id, str) or not isinstance(regulation, str):
        logger.warning("Skipped a Qdrant point that has no chunk, document, or regulation id.")
        return None
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        regulation=regulation,
        regulation_number=_optional_text(metadata.get("regulation_number")),
        text=text,
        similarity_score=float(point.score),
        dense_score=float(point.score),
        article=_optional_text(metadata.get("article")),
        paragraph=_optional_text(metadata.get("paragraph")),
        subparagraph=_optional_text(metadata.get("subparagraph")),
        annex=_optional_text(metadata.get("annex")),
        annex_section=_optional_text(metadata.get("annex_section")),
        page_start=_optional_int(metadata.get("page_start")),
        page_end=_optional_int(metadata.get("page_end")),
        consolidation_date=_optional_text(metadata.get("consolidation_date")),
        source_filename=_optional_text(metadata.get("source_filename")),
    )


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None
