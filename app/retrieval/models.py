"""Structured retrieval responses."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.retrieval.filters import SearchFilters


class RetrievedChunk(BaseModel):
    """One stored chunk and the scores that retrieved it.

    ``similarity_score`` and ``dense_score`` are the Qdrant cosine similarity.
    ``bm25_score`` is the lexical score. ``rrf_score`` is reciprocal rank
    fusion. Those three scales are not interchangeable.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    regulation: str
    regulation_number: str | None = None
    text: str
    similarity_score: float | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    article: str | None = None
    paragraph: str | None = None
    subparagraph: str | None = None
    annex: str | None = None
    annex_section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    consolidation_date: str | None = None
    source_filename: str | None = None


class RetrievalResponse(BaseModel):
    """Ranked chunks, or an insufficient-evidence result.

    ``best_score`` is the score used to order this response. For dense search
    that is cosine similarity. For BM25 it is the lexical score. For hybrid
    search it is the RRF score, which is not a cosine similarity.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "insufficient_evidence"]
    question: str
    method: Literal["dense", "bm25", "hybrid"] = "dense"
    top_k: int = Field(ge=1)
    minimum_similarity: float | None
    best_score: float | None
    best_dense_score: float | None = None
    best_bm25_score: float | None = None
    best_rrf_score: float | None = None
    results: list[RetrievedChunk]
    filters: SearchFilters
    message: str | None = None
