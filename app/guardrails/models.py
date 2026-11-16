"""Structured guardrail decisions. These models do not contain an answer."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.retrieval.models import RetrievedChunk

ScopeName = Literal["invalid", "out_of_scope", "clarification_required", "supported"]
DecisionStatus = Literal[
    "invalid",
    "out_of_scope",
    "clarification_required",
    "insufficient_evidence",
    "approved",
]
RegulationName = Literal["RoHS", "REACH"]


class CitationMetadata(BaseModel):
    """Citation fields copied from a retrieved chunk. Missing values stay null."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    regulation: str
    text: str
    article: str | None = None
    paragraph: str | None = None
    annex: str | None = None
    annex_section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    consolidation_date: str | None = None
    source_filename: str | None = None
    similarity_score: float | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None


class ScopeDecision(BaseModel):
    """Domain decision made before retrieval."""

    model_config = ConfigDict(extra="forbid")

    scope: ScopeName
    reason: str
    detected_regulation: RegulationName | None = None
    detected_unsupported_instrument: str | None = None
    clarification_question: str | None = None
    article: str | None = None
    annex: str | None = None


class GuardrailResponse(BaseModel):
    """Final guardrail decision for one question."""

    model_config = ConfigDict(extra="forbid")

    status: DecisionStatus
    scope: ScopeName
    reason: str
    detected_regulation: RegulationName | None = None
    detected_unsupported_instrument: str | None = None
    clarification_question: str | None = None
    evidence_available: bool = False
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    citation_metadata: list[CitationMetadata] = Field(default_factory=list)
    article: str | None = None
    annex: str | None = None


def citations_from_chunks(chunks: list[RetrievedChunk]) -> list[CitationMetadata]:
    """Copy citation fields and the three retrieval scores without adding any."""
    return [
        CitationMetadata(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            regulation=chunk.regulation,
            text=chunk.text,
            article=chunk.article,
            paragraph=chunk.paragraph,
            annex=chunk.annex,
            annex_section=chunk.annex_section,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            consolidation_date=chunk.consolidation_date,
            source_filename=chunk.source_filename,
            similarity_score=chunk.similarity_score,
            dense_score=chunk.dense_score,
            bm25_score=chunk.bm25_score,
            rrf_score=chunk.rrf_score,
        )
        for chunk in chunks
    ]
