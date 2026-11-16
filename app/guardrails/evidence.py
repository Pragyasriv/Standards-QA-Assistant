"""Check retrieved chunks against the regulation, article, and annex that were asked for.

Cosine, BM25, and RRF scores are copied through unchanged. A cosine cutoff is
not applied to an RRF score. A positive BM25 score is required and is not
sufficient when the question asks for a list and the chunk only points at it.
"""

from __future__ import annotations

from typing import Protocol

from app.guardrails.models import RegulationName, ScopeDecision
from app.guardrails.sufficiency import (
    asks_for_inventory,
    chunk_is_sufficient,
    cited_annex,
    is_procedural_pointer,
)
from app.retrieval.filters import SearchFilters
from app.retrieval.models import RetrievalResponse, RetrievedChunk


class EvidenceRetriever(Protocol):
    """The retrieval methods the evidence check calls."""

    def search(
        self,
        question: str,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        minimum_similarity: float | None = None,
        *,
        apply_threshold: bool = True,
    ) -> RetrievalResponse:
        """Dense search."""

    def search_bm25(
        self,
        question: str,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
    ) -> RetrievalResponse:
        """BM25 search."""

    def search_hybrid(
        self,
        question: str,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        dense_depth: int | None = None,
        sparse_depth: int | None = None,
        rrf_k: int | None = None,
    ) -> RetrievalResponse:
        """Hybrid search."""


class EvidenceAssessment:
    """Chunks that match the requested scope, plus the reason when none do."""

    def __init__(self, available: bool, reason: str, chunks: list[RetrievedChunk]) -> None:
        self.available = available
        self.reason = reason
        self.chunks = chunks


def assess_evidence(
    retriever: EvidenceRetriever,
    question: str,
    decision: ScopeDecision,
    top_k: int = 5,
    *,
    allow_annex_follow: bool = True,
) -> EvidenceAssessment:
    """Retrieve with the requested metadata filters and keep sufficient chunks."""
    regulation = decision.detected_regulation
    if regulation is None:
        return EvidenceAssessment(False, "No regulation was selected for retrieval.", [])
    filters = SearchFilters(
        regulation=regulation,
        article=decision.article,
        annex=decision.annex,
    )
    dense = retriever.search(question, filters=filters, top_k=top_k, apply_threshold=False)
    sparse = retriever.search_bm25(question, filters=filters, top_k=top_k)
    hybrid = retriever.search_hybrid(question, filters=filters, top_k=top_k)
    combined = _combine(dense.results, sparse.results, hybrid.results)
    matched = [chunk for chunk in combined if _matches_scope(chunk, regulation, decision.article, decision.annex)]
    sufficient = [
        chunk
        for chunk in matched
        if chunk_is_sufficient(
            chunk,
            question,
            requested_article=decision.article,
            requested_annex=decision.annex,
        )
    ]
    if sufficient:
        return EvidenceAssessment(True, _approved_reason(decision, question), sufficient)
    followed = _follow_annex(retriever, question, decision, matched, top_k, allow_annex_follow)
    if followed is not None:
        return followed
    if matched and asks_for_inventory(question) and decision.article is None and decision.annex is None:
        return EvidenceAssessment(
            False,
            f"Retrieved {regulation} chunks are related to the question but do not contain the requested list.",
            [],
        )
    return EvidenceAssessment(False, _insufficient_reason(decision, combined, matched), [])


def _follow_annex(
    retriever: EvidenceRetriever,
    question: str,
    decision: ScopeDecision,
    matched: list[RetrievedChunk],
    top_k: int,
    allow_annex_follow: bool,
) -> EvidenceAssessment | None:
    """Fetch an annex that a related article only points to.

    The annex number is read from that article. It is not inferred from the
    question. One follow-up is allowed, and only for an open list question.
    """
    if not allow_annex_follow or decision.article is not None or decision.annex is not None:
        return None
    if not asks_for_inventory(question):
        return None
    pointers = [cited_annex(chunk) for chunk in matched if is_procedural_pointer(chunk) or chunk.annex is None]
    names = {name for name in pointers if name is not None}
    if len(names) != 1:
        return None
    annex = next(iter(names))
    followed = assess_evidence(
        retriever,
        question,
        decision.model_copy(update={"annex": annex}),
        top_k,
        allow_annex_follow=False,
    )
    if not followed.available:
        return None
    return EvidenceAssessment(
        True,
        f"The related {decision.detected_regulation} provision points to Annex {annex}, which contains the list.",
        followed.chunks,
    )


def _approved_reason(decision: ScopeDecision, question: str) -> str:
    regulation = decision.detected_regulation or "the requested regulation"
    if asks_for_inventory(question) and decision.article is None:
        return f"Retrieved chunks match {regulation} and contain the requested list."
    return f"Retrieved chunks match {regulation} and the requested provision."


def _combine(
    dense: list[RetrievedChunk],
    sparse: list[RetrievedChunk],
    hybrid: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Merge the three lists by chunk id and keep each score on its own field."""
    order: list[str] = []
    merged: dict[str, RetrievedChunk] = {}
    for chunk in hybrid + sparse + dense:
        current = merged.get(chunk.chunk_id)
        if current is None:
            order.append(chunk.chunk_id)
            merged[chunk.chunk_id] = chunk.model_copy()
            continue
        merged[chunk.chunk_id] = current.model_copy(
            update={
                "similarity_score": _first(current.similarity_score, chunk.similarity_score),
                "dense_score": _first(current.dense_score, chunk.dense_score),
                "bm25_score": _first(current.bm25_score, chunk.bm25_score),
                "rrf_score": _first(current.rrf_score, chunk.rrf_score),
                "text": current.text or chunk.text,
            }
        )
    return [merged[chunk_id] for chunk_id in order]


def _matches_scope(
    chunk: RetrievedChunk,
    regulation: RegulationName,
    article: str | None,
    annex: str | None,
) -> bool:
    if chunk.regulation != regulation:
        return False
    if article is not None and (chunk.article != article or chunk.annex is not None):
        return False
    if annex is not None and chunk.annex != annex:
        return False
    return True


def _insufficient_reason(
    decision: ScopeDecision,
    combined: list[RetrievedChunk],
    matched: list[RetrievedChunk],
) -> str:
    regulation = decision.detected_regulation or "the requested regulation"
    if combined and not matched:
        others = sorted({chunk.regulation for chunk in combined if chunk.regulation != regulation})
        if others:
            return f"Retrieved evidence belongs to {', '.join(others)}, not {regulation}."
        if decision.article is not None:
            return f"No retrieved chunk is {regulation} Article {decision.article}."
        if decision.annex is not None:
            return f"No retrieved chunk is {regulation} Annex {decision.annex}."
        return f"No retrieved chunk matches {regulation}."
    if matched:
        return f"No retrieved {regulation} chunk shares the terms in the question."
    if decision.article is not None:
        return f"No chunks were retrieved for {regulation} Article {decision.article}."
    if decision.annex is not None:
        return f"No chunks were retrieved for {regulation} Annex {decision.annex}."
    return f"No chunks were retrieved for {regulation}."


def _first(current: float | None, incoming: float | None) -> float | None:
    if current is not None:
        return current
    return incoming
