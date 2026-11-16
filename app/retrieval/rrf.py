"""Reciprocal rank fusion for dense and sparse candidate lists.

Scores from cosine similarity and BM25 are not added together. Fusion uses
only the rank of each chunk inside each list:

    score(chunk) = sum over lists of 1 / (k + rank)

``k`` defaults to 60, the constant from Cormack, Clarke, and Buettcher (2009).
It was not selected by sweeping the evaluation questions.
"""

from __future__ import annotations

from app.retrieval.models import RetrievedChunk


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]],
    *,
    k: int = 60,
    limit: int,
) -> list[RetrievedChunk]:
    """Merge ranked lists by chunk id and return the fused order."""
    if k < 1:
        raise ValueError("RRF k must be at least 1.")
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    fused: dict[str, float] = {}
    merged: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item.chunk_id] = fused.get(item.chunk_id, 0.0) + 1.0 / (k + rank)
            current = merged.get(item.chunk_id)
            merged[item.chunk_id] = item if current is None else _merge(current, item)
    ordered = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], chunk_id))
    results: list[RetrievedChunk] = []
    for chunk_id in ordered[:limit]:
        results.append(merged[chunk_id].model_copy(update={"rrf_score": fused[chunk_id]}))
    return results


def _merge(existing: RetrievedChunk, incoming: RetrievedChunk) -> RetrievedChunk:
    """Keep one metadata record and both retrieval scores."""
    dense = existing.dense_score if existing.dense_score is not None else incoming.dense_score
    bm25 = existing.bm25_score if existing.bm25_score is not None else incoming.bm25_score
    cosine = dense if dense is not None else existing.similarity_score
    if cosine is None:
        cosine = incoming.similarity_score
    return existing.model_copy(
        update={
            "dense_score": dense,
            "bm25_score": bm25,
            "similarity_score": cosine,
            "text": existing.text or incoming.text,
        }
    )
