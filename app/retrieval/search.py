"""Qdrant similarity search over an existing collection."""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.retrieval.errors import RetrievalError

logger = logging.getLogger(__name__)


def search_similar(
    client: QdrantClient,
    collection_name: str,
    vector: list[float],
    limit: int,
    query_filter: models.Filter | None,
) -> list[models.ScoredPoint]:
    """Return the nearest stored points. The collection is not modified."""
    try:
        response = client.query_points(
            collection_name=collection_name,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        raise RetrievalError(f"Qdrant search failed: {exc}") from exc
    points = list(response.points)
    logger.info("Qdrant returned %s points from %s", len(points), collection_name)
    return points
