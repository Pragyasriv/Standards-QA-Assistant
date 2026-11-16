"""Open local retrieval resources once for the process.

The collection is opened with the existing open-only path. Nothing is created
or re-embedded here. The BM25 snapshot is required to already exist.
"""

from __future__ import annotations

import logging

from app.ask.errors import StartupError
from app.ask.service import AskService
from app.generation.service import GenerationService
from app.guardrails.service import GuardrailService
from app.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)


def prepare_ask_service() -> tuple[AskService, RetrievalService]:
    """Build the shared services, or raise StartupError without creating storage."""
    retrieval = RetrievalService()
    try:
        retrieval.store.open_existing()
    except Exception as exc:
        retrieval.close()
        raise StartupError(
            "Qdrant collection could not be opened. The collection was not created. "
            + _safe(exc)
        ) from exc
    index_path = retrieval.settings.bm25_index_path
    if not index_path.is_file():
        retrieval.close()
        raise StartupError(
            f"BM25 index was not found at {index_path}. The index was not rebuilt."
        )
    try:
        retrieval.embeddings.embed_query("startup")
    except Exception as exc:
        retrieval.close()
        raise StartupError("Embedding model could not be loaded. " + _safe(exc)) from exc
    service = AskService(GuardrailService(retrieval=retrieval), GenerationService())
    logger.info("Question resources are ready.")
    return service, retrieval


def _safe(exc: BaseException) -> str:
    text = str(exc)
    if "gsk_" in text:
        return "The error text was redacted."
    return text[:500]
