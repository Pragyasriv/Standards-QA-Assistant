"""Check the local Qdrant collection and BM25 file without calling Groq.

Opening the collection uses the existing open-only path. A missing collection
is reported as not ready and is not created.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.indexing.store import QdrantStore
from app.retrieval.service import RetrievalService
from app.retrieval.settings import RetrievalSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReadinessReport:
    """Local resources required before a question can be retrieved."""

    qdrant_available: bool
    bm25_index_available: bool

    @property
    def ready(self) -> bool:
        return self.qdrant_available and self.bm25_index_available


def assess_readiness(retrieval: RetrievalService | None = None) -> ReadinessReport:
    """Return whether the stored collection and BM25 snapshot can be opened."""
    settings = retrieval.settings if retrieval is not None else RetrievalSettings()
    bm25_ok = settings.bm25_index_path.is_file()
    qdrant_ok = _qdrant_available(retrieval, settings)
    if not qdrant_ok or not bm25_ok:
        logger.info(
            "Readiness qdrant=%s bm25=%s",
            qdrant_ok,
            bm25_ok,
        )
    return ReadinessReport(qdrant_available=qdrant_ok, bm25_index_available=bm25_ok)


def _qdrant_available(retrieval: RetrievalService | None, settings: RetrievalSettings) -> bool:
    if retrieval is not None and retrieval.store.client is not None:
        return True
    if not settings.qdrant_path.exists():
        return False
    store = QdrantStore(settings)
    try:
        store.open_existing()
    except Exception:
        logger.info("Readiness could not open the existing Qdrant collection.")
        return False
    finally:
        store.close()
    return True
