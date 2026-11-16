"""Retrieval settings, including the shared embedding and Qdrant configuration."""

from pathlib import Path

from pydantic import Field

from app.indexing.settings import IndexingSettings


class RetrievalSettings(IndexingSettings):
    """How many chunks to retrieve and which similarity scores count as evidence.

    ``retrieval_min_similarity`` is Qdrant's cosine similarity, where a higher
    score is closer. ``None`` keeps every retrieved chunk. A numeric value
    drops chunks below that score and reports insufficient evidence when none
    remain.

    The default 0.35 comes from a four-question probe on this index, not from
    the evaluation file. On that probe, "What are the restrictions on lead?"
    scored 0.649 and "What is the scope of REACH?" scored 0.512. "What is the
    capital of France?" scored 0.230. "How do I change a car tyre?" scored
    0.344 because Annex XVII restricts tyre oils, so the margin above that
    false match is thin. RoHS Article 4 scored about 0.25–0.32 for the lead
    question and does not clear this cutoff. The value is configurable.

    Dense and sparse candidate depths default to 20. Reciprocal rank fusion
    uses k = 60. BM25 uses k1 = 1.5 and b = 0.75. These are the usual published
    defaults. They were not chosen by sweeping the evaluation questions.
    """

    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    retrieval_min_similarity: float | None = 0.35
    retrieval_dense_depth: int = Field(default=20, ge=1, le=200)
    retrieval_sparse_depth: int = Field(default=20, ge=1, le=200)
    retrieval_rrf_k: int = Field(default=60, ge=1, le=1000)
    bm25_k1: float = Field(default=1.5, ge=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    bm25_index_path: Path = Path("storage/bm25/index.json")
