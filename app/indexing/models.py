"""Records produced by the indexing pipeline."""

from pydantic import BaseModel, ConfigDict, Field


class IndexingReport(BaseModel):
    """Measured result of one indexing run."""

    model_config = ConfigDict(extra="forbid")

    embedding_model: str
    embedding_dimensions: int
    rohs_chunks_read: int = Field(ge=0)
    reach_chunks_read: int = Field(ge=0)
    rohs_chunks_indexed: int = Field(ge=0)
    reach_chunks_indexed: int = Field(ge=0)
    total_vectors_stored: int = Field(ge=0)
    failed_chunks: list[str]
    skipped_chunks: list[str]
    collection_name: str
    distance: str
    storage_path: str
    duration_seconds: float = Field(ge=0)
    embeddings_reused_from_cache: int = Field(ge=0)
    source_unchanged: bool
    verification_passed: bool
    verification_notes: list[str]
