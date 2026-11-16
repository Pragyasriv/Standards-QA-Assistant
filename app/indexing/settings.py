"""Environment-backed settings for embeddings and Qdrant."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingDevice = Literal["cpu", "cuda"]


class IndexingSettings(BaseSettings):
    """Embedding model, batching, and local Qdrant storage settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = Field(default=384, ge=1)
    embedding_batch_size: int = Field(default=64, ge=1)
    embedding_device: EmbeddingDevice = "cpu"
    qdrant_path: Path = Path("storage/qdrant")
    qdrant_collection: str = "regulatory_documents"
    rohs_chunks_path: Path = Path("data/processed/rohs/chunks.jsonl")
    reach_chunks_path: Path = Path("data/processed/reach/chunks.jsonl")

    @property
    def storage_dir(self) -> Path:
        """Directory that holds the database, embedding cache, and reports."""
        return self.qdrant_path.parent
