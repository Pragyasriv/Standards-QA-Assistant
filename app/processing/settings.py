"""Configurable document-processing settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

TokenizerName = Literal["word", "char4"]


class ProcessingSettings(BaseSettings):
    """Chunking and path settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    chunk_target_tokens: int = Field(default=1000, ge=1)
    chunk_max_tokens: int = Field(default=1200, ge=1)
    chunk_overlap_tokens: int = Field(default=150, ge=0)
    tokenizer: TokenizerName = "word"
    manifest_path: Path = Path("data/manifest/documents.json")
    processed_dir: Path = Path("data/processed")
    ocr_review_char_threshold: int = Field(default=40, ge=0)
    column_gap_points: float = Field(default=24.0, gt=0)

    @model_validator(mode="after")
    def check_chunk_bounds(self) -> "ProcessingSettings":
        """Keep the target size between the overlap and the maximum."""
        if self.chunk_max_tokens < self.chunk_target_tokens:
            raise ValueError("chunk_max_tokens must be greater than or equal to chunk_target_tokens.")
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_target_tokens.")
        return self


@lru_cache
def get_processing_settings() -> ProcessingSettings:
    """Return the process-wide processing settings."""
    return ProcessingSettings()
