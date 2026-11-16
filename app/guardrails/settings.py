"""Guardrail configuration."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GuardrailSettings(BaseSettings):
    """Limits for question text. Instrument names live in the catalog."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    guardrail_max_question_length: int = Field(default=2000, ge=1, le=20000)
