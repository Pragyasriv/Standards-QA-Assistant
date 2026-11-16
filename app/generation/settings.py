"""Groq settings for grounded answer generation."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class GenerationSettings(BaseSettings):
    """Model name, temperature, and API key.

    The model name is required configuration. Business logic does not substitute
    a model when this value is empty or when the provider rejects it.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    groq_api_key: SecretStr = SecretStr("")
    groq_model: str = ""
    groq_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    def configured_key(self) -> str:
        """Return the API key, or an empty string when it is unset."""
        return self.groq_api_key.get_secret_value().strip()

    def configured_model(self) -> str:
        """Return the model name, or an empty string when it is unset."""
        return self.groq_model.strip()
