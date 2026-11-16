"""Evaluator settings. These do not change the application generator."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class EvaluationConfigError(RuntimeError):
    """The evaluator is not configured. No model is substituted."""


class EvaluationSettings(BaseSettings):
    """Groq judge and local embeddings used only by the RAGAS script.

    ``groq_model`` is read so the report can name the generator. The judge
    refuses to fall back to that name.
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
    ragas_evaluator_model: str = ""
    ragas_evaluator_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    ragas_evaluator_max_tokens: int = Field(default=2048, ge=256, le=8192)
    ragas_answer_relevancy_strictness: int = Field(default=2, ge=1, le=5)
    ragas_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ragas_max_retries: int = Field(default=3, ge=0, le=8)
    ragas_score_ids: str = ""
    ragas_skip_guardrail_report: bool = False

    def configured_key(self) -> str:
        """Return the shared Groq key, or an empty string when it is unset."""
        return self.groq_api_key.get_secret_value().strip()

    def evaluator_model(self) -> str:
        """Return the judge model name, or an empty string when it is unset."""
        return self.ragas_evaluator_model.strip()

    def generator_model(self) -> str:
        """Return the application generator model name."""
        return self.groq_model.strip()

    def score_id_filter(self, valid_ids: set[str]) -> set[str] | None:
        """Return selected ids from ``RAGAS_SCORE_IDS``, or ``None`` when unset."""
        raw = self.ragas_score_ids.strip()
        if not raw:
            return None
        selected = {part.strip() for part in raw.split(",") if part.strip()}
        unknown = sorted(selected - valid_ids)
        if unknown:
            raise EvaluationConfigError(
                "RAGAS_SCORE_IDS contains unknown question ids: " + ", ".join(unknown)
            )
        return selected

    def require_evaluator(self) -> str:
        """Return the judge model, or raise when it was not configured."""
        model = self.evaluator_model()
        if not model:
            raise EvaluationConfigError(
                "RAGAS_EVALUATOR_MODEL is empty. Set it to a Groq chat model. "
                "The evaluator does not fall back to GROQ_MODEL."
            )
        if not self.configured_key():
            raise EvaluationConfigError("GROQ_API_KEY is empty.")
        if not self.generator_model():
            raise EvaluationConfigError("GROQ_MODEL is empty.")
        return model
