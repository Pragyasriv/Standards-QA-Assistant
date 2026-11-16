"""Score saved pipeline rows with the installed RAGAS 0.4 collections API.

The judge is created with ``ragas.llms.llm_factory`` and a Groq client.
Embeddings are the local Hugging Face wrapper, not a Groq embedding model.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable

from app.evaluation.results import MetricRecord, PipelineRecord, utc_now
from app.evaluation.settings import EvaluationSettings

_SECRET = re.compile(r"gsk_[A-Za-z0-9_\-]+")
_TRANSIENT = ("429", "rate limit", "timeout", "timed out", "503", "502", "overloaded", "temporarily")


class RagasScorer:
    """Holds one judge and one local embedding model for the whole run."""

    def __init__(self, settings: EvaluationSettings) -> None:
        self.settings = settings
        model = settings.require_evaluator()
        from openai import OpenAI
        from ragas.embeddings import HuggingFaceEmbeddings
        from ragas.llms import llm_factory

        # RAGAS 0.4 patches the client with instructor.from_openai. The native
        # Groq client does not expose the chat attribute that patch expects.
        # This OpenAI client uses Groq's OpenAI-compatible endpoint.
        client = OpenAI(
            api_key=settings.configured_key(),
            base_url="https://api.groq.com/openai/v1",
        )
        self.llm = llm_factory(
            model,
            provider="groq",
            client=client,
            temperature=settings.ragas_evaluator_temperature,
            max_tokens=settings.ragas_evaluator_max_tokens,
        )
        self.embeddings = HuggingFaceEmbeddings(model=settings.ragas_embedding_model, use_api=False)
        self.model_name = model

    def score(self, record: PipelineRecord, existing: MetricRecord | None = None) -> MetricRecord:
        """Score one answered row. Metrics already stored on ``existing`` are not repeated."""
        return asyncio.run(self._score(record, existing))

    async def _score(self, record: PipelineRecord, existing: MetricRecord | None = None) -> MetricRecord:
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        answer = record.actual_answer or ""
        contexts = list(record.retrieved_contexts)
        retries = self.settings.ragas_max_retries
        operations: dict[str, Callable[[], Awaitable[object]]] = {
            "faithfulness": lambda: Faithfulness(llm=self.llm).ascore(
                user_input=record.question,
                response=answer,
                retrieved_contexts=contexts,
            ),
            "answer_relevancy": lambda: AnswerRelevancy(
                llm=self.llm,
                embeddings=self.embeddings,
                strictness=self.settings.ragas_answer_relevancy_strictness,
            ).ascore(user_input=record.question, response=answer),
            "context_precision": lambda: ContextPrecision(llm=self.llm).ascore(
                user_input=record.question,
                reference=record.reference_answer,
                retrieved_contexts=contexts,
            ),
            "context_recall": lambda: ContextRecall(llm=self.llm).ascore(
                user_input=record.question,
                retrieved_contexts=contexts,
                reference=record.reference_answer,
            ),
        }
        values: dict[str, float | None] = {name: None for name in operations}
        if existing is not None:
            for name in operations:
                values[name] = getattr(existing, name)
        errors: list[str] = []
        for name, operation in operations.items():
            if values[name] is not None:
                continue
            try:
                values[name] = await _call(operation, retries)
            except Exception as exc:
                errors.append(f"{name}: {_redact(exc)}")
        return MetricRecord(
            id=record.id,
            faithfulness=values["faithfulness"],
            answer_relevancy=values["answer_relevancy"],
            context_precision=values["context_precision"],
            context_recall=values["context_recall"],
            metric_error="; ".join(errors) or None,
            evaluator_model=self.model_name,
            evaluated_at=utc_now(),
        )


async def _call(operation: Callable[[], Awaitable[object]], retries: int) -> float:
    attempt = 0
    while True:
        try:
            result = await operation()
            return float(result.value)  # type: ignore[attr-defined]
        except Exception as exc:
            if attempt >= retries or not _transient(exc):
                raise
            time.sleep(_delay(exc, attempt))
            attempt += 1


def _delay(exc: BaseException, attempt: int) -> float:
    """Wait for the provider's stated retry time, otherwise back off."""
    match = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", str(exc))
    if match:
        minutes = float(match.group(1) or 0)
        seconds = float(match.group(2))
        return min(900.0, minutes * 60 + seconds + 2)
    return min(60.0, 2.0 * (2**attempt))


def _transient(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(token in text for token in _TRANSIENT)


def _redact(exc: BaseException) -> str:
    text = _SECRET.sub("[redacted]", str(exc)).replace("\n", " ")
    return text[:500]
