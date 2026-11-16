"""Local sentence-transformer embeddings for regulatory chunks."""

from __future__ import annotations

import logging
import math

from langchain_core.embeddings import Embeddings

from app.indexing.cache import EmbeddingCache
from app.indexing.errors import EmbeddingError
from app.indexing.settings import IndexingSettings

logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddings(Embeddings):
    """Encode chunk text with one reused SentenceTransformer model.

    The class implements the LangChain embeddings interface so the Qdrant
    integration can use the same vectors later. Indexing calls ``embed_documents``
    directly and does not ask Qdrant to encode text.
    """

    def __init__(
        self,
        settings: IndexingSettings,
        cache: EmbeddingCache | None = None,
    ) -> None:
        self.settings = settings
        self.cache = cache
        self._model: object | None = None
        self.cache_hits = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one validated embedding per text."""
        self._reject_invalid_texts(texts)
        if not texts:
            return []
        results: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        for index, text in enumerate(texts):
            cached = None if self.cache is None else self.cache.get(self.settings.embedding_model, text)
            if cached is None:
                missing.append(index)
                continue
            try:
                validate_embedding(cached, self.settings.embedding_dimensions)
            except EmbeddingError:
                missing.append(index)
                continue
            results[index] = cached
            self.cache_hits += 1
        if missing:
            encoded = self._encode([texts[index] for index in missing])
            if len(encoded) != len(missing):
                raise EmbeddingError(
                    f"Embedding model returned {len(encoded)} vectors for {len(missing)} texts."
                )
            for offset, index in enumerate(missing):
                vector = _as_float_list(encoded[offset])
                validate_embedding(vector, self.settings.embedding_dimensions)
                results[index] = vector
                if self.cache is not None:
                    self.cache.put(self.settings.embedding_model, texts[index], vector)
        if any(vector is None for vector in results):
            raise EmbeddingError("Embedding model did not return a vector for every text.")
        logger.info(
            "Embedded %s texts (%s reused from cache) with %s",
            len(texts),
            len(texts) - len(missing),
            self.settings.embedding_model,
        )
        return [vector for vector in results if vector is not None]

    def embed_query(self, text: str) -> list[float]:
        """Embed one query string with the same model and validation."""
        return self.embed_documents([text])[0]

    def _encode(self, texts: list[str]) -> list[object]:
        model = self._load_model()
        try:
            encoded = model.encode(  # type: ignore[attr-defined]
                texts,
                batch_size=self.settings.embedding_batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Embedding inference failed: {exc}") from exc
        if hasattr(encoded, "tolist"):
            return list(encoded)
        return list(encoded)

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model
        if self.settings.embedding_device == "cuda":
            self._require_cuda()
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Loading embedding model %s on %s",
                self.settings.embedding_model,
                self.settings.embedding_device,
            )
            self._model = SentenceTransformer(
                self.settings.embedding_model,
                device=self.settings.embedding_device,
            )
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model {self.settings.embedding_model}: {exc}"
            ) from exc
        return self._model

    @staticmethod
    def _require_cuda() -> None:
        try:
            import torch
        except Exception as exc:
            raise EmbeddingError(f"CUDA was requested but PyTorch could not be imported: {exc}") from exc
        if not torch.cuda.is_available():
            raise EmbeddingError("CUDA was requested but no GPU is available.")

    @staticmethod
    def _reject_invalid_texts(texts: list[str]) -> None:
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise EmbeddingError("Cannot embed empty or non-text input.")


def validate_embedding(vector: list[float], dimensions: int) -> None:
    """Require a finite vector of the configured length."""
    if len(vector) != dimensions:
        raise EmbeddingError(
            f"Expected an embedding of {dimensions} dimensions, received {len(vector)}."
        )
    if not vector:
        raise EmbeddingError("Embedding is empty.")
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmbeddingError("Embedding contains a non-numeric value.")
        if not math.isfinite(float(value)):
            raise EmbeddingError("Embedding contains a non-finite value.")


def _as_float_list(vector: object) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    if not isinstance(vector, list):
        raise EmbeddingError("Embedding model returned a value that is not a vector.")
    return [float(value) for value in vector]
