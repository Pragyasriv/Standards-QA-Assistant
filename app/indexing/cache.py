"""Cache embedding vectors so an unchanged chunk is not encoded again."""

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Append-only cache keyed by model name and chunk text."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._vectors: dict[tuple[str, str], list[float]] = {}
        self._load()

    @property
    def size(self) -> int:
        """Return the number of cached vectors currently loaded."""
        return len(self._vectors)

    def get(self, model_name: str, text: str) -> list[float] | None:
        """Return a cached vector for this model and text, if one exists."""
        return self._vectors.get((model_name, _text_key(text)))

    def put(self, model_name: str, text: str, vector: list[float]) -> None:
        """Store a vector and append it to the cache file."""
        key = (model_name, _text_key(text))
        if key in self._vectors:
            return
        self._vectors[key] = vector
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "model": model_name,
            "text_sha256": key[1],
            "vector": vector,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _load(self) -> None:
        if not self.path.is_file():
            return
        kept = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                vector = record["vector"]
                if not isinstance(vector, list):
                    continue
                self._vectors[(record["model"], record["text_sha256"])] = vector
                kept += 1
            except (KeyError, TypeError, json.JSONDecodeError):
                logger.warning("Skipped an unreadable embedding cache record in %s", self.path)
        logger.info("Loaded %s cached embeddings from %s", kept, self.path)


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
