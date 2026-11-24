"""Local persistent Qdrant collection for regulatory chunks."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.indexing.errors import CollectionCompatibilityError, IndexingError
from app.indexing.ids import point_id_for_chunk
from app.indexing.payload import chunk_to_payload
from app.indexing.settings import IndexingSettings
from app.processing.models import ChunkRecord

logger = logging.getLogger(__name__)


class QdrantStore:
    """Open one local Qdrant collection and upsert precomputed vectors.

    The LangChain ``QdrantVectorStore`` is the connection to that collection.
    Vectors are calculated by the embedding service before they are written,
    so this class does not encode text and does not run retrieval.
    """

    def __init__(self, settings: IndexingSettings) -> None:
        self.settings = settings
        self.client: QdrantClient | None = None
        self.vector_store: QdrantVectorStore | None = None

    def open(self) -> QdrantVectorStore:
        if not self.settings.qdrant_url and not self.settings.qdrant_api_key:
            self.settings.qdrant_path.mkdir(parents=True, exist_ok=True)

        self._connect()
        try:
            self._ensure_collection()
            return self._attach_vector_store()
        except Exception:
            self.close()
            raise

    def open_existing(self) -> QdrantVectorStore:
        """Open an existing collection without creating or changing it."""

        # For local Qdrant, verify the storage directory exists first.
        # QdrantClient(path=...) may create a missing directory.
        if not self.settings.qdrant_url and not self.settings.qdrant_api_key:
            if not self.settings.qdrant_path.exists():
                raise CollectionCompatibilityError(
                    f"Qdrant storage path {self.settings.qdrant_path} not found. "
                    "The storage directory was not created."
                )

        self._connect()

        try:
            name = self.settings.qdrant_collection

            if not self.client.collection_exists(name):
                raise CollectionCompatibilityError(
                    f"Qdrant collection {name!r} not found. "
                    "The collection was not created."
                )

            self._validate_vectors()
            return self._attach_vector_store()

        except Exception:
            self.close()
            raise

    def _connect(self) -> None:
        url = self.settings.qdrant_url
        api_key = self.settings.qdrant_api_key

        if bool(url) != bool(api_key):
            raise CollectionCompatibilityError(
                "Both QDRANT_URL and QDRANT_API_KEY must be configured together."
            )

        if url and api_key:
            self.client = QdrantClient(url=url, api_key=api_key)
            logger.info("Connected to Qdrant Cloud.")
        else:
            self.client = QdrantClient(path=str(self.settings.qdrant_path))
            logger.info("Connected to local Qdrant at %s", self.settings.qdrant_path)


    def close(self) -> None:
        """Release the Qdrant connection."""
        if self.client is not None:
            self.client.close()
            self.client = None
            self.vector_store = None

    def upsert(self, chunks: Sequence[ChunkRecord], vectors: Sequence[Sequence[float]]) -> None:
        """Insert or replace points. The point id is derived from ``chunk_id``."""
        if len(chunks) != len(vectors):
            raise IndexingError(f"Received {len(vectors)} embeddings for {len(chunks)} chunks.")
        points = [
            models.PointStruct(
                id=point_id_for_chunk(chunk.chunk_id),
                vector=list(vector),
                payload=chunk_to_payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        client = self._require_client()
        client.upsert(
            collection_name=self.settings.qdrant_collection,
            points=points,
            wait=True,
        )

    def count(self) -> int:
        """Return the exact number of points in the collection."""
        result = self._require_client().count(
            collection_name=self.settings.qdrant_collection,
            exact=True,
        )
        return int(result.count)

    def count_by_regulation(self) -> dict[str, int]:
        """Count stored points for each regulation value in the payload."""
        counts: dict[str, int] = {}
        offset = None
        client = self._require_client()
        while True:
            points, offset = client.scroll(
                collection_name=self.settings.qdrant_collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                metadata = payload.get("metadata") or {}
                regulation = metadata.get("regulation")
                if isinstance(regulation, str):
                    counts[regulation] = counts.get(regulation, 0) + 1
            if offset is None:
                break
        return counts

    def retrieve_payload(self, chunk_id: str) -> dict | None:
        """Return the stored payload for a chunk, or None when it is absent."""
        points = self._require_client().retrieve(
            collection_name=self.settings.qdrant_collection,
            ids=[point_id_for_chunk(chunk_id)],
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return None
        return points[0].payload

    def _ensure_collection(self) -> None:
        client = self._require_client()
        name = self.settings.qdrant_collection
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=self.settings.embedding_dimensions,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(
                "Created Qdrant collection %s (%s dimensions, cosine) at %s",
                name,
                self.settings.embedding_dimensions,
                self.settings.qdrant_path,
            )
            return
        self._validate_vectors()
        logger.info("Reusing Qdrant collection %s at %s", name, self.settings.qdrant_path)

    def _validate_vectors(self) -> None:
        client = self._require_client()
        name = self.settings.qdrant_collection
        info = client.get_collection(name)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict) or vectors is None:
            raise CollectionCompatibilityError(
                f"Qdrant collection {name!r} uses named vectors. "
                "This project requires one unnamed vector of "
                f"{self.settings.embedding_dimensions} dimensions with cosine distance. "
                "The existing collection was not modified."
            )
        if vectors.size != self.settings.embedding_dimensions or vectors.distance != models.Distance.COSINE:
            raise CollectionCompatibilityError(
                f"Qdrant collection {name!r} has size {vectors.size} and distance "
                f"{vectors.distance}. Required size is {self.settings.embedding_dimensions} "
                "and distance is cosine. The existing collection was not modified."
            )

    def _attach_vector_store(self) -> QdrantVectorStore:
        client = self._require_client()
        self.vector_store = QdrantVectorStore(
            client=client,
            collection_name=self.settings.qdrant_collection,
            embedding=None,
            distance=models.Distance.COSINE,
            validate_embeddings=False,
            validate_collection_config=False,
        )
        return self.vector_store

    def _require_client(self) -> QdrantClient:
        if self.client is None:
            raise CollectionCompatibilityError("Qdrant storage is not open.")
        return self.client
