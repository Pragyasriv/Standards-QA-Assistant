"""Serialize chunk records into the LangChain Qdrant payload shape."""

from app.processing.models import ChunkRecord

CONTENT_KEY = "page_content"
METADATA_KEY = "metadata"


def chunk_to_payload(chunk: ChunkRecord) -> dict:
    """Return a payload with the chunk text and its citation metadata.

    Missing citation fields stay null. Dates are stored as ISO strings because
    Qdrant payloads are JSON. No citation value is filled in.
    """
    data = chunk.model_dump(mode="json")
    text = data.pop("text")
    return {
        CONTENT_KEY: text,
        METADATA_KEY: data,
    }
