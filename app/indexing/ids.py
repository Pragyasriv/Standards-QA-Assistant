"""Deterministic Qdrant point identifiers."""

import uuid

# Stable namespace so the same chunk_id always maps to the same point id.
_POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "standards-qa-assistant.qdrant")


def point_id_for_chunk(chunk_id: str) -> str:
    """Return the UUID used as the Qdrant point id for a chunk.

    The mapping is a UUID5 of the chunk id. Repeating the index updates the
    same point instead of inserting a second copy.
    """
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))
