"""Startup failures for local retrieval resources."""


class StartupError(RuntimeError):
    """A required local resource is missing or incompatible.

    The Qdrant collection is not created and the BM25 index is not rebuilt.
    """
