"""Errors raised while embedding chunks or writing them to Qdrant."""


class IndexingError(Exception):
    """A chunk file or indexing step cannot continue."""


class EmbeddingError(IndexingError):
    """The embedding model failed to load or produced an unusable vector."""


class CollectionCompatibilityError(IndexingError):
    """An existing Qdrant collection does not match the required vector setup.

    The collection is left in place. It is not deleted or recreated.
    """
