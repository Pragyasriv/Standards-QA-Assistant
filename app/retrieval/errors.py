"""Errors raised while retrieving regulatory chunks."""


class RetrievalError(Exception):
    """A search could not be completed."""


class InvalidQuestionError(RetrievalError):
    """The question is missing or not usable text."""


class InvalidFilterError(RetrievalError):
    """A metadata filter is missing, blank, or not one of the allowed values."""
