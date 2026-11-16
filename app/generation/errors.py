"""Failures from answer generation. Messages never include an API key."""

from __future__ import annotations

import re

from app.core.exceptions import AppError

_SECRET = re.compile(r"gsk_[A-Za-z0-9_\-]+")


class GenerationError(AppError):
    """A generation request could not be completed."""

    def __init__(self, message: str, *, error_code: str = "generation_error") -> None:
        super().__init__(message, status_code=400, error_code=error_code)


class GenerationConfigError(GenerationError):
    """Groq settings are missing or unusable."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="generation_config")


class GenerationRejected(GenerationError):
    """The guardrail result is not approved, so no model call is made."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="generation_rejected")


class GenerationInvocationError(GenerationError):
    """The configured model could not be called."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="generation_invocation")


class GenerationValidationError(GenerationError):
    """The model output cannot be trusted as a grounded answer."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="generation_validation")


def public_error_text(exc: BaseException, secret: str | None = None) -> str:
    """Return a short error description with secrets removed."""
    text = str(exc)
    if secret:
        text = text.replace(secret, "[redacted]")
    text = _SECRET.sub("[redacted]", text)
    return text[:500]
