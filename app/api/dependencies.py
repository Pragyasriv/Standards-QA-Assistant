"""Shared FastAPI dependencies."""

from fastapi import Request

from app.ask.service import AskService
from app.core.config import Settings, get_settings
from app.core.exceptions import AppError


def get_app_settings() -> Settings:
    """Provide the cached settings object to routes."""
    return get_settings()


def get_ask_service(request: Request) -> AskService:
    """Return the process-wide question service opened during startup."""
    service = getattr(request.app.state, "ask_service", None)
    if service is None:
        raise AppError(
            "The question service is not ready.",
            status_code=503,
            error_code="not_ready",
        )
    return service
