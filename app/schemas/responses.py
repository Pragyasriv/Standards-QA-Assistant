"""Response models for the Phase 1 API."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness payload returned by the health endpoint."""

    status: str = Field(examples=["ok"])
    version: str


class ProjectInfoResponse(BaseModel):
    """Public description of this service."""

    name: str
    version: str
    description: str
    environment: str


class ErrorResponse(BaseModel):
    """Structured error payload shared by exception handlers."""

    error: str
    message: str
    details: list[dict[str, object]] | None = None
