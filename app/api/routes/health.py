"""Health check route."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_app_settings
from app.core.config import Settings
from app.schemas.responses import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check(
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> HealthResponse:
    """Report that the process is running."""
    return HealthResponse(status="ok", version=settings.app_version)
