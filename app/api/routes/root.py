"""Service information route."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_app_settings
from app.core.config import Settings
from app.schemas.responses import ProjectInfoResponse

router = APIRouter(tags=["info"])


@router.get("/", response_model=ProjectInfoResponse)
def project_info(
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> ProjectInfoResponse:
    """Return basic project information from configuration."""
    return ProjectInfoResponse(
        name=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
        environment=settings.app_env,
    )
