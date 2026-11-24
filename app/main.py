"""FastAPI application factory."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.exception_handlers import register_exception_handlers
from app.api.router import api_router
from app.ask.errors import StartupError
from app.ask.startup import prepare_ask_service
from app.core.config import get_settings
from app.core.logging_config import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Log process start and shutdown."""
    settings = get_settings()
    logger.info(
        "Starting %s version=%s environment=%s",
        settings.app_name,
        settings.app_version,
        settings.app_env,
    )
    _app.state.ask_service = None
    _app.state.retrieval = None
    if settings.app_env != "test":
        try:
            ask_service, retrieval = prepare_ask_service()
        except StartupError as exc:
            logger.error("Startup failed: %s", exc)
            raise
        _app.state.ask_service = ask_service
        _app.state.retrieval = retrieval
    try:
        yield
    finally:
        retrieval = getattr(_app.state, "retrieval", None)
        closer = getattr(retrieval, "close", None)
        if closer is not None:
            closer()
        logger.info("Stopping %s", settings.app_name)


def create_app() -> FastAPI:
    """Build a configured FastAPI application."""
    settings = get_settings()
    configure_logging(settings.log_level)

    expose_docs = settings.app_env != "production"
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
        lifespan=lifespan,
        docs_url="/docs" if expose_docs else None,
        redoc_url="/redoc" if expose_docs else None,
        openapi_url="/openapi.json" if expose_docs else None,
    )
    origins = [
        origin.strip()
        for origin in settings.frontend_origin.split(",")
        if origin.strip()
    ]

    if origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )
    register_exception_handlers(application)
    application.include_router(api_router)
    return application


app = create_app()


def main() -> None:
    """Run the ASGI server using configured host and port."""
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
