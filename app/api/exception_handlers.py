"""Convert exceptions into consistent JSON error responses."""

import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import AppError
from app.schemas.responses import ErrorResponse

logger = logging.getLogger(__name__)


def _error_response(status_code: int, payload: ErrorResponse) -> JSONResponse:
    """Serialize an error model without empty optional fields."""
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(exclude_none=True),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers for expected and unexpected failures."""

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.warning(
            "Application error path=%s code=%s status=%s",
            request.url.path,
            exc.error_code,
            exc.status_code,
        )
        return _error_response(
            exc.status_code,
            ErrorResponse(error=exc.error_code, message=exc.message),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        logger.info(
            "HTTP error path=%s status=%s",
            request.url.path,
            exc.status_code,
        )
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return _error_response(
            exc.status_code,
            ErrorResponse(error="http_error", message=message),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        logger.info("Validation error path=%s", request.url.path)
        return _error_response(
            422,
            ErrorResponse(
                error="validation_error",
                message="Request validation failed.",
                details=jsonable_encoder(exc.errors()),
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger.error(
            "Unhandled error path=%s",
            request.url.path,
            exc_info=exc,
        )
        return _error_response(
            500,
            ErrorResponse(
                error="internal_error",
                message="An unexpected error occurred.",
            ),
        )
