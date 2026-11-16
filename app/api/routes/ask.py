"""Question endpoint. Guardrails run before any model call."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.api.dependencies import get_ask_service
from app.ask.readiness import assess_readiness
from app.ask.service import AskService
from app.schemas.ask import AskRequest, AskResponse, ReadinessResponse
from app.schemas.responses import ErrorResponse

router = APIRouter(tags=["questions"])


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={
        422: {"model": ErrorResponse, "description": "The JSON body is not a question string."},
        502: {"model": ErrorResponse, "description": "The configured model could not be called."},
        503: {
            "model": ErrorResponse,
            "description": "Generation is not configured, or local retrieval resources are not ready.",
        },
    },
)
def ask_question(
    body: AskRequest,
    service: Annotated[AskService, Depends(get_ask_service)],
) -> AskResponse:
    """Ask a RoHS or REACH question.

    The guardrails run first. Generation is called only when they return
    approved. Citations in an answered or model-abstained response are copied
    from the retrieved chunks. A validation failure returns status abstained
    with no answer and no citations.
    """
    return service.ask(body.question)


@router.get("/ready", response_model=ReadinessResponse)
def ready(request: Request) -> ReadinessResponse:
    """Report whether the local Qdrant collection and BM25 index can be opened.

    This check does not call Groq and does not create a collection.
    """
    retrieval = getattr(request.app.state, "retrieval", None)
    report = assess_readiness(retrieval)
    return ReadinessResponse(
        status="ready" if report.ready else "not_ready",
        qdrant_available=report.qdrant_available,
        bm25_index_available=report.bm25_index_available,
    )
