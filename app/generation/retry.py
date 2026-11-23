"""When validation may trigger one targeted regeneration attempt."""

from __future__ import annotations

from app.generation.errors import GenerationValidationError
from app.generation.intent import asks_for_complete_list

_COMPLETE_LIST_RETRY_MARKERS = (
    "omits an application condition or exception",
    "presents a partial list as exhaustive",
)


def is_complete_list_fidelity_retry_eligible(
    exc: GenerationValidationError,
    question: str,
) -> bool:
    """True only for complete-list questions that failed entry or qualification coverage."""
    if not asks_for_complete_list(question):
        return False
    message = str(exc)
    return any(marker in message for marker in _COMPLETE_LIST_RETRY_MARKERS)