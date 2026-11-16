"""Validate question text before classification."""

import re

from pydantic import BaseModel, ConfigDict

from app.guardrails.settings import GuardrailSettings

_READABLE = re.compile(r"[A-Za-z0-9]")


class InputCheck(BaseModel):
    """Result of the input checks."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    text: str = ""
    reason: str | None = None


def validate_input(question: object, settings: GuardrailSettings | None = None) -> InputCheck:
    """Accept a readable question, or return a structured invalid result."""
    active = settings or GuardrailSettings()
    if not isinstance(question, str):
        return InputCheck(ok=False, reason="Question must be text.")
    if "\x00" in question:
        return InputCheck(ok=False, reason="Question contains a malformed null character.")
    stripped = question.strip()
    if not stripped:
        return InputCheck(ok=False, reason="Question is empty.")
    if len(stripped) > active.guardrail_max_question_length:
        return InputCheck(
            ok=False,
            reason=(
                f"Question exceeds the maximum length of {active.guardrail_max_question_length} characters."
            ),
        )
    if _READABLE.search(stripped) is None:
        return InputCheck(ok=False, reason="Question does not contain a readable request.")
    return InputCheck(ok=True, text=stripped)
