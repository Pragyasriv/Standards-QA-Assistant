"""Turn a LangChain model result into the four expected fields."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from app.generation.errors import GenerationValidationError
from app.generation.models import ModelDraft

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_model_output(value: object) -> ModelDraft:
    """Accept a structured object or a JSON message. Reject anything else."""
    if isinstance(value, ModelDraft):
        return value
    if isinstance(value, dict):
        return _validate(value)
    parsed = getattr(value, "parsed", None)
    if isinstance(parsed, ModelDraft):
        return parsed
    if isinstance(parsed, dict):
        return _validate(parsed)
    text = _message_text(value)
    text = _THINK.sub("", text)
    text = _FENCE.sub("", text.strip())
    return _validate(_json_object(text))


def _validate(payload: object) -> ModelDraft:
    try:
        return ModelDraft.model_validate(payload)
    except ValidationError as exc:
        raise GenerationValidationError(
            "The model returned malformed structured output."
        ) from exc


def _json_object(text: str) -> object:
    candidate = text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            raise GenerationValidationError("The model did not return valid structured output.")
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise GenerationValidationError(
                "The model did not return valid structured output."
            ) from exc


def _message_text(value: object) -> str:
    if isinstance(value, str):
        return value
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        if parts:
            return "\n".join(parts)
    raise GenerationValidationError("The model did not return valid structured output.")
