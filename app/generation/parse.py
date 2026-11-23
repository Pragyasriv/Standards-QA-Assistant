"""Turn a LangChain model result into the four expected fields."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from app.generation.errors import GenerationValidationError
from app.generation.models import ModelDraft

_THINK = re.compile(r".*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def message_content_shape(value: object) -> tuple[str, list[str]]:
    """Describe message.content and content block types for logging."""
    content = getattr(value, "content", None)
    content_type = type(content).__name__
    block_types: list[str] = []
    blocks = _read_content_blocks(value)
    for item in blocks:
        if isinstance(item, dict) and item.get("type") is not None:
            block_types.append(str(item.get("type")))
    return content_type, block_types


def assistant_text_candidates(value: object) -> list[str]:
    """Ordered, deduped text sources from content then content_blocks.

    Sources are not merged across content and content_blocks. When both carry
    text, each distinct candidate is returned for independent JSON parsing.
    """
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []

    candidates: list[str] = []
    seen: set[str] = set()
    content = getattr(value, "content", None)
    _extend_candidates(candidates, seen, _text_parts_from_blocks(content))
    _extend_candidates(candidates, seen, _text_parts_from_blocks(_read_content_blocks(value)))
    return candidates


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

    candidates = assistant_text_candidates(value)
    if not candidates:
        raise GenerationValidationError(
            "The model did not return valid structured output (no assistant text content)."
        )

    errors: list[GenerationValidationError] = []
    for candidate in candidates:
        try:
            text = _THINK.sub("", candidate)
            text = _FENCE.sub("", text.strip())
            return _validate(_json_object(text))
        except GenerationValidationError as exc:
            errors.append(exc)
    raise errors[-1]


def _validate(payload: object) -> ModelDraft:
    try:
        return ModelDraft.model_validate(payload)
    except ValidationError as exc:
        raise GenerationValidationError(
            "The model returned malformed structured output."
        ) from exc


def _json_object(text: str) -> object:
    candidate = text.strip()
    if not candidate:
        raise GenerationValidationError(
            "The model did not return valid structured output (empty message content)."
        )
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            raise GenerationValidationError(
                "The model did not return valid structured output (no JSON object found)."
            )
        fragment = candidate[start : end + 1]
        try:
            return json.loads(fragment)
        except json.JSONDecodeError as exc:
            raise GenerationValidationError(
                "The model did not return valid structured output "
                f"(JSON decode error: {exc.msg} at position {exc.pos})."
            ) from exc


def _extend_candidates(
    candidates: list[str],
    seen: set[str],
    parts: list[str],
) -> None:
    if not parts:
        return
    joined = "\n".join(parts)
    _append_candidate(candidates, seen, joined)
    for part in parts:
        _append_candidate(candidates, seen, part)


def _append_candidate(candidates: list[str], seen: set[str], text: str) -> None:
    normalized = text.strip()
    if not normalized or normalized in seen:
        return
    seen.add(normalized)
    candidates.append(normalized)


def _read_content_blocks(value: object) -> list[object]:
    blocks = getattr(value, "content_blocks", None)
    if blocks is None:
        return []
    if callable(blocks):
        blocks = blocks()
    if isinstance(blocks, list):
        return blocks
    return []


def _text_parts_from_blocks(content: object) -> list[str]:
    """Collect final assistant text blocks, not chain-of-thought reasoning."""
    if isinstance(content, str):
        return [content] if content.strip() else []
    if not isinstance(content, list):
        return []
    parts: list[str] = []
    for block in content:
        if isinstance(block, str) and block.strip():
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "reasoning":
            continue
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text)
    return parts