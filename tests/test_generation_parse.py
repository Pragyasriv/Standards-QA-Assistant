"""Parsing structured model output from LangChain Groq responses."""

import json

import pytest

from app.generation.errors import GenerationValidationError
from app.generation.models import ModelDraft
from app.generation.parse import assistant_text_candidates, parse_model_output


def _payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "answer": "Lead (0,1 %)",
        "cited_chunk_ids": ["02011L0065-20260701:00141"],
        "abstained": False,
        "abstention_reason": None,
    }
    base.update(overrides)
    return base


def test_parse_dict_payload() -> None:
    draft = parse_model_output(_payload())
    assert isinstance(draft, ModelDraft)
    assert draft.answer == "Lead (0,1 %)"


def test_parse_json_string_content() -> None:
    draft = parse_model_output(json.dumps(_payload()))
    assert draft.cited_chunk_ids == ["02011L0065-20260701:00141"]


def test_parse_groq_text_content_block_list() -> None:
    """Groq may place the final JSON in a text block alongside a reasoning block."""
    from langchain_core.messages import AIMessage

    draft = parse_model_output(
        AIMessage(
            content=[
                {"type": "reasoning", "reasoning": "plan the JSON"},
                {"type": "text", "text": json.dumps(_payload())},
            ]
        )
    )
    assert "Lead (0,1 %)" in draft.answer


def test_parse_reads_content_blocks_when_content_empty() -> None:
    """LangChain may expose the JSON only through content_blocks."""

    class _GroqStyleMessage:
        content = ""

        @property
        def content_blocks(self) -> list[dict[str, str]]:
            return [
                {"type": "reasoning", "reasoning": "chain of thought"},
                {"type": "text", "text": json.dumps(_payload())},
            ]

    draft = parse_model_output(_GroqStyleMessage())
    assert draft.cited_chunk_ids == ["02011L0065-20260701:00141"]

    class _Empty:
        content = ""

        @property
        def content_blocks(self) -> list[dict[str, str]]:
            return [{"type": "reasoning", "reasoning": "no final text"}]

    with pytest.raises(GenerationValidationError, match="no assistant text"):
        parse_model_output(_Empty())


def test_parse_json_embedded_in_prose() -> None:
    wrapped = f"Here is the response:\n{json.dumps(_payload())}\nThanks."
    draft = parse_model_output(wrapped)
    assert draft.answer == "Lead (0,1 %)"


def test_parse_partial_content_uses_complete_content_blocks() -> None:
    complete = json.dumps(_payload())

    class _SplitMessage:
        content = '{"answer": "incomplete'

        @property
        def content_blocks(self) -> list[dict[str, str]]:
            return [{"type": "text", "text": complete}]

    draft = parse_model_output(_SplitMessage())
    assert draft.answer == "Lead (0,1 %)"


def test_assistant_text_candidates_dedupe_identical_content_and_blocks() -> None:
    complete = json.dumps(_payload())

    class _Duplicated:
        content = complete

        @property
        def content_blocks(self) -> list[dict[str, str]]:
            return [{"type": "text", "text": complete}]

    assert assistant_text_candidates(_Duplicated()) == [complete]


def test_parse_ignores_reasoning_blocks_in_both_sources() -> None:
    complete = json.dumps(_payload())

    class _ReasoningOnlyInContent:
        content = [
            {"type": "reasoning", "reasoning": "plan"},
            {"type": "text", "text": complete},
        ]

        @property
        def content_blocks(self) -> list[dict[str, str]]:
            return [{"type": "reasoning", "reasoning": "more"}]

    draft = parse_model_output(_ReasoningOnlyInContent())
    assert draft.cited_chunk_ids == ["02011L0065-20260701:00141"]


def test_parse_rejects_malformed_model_draft() -> None:
    with pytest.raises(GenerationValidationError, match="malformed structured output"):
        parse_model_output(json.dumps({"answer": "text", "abstained": False}))


def test_parse_rejects_prose_without_json() -> None:
    with pytest.raises(GenerationValidationError):
        parse_model_output("The supplied text lists Lead only.")