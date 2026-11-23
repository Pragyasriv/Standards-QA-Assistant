"""Regression checks for grounded-answer prompt instructions."""

import re

from app.generation.prompt import SYSTEM_PROMPT, build_messages
from app.retrieval.models import RetrievedChunk


def test_complete_list_instructions_cover_entry_lines_and_qualifications() -> None:
    lowered = SYSTEM_PROMPT.lower()
    assert "comma-separated" in lowered
    assert "concentration" in lowered or "percent" in lowered
    assert "bullet" in lowered or "own line" in lowered
    assert "shall apply" in lowered or "shall not apply" in lowered
    assert "paraphrase" in lowered or "shortened" in lowered


def test_build_messages_includes_system_prompt_for_list_question() -> None:
    chunk = RetrievedChunk(
        chunk_id="example:1",
        document_id="doc",
        regulation="RoHS",
        text="Lead (0,1 %)",
    )
    messages = build_messages("What substances are restricted under RoHS?", [chunk])
    system = messages[0].content
    assert isinstance(system, str)
    assert system == SYSTEM_PROMPT
    user = messages[1].content
    assert isinstance(user, str)
    assert "Lead (0,1 %)" in user
    assert not re.search(r"\bhardcoded\b", system, re.IGNORECASE)