"""Temporary generation diagnostics for model responses."""

import json
import logging

import pytest
from langchain_core.messages import AIMessage

from tests.test_generation import _approved, _chunk, _draft, _service


def test_model_response_diagnostic_logs_before_parse(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="app.generation.service")
    payload = json.dumps(_draft())
    service, _client = _service(AIMessage(content=payload))
    service.generate("What substances are restricted under the RoHS Directive?", _approved([_chunk()]))
    assert "TEMPORARY DIAGNOSTIC model_response" in caplog.text
    assert "content_type=str" in caplog.text or "content_type" in caplog.text
    assert "candidate_lengths=" in caplog.text


def test_diagnostics_match_parser_candidates_for_content_blocks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from app.generation.parse import assistant_text_candidates

    caplog.set_level(logging.INFO, logger="app.generation.service")

    class _BlockMessage:
        content = ""

        @property
        def content_blocks(self) -> list[dict[str, str]]:
            return [{"type": "text", "text": json.dumps(_draft())}]

    message = _BlockMessage()
    expected = assistant_text_candidates(message)
    service, _client = _service(message)
    service.generate(
        "What substances are restricted under the RoHS Directive?",
        _approved([_chunk()]),
    )
    assert expected
    assert f"candidate_lengths=[{len(expected[0])}]" in caplog.text


def test_parse_phase_logged_when_json_invalid(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="app.generation.service")
    service, _client = _service("not json")
    with pytest.raises(Exception):
        service.generate(
            "What substances are restricted under the RoHS Directive?",
            _approved([_chunk()]),
        )
    assert "TEMPORARY DIAGNOSTIC model_response" in caplog.text
    assert "phase=parse" in caplog.text