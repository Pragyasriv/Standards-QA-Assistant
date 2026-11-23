"""Regression tests for list-question intent detection across equivalent wordings."""

import pytest

from app.generation.intent import asks_for_complete_list
from app.guardrails.sufficiency import asks_for_inventory


@pytest.mark.parametrize(
    "question",
    [
        "What substances are restricted under the RoHS Directive?",
        "What are the restricted substances under RoHS?",
        "Which substances are restricted by RoHS?",
        "List all restricted substances under RoHS.",
    ],
)
def test_equivalent_list_questions_request_inventory_and_complete_list(question: str) -> None:
    assert asks_for_inventory(question) is True
    assert asks_for_complete_list(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "What is the concentration limit for lead?",
        "What does RoHS Article 6 require?",
        "When does the DEHP restriction apply?",
        "What is the scope of REACH?",
        "What maximum concentration value does RoHS Annex II tolerate for lead?",
    ],
)
def test_non_inventory_questions_are_not_complete_list_requests(question: str) -> None:
    assert asks_for_complete_list(question) is False


@pytest.mark.parametrize(
    "question",
    [
        "What is the concentration limit for lead?",
        "What does RoHS Article 6 require?",
        "When does the DEHP restriction apply?",
        "What is the scope of REACH?",
    ],
)
def test_non_inventory_questions_do_not_request_inventory(question: str) -> None:
    assert asks_for_inventory(question) is False