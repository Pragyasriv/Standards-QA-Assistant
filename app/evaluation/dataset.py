"""Schema for the 50-question RAGAS dataset."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RegulationName = Literal["RoHS", "REACH"]

_CHUNK_ID = re.compile(r"^(02011L0065-20260701|02006R1907-20260622):(\d{5})$")
_PREFIX = {"RoHS": "02011L0065-20260701", "REACH": "02006R1907-20260622"}
EXPECTED_COUNT = 50
EXPECTED_PER_REGULATION = 25


class RagasQuestion(BaseModel):
    """One answerable question with a source-checked reference."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    regulation: RegulationName
    reference_answer: str
    reference_contexts: list[str] = Field(min_length=1)
    expected_chunk_ids: list[str] = Field(min_length=1)
    article: str | None = None
    annex: str | None = None
    paragraph: str | None = None
    verified_phrases: list[str] = Field(min_length=1)
    notes: str | None = None


class DatasetProblems(ValueError):
    """The question file failed schema or source checks."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


def load_questions(path: Path) -> list[RagasQuestion]:
    """Read and validate the question file. This does not open the chunk corpus."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise DatasetProblems(["the question file must be a JSON list"])
    questions = [RagasQuestion.model_validate(item) for item in raw]
    problems = schema_problems(questions)
    if problems:
        raise DatasetProblems(problems)
    return questions


def schema_problems(questions: list[RagasQuestion]) -> list[str]:
    """Return schema errors. An empty list means the dataset shape is valid."""
    problems: list[str] = []
    if len(questions) != EXPECTED_COUNT:
        problems.append(f"expected {EXPECTED_COUNT} questions, found {len(questions)}")
    ids = [item.id for item in questions]
    if len(ids) != len(set(ids)):
        problems.append("question ids are not unique")
    counts = {name: sum(item.regulation == name for item in questions) for name in ("RoHS", "REACH")}
    for name, expected in (("RoHS", EXPECTED_PER_REGULATION), ("REACH", EXPECTED_PER_REGULATION)):
        if counts[name] != expected:
            problems.append(f"expected {expected} {name} questions, found {counts[name]}")
    for item in questions:
        if not item.reference_answer.strip():
            problems.append(f"{item.id} is missing a reference answer")
        if any(not context.strip() for context in item.reference_contexts):
            problems.append(f"{item.id} has an empty reference context")
        if len(item.reference_contexts) != len(item.expected_chunk_ids):
            problems.append(f"{item.id} context count does not match expected chunk ids")
        for chunk_id in item.expected_chunk_ids:
            match = _CHUNK_ID.fullmatch(chunk_id)
            if match is None or match.group(1) != _PREFIX[item.regulation]:
                problems.append(f"{item.id} has an invalid source reference {chunk_id}")
        combined = "\n".join(item.reference_contexts)
        for phrase in item.verified_phrases:
            if phrase not in combined:
                problems.append(f"{item.id} verified phrase is not in its reference contexts")
    return problems


def source_problems(questions: list[RagasQuestion], chunks: dict[str, str]) -> list[str]:
    """Return errors when a cited chunk is missing or the context is not that chunk."""
    problems: list[str] = []
    for item in questions:
        for chunk_id, context in zip(item.expected_chunk_ids, item.reference_contexts, strict=True):
            source = chunks.get(chunk_id)
            if source is None:
                problems.append(f"{item.id} cites missing chunk {chunk_id}")
                continue
            if context != source:
                problems.append(f"{item.id} reference context is not the text of {chunk_id}")
        for phrase in item.verified_phrases:
            if not any(phrase in chunks.get(chunk_id, "") for chunk_id in item.expected_chunk_ids):
                problems.append(f"{item.id} verified phrase is not in the source chunk")
    return problems


def load_chunk_texts(*paths: Path) -> dict[str, str]:
    """Read chunk id to text from JSONL files. Missing files are skipped."""
    chunks: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            chunk_id = row.get("chunk_id")
            text = row.get("text")
            if isinstance(chunk_id, str) and isinstance(text, str):
                chunks[chunk_id] = text
    return chunks
