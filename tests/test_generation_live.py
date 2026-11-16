"""Optional Groq call. The default suite does not set these variables."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.generation.service import GenerationService
from app.guardrails.models import GuardrailResponse
from app.retrieval.models import RetrievedChunk

_KEY = os.environ.get("GROQ_API_KEY", "").strip()
_MODEL = os.environ.get("GROQ_MODEL", "").strip()
_ENABLED = bool(_KEY) and not _KEY.lower().startswith("your_") and bool(_MODEL) and not _MODEL.lower().startswith("your_")

pytestmark = pytest.mark.skipif(
    not _ENABLED,
    reason="GROQ_API_KEY and GROQ_MODEL are not set in the process environment.",
)


def test_live_groq_answers_from_rohs_annex_ii() -> None:
    chunk = _annex_ii_chunk()
    guardrail = GuardrailResponse(
        status="approved",
        scope="supported",
        reason="Supplied Annex II evidence.",
        detected_regulation="RoHS",
        evidence_available=True,
        retrieved_chunks=[chunk],
        annex="II",
    )
    response = GenerationService().generate(
        "What are the restricted substances under RoHS?",
        guardrail,
    )
    assert response.model == _MODEL
    assert response.status in {"answered", "abstained"}
    assert "gsk_" not in response.answer
    assert response.generation_metadata.evidence_chunk_count == 1
    if response.status == "answered":
        assert response.citations
        assert {item.chunk_id for item in response.citations} == {chunk.chunk_id}
        citation = response.citations[0]
        assert citation.regulation == "RoHS"
        assert citation.regulation_number == "2011/65/EU"
        assert citation.annex == "II"
        assert citation.page_start == 22
        assert citation.source_filename == chunk.source_filename


def _annex_ii_chunk() -> RetrievedChunk:
    path = Path("data/processed/rohs/chunks.jsonl")
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["chunk_id"] == "02011L0065-20260701:00141":
            return RetrievedChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                regulation=row["regulation"],
                regulation_number=row["regulation_number"],
                text=row["text"],
                article=row["article"],
                paragraph=row["paragraph"],
                subparagraph=row["subparagraph"],
                annex=row["annex"],
                annex_section=row["annex_section"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                consolidation_date=row["consolidation_date"],
                source_filename=row["source_filename"],
            )
    raise AssertionError("RoHS Annex II chunk 00141 was not in the local chunk file.")
