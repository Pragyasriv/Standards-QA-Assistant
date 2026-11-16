"""Validate model-selected chunk IDs and copy citation fields from evidence."""

from __future__ import annotations

import re

from app.generation.errors import GenerationValidationError
from app.generation.fidelity import check_fidelity
from app.generation.models import ModelDraft, ResolvedCitation
from app.retrieval.models import RetrievedChunk

_LIMITATION = re.compile(
    r"do not establish|does not establish|cannot be established|not established|"
    r"do not contain|does not contain|not enough evidence|insufficient evidence|"
    r"supplied documents do not|supplied evidence does not",
    re.IGNORECASE,
)
_CLAIM = re.compile(
    r"\bshall\b|\bmust\b|\bis required\b|\d+,\d+\s*%|\d+\.\d+\s*%",
    re.IGNORECASE,
)


def validate_draft(
    draft: ModelDraft,
    chunks: list[RetrievedChunk],
    question: str = "",
) -> tuple[str, str | None, list[ResolvedCitation]]:
    """Return status, abstention reason, and trusted citations.

    An unknown chunk ID rejects the whole output. Valid IDs are not returned
    beside an ID that was not in the evidence.
    """
    known = {chunk.chunk_id: chunk for chunk in chunks}
    cited = [chunk_id.strip() for chunk_id in draft.cited_chunk_ids]
    unknown = [chunk_id for chunk_id in cited if chunk_id not in known]
    if unknown:
        raise GenerationValidationError(
            "The model cited chunk IDs that were not in the supplied evidence: "
            + ", ".join(unknown)
            + "."
        )
    answer = draft.answer.strip()
    reason = (draft.abstention_reason or "").strip() or None
    if draft.abstained:
        return _abstention(answer, reason, cited)
    if not answer:
        raise GenerationValidationError(
            "The model returned an empty answer without a valid abstention."
        )
    if not cited:
        raise GenerationValidationError("An answer must cite at least one supplied chunk.")
    citations = _citations(cited, known)
    check_fidelity(answer, [known[citation.chunk_id] for citation in citations], question)
    return "answered", None, citations


def _abstention(
    answer: str,
    reason: str | None,
    cited: list[str],
) -> tuple[str, str | None, list[ResolvedCitation]]:
    if reason is None:
        raise GenerationValidationError("The model abstained without an abstention reason.")
    if cited:
        raise GenerationValidationError(
            "The model abstained and also cited evidence as support for an answer."
        )
    text = answer or reason
    if _CLAIM.search(text) and not _LIMITATION.search(text):
        raise GenerationValidationError(
            "The model abstained and also stated a conclusion the abstention does not support."
        )
    return "abstained", reason, []


def _citations(
    cited: list[str],
    known: dict[str, RetrievedChunk],
) -> list[ResolvedCitation]:
    ordered: list[str] = []
    seen: set[str] = set()
    for chunk_id in cited:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        ordered.append(chunk_id)
    return [_from_chunk(known[chunk_id]) for chunk_id in ordered]


def _from_chunk(chunk: RetrievedChunk) -> ResolvedCitation:
    """Copy stored metadata. Missing optional fields stay null."""
    return ResolvedCitation(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        regulation=chunk.regulation,
        regulation_number=chunk.regulation_number,
        article=chunk.article,
        paragraph=chunk.paragraph,
        subparagraph=chunk.subparagraph,
        annex=chunk.annex,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        source_filename=chunk.source_filename,
    )
