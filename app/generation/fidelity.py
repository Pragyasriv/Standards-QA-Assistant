"""Reject answers that change or drop qualifications present in the cited text.

The checks compare the answer with the cited chunk text. They do not decide
whether that text is a complete statement of the law, and they do not add
conditions that the chunk does not contain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.generation.errors import GenerationValidationError
from app.generation.intent import asks_for_complete_list, requested_items
from app.retrieval.models import RetrievedChunk

_PERCENT = re.compile(r"(\d+)([.,])(\d+)\s*%")
_SOURCE_DATE = re.compile(r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b")
_REWRITTEN_DATE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|[A-Z][a-z]+\s+\d{1,2},\s*\d{4})\b"
)
_ENTRY = re.compile(r"^(.+?)\s*\((\d+,\d+)\s*%\)\s*$")
_QUALIFIED = re.compile(r"shall not apply|shall apply", re.IGNORECASE)
_APPLY_TO = re.compile(r"shall not apply to\s+(.+)|shall apply to\s+(.+)", re.IGNORECASE)
_STOP_WORD = re.compile(r"\b(?:for|including|which|from|before|through)\b", re.IGNORECASE)
_UNIVERSAL = re.compile(
    r"universally|without exception|in all cases|apply to all|applies to all|"
    r"all equipment|all EEE|always applies|in every case",
    re.IGNORECASE,
)
_EXHAUSTIVE = re.compile(
    r"complete list|all restricted|the only substances|only the following|"
    r"consists of|these are the only|the restricted substances are|"
    r"restricted substances listed|the following are",
    re.IGNORECASE,
)
_LIMITATION = re.compile(
    r"not a complete list|not exhaustive|not restated|are not restated|"
    r"partial list|further entries|not every|does not include every|"
    r"conditions that are not|not reproduced|does not restate",
    re.IGNORECASE,
)
_ABBREVIATION = re.compile(r"\b[A-Z]{2,}\b")
_SKIP_ANCHORS = frozenset({"ANNEX", "THE", "AND"})


@dataclass(frozen=True)
class _Qualification:
    anchors: tuple[str, ...]
    fragments: tuple[str, ...]


def check_fidelity(answer: str, chunks: list[RetrievedChunk], question: str = "") -> None:
    """Raise when the answer changes numbers or drops cited conditions.

    A complete-list question must include every listed entry and attached
    condition in the cited text. A disclaimer does not replace those entries.
    A question about one substance or exception is checked only for that item.
    """
    evidence = "\n".join(chunk.text for chunk in chunks)
    _check_percentages(answer, evidence)
    _check_date_notation(answer, evidence)
    qualifications = _qualifications(evidence)
    entries = _entry_lines(evidence)
    if asks_for_complete_list(question):
        _require_complete_list(answer, entries, qualifications)
        return
    items = requested_items(question)
    if items:
        _require_specific_items(answer, items, entries, qualifications)
        return
    triggered = [item for item in qualifications if _mentions(answer, item.anchors)]
    if _UNIVERSAL.search(answer) and triggered:
        raise GenerationValidationError(
            "The answer states a conditional requirement as unconditional."
        )
    if _LIMITATION.search(answer):
        return
    if _EXHAUSTIVE.search(answer):
        _require_entry_lines(answer, entries)
        triggered = qualifications
    missing = _missing_fragments(answer, triggered)
    if missing:
        _raise_missing(missing)


def _require_complete_list(
    answer: str,
    entries: list[str],
    qualifications: list[_Qualification],
) -> None:
    if not entries and not qualifications:
        raise GenerationValidationError(
            "The supplied evidence does not contain the requested list. The model must abstain."
        )
    if _UNIVERSAL.search(answer) and qualifications:
        raise GenerationValidationError(
            "The answer states a conditional requirement as unconditional."
        )
    _require_entry_lines(answer, entries)
    missing = _missing_fragments(answer, qualifications)
    if missing:
        _raise_missing(missing)


def _require_specific_items(
    answer: str,
    items: tuple[str, ...],
    entries: list[str],
    qualifications: list[_Qualification],
) -> None:
    relevant_entries = [line for line in entries if _line_mentions(line, items)]
    relevant_qualifications = [item for item in qualifications if _qualification_mentions(item, items)]
    if not relevant_entries and not relevant_qualifications:
        raise GenerationValidationError(
            "The supplied evidence does not establish the requested item. The model must abstain."
        )
    if _UNIVERSAL.search(answer) and relevant_qualifications:
        raise GenerationValidationError(
            "The answer states a conditional requirement as unconditional."
        )
    _require_entry_lines(answer, relevant_entries)
    missing = _missing_fragments(answer, relevant_qualifications)
    if missing:
        _raise_missing(missing)


def _raise_missing(fragment: str) -> None:
    raise GenerationValidationError(
        "The answer omits an application condition or exception present in the cited evidence: "
        + fragment
        + "."
    )


def _check_percentages(answer: str, evidence: str) -> None:
    if re.search(r"\bper\s*cent\b", answer, re.IGNORECASE):
        raise GenerationValidationError(
            "The answer alters a numerical value from the cited evidence: percent."
        )
    allowed = {(whole, separator, fraction) for whole, separator, fraction in _PERCENT.findall(evidence)}
    for token in _PERCENT.finditer(answer):
        key = (token.group(1), token.group(2), token.group(3))
        if key not in allowed:
            shown = f"{key[0]}{key[1]}{key[2]} %"
            raise GenerationValidationError(
                "The answer alters a numerical value from the cited evidence: " + shown + "."
            )


def _check_date_notation(answer: str, evidence: str) -> None:
    if _REWRITTEN_DATE.search(answer):
        raise GenerationValidationError(
            "The answer alters a numerical value from the cited evidence: date notation."
        )
    for date in _SOURCE_DATE.findall(evidence):
        year = date.rsplit(" ", 1)[-1]
        if re.search(rf"\b{year}\b", answer) and date not in answer:
            raise GenerationValidationError(
                "The answer alters a numerical value from the cited evidence: " + date + "."
            )


def _entry_lines(evidence: str) -> list[str]:
    entries: list[str] = []
    for line in evidence.splitlines():
        cleaned = re.sub(r"\s+", " ", line.strip())
        if _ENTRY.match(cleaned):
            entries.append(cleaned)
    return entries


def _require_entry_lines(answer: str, entries: list[str]) -> None:
    squashed = re.sub(r"\s+", " ", answer)
    for line in entries:
        if line not in squashed:
            raise GenerationValidationError(
                "The answer presents a partial list as exhaustive. The cited evidence also contains: "
                + line
                + "."
            )


def _line_mentions(line: str, items: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(item)}\b", line, re.IGNORECASE) for item in items)


def _qualification_mentions(qualification: _Qualification, items: tuple[str, ...]) -> bool:
    for item in items:
        for anchor in qualification.anchors:
            if anchor.lower() == item or item in anchor.lower():
                return True
    return False


def _qualifications(evidence: str) -> list[_Qualification]:
    found: list[_Qualification] = []
    for sentence in _sentences(evidence):
        if not _QUALIFIED.search(sentence):
            continue
        target = _target(sentence)
        anchors = [item for item in _ABBREVIATION.findall(sentence) if item not in _SKIP_ANCHORS]
        if target:
            anchors.append(target)
        fragments: list[str] = []
        if re.search(r"shall not apply", sentence, re.IGNORECASE):
            fragments.append("shall not apply")
        elif re.search(r"shall apply", sentence, re.IGNORECASE):
            fragments.append("shall apply")
        if target:
            fragments.append(target)
        fragments.extend(_SOURCE_DATE.findall(sentence))
        if anchors and fragments:
            found.append(_Qualification(tuple(anchors), tuple(fragments)))
    return found


def _sentences(evidence: str) -> list[str]:
    parts: list[str] = []
    for line in evidence.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        parts.extend(part.strip() for part in re.split(r"(?<=\.)\s+(?=[A-Z])", cleaned) if part.strip())
    return parts


def _target(sentence: str) -> str:
    match = _APPLY_TO.search(sentence)
    if match is None:
        return ""
    tail = match.group(1) or match.group(2) or ""
    stop = _STOP_WORD.search(tail)
    if stop:
        tail = tail[: stop.start()]
    words = [word.strip(".,;") for word in tail.split() if word.strip(".,;")]
    return " ".join(words[:4])


def _mentions(answer: str, anchors: tuple[str, ...]) -> bool:
    for anchor in anchors:
        if " " in anchor:
            if anchor.lower() in answer.lower():
                return True
            continue
        if re.search(rf"\b{re.escape(anchor)}\b", answer):
            return True
    return False


def _missing_fragments(answer: str, qualifications: list[_Qualification]) -> str:
    folded = answer.lower()
    for item in qualifications:
        for fragment in item.fragments:
            if fragment.lower() not in folded and fragment not in answer:
                return fragment
    return ""
