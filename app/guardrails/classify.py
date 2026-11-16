"""Assign a question to RoHS, REACH, a clarification, or out of scope.

An explicit supported name selects that regulation. An explicit unsupported
instrument rejects the question. Overlap topics do not select a regulation.
"""

from __future__ import annotations

import re

from app.guardrails.catalog import GuardrailCatalog, NamedAliases, default_catalog
from app.guardrails.models import RegulationName, ScopeDecision

_ANNEX = re.compile(
    r"\bannex\s+(xvii|xvi|xiii|xii|xiv|xv|xi|viii|vii|vi|ix|iv|iii|ii|x|v|i|\d{1,2})\b",
    re.IGNORECASE,
)
_ARTICLE = re.compile(r"\barticles?\s+(\d{1,3})\b", re.IGNORECASE)
_ARABIC_ANNEX = {
    "1": "I",
    "2": "II",
    "3": "III",
    "4": "IV",
    "5": "V",
    "6": "VI",
    "7": "VII",
    "8": "VIII",
    "9": "IX",
    "10": "X",
    "11": "XI",
    "12": "XII",
    "13": "XIII",
    "14": "XIV",
    "15": "XV",
    "16": "XVI",
    "17": "XVII",
}


def classify_question(question: str, catalog: GuardrailCatalog | None = None) -> ScopeDecision:
    """Classify one already validated question."""
    active = catalog or default_catalog()
    text = " ".join(question.lower().split())
    supported = _supported_regulations(text, active)
    unsupported = _earliest_unsupported(text, active)
    article = _single_article(text)
    annex = _single_annex(text)
    topics = _matched_topics(text, active)

    if unsupported is not None:
        return ScopeDecision(
            scope="out_of_scope",
            reason=f"{unsupported.name} is outside the RoHS and REACH documents this assistant can use.",
            detected_unsupported_instrument=unsupported.name,
        )
    if len(supported) > 1:
        return ScopeDecision(
            scope="clarification_required",
            reason="The question names both RoHS and REACH.",
            clarification_question="Your question mentions both RoHS and REACH. Which regulation should I answer from?",
            article=article,
            annex=annex,
        )
    if len(supported) == 1:
        regulation = supported[0]
        return ScopeDecision(
            scope="supported",
            reason=f"The question names {regulation}.",
            detected_regulation=regulation,
            article=article,
            annex=annex,
        )
    if topics or article or annex or _has_regulatory_cue(text, active):
        return ScopeDecision(
            scope="clarification_required",
            reason="The question does not name RoHS or REACH, and it overlaps more than one instrument.",
            clarification_question=_clarification(topics, article, annex, active),
            article=article,
            annex=annex,
        )
    return ScopeDecision(
        scope="out_of_scope",
        reason="This question is not about the RoHS Directive or the REACH Regulation.",
    )


def _supported_regulations(text: str, catalog: GuardrailCatalog) -> list[RegulationName]:
    found: list[RegulationName] = []
    occupied = _occupy(text, catalog.rohs)
    if occupied:
        found.append("RoHS")
    reach_spans = _occupy(text, catalog.reach)
    if reach_spans:
        found.append("REACH")
    return found


def _earliest_unsupported(text: str, catalog: GuardrailCatalog) -> NamedAliases | None:
    earliest: tuple[int, NamedAliases] | None = None
    for instrument in catalog.unsupported:
        spans = _occupy(text, instrument)
        if not spans:
            continue
        start = min(span[0] for span in spans)
        if earliest is None or start < earliest[0]:
            earliest = (start, instrument)
    if earliest is None:
        return None
    return earliest[1]


def _matched_topics(text: str, catalog: GuardrailCatalog) -> list[str]:
    matched: list[str] = []
    for topic in catalog.topics:
        if _occupy(text, NamedAliases(name=topic.topic, aliases=topic.phrases)):
            matched.append(topic.topic)
    return matched


def _has_regulatory_cue(text: str, catalog: GuardrailCatalog) -> bool:
    return bool(_occupy(text, NamedAliases(name="cue", aliases=catalog.regulatory_cues)))


def _clarification(
    topics: list[str],
    article: str | None,
    annex: str | None,
    catalog: GuardrailCatalog,
) -> str:
    sentences: list[str] = []
    if article is not None:
        sentences.append(f"Both RoHS and REACH contain Article {article}.")
    if annex is not None:
        sentences.append(f"Annex {annex} cannot be identified until the regulation is named.")
    by_topic = {topic.topic: topic.clarification for topic in catalog.topics}
    for topic in topics:
        clarification = by_topic.get(topic)
        if clarification and clarification not in sentences:
            sentences.append(clarification)
    if not sentences:
        sentences.append("This question could fall under RoHS or REACH.")
    sentences.append("Which regulation should I use?")
    return " ".join(sentences)


def _occupy(text: str, instrument: NamedAliases) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    occupied = [False] * len(text)
    aliases = sorted((alias.lower() for alias in instrument.aliases), key=len, reverse=True)
    for alias in aliases:
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])")
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(occupied[start:end]):
                continue
            for index in range(start, end):
                occupied[index] = True
            spans.append((start, end))
    return spans


def _single_article(text: str) -> str | None:
    numbers = list(dict.fromkeys(_ARTICLE.findall(text)))
    if len(numbers) != 1:
        return None
    return numbers[0]


def _single_annex(text: str) -> str | None:
    raw = [item.upper() for item in _ANNEX.findall(text)]
    normalized = list(dict.fromkeys(_ARABIC_ANNEX.get(item, item) for item in raw))
    if len(normalized) != 1:
        return None
    return normalized[0]
