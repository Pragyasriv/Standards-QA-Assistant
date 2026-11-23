"""Decide whether a retrieved chunk actually answers the question.

The rules are structural. They do not mix cosine, BM25, and RRF, and they do
not apply the cosine cutoff to either of the other scores.

A positive BM25 score only shows that some question words occur in the chunk.
For a question that asks what a list contains, an article that merely points at
that list is not the list. In these two instruments the lists are annexes.
Naming an article or an annex in the question still requires that exact
provision, and that provision is then eligible on its own.
"""

from __future__ import annotations

import re

from app.retrieval.models import RetrievedChunk
from app.retrieval.tokenize import tokenize

_INVENTORY = re.compile(
    r"\brestricted substances\b"
    r"|\bwhat substances\b.{0,80}\brestricted\b"
    r"|\bwhich substances\b.{0,80}\brestricted\b"
    r"|\bsubstances\b.{0,40}\bare restricted\b"
    r"|\blist all\b.{0,60}\bsubstances\b"
    r"|\bconcentration limits\b"
    r"|\bwhat are the exemptions\b"
    r"|\bwhich .+ exemptions\b"
    r"|\bcategories are listed\b"
    r"|\bwhich lamps\b"
    r"|\blist of\b",
    re.IGNORECASE,
)
_PROCEDURE = re.compile(
    r"review and amendment|amendment of the list of restricted substances|"
    r"shall be considered by the commission",
    re.IGNORECASE,
)
_PERCENT = re.compile(r"\d+,\d+\s*%|\d+\.\d+\s*%")
_EXPIRES = re.compile(r"\bexpires on\b", re.IGNORECASE)
_ITEM = re.compile(r"\d+\.\s+[A-Z]")
_ANNEX = re.compile(
    r"\bannex\s+(xvii|xvi|xiii|xii|xiv|xv|xi|viii|vii|vi|ix|iv|iii|ii|x|v|i)\b",
    re.IGNORECASE,
)
_BOILERPLATE = frozenset(
    {
        "rohs",
        "reach",
        "directive",
        "regulation",
        "article",
        "annex",
        "eu",
        "ec",
        "2011",
        "65",
        "1907",
        "2006",
        "eee",
    }
)
_VAGUE = frozenset(
    {
        "cover",
        "set",
        "say",
        "says",
        "state",
        "states",
        "place",
        "places",
        "require",
        "requires",
        "mean",
        "means",
        "apply",
        "applies",
        "rules",
        "rule",
    }
)


def asks_for_inventory(question: str) -> bool:
    """Return whether the question asks for the contents of a list."""
    return _INVENTORY.search(question) is not None


def specific_terms(question: str) -> set[str]:
    """Return question words that are not regulation names or vague verbs."""
    return {
        token
        for token in tokenize(question)
        if token not in _BOILERPLATE and token not in _VAGUE
    }


def is_procedural_pointer(chunk: RetrievedChunk) -> bool:
    """Return whether the chunk discusses reviewing a list instead of stating it."""
    if has_list_entries(chunk):
        return False
    return _PROCEDURE.search(chunk.text[:600]) is not None


def has_list_entries(chunk: RetrievedChunk) -> bool:
    """Return whether the chunk states list entries rather than one numbered paragraph.

    A concentration such as ``0,1 %``, an exemption date, or two numbered
    category lines counts. One paragraph that begins ``1. With a view`` does
    not, so a review article is not treated as the list it points to.
    """
    text = chunk.text
    if _PERCENT.search(text) or _EXPIRES.search(text):
        return True
    return len(_ITEM.findall(text)) >= 2


def cited_annex(chunk: RetrievedChunk) -> str | None:
    """Return the single annex named in a chunk, if it names only one."""
    found = list(dict.fromkeys(item.upper() for item in _ANNEX.findall(chunk.text)))
    if len(found) != 1:
        return None
    return found[0]


def chunk_is_sufficient(
    chunk: RetrievedChunk,
    question: str,
    *,
    requested_article: str | None,
    requested_annex: str | None,
) -> bool:
    """Return whether this chunk is enough evidence for the question.

    BM25 must be positive. That is necessary and is not compared with cosine
    or RRF. It is not sufficient for an open list question.
    """
    if chunk.bm25_score is None or chunk.bm25_score <= 0:
        return False
    if requested_article is not None or requested_annex is not None:
        return True
    if not asks_for_inventory(question):
        return True
    if chunk.annex is None or is_procedural_pointer(chunk) or not has_list_entries(chunk):
        return False
    terms = specific_terms(question)
    if not terms:
        return True
    text_terms = set(tokenize(chunk.text))
    return bool(terms & text_terms)
