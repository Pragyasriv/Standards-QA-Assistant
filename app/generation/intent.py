"""Decide whether a question asks for a whole list or for one item.

The patterns are the question wordings this project asks about. They do not
infer a list that the question does not request.
"""

from __future__ import annotations

import re

_COMPLETE_LIST = re.compile(
    r"\bwhat are the\b.{0,80}\b(?:restricted substances|substances|exemptions|exceptions|categories)\b"
    r"|\blist all\b.{0,80}\b(?:substances|exemptions|exceptions|categories)\b"
    r"|\bcomplete list\b"
    r"|\bgive me the (?:complete )?list\b"
    r"|\bwhich substances\b"
    r"|\ball (?:the )?(?:restricted substances|exemptions)\b",
    re.IGNORECASE | re.DOTALL,
)

# Longer names are listed first so a specific name is available to the caller
# in a stable order. Matching itself uses word boundaries.
_ITEMS = (
    "hexavalent chromium",
    "polybrominated biphenyls",
    "polybrominated diphenyl ethers",
    "bis(2-ethylhexyl) phthalate",
    "butyl benzyl phthalate",
    "dibutyl phthalate",
    "diisobutyl phthalate",
    "medical devices",
    "cadmium",
    "mercury",
    "dehp",
    "pbde",
    "dibp",
    "bbp",
    "dbp",
    "pbb",
    "lead",
    "cables",
    "toys",
)


def asks_for_complete_list(question: str) -> bool:
    """Return whether the question asks for every entry of a list."""
    return _COMPLETE_LIST.search(question) is not None


def requested_items(question: str) -> tuple[str, ...]:
    """Return named substances or exceptions the question asks about.

    A complete-list question does not also select one item. A question that
    names no known item is not treated as a specific-item question.
    """
    if asks_for_complete_list(question):
        return ()
    found: list[str] = []
    for name in _ITEMS:
        if re.search(rf"\b{re.escape(name)}\b", question, re.IGNORECASE):
            found.append(name)
    return tuple(found)
