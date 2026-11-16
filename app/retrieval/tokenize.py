"""Tokenize regulatory text for BM25.

Tokens are lowercase letters and numbers. A fixed English function-word list
is removed. No stemmer is applied, and the list is not derived from the
evaluation questions.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """
    a an the of to and or for in on at by from with without into onto over under
    as is are was were be been being this that these those it its itself
    what which who whom whose when where why how
    do does did doing done can could should would may might must
    not no nor if then than so such
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Return the BM25 tokens for a chunk or a question."""
    tokens: list[str] = []
    for token in _TOKEN.findall(text.lower()):
        if token in _STOPWORDS:
            continue
        if token.isalpha() and len(token) == 1:
            continue
        tokens.append(token)
    return tokens
