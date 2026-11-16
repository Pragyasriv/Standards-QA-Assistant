"""Local token counters used to size chunks.

These are not language-model calls. ``word`` counts words and punctuation.
``char4`` estimates model tokens as one token per four characters.
"""

from __future__ import annotations

import re
from typing import Protocol

from app.processing.settings import TokenizerName

_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class Tokenizer(Protocol):
    """Count tokens and return a trailing slice of text."""

    name: str

    def count(self, text: str) -> int:
        """Return the token count of text."""

    def tail(self, text: str, token_count: int) -> str:
        """Return the trailing token_count tokens of text."""


class WordTokenizer:
    """Split text into words and punctuation marks."""

    name = "word"

    def count(self, text: str) -> int:
        if not text.strip():
            return 0
        return len(_WORD.findall(text))

    def tail(self, text: str, token_count: int) -> str:
        if token_count <= 0 or not text.strip():
            return ""
        matches = list(_WORD.finditer(text))
        if not matches:
            return ""
        if token_count >= len(matches):
            start = matches[0].start()
        else:
            start = matches[-token_count].start()
        return text[start:].strip()


class Char4Tokenizer:
    """Estimate tokens as one per four characters."""

    name = "char4"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return (len(text) + 3) // 4

    def tail(self, text: str, token_count: int) -> str:
        if token_count <= 0 or not text:
            return ""
        return text[-(token_count * 4) :]


def build_tokenizer(name: TokenizerName) -> Tokenizer:
    """Return the configured tokenizer."""
    if name == "char4":
        return Char4Tokenizer()
    return WordTokenizer()
