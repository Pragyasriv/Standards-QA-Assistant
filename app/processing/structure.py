"""Identify legal hierarchy in cleaned RoHS and REACH text.

Heading decisions are conservative. Contents-list entries are kept as text
and do not create article or annex metadata. Unrecognized provisions stay
in the current section with null fields rather than a guessed citation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.processing.models import CleanedPage

_ARTICLE_RE = re.compile(r"^Article\s+(\d+[a-zA-Z]?)$")
_TITLE_RE = re.compile(r"^TITLE\s+([IVXLCDM]+)$")
_CHAPTER_RE = re.compile(r"^CHAPTER\s+(\d+[a-zA-Z]?)$")
_ANNEX_RE = re.compile(r"^ANNEX\s+([IVXLCDM]+)$")
_PARA_RE = re.compile(r"^(\d+)\.\s*(.*)$")
_POINT_RE = re.compile(r"^\(([a-z])\)\s*(.*)$")
_ROMAN_RE = re.compile(r"^\((i{1,3}|iv|vi{0,3}|ix|x|xi|xii)\)\s*(.*)$")
_ANNEX_ITEM_RE = re.compile(
    r"^(\d{1,3}(?:\([a-z]\))?)\s+\|\s*(\S.*)$"
    r"|^(\d{1,3}\([a-z]\))\s+(\S.*)$"
    r"|^(\d{1,3})\.\s+(\S.*)$"
)
_SENTENCE_START = re.compile(
    r"^(This|The|A|An|It|Where|If|In|For|When|Such|Any|No|Member|These|Those)\b"
)


@dataclass
class Line:
    """A cleaned line tied to its PDF page."""

    page: int
    text: str


@dataclass
class Block:
    """A contiguous provision before chunking."""

    kind: str
    lines: list[str] = field(default_factory=list)
    page_start: int = 1
    page_end: int = 1
    title: str | None = None
    chapter: str | None = None
    article: str | None = None
    article_heading: str | None = None
    paragraph: str | None = None
    subparagraph: str | None = None
    annex: str | None = None
    annex_section: str | None = None

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


@dataclass
class _State:
    title: str | None = None
    chapter: str | None = None
    article: str | None = None
    article_heading: str | None = None
    paragraph: str | None = None
    letter: str | None = None
    roman: str | None = None
    annex: str | None = None
    annex_section: str | None = None
    kind: str = "preamble"

    def subparagraph(self) -> str | None:
        if self.letter and self.roman:
            return f"({self.letter})({self.roman})"
        if self.letter:
            return f"({self.letter})"
        if self.roman:
            return f"({self.roman})"
        return None


def build_blocks(pages: list[CleanedPage]) -> list[Block]:
    """Turn cleaned pages into structural blocks without dropping lines."""
    lines = _iter_lines(pages)
    contents = _contents_indexes(lines)
    state = _State()
    blocks: list[Block] = []
    current: Block | None = None

    def flush() -> None:
        nonlocal current
        if current and current.text:
            blocks.append(current)
        current = None

    def start(kind: str, page: int) -> Block:
        nonlocal current
        flush()
        current = Block(
            kind=kind,
            page_start=page,
            page_end=page,
            title=state.title,
            chapter=state.chapter,
            article=state.article,
            article_heading=state.article_heading,
            paragraph=state.paragraph,
            subparagraph=state.subparagraph(),
            annex=state.annex,
            annex_section=state.annex_section,
        )
        return current

    expecting_heading_for: str | None = None
    for index, line in enumerate(lines):
        if index in contents:
            state.kind = "unknown"
            if current is None or current.kind != "unknown" or current.annex or current.article:
                start("unknown", line.page)
            assert current is not None
            _append(current, line.text, line.page)
            expecting_heading_for = None
            continue

        article = _ARTICLE_RE.match(line.text)
        title = _TITLE_RE.match(line.text)
        chapter = _CHAPTER_RE.match(line.text)
        annex = _ANNEX_RE.match(line.text)
        if state.annex is None and article:
            state.article = article.group(1)
            state.article_heading = None
            state.paragraph = None
            state.letter = None
            state.roman = None
            state.kind = "article"
            start("article", line.page)
            assert current is not None
            _append(current, line.text, line.page)
            expecting_heading_for = "article"
            continue
        if state.annex is None and title:
            state.title = title.group(1)
            state.chapter = None
            state.article = None
            state.article_heading = None
            state.paragraph = None
            state.letter = None
            state.roman = None
            state.kind = "title"
            start("title", line.page)
            assert current is not None
            _append(current, line.text, line.page)
            expecting_heading_for = "title"
            continue
        if state.annex is None and chapter:
            state.chapter = chapter.group(1)
            state.article = None
            state.article_heading = None
            state.paragraph = None
            state.letter = None
            state.roman = None
            state.kind = "chapter"
            start("chapter", line.page)
            assert current is not None
            _append(current, line.text, line.page)
            expecting_heading_for = "chapter"
            continue
        if annex:
            state.annex = annex.group(1)
            state.annex_section = None
            state.title = None
            state.chapter = None
            state.article = None
            state.article_heading = None
            state.paragraph = None
            state.letter = None
            state.roman = None
            state.kind = "annex"
            start("annex", line.page)
            assert current is not None
            _append(current, line.text, line.page)
            expecting_heading_for = "annex"
            continue

        if expecting_heading_for and _looks_like_heading(line.text):
            assert current is not None
            _append(current, line.text, line.page)
            if expecting_heading_for == "article":
                state.article_heading = line.text
                current.article_heading = line.text
            elif expecting_heading_for == "annex":
                state.annex_section = line.text
                current.annex_section = line.text
            expecting_heading_for = None
            continue
        expecting_heading_for = None

        if state.annex is None:
            paragraph = _PARA_RE.match(line.text)
            roman = _ROMAN_RE.match(line.text)
            point = None if roman and state.letter else _POINT_RE.match(line.text)
            if roman and state.letter:
                point = None
            if paragraph and not point and not roman:
                state.paragraph = paragraph.group(1)
                state.letter = None
                state.roman = None
                state.kind = "paragraph"
                start("paragraph", line.page)
                assert current is not None
                body = paragraph.group(2).strip()
                _append(current, f"{state.paragraph}. {body}".rstrip(), line.page)
                continue
            if point:
                state.letter = point.group(1)
                state.roman = None
                state.kind = "point"
                start("point", line.page)
                assert current is not None
                body = point.group(2).strip()
                _append(current, f"({state.letter}) {body}".rstrip(), line.page)
                continue
            if roman:
                state.roman = roman.group(1)
                state.kind = "subparagraph"
                start("subparagraph", line.page)
                assert current is not None
                body = roman.group(2).strip()
                _append(current, f"({state.roman}) {body}".rstrip(), line.page)
                continue
        else:
            item_label = _annex_item_label(line.text)
            bare_item = re.fullmatch(r"(\d{1,3})\.", line.text)
            if bare_item:
                item_label = bare_item.group(1)
            if item_label:
                state.paragraph = item_label
                state.letter = None
                state.roman = None
                state.kind = "annex_section"
                start("annex_section", line.page)
                assert current is not None
                _append(current, line.text, line.page)
                continue

        if current is None:
            start(state.kind, line.page)
        assert current is not None
        _append(current, line.text, line.page)

    flush()
    return _merge_headings(blocks)


def _iter_lines(pages: list[CleanedPage]) -> list[Line]:
    lines: list[Line] = []
    for page in pages:
        for raw in page.text.splitlines():
            text = raw.strip()
            if text:
                lines.append(Line(page=page.page_number, text=text))
    return lines


_CONTENTS_ENTRY = re.compile(
    r"^(?:TITLE\s+[IVXLCDM]+|CHAPTER\s+\d+[a-zA-Z]?|Chapter\s+\d+[a-zA-Z]?|"
    r"Article\s+\d+[a-zA-Z]?|ANNEX\s+[IVXLCDM]+)\s+\|"
)


def _contents_indexes(lines: list[Line]) -> set[int]:
    """Mark headings that belong to a printed table of contents.

    REACH lays contents entries out as ``TITLE I | GENERAL ISSUES``. The
    operative heading is the same label on a line of its own. RoHS has no
    contents list, so article headings are never guessed from spacing.
    """
    start = next(
        (index for index, line in enumerate(lines) if line.text.upper() == "TABLE OF CONTENTS"),
        None,
    )
    if start is None:
        return set()
    indexes = {start}
    for index in range(start + 1, len(lines)):
        text = lines[index].text
        if _is_heading(text):
            break
        if _CONTENTS_ENTRY.match(text):
            indexes.add(index)
    return indexes


def _is_heading(text: str) -> bool:
    return any(
        pattern.match(text)
        for pattern in (_ARTICLE_RE, _TITLE_RE, _CHAPTER_RE, _ANNEX_RE)
    )


def _looks_like_heading(text: str) -> bool:
    if _is_heading(text) or _PARA_RE.match(text) or _POINT_RE.match(text):
        return False
    if _SENTENCE_START.match(text):
        return False
    if len(text) > 120 or text.endswith((".", ";", ",")):
        return False
    return len(text.split()) <= 12


def _annex_item_label(text: str) -> str | None:
    """Return an annex row or paragraph label when the line clearly starts one."""
    if len(text) > 220:
        return None
    match = _ANNEX_ITEM_RE.match(text)
    if match is None:
        return None
    return next(group for group in match.groups() if group)


def _append(block: Block, text: str, page: int) -> None:
    if block.lines and _can_join(block.lines[-1], text):
        block.lines[-1] = _join(block.lines[-1], text)
    else:
        block.lines.append(text)
    block.page_end = page


def _can_join(previous: str, following: str) -> bool:
    if _is_heading(previous) or _is_heading(following):
        return False
    if _looks_like_heading(previous):
        return False
    if _PARA_RE.match(following) or _POINT_RE.match(following) or _ROMAN_RE.match(following):
        return False
    if re.fullmatch(r"\d+\.", previous):
        return True
    if previous.endswith((".", ";", ":")) and following[:1].isupper():
        return False
    return True


def _join(previous: str, following: str) -> str:
    if previous.endswith("-") and following[:1].islower():
        stem = previous[:-1]
        # "manufac-turers" is a line-break split. "m-xylene" is a real hyphen.
        if len(stem) >= 2 and stem[-1].isalpha() and stem[-2].isalpha():
            return stem + following
        return previous + following
    if previous.endswith(_SOFT_HYPHEN):
        return previous[:-1] + following
    return f"{previous} {following}"


_SOFT_HYPHEN = "\u00ad"


def _merge_headings(blocks: list[Block]) -> list[Block]:
    """Keep an article or annex heading with the text that follows it."""
    merged: list[Block] = []
    index = 0
    while index < len(blocks):
        block = blocks[index]
        if (
            index + 1 < len(blocks)
            and _heading_only(block)
            and _same_parent(block, blocks[index + 1])
        ):
            follower = blocks[index + 1]
            follower.lines = block.lines + follower.lines
            follower.page_start = block.page_start
            if block.article_heading and not follower.article_heading:
                follower.article_heading = block.article_heading
            if block.annex_section and not follower.annex_section:
                follower.annex_section = block.annex_section
            index += 1
            continue
        merged.append(block)
        index += 1
    return merged


def _heading_only(block: Block) -> bool:
    if block.kind not in {"article", "title", "chapter", "annex"}:
        return False
    word_count = len(block.text.split())
    return word_count <= 30 and block.paragraph is None


def _same_parent(heading: Block, follower: Block) -> bool:
    return (
        heading.article == follower.article
        and heading.annex == follower.annex
        and heading.title == follower.title
        and heading.chapter == follower.chapter
    )
