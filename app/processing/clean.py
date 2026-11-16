"""Clean extracted regulatory text without removing operative provisions."""

from __future__ import annotations

import re

from app.processing.models import CleanedPage, ExtractedPage

_HEADER_RE = re.compile(
    r"^0\d{4}[A-Z]\d+\s*[—―–\-]\s*EN\s*[—―–\-]\s*"
    r"\d{2}\.\d{2}\.\d{4}\s*[—―–\-]\s*\d{3}\.\d{3}"
    r"(?:\s*[—―–\-]\s*\d+)?\s*$"
)
_MARKER_RE = re.compile(r"^[▼►]\s*[A-Z]\d*\s*_{0,20}$")
_SOFT_HYPHEN = "\u00ad"


def clean_pages(pages: list[ExtractedPage]) -> list[CleanedPage]:
    """Normalize each page and remove confident running furniture."""
    return [_clean_page(page) for page in pages]


def clean_line(text: str) -> str:
    """Normalize spaces on one line and drop discretionary hyphens.

    A soft hyphen at the end of a line is kept as a hyphen so the structure
    step can rejoin the split word. Soft hyphens inside a line are removed.
    """
    normalized = text.replace("\u00a0", " ")
    if normalized.rstrip().endswith(_SOFT_HYPHEN):
        trimmed = normalized.rstrip()
        normalized = trimmed[:-1] + "-" + normalized[len(trimmed):]
        normalized = normalized.replace(_SOFT_HYPHEN, "")
    else:
        normalized = normalized.replace(_SOFT_HYPHEN, "")
    return re.sub(r"[ \t]+", " ", normalized).strip()


def is_running_header(text: str) -> bool:
    """Return True for a consolidated-text running header."""
    return _HEADER_RE.match(clean_line(text)) is not None


def is_margin_marker(text: str) -> bool:
    """Return True for a standalone consolidation marker such as ``▼B``."""
    return _MARKER_RE.match(clean_line(text)) is not None


def _clean_page(page: ExtractedPage) -> CleanedPage:
    removed_header = False
    removed_markers = 0
    kept: list[str] = []
    for raw_line in page.text.splitlines():
        line = clean_line(raw_line)
        if not line:
            continue
        if is_running_header(line):
            removed_header = True
            continue
        if is_margin_marker(line):
            removed_markers += 1
            continue
        if line == str(page.page_number):
            continue
        kept.append(line)
    return CleanedPage(
        document_id=page.document_id,
        page_number=page.page_number,
        text="\n".join(kept),
        removed_header=removed_header,
        removed_marker_lines=removed_markers,
    )
