"""Extract page text from regulatory PDFs without modifying the files."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import fitz

from app.processing.models import ExtractedPage

logger = logging.getLogger(__name__)

_HEADER_HINT = re.compile(r"0\d{4}[A-Z]\d+")


def extract_pdf(
    path: Path,
    document_id: str,
    *,
    ocr_review_char_threshold: int,
    column_gap_points: float,
) -> list[ExtractedPage]:
    """Extract each page. The PDF is opened read-only and is not saved."""
    document = fitz.open(path)
    pages: list[ExtractedPage] = []
    try:
        for index, page in enumerate(document):
            page_number = index + 1
            try:
                pages.append(
                    _extract_page(
                        page,
                        document_id,
                        page_number,
                        ocr_review_char_threshold,
                        column_gap_points,
                    )
                )
            except Exception as exc:
                logger.exception("Failed to extract page %s of %s", page_number, path.name)
                pages.append(
                    ExtractedPage(
                        document_id=document_id,
                        page_number=page_number,
                        text="",
                        char_count=0,
                        is_empty=True,
                        extraction_failed=True,
                        ocr_review=True,
                        image_count=0,
                        warnings=[f"Extraction failed: {exc}"],
                    )
                )
    finally:
        document.close()
    return pages


def _extract_page(
    page: fitz.Page,
    document_id: str,
    page_number: int,
    ocr_review_char_threshold: int,
    column_gap_points: float,
) -> ExtractedPage:
    warnings: list[str] = []
    raw_text = page.get_text("text")
    header_versions = _header_versions(raw_text)
    if len(header_versions) > 1:
        warnings.append(
            "Multiple running-header versions are painted on this page: "
            + "; ".join(header_versions)
        )
    if page.rotation:
        text = raw_text
        warnings.append(
            f"Page is rotated {page.rotation} degrees. Text was read in display "
            "order instead of reconstructed from word positions."
        )
    else:
        text = _lines_from_words(page, column_gap_points)
    if not text.strip():
        text = raw_text
    stripped = text.strip()
    char_count = len(stripped)
    image_count = len(page.get_images())
    is_empty = char_count == 0
    ocr_review = is_empty or (image_count > 0 and char_count < ocr_review_char_threshold)
    if ocr_review and image_count > 0 and char_count < ocr_review_char_threshold:
        warnings.append("Page has images and little or no extractable text.")
    elif is_empty:
        warnings.append("Page has no extractable text.")
    return ExtractedPage(
        document_id=document_id,
        page_number=page_number,
        text=text,
        char_count=char_count,
        is_empty=is_empty,
        extraction_failed=False,
        ocr_review=ocr_review,
        image_count=image_count,
        warnings=warnings,
    )


def _lines_from_words(page: fitz.Page, column_gap_points: float) -> str:
    """Rebuild lines in reading order and mark wide column gaps."""
    words = _drop_overlapping_words(page.get_text("words"))
    if not words:
        return ""
    rows: list[list[tuple]] = []
    for word in sorted(words, key=lambda item: (round(item[1], 1), item[0])):
        if not rows or abs(word[1] - rows[-1][0][1]) > 3:
            rows.append([word])
        else:
            rows[-1].append(word)
    lines: list[str] = []
    for row in rows:
        row.sort(key=lambda item: item[0])
        parts: list[str] = []
        previous_x1: float | None = None
        for word in row:
            if previous_x1 is not None and word[0] - previous_x1 >= column_gap_points:
                parts.append("|")
            parts.append(str(word[4]))
            previous_x1 = float(word[2])
        line = " ".join(parts).replace(" | ", " | ").strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _drop_overlapping_words(words: list[tuple]) -> list[tuple]:
    """Keep the first text layer when consolidated PDFs paint several headers."""
    kept: list[tuple] = []
    for word in words:
        if any(_mostly_overlaps(existing[:4], word[:4]) for existing in kept):
            continue
        kept.append(word)
    return kept


def _mostly_overlaps(first: tuple, second: tuple) -> bool:
    overlap_width = min(first[2], second[2]) - max(first[0], second[0])
    overlap_height = min(first[3], second[3]) - max(first[1], second[1])
    if overlap_width <= 0 or overlap_height <= 0:
        return False
    second_area = max((second[2] - second[0]) * (second[3] - second[1]), 0.01)
    return (overlap_width * overlap_height) / second_area > 0.5


def _header_versions(raw_text: str) -> list[str]:
    versions: list[str] = []
    for line in raw_text.splitlines():
        compact = " ".join(line.split())
        if _HEADER_HINT.search(compact) and "EN" in compact:
            versions.append(compact)
    return list(dict.fromkeys(versions))
