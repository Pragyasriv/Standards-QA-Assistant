"""Run extraction, cleaning, structure detection, and chunking."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.documents.models import DocumentManifest, DocumentRecord
from app.documents.validation import load_manifest, sha256_file
from app.processing.chunking import build_chunks
from app.processing.clean import clean_pages
from app.processing.extract import extract_pdf
from app.processing.models import ChunkRecord, ExtractionReport
from app.processing.settings import ProcessingSettings, get_processing_settings
from app.processing.structure import Block, build_blocks
from app.processing.tokenize import build_tokenizer

logger = logging.getLogger(__name__)


def project_root() -> Path:
    """Return the repository root that contains the app package."""
    return Path(__file__).resolve().parents[2]


def process_manifest(
    root: Path | None = None,
    settings: ProcessingSettings | None = None,
) -> list[ExtractionReport]:
    """Process every document listed in the manifest."""
    repository = root or project_root()
    active = settings or get_processing_settings()
    manifest_path = _resolve(repository, active.manifest_path)
    manifest = load_manifest(manifest_path)
    reports: list[ExtractionReport] = []
    for record in manifest.documents:
        reports.append(process_document(record, repository, active))
    return reports


def process_document(
    record: DocumentRecord,
    root: Path,
    settings: ProcessingSettings,
) -> ExtractionReport:
    """Process one official PDF and write the Phase 3 output files."""
    source = _resolve(root, Path(record.relative_path))
    digest_before = sha256_file(source)
    if digest_before != record.sha256:
        raise ValueError(
            f"{record.document_id} SHA-256 does not match the manifest. "
            "The original PDF was not processed."
        )
    collection = _collection_name(record.relative_path)
    output_dir = _resolve(root, settings.processed_dir) / collection
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(settings.tokenizer)
    pages = extract_pdf(
        source,
        record.document_id,
        ocr_review_char_threshold=settings.ocr_review_char_threshold,
        column_gap_points=settings.column_gap_points,
    )
    cleaned = clean_pages(pages)
    blocks = build_blocks(cleaned)
    chunks = build_chunks(blocks, record, settings, tokenizer)
    report = _build_report(record, pages, cleaned, blocks, chunks)

    _write_jsonl(output_dir / "extracted_pages.jsonl", pages)
    _write_jsonl(output_dir / "cleaned_text.jsonl", cleaned)
    _write_jsonl(output_dir / "chunks.jsonl", chunks)
    (output_dir / "extraction_report.json").write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    digest_after = sha256_file(source)
    if digest_after != digest_before:
        raise ValueError(f"{record.document_id} PDF bytes changed during processing.")
    logger.info(
        "Processed %s pages=%s chunks=%s articles=%s annexes=%s",
        record.regulation_name,
        report.total_pdf_pages,
        report.generated_chunks,
        report.detected_articles,
        report.detected_annexes,
    )
    return report


def load_processed_manifest(path: Path) -> DocumentManifest:
    """Load a manifest from an explicit path."""
    return load_manifest(path)


def _build_report(record, pages, cleaned, blocks: list[Block], chunks: list[ChunkRecord]) -> ExtractionReport:
    failed = [page.page_number for page in pages if page.extraction_failed]
    empty = [page.page_number for page in pages if page.is_empty and not page.extraction_failed]
    ocr_pages = [page.page_number for page in pages if page.ocr_review]
    successful = len(pages) - len(failed)
    article_numbers = list(dict.fromkeys(block.article for block in blocks if block.article))
    annex_labels = list(dict.fromkeys(block.annex for block in blocks if block.annex))
    token_counts = [chunk.token_count for chunk in chunks]
    warnings: list[str] = []
    header_pages = sum(1 for page in cleaned if page.removed_header)
    marker_lines = sum(page.removed_marker_lines for page in cleaned)
    if header_pages:
        warnings.append(f"Removed running headers from {header_pages} pages.")
    if marker_lines:
        warnings.append(f"Removed {marker_lines} standalone consolidation marker lines.")
    stacked = [
        page.page_number
        for page in pages
        if any("Multiple running-header versions" in warning for warning in page.warnings)
    ]
    if stacked:
        warnings.append(
            "Stacked running headers were detected on "
            f"{len(stacked)} pages. Consolidation dates come from the manifest, "
            "not from a PDF header layer."
        )
    if empty:
        warnings.append(f"Empty pages: {empty}.")
    if failed:
        warnings.append(f"Failed pages: {failed}.")
    if ocr_pages:
        warnings.append(f"Pages flagged for OCR review: {ocr_pages}.")
    rotated = [
        page.page_number
        for page in pages
        if any(warning.startswith("Page is rotated") for warning in page.warnings)
    ]
    if rotated:
        warnings.append(
            "Rotated pages were read in display order, so table columns on "
            f"those pages are less reliable: {rotated}."
        )
    unknown = sum(1 for block in blocks if block.kind == "unknown")
    average = round(sum(token_counts) / len(token_counts), 2) if token_counts else 0
    return ExtractionReport(
        document_id=record.document_id,
        regulation=record.regulation_name,
        source_filename=record.local_filename,
        total_pdf_pages=len(pages),
        successfully_extracted_pages=successful,
        empty_pages=len(empty),
        failed_pages=len(failed),
        total_extracted_characters=sum(page.char_count for page in pages),
        detected_articles=len(article_numbers),
        detected_annexes=len(annex_labels),
        generated_chunks=len(chunks),
        average_chunk_size=average,
        maximum_chunk_size=max(token_counts, default=0),
        pages_requiring_ocr_review=ocr_pages,
        warnings=warnings,
        detected_article_numbers=article_numbers,
        detected_annex_labels=annex_labels,
        unknown_block_count=unknown,
    )


def _collection_name(relative_path: str) -> str:
    parts = Path(relative_path).parts
    if len(parts) >= 2:
        return parts[-2]
    return "documents"


def _resolve(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def _write_jsonl(path: Path, records: list) -> None:
    lines = [record.model_dump_json() for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file produced by the pipeline."""
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
