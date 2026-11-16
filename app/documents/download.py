"""Download the official consolidated RoHS and REACH PDFs."""

from __future__ import annotations

import argparse
import logging
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

from app.core.logging_config import configure_logging
from app.documents.cellar import download_pdf, resolve_official_pdf
from app.documents.errors import DocumentCollectionError
from app.documents.models import DocumentManifest, DocumentRecord
from app.documents.sources import MANIFEST_RELATIVE_PATH, OFFICIAL_SOURCES, OfficialSource
from app.documents.validation import sha256_file, validate_pdf_file

logger = logging.getLogger(__name__)


def project_root() -> Path:
    """Return the repository root that contains the app package."""
    return Path(__file__).resolve().parents[2]


def download_official_documents(root: Path, force: bool = False) -> DocumentManifest:
    """Download every configured official PDF and write the manifest."""
    records: list[DocumentRecord] = []
    downloaded_on = datetime.now(timezone.utc).date()
    for source in OFFICIAL_SOURCES:
        records.append(_download_source(root, source, downloaded_on, force))
    manifest = DocumentManifest(documents=records)
    manifest_path = root / MANIFEST_RELATIVE_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote manifest %s", manifest_path)
    return manifest


def _download_source(
    root: Path,
    source: OfficialSource,
    downloaded_on: date,
    force: bool,
) -> DocumentRecord:
    logger.info("Resolving %s CELEX %s", source.regulation_name, source.celex)
    resolved = resolve_official_pdf(source.celex)
    if source.title_token not in resolved.title:
        raise DocumentCollectionError(
            f"CELLAR title for {source.celex} does not contain {source.title_token!r}: "
            f"{resolved.title}"
        )
    relative_path = f"data/raw/{source.collection}/{resolved.official_filename}"
    destination = root / relative_path
    _store_official_pdf(resolved.download_url, destination, force)
    pdf_problems = validate_pdf_file(destination)
    if pdf_problems:
        raise DocumentCollectionError(
            f"{source.celex} failed PDF validation: {'; '.join(pdf_problems)}"
        )
    digest = sha256_file(destination)
    record = DocumentRecord(
        document_id=source.document_id,
        regulation_name=source.regulation_name,
        document_title=resolved.title,
        regulation_number=source.regulation_number,
        source_url=source.source_url,
        local_filename=resolved.official_filename,
        version_date=resolved.version_date,
        download_date=downloaded_on,
        file_format="pdf",
        sha256=digest,
        celex=source.celex,
        consolidation_reference=resolved.consolidation_reference,
        download_url=resolved.download_url,
        file_size_bytes=destination.stat().st_size,
        relative_path=relative_path,
    )
    logger.info(
        "Acquired %s file=%s bytes=%s sha256=%s version=%s",
        source.regulation_name,
        record.local_filename,
        record.file_size_bytes,
        record.sha256,
        record.consolidation_reference,
    )
    return record


def _store_official_pdf(url: str, destination: Path, force: bool) -> None:
    """Save the server bytes. An identical local file is left in place."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=".download-",
        suffix=".pdf",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        download_pdf(url, temporary_path)
        problems = validate_pdf_file(temporary_path)
        if problems:
            raise DocumentCollectionError(
                f"Downloaded file is not a valid PDF: {'; '.join(problems)}"
            )
        if destination.is_file() and sha256_file(destination) == sha256_file(temporary_path):
            temporary_path.unlink()
            logger.info("Existing file matches the official PDF and was left unchanged.")
            return
        if destination.is_file() and not force:
            raise DocumentCollectionError(
                f"{destination.name} already exists and differs from the official download. "
                "Re-run with --force to replace it."
            )
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    logger.info("Saved original PDF %s", destination)


def main(argv: list[str] | None = None) -> int:
    """Download the configured official documents."""
    parser = argparse.ArgumentParser(
        description="Download official consolidated RoHS and REACH PDFs."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing local PDF with a fresh download.",
    )
    args = parser.parse_args(argv)
    configure_logging("INFO")
    try:
        download_official_documents(project_root(), force=args.force)
    except DocumentCollectionError as exc:
        logger.error("%s", exc)
        logger.error(
            "Automatic download failed. Open the EUR-Lex document page, choose "
            "the English PDF, and save that unchanged file under data/raw/. "
            "RoHS: %s REACH: %s",
            OFFICIAL_SOURCES[0].eur_lex_pdf_url,
            OFFICIAL_SOURCES[1].eur_lex_pdf_url,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
