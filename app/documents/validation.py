"""Checks for the local regulatory document manifest and PDF files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from app.documents.models import DocumentManifest, DocumentRecord

PDF_HEADER = b"%PDF-"
PDF_EOF = b"%%EOF"


def load_manifest(path: Path) -> DocumentManifest:
    """Parse a manifest file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return DocumentManifest.model_validate(payload)


def validate_manifest_file(manifest_path: Path, project_root: Path) -> list[str]:
    """Return every integrity problem found for a manifest and its files."""
    if not manifest_path.is_file():
        return [f"Manifest does not exist: {manifest_path}"]
    try:
        manifest = load_manifest(manifest_path)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        return [f"Manifest is invalid: {exc}"]
    return validate_manifest(manifest, project_root)


def validate_manifest(manifest: DocumentManifest, project_root: Path) -> list[str]:
    """Check identifiers, file presence, PDF structure, and checksums."""
    problems: list[str] = []
    seen_ids: set[str] = set()
    seen_filenames: set[str] = set()
    for record in manifest.documents:
        if record.document_id in seen_ids:
            problems.append(f"Duplicate document ID: {record.document_id}")
        seen_ids.add(record.document_id)
        if record.local_filename in seen_filenames:
            problems.append(f"Duplicate local filename: {record.local_filename}")
        seen_filenames.add(record.local_filename)
        problems.extend(_validate_record(record, project_root))
    return problems


def validate_pdf_file(path: Path) -> list[str]:
    """Check that a file exists, is non-empty, and has PDF header and EOF markers."""
    if not path.is_file():
        return ["File does not exist."]
    file_size = path.stat().st_size
    if file_size == 0:
        return ["File is empty."]
    with path.open("rb") as handle:
        header = handle.read(len(PDF_HEADER))
        handle.seek(max(0, file_size - 2048))
        tail = handle.read()
    problems: list[str] = []
    if header != PDF_HEADER:
        problems.append("File does not start with a PDF header.")
    if PDF_EOF not in tail:
        problems.append("File does not end with a PDF EOF marker.")
    return problems


def validate_pdf_bytes(payload: bytes) -> list[str]:
    """Check that a byte string looks like a non-empty PDF."""
    if not payload:
        return ["File is empty."]
    problems: list[str] = []
    if not payload.startswith(PDF_HEADER):
        problems.append("File does not start with a PDF header.")
    if PDF_EOF not in payload[-2048:]:
        problems.append("File does not end with a PDF EOF marker.")
    return problems


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(256 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _validate_record(record: DocumentRecord, project_root: Path) -> list[str]:
    problems: list[str] = []
    if "/" in record.local_filename or "\\" in record.local_filename:
        problems.append(f"{record.document_id}: local filename must be a single path segment.")
        return problems
    if not record.local_filename.lower().endswith(".pdf"):
        problems.append(f"{record.document_id}: local filename must end with .pdf.")
    expected_prefix = f"data/raw/{_collection_name(record)}/"
    if not record.relative_path.startswith(expected_prefix):
        problems.append(
            f"{record.document_id}: relative path must stay under {expected_prefix}."
        )
    if Path(record.relative_path).name != record.local_filename:
        problems.append(f"{record.document_id}: relative path does not end with the local filename.")
    if record.celex != record.document_id:
        problems.append(f"{record.document_id}: document ID and CELEX differ.")
    base_celex = record.celex.split("-", maxsplit=1)[0]
    version_label = record.version_date.strftime("%d.%m.%Y")
    if base_celex not in record.consolidation_reference:
        problems.append(f"{record.document_id}: consolidation reference does not contain its CELEX.")
    if version_label not in record.consolidation_reference:
        problems.append(
            f"{record.document_id}: consolidation reference does not contain {version_label}."
        )

    file_path = _safe_path(project_root, record.relative_path)
    if file_path is None:
        problems.append(f"{record.document_id}: path escapes data/raw.")
        return problems
    if not file_path.is_file():
        problems.append(f"{record.document_id}: file does not exist: {record.relative_path}")
        return problems

    file_size = file_path.stat().st_size
    if file_size == 0:
        problems.append(f"{record.document_id}: file is empty.")
        return problems
    if file_size != record.file_size_bytes:
        problems.append(
            f"{record.document_id}: size is {file_size} bytes, manifest says {record.file_size_bytes}."
        )
    with file_path.open("rb") as handle:
        header = handle.read(len(PDF_HEADER))
        handle.seek(max(0, file_size - 2048))
        tail = handle.read()
    if header != PDF_HEADER:
        problems.append(f"{record.document_id}: file does not start with a PDF header.")
    if PDF_EOF not in tail:
        problems.append(f"{record.document_id}: file does not end with a PDF EOF marker.")
    digest = sha256_file(file_path)
    if digest != record.sha256:
        problems.append(f"{record.document_id}: SHA-256 mismatch.")
    return problems


def _collection_name(record: DocumentRecord) -> str:
    if record.regulation_name == "RoHS":
        return "rohs"
    if record.regulation_name == "REACH":
        return "reach"
    return "unknown"


def _safe_path(project_root: Path, relative_path: str) -> Path | None:
    raw_root = (project_root / "data" / "raw").resolve()
    candidate = (project_root / relative_path).resolve()
    if not candidate.is_relative_to(raw_root):
        return None
    return candidate
