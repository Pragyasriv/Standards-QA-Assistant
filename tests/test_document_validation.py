"""Tests for regulatory document metadata parsing and file validation."""

from datetime import date
from pathlib import Path

import pytest

from app.documents.cellar import (
    parse_consolidation_reference,
    parse_expression_title,
    parse_official_filename,
)
from app.documents.errors import DocumentCollectionError
from app.documents.models import DocumentManifest, DocumentRecord
from app.documents.validation import sha256_file, validate_manifest, validate_pdf_bytes

MINIMAL_PDF = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
ROHS_REFERENCE = "02011L0065 — EN — 01.07.2026 — 023.001"
REACH_REFERENCE = "02006R1907 — EN — 22.06.2026 — 069.001"


def test_parse_consolidation_reference_reads_official_header() -> None:
    reference, version_date, version = parse_consolidation_reference(
        f'<p class="reference">{ROHS_REFERENCE}</p>',
        "02011L0065-20260701",
    )

    assert reference == ROHS_REFERENCE
    assert version_date == date(2026, 7, 1)
    assert version == "023.001"


def test_parse_consolidation_reference_rejects_a_different_celex() -> None:
    with pytest.raises(DocumentCollectionError):
        parse_consolidation_reference(REACH_REFERENCE, "02011L0065-20260701")


def test_parse_expression_title_normalizes_official_whitespace() -> None:
    rdf = (
        '<j.0:expression_title xml:lang="en">'
        "Directive 2011/65/EU of 8\u00a0June 2011</j.0:expression_title>"
    )

    assert parse_expression_title(rdf, "02011L0065-20260701") == (
        "Directive 2011/65/EU of 8 June 2011"
    )


def test_parse_official_filename_requires_one_pdf() -> None:
    listing = (
        "CL2011L0065EN0230010.0001.pdf --> "
        "http://publications.europa.eu/resource/cellar/example/DOC_1"
    )

    assert (
        parse_official_filename(listing, "02011L0065-20260701")
        == "CL2011L0065EN0230010.0001.pdf"
    )


def test_validate_pdf_bytes_accepts_a_small_pdf_and_rejects_other_bytes() -> None:
    assert validate_pdf_bytes(MINIMAL_PDF) == []
    assert validate_pdf_bytes(b"") == ["File is empty."]
    assert validate_pdf_bytes(b"not a pdf%%EOF") == [
        "File does not start with a PDF header."
    ]


def test_manifest_validation_accepts_a_matching_pdf(tmp_path: Path) -> None:
    relative_path = "data/raw/rohs/CL2011L0065EN0230010.0001.pdf"
    pdf_path = tmp_path / relative_path
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(MINIMAL_PDF)
    manifest = DocumentManifest(
        documents=[_record(relative_path, sha256_file(pdf_path), len(MINIMAL_PDF))]
    )

    assert validate_manifest(manifest, tmp_path) == []


def test_manifest_validation_reports_checksum_and_identity_problems(tmp_path: Path) -> None:
    relative_path = "data/raw/rohs/CL2011L0065EN0230010.0001.pdf"
    pdf_path = tmp_path / relative_path
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(MINIMAL_PDF)
    first = _record(relative_path, "a" * 64, len(MINIMAL_PDF))
    duplicate = first.model_copy(update={"sha256": "b" * 64})
    manifest = DocumentManifest(documents=[first, duplicate])

    problems = validate_manifest(manifest, tmp_path)

    assert any("Duplicate document ID" in problem for problem in problems)
    assert any("SHA-256 mismatch" in problem for problem in problems)


def test_manifest_validation_rejects_paths_outside_raw_data(tmp_path: Path) -> None:
    record = _record("data/raw/rohs/../../outside.pdf", "a" * 64, 10)
    manifest = DocumentManifest(documents=[record])

    problems = validate_manifest(manifest, tmp_path)

    assert any("path escapes data/raw" in problem for problem in problems)


def test_manifest_validation_reports_a_missing_file(tmp_path: Path) -> None:
    record = _record(
        "data/raw/reach/CL2006R1907EN0690010.0001.pdf",
        "a" * 64,
        10,
        regulation_name="REACH",
        document_id="02006R1907-20260622",
        celex="02006R1907-20260622",
        filename="CL2006R1907EN0690010.0001.pdf",
        reference=REACH_REFERENCE,
        regulation_number="(EC) No 1907/2006",
    )
    manifest = DocumentManifest(documents=[record])

    problems = validate_manifest(manifest, tmp_path)

    assert any("does not exist" in problem for problem in problems)


def _record(
    relative_path: str,
    digest: str,
    size: int,
    *,
    regulation_name: str = "RoHS",
    document_id: str = "02011L0065-20260701",
    celex: str = "02011L0065-20260701",
    filename: str = "CL2011L0065EN0230010.0001.pdf",
    reference: str = ROHS_REFERENCE,
    regulation_number: str = "2011/65/EU",
) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        regulation_name=regulation_name,
        document_title="Official consolidated text",
        regulation_number=regulation_number,
        source_url="https://eur-lex.europa.eu/eli/dir/2011/65/2026-07-01/eng",
        local_filename=filename,
        version_date=date(2026, 7, 1),
        download_date=date(2026, 11, 13),
        file_format="pdf",
        sha256=digest,
        celex=celex,
        consolidation_reference=reference,
        download_url="https://publications.europa.eu/resource/cellar/example/DOC_1",
        file_size_bytes=size,
        relative_path=relative_path,
    )
