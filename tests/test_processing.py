"""Tests for PDF extraction, cleaning, structure detection, and chunking."""

from datetime import date
from pathlib import Path

import fitz
import pytest

from app.documents.models import DocumentRecord
from app.processing.chunking import build_chunks
from app.processing.clean import clean_pages, is_margin_marker, is_running_header
from app.processing.extract import extract_pdf
from app.processing.models import CleanedPage, ExtractedPage
from app.processing.pipeline import process_document
from app.processing.settings import ProcessingSettings
from app.processing.structure import build_blocks
from app.processing.tokenize import WordTokenizer, build_tokenizer

SAMPLE = """\
TABLE OF CONTENTS
Article 1 | Aim and scope
Article 2 | Definitions
TITLE I | GENERAL ISSUES
ANNEX I | Categories
ANNEX II | Substances
TITLE I
GENERAL ISSUES
Article 1
Subject matter
This Directive lays down rules on the restriction
of hazardous substances.
1.
This Directive shall apply to equipment.
(a) equipment used for security;
(b) equipment designed to be sent into space.
Article 2
Scope
2.
The second article continues here.
ANNEX I
Categories
ANNEX II
Substances
ANNEX I
Categories of EEE covered by this Directive
1 | Large household appliances
2 | Small household appliances
3 | IT and telecommunications equipment
4 | Consumer equipment
5 | Lighting equipment
ANNEX II
Restricted substances
Lead (0,1 %)
Mercury (0,1 %)
Cadmium (0,01 %)
"""


def test_running_header_and_marker_detection() -> None:
    assert is_running_header("02011L0065 — EN — 01.07.2026 — 023.001 — 5")
    assert is_running_header("02006R1907 ― EN ― 22.06.2026 ― 069.001 ― 7")
    assert not is_running_header("Article 1")
    assert is_margin_marker("▼B")
    assert is_margin_marker("▼M37")
    assert not is_margin_marker("►M3 mixtures ◄")


def test_cleaning_removes_headers_and_keeps_operative_text() -> None:
    page = ExtractedPage(
        document_id="02011L0065-20260701",
        page_number=5,
        text="02011L0065 — EN — 01.07.2026 — 023.001 — 5\n▼B\nArticle 1\nSubject matter",
        char_count=10,
        is_empty=False,
        extraction_failed=False,
        ocr_review=False,
        image_count=0,
        warnings=[],
    )

    cleaned = clean_pages([page])[0]

    assert cleaned.removed_header is True
    assert cleaned.removed_marker_lines == 1
    assert "Article 1" in cleaned.text
    assert "02011L0065" not in cleaned.text
    assert "▼B" not in cleaned.text


def test_single_letter_hyphen_is_not_swallowed() -> None:
    pages = [
        CleanedPage(
            document_id="doc",
            page_number=1,
            text="1.\n5-tert-butyl-2,4,6-trinitro-m-\nxylene",
        )
    ]

    blocks = build_blocks(pages)

    assert "trinitro-m-xylene" in blocks[0].text


def test_line_end_soft_hyphen_is_rejoined() -> None:
    page = ExtractedPage(
        document_id="doc",
        page_number=1,
        text="It is for manufac\u00ad\nturers to comply.",
        char_count=10,
        is_empty=False,
        extraction_failed=False,
        ocr_review=False,
        image_count=0,
        warnings=[],
    )

    cleaned = clean_pages([page])
    blocks = build_blocks(cleaned)

    assert "manufacturers" in blocks[0].text


def test_word_tail_keeps_the_original_characters() -> None:
    assert WordTokenizer().tail("equipment for professional use;", 3) == "professional use;"


def test_annex_entry_number_on_its_own_line_is_kept() -> None:
    pages = [
        CleanedPage(
            document_id="doc",
            page_number=216,
            text="ANNEX XIV\nLIST OF SUBSTANCES\n1.\nMusk xylene\n2.\nMDA\n",
        )
    ]

    blocks = build_blocks(pages)

    assert any(block.paragraph == "1" and "Musk xylene" in block.text for block in blocks)
    assert any(block.paragraph == "2" and "MDA" in block.text for block in blocks)


def test_article_paragraph_and_annex_detection_skips_contents() -> None:
    pages = [
        CleanedPage(
            document_id="doc",
            page_number=5,
            text=SAMPLE,
            removed_header=False,
        )
    ]

    blocks = build_blocks(pages)
    articles = [block.article for block in blocks if block.article]
    annexes = [block.annex for block in blocks if block.annex]

    assert articles[0] == "1"
    assert "Subject matter" in blocks[0].text or any(
        "Subject matter" in block.text for block in blocks
    )
    assert any(block.paragraph == "1" and block.article == "1" for block in blocks)
    assert any(block.subparagraph == "(a)" and "security" in block.text for block in blocks)
    assert any(block.subparagraph == "(b)" for block in blocks)
    assert articles.count("1") >= 1
    assert annexes[0] == "I"
    assert "II" in annexes
    assert any(
        block.annex == "I" and "Large household appliances" in block.text for block in blocks
    )
    assert any(block.kind == "unknown" and "Definitions" in block.text for block in blocks)


def test_unknown_contents_text_is_not_given_an_article_number() -> None:
    pages = [
        CleanedPage(
            document_id="doc",
            page_number=1,
            text=(
                "TABLE OF CONTENTS\n"
                "TITLE I | GENERAL ISSUES\n"
                "Article 1 | Aim\nArticle 2 | Scope\n"
                "TITLE I\nGENERAL ISSUES\n"
                "Article 1\nSubject matter\n"
                "1. This Directive applies.\n"
            ),
            removed_header=False,
        )
    ]

    blocks = build_blocks(pages)
    contents = [block for block in blocks if block.kind == "unknown"]
    operative = [block for block in blocks if block.article == "1"]

    assert contents
    assert "Aim" in "\n".join(block.text for block in contents)
    assert "Article 1 | Aim" in "\n".join(block.text for block in contents)
    assert all(block.article is None for block in contents)
    assert operative
    assert any("This Directive applies" in block.text for block in operative)


def test_chunk_size_metadata_and_deterministic_ids() -> None:
    pages = [
        CleanedPage(
            document_id="02011L0065-20260701",
            page_number=2,
            text=(
                "Article 4\nPrevention\n"
                + " ".join(f"word{index}" for index in range(300))
                + "\n2.\nA second paragraph keeps its own metadata."
            ),
            removed_header=False,
        )
    ]
    blocks = build_blocks(pages)
    record = _record()
    settings = ProcessingSettings(
        chunk_target_tokens=40,
        chunk_max_tokens=60,
        chunk_overlap_tokens=5,
        tokenizer="word",
    )
    tokenizer = build_tokenizer("word")

    first = build_chunks(blocks, record, settings, tokenizer)
    second = build_chunks(blocks, record, settings, tokenizer)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].chunk_id == "02011L0065-20260701:00001"
    assert all(chunk.token_count <= settings.chunk_max_tokens for chunk in first)
    assert all(chunk.article == "4" for chunk in first)
    assert any(chunk.paragraph == "2" for chunk in first)
    assert first[0].regulation == "RoHS"
    assert first[0].consolidation_date == date(2026, 7, 1)
    assert first[0].page_start == 2
    assert _covers(blocks, first)


def test_missing_structure_stays_null() -> None:
    pages = [
        CleanedPage(
            document_id="doc",
            page_number=1,
            text="Recitals that do not identify an article.",
            removed_header=False,
        )
    ]
    blocks = build_blocks(pages)
    chunks = build_chunks(
        blocks,
        _record(),
        ProcessingSettings(chunk_target_tokens=100, chunk_max_tokens=120, chunk_overlap_tokens=10),
        WordTokenizer(),
    )

    assert chunks[0].article is None
    assert chunks[0].annex is None
    assert chunks[0].paragraph is None
    assert chunks[0].chapter is None
    assert chunks[0].title is None


def test_extraction_preserves_page_numbers_and_empty_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    document = fitz.open()
    first = document.new_page()
    first.insert_text((72, 72), "Article 1")
    first.insert_text((72, 96), "Subject matter")
    first.insert_text((72, 120), "1. Equipment shall comply with the limit.")
    document.new_page()
    document.save(pdf_path)
    document.close()

    pages = extract_pdf(
        pdf_path,
        "sample",
        ocr_review_char_threshold=40,
        column_gap_points=24,
    )

    assert [page.page_number for page in pages] == [1, 2]
    assert "Article 1" in pages[0].text
    assert pages[0].is_empty is False
    assert pages[1].is_empty is True
    assert pages[1].ocr_review is True
    assert pages[1].text.strip() == ""


def test_rotated_page_uses_display_order(tmp_path: Path) -> None:
    pdf_path = tmp_path / "rotated.pdf"
    document = fitz.open()
    page = document.new_page(width=842, height=595)
    page.insert_text((72, 72), "ANNEX XIV")
    page.set_rotation(90)
    document.save(pdf_path)
    document.close()

    pages = extract_pdf(
        pdf_path,
        "rotated",
        ocr_review_char_threshold=40,
        column_gap_points=24,
    )

    assert "ANNEX XIV" in pages[0].text
    assert any("rotated 90" in warning for warning in pages[0].warnings)


def test_process_document_leaves_the_pdf_unchanged(tmp_path: Path) -> None:
    pdf_path = tmp_path / "data" / "raw" / "rohs" / "sample.pdf"
    pdf_path.parent.mkdir(parents=True)
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Article 3")
    page.insert_text((72, 96), "Definitions")
    page.insert_text((72, 120), "1. A definition used for the extraction test.")
    document.save(pdf_path)
    document.close()
    before = pdf_path.read_bytes()
    record = _record()
    record = record.model_copy(
        update={
            "relative_path": "data/raw/rohs/sample.pdf",
            "local_filename": "sample.pdf",
            "sha256": __import__("hashlib").sha256(before).hexdigest(),
            "file_size_bytes": len(before),
        }
    )
    settings = ProcessingSettings(
        manifest_path=Path("missing.json"),
        processed_dir=tmp_path / "data" / "processed",
        chunk_target_tokens=50,
        chunk_max_tokens=80,
        chunk_overlap_tokens=5,
    )

    report = process_document(record, tmp_path, settings)

    assert pdf_path.read_bytes() == before
    assert report.total_pdf_pages == 1
    assert report.empty_pages == 0
    assert report.detected_articles == 1
    output = tmp_path / "data" / "processed" / "rohs"
    assert (output / "extracted_pages.jsonl").is_file()
    assert (output / "cleaned_text.jsonl").is_file()
    assert (output / "chunks.jsonl").is_file()
    assert (output / "extraction_report.json").is_file()


@pytest.mark.skipif(
    not Path("data/manifest/documents.json").is_file(),
    reason="Official PDFs are stored locally and are not in Git.",
)
def test_real_rohs_and_reach_documents_are_processed(tmp_path: Path) -> None:
    from app.documents.validation import load_manifest
    from app.processing.pipeline import read_jsonl

    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(root / "data" / "manifest" / "documents.json")
    settings = ProcessingSettings(
        processed_dir=tmp_path / "processed",
        chunk_target_tokens=1000,
        chunk_max_tokens=1200,
        chunk_overlap_tokens=150,
        tokenizer="word",
    )
    reports = [process_document(record, root, settings) for record in manifest.documents]

    by_name = {report.regulation: report for report in reports}
    assert by_name["RoHS"].total_pdf_pages == 46
    assert by_name["REACH"].total_pdf_pages == 603
    assert by_name["RoHS"].failed_pages == 0
    assert by_name["REACH"].failed_pages == 0
    assert by_name["RoHS"].empty_pages == 0
    assert by_name["REACH"].empty_pages == 0
    assert by_name["RoHS"].pages_requiring_ocr_review == []
    assert by_name["REACH"].pages_requiring_ocr_review == []
    assert by_name["RoHS"].detected_articles >= 20
    assert by_name["REACH"].detected_articles >= 100
    assert by_name["RoHS"].detected_annex_labels[:2] == ["I", "II"]
    assert by_name["REACH"].detected_annex_labels == [
        "I",
        "II",
        "III",
        "IV",
        "V",
        "VI",
        "VII",
        "VIII",
        "IX",
        "X",
        "XI",
        "XII",
        "XIII",
        "XIV",
        "XV",
        "XVI",
        "XVII",
    ]
    for record, report in zip(manifest.documents, reports):
        chunks = read_jsonl(tmp_path / "processed" / _collection(record.relative_path) / "chunks.jsonl")
        assert chunks
        assert all(chunk["token_count"] <= 1200 for chunk in chunks)
        assert [chunk["chunk_id"] for chunk in chunks] == [
            f"{record.document_id}:{index:05d}" for index in range(1, len(chunks) + 1)
        ]
        assert report.generated_chunks == len(chunks)
        source = root / record.relative_path
        assert __import__("hashlib").sha256(source.read_bytes()).hexdigest() == record.sha256


def _covers(blocks, chunks) -> bool:
    expected = []
    for block in blocks:
        expected.extend(_tokens(block.text))
    observed = []
    for chunk in chunks:
        observed.extend(_tokens(chunk.text))
    position = 0
    for token in expected:
        while position < len(observed) and observed[position] != token:
            position += 1
        if position >= len(observed):
            return False
        position += 1
    return True


def _tokens(text: str) -> list[str]:
    import re

    return re.findall(r"\w+|[^\w\s]", text)


def _collection(relative_path: str) -> str:
    return Path(relative_path).parts[-2]


def _record() -> DocumentRecord:
    return DocumentRecord(
        document_id="02011L0065-20260701",
        regulation_name="RoHS",
        document_title="Directive 2011/65/EU",
        regulation_number="2011/65/EU",
        source_url="https://eur-lex.europa.eu/eli/dir/2011/65/2026-07-01/eng",
        local_filename="sample.pdf",
        version_date=date(2026, 7, 1),
        download_date=date(2026, 11, 13),
        file_format="pdf",
        sha256="a" * 64,
        celex="02011L0065-20260701",
        consolidation_reference="02011L0065 — EN — 01.07.2026 — 023.001",
        download_url="https://publications.europa.eu/resource/cellar/example/DOC_1",
        file_size_bytes=10,
        relative_path="data/raw/rohs/sample.pdf",
    )
