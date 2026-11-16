"""Records written by the document-processing pipeline."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class ExtractedPage(BaseModel):
    """One PDF page as extracted, before cleaning."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    page_number: int = Field(ge=1)
    text: str
    char_count: int = Field(ge=0)
    is_empty: bool
    extraction_failed: bool
    ocr_review: bool
    image_count: int = Field(ge=0)
    warnings: list[str]


class CleanedPage(BaseModel):
    """One page after header, footer, and whitespace cleaning."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    page_number: int = Field(ge=1)
    text: str
    removed_header: bool = False
    removed_marker_lines: int = Field(default=0, ge=0)


class ChunkRecord(BaseModel):
    """A structure-aware chunk ready for a later retrieval phase."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    regulation: str
    document_title: str
    regulation_number: str
    consolidation_date: date
    chapter: str | None = None
    title: str | None = None
    article: str | None = None
    paragraph: str | None = None
    subparagraph: str | None = None
    annex: str | None = None
    annex_section: str | None = None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    source_filename: str
    text: str
    token_count: int = Field(ge=0)


class ExtractionReport(BaseModel):
    """Quality statistics for one processed document."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    regulation: str
    source_filename: str
    total_pdf_pages: int = Field(ge=0)
    successfully_extracted_pages: int = Field(ge=0)
    empty_pages: int = Field(ge=0)
    failed_pages: int = Field(ge=0)
    total_extracted_characters: int = Field(ge=0)
    detected_articles: int = Field(ge=0)
    detected_annexes: int = Field(ge=0)
    generated_chunks: int = Field(ge=0)
    average_chunk_size: float = Field(ge=0)
    maximum_chunk_size: int = Field(ge=0)
    pages_requiring_ocr_review: list[int]
    warnings: list[str]
    detected_article_numbers: list[str]
    detected_annex_labels: list[str]
    unknown_block_count: int = Field(ge=0)
