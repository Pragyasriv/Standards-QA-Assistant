"""Manifest records for downloaded regulatory documents."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class DocumentRecord(BaseModel):
    """One official file stored under data/raw."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    regulation_name: str = Field(min_length=1)
    document_title: str = Field(min_length=1)
    regulation_number: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    local_filename: str = Field(min_length=1)
    version_date: date
    download_date: date
    file_format: str = Field(pattern=r"^pdf$")
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    celex: str = Field(min_length=1)
    consolidation_reference: str = Field(min_length=1)
    download_url: str = Field(min_length=1)
    file_size_bytes: int = Field(gt=0)
    relative_path: str = Field(min_length=1)


class DocumentManifest(BaseModel):
    """The local catalogue of acquired regulatory files."""

    model_config = ConfigDict(extra="forbid")

    documents: list[DocumentRecord] = Field(min_length=1)
