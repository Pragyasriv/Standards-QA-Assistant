"""Structured generation output. Citation fields come from stored chunks."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelDraft(BaseModel):
    """The only fields the model is asked to return.

    Extra keys are ignored. Citation metadata is not accepted from the model.
    """

    model_config = ConfigDict(extra="ignore")

    answer: str
    cited_chunk_ids: list[str]
    abstained: bool
    abstention_reason: str | None = None


class ResolvedCitation(BaseModel):
    """One citation copied from the evidence chunk that was supplied."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    regulation: str
    regulation_number: str | None = None
    article: str | None = None
    paragraph: str | None = None
    subparagraph: str | None = None
    annex: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    source_filename: str | None = None


class GenerationMetadata(BaseModel):
    """Operational details that are safe to return and log."""

    model_config = ConfigDict(extra="forbid")

    evidence_chunk_count: int = Field(ge=0)
    validated_citation_count: int = Field(ge=0)
    abstained: bool
    latency_ms: float = Field(ge=0)
    temperature: float


class GenerationResponse(BaseModel):
    """A grounded answer, or an abstention, with backend-resolved citations."""

    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    status: Literal["answered", "abstained"]
    citations: list[ResolvedCitation]
    abstention_reason: str | None = None
    model: str
    generation_metadata: GenerationMetadata
