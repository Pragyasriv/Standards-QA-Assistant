"""Optional Qdrant filters for regulation, document, article, and annex."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator
from qdrant_client.http import models

from app.retrieval.errors import InvalidFilterError

RegulationName = Literal["RoHS", "REACH"]
_REGULATION_NAMES = {"rohs": "RoHS", "reach": "REACH"}


class SearchFilters(BaseModel):
    """Filters applied inside Qdrant before candidates are returned."""

    model_config = ConfigDict(extra="forbid")

    regulation: RegulationName | None = None
    document_id: str | None = None
    article: str | None = None
    annex: str | None = None

    @field_validator("regulation", mode="before")
    @classmethod
    def normalize_regulation(cls, value: object) -> object:
        """Accept the stored regulation names in either case."""
        if value is None or not isinstance(value, str):
            return value
        canonical = _REGULATION_NAMES.get(value.strip().lower())
        if canonical is None:
            raise InvalidFilterError("regulation must be RoHS or REACH.")
        return canonical

    @field_validator("document_id", "article", "annex")
    @classmethod
    def reject_blank(cls, value: str | None) -> str | None:
        """Keep a supplied identifier, and reject a blank one."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise InvalidFilterError("A filter value cannot be blank.")
        return stripped


def build_qdrant_filter(filters: SearchFilters | None) -> models.Filter | None:
    """Build a Qdrant payload filter. Unset fields are not constrained."""
    if filters is None:
        return None
    must: list[models.FieldCondition] = []
    for key, value in (
        ("metadata.regulation", filters.regulation),
        ("metadata.document_id", filters.document_id),
        ("metadata.article", filters.article),
        ("metadata.annex", filters.annex),
    ):
        if value is None:
            continue
        must.append(
            models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value),
            )
        )
    if not must:
        return None
    return models.Filter(must=must)
