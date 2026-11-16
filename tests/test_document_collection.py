"""Integrity check for the acquired official documents, when they are present."""

from pathlib import Path

import pytest

from app.documents.sources import MANIFEST_RELATIVE_PATH
from app.documents.validation import validate_manifest_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / MANIFEST_RELATIVE_PATH


@pytest.mark.skipif(
    not MANIFEST_PATH.is_file(),
    reason="Official PDFs are local and are not stored in Git.",
)
def test_acquired_documents_match_the_manifest() -> None:
    problems = validate_manifest_file(MANIFEST_PATH, PROJECT_ROOT)

    assert problems == []
