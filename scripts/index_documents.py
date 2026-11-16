"""Embed Phase 3 chunks and store them in local Qdrant.

Run from the project root:

    python scripts/index_documents.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.logging_config import configure_logging
from app.indexing.indexer import index_regulatory_documents


def main() -> int:
    """Index the configured chunk files and return a process exit code."""
    configure_logging("INFO")
    report = index_regulatory_documents(ROOT)
    if report.verification_passed and not report.failed_chunks:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
