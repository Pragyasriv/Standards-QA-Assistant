"""Build the BM25 index from the Phase 3 chunk files.

Run from the project root:

    python scripts/build_bm25_index.py

The script reads the chunk JSONL files and writes storage/bm25/index.json.
It does not modify those chunks or the Qdrant collection.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.logging_config import configure_logging
from app.retrieval.bm25 import BM25Index
from app.retrieval.settings import RetrievalSettings


def main() -> int:
    """Build and save the lexical index."""
    configure_logging("INFO")
    settings = RetrievalSettings()
    index = BM25Index.build(
        [settings.rohs_chunks_path, settings.reach_chunks_path],
        k1=settings.bm25_k1,
        b=settings.bm25_b,
    )
    index.save(settings.bm25_index_path)
    print(f"chunks {len(index.documents)}")
    print(f"path {settings.bm25_index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
