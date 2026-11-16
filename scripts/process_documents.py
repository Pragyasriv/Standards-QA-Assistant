"""Extract, clean, and chunk the official RoHS and REACH PDFs.

Run from the project root:

    python scripts/process_documents.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.logging_config import configure_logging
from app.processing.pipeline import process_manifest


def main() -> int:
    """Process the manifest and return a process exit code."""
    configure_logging("INFO")
    process_manifest(ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
