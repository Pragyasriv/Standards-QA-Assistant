"""Verify official regulatory PDFs against the local manifest.

Run from the project root:

    python scripts/verify_documents.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.documents.verify import main

if __name__ == "__main__":
    raise SystemExit(main())
