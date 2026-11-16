"""Verify the local regulatory PDFs against the document manifest."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.core.logging_config import configure_logging
from app.documents.sources import MANIFEST_RELATIVE_PATH
from app.documents.validation import validate_manifest_file

logger = logging.getLogger(__name__)


def project_root() -> Path:
    """Return the repository root that contains the app package."""
    return Path(__file__).resolve().parents[2]


def verify_documents(root: Path | None = None) -> list[str]:
    """Check the manifest, PDF files, and SHA-256 checksums."""
    project = root or project_root()
    return validate_manifest_file(project / MANIFEST_RELATIVE_PATH, project)


def main(argv: list[str] | None = None) -> int:
    """Run the integrity check and return a process exit code."""
    argparse.ArgumentParser(
        description="Verify downloaded RoHS and REACH PDFs against the manifest."
    ).parse_args(argv)
    configure_logging("INFO")
    problems = verify_documents()
    if problems:
        for problem in problems:
            logger.error("%s", problem)
        logger.error("Document verification failed with %s problem(s).", len(problems))
        return 1
    logger.info("Document verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
