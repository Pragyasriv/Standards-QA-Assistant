"""Score the guardrail question file.

Run from the project root:

    python scripts/evaluate_guardrails.py

Supported questions retrieve from the existing indexes. The script does not
edit retrieval_questions.json, the chunk files, or the Qdrant collection.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.logging_config import configure_logging
from app.guardrails.evaluate import evaluate_guardrails, load_guardrail_questions
from app.guardrails.service import GuardrailService


def main() -> int:
    """Print measured guardrail rates."""
    configure_logging("WARNING")
    questions = load_guardrail_questions(ROOT / "evaluation" / "guardrail_questions.json")
    service = GuardrailService()
    try:
        report = evaluate_guardrails(service, questions)
    finally:
        service.close()
    destination = ROOT / "evaluation" / "guardrail_report.json"
    destination.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(report.model_dump_json(indent=2))
    print(f"report {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
