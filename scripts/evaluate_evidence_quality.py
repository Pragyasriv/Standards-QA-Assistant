"""Score the separate evidence-quality questions against the existing indexes.

Run from the project root:

    python scripts/evaluate_evidence_quality.py

The script does not edit the original retrieval or guardrail question files,
the chunk files, or the Qdrant collection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.logging_config import configure_logging
from app.guardrails.service import GuardrailService


def main() -> int:
    """Write one result per evidence-quality question."""
    configure_logging("WARNING")
    source = ROOT / "evaluation" / "evidence_quality_questions.json"
    questions = json.loads(source.read_text(encoding="utf-8"))
    service = GuardrailService()
    rows: list[dict[str, object]] = []
    try:
        for item in questions:
            response = service.review(item["question"])
            chunks = response.retrieved_chunks
            problems = _problems(item, response.status, response.detected_regulation, chunks)
            rows.append(
                {
                    "id": item["id"],
                    "case": item["case"],
                    "question": item["question"],
                    "status": response.status,
                    "regulation": response.detected_regulation,
                    "reason": response.reason,
                    "chunk_ids": [chunk.chunk_id for chunk in chunks],
                    "articles": [chunk.article for chunk in chunks],
                    "annexes": [chunk.annex for chunk in chunks],
                    "scores": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "dense_score": chunk.dense_score,
                            "bm25_score": chunk.bm25_score,
                            "rrf_score": chunk.rrf_score,
                        }
                        for chunk in chunks
                    ],
                    "passed": not problems,
                    "problems": problems,
                }
            )
    finally:
        service.close()
    passed = sum(1 for row in rows if row["passed"])
    report = {"passed": passed, "failed": len(rows) - passed, "total": len(rows), "results": rows}
    destination = ROOT / "evaluation" / "evidence_quality_report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report {destination}")
    return 0 if report["failed"] == 0 else 1


def _problems(item: dict[str, object], status: str, regulation: str | None, chunks: list) -> list[str]:
    problems: list[str] = []
    if status != item["expect_status"]:
        problems.append(f"status {status}")
    if regulation != item["expect_regulation"]:
        problems.append(f"regulation {regulation}")
    require_annex = item.get("require_annex")
    if require_annex is not None:
        if not chunks or any(chunk.annex != require_annex for chunk in chunks):
            problems.append(f"annex {require_annex} was not the evidence")
    require_article = item.get("require_article")
    if require_article is not None:
        if not chunks or any(chunk.article != require_article for chunk in chunks):
            problems.append(f"article {require_article} was not the evidence")
    for article in item.get("forbid_article_only", []):
        if any(chunk.article == article and chunk.annex is None for chunk in chunks):
            problems.append(f"article {article} was returned without the list")
    for chunk in chunks:
        if chunk.regulation != item["expect_regulation"]:
            problems.append(f"{chunk.chunk_id} is {chunk.regulation}")
    return problems


if __name__ == "__main__":
    raise SystemExit(main())
