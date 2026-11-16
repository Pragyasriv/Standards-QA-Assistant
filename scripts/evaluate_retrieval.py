"""Compare dense, BM25, and hybrid retrieval on the shared question file.

Run from the project root:

    python scripts/evaluate_retrieval.py

The recorded Phase 5 baseline is evaluation/baseline_dense.json. This script
does not edit retrieval_questions.json and does not change the cutoff.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.logging_config import configure_logging
from app.retrieval.evaluate import compare_methods, evaluate_questions, load_baseline, load_questions
from app.retrieval.service import RetrievalService


def main() -> int:
    """Print measured metrics for the three retrieval methods."""
    configure_logging("WARNING")
    questions = load_questions(ROOT / "evaluation" / "retrieval_questions.json")
    baseline = load_baseline(ROOT / "evaluation" / "baseline_dense.json")
    service = RetrievalService()
    try:
        dense = evaluate_questions(service, questions, top_k=service.settings.retrieval_top_k)
        comparison = compare_methods(
            service,
            questions,
            service.settings.retrieval_top_k,
            baseline_recall=baseline["recall_at_k"],
            baseline_precision=baseline["precision_at_k"],
            baseline_mrr=baseline["mean_reciprocal_rank"],
            baseline_off_topic_rejection=baseline["off_topic_rejection_rate"],
        )
    finally:
        service.close()
    report_path = ROOT / "evaluation" / "phase5_1_report.json"
    report_path.write_text(comparison.model_dump_json(indent=2), encoding="utf-8")
    print(dense.model_dump_json(indent=2))
    for method in comparison.methods:
        print(
            f"{method.method}\trecall={method.recall_at_k}\tprecision={method.precision_at_k}"
            f"\tmrr={method.mean_reciprocal_rank}\toff_topic_rejected={method.off_topic_rejected}"
            f"\tfailures={','.join(method.ranking_failures)}"
        )
    for change in comparison.changes:
        print(
            f"{change.id}\tdense={change.dense_rank}\tbm25={change.bm25_rank}"
            f"\thybrid={change.hybrid_rank}\tbm25={change.bm25_versus_dense}"
            f"\thybrid={change.hybrid_versus_dense}"
        )
    print(f"dense_matches_recorded_baseline {comparison.dense_matches_recorded_baseline}")
    print(f"report {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
