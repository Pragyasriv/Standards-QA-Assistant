"""Evaluate the live RAG pipeline with the curated 50-question dataset.

Run from the project root after installing requirements-eval.txt:

    python scripts/evaluate_ragas.py

The script calls the existing guardrail and generation services, then scores
answered rows with RAGAS. It writes intermediate JSONL files and can resume.
It does not change retrieval thresholds, guardrail rules, or the indexes.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.logging_config import configure_logging
from app.evaluation.dataset import load_questions
from app.evaluation.metrics import RagasScorer
from app.evaluation.pipeline import run_question
from app.evaluation.results import (
    MetricRecord,
    PipelineRecord,
    append_jsonl,
    build_report,
    load_guardrail_report_json,
    load_jsonl,
    pending_ids,
    should_attempt_ragas_scoring,
    utc_now,
    write_report,
)
from app.evaluation.settings import EvaluationConfigError, EvaluationSettings
from app.generation.service import GenerationService
from app.guardrails.evaluate import evaluate_guardrails, load_guardrail_questions
from app.guardrails.service import GuardrailService

logger = logging.getLogger(__name__)

QUESTIONS = ROOT / "evaluation" / "ragas_questions.json"
INTERMEDIATE = ROOT / "evaluation" / "reports" / "ragas_intermediate.jsonl"
SCORES = ROOT / "evaluation" / "reports" / "ragas_scores.jsonl"
REPORT_JSON = ROOT / "evaluation" / "reports" / "ragas_report.json"
REPORT_MD = ROOT / "evaluation" / "reports" / "ragas_report.md"


def main() -> int:
    """Run the pipeline, score answered rows, then the separate guardrail file."""
    configure_logging("WARNING")
    settings = EvaluationSettings()
    settings.require_evaluator()
    questions = load_questions(QUESTIONS)
    valid_ids = {item.id for item in questions}
    try:
        score_filter = settings.score_id_filter(valid_ids)
    except EvaluationConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    pipeline = load_jsonl(INTERMEDIATE, PipelineRecord)
    guardrails = GuardrailService()
    generation = GenerationService()
    try:
        for item in pending_ids(questions, pipeline):
            record = run_question(
                item,
                guardrails,
                generation,
                max_retries=settings.ragas_max_retries,
            )
            append_jsonl(INTERMEDIATE, record)
            pipeline[item.id] = record
            logger.warning("pipeline %s status=%s", item.id, record.final_status)
    finally:
        guardrails.close()

    scores = load_jsonl(SCORES, MetricRecord)
    scorer = RagasScorer(settings)
    for item in questions:
        record = pipeline.get(item.id)
        if record is None:
            continue
        existing = scores.get(item.id)
        if not should_attempt_ragas_scoring(item.id, score_filter, record, existing):
            continue
        scored = scorer.score(record, existing if isinstance(existing, MetricRecord) else None)
        append_jsonl(SCORES, scored)
        scores[item.id] = scored
        logger.warning("scored %s", item.id)

    guardrail_report = _guardrail_report(settings)
    import ragas

    models = {row.generation_model for row in pipeline.values() if row.generation_model}
    temperatures = {
        row.generation_temperature
        for row in pipeline.values()
        if row.generation_temperature is not None
    }
    report = build_report(
        questions,
        pipeline,
        scores,
        generated_at=utc_now(),
        ragas_version=ragas.__version__,
        evaluator_model=settings.evaluator_model(),
        evaluator_temperature=settings.ragas_evaluator_temperature,
        evaluator_max_tokens=settings.ragas_evaluator_max_tokens,
        answer_relevancy_strictness=settings.ragas_answer_relevancy_strictness,
        embedding_model=settings.ragas_embedding_model,
        generation_model=next(iter(models), settings.generator_model()),
        generation_temperature=next(iter(temperatures), settings.groq_temperature),
        guardrail_evaluation=guardrail_report,
        live_qdrant=True,
        live_bm25=True,
        live_groq=True,
    )
    write_report(report, REPORT_JSON, REPORT_MD)
    print(
        f"questions={report.questions} answered={report.successfully_answered} "
        f"scored={report.scored_all_four_metrics} abstained={report.unexpected_abstentions} "
        f"provider_failures={report.provider_failures}"
    )
    for name, summary in report.metrics.items():
        print(f"{name} mean={summary.mean} count={summary.count} excluded={summary.excluded}")
    print(f"report {REPORT_JSON}")
    return 0


def _guardrail_report(settings: EvaluationSettings) -> dict[str, object] | None:
    """Score the guardrail file or reload the saved report when skipping."""
    destination = ROOT / "evaluation" / "guardrail_report.json"
    if settings.ragas_skip_guardrail_report:
        return load_guardrail_report_json(destination)
    questions = load_guardrail_questions(ROOT / "evaluation" / "guardrail_questions.json")
    service = GuardrailService()
    try:
        report = evaluate_guardrails(service, questions)
    finally:
        service.close()
    destination.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report.model_dump()


if __name__ == "__main__":
    raise SystemExit(main())
