"""Pipeline records, score aggregation, resume files, and reports.

Abstentions and provider failures stay in the dataset. A metric mean uses only
questions where that metric returned a number. Missing scores stay null. They
are not filled with 0 or 1.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.dataset import RagasQuestion

FinalStatus = Literal[
    "invalid",
    "out_of_scope",
    "clarification_required",
    "insufficient_evidence",
    "answered",
    "abstained",
    "provider_failure",
    "pipeline_error",
]

_ANSWERED = "answered"
_ABSTAINED_STATUSES = {
    "invalid",
    "out_of_scope",
    "clarification_required",
    "insufficient_evidence",
    "abstained",
}
_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
_BUCKETS = ("[0,0.25)", "[0.25,0.5)", "[0.5,0.75)", "[0.75,1]")

CURATION_NOTES = [
    (
        "A question asking which substances are listed in REACH Annex XIV was not used. "
        "Chunk 02006R1907-20260622:01070 contains the annex title and column headings, "
        "not substance rows. The authorisation question uses Article 56(1), which states "
        "the prohibition for a substance included in Annex XIV and stops at 'unless:'."
    ),
    (
        "REACH Article 7(1)(b) is not in the indexed chunks 00111 or 00112. The Article 7 "
        "question states condition (a) only and does not invent condition (b)."
    ),
]

SCORING_POLICY = (
    "Every curated question stays in the dataset. RAGAS is called only when the "
    "pipeline returned a non-empty answer and at least one retrieved context. "
    "Abstentions, guardrail rejections, validation failures, and provider failures "
    "are counted separately and receive null metric scores. Each mean is the "
    "arithmetic mean of the non-null scores for that metric. Nulls are not replaced "
    "with zero. A transient provider error is retried; a validation rejection is not."
)

BASELINE_COMPARISON = {
    "phase5_dense_file": "evaluation/baseline_dense.json",
    "phase5_dense": {
        "recall_at_5": 0.4,
        "precision_at_5": 0.18,
        "mean_reciprocal_rank": 0.2533,
        "off_topic_rejection_rate": 0.5,
        "question_file": "evaluation/retrieval_questions.json",
    },
    "phase5_1_file": "evaluation/phase5_1_report.json",
    "phase5_1_bm25": {
        "recall_at_5": 0.4,
        "precision_at_5": 0.2,
        "mean_reciprocal_rank": 0.3333,
    },
    "phase5_1_hybrid": {
        "recall_at_5": 0.4,
        "precision_at_5": 0.22,
        "mean_reciprocal_rank": 0.275,
    },
    "note": (
        "RAGAS context precision and context recall judge retrieved text against "
        "reference answers on the 50-question file. Phase 5 Recall@5, Precision@5, "
        "and MRR rank expected chunk ids on a different 16-question file. The two "
        "sets of numbers are not the same metric and this evaluation does not claim "
        "that one improved on the other."
    ),
}


class PipelineRecord(BaseModel):
    """One live pipeline result. Scores are stored separately so a rerun can resume."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    regulation: Literal["RoHS", "REACH"]
    reference_answer: str
    expected_chunk_ids: list[str]
    actual_answer: str | None = None
    retrieved_contexts: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    actual_citations: list[dict[str, Any]] = Field(default_factory=list)
    guardrail_status: str
    final_status: FinalStatus
    generation_model: str | None = None
    generation_temperature: float | None = None
    evaluated_at: str
    failure_reason: str | None = None
    abstention_reason: str | None = None


class MetricRecord(BaseModel):
    """RAGAS scores for one question. Null means that metric was not computed."""

    model_config = ConfigDict(extra="forbid")

    id: str
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    metric_error: str | None = None
    evaluator_model: str | None = None
    evaluated_at: str


class MetricSummary(BaseModel):
    """Distribution of one metric over the questions that produced a number."""

    model_config = ConfigDict(extra="forbid")

    count: int
    excluded: int
    mean: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    buckets: dict[str, int]


class QuestionResult(BaseModel):
    """Per-question row in the report."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    regulation: str
    reference_answer: str
    actual_answer: str | None = None
    expected_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    missing_expected_chunk_ids: list[str]
    actual_citations: list[dict[str, Any]]
    guardrail_status: str
    final_status: str
    failure_reason: str | None = None
    abstention_reason: str | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    metric_error: str | None = None
    scored: bool


class RagasReport(BaseModel):
    """Measured evaluation report. Fields stay null until a run fills them."""

    model_config = ConfigDict(extra="forbid")

    questions: int
    rohs_questions: int
    reach_questions: int
    successfully_answered: int
    scored_all_four_metrics: int
    unexpected_abstentions: int
    provider_failures: int
    pipeline_errors: int
    evaluation_failures: int
    metrics: dict[str, MetricSummary]
    per_question: list[QuestionResult]
    lowest_scoring: list[QuestionResult]
    failure_examples: list[dict[str, Any]]
    scoring_policy: str = SCORING_POLICY
    curation_notes: list[str] = Field(default_factory=lambda: list(CURATION_NOTES))
    baseline_comparison: dict[str, Any] = Field(default_factory=lambda: dict(BASELINE_COMPARISON))
    guardrail_evaluation: dict[str, Any] | None = None
    ragas_version: str | None = None
    evaluator_model: str | None = None
    evaluator_temperature: float | None = None
    evaluator_max_tokens: int | None = None
    answer_relevancy_strictness: int | None = None
    embedding_model: str | None = None
    generation_model: str | None = None
    generation_temperature: float | None = None
    live_qdrant: bool = False
    live_bm25: bool = False
    live_groq: bool = False
    generated_at: str


def utc_now() -> str:
    """Return an ISO timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path, model: type[PipelineRecord] | type[MetricRecord]) -> dict[str, Any]:
    """Load completed records by id. A missing file is an empty resume state."""
    if not path.is_file():
        return {}
    found: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = model.model_validate_json(line)
        found[row.id] = row
    return found


def append_jsonl(path: Path, record: PipelineRecord | MetricRecord) -> None:
    """Append one completed record so a later run can skip it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")


def pending_ids(questions: list[RagasQuestion], done: dict[str, Any]) -> list[RagasQuestion]:
    """Return questions that do not yet have a saved record."""
    return [item for item in questions if item.id not in done]


def can_score(record: PipelineRecord) -> bool:
    """RAGAS needs an answer and retrieved context. Abstentions do not qualify."""
    return bool(
        record.final_status == _ANSWERED
        and record.actual_answer
        and record.actual_answer.strip()
        and record.retrieved_contexts
    )


def metric_record_is_complete(record: MetricRecord) -> bool:
    """True when all four RAGAS metrics returned a number."""
    return all(getattr(record, name) is not None for name in _METRICS)


def needs_ragas_rescoring(existing: MetricRecord | None) -> bool:
    """True when scoring should run again for this saved row."""
    if existing is None:
        return True
    if existing.metric_error is not None:
        return True
    return not metric_record_is_complete(existing)


def should_attempt_ragas_scoring(
    question_id: str,
    score_filter: set[str] | None,
    record: PipelineRecord,
    existing: MetricRecord | None,
) -> bool:
    """Decide whether the live scorer should run for one question."""
    if score_filter is not None and question_id not in score_filter:
        return False
    if not can_score(record):
        return False
    return needs_ragas_rescoring(existing)


def load_guardrail_report_json(path: Path) -> dict[str, object] | None:
    """Read a saved guardrail report without running guardrails again."""
    if not path.is_file():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return loaded


def exclusion_reason(record: PipelineRecord) -> str | None:
    """Explain why a saved pipeline row is left out of the metric means."""
    if can_score(record):
        return None
    if record.final_status == "provider_failure":
        return record.failure_reason or "provider failure"
    if record.final_status == "pipeline_error":
        return record.failure_reason or "pipeline error"
    if record.final_status in _ABSTAINED_STATUSES:
        return record.failure_reason or record.abstention_reason or record.final_status
    if not record.retrieved_contexts:
        return "no retrieved context"
    return "no answer to score"


def summarize_metric(values: list[float | None], total: int) -> MetricSummary:
    """Summarize one metric. Nulls are excluded from the mean and the buckets."""
    present = [value for value in values if value is not None]
    buckets = {name: 0 for name in _BUCKETS}
    for value in present:
        buckets[_bucket(value)] += 1
    if not present:
        return MetricSummary(count=0, excluded=total, buckets=buckets)
    quartiles = statistics.quantiles(present, n=4, method="inclusive") if len(present) >= 2 else [present[0]] * 3
    return MetricSummary(
        count=len(present),
        excluded=total - len(present),
        mean=sum(present) / len(present),
        minimum=min(present),
        maximum=max(present),
        p25=quartiles[0],
        p50=quartiles[1],
        p75=quartiles[2],
        buckets=buckets,
    )


def build_report(
    questions: list[RagasQuestion],
    pipeline: dict[str, PipelineRecord],
    metrics: dict[str, MetricRecord],
    *,
    generated_at: str | None = None,
    ragas_version: str | None = None,
    evaluator_model: str | None = None,
    evaluator_temperature: float | None = None,
    evaluator_max_tokens: int | None = None,
    answer_relevancy_strictness: int | None = None,
    embedding_model: str | None = None,
    generation_model: str | None = None,
    generation_temperature: float | None = None,
    guardrail_evaluation: dict[str, Any] | None = None,
    live_qdrant: bool = False,
    live_bm25: bool = False,
    live_groq: bool = False,
) -> RagasReport:
    """Join questions, pipeline rows, and scores without dropping unanswered items."""
    rows: list[QuestionResult] = []
    for item in questions:
        record = pipeline.get(item.id)
        score = metrics.get(item.id)
        rows.append(_question_result(item, record, score))
    metric_map = {
        name: summarize_metric([getattr(row, name) for row in rows], len(rows))
        for name in _METRICS
    }
    lowest = sorted(
        [row for row in rows if row.scored],
        key=lambda row: _row_mean(row),
    )[:5]
    return RagasReport(
        questions=len(questions),
        rohs_questions=sum(item.regulation == "RoHS" for item in questions),
        reach_questions=sum(item.regulation == "REACH" for item in questions),
        successfully_answered=sum(row.final_status == _ANSWERED and bool(row.actual_answer) for row in rows),
        scored_all_four_metrics=sum(
            all(getattr(row, name) is not None for name in _METRICS) for row in rows
        ),
        unexpected_abstentions=sum(row.final_status in _ABSTAINED_STATUSES for row in rows),
        provider_failures=sum(row.final_status == "provider_failure" for row in rows),
        pipeline_errors=sum(row.final_status == "pipeline_error" for row in rows),
        evaluation_failures=sum(row.metric_error is not None for row in rows),
        metrics=metric_map,
        per_question=rows,
        lowest_scoring=lowest,
        failure_examples=_examples(rows),
        guardrail_evaluation=guardrail_evaluation,
        ragas_version=ragas_version,
        evaluator_model=evaluator_model,
        evaluator_temperature=evaluator_temperature,
        evaluator_max_tokens=evaluator_max_tokens,
        answer_relevancy_strictness=answer_relevancy_strictness,
        embedding_model=embedding_model,
        generation_model=generation_model,
        generation_temperature=generation_temperature,
        live_qdrant=live_qdrant,
        live_bm25=live_bm25,
        live_groq=live_groq,
        generated_at=generated_at or utc_now(),
    )


def write_report(report: RagasReport, json_path: Path, markdown_path: Path) -> None:
    """Write the machine-readable report and the readable report."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: RagasReport) -> str:
    """Render the report without rounding away the stored scores."""
    lines = [
        "# RAGAS evaluation",
        "",
        f"Generated at {report.generated_at}.",
        "",
        "## Dataset",
        "",
        f"- Questions: {report.questions}",
        f"- RoHS: {report.rohs_questions}",
        f"- REACH: {report.reach_questions}",
        f"- Successfully answered: {report.successfully_answered}",
        f"- Scored on all four metrics: {report.scored_all_four_metrics}",
        f"- Unexpected abstentions: {report.unexpected_abstentions}",
        f"- Provider failures: {report.provider_failures}",
        f"- Pipeline errors: {report.pipeline_errors}",
        f"- Evaluation failures: {report.evaluation_failures}",
        "",
        "## Configuration",
        "",
        f"- RAGAS version: {report.ragas_version}",
        f"- Evaluator model: {report.evaluator_model}",
        f"- Evaluator temperature: {report.evaluator_temperature}",
        f"- Evaluator max tokens: {report.evaluator_max_tokens}",
        f"- Answer relevancy strictness: {report.answer_relevancy_strictness}",
        f"- Embedding model: {report.embedding_model}",
        f"- Generation model: {report.generation_model}",
        f"- Generation temperature: {report.generation_temperature}",
        f"- Live Qdrant: {report.live_qdrant}",
        f"- Live BM25: {report.live_bm25}",
        f"- Live Groq: {report.live_groq}",
        "",
        "## Scoring policy",
        "",
        report.scoring_policy,
        "",
        "## Metric distributions",
        "",
        "| Metric | Count | Excluded | Mean | Min | P25 | P50 | P75 | Max |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in _METRICS:
        summary = report.metrics[name]
        lines.append(
            "| {name} | {count} | {excluded} | {mean} | {minimum} | {p25} | {p50} | {p75} | {maximum} |".format(
                name=name.replace("_", " "),
                count=summary.count,
                excluded=summary.excluded,
                mean=_fmt(summary.mean),
                minimum=_fmt(summary.minimum),
                p25=_fmt(summary.p25),
                p50=_fmt(summary.p50),
                p75=_fmt(summary.p75),
                maximum=_fmt(summary.maximum),
            )
        )
    lines.extend(["", "Buckets are [0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1].", ""])
    for name in _METRICS:
        summary = report.metrics[name]
        rendered = ", ".join(f"{bucket}={summary.buckets[bucket]}" for bucket in _BUCKETS)
        lines.append(f"- {name}: {rendered}")
    lines.extend(["", "## Per-question scores", "", "| ID | Regulation | Status | Faithfulness | Relevancy | Precision | Recall |", "| --- | --- | --- | --- | --- | --- | --- |"])
    for row in report.per_question:
        lines.append(
            f"| {row.id} | {row.regulation} | {row.final_status} | {_fmt(row.faithfulness)} | "
            f"{_fmt(row.answer_relevancy)} | {_fmt(row.context_precision)} | {_fmt(row.context_recall)} |"
        )
    lines.extend(["", "## Lowest-scoring answered questions", ""])
    if not report.lowest_scoring:
        lines.append("No question produced a RAGAS score.")
    for row in report.lowest_scoring:
        lines.extend(_example_lines(row))
    lines.extend(["", "## Failure examples", ""])
    if not report.failure_examples:
        lines.append("No abstention, provider failure, retrieval miss, or metric error was recorded.")
    for example in report.failure_examples:
        lines.append(f"- `{example['id']}` ({example['kind']}): {example['detail']}")
    lines.extend(["", "## Guardrail evaluation", ""])
    if report.guardrail_evaluation is None:
        lines.append("The separate guardrail evaluation was not attached to this report.")
    else:
        guard = report.guardrail_evaluation
        lines.extend(
            [
                "These rates come from `evaluation/guardrail_questions.json`. They are not combined with the RAGAS means.",
                "",
                f"- Supported acceptance: {_fmt(guard.get('supported_acceptance_rate'))}",
                f"- Ambiguous clarification: {_fmt(guard.get('ambiguous_clarification_rate'))}",
                f"- Unsupported rejection: {_fmt(guard.get('unsupported_rejection_rate'))}",
                f"- False acceptance: {_fmt(guard.get('false_acceptance_rate'))}",
                f"- False rejection: {_fmt(guard.get('false_rejection_rate'))}",
            ]
        )
    lines.extend(["", "## Baseline comparison", "", report.baseline_comparison["note"], ""])
    dense = report.baseline_comparison["phase5_dense"]
    hybrid = report.baseline_comparison["phase5_1_hybrid"]
    lines.append(
        f"Recorded dense Recall@5 {dense['recall_at_5']}, Precision@5 {dense['precision_at_5']}, "
        f"MRR {dense['mean_reciprocal_rank']}."
    )
    lines.append(
        f"Recorded hybrid Recall@5 {hybrid['recall_at_5']}, Precision@5 {hybrid['precision_at_5']}, "
        f"MRR {hybrid['mean_reciprocal_rank']}."
    )
    lines.extend(["", "## Curation notes", ""])
    for note in report.curation_notes:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "The judge is a language model, so a score is an evaluation judgment rather than a legal conclusion. "
            "Answer relevancy in RAGAS 0.4 compares the question with questions generated from the answer and does not read the retrieved context. "
            "Context precision and context recall use the reference answer, not the expected chunk ids. "
            "A validation abstention keeps the public API reason and does not keep the rejected answer text.",
            "",
        ]
    )
    return "\n".join(lines)


def _question_result(
    item: RagasQuestion,
    record: PipelineRecord | None,
    score: MetricRecord | None,
) -> QuestionResult:
    if record is None:
        return QuestionResult(
            id=item.id,
            question=item.question,
            regulation=item.regulation,
            reference_answer=item.reference_answer,
            expected_chunk_ids=list(item.expected_chunk_ids),
            retrieved_chunk_ids=[],
            missing_expected_chunk_ids=list(item.expected_chunk_ids),
            actual_citations=[],
            guardrail_status="not_run",
            final_status="pipeline_error",
            failure_reason="no pipeline result was saved",
            scored=False,
        )
    missing = [chunk_id for chunk_id in item.expected_chunk_ids if chunk_id not in record.retrieved_chunk_ids]
    return QuestionResult(
        id=item.id,
        question=item.question,
        regulation=item.regulation,
        reference_answer=item.reference_answer,
        actual_answer=record.actual_answer,
        expected_chunk_ids=list(item.expected_chunk_ids),
        retrieved_chunk_ids=list(record.retrieved_chunk_ids),
        missing_expected_chunk_ids=missing,
        actual_citations=list(record.actual_citations),
        guardrail_status=record.guardrail_status,
        final_status=record.final_status,
        failure_reason=record.failure_reason,
        abstention_reason=record.abstention_reason,
        faithfulness=None if score is None else score.faithfulness,
        answer_relevancy=None if score is None else score.answer_relevancy,
        context_precision=None if score is None else score.context_precision,
        context_recall=None if score is None else score.context_recall,
        metric_error=None if score is None else score.metric_error,
        scored=score is not None and any(getattr(score, name) is not None for name in _METRICS),
    )


def _examples(rows: list[QuestionResult]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        if row.final_status in _ABSTAINED_STATUSES or row.final_status in {"provider_failure", "pipeline_error"}:
            examples.append(
                {
                    "id": row.id,
                    "kind": row.final_status,
                    "detail": row.failure_reason or row.abstention_reason or row.final_status,
                }
            )
        elif row.missing_expected_chunk_ids and row.final_status == _ANSWERED:
            examples.append(
                {
                    "id": row.id,
                    "kind": "missing_expected_chunk",
                    "detail": "expected chunks not retrieved: " + ", ".join(row.missing_expected_chunk_ids),
                }
            )
        if row.metric_error:
            examples.append({"id": row.id, "kind": "metric_error", "detail": row.metric_error})
    return examples


def _example_lines(row: QuestionResult) -> list[str]:
    return [
        f"### {row.id}",
        "",
        row.question,
        "",
        f"Status `{row.final_status}`. Faithfulness {_fmt(row.faithfulness)}, "
        f"answer relevancy {_fmt(row.answer_relevancy)}, "
        f"context precision {_fmt(row.context_precision)}, "
        f"context recall {_fmt(row.context_recall)}.",
        "",
    ]


def _row_mean(row: QuestionResult) -> float:
    values = [getattr(row, name) for name in _METRICS if getattr(row, name) is not None]
    if not values:
        return 1.0
    return sum(values) / len(values)


def _bucket(value: float) -> str:
    if value < 0.25:
        return "[0,0.25)"
    if value < 0.5:
        return "[0.25,0.5)"
    if value < 0.75:
        return "[0.5,0.75)"
    return "[0.75,1]"


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)
