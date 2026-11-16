"""Offline checks for the RAGAS dataset, aggregation, and resume files."""

from pathlib import Path

import pytest

from app.ask.service import _VALIDATION_REASON
from app.evaluation.dataset import (
    DatasetProblems,
    RagasQuestion,
    load_chunk_texts,
    load_questions,
    schema_problems,
    source_problems,
)
from app.evaluation.pipeline import is_transient, run_question
from app.evaluation.results import (
    MetricRecord,
    PipelineRecord,
    append_jsonl,
    build_report,
    load_guardrail_report_json,
    load_jsonl,
    metric_record_is_complete,
    needs_ragas_rescoring,
    pending_ids,
    should_attempt_ragas_scoring,
    write_report,
)
from app.evaluation.settings import EvaluationConfigError, EvaluationSettings
from app.generation.errors import GenerationInvocationError, GenerationValidationError
from app.generation.models import GenerationMetadata, GenerationResponse, ResolvedCitation
from app.guardrails.models import GuardrailResponse
from app.retrieval.models import RetrievedChunk

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "evaluation" / "ragas_questions.json"


def test_curated_dataset_has_fifty_unique_questions() -> None:
    questions = load_questions(QUESTIONS)
    assert len(questions) == 50
    assert len({item.id for item in questions}) == 50
    assert sum(item.regulation == "RoHS" for item in questions) == 25
    assert sum(item.regulation == "REACH" for item in questions) == 25
    assert all(item.reference_answer.strip() for item in questions)
    assert all(item.reference_contexts for item in questions)


def test_curated_contexts_match_source_chunks() -> None:
    chunks = load_chunk_texts(
        ROOT / "data" / "processed" / "rohs" / "chunks.jsonl",
        ROOT / "data" / "processed" / "reach" / "chunks.jsonl",
    )
    if not chunks:
        pytest.skip("local chunk files are not present")
    questions = load_questions(QUESTIONS)
    assert source_problems(questions, chunks) == []


def test_schema_rejects_missing_answer_context_and_bad_source() -> None:
    question = _question()
    question.reference_answer = " "
    question.reference_contexts = [""]
    question.expected_chunk_ids = ["not-a-chunk"]
    question.verified_phrases = ["missing"]
    problems = schema_problems([question] * 50)
    assert any("missing a reference answer" in item for item in problems)
    assert any("empty reference context" in item for item in problems)
    assert any("invalid source reference" in item for item in problems)
    assert any("not unique" in item for item in problems)


def test_schema_rejects_the_wrong_regulation_split() -> None:
    questions = [_question(regulation="RoHS") for _ in range(50)]
    for index, item in enumerate(questions):
        item.id = f"q-{index}"
    problems = schema_problems(questions)
    assert any("REACH" in item for item in problems)


def test_source_check_rejects_a_context_that_is_not_the_chunk() -> None:
    question = _question()
    problems = source_problems([question], {question.expected_chunk_ids[0]: "different text"})
    assert any("not the text" in item for item in problems)
    missing = source_problems([question], {})
    assert any("missing chunk" in item for item in missing)


def test_load_questions_rejects_a_non_list(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(DatasetProblems):
        load_questions(path)


def test_abstentions_stay_in_the_dataset_and_out_of_the_mean() -> None:
    questions = [_question(), _question()]
    questions[1].id = "second"
    pipeline = {
        questions[0].id: run_question(questions[0], _Guardrail("approved"), _Generator("answer"), clock=lambda: "t"),
        questions[1].id: run_question(questions[1], _Guardrail("approved"), _Generator("invalid"), clock=lambda: "t"),
    }
    scores = {
        questions[0].id: MetricRecord(
            id=questions[0].id,
            faithfulness=0.5,
            answer_relevancy=1.0,
            context_precision=0.25,
            context_recall=0.0,
            evaluated_at="t",
        )
    }
    report = build_report(questions, pipeline, scores, generated_at="t")
    assert report.questions == 2
    assert report.successfully_answered == 1
    assert report.unexpected_abstentions == 1
    assert report.metrics["faithfulness"].count == 1
    assert report.metrics["faithfulness"].excluded == 1
    assert report.metrics["faithfulness"].mean == 0.5
    assert report.per_question[1].faithfulness is None
    assert report.per_question[1].abstention_reason == _VALIDATION_REASON


def test_provider_failure_is_counted_and_not_retried_when_it_is_permanent() -> None:
    question = _question()
    sleeps: list[float] = []
    record = run_question(
        question,
        _Guardrail("approved"),
        _Generator("down"),
        max_retries=2,
        sleeper=sleeps.append,
        clock=lambda: "t",
    )
    assert record.final_status == "provider_failure"
    assert sleeps == []
    report = build_report([question], {question.id: record}, {}, generated_at="t")
    assert report.provider_failures == 1
    assert report.metrics["context_recall"].count == 0


def test_transient_provider_failure_is_retried() -> None:
    question = _question()
    generator = _Generator("rate-limit")
    record = run_question(
        question,
        _Guardrail("approved"),
        generator,
        max_retries=1,
        sleeper=lambda _seconds: None,
        clock=lambda: "t",
    )
    assert generator.calls == 2
    assert record.final_status == "provider_failure"
    assert is_transient(GenerationInvocationError("429 rate limit"))
    assert not is_transient(GenerationValidationError("omits an application condition"))


def test_report_round_trip_and_resume(tmp_path: Path) -> None:
    question = _question()
    record = run_question(question, _Guardrail("insufficient_evidence"), _Generator("answer"), clock=lambda: "t")
    path = tmp_path / "intermediate.jsonl"
    append_jsonl(path, record)
    loaded = load_jsonl(path, type(record))
    assert pending_ids([question], loaded) == []
    report = build_report([question], loaded, {}, generated_at="t")
    write_report(report, tmp_path / "report.json", tmp_path / "report.md")
    text = (tmp_path / "report.json").read_text(encoding="utf-8")
    assert question.id in text
    assert "null" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert report.unexpected_abstentions == 1


def test_evaluator_model_is_required() -> None:
    settings = EvaluationSettings(groq_api_key="present", groq_model="generator", ragas_evaluator_model=" ")
    with pytest.raises(EvaluationConfigError):
        settings.require_evaluator()


def test_empty_score_id_filter_preserves_full_resume() -> None:
    valid = {"a", "b"}
    settings = EvaluationSettings(ragas_score_ids="  ")
    assert settings.score_id_filter(valid) is None


def test_score_id_filter_selects_known_ids() -> None:
    valid = {"reach-article-5-no-data", "reach-article-6-registration"}
    settings = EvaluationSettings(
        ragas_score_ids="reach-article-5-no-data, reach-article-6-registration"
    )
    assert settings.score_id_filter(valid) == valid


def test_score_id_filter_rejects_unknown_ids() -> None:
    settings = EvaluationSettings(ragas_score_ids="missing-id,reach-article-5-no-data")
    with pytest.raises(EvaluationConfigError, match="unknown question ids: missing-id"):
        settings.score_id_filter({"reach-article-5-no-data"})


def test_should_attempt_ragas_scoring_respects_selected_ids() -> None:
    record = _pipeline_record()
    partial = MetricRecord(
        id=record.id,
        faithfulness=0.5,
        answer_relevancy=None,
        context_precision=None,
        context_recall=None,
        metric_error="answer_relevancy: rate limit",
        evaluated_at="t",
    )
    selected = {record.id}
    assert should_attempt_ragas_scoring(record.id, selected, record, partial) is True
    assert should_attempt_ragas_scoring("other-id", selected, record, None) is False


def test_should_attempt_ragas_scoring_skips_completed_rows() -> None:
    record = _pipeline_record()
    complete = MetricRecord(
        id=record.id,
        faithfulness=1.0,
        answer_relevancy=1.0,
        context_precision=1.0,
        context_recall=1.0,
        evaluated_at="t",
    )
    assert needs_ragas_rescoring(complete) is False
    assert should_attempt_ragas_scoring(record.id, None, record, complete) is False


def test_should_attempt_ragas_scoring_reuses_partial_metrics() -> None:
    record = _pipeline_record()
    partial = MetricRecord(
        id=record.id,
        faithfulness=1.0,
        answer_relevancy=None,
        context_precision=1.0,
        context_recall=1.0,
        metric_error="answer_relevancy: rate limit",
        evaluated_at="t",
    )
    assert metric_record_is_complete(partial) is False
    assert needs_ragas_rescoring(partial) is True


def test_pipeline_pending_ids_unaffected_by_score_filter() -> None:
    second = _question(
        id="second",
        question="What does RoHS Article 5 require?",
        verified_phrases=["evidence"],
    )
    questions = [_question(), second]
    done = {
        questions[0].id: _pipeline_record(),
        second.id: _pipeline_record(id=second.id),
    }
    assert pending_ids(questions, done) == []


def test_guardrail_report_suppression_keeps_ragas_aggregation() -> None:
    question = _question()
    pipeline = {question.id: _pipeline_record()}
    scores = {
        question.id: MetricRecord(
            id=question.id,
            faithfulness=0.5,
            answer_relevancy=0.5,
            context_precision=0.5,
            context_recall=0.5,
            evaluated_at="t",
        )
    }
    without = build_report([question], pipeline, scores, generated_at="t", guardrail_evaluation=None)
    with_guard = build_report(
        [question],
        pipeline,
        scores,
        generated_at="t",
        guardrail_evaluation={"supported_acceptance_rate": 1.0},
    )
    assert without.metrics == with_guard.metrics
    assert without.scored_all_four_metrics == with_guard.scored_all_four_metrics
    assert without.guardrail_evaluation is None
    assert with_guard.guardrail_evaluation == {"supported_acceptance_rate": 1.0}


def test_load_guardrail_report_json_reads_saved_file(tmp_path: Path) -> None:
    path = tmp_path / "guardrail_report.json"
    path.write_text('{"supported_acceptance_rate": 1.0}', encoding="utf-8")
    assert load_guardrail_report_json(path) == {"supported_acceptance_rate": 1.0}


class _Guardrail:
    def __init__(self, status: str) -> None:
        self.status = status

    def review(self, question: object) -> GuardrailResponse:
        chunks = []
        if self.status == "approved":
            chunks = [
                RetrievedChunk(
                    chunk_id="02011L0065-20260701:00022",
                    document_id="02011L0065-20260701",
                    regulation="RoHS",
                    text="Article 4 evidence",
                )
            ]
        return GuardrailResponse(
            status=self.status,  # type: ignore[arg-type]
            scope="supported" if self.status in {"approved", "insufficient_evidence"} else "invalid",
            reason="recorded reason",
            retrieved_chunks=chunks,
        )


class _Generator:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.calls = 0

    def generate(self, question: str, guardrail: GuardrailResponse) -> GenerationResponse:
        self.calls += 1
        if self.mode == "invalid":
            raise GenerationValidationError("omits an application condition")
        if self.mode == "down":
            raise GenerationInvocationError("model rejected the request")
        if self.mode == "rate-limit":
            raise GenerationInvocationError("429 rate limit")
        return GenerationResponse(
            question=question,
            answer="Member States shall ensure the Annex II limit.",
            status="answered",
            citations=[
                ResolvedCitation(
                    chunk_id="02011L0065-20260701:00022",
                    document_id="02011L0065-20260701",
                    regulation="RoHS",
                    article="4",
                    page_start=8,
                    page_end=8,
                    source_filename="CL2011L0065EN0230010.0001.pdf",
                )
            ],
            model="test-model",
            generation_metadata=GenerationMetadata(
                evidence_chunk_count=1,
                validated_citation_count=1,
                abstained=False,
                latency_ms=1,
                temperature=0,
            ),
        )


def _pipeline_record(**overrides: object) -> PipelineRecord:
    payload = {
        "id": "rohs-article-4-prevention",
        "question": "What does RoHS Article 4(1) require?",
        "regulation": "RoHS",
        "reference_answer": "EEE does not contain the substances listed in Annex II.",
        "expected_chunk_ids": ["02011L0065-20260701:00022"],
        "actual_answer": "Member States shall ensure the Annex II limit.",
        "retrieved_contexts": ["Article 4 evidence"],
        "retrieved_chunk_ids": ["02011L0065-20260701:00022"],
        "actual_citations": [],
        "guardrail_status": "approved",
        "final_status": "answered",
        "evaluated_at": "t",
    }
    payload.update(overrides)
    return PipelineRecord.model_validate(payload)


def _question(**overrides: object) -> RagasQuestion:
    payload = {
        "id": "rohs-article-4-prevention",
        "question": "What does RoHS Article 4(1) require?",
        "regulation": "RoHS",
        "reference_answer": "EEE does not contain the substances listed in Annex II.",
        "reference_contexts": ["does not contain the substances listed in Annex II"],
        "expected_chunk_ids": ["02011L0065-20260701:00022"],
        "article": "4",
        "paragraph": "1",
        "verified_phrases": ["does not contain the substances listed in Annex II"],
    }
    payload.update(overrides)
    return RagasQuestion.model_validate(payload)
