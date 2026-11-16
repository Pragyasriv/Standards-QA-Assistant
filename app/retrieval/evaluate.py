"""Score a manually written retrieval set.

Ranking metrics use the raw top-K list. Rejection of unrelated questions is
counted separately from the configured similarity cutoff. The cutoff is not
chosen from these questions.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.retrieval.models import RetrievedChunk
from app.retrieval.service import RetrievalService


class ExpectedReference(BaseModel):
    """A citation that should appear for a question. Unused fields stay unset."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str | None = None
    regulation: str | None = None
    article: str | None = None
    annex: str | None = None


class EvaluationQuestion(BaseModel):
    """One manually written retrieval question."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    category: str
    expected: list[ExpectedReference] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    """Measured retrieval metrics for one run."""

    model_config = ConfigDict(extra="forbid")

    questions: int
    ranked_questions: int
    recall_at_k: float
    precision_at_k: float
    mean_reciprocal_rank: float
    off_topic_questions: int
    off_topic_rejected: int
    off_topic_rejection_rate: float
    failures: list[str]
    top_k: int
    minimum_similarity: float | None


def load_questions(path: Path) -> list[EvaluationQuestion]:
    """Read the curated question file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [EvaluationQuestion.model_validate(item) for item in raw]


def evaluate_questions(
    service: RetrievalService,
    questions: list[EvaluationQuestion],
    top_k: int,
) -> EvaluationReport:
    """Run every question and aggregate ranking and rejection metrics."""
    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    failures: list[str] = []
    off_topic = [item for item in questions if not item.expected]
    rejected = 0
    for item in questions:
        ranked = service.search(
            item.question,
            top_k=top_k,
            apply_threshold=False,
        )
        decided = service.search(item.question, top_k=top_k)
        if item.expected:
            recalls.append(_recall(ranked.results, item.expected))
            precisions.append(_precision(ranked.results, item.expected, top_k))
            reciprocal_ranks.append(_reciprocal_rank(ranked.results, item.expected))
            if _recall(ranked.results, item.expected) == 0:
                failures.append(item.id)
        elif decided.status == "insufficient_evidence":
            rejected += 1
        else:
            failures.append(item.id)
    return EvaluationReport(
        questions=len(questions),
        ranked_questions=len(recalls),
        recall_at_k=_mean(recalls),
        precision_at_k=_mean(precisions),
        mean_reciprocal_rank=_mean(reciprocal_ranks),
        off_topic_questions=len(off_topic),
        off_topic_rejected=rejected,
        off_topic_rejection_rate=round(rejected / len(off_topic), 4) if off_topic else 0.0,
        failures=failures,
        top_k=top_k,
        minimum_similarity=service.settings.retrieval_min_similarity,
    )


def matches_reference(result: RetrievedChunk, reference: ExpectedReference) -> bool:
    """Return whether a retrieved chunk is one of the recorded references."""
    if reference.chunk_id is not None:
        return result.chunk_id == reference.chunk_id
    if reference.regulation is not None and result.regulation != reference.regulation:
        return False
    if reference.article is not None:
        return result.article == reference.article and result.annex is None
    if reference.annex is not None:
        return result.annex == reference.annex
    return False


def _recall(results: list[RetrievedChunk], expected: list[ExpectedReference]) -> float:
    found = sum(1 for reference in expected if any(matches_reference(result, reference) for result in results))
    return found / len(expected)


def _precision(results: list[RetrievedChunk], expected: list[ExpectedReference], top_k: int) -> float:
    relevant = sum(1 for result in results if any(matches_reference(result, reference) for reference in expected))
    return relevant / top_k


def _reciprocal_rank(results: list[RetrievedChunk], expected: list[ExpectedReference]) -> float:
    for rank, result in enumerate(results, start=1):
        if any(matches_reference(result, reference) for reference in expected):
            return 1 / rank
    return 0.0


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


class HitSummary(BaseModel):
    """A compact retrieved hit for the comparison report."""

    model_config = ConfigDict(extra="forbid")

    rank: int
    chunk_id: str
    regulation: str
    article: str | None = None
    annex: str | None = None
    similarity_score: float | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    text_preview: str


class QuestionRanking(BaseModel):
    """One question scored against one retrieval method."""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    first_relevant_rank: int | None = None
    reciprocal_rank: float
    recall: float
    precision: float
    hits: list[HitSummary]


class MethodReport(BaseModel):
    """Ranking metrics for one method on the shared question file."""

    model_config = ConfigDict(extra="forbid")

    method: str
    recall_at_k: float
    precision_at_k: float
    mean_reciprocal_rank: float
    ranking_failures: list[str]
    off_topic_questions: int
    off_topic_rejected: int | None = None
    off_topic_rejection_rate: float | None = None
    off_topic_rule: str
    questions: list[QuestionRanking]
    top_k: int


class QuestionChange(BaseModel):
    """How BM25 and hybrid ranks compare with dense retrieval for one question."""

    model_config = ConfigDict(extra="forbid")

    id: str
    dense_rank: int | None
    bm25_rank: int | None
    hybrid_rank: int | None
    bm25_versus_dense: str
    hybrid_versus_dense: str


class ComparisonReport(BaseModel):
    """Dense, BM25, and hybrid results on the same questions and references."""

    model_config = ConfigDict(extra="forbid")

    top_k: int
    baseline_recall_at_k: float
    baseline_precision_at_k: float
    baseline_mean_reciprocal_rank: float
    baseline_off_topic_rejection_rate: float
    dense_matches_recorded_baseline: bool
    methods: list[MethodReport]
    changes: list[QuestionChange]


def compare_methods(
    service: RetrievalService,
    questions: list[EvaluationQuestion],
    top_k: int,
    *,
    baseline_recall: float,
    baseline_precision: float,
    baseline_mrr: float,
    baseline_off_topic_rejection: float,
) -> ComparisonReport:
    """Score dense, BM25, and hybrid retrieval with the same metric definitions."""
    dense_rows = [_rank_question(service, item, top_k, "dense") for item in questions]
    bm25_rows = [_rank_question(service, item, top_k, "bm25") for item in questions]
    hybrid_rows = [_rank_question(service, item, top_k, "hybrid") for item in questions]
    cutoff = service.settings.retrieval_min_similarity
    dense = _method_report("dense", dense_rows, questions, top_k, "cosine_cutoff", cutoff)
    bm25 = _method_report("bm25", bm25_rows, questions, top_k, "no_positive_bm25", None)
    hybrid = _method_report("hybrid", hybrid_rows, questions, top_k, "not_gated", None)
    changes = [
        QuestionChange(
            id=item.id,
            dense_rank=dense_rows[index].first_relevant_rank,
            bm25_rank=bm25_rows[index].first_relevant_rank,
            hybrid_rank=hybrid_rows[index].first_relevant_rank,
            bm25_versus_dense=_change(dense_rows[index], bm25_rows[index]),
            hybrid_versus_dense=_change(dense_rows[index], hybrid_rows[index]),
        )
        for index, item in enumerate(questions)
        if item.expected
    ]
    return ComparisonReport(
        top_k=top_k,
        baseline_recall_at_k=baseline_recall,
        baseline_precision_at_k=baseline_precision,
        baseline_mean_reciprocal_rank=baseline_mrr,
        baseline_off_topic_rejection_rate=baseline_off_topic_rejection,
        dense_matches_recorded_baseline=(
            dense.recall_at_k == baseline_recall
            and dense.precision_at_k == baseline_precision
            and dense.mean_reciprocal_rank == baseline_mrr
            and dense.off_topic_rejection_rate == baseline_off_topic_rejection
        ),
        methods=[dense, bm25, hybrid],
        changes=changes,
    )


def load_baseline(path: Path) -> dict[str, float]:
    """Read the recorded Phase 5 dense metrics."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "recall_at_k": float(payload["recall_at_k"]),
        "precision_at_k": float(payload["precision_at_k"]),
        "mean_reciprocal_rank": float(payload["mean_reciprocal_rank"]),
        "off_topic_rejection_rate": float(payload["off_topic_rejection_rate"]),
    }


def _rank_question(
    service: RetrievalService,
    item: EvaluationQuestion,
    top_k: int,
    method: str,
) -> QuestionRanking:
    if method == "dense":
        response = service.search(item.question, top_k=top_k, apply_threshold=False)
    elif method == "bm25":
        response = service.search_bm25(item.question, top_k=top_k)
    elif method == "hybrid":
        response = service.search_hybrid(item.question, top_k=top_k)
    else:
        raise ValueError(f"Unknown retrieval method {method!r}.")
    results = response.results
    rank = _first_rank(results, item.expected)
    return QuestionRanking(
        id=item.id,
        category=item.category,
        first_relevant_rank=rank,
        reciprocal_rank=_reciprocal_rank(results, item.expected) if item.expected else 0.0,
        recall=_recall(results, item.expected) if item.expected else 0.0,
        precision=_precision(results, item.expected, top_k) if item.expected else 0.0,
        hits=[_hit(result, index) for index, result in enumerate(results, start=1)],
    )


def _method_report(
    method: str,
    rows: list[QuestionRanking],
    questions: list[EvaluationQuestion],
    top_k: int,
    off_topic_rule: str,
    cutoff: float | None,
) -> MethodReport:
    ranked_rows = [row for row, item in zip(rows, questions, strict=True) if item.expected]
    off_topic = [item for item in questions if not item.expected]
    rejected = _off_topic_rejected(method, rows, questions, cutoff) if off_topic_rule != "not_gated" else None
    return MethodReport(
        method=method,
        recall_at_k=_mean([row.recall for row in ranked_rows]),
        precision_at_k=_mean([row.precision for row in ranked_rows]),
        mean_reciprocal_rank=_mean([row.reciprocal_rank for row in ranked_rows]),
        ranking_failures=[row.id for row in ranked_rows if row.recall == 0],
        off_topic_questions=len(off_topic),
        off_topic_rejected=rejected,
        off_topic_rejection_rate=(
            round(rejected / len(off_topic), 4) if rejected is not None and off_topic else None
        ),
        off_topic_rule=off_topic_rule,
        questions=rows,
        top_k=top_k,
    )


def _off_topic_rejected(
    method: str,
    rows: list[QuestionRanking],
    questions: list[EvaluationQuestion],
    cutoff: float | None,
) -> int:
    rejected = 0
    for row, item in zip(rows, questions, strict=True):
        if item.expected:
            continue
        if method == "bm25":
            if not row.hits:
                rejected += 1
            continue
        best = row.hits[0].similarity_score if row.hits else None
        if not row.hits or cutoff is not None and (best is None or best < cutoff):
            rejected += 1
    return rejected


def _first_rank(results: list[RetrievedChunk], expected: list[ExpectedReference]) -> int | None:
    if not expected:
        return None
    for rank, result in enumerate(results, start=1):
        if any(matches_reference(result, reference) for reference in expected):
            return rank
    return None


def _change(baseline: QuestionRanking, candidate: QuestionRanking) -> str:
    baseline_rank = baseline.first_relevant_rank if baseline.first_relevant_rank is not None else 10**9
    candidate_rank = candidate.first_relevant_rank if candidate.first_relevant_rank is not None else 10**9
    if candidate_rank < baseline_rank or (candidate_rank == baseline_rank and candidate.recall > baseline.recall):
        return "improved"
    if candidate_rank > baseline_rank or (candidate_rank == baseline_rank and candidate.recall < baseline.recall):
        return "worsened"
    return "unchanged"


def _hit(result: RetrievedChunk, rank: int) -> HitSummary:
    preview = " ".join(result.text.split())
    return HitSummary(
        rank=rank,
        chunk_id=result.chunk_id,
        regulation=result.regulation,
        article=result.article,
        annex=result.annex,
        similarity_score=result.similarity_score,
        dense_score=result.dense_score,
        bm25_score=result.bm25_score,
        rrf_score=result.rrf_score,
        text_preview=preview[:180],
    )
