"""Combine input checks, domain classification, and retrieval evidence.

This service does not generate an answer and does not call an LLM.
"""

from __future__ import annotations

from app.guardrails.catalog import GuardrailCatalog, default_catalog
from app.guardrails.classify import classify_question
from app.guardrails.evidence import EvidenceRetriever, assess_evidence
from app.guardrails.models import GuardrailResponse, ScopeDecision, citations_from_chunks
from app.guardrails.settings import GuardrailSettings
from app.guardrails.validate import validate_input
from app.retrieval.service import RetrievalService


class GuardrailService:
    """Decide whether a question may proceed to answer generation."""

    def __init__(
        self,
        retrieval: EvidenceRetriever | None = None,
        settings: GuardrailSettings | None = None,
        catalog: GuardrailCatalog | None = None,
    ) -> None:
        self.settings = settings or GuardrailSettings()
        self.catalog = catalog or default_catalog()
        self.retrieval = retrieval

    def review(self, question: object) -> GuardrailResponse:
        """Validate, classify, and, for a supported question, check retrieved evidence."""
        checked = validate_input(question, self.settings)
        if not checked.ok:
            return GuardrailResponse(
                status="invalid",
                scope="invalid",
                reason=checked.reason or "Question is invalid.",
            )
        decision = classify_question(checked.text, self.catalog)
        if decision.scope != "supported":
            return _from_scope(decision)
        assessment = assess_evidence(self._retriever(), checked.text, decision)
        status = "approved" if assessment.available else "insufficient_evidence"
        return GuardrailResponse(
            status=status,
            scope="supported",
            reason=assessment.reason,
            detected_regulation=decision.detected_regulation,
            clarification_question=None,
            evidence_available=assessment.available,
            retrieved_chunks=assessment.chunks,
            citation_metadata=citations_from_chunks(assessment.chunks),
            article=decision.article,
            annex=decision.annex,
        )

    def close(self) -> None:
        """Release a retrieval connection this service opened."""
        closer = getattr(self.retrieval, "close", None)
        if closer is not None:
            closer()

    def _retriever(self) -> EvidenceRetriever:
        if self.retrieval is None:
            self.retrieval = RetrievalService()
        return self.retrieval


def _from_scope(decision: ScopeDecision) -> GuardrailResponse:
    status = decision.scope
    if status == "supported":
        status = "insufficient_evidence"
    return GuardrailResponse(
        status=status,
        scope=decision.scope,
        reason=decision.reason,
        detected_regulation=decision.detected_regulation,
        detected_unsupported_instrument=decision.detected_unsupported_instrument,
        clarification_question=decision.clarification_question,
        evidence_available=False,
        article=decision.article,
        annex=decision.annex,
    )
