"""FastAPI router for Clinician Review and Model Feedback."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any

from services.evidence_tracker import (
    build_evidence_response,
    format_evidence_citations,
)
import metrics as m

router = APIRouter(prefix="/review", tags=["clinician-review"])


class EvidenceRequest(BaseModel):
    response_text: str
    response_type: str = "general"
    patient_context: dict[str, Any] | None = None


class FeedbackRecord(BaseModel):
    user_id: int
    model_name: str
    prediction_id: str = ""
    feedback_type: str  # correct, incorrect, partially_correct, hallucination
    original_output: str = ""
    corrected_output: str = ""
    notes: str = ""


class ReviewDecisionRecord(BaseModel):
    user_id: int
    recommendation_type: str
    recommendation_id: str = ""
    ai_summary: str = ""
    clinician_decision: str  # accepted, modified, rejected, deferred
    clinician_notes: str = ""


@router.post("/build-evidence")
async def build_evidence(req: EvidenceRequest) -> dict[str, Any]:
    """Build an evidence-linked response with provenance tracking."""
    result = build_evidence_response(
        response_text=req.response_text,
        response_type=req.response_type,
        patient_context=req.patient_context,
    )
    output = result.to_dict()
    output["citations"] = format_evidence_citations(result.evidence_sources)
    return output


@router.post("/record-feedback")
async def record_feedback(req: FeedbackRecord) -> dict[str, Any]:
    """
    Record model feedback for quality monitoring.

    This endpoint records feedback in the Django backend via the
    ai_proxy. The caller (Django view) will persist it to the
    reviews_Modelfeedback table.
    """
    from clinical.django_client import record_model_feedback

    m.model_feedback_events.labels(
        feedback_type=req.feedback_type,
        model_name=req.model_name,
    ).inc()
    return record_model_feedback(req.model_dump())


@router.post("/record-decision")
async def record_decision(req: ReviewDecisionRecord) -> dict[str, Any]:
    """
    Record a clinician review decision.

    This endpoint records the decision in the Django backend.
    """
    from clinical.django_client import record_review_decision

    m.review_decisions.labels(decision_type=req.clinician_decision).inc()
    return record_review_decision(req.model_dump())
