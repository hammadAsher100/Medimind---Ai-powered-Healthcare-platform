"""FastAPI router for Patient State Intelligence Engine."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Any

from services.patient_state import (
    build_patient_state,
    check_emergency_keywords,
)
import metrics as m

router = APIRouter(prefix="/patient-state", tags=["patient-state"])


class BuildStateRequest(BaseModel):
    user_id: int
    observations: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    medications: list[dict[str, Any]] = []
    allergies: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    health_score: dict[str, Any] | None = None
    medical_profile: dict[str, Any] | None = None


class EmergencyCheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


@router.post("/build")
async def build_state(req: BuildStateRequest) -> dict[str, Any]:
    """Build a comprehensive patient state snapshot from all data sources."""
    state = build_patient_state(
        user_id=req.user_id,
        observations=req.observations,
        predictions=req.predictions,
        medications=req.medications,
        allergies=req.allergies,
        reports=req.reports,
        health_score=req.health_score,
        medical_profile=req.medical_profile,
    )
    m.patient_state_builds.labels(priority_level=state.priority_level).inc()
    result = state.to_dict()
    result["disclaimer"] = (
        "This patient state is an automated aggregation of available data. "
        "It does not constitute a clinical assessment. All findings "
        "require clinical validation."
    )
    return result


@router.post("/emergency-check")
async def emergency_check(req: EmergencyCheckRequest) -> dict[str, Any]:
    """Check text for emergency keywords. Returns triage result."""
    result = check_emergency_keywords(req.text)
    if result:
        m.emergency_checks.labels(is_emergency="true").inc()
        return result
    m.emergency_checks.labels(is_emergency="false").inc()
    return {
        "is_emergency": False,
        "message": "No emergency keywords detected.",
    }
