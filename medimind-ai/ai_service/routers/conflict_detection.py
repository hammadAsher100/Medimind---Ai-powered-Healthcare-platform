"""FastAPI router for Clinical Contradiction Detection."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any

from services.conflict_detection import run_conflict_detection
import metrics as m

router = APIRouter(prefix="/conflicts", tags=["conflict-detection"])


class ConflictDetectionRequest(BaseModel):
    user_id: int
    observations: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    timeline_events: list[dict[str, Any]] = []


@router.post("/detect")
async def detect_conflicts(req: ConflictDetectionRequest) -> dict[str, Any]:
    """Run all conflict detection checks across data sources."""
    result = run_conflict_detection(
        observations=req.observations,
        predictions=req.predictions,
        reports=req.reports,
        timeline_events=req.timeline_events,
    )
    m.conflict_detections.labels(
        has_conflicts=str(result["total_conflicts"] > 0).lower()
    ).inc()
    for conflict in result.get("conflicts", []):
        m.conflicts_found.labels(
            conflict_type=conflict.get("conflict_type", "unknown"),
            severity=conflict.get("severity", "info"),
        ).inc()
    result["user_id"] = req.user_id
    result["disclaimer"] = (
        "Conflict detection is based on rule-based heuristics. "
        "Apparent conflicts may have legitimate clinical explanations. "
        "All findings should be reviewed by a qualified clinician."
    )
    return result
