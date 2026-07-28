"""FastAPI router for Medication Safety Passport."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any

from services.medication_safety import (
    run_safety_check,
    extract_medications_from_text,
)
import metrics as m

router = APIRouter(prefix="/medication", tags=["medication-safety"])


class MedicationInput(BaseModel):
    medication_name: str
    status: str = "active"
    dosage: str = ""
    frequency: str = ""
    drug_class: str = ""


class AllergyInput(BaseModel):
    allergen: str
    severity: str = "unknown"


class SafetyCheckRequest(BaseModel):
    user_id: int
    medications: list[MedicationInput]
    allergies: list[AllergyInput] = []


class ExtractFromTextRequest(BaseModel):
    user_id: int
    text: str


@router.post("/safety-check")
async def safety_check(req: SafetyCheckRequest) -> dict[str, Any]:
    """Run comprehensive medication safety check."""
    meds = [m.model_dump() for m in req.medications]
    allergies = [a.model_dump() for a in req.allergies]
    result = run_safety_check(meds, allergies)
    m.medication_safety_checks.labels(has_alerts=str(result["total_alerts"] > 0).lower()).inc()
    for alert in result.get("alerts", []):
        m.medication_alerts_generated.labels(
            alert_type=alert.get("alert_type", "unknown"),
            severity=alert.get("severity", "info"),
        ).inc()
    result["user_id"] = req.user_id
    return result


@router.post("/extract-from-text")
async def extract_from_text(req: ExtractFromTextRequest) -> dict[str, Any]:
    """Extract medications from free text using pattern matching."""
    medications = extract_medications_from_text(req.text)
    return {
        "user_id": req.user_id,
        "medications_found": medications,
        "count": len(medications),
        "disclaimer": (
            "Text-based medication extraction uses pattern matching with "
            "limited accuracy. For reliable extraction, use the AI-powered "
            "report analyzer or manual entry."
        ),
    }
