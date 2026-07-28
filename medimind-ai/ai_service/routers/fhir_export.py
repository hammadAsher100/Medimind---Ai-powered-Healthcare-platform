"""FastAPI router for FHIR Compatible Patient Record Export."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any

from services.fhir_export import export_patient_bundle
import metrics as m

router = APIRouter(prefix="/fhir", tags=["fhir-export"])


class FHIRExportRequest(BaseModel):
    user_id: int
    patient_data: dict[str, Any]
    observations: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    health_score: dict[str, Any] | None = None
    allergies: list[dict[str, Any]] = []


@router.post("/export")
async def export_bundle(req: FHIRExportRequest) -> dict[str, Any]:
    """Generate a FHIR R4 Bundle from patient data."""
    bundle = export_patient_bundle(
        patient_data=req.patient_data,
        observations=req.observations,
        predictions=req.predictions,
        health_score=req.health_score,
        allergies=req.allergies,
    )
    m.fhir_exports.labels(
        resource_count_bucket=str(len(bundle.get("entry", [])))
    ).inc()
    return bundle


@router.get("/capabilities")
async def capabilities() -> dict[str, Any]:
    """Return FHIR capability statement."""
    return {
        "resourceType": "CapabilityStatement",
        "status": "active",
        "kind": "instance",
        "fhirVersion": "4.0.1",
        "format": ["json"],
        "software": {
            "name": "MediMind AI",
            "version": "1.0.0",
        },
        "rest": [
            {
                "mode": "server",
                "resources": [
                    {"type": "Patient", "interaction": ["read"]},
                    {"type": "Observation", "interaction": ["read"]},
                    {"type": "RiskAssessment", "interaction": ["read"]},
                    {"type": "AllergyIntolerance", "interaction": ["read"]},
                    {"type": "Bundle", "interaction": ["read"]},
                ],
            }
        ],
        "disclaimer": (
            "This FHIR endpoint is for educational purposes. "
            "It does not implement the full FHIR R4 specification "
            "and should not be used for clinical data exchange "
            "without proper certification."
        ),
    }
