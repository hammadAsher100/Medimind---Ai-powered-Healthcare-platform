"""FHIR Compatible Patient Record Export — maps internal models to FHIR R4 resources."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

FHIR_VERSION = "4.0.1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


def _bundle(resource_type: str, resources: list[dict]) -> dict:
    """Wrap resources in a FHIR Bundle."""
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "meta": {
            "lastUpdated": _now_iso(),
            "profile": [
                "http://hl7.org/fhir/StructureDefinition/Bundle"
            ],
        },
        "total": len(resources),
        "entry": [
            {
                "fullUrl": f"urn:uuid:{r.get('id', _new_id())}",
                "resource": r,
            }
            for r in resources
        ],
    }


def patient_to_fhir(patient_data: dict[str, Any]) -> dict:
    """Map internal patient data to a FHIR Patient resource."""
    profile = patient_data.get("medical_profile", {})
    user = patient_data.get("user", {})

    gender_map = {"male": "male", "female": "female", "other": "other"}
    gender = gender_map.get(
        profile.get("gender", "").lower(),
        "unknown",
    )

    resource: dict[str, Any] = {
        "resourceType": "Patient",
        "id": _new_id(),
        "meta": {
            "lastUpdated": _now_iso(),
        },
        "identifier": [
            {
                "system": "http://medimind.ai/user-id",
                "value": str(user.get("id", "")),
            }
        ],
        "name": [
            {
                "use": "official",
                "family": user.get("last_name", ""),
                "given": [user.get("first_name", "")],
            }
        ],
        "gender": gender,
        "birthDate": user.get("date_of_birth", ""),
        "active": True,
    }

    # Add extensions for non-standard fields
    if profile.get("blood_type"):
        resource.setdefault("extension", []).append(
            {
                "url": "http://medimind.ai/fhir/StructureDefinition/blood-type",
                "valueString": profile["blood_type"],
            }
        )

    if profile.get("smoking_status"):
        coding_map = {
            "current": {"code": "449804003", "display": "Current smoker"},
            "former": {"code": "8517006", "display": "Ex-smoker"},
            "never": {"code": "266919007", "display": "Non-smoker"},
        }
        status = profile["smoking_status"].lower()
        if status in coding_map:
            resource.setdefault("extension", []).append(
                {
                    "url": (
                        "http://medimind.ai/fhir/StructureDefinition/"
                        "smoking-status"
                    ),
                    "valueCodeableConcept": {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                **coding_map[status],
                            }
                        ]
                    },
                }
            )

    # Add physical measurements
    if profile.get("height_cm"):
        resource.setdefault("extension", []).append(
            {
                "url": "http://medimind.ai/fhir/StructureDefinition/height",
                "valueQuantity": {
                    "value": profile["height_cm"],
                    "unit": "cm",
                    "system": "http://unitsofmeasure.org",
                    "code": "cm",
                },
            }
        )

    if profile.get("weight_kg"):
        resource.setdefault("extension", []).append(
            {
                "url": "http://medimind.ai/fhir/StructureDefinition/weight",
                "valueQuantity": {
                    "value": profile["weight_kg"],
                    "unit": "kg",
                    "system": "http://unitsofmeasure.org",
                    "code": "kg",
                },
            }
        )

    return resource


def observation_to_fhir(obs: dict[str, Any]) -> dict:
    """Map internal ClinicalObservation to a FHIR Observation resource."""
    loinc_map = {
        "glucose": "2345-7",
        "hba1c": "4548-4",
        "cholesterol": "2093-3",
        "hdl": "2085-9",
        "ldl": "2089-1",
        "triglycerides": "2571-8",
        "creatinine": "2160-0",
        "urea": "3094-0",
        "hemoglobin": "718-7",
        "wbc": "6690-2",
        "rbc": "789-8",
        "platelets": "777-3",
        "tsh": "3016-3",
        "vitamin_d": "1989-3",
        "vitamin_b12": "2132-9",
    }

    test_name = obs.get("standardised_name", obs.get("test_name", "")).lower()
    loinc_code = loinc_map.get(test_name, "")

    resource: dict[str, Any] = {
        "resourceType": "Observation",
        "id": _new_id(),
        "meta": {"lastUpdated": _now_iso()},
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": (
                            "http://terminology.hl7.org/CodeSystem/"
                            "observation-category"
                        ),
                        "code": "laboratory",
                        "display": "Laboratory",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": loinc_code,
                    "display": test_name,
                }
            ],
            "text": obs.get("test_name", test_name),
        },
    }

    # Value
    if obs.get("numeric_value") is not None:
        resource["valueQuantity"] = {
            "value": obs["numeric_value"],
            "unit": obs.get("original_unit", ""),
            "system": "http://unitsofmeasure.org",
        }
    else:
        resource["valueString"] = obs.get("original_value", "")

    # Reference range
    if obs.get("reference_range_low") is not None or obs.get(
        "reference_range_high"
    ) is not None:
        ref_range: dict[str, Any] = {}
        if obs.get("reference_range_low") is not None:
            ref_range["low"] = {
                "value": obs["reference_range_low"],
                "unit": obs.get("original_unit", ""),
            }
        if obs.get("reference_range_high") is not None:
            ref_range["high"] = {
                "value": obs["reference_range_high"],
                "unit": obs.get("original_unit", ""),
            }
        resource["referenceRange"] = [ref_range]

    # Effective date
    if obs.get("collection_date"):
        resource["effectiveDateTime"] = obs["collection_date"]

    return resource


def prediction_to_fhir(pred: dict[str, Any]) -> dict:
    """Map internal prediction to a FHIR RiskAssessment resource."""
    resource: dict[str, Any] = {
        "resourceType": "RiskAssessment",
        "id": _new_id(),
        "meta": {"lastUpdated": _now_iso()},
        "status": "final",
        "subject": {
            "reference": f"Patient/{pred.get('user_id', '')}",
        },
        "basis": [
            {
                "display": pred.get("disease", "unknown"),
            }
        ],
        "prediction": [
            {
                "outcome": {
                    "text": pred.get("disease", "unknown"),
                },
                "probabilityDecimal": (
                    pred.get("risk_percentage", 0) / 100.0
                ),
                "text": (
                    f"Risk: {pred.get('risk_percentage', 0):.1f}% "
                    f"({pred.get('risk_level', 'unknown')})"
                ),
            }
        ],
    }

    if pred.get("created_at"):
        resource["occurrenceDateTime"] = pred["created_at"]

    return resource


def health_score_to_fhir(score: dict[str, Any]) -> dict:
    """Map internal health score to a FHIR Observation (assessment)."""
    return {
        "resourceType": "Observation",
        "id": _new_id(),
        "meta": {"lastUpdated": _now_iso()},
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": (
                            "http://terminology.hl7.org/CodeSystem/"
                            "observation-category"
                        ),
                        "code": "survey",
                        "display": "Survey",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://medimind.ai",
                    "code": "health-score",
                    "display": "MediMind Health Score",
                }
            ],
            "text": "Health Score",
        },
        "valueQuantity": {
            "value": score.get("score", 0),
            "unit": "points",
            "system": "http://unitsofmeasure.org",
            "code": "{score}",
        },
        "effectiveDateTime": score.get("created_at", _now_iso()),
    }


def allergy_to_fhir(allergy: dict[str, Any]) -> dict:
    """Map internal allergy to a FHIR AllergyIntolerance resource."""
    severity_map = {
        "mild": "mild",
        "moderate": "moderate",
        "severe": "severe",
    }

    resource: dict[str, Any] = {
        "resourceType": "AllergyIntolerance",
        "id": _new_id(),
        "meta": {"lastUpdated": _now_iso()},
        "clinicalStatus": {
            "coding": [
                {
                    "system": (
                        "http://terminology.hl7.org/CodeSystem/"
                        "allergyintolerance-clinical"
                    ),
                    "code": "active",
                }
            ]
        },
        "verificationStatus": {
            "coding": [
                {
                    "system": (
                        "http://terminology.hl7.org/CodeSystem/"
                        "allergyintolerance-verification"
                    ),
                    "code": "confirmed",
                }
            ]
        },
        "type": "allergy",
        "category": ["medication"],
        "code": {
            "text": allergy.get("allergen", ""),
        },
        "criticality": severity_map.get(
            allergy.get("severity", "unknown"), "unknown"
        ),
    }

    return resource


def export_patient_bundle(
    patient_data: dict[str, Any],
    observations: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    health_score: dict[str, Any] | None = None,
    allergies: list[dict[str, Any]] | None = None,
) -> dict:
    """Generate a complete FHIR Bundle for a patient.

    This is an educational export — not certified for clinical exchange.
    """
    resources: list[dict] = []

    # Patient
    resources.append(patient_to_fhir(patient_data))

    # Observations
    for obs in observations:
        resources.append(observation_to_fhir(obs))

    # Predictions
    for pred in predictions:
        resources.append(prediction_to_fhir(pred))

    # Health score
    if health_score:
        resources.append(health_score_to_fhir(health_score))

    # Allergies
    for allergy in (allergies or []):
        resources.append(allergy_to_fhir(allergy))

    bundle = _bundle("collection", resources)

    # Add MetaMediMind metadata
    bundle["meta"]["extension"] = [
        {
            "url": "http://medimind.ai/fhir/StructureDefinition/export-info",
            "extension": [
                {
                    "url": "generatedBy",
                    "valueString": "MediMind AI",
                },
                {
                    "url": "fhirVersion",
                    "valueString": FHIR_VERSION,
                },
                {
                    "url": "disclaimer",
                    "valueString": (
                        "This FHIR bundle was generated for educational "
                        "and informational purposes. It may not contain "
                        "all clinically relevant data. Do not use for "
                        "clinical decision-making without verification "
                        "by a qualified healthcare provider."
                    ),
                },
            ],
        }
    ]

    return bundle
