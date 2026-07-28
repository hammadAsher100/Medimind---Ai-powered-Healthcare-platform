"""FastAPI router for Longitudinal Laboratory Intelligence."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any

from services.lab_intelligence import (
    analyze_longitudinal_data,
    standardise_test_name,
    classify_abnormality,
)
import metrics as m

router = APIRouter(prefix="/lab", tags=["lab-intelligence"])


class ObservationInput(BaseModel):
    test_name: str
    numeric_value: float | None = None
    original_value: str = ""
    original_unit: str = ""
    reference_range_low: float | None = None
    reference_range_high: float | None = None
    collection_date: str = ""
    source_label: str = ""


class AnalyzeTrendRequest(BaseModel):
    user_id: int
    test_name: str
    observations: list[ObservationInput] = Field(
        ..., min_length=1, description="Observations sorted chronologically"
    )


class BulkAnalyzeRequest(BaseModel):
    user_id: int
    observations: list[ObservationInput]


@router.post("/analyze-trend")
async def analyze_trend(req: AnalyzeTrendRequest) -> dict[str, Any]:
    """Analyze longitudinal trends for a single test."""
    obs_dicts = [
        {
            "test_name": o.test_name,
            "numeric_value": o.numeric_value,
            "original_value": o.original_value,
            "original_unit": o.original_unit,
            "reference_range_low": o.reference_range_low,
            "reference_range_high": o.reference_range_high,
            "collection_date": o.collection_date,
            "source_label": o.source_label,
        }
        for o in req.observations
    ]
    result = analyze_longitudinal_data(obs_dicts)
    m.lab_trend_analyses.labels(status="success").inc()
    result["user_id"] = req.user_id
    result["disclaimer"] = (
        "Trend analysis is deterministic and based on reference ranges. "
        "Clinical interpretation should be performed by a qualified "
        "healthcare provider."
    )
    return result


@router.post("/analyze-all")
async def analyze_all(req: BulkAnalyzeRequest) -> dict[str, Any]:
    """Analyze trends for all tests in a patient's observation set."""
    # Group by test_name
    grouped: dict[str, list[dict]] = {}
    for o in req.observations:
        name = standardise_test_name(o.test_name)
        grouped.setdefault(name, []).append({
            "test_name": o.test_name,
            "numeric_value": o.numeric_value,
            "original_value": o.original_value,
            "original_unit": o.original_unit,
            "reference_range_low": o.reference_range_low,
            "reference_range_high": o.reference_range_high,
            "collection_date": o.collection_date,
            "source_label": o.source_label,
        })

    results = {}
    alerts_all: list[dict] = []
    for test_name, obs_list in grouped.items():
        obs_list.sort(key=lambda x: x.get("collection_date", ""))
        result = analyze_longitudinal_data(obs_list)
        results[test_name] = result
        alerts_all.extend(result.get("alerts", []))

    critical_count = sum(
        1 for a in alerts_all if a.get("severity") == "critical"
    )

    return {
        "user_id": req.user_id,
        "test_results": results,
        "summary": {
            "tests_analyzed": len(results),
            "total_alerts": len(alerts_all),
            "critical_alerts": critical_count,
        },
        "disclaimer": (
            "This analysis is for educational and decision-support "
            "purposes. All findings must be verified by a clinician."
        ),
    }


@router.post("/classify-abnormality")
async def classify_observation(
    numeric_value: float,
    reference_range_low: float | None = None,
    reference_range_high: float | None = None,
) -> dict[str, Any]:
    """Classify a single observation against its reference range."""
    status = classify_abnormality(
        numeric_value, reference_range_low, reference_range_high
    )
    return {
        "numeric_value": numeric_value,
        "reference_range": {
            "low": reference_range_low,
            "high": reference_range_high,
        },
        "abnormality_status": status,
    }
