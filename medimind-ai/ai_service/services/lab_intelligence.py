"""Longitudinal Laboratory Intelligence — trend calculation and analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Standardised test name mapping (lowercase original → canonical)
TEST_NAME_MAP: dict[str, str] = {
    # Blood glucose
    "glucose": "glucose",
    "blood glucose": "glucose",
    "fasting glucose": "glucose",
    "fasting blood glucose": "glucose",
    "fbg": "glucose",
    "blood sugar": "glucose",
    # HbA1c
    "hba1c": "hba1c",
    "glycated hemoglobin": "hba1c",
    "glycated haemoglobin": "hba1c",
    # Lipids
    "cholesterol": "cholesterol",
    "total cholesterol": "cholesterol",
    "hdl": "hdl",
    "hdl cholesterol": "hdl",
    "ldl": "ldl",
    "ldl cholesterol": "ldl",
    "triglycerides": "triglycerides",
    "triglyceride": "triglycerides",
    # Kidney
    "creatinine": "creatinine",
    "serum creatinine": "creatinine",
    "urea": "urea",
    "blood urea": "urea",
    "bun": "urea",
    # Blood count
    "hemoglobin": "hemoglobin",
    "haemoglobin": "hemoglobin",
    "hgb": "hemoglobin",
    "wbc": "wbc",
    "white blood cells": "wbc",
    "rbc": "rbc",
    "red blood cells": "rbc",
    "platelets": "platelets",
    "platelet count": "platelets",
    # Thyroid
    "tsh": "tsh",
    "thyroid stimulating hormone": "tsh",
    # Vitamins
    "vitamin d": "vitamin_d",
    "25-hydroxyvitamin d": "vitamin_d",
    "vitamin b12": "vitamin_b12",
}

# Known reference ranges (lower bound, upper bound, unit)
REFERENCE_RANGES: dict[str, tuple[float, float, str]] = {
    "glucose": (70.0, 100.0, "mg/dL"),
    "hba1c": (4.0, 5.7, "%"),
    "cholesterol": (100.0, 200.0, "mg/dL"),
    "hdl": (40.0, 60.0, "mg/dL"),
    "ldl": (0.0, 100.0, "mg/dL"),
    "triglycerides": (0.0, 150.0, "mg/dL"),
    "creatinine": (0.6, 1.2, "mg/dL"),
    "urea": (7.0, 20.0, "mg/dL"),
    "hemoglobin": (12.0, 17.5, "g/dL"),
    "wbc": (4500.0, 11000.0, "/uL"),
    "rbc": (4.2, 5.9, "million/uL"),
    "platelets": (150000.0, 400000.0, "/uL"),
    "tsh": (0.4, 4.0, "mIU/L"),
    "vitamin_d": (30.0, 100.0, "ng/mL"),
    "vitamin_b12": (200.0, 900.0, "pg/mL"),
}


def standardise_test_name(raw_name: str) -> str:
    """Map a raw test name to its canonical form."""
    key = raw_name.strip().lower()
    return TEST_NAME_MAP.get(key, key)


def classify_abnormality(
    numeric_value: float | None,
    ref_low: float | None,
    ref_high: float | None,
) -> str:
    """Classify a numeric result against its reference range."""
    if numeric_value is None or ref_low is None or ref_high is None:
        return "unable_to_assess"
    if numeric_value < ref_low * 0.7:
        return "critically_low"
    if numeric_value < ref_low:
        return "low"
    if numeric_value > ref_high * 1.3:
        return "critically_high"
    if numeric_value > ref_high:
        return "high"
    return "normal"


@dataclass
class TrendResult:
    """Result of a trend calculation between two observations."""

    test_name: str
    earlier_value: float
    later_value: float
    absolute_change: float
    percentage_change: float
    trend_direction: str  # improving | worsening | stable | fluctuating
    is_sudden_change: bool = False
    is_persistent_abnormality: bool = False
    is_conflicting: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "earlier_value": self.earlier_value,
            "later_value": self.later_value,
            "absolute_change": round(self.absolute_change, 4),
            "percentage_change": round(self.percentage_change, 2),
            "trend_direction": self.trend_direction,
            "is_sudden_change": self.is_sudden_change,
            "is_persistent_abnormality": self.is_persistent_abnormality,
            "is_conflicting": self.is_conflicting,
        }


def calculate_trend(
    test_name: str,
    earlier_value: float,
    later_value: float,
    ref_low: float | None = None,
    ref_high: float | None = None,
) -> TrendResult:
    """Calculate the trend between two numeric observations."""
    canonical = standardise_test_name(test_name)
    abs_change = later_value - earlier_value
    avg = (abs(later_value) + abs(earlier_value)) / 2
    pct_change = (abs_change / avg * 100) if avg != 0 else 0.0

    # Determine direction
    threshold = max(abs(earlier_value) * 0.05, 0.5)
    if abs(abs_change) <= threshold:
        direction = "stable"
    elif abs_change > 0:
        direction = "worsening"  # values going up is usually worse
    else:
        direction = "improving"

    # Sudden change: >20% jump
    is_sudden = abs(pct_change) > 20

    # Persistent abnormality: both values outside range
    is_persistent = False
    if ref_low is not None and ref_high is not None:
        both_high = earlier_value > ref_high and later_value > ref_high
        both_low = earlier_value < ref_low and later_value < ref_low
        is_persistent = both_high or both_low

    # Conflicting: trend crosses the range (e.g., was high, now normal)
    is_conflicting = False
    if ref_low is not None and ref_high is not None:
        earlier_out = earlier_value < ref_low or earlier_value > ref_high
        later_normal = ref_low <= later_value <= ref_high
        is_conflicting = earlier_out and later_normal

    return TrendResult(
        test_name=canonical,
        earlier_value=earlier_value,
        later_value=later_value,
        absolute_change=abs_change,
        percentage_change=pct_change,
        trend_direction=direction,
        is_sudden_change=is_sudden,
        is_persistent_abnormality=is_persistent,
        is_conflicting=is_conflicting,
    )


def analyze_longitudinal_data(
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Analyze a list of observations for a single test over time.

    Args:
        observations: sorted by collection_date ascending. Each dict has:
            test_name, numeric_value, reference_range_low, reference_range_high,
            collection_date

    Returns:
        dict with trends, latest_status, summary, alerts
    """
    if not observations:
        return {
            "test_name": "",
            "trends": [],
            "latest_status": "not_tested",
            "summary": "No observations available.",
            "alerts": [],
        }

    test_name = standardise_test_name(observations[0].get("test_name", ""))
    ref = REFERENCE_RANGES.get(test_name)
    ref_low = observations[0].get("reference_range_low") or (ref[0] if ref else None)
    ref_high = observations[0].get("reference_range_high") or (ref[1] if ref else None)

    trends: list[TrendResult] = []
    for i in range(1, len(observations)):
        prev_val = observations[i - 1].get("numeric_value")
        curr_val = observations[i].get("numeric_value")
        if prev_val is None or curr_val is None:
            continue
        trend = calculate_trend(test_name, prev_val, curr_val, ref_low, ref_high)
        trends.append(trend)

    latest = observations[-1]
    latest_status = classify_abnormality(
        latest.get("numeric_value"), ref_low, ref_high
    )

    # Build alerts
    alerts: list[dict[str, str]] = []
    for t in trends:
        if t.is_sudden_change:
            alerts.append({
                "severity": "warning",
                "message": (
                    f"Sudden {t.percentage_change:+.1f}% change in {test_name} "
                    f"({t.earlier_value} → {t.later_value})"
                ),
            })
        if t.is_persistent_abnormality:
            alerts.append({
                "severity": "critical",
                "message": (
                    f"Persistent abnormal {test_name}: "
                    f"{t.earlier_value} → {t.later_value}"
                ),
            })

    # Summary (deterministic, no LLM)
    if latest_status in ("critically_high", "critically_low"):
        summary = (
            f"{test_name} is critically {'high' if 'high' in latest_status else 'low'} "
            f"at {latest.get('numeric_value')} — requires urgent attention."
        )
    elif latest_status in ("high", "low"):
        summary = (
            f"{test_name} is {'elevated' if latest_status == 'high' else 'low'} "
            f"at {latest.get('numeric_value')}. "
            f"{len(trends)} prior measurement(s) available."
        )
    else:
        summary = (
            f"{test_name} is within normal range at "
            f"{latest.get('numeric_value')}."
        )

    return {
        "test_name": test_name,
        "trends": [t.to_dict() for t in trends],
        "latest_status": latest_status,
        "latest_value": latest.get("numeric_value"),
        "reference_range": {"low": ref_low, "high": ref_high},
        "summary": summary,
        "alerts": alerts,
        "observation_count": len(observations),
    }
