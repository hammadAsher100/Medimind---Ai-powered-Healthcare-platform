"""Clinical Contradiction Detection — identifies conflicting data across sources."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Conflict:
    """A detected conflict between two data sources."""

    conflict_type: str
    severity: str
    first_source_label: str
    first_source_detail: str
    first_source_record_id: str
    second_source_label: str
    second_source_detail: str
    second_source_record_id: str
    explanation: str
    detection_method: str = "rule_based"

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflict_type": self.conflict_type,
            "severity": self.severity,
            "first_source_label": self.first_source_label,
            "first_source_detail": self.first_source_detail,
            "first_source_record_id": self.first_source_record_id,
            "second_source_label": self.second_source_label,
            "second_source_detail": self.second_source_detail,
            "second_source_record_id": self.second_source_record_id,
            "explanation": self.explanation,
            "detection_method": self.detection_method,
        }


def detect_value_discrepancies(
    observations: list[dict[str, Any]],
    threshold_pct: float = 20.0,
) -> list[Conflict]:
    """Detect duplicate observations with significantly different values."""
    conflicts: list[Conflict] = []

    # Group by standardised test name
    by_test: dict[str, list[dict]] = {}
    for obs in observations:
        name = obs.get("standardised_name", obs.get("test_name", "")).lower()
        val = obs.get("numeric_value")
        if name and val is not None:
            by_test.setdefault(name, []).append(obs)

    for test_name, obs_list in by_test.items():
        if len(obs_list) < 2:
            continue
        # Compare each pair
        for i in range(len(obs_list)):
            for j in range(i + 1, len(obs_list)):
                a, b = obs_list[i], obs_list[j]
                val_a = a.get("numeric_value")
                val_b = b.get("numeric_value")
                if val_a is None or val_b is None:
                    continue
                if val_a == 0:
                    continue

                pct_diff = abs(val_b - val_a) / abs(val_a) * 100
                if pct_diff >= threshold_pct:
                    conflicts.append(
                        Conflict(
                            conflict_type="value_discrepancy",
                            severity="warning",
                            first_source_label=a.get(
                                "source_label", f"Observation #{a.get('id', '?')}"
                            ),
                            first_source_detail=(
                                f"{test_name} = {val_a} "
                                f"{a.get('original_unit', '')}"
                            ),
                            first_source_record_id=str(a.get("id", "")),
                            second_source_label=b.get(
                                "source_label", f"Observation #{b.get('id', '?')}"
                            ),
                            second_source_detail=(
                                f"{test_name} = {val_b} "
                                f"{b.get('original_unit', '')}"
                            ),
                            second_source_record_id=str(b.get("id", "")),
                            explanation=(
                                f"{test_name} differs by {pct_diff:.1f}% "
                                f"between sources ({val_a} vs {val_b}). "
                                f"This may indicate a data entry error or "
                                f"legitimate change over time."
                            ),
                        )
                    )
    return conflicts


def detect_prediction_vs_input(
    predictions: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> list[Conflict]:
    """Detect when disease prediction inputs don't match recorded observations."""
    conflicts: list[Conflict] = []

    for pred in predictions:
        disease = pred.get("disease", "")
        input_data = pred.get("input_data", {})
        if not input_data:
            continue

        # Map prediction inputs to observation names
        input_obs_map = {
            "glucose": "glucose",
            "cholesterol": "cholesterol",
            "blood_pressure_systolic": "blood_pressure",
            "bmi": "bmi",
            "creatinine": "creatinine",
        }

        for input_key, obs_name in input_obs_map.items():
            input_val = input_data.get(input_key)
            if input_val is None:
                continue

            # Find matching observation
            matching = [
                o for o in observations
                if obs_name in o.get("standardised_name", "").lower()
                or obs_name in o.get("test_name", "").lower()
            ]
            if not matching:
                continue

            latest = matching[-1]
            obs_val = latest.get("numeric_value")
            if obs_val is None:
                continue

            # Allow 10% tolerance for rounding / different units
            if abs(float(input_val) - obs_val) / max(abs(obs_val), 0.01) > 0.1:
                conflicts.append(
                    Conflict(
                        conflict_type="prediction_vs_input",
                        severity="info",
                        first_source_label=f"{disease} prediction input",
                        first_source_detail=(
                            f"{input_key} = {input_val}"
                        ),
                        first_source_record_id=str(pred.get("id", "")),
                        second_source_label="Lab observation",
                        second_source_detail=(
                            f"{obs_name} = {obs_val} "
                            f"({latest.get('original_unit', '')})"
                        ),
                        second_source_record_id=str(
                            latest.get("id", "")
                        ),
                        explanation=(
                            f"The {disease} prediction used "
                            f"{input_key}={input_val} but the latest "
                            f"recorded {obs_name} is {obs_val}."
                        ),
                    )
                )
    return conflicts


def detect_timeline_vs_record(
    timeline_events: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    reports: list[dict[str, Any]],
) -> list[Conflict]:
    """Detect timeline events that reference missing or inconsistent records."""
    conflicts: list[Conflict] = []

    pred_ids = {str(p.get("id", "")) for p in predictions}
    report_ids = {str(r.get("id", "")) for r in reports}

    for event in timeline_events:
        metadata = event.get("metadata", {})
        if not isinstance(metadata, dict):
            continue

        ref_id = str(metadata.get("prediction_id", ""))
        if ref_id and ref_id not in pred_ids:
            conflicts.append(
                Conflict(
                    conflict_type="timeline_vs_record",
                    severity="warning",
                    first_source_label="Timeline event",
                    first_source_detail=(
                        f"References prediction #{ref_id}"
                    ),
                    first_source_record_id=str(event.get("id", "")),
                    second_source_label="Predictions",
                    second_source_detail=(
                        f"Prediction #{ref_id} not found in records"
                    ),
                    second_source_record_id=ref_id,
                    explanation=(
                        f"Timeline event '{event.get('title', '')}' "
                        f"references prediction #{ref_id} which does "
                        f"not exist in the prediction records."
                    ),
                )
            )
    return conflicts


def run_conflict_detection(
    observations: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    reports: list[dict[str, Any]],
    timeline_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run all conflict detection checks."""
    all_conflicts: list[Conflict] = []

    all_conflicts.extend(detect_value_discrepancies(observations))
    all_conflicts.extend(
        detect_prediction_vs_input(predictions, observations)
    )
    all_conflicts.extend(
        detect_timeline_vs_record(timeline_events, predictions, reports)
    )

    critical = sum(1 for c in all_conflicts if c.severity == "critical")
    warnings = sum(1 for c in all_conflicts if c.severity == "warning")

    return {
        "conflicts": [c.to_dict() for c in all_conflicts],
        "total_conflicts": len(all_conflicts),
        "critical_count": critical,
        "warning_count": warnings,
        "info_count": len(all_conflicts) - critical - warnings,
    }
