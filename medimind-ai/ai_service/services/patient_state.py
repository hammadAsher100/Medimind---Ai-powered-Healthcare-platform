"""Patient State Intelligence Engine — aggregates all data into a unified snapshot."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .lab_intelligence import analyze_longitudinal_data, standardise_test_name

logger = logging.getLogger(__name__)

# Emergency keywords that override all routing
EMERGENCY_KEYWORDS = [
    "chest pain",
    "difficulty breathing",
    "shortness of breath",
    "severe bleeding",
    "unconscious",
    "seizure",
    "stroke symptoms",
    "heart attack",
    "allergic reaction",
    "anaphylaxis",
    "overdose",
    "suicidal",
    "cannot breathe",
    "choking",
    "severe head injury",
]


def check_emergency_keywords(text: str) -> dict[str, Any] | None:
    """Check text for emergency keywords. Returns triage result or None."""
    lower_text = text.lower()
    for kw in EMERGENCY_KEYWORDS:
        if kw in lower_text:
            return {
                "is_emergency": True,
                "matched_keyword": kw,
                "priority_level": "emergency",
                "message": (
                    "EMERGENCY DETECTED: If you or someone is experiencing "
                    "a medical emergency, call your local emergency number "
                    "immediately (e.g., 911, 999, 112). This system cannot "
                    "provide emergency care."
                ),
                "disclaimer": (
                    "This is an automated detection. Always seek professional "
                    "medical help in emergencies."
                ),
            }
    return None


@dataclass
class PatientState:
    """Aggregated patient state at a point in time."""

    user_id: int
    priority_level: str = "routine"
    critical_findings: list[str] = field(default_factory=list)
    recent_changes: list[str] = field(default_factory=list)
    active_risks: list[str] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)
    contradictory_evidence: list[str] = field(default_factory=list)
    medication_concerns: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    uncertainty_notes: list[str] = field(default_factory=list)
    suggested_next_steps: list[str] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
    snapshot_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "priority_level": self.priority_level,
            "critical_findings": self.critical_findings,
            "recent_changes": self.recent_changes,
            "active_risks": self.active_risks,
            "supporting_evidence": self.supporting_evidence,
            "contradictory_evidence": self.contradictory_evidence,
            "medication_concerns": self.medication_concerns,
            "missing_information": self.missing_information,
            "uncertainty_notes": self.uncertainty_notes,
            "suggested_next_steps": self.suggested_next_steps,
            "data_sources": self.data_sources,
            "snapshot_data": self.snapshot_data,
        }


def build_patient_state(
    user_id: int,
    *,
    observations: list[dict[str, Any]] | None = None,
    predictions: list[dict[str, Any]] | None = None,
    medications: list[dict[str, Any]] | None = None,
    allergies: list[dict[str, Any]] | None = None,
    reports: list[dict[str, Any]] | None = None,
    health_score: dict[str, Any] | None = None,
    medical_profile: dict[str, Any] | None = None,
) -> PatientState:
    """Build a comprehensive patient state from all available data sources.

    All deterministic — no LLM calls. This is the data aggregation layer.
    LLM can add natural-language explanations on top later.
    """
    state = PatientState(user_id=user_id)

    if observations:
        state.data_sources.append("lab_observations")
        _analyze_observations(state, observations)
    if predictions:
        state.data_sources.append("disease_predictions")
        _analyze_predictions(state, predictions)
    if medications:
        state.data_sources.append("medications")
        _analyze_medications(state, medications, allergies or [])
    if reports:
        state.data_sources.append("medical_reports")
        _analyze_reports(state, reports)
    if health_score:
        state.data_sources.append("health_score")
        _analyze_health_score(state, health_score)
    if medical_profile:
        state.data_sources.append("medical_profile")
        _analyze_profile(state, medical_profile)

    # Missing information detection
    _detect_missing_info(
        state, observations, predictions, medications, medical_profile
    )

    # Determine priority from critical findings
    if state.critical_findings:
        state.priority_level = "urgent"
    elif len(state.active_risks) >= 3:
        state.priority_level = "review_today"
    elif state.recent_changes:
        state.priority_level = "review_soon"
    else:
        state.priority_level = "routine"

    return state


def _analyze_observations(
    state: PatientState, observations: list[dict[str, Any]]
) -> None:
    """Analyze lab observations for trends and abnormalities."""
    # Group by test_name
    grouped: dict[str, list[dict]] = {}
    for obs in observations:
        name = standardise_test_name(obs.get("test_name", ""))
        grouped.setdefault(name, []).append(obs)

    for test_name, obs_list in grouped.items():
        # Sort by collection_date
        obs_list.sort(key=lambda x: x.get("collection_date", ""))
        result = analyze_longitudinal_data(obs_list)

        if result.get("latest_status") in (
            "critically_high",
            "critically_low",
        ):
            state.critical_findings.append(
                f"{test_name}: critically "
                f"{'high' if 'high' in result['latest_status'] else 'low'} "
                f"at {result.get('latest_value')}"
            )

        if result.get("latest_status") in ("high", "low"):
            state.active_risks.append(
                f"{test_name}: {result['latest_status']} "
                f"at {result.get('latest_value')}"
            )

        for alert in result.get("alerts", []):
            if alert.get("severity") == "critical":
                state.critical_findings.append(alert["message"])
            else:
                state.recent_changes.append(alert["message"])


def _analyze_predictions(
    state: PatientState, predictions: list[dict[str, Any]]
) -> None:
    """Analyze disease risk predictions."""
    for pred in predictions:
        disease = pred.get("disease", "unknown")
        risk = pred.get("risk_percentage", 0)
        risk_level = pred.get("risk_level", "")

        if risk_level == "high" or risk >= 70:
            state.active_risks.append(
                f"High risk of {disease} ({risk:.1f}%)"
            )
        elif risk_level == "moderate" or risk >= 40:
            state.recent_changes.append(
                f"Moderate risk of {disease} ({risk:.1f}%)"
            )

        # Check if SHAP shows specific risk factors
        shap = pred.get("shap_explanation", {})
        if isinstance(shap, dict):
            top_factors = shap.get("top_features", [])
            for factor in top_factors[:2]:
                if isinstance(factor, dict):
                    state.supporting_evidence.append(
                        f"{disease} risk factor: {factor.get('feature', 'unknown')}"
                    )


def _analyze_medications(
    state: PatientState,
    medications: list[dict[str, Any]],
    allergies: list[dict[str, Any]],
) -> None:
    """Analyze medications for safety concerns."""
    active = [m for m in medications if m.get("status") == "active"]

    if len(active) > 5:
        state.medication_concerns.append(
            f"Patient is on {len(active)} active medications — "
            f"polypharmacy risk"
        )

    # Cross-check active meds against allergies
    allergy_set = {a.get("allergen", "").lower() for a in allergies}
    for med in active:
        med_name = med.get("medication_name", "").lower()
        for allergy in allergy_set:
            if allergy in med_name or med_name in allergy:
                state.critical_findings.append(
                    f"Allergy conflict: {med.get('medication_name')} vs "
                    f"known allergy to {allergy}"
                )

    # Missing medication info
    for med in active:
        if not med.get("dosage") or not med.get("frequency"):
            state.medication_concerns.append(
                f"{med.get('medication_name', 'Unknown')}: "
                f"missing dosage or frequency info"
            )


def _analyze_reports(
    state: PatientState, reports: list[dict[str, Any]]
) -> None:
    """Analyze medical reports for findings."""
    for report in reports:
        summary = report.get("summary", "")
        if summary:
            lower = summary.lower()
            for keyword in ["critical", "urgent", "abnormal", "elevated"]:
                if keyword in lower:
                    state.recent_changes.append(
                        f"Report '{report.get('title', 'untitled')}': "
                        f"contains '{keyword}' finding"
                    )
                    break


def _analyze_health_score(
    state: PatientState, score: dict[str, Any]
) -> None:
    """Analyze health score for concerns."""
    val = score.get("score", 100)
    if val < 40:
        state.active_risks.append(f"Low health score: {val}/100")
    elif val < 60:
        state.recent_changes.append(f"Health score below average: {val}/100")

    # Check individual components
    if score.get("sugar_level") and float(score["sugar_level"]) > 126:
        state.active_risks.append("Elevated blood sugar")
    if score.get("cholesterol") and float(score["cholesterol"]) > 240:
        state.active_risks.append("High cholesterol")


def _analyze_profile(
    state: PatientState, profile: dict[str, Any]
) -> None:
    """Analyze medical profile for static risk factors."""
    smoking = profile.get("smoking_status", "")
    if smoking == "current":
        state.active_risks.append("Current smoker — cardiovascular risk")

    bmi = profile.get("bmi")
    if bmi:
        if bmi > 30:
            state.active_risks.append(f"BMI {bmi:.1f} — obesity category")
        elif bmi < 18.5:
            state.active_risks.append(f"BMI {bmi:.1f} — underweight")


def _detect_missing_info(
    state: PatientState,
    observations: list[dict[str, Any]] | None,
    predictions: list[dict[str, Any]] | None,
    medications: list[dict[str, Any]] | None,
    medical_profile: dict[str, Any] | None,
) -> None:
    """Detect commonly expected data that is missing."""
    critical_tests = ["glucose", "hba1c", "cholesterol", "creatinine"]
    if observations:
        present = {
            standardise_test_name(o.get("test_name", ""))
            for o in observations
        }
        for test in critical_tests:
            if test not in present:
                state.missing_information.append(
                    f"No {test} measurement in record"
                )
    else:
        state.missing_information.append("No laboratory observations recorded")

    if not predictions:
        state.missing_information.append("No disease risk assessments performed")

    if medical_profile:
        if not medical_profile.get("blood_type"):
            state.missing_information.append("Blood type not recorded")
        if not medical_profile.get("height_cm"):
            state.missing_information.append("Height not recorded")
