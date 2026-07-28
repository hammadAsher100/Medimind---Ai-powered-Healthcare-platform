"""Counterfactual Health Simulator — 'what if' health scenario modeling."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Known impact factors: how changing a factor affects disease risk
# These are simplified approximations for educational purposes
# In production, use validated clinical risk calculators
IMPACT_FACTORS: dict[str, dict[str, dict[str, float]]] = {
    "diabetes": {
        "glucose": {"per_unit_above_100": 0.8, "per_unit_below_100": -0.3},
        "bmi": {"per_unit_above_25": 0.6, "per_unit_below_25": -0.2},
        "exercise": {"add_30min_weekly": -3.0, "remove_30min_weekly": 2.0},
        "smoking": {"quit": -2.0, "start": 3.0},
    },
    "heart": {
        "cholesterol": {"per_10mg_above_200": 1.5, "per_10mg_below_200": -0.5},
        "blood_pressure": {"per_10mmhg_above_140": 2.0, "per_10mmhg_below_140": -0.8},
        "exercise": {"add_30min_weekly": -4.0, "remove_30min_weekly": 3.0},
        "smoking": {"quit": -5.0, "start": 6.0},
    },
    "kidney": {
        "creatinine": {"per_0.5mg_above_1.2": 5.0, "per_0.5mg_below_1.2": -1.0},
        "blood_pressure": {"per_10mmhg_above_140": 1.5, "per_10mmhg_below_140": -0.5},
        "glucose": {"per_unit_above_100": 0.5, "per_unit_below_100": -0.2},
    },
    "stroke": {
        "blood_pressure": {"per_10mmhg_above_140": 3.0, "per_10mmhg_below_140": -1.5},
        "smoking": {"quit": -4.0, "start": 5.0},
        "bmi": {"per_unit_above_25": 0.8, "per_unit_below_25": -0.3},
    },
}


@dataclass
class CounterfactualResult:
    """Result of a counterfactual simulation."""

    original_risk: dict[str, float]
    modified_risk: dict[str, float]
    changes_applied: list[dict[str, Any]]
    risk_deltas: dict[str, float]
    recommendations: list[str]
    is_synthetic_data: bool = True
    disclaimer: str = (
        "This is a simplified educational simulation. "
        "Actual health outcomes depend on many complex factors "
        "not captured here. Always consult a healthcare provider "
        "for medical decisions."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_risk": self.original_risk,
            "modified_risk": self.modified_risk,
            "changes_applied": self.changes_applied,
            "risk_deltas": self.risk_deltas,
            "recommendations": self.recommendations,
            "is_synthetic_data": self.is_synthetic_data,
            "disclaimer": self.disclaimer,
        }


def simulate_counterfactual(
    original_risk: dict[str, float],
    modifications: list[dict[str, Any]],
    user_profile: dict[str, Any] | None = None,
) -> CounterfactualResult:
    """Simulate health risks under modified conditions.

    Args:
        original_risk: {"diabetes": 45.2, "heart": 30.1, ...} — current risks
        modifications: list of changes like
            [{"factor": "smoking", "change": "quit"},
             {"factor": "exercise", "change": "add_30min_weekly"},
             {"factor": "glucose", "current": 140, "target": 100}]
        user_profile: optional current patient profile

    Returns:
        CounterfactualResult with original/modified risks and recommendations
    """
    modified_risk = dict(original_risk)
    changes_applied: list[dict[str, Any]] = []

    for mod in modifications:
        factor = mod.get("factor", "").lower()
        change = mod.get("change", "")
        current_val = mod.get("current")
        target_val = mod.get("target")

        # Find which disease models this factor affects
        for disease, factors in IMPACT_FACTORS.items():
            if disease not in modified_risk:
                continue
            if factor not in factors:
                continue

            impact_rules = factors[factor]
            delta = 0.0

            if change in ("quit", "start"):
                delta = impact_rules.get(change, 0.0)
            elif change.startswith("add_") or change.startswith("remove_"):
                delta = impact_rules.get(change, 0.0)
            elif current_val is not None and target_val is not None:
                diff = target_val - current_val
                if diff > 0:
                    # Target is above current — use "above" rule
                    key = next(
                        (k for k in impact_rules if k.startswith("per_") and "above" in k),
                        None,
                    )
                    if key:
                        import re
                        m = re.search(r"per_(\d+\.?\d*)", key)
                        per_unit = float(m.group(1)) if m else 1.0
                        delta = (abs(diff) / per_unit) * impact_rules[key]
                elif diff < 0:
                    # Target is below current — use "below" rule
                    key = next(
                        (k for k in impact_rules if k.startswith("per_") and "below" in k),
                        None,
                    )
                    if key:
                        import re
                        m = re.search(r"per_(\d+\.?\d*)", key)
                        per_unit = float(m.group(1)) if m else 1.0
                        delta = (abs(diff) / per_unit) * impact_rules[key]

            modified_risk[disease] = max(
                0.0, min(100.0, modified_risk[disease] + delta)
            )
            changes_applied.append({
                "factor": factor,
                "change": change,
                "current": current_val,
                "target": target_val,
                "disease": disease,
                "delta": round(delta, 2),
            })

    # Calculate deltas
    risk_deltas = {
        d: round(modified_risk.get(d, 0) - original_risk.get(d, 0), 2)
        for d in original_risk
    }

    # Generate recommendations
    recommendations = _generate_recommendations(
        original_risk, modified_risk, risk_deltas, modifications
    )

    return CounterfactualResult(
        original_risk=original_risk,
        modified_risk=modified_risk,
        changes_applied=changes_applied,
        risk_deltas=risk_deltas,
        recommendations=recommendations,
    )


def _generate_recommendations(
    original: dict[str, float],
    modified: dict[str, float],
    deltas: dict[str, float],
    modifications: list[dict[str, Any]],
) -> list[str]:
    """Generate actionable recommendations from simulation results."""
    recs: list[str] = []

    # Improvements
    improving = {d: d for d, v in deltas.items() if v < -2}
    if improving:
        for disease in improving:
            recs.append(
                f"Changes would reduce {disease} risk by "
                f"{abs(deltas[disease]):.1f}%"
            )

    # Worsening
    worsening = {d: d for d, v in deltas.items() if v > 2}
    if worsening:
        for disease in worsening:
            recs.append(
                f"Warning: Changes would increase {disease} risk by "
                f"{deltas[disease]:.1f}%"
            )

    # High residual risk
    for disease, risk in modified.items():
        if risk >= 70:
            recs.append(
                f"⚠ {disease} risk remains HIGH ({risk:.1f}%) — "
                f"additional intervention needed"
            )
        elif risk >= 50:
            recs.append(
                f"{disease} risk moderate ({risk:.1f}%) — "
                f"continued monitoring recommended"
            )

    # Specific factor recommendations
    change_types = {c.get("factor", "") for c in modifications}
    if "smoking" in change_types:
        recs.append(
            "Smoking cessation provides the single largest risk "
            "reduction across multiple diseases"
        )
    if "exercise" in change_types:
        recs.append(
            "Regular physical activity (150 min/week moderate) "
            "is associated with lower cardiovascular and metabolic risk"
        )

    if not recs:
        recs.append(
            "The proposed changes have minimal impact on calculated risk. "
            "Consider discussing broader lifestyle modifications with "
            "your healthcare provider."
        )

    return recs
