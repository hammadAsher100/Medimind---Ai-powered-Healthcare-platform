"""Medication Safety Passport — interaction checks, allergy screening, alerts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Known significant drug-drug interactions (simplified)
# In production, use RxNorm / openFDA / DrugBank
DRUG_INTERACTIONS: dict[tuple[str, str], dict[str, str]] = {
    ("warfarin", "aspirin"): {
        "severity": "critical",
        "description": (
            "Increased bleeding risk when combined. Monitor INR closely."
        ),
    },
    ("metformin", "alcohol"): {
        "severity": "warning",
        "description": (
            "Alcohol increases risk of lactic acidosis with metformin."
        ),
    },
    ("lisinopril", "potassium"): {
        "severity": "warning",
        "description": (
            "ACE inhibitors can raise potassium levels. "
            "Monitor serum potassium."
        ),
    },
    ("atorvastatin", "grapefruit"): {
        "severity": "info",
        "description": (
            "Grapefruit can increase statin levels. "
            "Limit grapefruit intake."
        ),
    },
    ("clopidogrel", "omeprazole"): {
        "severity": "warning",
        "description": (
            "Proton pump inhibitors may reduce antiplatelet effect "
            "of clopidogrel."
        ),
    },
    ("ibuprofen", "warfarin"): {
        "severity": "critical",
        "description": (
            "NSAIDs increase bleeding risk with anticoagulants."
        ),
    },
    ("lithium", "ibuprofen"): {
        "severity": "critical",
        "description": (
            "NSAIDs can increase lithium levels to toxic range."
        ),
    },
    ("ssri", "tramadol"): {
        "severity": "critical",
        "description": (
            "Risk of serotonin syndrome. Avoid combination."
        ),
    },
}

# Drug-allergy cross-reactivity patterns
ALLERGY_CROSS_REACT: dict[str, list[str]] = {
    "penicillin": ["amoxicillin", "ampicillin", "piperacillin"],
    "sulfonamide": ["sulfamethoxazole", "sulfadiazine", "celecoxib"],
    "nsaid": ["aspirin", "ibuprofen", "naproxen", "diclofenac"],
    "aspirin": ["ibuprofen", "naproxen"],
}


@dataclass
class MedicationSafetyAlert:
    """A safety alert for a patient's medication regimen."""

    alert_type: str
    severity: str  # info, warning, critical
    title: str
    description: str
    medications_involved: list[str] = field(default_factory=list)
    is_synthetic_data: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "medications_involved": self.medications_involved,
            "is_synthetic_data": self.is_synthetic_data,
        }


def check_drug_interactions(
    medications: list[dict[str, Any]],
) -> list[MedicationSafetyAlert]:
    """Check a medication list for known drug-drug interactions."""
    alerts: list[MedicationSafetyAlert] = []
    active_names = [
        m.get("medication_name", "").lower()
        for m in medications
        if m.get("status") == "active"
    ]

    for i, name_a in enumerate(active_names):
        for name_b in active_names[i + 1:]:
            # Check both orderings
            pair = (name_a, name_b)
            pair_rev = (name_b, name_a)
            interaction = DRUG_INTERACTIONS.get(pair) or DRUG_INTERACTIONS.get(
                pair_rev
            )
            if interaction:
                alerts.append(
                    MedicationSafetyAlert(
                        alert_type="drug_interaction",
                        severity=interaction["severity"],
                        title=f"Interaction: {name_a} + {name_b}",
                        description=interaction["description"],
                        medications_involved=[name_a, name_b],
                    )
                )
    return alerts


def check_allergy_conflicts(
    medications: list[dict[str, Any]],
    allergies: list[dict[str, Any]],
) -> list[MedicationSafetyAlert]:
    """Check medications against known allergies."""
    alerts: list[MedicationSafetyAlert] = []
    allergy_terms = {
        a.get("allergen", "").lower() for a in allergies
    }
    active_meds = [
        m for m in medications if m.get("status") == "active"
    ]

    for med in active_meds:
        med_name = med.get("medication_name", "").lower()

        # Direct allergy match
        for allergy in allergy_terms:
            if allergy in med_name or med_name in allergy:
                alerts.append(
                    MedicationSafetyAlert(
                        alert_type="allergy_conflict",
                        severity="critical",
                        title=f"Allergy: {med_name} vs {allergy}",
                        description=(
                            f"Patient has recorded allergy to '{allergy}'. "
                            f"Medication '{med_name}' may cause a reaction."
                        ),
                        medications_involved=[med_name],
                    )
                )

        # Cross-reactivity check
        for allergy in allergy_terms:
            cross_matches = ALLERGY_CROSS_REACT.get(allergy, [])
            for cross in cross_matches:
                if cross in med_name:
                    alerts.append(
                        MedicationSafetyAlert(
                            alert_type="allergy_conflict",
                            severity="warning",
                            title=(
                                f"Cross-reactivity: {med_name} "
                                f"(allergy to {allergy})"
                            ),
                            description=(
                                f"Patient is allergic to '{allergy}' which "
                                f"may cross-react with '{med_name}'."
                            ),
                            medications_involved=[med_name, allergy],
                        )
                    )
    return alerts


def check_duplicate_therapy(
    medications: list[dict[str, Any]],
) -> list[MedicationSafetyAlert]:
    """Check for duplicate therapeutic classes."""
    alerts: list[MedicationSafetyAlert] = []
    active = [
        m for m in medications if m.get("status") == "active"
    ]

    # Group by drug_class
    by_class: dict[str, list[str]] = {}
    for med in active:
        drug_class = med.get("drug_class", "").lower()
        if drug_class:
            by_class.setdefault(drug_class, []).append(
                med.get("medication_name", "")
            )

    for drug_class, meds in by_class.items():
        if len(meds) > 1:
            alerts.append(
                MedicationSafetyAlert(
                    alert_type="duplicate_therapy",
                    severity="warning",
                    title=f"Duplicate therapy: {drug_class}",
                    description=(
                        f"Multiple medications in class '{drug_class}': "
                        f"{', '.join(meds)}. Review for necessity."
                    ),
                    medications_involved=meds,
                )
            )
    return alerts


def run_safety_check(
    medications: list[dict[str, Any]],
    allergies: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run the full medication safety check.

    Returns:
        dict with alerts, medication_count, critical_count, etc.
    """
    all_alerts: list[MedicationSafetyAlert] = []

    all_alerts.extend(check_drug_interactions(medications))
    all_alerts.extend(check_allergy_conflicts(medications, allergies))
    all_alerts.extend(check_duplicate_therapy(medications))

    critical_count = sum(
        1 for a in all_alerts if a.severity == "critical"
    )
    warning_count = sum(
        1 for a in all_alerts if a.severity == "warning"
    )

    return {
        "alerts": [a.to_dict() for a in all_alerts],
        "total_alerts": len(all_alerts),
        "critical_count": critical_count,
        "warning_count": warning_count,
        "medication_count": len(
            [m for m in medications if m.get("status") == "active"]
        ),
        "disclaimer": (
            "This is an automated safety screening. It does not replace "
            "professional pharmacist or physician review. Always consult "
            "a qualified healthcare provider before making medication "
            "decisions."
        ),
    }


def extract_medications_from_text(
    text: str,
) -> list[dict[str, Any]]:
    """Extract medication names from free text using pattern matching.

    This is a deterministic extraction — no LLM. For full NLP extraction,
    use the LLM-based report analyzer.
    """
    import re

    med_pattern = re.compile(
        r"(?:take|taking|prescribed|started on|discontinued)\s+"
        r"(\w+(?:\s+\w+)?)",
        re.IGNORECASE,
    )
    matches = med_pattern.findall(text)

    # Deduplicate
    seen = set()
    results = []
    for name in matches:
        name_lower = name.lower().strip()
        if name_lower not in seen:
            seen.add(name_lower)
            results.append({
                "medication_name": name.strip(),
                "status": "active",
                "source": "text_extraction",
                "confidence": 0.6,  # pattern matching is low confidence
            })
    return results
