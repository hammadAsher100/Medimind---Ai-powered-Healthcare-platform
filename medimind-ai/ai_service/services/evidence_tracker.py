"""Evidence-Linked Medical Assistant — tracks provenance for AI responses."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvidenceSource:
    """A single evidence source supporting an AI claim."""

    source_type: str  # prediction, observation, report, knowledge, medication
    source_id: str
    label: str
    detail: str
    confidence: float = 1.0  # 0–1
    is_primary: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "label": self.label,
            "detail": self.detail,
            "confidence": round(self.confidence, 2),
            "is_primary": self.is_primary,
        }


@dataclass
class EvidenceLinkedResponse:
    """An AI response with tracked evidence sources."""

    response_text: str
    evidence_sources: list[EvidenceSource] = field(default_factory=list)
    disclaimers: list[str] = field(default_factory=list)
    response_type: str = "general"  # general, diagnosis, medication, emergency

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_text": self.response_text,
            "evidence_sources": [e.to_dict() for e in self.evidence_sources],
            "disclaimers": self.disclaimers,
            "response_type": self.response_type,
            "is_synthetic_data": False,
        }


# Standard disclaimers for different response types
DISCLAIMERS: dict[str, list[str]] = {
    "general": [
        "This information is for educational purposes only and does "
        "not constitute medical advice.",
    ],
    "diagnosis": [
        "This analysis is for decision support only. All findings "
        "must be confirmed by a qualified clinician.",
        "AI-generated assessments may contain errors and should not "
        "replace professional clinical judgment.",
    ],
    "medication": [
        "Medication information is for reference only. Always verify "
        "with a pharmacist or physician before making changes.",
        "Drug interaction checks are based on a limited database. "
        "Complete interactions may not be detected.",
    ],
    "emergency": [
        "If this is a medical emergency, call your local emergency "
        "number immediately.",
        "Automated triage is not a substitute for professional "
        "emergency assessment.",
    ],
}


def build_evidence_response(
    response_text: str,
    response_type: str = "general",
    patient_context: dict[str, Any] | None = None,
) -> EvidenceLinkedResponse:
    """Build an evidence-linked response with proper provenance tracking.

    Args:
        response_text: The AI-generated response text
        response_type: Type of response (general, diagnosis, medication, emergency)
        patient_context: Patient data used in generating the response

    Returns:
        EvidenceLinkedResponse with tracked sources and disclaimers
    """
    evidence: list[EvidenceSource] = []

    if patient_context:
        # Track which data sources were used
        if patient_context.get("observations"):
            for obs in patient_context["observations"][:5]:
                evidence.append(
                    EvidenceSource(
                        source_type="observation",
                        source_id=str(obs.get("id", "")),
                        label=f"{obs.get('test_name', 'test')}: "
                        f"{obs.get('numeric_value', 'N/A')}",
                        detail=(
                            f"Collected: {obs.get('collection_date', 'unknown')}"
                        ),
                        confidence=obs.get("extraction_confidence", 1.0),
                    )
                )

        if patient_context.get("predictions"):
            for pred in patient_context["predictions"][:3]:
                evidence.append(
                    EvidenceSource(
                        source_type="prediction",
                        source_id=str(pred.get("id", "")),
                        label=(
                            f"{pred.get('disease', 'disease')}: "
                            f"{pred.get('risk_percentage', 0):.1f}%"
                        ),
                        detail=pred.get("risk_level", ""),
                        is_primary=True,
                    )
                )

        if patient_context.get("medications"):
            for med in patient_context["medications"][:5]:
                evidence.append(
                    EvidenceSource(
                        source_type="medication",
                        source_id=str(med.get("id", "")),
                        label=med.get("medication_name", "unknown"),
                        detail=(
                            f"{med.get('dosage', '')} "
                            f"{med.get('frequency', '')}"
                        ),
                    )
                )

        if patient_context.get("reports"):
            for report in patient_context["reports"][:3]:
                evidence.append(
                    EvidenceSource(
                        source_type="report",
                        source_id=str(report.get("id", "")),
                        label=report.get("title", "Medical report"),
                        detail=report.get("report_type", ""),
                    )
                )

    # Attach standard disclaimers
    disclaimers = list(DISCLAIMERS.get(response_type, DISCLAIMERS["general"]))

    return EvidenceLinkedResponse(
        response_text=response_text,
        evidence_sources=evidence,
        disclaimers=disclaimers,
        response_type=response_type,
    )


def format_evidence_citations(
    evidence: list[EvidenceSource],
) -> str:
    """Format evidence sources as numbered citations."""
    if not evidence:
        return ""

    lines = ["**Evidence Sources:**"]
    for i, e in enumerate(evidence, 1):
        primary = " ★" if e.is_primary else ""
        lines.append(
            f"[{i}] {e.label} — {e.detail}{primary}"
        )

    lines.append("")
    lines.append(
        "★ = Primary source used in this assessment. "
        "All sources should be independently verified."
    )
    return "\n".join(lines)
