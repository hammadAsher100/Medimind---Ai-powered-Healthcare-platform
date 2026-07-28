"""Tests for clinical intelligence FastAPI services."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AI_SERVICE_DIR = PROJECT_ROOT / "ai_service"
if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))


# ── Lab Intelligence Tests ─────────────────────────────────────────────────────

class TestLabIntelligence:
    @pytest.fixture(autouse=True)
    def setup(self):
        from services import lab_intelligence
        self.li = lab_intelligence

    def test_standardise_test_name(self):
        assert self.li.standardise_test_name("Fasting Blood Glucose") == "glucose"
        assert self.li.standardise_test_name("HDL Cholesterol") == "hdl"
        assert self.li.standardise_test_name("Unknown Test") == "unknown test"

    def test_classify_abnormality_normal(self):
        status = self.li.classify_abnormality(90.0, 70.0, 100.0)
        assert status == "normal"

    def test_classify_abnormality_high(self):
        status = self.li.classify_abnormality(110.0, 70.0, 100.0)
        assert status == "high"

    def test_classify_abnormality_low(self):
        status = self.li.classify_abnormality(50.0, 70.0, 100.0)
        assert status == "low"

    def test_classify_abnormality_critically_high(self):
        status = self.li.classify_abnormality(150.0, 70.0, 100.0)
        assert status == "critically_high"

    def test_classify_abnormality_critically_low(self):
        status = self.li.classify_abnormality(40.0, 70.0, 100.0)
        assert status == "critically_low"

    def test_calculate_trend_stable(self):
        trend = self.li.calculate_trend("glucose", 95.0, 97.0, 70.0, 100.0)
        assert trend.trend_direction == "stable"

    def test_calculate_trend_worsening(self):
        trend = self.li.calculate_trend("glucose", 95.0, 145.0, 70.0, 100.0)
        assert trend.trend_direction == "worsening"

    def test_calculate_trend_improving(self):
        trend = self.li.calculate_trend("glucose", 145.0, 100.0, 70.0, 100.0)
        assert trend.trend_direction == "improving"

    def test_calculate_trend_sudden_change(self):
        trend = self.li.calculate_trend("glucose", 100.0, 150.0)
        assert trend.is_sudden_change

    def test_calculate_trend_persistent_abnormality(self):
        trend = self.li.calculate_trend("glucose", 110.0, 120.0, 70.0, 100.0)
        assert trend.is_persistent_abnormality

    def test_analyze_longitudinal_data_empty(self):
        result = self.li.analyze_longitudinal_data([])
        assert result["latest_status"] == "not_tested"

    def test_analyze_longitudinal_single(self):
        result = self.li.analyze_longitudinal_data([
            {"test_name": "Glucose", "numeric_value": 95.0, "collection_date": "2025-01-01"},
        ])
        assert result["observation_count"] == 1
        assert result["latest_status"] in ("normal", "unable_to_assess")


# ── Patient State Tests ───────────────────────────────────────────────────────

class TestPatientState:
    @pytest.fixture(autouse=True)
    def setup(self):
        from services import patient_state
        self.ps = patient_state

    def test_emergency_keyword_detection(self):
        result = self.ps.check_emergency_keywords("I have chest pain and difficulty breathing")
        assert result is not None
        assert result["is_emergency"]
        assert result["priority_level"] == "emergency"

    def test_no_emergency_normal_text(self):
        result = self.ps.check_emergency_keywords("I have a mild headache")
        assert result is None

    def test_emergency_detected_priority(self):
        result = self.ps.check_emergency_keywords("shortness of breath")
        assert result is not None
        assert result["matched_keyword"] == "shortness of breath"

    def test_build_patient_state_without_data(self):
        state = self.ps.build_patient_state(1)
        assert state.priority_level == "routine"
        assert state.data_sources == []

    def test_build_state_with_observations(self):
        state = self.ps.build_patient_state(
            1,
            observations=[{
                "test_name": "Glucose",
                "numeric_value": 250.0,
                "reference_range_low": 70.0,
                "reference_range_high": 100.0,
                "collection_date": "2025-01-01",
            }],
        )
        assert len(state.critical_findings) >= 1
        assert state.priority_level == "urgent"

    def test_build_state_with_predictions(self):
        state = self.ps.build_patient_state(
            1,
            predictions=[{
                "disease": "diabetes",
                "risk_percentage": 85.0,
                "risk_level": "high",
                "shap_explanation": {"top_features": [{"feature": "glucose"}]},
            }],
        )
        assert len(state.active_risks) >= 1
        assert any("diabetes" in r for r in state.active_risks)

    def test_build_state_missing_detection(self):
        state = self.ps.build_patient_state(1)
        assert len(state.missing_information) >= 1
        assert any("observations" in m for m in state.missing_information)

    def test_profile_risk_detection(self):
        state = self.ps.build_patient_state(
            1,
            medical_profile={"smoking_status": "current", "bmi": 32.0},
        )
        risk_texts = " ".join(state.active_risks).lower()
        assert "smoker" in risk_texts
        assert "bmi" in risk_texts


# ── Medication Safety Tests ───────────────────────────────────────────────────

class TestMedicationSafety:
    @pytest.fixture(autouse=True)
    def setup(self):
        from services import medication_safety
        self.ms = medication_safety

    def test_no_interactions_no_meds(self):
        result = self.ms.run_safety_check([], [])
        assert result["total_alerts"] == 0

    def test_drug_interaction_detected(self):
        meds = [
            {"medication_name": "warfarin", "status": "active"},
            {"medication_name": "aspirin", "status": "active"},
        ]
        result = self.ms.run_safety_check(meds, [])
        assert result["total_alerts"] >= 1
        assert result["critical_count"] >= 1

    def test_allergy_conflict_detected(self):
        meds = [
            {"medication_name": "Amoxicillin", "status": "active"},
        ]
        allergies = [
            {"allergen": "Penicillin", "severity": "severe"},
        ]
        result = self.ms.run_safety_check(meds, allergies)
        assert result["total_alerts"] >= 1

    def test_duplicate_therapy_detected(self):
        meds = [
            {"medication_name": "Lisinopril", "status": "active", "drug_class": "ACE inhibitor"},
            {"medication_name": "Ramipril", "status": "active", "drug_class": "ACE inhibitor"},
        ]
        result = self.ms.run_safety_check(meds, [])
        assert result["total_alerts"] >= 1
        assert any(
            a["alert_type"] == "duplicate_therapy"
            for a in result["alerts"]
        )

    def test_warning_count(self):
        meds = [
            {"medication_name": "warfarin", "status": "active"},
            {"medication_name": "aspirin", "status": "active"},
            {"medication_name": "ibuprofen", "status": "active"},
        ]
        result = self.ms.run_safety_check(meds, [])
        assert result["critical_count"] >= 1
        assert result["warning_count"] >= 0

    def test_text_extraction(self):
        text = "The patient is take Aspirin and prescribed Metformin"
        meds = self.ms.extract_medications_from_text(text)
        assert len(meds) >= 1


# ── Conflict Detection Tests ──────────────────────────────────────────────────

class TestConflictDetection:
    @pytest.fixture(autouse=True)
    def setup(self):
        from services import conflict_detection
        self.cd = conflict_detection

    def test_no_conflicts_no_data(self):
        result = self.cd.run_conflict_detection([], [], [], [])
        assert result["total_conflicts"] == 0

    def test_value_discrepancy_detected(self):
        observations = [
            {"id": 1, "standardised_name": "glucose", "numeric_value": 95.0, "original_unit": "mg/dL", "source_label": "Lab A"},
            {"id": 2, "standardised_name": "glucose", "numeric_value": 145.0, "original_unit": "mg/dL", "source_label": "Lab B"},
        ]
        result = self.cd.run_conflict_detection(observations, [], [], [])
        assert result["total_conflicts"] >= 1

    def test_prediction_vs_observation(self):
        predictions = [
            {"id": 1, "disease": "diabetes", "input_data": {"glucose": 95.0}},
        ]
        observations = [
            {"id": 10, "standardised_name": "glucose", "numeric_value": 180.0, "original_unit": "mg/dL"},
        ]
        result = self.cd.run_conflict_detection(observations, predictions, [], [])
        assert result["total_conflicts"] >= 1

    def test_no_conflict_on_similar_values(self):
        observations = [
            {"id": 1, "standardised_name": "glucose", "numeric_value": 100.0, "original_unit": "mg/dL", "source_label": "A"},
            {"id": 2, "standardised_name": "glucose", "numeric_value": 105.0, "original_unit": "mg/dL", "source_label": "B"},
        ]
        result = self.cd.run_conflict_detection(observations, [], [], [])
        assert result["total_conflicts"] == 0


# ── Counterfactual Simulator Tests ────────────────────────────────────────────

class TestCounterfactual:
    @pytest.fixture(autouse=True)
    def setup(self):
        from services import counterfactual
        self.cf = counterfactual

    def test_simulate_no_changes(self):
        result = self.cf.simulate_counterfactual(
            {"diabetes": 50.0, "heart": 30.0},
            [],
        )
        # Should have no delta
        assert all(d == 0 for d in result.risk_deltas.values())

    def test_smoking_cessation_improves_risk(self):
        result = self.cf.simulate_counterfactual(
            {"diabetes": 50.0, "heart": 50.0, "stroke": 50.0},
            [{"factor": "smoking", "change": "quit"}],
        )
        assert result.risk_deltas["heart"] < 0
        assert result.risk_deltas["stroke"] < 0

    def test_add_exercise_reduces_risk(self):
        result = self.cf.simulate_counterfactual(
            {"diabetes": 50.0, "heart": 50.0},
            [{"factor": "exercise", "change": "add_30min_weekly"}],
        )
        assert result.risk_deltas["diabetes"] < 0
        assert result.risk_deltas["heart"] < 0

    def test_glucose_reduction_improves_diabetes_risk(self):
        result = self.cf.simulate_counterfactual(
            {"diabetes": 50.0},
            [{"factor": "glucose", "current": 140.0, "target": 100.0}],
        )
        assert result.risk_deltas["diabetes"] < 0

    def test_result_contains_recommendations(self):
        result = self.cf.simulate_counterfactual(
            {"diabetes": 50.0, "heart": 30.0},
            [{"factor": "smoking", "change": "quit"}],
        )
        assert len(result.recommendations) >= 1

    def test_risk_clamped_between_0_and_100(self):
        result = self.cf.simulate_counterfactual(
            {"diabetes": 10.0},
            [{"factor": "smoking", "change": "start"}],
        )
        for disease, risk in result.modified_risk.items():
            assert 0.0 <= risk <= 100.0


# ── Evidence Tracker Tests ────────────────────────────────────────────────────

class TestEvidenceTracker:
    @pytest.fixture(autouse=True)
    def setup(self):
        from services import evidence_tracker
        self.et = evidence_tracker

    def test_build_general_response(self):
        result = self.et.build_evidence_response("Your health is good.", "general")
        assert result.response_text == "Your health is good."
        assert len(result.disclaimers) >= 1

    def test_response_with_patient_context(self):
        result = self.et.build_evidence_response(
            "Assessment complete.",
            "diagnosis",
            patient_context={
                "observations": [
                    {"id": 1, "test_name": "Glucose", "numeric_value": 95.0, "collection_date": "2025-01-01", "extraction_confidence": 0.95},
                ],
                "predictions": [
                    {"id": 1, "disease": "diabetes", "risk_percentage": 45.0, "risk_level": "moderate"},
                ],
            },
        )
        assert len(result.evidence_sources) >= 2
        assert any(e.source_type == "prediction" for e in result.evidence_sources)

    def test_medication_disclaimer(self):
        result = self.et.build_evidence_response("Medication info.", "medication")
        assert any("pharmacist" in d.lower() for d in result.disclaimers)

    def test_emergency_response(self):
        result = self.et.build_evidence_response("Emergency.", "emergency")
        assert any("emergency" in d.lower() for d in result.disclaimers)

    def test_citation_formatting(self):
        from services.evidence_tracker import EvidenceSource, format_evidence_citations
        sources = [
            EvidenceSource("observation", "1", "Glucose: 95", "Date: 2025-01-01", is_primary=True),
            EvidenceSource("prediction", "2", "Diabetes: 45%", "Level: moderate"),
        ]
        citations = format_evidence_citations(sources)
        assert "[1]" in citations
        assert "★" in citations


# ── FHIR Export Tests ───────────────────────────────────────────────────────

class TestFHIRExport:
    @pytest.fixture(autouse=True)
    def setup(self):
        from services import fhir_export
        self.fe = fhir_export

    def test_bundle_structure(self):
        bundle = self.fe.export_patient_bundle(
            patient_data={"user": {"id": 1, "first_name": "John", "last_name": "Doe"}, "medical_profile": {}},
            observations=[
                {"test_name": "Glucose", "standardised_name": "glucose", "numeric_value": 95.0, "original_unit": "mg/dL", "collection_date": "2025-01-01"},
            ],
            predictions=[],
        )
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "collection"
        assert len(bundle["entry"]) >= 2  # patient + observation

    def test_patient_resource(self):
        resource = self.fe.patient_to_fhir({
            "user": {"id": 1, "first_name": "Jane", "last_name": "Doe", "date_of_birth": "1990-01-01"},
            "medical_profile": {"gender": "female", "blood_type": "O+", "height_cm": 165.0},
        })
        assert resource["resourceType"] == "Patient"
        assert resource["gender"] == "female"
        assert len(resource.get("extension", [])) >= 1

    def test_observation_fhir_format(self):
        resource = self.fe.observation_to_fhir({
            "test_name": "Glucose",
            "standardised_name": "glucose",
            "numeric_value": 95.0,
            "original_unit": "mg/dL",
            "original_value": "95",
            "collection_date": "2025-01-01",
        })
        assert resource["resourceType"] == "Observation"
        assert resource["code"]["coding"][0]["code"] == "2345-7"
        assert resource["valueQuantity"]["value"] == 95.0

    def test_prediction_to_riskassessment(self):
        resource = self.fe.prediction_to_fhir({
            "disease": "diabetes",
            "risk_percentage": 75.0,
            "risk_level": "high",
            "user_id": 1,
        })
        assert resource["resourceType"] == "RiskAssessment"
        assert resource["prediction"][0]["probabilityDecimal"] == 0.75


# ── Trust Gate Tests ────────────────────────────────────────────────────────

class TestTrustGate:
    @pytest.fixture(autouse=True)
    def setup(self):
        from cnn import trust_gate
        self.tg = trust_gate

    def test_high_confidence_trust(self):
        trust = self.tg.compute_trust(
            np.array([0.95, 0.05]),
            reference_stats=self.tg.DEFAULT_REFERENCE_STATS,
        )
        assert trust.confidence_score > 0.9
        assert trust.trust_status in ("trust", "uncertain")

    def test_low_confidence_abstain(self):
        trust = self.tg.compute_trust(
            np.array([0.55, 0.45]),
            reference_stats=self.tg.DEFAULT_REFERENCE_STATS,
        )
        assert trust.confidence_score < 0.75 or trust.overall_score < 0.5

    def test_image_quality_assessment(self):
        from PIL import Image
        import numpy as np
        img = Image.new("L", (224, 224), 128)
        quality, issues, checks = self.tg.assess_image_quality(img)
        assert 0.0 <= quality <= 1.0
        assert "grayscale_ratio" in checks

    def test_default_reference_stats(self):
        assert "mean" in self.tg.DEFAULT_REFERENCE_STATS
        assert 0.0 < self.tg.DEFAULT_REFERENCE_STATS["mean"] < 1.0

    def test_trust_data_includes_disclaimer(self):
        trust = self.tg.compute_trust(np.array([0.9, 0.1]))
        d = trust.to_dict()
        assert "disclaimer" in d
